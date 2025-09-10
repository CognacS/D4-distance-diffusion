##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/8757353a61235fa499dea0cbcd4771eb79b22901/dgd/diffusion_model_discrete.py
#
##########################################################################################################

from typing import Dict, Tuple, Optional, List

import time
from copy import deepcopy, copy

from logging import Logger

################  TORCH IMPORTS  #################
import torch
from torch import Tensor, IntTensor
import torch.nn as nn

from torch_geometric.utils import to_dense_batch


##############  DATATYPES IMPORTS  ###############
from src.datatypes import dense
from src.datatypes.dense import DenseGraph, DenseEdges, DenseNodes
from src.datatypes.sparse import SparseGraph, SparseEdges


###############  METRICS IMPORTS  ################
from src.models.digress.losses.train_denoising import SimpleTrainLossDiscrete


from src.models.generator import Generator, GeneratorWithEvaluation
from src.models import reg_models, reg_architectures
from src.evaluation.assignment.core import Assignment
from src.datatypes.features import get_features_list
from src.datatypes.features.core import increase_dims_list, increase_dims, Feature, get_dims_list

from src.noise.graph_diffusion import (
    MarginalGraphDiffusionProcess,
    GraphDiffusionProcess
)

from src.models.digress import labels

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.aggregation import MeanMetric

from src.models.architectures.distributions.empirical import EmpiricalSampler

from src.datatypes.features.posenc import SinusoidalPosEmb

from src.models.utils.batch_ops import (
    to_onehot_all,
    mask_all,
)
from src.models.utils.diffusion import (
    append_time_to_graph_globals,
    change_time_in_graph_globals
)

from src.noise import (
    reg_diffusion,
    reg_schedule,
    reg_timesampler
)
from src.noise.core import TimeSampler


KEY_TRAIN = 'TRAIN'
KEY_VALID = 'VALID'
KEY_TEST = 'TEST'


@reg_models.register()
class DiscreteDenoisingDiffusionModel(GeneratorWithEvaluation):

    def __init__(
            self,

            ########### configurations ###########
            # model configurations
            denoising: Dict,
            diffusion: Dict,

            # optimizer configuration
            optimizer: Dict,
            
            # features configurations
            features: Dict = None,

            # generation configuration
            # e.g., conditional, batch size
            generation: Dict = None,

            # validation config
            validation: Dict = None,
            
            discard_conditioning: bool = True,
            received_dims: Optional[Dict] = None,

            embed_time: bool = None,

            ######## passed by configurator ######
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):
        
        super().__init__(
            validation=validation,
            dataset_info=dataset_info,
            test_assignment=test_assignment,
            console_logger=console_logger
        )

        ############################  CONFIGS SETUP  ###########################

        # setup console logger
        self.console_logger = console_logger

        # setup config on how to build the model and noise processes
        self.denoising_config = denoising
        self.diffusion_config = diffusion

        self.time_enc_dim = 16
        self.embed_time = embed_time
        self.positional_embedding = SinusoidalPosEmb(self.time_enc_dim)

        # setup optimizer configuration
        self.optimizer_config = optimizer

        # setup additional features
        self.additional_features: List[Feature] = get_features_list(features) if features else []

        # setup generation
        self.generation_config = generation

        #######################  GRAPHS DIMENSIONS SETUP  ######################
        # setup model input and output dimensions (based on the dataset)
        self.data_dims = {
            'x': dataset_info['num_cls_nodes'],
            'e': dataset_info['num_cls_edges'],
            'y': 0 if discard_conditioning else dataset_info['dim_targets']
        }

        self.data_dims['e'] += 1  # account for no-edge class

        if received_dims:
            self.received_dims = deepcopy(received_dims)
            self.received_dims['e'] += 1
        else:
            self.received_dims = self.data_dims

        # increase dimensions based on additional features (creates a copy)
        self.augmented_dims = increase_dims_list(self.received_dims, self.additional_features)
        
        self.augmented_dims = increase_dims(self.augmented_dims, {
            'y': self.time_enc_dim if self.using_pos_emb() else 1
            # account for diffusion time as a global y feature
        })

        self.console_logger.info(f'{self.__class__.__name__} dimensions:')
        self.console_logger.info(f"Size of input features: {self.augmented_dims}")
        self.console_logger.info(f"Size of output features: {self.data_dims}")


        ########################  BUILD DENOISING MODEL  #######################
        # use an empirical sampler when the number of nodes is not known
        self.empirical_sampler = EmpiricalSampler(
            dataset_info =      dataset_info,
            device =            self.device
        )

        # by default, the architecture is a GraphTransformer
        self.denoising_model = reg_architectures.get_instance_from_dict(
            config =        self.denoising_config.architecture,
            input_dims =    self.augmented_dims,
            output_dims =   self.data_dims,
        )


        enc_x_dim = self.denoising_model.get_external_nodes_dim()

        self.ext_x_enc = nn.Linear(
            enc_x_dim + get_dims_list(self.additional_features)['x'],
            enc_x_dim
        )

        ######################  BUILD DIFFUSION PROCESS  #######################

        self.diffusion_timesampler: TimeSampler
        self.diffusion_timesampler = reg_timesampler.get_instance_from_cfg(
            self.diffusion_config.timesampler
        )

        self.diffusion_process: GraphDiffusionProcess
        self.diffusion_process = reg_diffusion.get_instance_from_cfg(
            self.diffusion_config.process,
            schedule = reg_schedule.get_instance_from_cfg(
                self.diffusion_config.schedule
            ),
            num_cls_x = self.data_dims['x'],
            num_cls_e = self.data_dims['e']
        )

        self.diffusion_process_edges: GraphDiffusionProcess = deepcopy(self.diffusion_process)


        ######################  BUILD LOSSES AND METRICS  ######################
        self.train_loss = SimpleTrainLossDiscrete(
            **self.denoising_config.loss
        )

        metrics = nn.ModuleDict({
            labels.DENOISE_CE_X: MeanMetric(),
            labels.DENOISE_CE_E: MeanMetric(),
            labels.DENOISE_CE_EXT_E: MeanMetric(),
            labels.DENOISE_ACC_X: MulticlassAccuracy(num_classes=self.data_dims['x'], validate_args=False),
            labels.DENOISE_ACC_E: MulticlassAccuracy(num_classes=self.data_dims['e'], validate_args=False),
            labels.DENOISE_ACC_EXT_E: MulticlassAccuracy(num_classes=self.data_dims['e'], validate_args=False)
        })

        self.metrics = nn.ModuleDict({
            KEY_TRAIN: deepcopy(metrics),
            KEY_VALID: deepcopy(metrics),
            KEY_TEST: deepcopy(metrics)
        })

        # save hyperaparameters (but those not in the Generator ignored list)
        self.save_hyperparameters(ignore=Generator.IGNORED_HPARAMS + ['received_dims'])


    def is_conditional(self):
        return self.generation_config['conditional']
    
    def get_external_nodes_dim(self):
        return self.denoising_model.get_external_nodes_dim()


    ############################################################################
    #                 SHORTHANDS FOR TRAINING/VALIDATION STEPS                 #
    ############################################################################
    
    def compute_true_pred_denoising(
            self,
            batch_to_generate: SparseGraph,
            batch_external: Optional[SparseGraph] = None,
            edges_external: Optional[SparseEdges] = None
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate the true and predicted nodes and egdes for the denoising
        process. The flow is as follows:
        1 - encode the batch_external to get encoded nodes
        2 - densify batch_to_generate as a DenseGraph, the encoded nodes,
            and the external edges, with onehot and masking
        3 - sample the diffusion process at uniformly random timesteps to
            make a noisy version of batch_to_generate (again requires onehot
            and masking)
        4 - try to denoise the above data which include the batch_to_generate
            and edges_external
        5 - flatten and pack the true and predicted nodes and edges
        The final order is: nodes, edges, external_edges.
        Predicted values are in expanded form, true values are collapsed. This is
        ideal for the cross-entropy loss function.

        Parameters
        ----------
        batch_to_generate : SparseGraph
            sparse graph with collapsed classes (i.e. class indices). This graph
            will be noised and denoised.
        batch_external : Optional[SparseGraph]
            sparse graph with onehot classes. The nodes of this graph will be
            encoded and used to denoise the batch_to_generate. Default is None,
            in which case only the batch_to_generate is noised and denoised.
        edges_external : Optional[Tuple[Tensor, Tensor]]
            external edges in edge_index and edge_attr form, to be noised and
            denoised. Default is None, in which case only the batch_to_generate
            is noised and denoised.

        Returns
        -------
        true_values : List[Tensor]
            list of true values of nodes and edges, in collapsed form.
        pred_values : List[Tensor]
            list of predicted values of nodes and edges, in expanded form.
        """

        #self.console_logger.info(f'nonzeros before: {batch_to_generate.edge_attr}')

        ####################  FORMAT INPUT FOR PREDICTION  #####################
        # 1 - densify
        # transform the current nodes to dense format
        # transform the external nodes and edges to dense format if needed
        batch_to_generate_dense: DenseGraph
        ext_nodes: DenseNodes
        ext_adjmat: DenseEdges      # None if no external graph
        #batch_to_generate_dense, ext_x, ext_node_mask, ext_adjmat, _ = format_generation_task_data(
        batch_to_generate_dense, ext_adjmat, _ = format_generation_task_data(
            curr_graph =		batch_to_generate,
            ext_graph =         batch_external,
            edges_curr_ext =	edges_external
        )

        #self.console_logger.info(f'adjmat: {batch_to_generate_dense.edge_adjmat[0]}')

        #self.console_logger.info(f'{batch_to_generate_dense}')
        #self.console_logger.info(f'{batch_to_generate_dense.edge_mask[0]}')
        #self.console_logger.info(f'{batch_to_generate_dense.edge_mask[1]}')
        #self.console_logger.info(f'{batch_to_generate_dense.edge_mask[2]}')

        # setup masks for edges
        node_mask = batch_to_generate_dense.node_mask
        triang_edge_mask = torch.tril(batch_to_generate_dense.edge_mask, diagonal=-1)

        # 2 - copy true masked data (to be returned later)
        true_x = batch_to_generate_dense.x.argmax(dim=-1)[node_mask]
        true_e = batch_to_generate_dense.edge_adjmat.argmax(dim=-1)[triang_edge_mask]

        if ext_adjmat is not None:
            ext_edge_mask = ext_adjmat.edge_mask
            true_ext_e = ext_adjmat.edge_adjmat.argmax(dim=-1)[ext_edge_mask]
        else:
            ext_edge_mask = None
            true_ext_e = None

        ##################  UPDATE MARGINAL PROCESS IF NEEDED  #################
        if hasattr(self.diffusion_process, 'update'):

            self.diffusion_process.update(x_labels=true_x, e_labels=true_e)

            if edges_external is not None:
                self.diffusion_process_edges.update(e_labels=true_ext_e)

        #######################  APPLY GRAPH DIFFUSION  ########################
        # sample the timesteps for the diffusion process
        max_times = torch.full((batch_to_generate.num_graphs,), self.diffusion_process.get_max_time()-1) # must be in cpu
        u: Tensor = self.diffusion_timesampler.sample_time(max_time=max_times).to(self.device) + 1 # do not sample u=0

        self.append_time(
            batch_to_generate_dense,
            time = u
        )

        # sample the noisy graph at timestep u

        # WARNING: here selfloops are not masked!!!
        # noisy_data = self.diffusion_process.sample_from_original(
        #     original_datapoint=(batch_to_generate_dense, ext_adjmat),
        #     t=u
        # )

        noisy_graph = self.diffusion_process.sample_from_original(batch_to_generate_dense, t=u)
        noisy_edges = self.diffusion_process_edges.sample_from_original(ext_adjmat, t=u)

        noisy_data = (noisy_graph, noisy_edges)

        # onehot and mask the noisy data again (to remove the fake noisy components)
        onehot_data = to_onehot_all(
            *noisy_data,
            **self.data_dims
        )

        ext_nodes = None
        # add features to the noisy data
        if batch_external is not None:
            #print('train1:', ext_nodes.indegree)
            #print('train2:', onehot_data[1].indegree)
            ext_edges = onehot_data[1]
            self.add_additional_features((onehot_data[0], ext_edges, ext_edges.transpose(), batch_external))
            ext_nodes = to_dense_nodes(batch_external)  # to dense nodes
            ext_nodes.x = self.ext_x_enc(ext_nodes.x)   # encode external nodes to required dimension
            ext_nodes.apply_mask()                      # apply mask to non existing nodes
            onehot_data = (onehot_data[0], ext_edges)   # repack current subgraph
        else:
            self.add_additional_features(onehot_data)

        masked_data = mask_all(
            *onehot_data
        )

        noisy_batch_to_generate_dense_onehot, noisy_ext_edges_onehot = masked_data

        #####################  PREDICT THE ORIGINAL GRAPH  #####################
        gen_batch_dense: DenseGraph
        gen_ext_edges: DenseEdges   # None if no external graph
        gen_batch_dense, gen_ext_edges = self.denoising_model(
            graph =                noisy_batch_to_generate_dense_onehot,
            ext_X =                None if ext_nodes is None else ext_nodes.x,
            ext_node_mask =        None if ext_nodes is None else ext_nodes.node_mask,
            ext_edges =            noisy_ext_edges_onehot
        )

        pred_x = gen_batch_dense.x[node_mask]
        pred_e = gen_batch_dense.edge_adjmat[triang_edge_mask]
        if gen_ext_edges is not None:
            pred_ext_e = gen_ext_edges.edge_adjmat[ext_edge_mask]
        else:
            pred_ext_e = None

        ###########################  FINAL PACKING  ############################
        true_values = [true_x, true_e, true_ext_e]
        pred_values = [pred_x, pred_e, pred_ext_e, node_mask, triang_edge_mask, ext_edge_mask]
        
        #self.console_logger.info(f'{true_e.shape}, {pred_e.shape}')
        #self.console_logger.info(f'{true_e}')
        #self.console_logger.info(f'{pred_e}')
        #self.console_logger.info(f'nonzeros: {torch.nonzero(true_e)}')

        #self.console_logger.info(f'{true_ext_e.shape}, {pred_ext_e.shape}')
        #self.console_logger.info(f'{true_ext_e}')
        #self.console_logger.info(f'{pred_ext_e}')

        return true_values, pred_values
    

    @torch.no_grad()
    def compute_metrics(
            self,
            loss_logs: Dict[str, Tensor],
            pred_values: List[Tensor],
            true_values: List[Tensor],
            split: str
        ):
        
        metrics = self.metrics[split]

        metrics[labels.DENOISE_CE_X](loss_logs[labels.DENOISE_CE_X])
        metrics[labels.DENOISE_CE_E](loss_logs[labels.DENOISE_CE_E])
        if pred_values[0].numel() > 0:
            metrics[labels.DENOISE_ACC_X](pred_values[0], true_values[0])
        if pred_values[1].numel() > 0:
            metrics[labels.DENOISE_ACC_E](pred_values[1], true_values[1])

        if pred_values[2] is not None:
            metrics[labels.DENOISE_CE_EXT_E](loss_logs[labels.DENOISE_CE_EXT_E])
            if pred_values[2].numel() > 0:
                metrics[labels.DENOISE_ACC_EXT_E](pred_values[2], true_values[2])

        return metrics



    def prepare_batch(self, batch: SparseGraph):

        if self.received_dims['y'] == 0:
            batch.y = None

        return batch
    

    ############################################################################
    #                          TRAINING PHASE SECTION                          #
    ############################################################################

    def on_train_epoch_start(self) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(self) -> None:
        """"Recall that this method is called AFTER the validation epoch, if there is any!"""

        if isinstance(self.diffusion_process, MarginalGraphDiffusionProcess):
            # stop updating marginals at the end of the first training epoch
            self.diffusion_process.stop_updating()
            self.diffusion_process_edges.stop_updating()
        
        denoise_logs = self.apply_prefix(
            metrics = self.metrics[KEY_TRAIN],
            prefix = f'train_denoising'
        )
        self.log_dict(denoise_logs)

        self.total_elapsed_time += time.time() - self.start_time
        self.max_memory_reserved = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)


    def training_step(self, batch: SparseGraph|Dict, batch_idx: int):

        # compute true and predicted nodes and edges from the denoising process
        if isinstance(batch, dict):

            curr_batch = self.prepare_batch(batch['curr'])
            ext_batch = self.prepare_batch(batch['ext'])
            ext_edges = batch['edges_curr_ext']

            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = curr_batch,
                batch_external =	ext_batch,
                edges_external =	ext_edges
            )
        else:
            batch = self.prepare_batch(batch)
            
            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = batch
            )

        # compute denoising training loss
        denoise_loss, denoise_logs = self.train_loss(
            pred_data,
            true_data,
            ret_log=True
        )

        # compute metrics
        self.compute_metrics(denoise_logs, pred_data, true_data, split=KEY_TRAIN)

        # apply prefix to logs
        logs = self.apply_prefix(
            metrics = self.metrics[KEY_TRAIN],
            prefix = f'train_denoising'
        )

        self.log_dict(logs)

        return {'loss': denoise_loss}


    def configure_optimizers(self):

        # currently using the AdamW optimizer
        # NOTE: the original code used the option "amsgrad=True"
        params = list(self.denoising_model.parameters()) + list(self.ext_x_enc.parameters())

        return torch.optim.AdamW(
            params, **self.optimizer_config
        )
    
    ############################################################################
    #                         VALID/TEST PHASE SECTION                         #
    ############################################################################

    @torch.no_grad()
    def on_evaluation_epoch_start(self, which=KEY_VALID) -> None:

        # part used for gathering conditioning
        # attributes from the validation or test set
        # to be used for generation
        self.conditioning_y = None
        if self.is_conditional():
            self.conditioning_y = []
            self.num_cond_y = 0


    @torch.no_grad()
    def evaluation_step(self, batch: SparseGraph, batch_idx: int, which=KEY_VALID) -> None:

        batch = self.prepare_batch(batch)

        #############  SAVE PROPERTIES FOR CONDITIONAL GENERATION  #############
        # save some target properties if needed for conditional generation
        if self.is_conditional():

            # get how many will be sampled
            sampling_metrics = self.losses['sampling']
            if which in sampling_metrics:
                sampling_metrics = sampling_metrics[which]

            num_to_sample = sampling_metrics.generation_cfg['num_samples']

            # get the conditioning attributes from the batch
            if self.num_cond_y < num_to_sample:
                to_grab = min(num_to_sample - self.num_cond_y, batch.num_graphs)
                self.conditioning_y.append(batch.y[:to_grab, -2:].float())
                self.num_cond_y += to_grab

        #######################  TRAIN DENOISING MODEL  ########################

        # FLOW DEFINITION
        # survived graph -> encoded survived graph
        # removed graph -> noisy graph -> denoised graph

        # compute true and predicted nodes and edges from the denoising process
        if isinstance(batch, dict):

            curr_batch = self.prepare_batch(batch['curr'])
            ext_batch = self.prepare_batch(batch['ext'])
            ext_edges = batch['edges_curr_ext']

            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = curr_batch,
                batch_external =	ext_batch,
                edges_external =	ext_edges
            )
        else:
            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = batch
            )

        # compute denoising training loss
        denoise_loss, denoise_logs = self.train_loss(
            pred_data,
            true_data,
            reduce=False,
            ret_log=True
        )

        # compute denoising training loss
        #nll = self.compute_val_loss(pred_data, noisy_data, true_data, node_mask, test=False)

        # compute metrics
        self.compute_metrics(denoise_logs, pred_data, true_data, split=which)

        return {'loss': denoise_loss}


    @torch.no_grad()
    def on_evaluation_epoch_end(self, which=KEY_VALID) -> None:

        if which == KEY_VALID:
            assignment = self.valid_assignment
        else:
            assignment = self.test_assignment


        # start with already computed metrics (during evaluation epochs)
        metrics = {
            **self.metrics[which]
        }

        if assignment is not None:
        
            batch_size = self.generation_config['batch_size']
        
            # compute sampling metrics
            assignment_results, hists = self.perform_assignment(
                assignment=assignment, other_metrics=metrics,
                sampling_kwargs={'batch_size': batch_size}
            )
            
            # add the assignment results to the metrics
            metrics.update(assignment_results)

            # log histograms with wandb if available (check is inside)
            self.log_wandb_histograms(self.apply_prefix(hists, f'{which}/sampling'))

        # output the metrics for reading purposes
        self.console_logger.info(str(self.get_metrics_values(metrics)))

        # add prefix to logs
        to_log = self.apply_prefix(
            metrics = metrics,
            prefix = f'{which}'
        )

        self.log_dict(to_log)


    ############################################################################
    #           VALIDATION PHASE SECTION (executed during validation)          #
    ############################################################################

    def on_validation_epoch_start(self):
        self.on_evaluation_epoch_start(which=KEY_VALID)

    def validation_step(self, batch: SparseGraph, batch_idx: int):
        return self.evaluation_step(batch, batch_idx, which=KEY_VALID)

    def on_validation_epoch_end(self):
        return self.on_evaluation_epoch_end(which=KEY_VALID)

    ############################################################################
    #               TEST PHASE SECTION (executed during testing)               #
    ############################################################################

    def on_test_epoch_start(self):
        self.on_evaluation_epoch_start(which=KEY_TEST)

    def test_step(self, batch: SparseGraph, batch_idx: int):
        return self.evaluation_step(batch, batch_idx, which=KEY_TEST)

    def on_test_epoch_end(self):
        return self.on_evaluation_epoch_end(which=KEY_TEST)
    

    ############################################################################
    #                           MODEL CALL FUNCTIONS                           #
    ############################################################################
    
    @torch.no_grad()
    def forward_denoising(
            self,
            graph_to_gen: DenseGraph,
            ext_edges_to_gen: DenseEdges,
            encoded_ext_graph: Optional[SparseGraph],
            denoising_time: IntTensor,
            return_onehot: bool=True,
            return_masked: bool=True,
            copy_globals_to_output: bool=True
        ) -> Tuple[DenseGraph, Tensor]:	

        #assert_is_onehot(graph_to_gen, ext_edges_to_gen)

        #augmented_graph_to_gen = graph_to_gen.clone()
        augmented_graph_to_gen = copy(graph_to_gen)

        encoded_ext_nodes: Optional[DenseNodes] = None
        if encoded_ext_graph is not None:
            self.add_additional_features((
                augmented_graph_to_gen,
                ext_edges_to_gen, ext_edges_to_gen.transpose(),
                encoded_ext_graph))
            encoded_ext_nodes = to_dense_nodes(encoded_ext_graph)
            encoded_ext_nodes.x = self.ext_x_enc(encoded_ext_nodes.x)
            encoded_ext_nodes.apply_mask()
            ext_edges_to_gen.apply_mask()
        else:
            self.add_additional_features((augmented_graph_to_gen, ext_edges_to_gen))

        augmented_graph_to_gen.apply_mask()


        # predict final graph and edges
        final_graph: DenseGraph
        final_ext_edges: DenseEdges
        final_graph, final_ext_edges = self.denoising_model(
            graph =				    augmented_graph_to_gen,
            ext_X =					None if encoded_ext_nodes is None else encoded_ext_nodes.x,
            ext_node_mask =			None if encoded_ext_nodes is None else encoded_ext_nodes.node_mask,
            ext_edges =         	ext_edges_to_gen
        )

        has_ext_edges = final_ext_edges is not None
        
        # transform the logits to probabilities
        final_graph.x = torch.softmax(final_graph.x, dim=-1)
        final_graph.edge_adjmat = torch.softmax(final_graph.edge_adjmat, dim=-1)
        if has_ext_edges:
            final_ext_edges.edge_adjmat = torch.softmax(final_ext_edges.edge_adjmat, dim=-1)
        else:
            final_ext_edges = None


        # sample graph at step t-1 from posterior
        generated_graph = self.diffusion_process.sample_posterior(
            original_datapoint =	final_graph,
            current_datapoint =		graph_to_gen,
            t =						denoising_time
        )

        generated_ext_edges = self.diffusion_process_edges.sample_posterior(
            original_datapoint =	final_ext_edges,
            current_datapoint =		ext_edges_to_gen,
            t =						denoising_time
        )

        if return_onehot:
            generated_graph, generated_ext_edges = to_onehot_all(
                generated_graph, generated_ext_edges,
                **self.data_dims
            )

        if return_masked:
            generated_graph, generated_ext_adjmat = mask_all(
                generated_graph, generated_ext_edges
            )


        if copy_globals_to_output:
            generated_graph.y = graph_to_gen.y

        return generated_graph, generated_ext_adjmat

    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        conditioning_y: Optional[Tensor]=None,
        ext_graph: Optional[SparseGraph]=None,
        encoded_ext_x: Optional[Tensor]=None,
        number_of_nodes: Optional[IntTensor]=None,
        return_directed: bool=True,
        save_chains: int=0
    ):
        ########################################################################
        #                        INITIAL SAMPLING SETUP                        #
        ########################################################################

        #########################  SETUP CONDITIONING  #########################

        # TODO: implement the generation chain saving
        do_save_chains = save_chains > 0


        # elaborate external graph
        # if it is None, then all parts about it will be skipped
        if ext_graph is not None:

            ext_graph = copy(ext_graph)
            ext_graph.x = encoded_ext_x

            _, ext_node_mask = to_dense_batch(
                x =             ext_graph.x,
                batch =         ext_graph.batch,
                batch_size =    batch_size
            )

        else:
            ext_node_mask = None

        # if the number of nodes is not given, sample it from the empirical distribution
        if number_of_nodes is None:
            number_of_nodes = self.empirical_sampler(batch_size)

        ##############  SAMPLE THE STARTING SUBGRAPHS (AS NOISE)  ##############
        new_graph: DenseGraph
        new_ext_edges: DenseEdges
        new_graph = self.diffusion_process.sample_stationary(
            num_new_nodes = number_of_nodes,
            ext_node_mask = ext_node_mask,
            device = self.device
        )
        if ext_node_mask is not None:
            new_ext_edges = self.diffusion_process_edges.sample_stationary(
                num_new_nodes = number_of_nodes,
                ext_node_mask = ext_node_mask,
                device = self.device,
                generate_edges = True
            )
        else:
            new_ext_edges = None

        # convert the new subgraph to one-hot
        new_graph, new_ext_edges = to_onehot_all(
            *(new_graph, new_ext_edges),
            **self.data_dims
        )

        # copy the global information to the new subgraph
        if ext_graph is not None:
            new_graph.y = ext_graph.y.clone()
        elif conditioning_y is not None:
            new_graph.y = conditioning_y.clone()
        else:
            new_graph.y = None
        
        ###################  INITIALIZE DENOISING TIME AS T  ###################
        diffusion_max_time = self.diffusion_process.get_max_time()

        diff_time = torch.full((batch_size,), diffusion_max_time, dtype=torch.int, device=self.device)

        self.append_time(
            graph = new_graph,
            time = diff_time
        )

        new_graph_dense = new_graph

        ########################################################################
        #                            DENOISING LOOP                            #
        ########################################################################

        t_tensor = torch.empty(batch_size, dtype=torch.int, device=self.device)

        # denoise going backwards in time
        for t in reversed(range(1,diffusion_max_time+1)):

            t_tensor.fill_(t)

            # sample graph at step u-1
            new_graph_dense, new_ext_edges = self.forward_denoising(
                graph_to_gen =		new_graph_dense,
                ext_edges_to_gen =	new_ext_edges,
                encoded_ext_graph = copy(ext_graph),
                denoising_time = 	t_tensor,
                return_onehot =		True
            )

            # update denoising time (in-place), denoising go down!
            self.change_time(new_graph_dense, t_tensor-1)
            
            # new_graph_dense.y[..., 0] = self.diffusion_process.normalize_time(
            #     t = t-1
            # )

        #######################  END OF DENOISING LOOP  ########################

        ########################################################################
        #                 SPARSIFY THE PRODUCED GRAPH AND EDGES                #
        ########################################################################

        output_graph, output_edges = sparsify_data(
            subgraph = new_graph_dense,
            ext_edges = new_ext_edges,
            subgraph_nodes_num = number_of_nodes,
            ext_ptr = None if ext_graph is None else ext_graph.ptr
        )

        ########################################################################
        #                                RETURN                                #
        ########################################################################

        # replace globals with starting variables, removing time
        if ext_graph is not None:
            output_graph.y = ext_graph.y
        elif conditioning_y is not None:
            output_graph.y = conditioning_y
        else:
            output_graph.y = None

        if output_edges is not None:
            return output_graph, output_edges
        else:
            return output_graph
            
    
    @torch.no_grad()
    def sample(
            self,
            num_samples: int,
            condition: Optional[Dict]=None,
            batch_size: Optional[int]=None
        ):

        if batch_size is None:
            batch_size = self.generation_config['batch_size']

        samples_left_to_generate = num_samples
        batch_idx = 0
        samples = []

        while samples_left_to_generate > 0:
            to_generate = min(samples_left_to_generate, batch_size)
            self.console_logger.info(f'Generating {to_generate} graphs...')

            graph_batch = self.sample_batch(
                batch_size=to_generate,
                conditioning_y=condition[batch_idx] if condition is not None else None
            )

            graph_batch.collapse('x', 'edge_attr')

            output_batch = graph_batch.to_data_list()

            output_batch = [g.cpu() for g in output_batch]

            samples.extend(output_batch)

            samples_left_to_generate -= to_generate
            batch_idx += 1
            self.console_logger.info(f'Generated {len(samples)}/{num_samples} graphs')

        return samples
    

    ############################################################################
    #                         UTILITY MODULE FUNCTIONS                         #
    ############################################################################


    def add_additional_features(self, graph: SparseGraph|DenseGraph|Tuple[DenseGraph, DenseEdges]) -> Tensor:

        for feature in self.additional_features:
            feature(graph)

        return graph


    def using_pos_emb(self):
        return self.embed_time is not None and self.embed_time


    def append_time(self, graph, time):
        if self.using_pos_emb():
            emb = self.positional_embedding
        else:
            time = self.diffusion_process.normalize_time(
                t = time
            )
            emb = None

        append_time_to_graph_globals(graph, time, emb)


    def change_time(self, graph, time):
        if self.using_pos_emb():
            emb = self.positional_embedding
        else:
            time = self.diffusion_process.normalize_time(
                t = time
            )
            emb = None

        change_time_in_graph_globals(graph, time, emb)





################################################################################
#                               UTILITY METHODS                                #
################################################################################

# the following methods are utility methods which could be an integral part of
# the main class, but have been put outside for readability

##############################  DATA FORMATTING  ###############################

def format_generation_task_data(
        curr_graph: SparseGraph,
        ext_graph: SparseGraph,
        edges_curr_ext: SparseEdges=None,
        edges_ext_curr: SparseEdges=None
    ) -> Tuple[DenseGraph, DenseEdges, DenseEdges]:
    """transform the splitting of the two graphs into the format required by the
    model, that is:
    - extract a dense representation (and a node mask) of the Ne nodes from ext_graph
    - transform curr_graph into a DenseGraph (with Nc nodes, and adjmat of shape (*, Nc, Nc, *))
    - transform edges_ext_curr and edges_curr_ext into dense adjacency matrices
      each of shape (*, Ne, Nc, *) and (*, Nc, Ne, *) respectively.
      If one of the two is None, it is assumed that the graph is undirected and
      a single adjacency matrix ((*, Ne, Nc, *) or (*, Nc, Ne, *)) is returned.

    Notice that the possibly very big adjacency matrix of curr_graph (*, Nc, Nc, *)
    is never computed, so Nc >> Ne is allowed, avoiding a squared dependency on
    Nc.

    Parameters
    ----------
    curr_graph : SparseGraph
        graph of nodes in the current graph to be generated
    ext_graph : SparseGraph
        graph of nodes from an external graph
    edges_ext_curr : Tuple[Tensor, Tensor]
        edges going from the external nodes to the current nodes. The first
        component is the edge_index, the second the edge_attr. If is is None,
        the dense version is not returned (default: None)
    edges_curr_ext : Tuple[Tensor, Tensor], optional
        edges going from the current nodes to the external nodes. The first
        component is the edge_index, the second the edge_attr. If is is None,
        the dense version is not returned (default: None)

    Returns
    -------
    curr_graph_dense : DenseGraph
        graph of nodes removed by the removal process as a dense graph.
    ext_x_tensor : Tensor
        tensor of the external nodes, as a batched dense representation.
    ext_node_mask : BoolTensor
        mask of the true external nodes, as the process of densifying generates
        some dummy nodes.
    adjmat_ext_curr : Optional[Tensor]
        edges going from the external nodes to the current nodes, as a dense
        adjacency matrix. If edges_ext_curr is None, this is not returned.
    adjmat_curr_ext : Optional[Tensor]
        edges going from the current nodes to the external nodes, as a dense
        adjacency matrix. If edges_curr_ext is None, this is not returned.
    """

    batch_size = curr_graph.num_graphs

    # transform the current graph into a dense representation
    curr_graph_dense = dense.sparse_graph_to_dense_graph(
        sparse_graph =		curr_graph,
        handle_one_hot =    True
    )

    # initialize to None
    ext_nodes = None
    edges_ext_curr_dense = None
    edges_curr_ext_dense = None

    # if conditioned on an external graph
    if ext_graph is not None:

        # extract the dense representation of the surviving nodes
        _, ext_node_mask = to_dense_batch(
            x =             ext_graph.x,
            batch =         ext_graph.batch,
            batch_size =    batch_size
        )

        if (edges_ext_curr is not None) or (edges_curr_ext is not None):
            edge_mask_ext_curr = dense.get_bipartite_edge_mask_dense(
                node_mask_a = ext_node_mask,
                node_mask_b = curr_graph_dense.node_mask
            )

        # transform the edges_curr_ext into a dense adjacency matrix
        if edges_curr_ext is not None:
            # transpose
            edge_mask_curr_ext = edge_mask_ext_curr.transpose(1, 2)

            adjmat_curr_ext = dense.to_dense_adj_bipartite(
                edge_index =	edges_curr_ext.edge_index,
                edge_attr =		edges_curr_ext.edge_attr,
                batch_s =		curr_graph.batch,
                batch_t =		ext_graph.batch,
                batch_size =	batch_size,
                handle_one_hot =True
            )

            edges_curr_ext_dense = DenseEdges(
                edge_adjmat =   adjmat_curr_ext,
                edge_mask =     edge_mask_curr_ext
            )

        # transform the edges_ext_curr into a dense adjacency matrix
        if edges_ext_curr is not None:

            adjmat_ext_curr =   dense.to_dense_adj_bipartite(
                edge_index =    edges_ext_curr.edge_index,
                edge_attr =     edges_ext_curr.edge_attr,
                batch_s =       ext_graph.batch,
                batch_t =       curr_graph.batch,
                batch_size =    batch_size,
                handle_one_hot =True
            )

            edges_ext_curr_dense = DenseEdges(
                edge_adjmat =   adjmat_ext_curr,
                edge_mask =     edge_mask_ext_curr
            )   

    return curr_graph_dense, edges_curr_ext_dense, edges_ext_curr_dense


def to_dense_nodes(graph: SparseGraph) -> DenseNodes:
    """transform a sparse graph to a dense representation of the nodes, with
    the node mask and the node degree.

    Parameters
    ----------
    graph : SparseGraph
        sparse graph to be transformed

    Returns
    -------
    dense_nodes : DenseNodes
        dense representation of the nodes
    """

    x_tensor, node_mask = to_dense_batch(
        x =				graph.x,
        batch =			graph.batch,
        batch_size =	graph.num_graphs
    )

    # extract the dense degree
    if hasattr(graph, 'node_indegree'):
        degree = graph.node_indegree # if indegree is passed as a node property use it
        # e.g., due to sub selection of nodes
    else:
        degree = graph.indegree # else compute it

    degree_tensor, _ = to_dense_batch(
        x =             degree,
        batch =         graph.batch,
        batch_size =    graph.num_graphs
    )

    return DenseNodes(
        x = x_tensor,
        node_mask = node_mask,
        indegree = degree_tensor
    )



def sparsify_data(
        subgraph: DenseGraph,
        ext_edges: DenseEdges,
        subgraph_nodes_num: IntTensor,
        ext_ptr: Tensor = None,
    ) -> Tuple[SparseGraph, SparseEdges]:

    ########################  SPARSIFY DENSE SUBGRAPH  #########################
    #subgraph = subgraph.clone()
    subgraph = copy(subgraph)

    # remove self-loops from dense adjacency matrices
    subgraph.edge_adjmat = dense.dense_remove_self_loops(
        subgraph.edge_adjmat
    )

    # remove no edge class from dense adjacency
    # matrices
    subgraph.edge_adjmat = dense.remove_no_edge(
        subgraph.edge_adjmat,
        sparse = False,
        collapsed = False
    )

    # transform the new graph to sparse format
    new_subgraph = dense.dense_graph_to_sparse_graph(
        dense_graph =	subgraph,
        num_nodes =		subgraph_nodes_num,
        batchify =      True
    )

    ##########################  SPARSIFY DENSE EDGES  ##########################

    if ext_edges is not None:

        ext_edges.edge_adjmat = dense.remove_no_edge(
            ext_edges.edge_adjmat,
            sparse = False,
            collapsed = False
        )

        new_edges = dense.dense_edges_to_sparse_edges(
            dense_edges =		ext_edges,
            cum_num_nodes_s =	new_subgraph.ptr,
            cum_num_nodes_t =	ext_ptr
        )

    else:
        new_edges = None

    return new_subgraph, new_edges