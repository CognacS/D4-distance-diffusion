##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/8757353a61235fa499dea0cbcd4771eb79b22901/dgd/diffusion_model_discrete.py
#
##########################################################################################################

from typing import Dict, Tuple, Union, Optional, List, Callable, Any

import time
import os
from copy import deepcopy

from logging import Logger
import wandb

import numpy as np


################  TORCH IMPORTS  #################
import torch
from torch import Tensor, BoolTensor, IntTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from torchmetrics import Metric

##############  DATATYPES IMPORTS  ###############
from src.datatypes import (
    dense,
    sparse,
    split
)
from src.datatypes.dense import DenseGraph, DenseEdges, DenseNodes
from src.datatypes.sparse import SparseGraph, SparseEdges

################  NOISE IMPORTS  #################

from copy import copy
from src.noise.config_support import build_noise_process
from src.noise.batch_transform.sequence_sampler import sample_sequences

###############  METRICS IMPORTS  ################
import src.evaluation.metrics as m_list
from src.models.d4.losses.train_denoising import TrainLossDistance


from src.models.generator import GeneratorWithEvaluation
from src.models import reg_models, reg_architectures
from src.evaluation.assignment import Assignment
from src.datatypes.features import get_features_list
from src.datatypes.features.core import increase_dims_list, increase_dims, Feature, get_dims_list
from src.datatypes.features.posenc import SinusoidalPosEmb

from src.models.d4.geometry import mds, compute_affine, transform_vectors, fit_dist_ext

from src.noise import reg_schedule, reg_diffusion, reg_timesampler
from src.noise.timesample import TimeSampler
from src.models.utils.diffusion import (
    append_time_to_graph_globals,
    change_time_in_graph_globals
)
from src.noise.graph_diffusion import (
    MarginalGraphDiffusionProcess,
    GraphDiffusionProcess
)

from src.noise.graph_diffusion import GraphDiffusionProcess
from src.noise.graph_cont_diffusion import DistanceGaussianDiffusionProcess

from src.models.d4 import labels

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.aggregation import MeanMetric
from torchmetrics.regression import MeanAbsoluteError

from src.models.architectures.distributions.empirical import EmpiricalSampler
from pytorch_lightning.loggers import WandbLogger
import collections


KEY_TRAIN = 'TRAIN'
KEY_VALID = 'VALID'
KEY_TEST = 'TEST'

@reg_models.register('D4Model')
class DistanceDiscreteDenoisingDiffusionModel(GeneratorWithEvaluation):

    def __init__(
            self,

            ########### configurations ###########
            # model configurations
            denoising: Dict,
            diffusion: Dict,
            dist_diffusion: Dict,

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
        self.dist_diffusion_config = dist_diffusion

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
            'y': 0 if discard_conditioning else dataset_info['dim_targets'],
            "dist": 1
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

        
        self.diffusion_process_dists: DistanceGaussianDiffusionProcess
        self.diffusion_process_dists = reg_diffusion.get_instance_from_cfg(
            self.dist_diffusion_config.process,
            schedule = reg_schedule.get_instance_from_cfg(
                self.dist_diffusion_config.schedule
            ),
            undirected=True
        )


        ######################  BUILD LOSSES AND METRICS  ######################
        self.train_loss = TrainLossDistance(
            **self.denoising_config.loss
        )

        metrics = nn.ModuleDict({
            labels.DENOISE_CE_X: MeanMetric(),
            labels.DENOISE_CE_E: MeanMetric(),
            labels.DENOISE_ACC_X: MulticlassAccuracy(num_classes=self.data_dims['x'], validate_args=False),
            labels.DENOISE_ACC_E: MulticlassAccuracy(num_classes=self.data_dims['e'], validate_args=False),
            labels.DENOISE_MSE_DIST: MeanMetric(),
            labels.DENOISE_MAE_DIST: MeanAbsoluteError(),
            labels.DENOISE_TOTAL: MeanMetric()
        })

        self.metrics = nn.ModuleDict({
            KEY_TRAIN: deepcopy(metrics),
            KEY_VALID: deepcopy(metrics),
            KEY_TEST: deepcopy(metrics)
        })

        # save hyperaparameters (but those not in the Generator ignored list)
        self.save_hyperparameters(ignore=GeneratorWithEvaluation.IGNORED_HPARAMS + ['received_dims'])


    def is_conditional(self):
        return self.generation_config['conditional']
    
    def get_external_nodes_dim(self):
        return self.denoising_model.get_external_nodes_dim()


    ############################################################################
    #                 SHORTHANDS FOR TRAINING/VALIDATION STEPS                 #
    ############################################################################
    
    
    
    def compute_true_pred_denoising(
            self,
            batch_to_generate: SparseGraph
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

        ####################  FORMAT INPUT FOR PREDICTION  #####################
        # 1 - densify
        # transform the current nodes to dense format
        # transform the external nodes and edges to dense format if needed
        batch_to_generate_dense: DenseGraph
        batch_to_generate_dense = format_generation_task_data(
            curr_graph =		batch_to_generate
        )
        
        # setup masks for edges
        node_mask = batch_to_generate_dense.node_mask
        triang_edge_mask = torch.tril(batch_to_generate_dense.edge_mask, diagonal=-1)

        # 2 - copy true masked data (to be returned later)
        true_x = batch_to_generate_dense.x.argmax(dim=-1)[node_mask]
        true_e = batch_to_generate_dense.edge_adjmat.argmax(dim=-1)[triang_edge_mask]
        true_dist = batch_to_generate_dense.edge_dist[triang_edge_mask]
            
        ##################  UPDATE MARGINAL PROCESS IF NEEDED  #################
        if hasattr(self.diffusion_process, 'update'):

            self.diffusion_process.update(x_labels=true_x, e_labels=true_e)

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
        # apply discrete noise, then continuous noise to dists
        noisy_graph = self.diffusion_process.sample_from_original(batch_to_generate_dense, t=u)
        noisy_graph = self.diffusion_process_dists.sample_from_original(noisy_graph, t=u)
        
        
        noisy_data = noisy_graph

        # onehot and mask the noisy data again (to remove the fake noisy components)
        onehot_data = to_onehot_all(
            *noisy_data,
            **self.data_dims
        )

        self.add_additional_features(onehot_data)

        masked_data = mask_all(
            *onehot_data
        )

        noisy_batch_to_generate_dense_onehot, noisy_ext_edges_onehot = masked_data

        #####################  PREDICT THE ORIGINAL GRAPH  #####################
        gen_batch_dense: DenseGraph
        gen_ext_edges: DenseEdges   # None if no external graph
        gen_batch_dense, gen_ext_edges = self.denoising_model(
            graph =                noisy_batch_to_generate_dense_onehot
        )

        pred_x = gen_batch_dense.x[node_mask]
        pred_e = gen_batch_dense.edge_adjmat[triang_edge_mask]
        pred_dist = gen_batch_dense.edge_dist[triang_edge_mask]
            
        ###########################  DISTANCE PREDICTION  ############################
        
        ###########################  FINAL PACKING  ############################

        true_values = [true_x, true_e, true_dist]
        pred_values = [pred_x, pred_e, pred_dist, node_mask, triang_edge_mask,]
        
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
        metrics[labels.DENOISE_MSE_DIST](loss_logs[labels.DENOISE_MSE_DIST])
        metrics[labels.DENOISE_TOTAL](loss_logs[labels.DENOISE_TOTAL])
        if pred_values[0].numel() > 0:
            metrics[labels.DENOISE_ACC_X](pred_values[0], true_values[0])
        if pred_values[1].numel() > 0:
            metrics[labels.DENOISE_ACC_E](pred_values[1], true_values[1])
        if pred_values[2].numel() > 0:
            metrics[labels.DENOISE_MAE_DIST](pred_values[2], true_values[2])

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
        batch = self.prepare_batch(batch)

        true_data, pred_data = self.compute_true_pred_denoising(
            batch_to_generate = batch,
            train_step=True
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

        # compute metrics
        self.compute_metrics(denoise_logs, pred_data, true_data, split=which)

        logs = self.apply_prefix(
            metrics = self.metrics[KEY_VALID],
            prefix = f'valid_denoising'
        )

        self.log_dict(logs)

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
            denoising_time: IntTensor,
            return_onehot: bool=True,
            return_masked: bool=True,
            copy_globals_to_output: bool=True
        ) -> Tuple[DenseGraph, Tensor]:	

        augmented_graph_to_gen = copy(graph_to_gen)

        self.add_additional_features(augmented_graph_to_gen)

        augmented_graph_to_gen.apply_mask()
        

        # predict final graph and edges
        final_graph: DenseGraph
        final_graph, final_ext_edges = self.denoising_model(
            graph =				    augmented_graph_to_gen
        )

        has_ext_edges = final_ext_edges is not None
        
        # transform the logits to probabilities
        final_graph.x = torch.softmax(final_graph.x, dim=-1)
        final_graph.edge_adjmat = torch.softmax(final_graph.edge_adjmat, dim=-1)
        

        # sample graph at step t-1 from posterior
        generated_graph = self.diffusion_process.sample_posterior(
            original_datapoint =	final_graph,
            current_datapoint =		graph_to_gen,
            t =						denoising_time
        )
        generated_graph = self.diffusion_process_dists.sample_posterior(
            original_datapoint =	final_graph,
            current_datapoint =		generated_graph,
            t =						denoising_time
        )

        if return_onehot:
            generated_graph = to_onehot_all(
                generated_graph,
                **self.data_dims
            )

        if return_masked:
            generated_graph = mask_all(
                generated_graph
            )


        if copy_globals_to_output:
            generated_graph.y = graph_to_gen.y

        return generated_graph
    
    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        conditioning_y: Optional[Tensor]=None,
        return_directed: bool=True,
        save_chains: int=0
    ):
        ########################################################################
        #                        INITIAL SAMPLING SETUP                        #
        ########################################################################

        #########################  SETUP CONDITIONING  #########################

        # TODO: implement the generation chain saving
        do_save_chains = save_chains > 0

        number_of_nodes = self.empirical_sampler(batch_size)

        ##############  SAMPLE THE STARTING SUBGRAPHS (AS NOISE)  ##############
        new_graph: DenseGraph
        new_graph = self.diffusion_process.sample_stationary(
            num_new_nodes = number_of_nodes,
            device = self.device
        )
        new_dists = self.diffusion_process_dists.sample_stationary(
            num_new_nodes = number_of_nodes,
            device = self.device
        )
        new_graph.edge_dist = new_dists.edge_dist
        del new_dists

        # convert the new subgraph to one-hot
        new_graph = to_onehot_all(
            new_graph,
            **self.data_dims
        )

        # copy the global information
        if conditioning_y is not None:
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
                denoising_time = 	t_tensor,
                return_onehot =		True
            )

            # update denoising time (in-place), denoising go down!
            self.change_time(new_graph_dense, t_tensor-1)

        #######################  END OF DENOISING LOOP  ########################
        
        ########################################################################
        #                 TRANSFORM DISTANCE BACK TO POSITIONS                 #
        ########################################################################
        
        new_dist = new_graph_dense.edge_dist
        
        # standard MDS
        self.console_logger.info('Computing distances with MDS fully connected')

        edge_mask = new_graph_dense.edge_mask
        
        updated_node_pos = mds(new_dist.float(), edge_mask=edge_mask)
            
        new_graph_dense.node_pos = updated_node_pos
        del new_graph_dense.edge_dist
        if new_ext_edges is not None:
            del new_ext_edges.edge_dist

        ########################################################################
        #                    MERGE THE OLD AND NEW SUBGRAPHS                   #
        ########################################################################

        output_graph, output_edges = sparsify_data(
            subgraph = new_graph_dense,
            subgraph_nodes_num = number_of_nodes,
        )

        ########################################################################
        #                                RETURN                                #
        ########################################################################

        # replace globals with starting variables, removing time
        if conditioning_y is not None:
            output_graph.y = conditioning_y
        else:
            output_graph.y = None

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

            graph_batch.collapse()

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
    ) -> Tuple[DenseGraph, Tensor, BoolTensor, DenseEdges, DenseEdges]:

    batch_size = curr_graph.num_graphs

    # transform the current graph into a dense representation
    curr_graph_dense = dense.sparse_graph_to_dense_graph(
        sparse_graph =		curr_graph,
        handle_one_hot =    True
    )

    # compute distances between the current nodes
    curr_graph_dense.edge_dist = torch.cdist(curr_graph_dense.node_pos, curr_graph_dense.node_pos)

    return curr_graph_dense


def sparsify_data(
        subgraph: DenseGraph,
        subgraph_nodes_num: IntTensor,
    ) -> SparseGraph:

    ########################  SPARSIFY DENSE SUBGRAPH  #########################
    subgraph = subgraph.clone()

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

    return new_subgraph


###########################  BULK OPERATION METHODS  ###########################

def to_onehot_all(*data, **classes_nums):

    ret_data = []

    for i, d in enumerate(data):
        if isinstance(d, tuple):
            k, d = d
            ret_d = F.one_hot(
                d.long(), num_classes = classes_nums[k]
            ).float()

        elif isinstance(d, DenseEdges):
            ret_d = d.to_onehot(
                num_classes_e =	classes_nums['e']
            )
        
        elif isinstance(d, (DenseGraph, SparseGraph)):
            ret_d = d.to_onehot(
                num_classes_x =	classes_nums['x'],
                num_classes_e =	classes_nums['e'],
            )

        elif isinstance(d, Tensor):
            if d.dtype == torch.bool:
                ret_d = d.unsqueeze(-1)

        elif d is None:
            ret_d = None

        else:
            raise NotImplementedError(f'{i}-th data of type {type(d)} during to_onehot_all')
        
        ret_data.append(ret_d)

    return ret_data


def mask_all(*data, **masks):

    ret_data = []

    for i, d in enumerate(data):
        if isinstance(d, tuple):
            k, d = d
            ret_d = d * masks[k].unsqueeze(-1)
        
        elif isinstance(d, DenseGraph):
            ret_d = d.apply_mask()

        elif d is None:
            ret_d = None

        else:
            raise NotImplementedError(f'{i}-th data of type {type(d)} during mask_all')

        ret_data.append(ret_d)

    return ret_data


#################################  ASSERTIONS  #################################

def assert_is_onehot(*data):

    tensor_dims = {
        'xd': ('dense nodes', 3),
        'xs': ('sparse nodes', 2),
        'ed': ('dense edges', 4),
        'es': ('sparse edges', 2)
    }

    for i, d in enumerate(data):
        if isinstance(d, tuple):

            k: str
            d: Tensor
            k, d = d
            
            assert d.ndim == tensor_dims[k][1], \
                f'Expected {tensor_dims[k][0]} to be of dimension {tensor_dims[k][1]}, got {d.ndim}'

        elif isinstance(d, DenseGraph):
            assert not d.collapsed, \
                'Expected the dense graph to be onehot'
        
        elif isinstance(d, SparseGraph):
            assert_is_onehot(
                ('xs', d.x),
                ('es', d.edge_attr)
            )

        else:
            raise NotImplementedError(f'Expected {i}-th data to be of type tuple, DenseGraph or SparseGraph, got {type(d)}')
            
            