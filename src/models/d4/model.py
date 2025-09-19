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
from src.noise.core import dict_of_noise_processes_from_config
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
from src.noise.graph_diffusion import GraphDiffusionProcess
from src.noise.multimodal_diffusion import ChainedNoiseProcess

from src.noise.discrete_diffusion import DiscreteDiffusionProcess
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
        
        self.distance_output_mode = self.denoising_config.get('distance_output_mode', 'single')
        assert self.distance_output_mode in ['single', 'periscopic', 'conditional'], \
            "distance_output_mode must be one of 'single', 'periscopic', or 'conditional'"
            
        self.always_apply_mds = self.denoising_config.get('always_apply_mds', False)

        #######################  GRAPHS DIMENSIONS SETUP  ######################
        # setup model input and output dimensions (based on the dataset)
        self.data_dims = {
            'x': dataset_info['num_cls_nodes'],
            'edge_adjmat': dataset_info['num_cls_edges'],
            'node_charges': dataset_info['num_cls_charges'],
            'y': 0 if discard_conditioning else dataset_info['dim_targets'],
            "edge_dist": 1
        }

        self.data_dims['edge_adjmat'] += 1  # account for no-edge class

        if received_dims:
            self.received_dims = deepcopy(received_dims)
            self.received_dims['edge_adjmat'] += 1
        else:
            self.received_dims = self.data_dims

        # increase dimensions based on additional features (creates a copy)
        self.augmented_dims = increase_dims_list(self.received_dims, self.additional_features)
        
        self.augmented_dims = increase_dims(self.augmented_dims, {
            'y': self.time_enc_dim if self.using_pos_emb() else 1
            # account for diffusion time as a global y feature
        })
        
        self.out_dims = deepcopy(self.data_dims)
        if not self.distance_output_mode == 'single':
            self.out_dims['edge_dist'] = self.data_dims['edge_adjmat']

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
        self.denoising_model = reg_architectures.get_instance_from_cfg(
            config =        self.denoising_config.architecture,
            input_dims =    self.augmented_dims,
            output_dims =   self.out_dims,
        )

        ######################  BUILD DIFFUSION PROCESS  #######################

        self.diffusion_timesampler: TimeSampler
        self.diffusion_timesampler = reg_timesampler.get_instance_from_cfg(
            self.diffusion_config.timesampler
        )
        
        # prepare additional parameters for each process (number of classes)
        process_kwargs = {
            'x': {'num_cls': self.data_dims['x']},
            'edge_adjmat': {'num_cls': self.data_dims['edge_adjmat']},
            'node_charges': {'num_cls': self.data_dims['node_charges']}
        }
        # directly add marginals if they are available in the dataset_info
        if 'marginals' in dataset_info:
            marginals = dataset_info['marginals']
            process_kwargs['x']['marginals'] = marginals['x']
            process_kwargs['edge_adjmat']['marginals'] = marginals['edge_attr']
            process_kwargs['node_charges']['marginals'] = marginals['node_charges']
        
        # build all noise processes
        diffusion_procs_per_data = dict_of_noise_processes_from_config(
            config = self.diffusion_config.process.params,
            process_kwargs=process_kwargs
        )
        # check that all required processes are present
        assert all(s in diffusion_procs_per_data for s in ['x', 'edge_adjmat', 'edge_dist', 'node_charges']), \
            "Diffusion processes for x, edge_adjmat, edge_dist, node_charges must be specified in D4Model"
        # build the graph diffusion process
        # this computes all processes at the same time in a single call
        
        if self.distance_output_mode == 'single':
            # here there is no internal conditioning, then aggregate all processes together
            self.diffusion_process = GraphDiffusionProcess(
                diffusion_procs_per_data=diffusion_procs_per_data,
            )
        else:
            # here first the structure is compute, then the distances conditioned on the structure
            diffusion_struct = {k: diffusion_procs_per_data[k] for k in ['x', 'edge_adjmat', 'node_charges']}
            diffusion_dist = {'edge_dist': diffusion_procs_per_data['edge_dist']}
            
            # the samples are combined by just adding the distances
            def combine_stationary(x_before, x_after, *args, **kwargs):
                x_before.edge_dist = x_after.edge_dist
                return x_before

            # after sampling the structure, distances are sampled conditioned on the structure
            def chain_sample_posterior(datapoint, *args, **kwargs):
                datapoint.edge_dist = self.postprocess_distances(
                    datapoint.edge_dist, datapoint.edge_adjmat, datapoint.edge_mask
                )
                
                return datapoint
            
            # chain the two processes one after the other
            self.diffusion_process = ChainedNoiseProcess(
                noise_process_before = GraphDiffusionProcess(diffusion_struct, undirected=True),
                noise_process_after = GraphDiffusionProcess(diffusion_dist, undirected=True),
                combine_stationary = combine_stationary,
                chain_sample_posterior = chain_sample_posterior
            )


        ######################  BUILD LOSSES AND METRICS  ######################
        self.train_loss = TrainLossDistance(
            **self.denoising_config.loss
        )

        metrics = nn.ModuleDict({
            labels.DENOISE_CE_X: MeanMetric(),
            labels.DENOISE_CE_C: MeanMetric(),
            labels.DENOISE_CE_E: MeanMetric(),
            labels.DENOISE_ACC_X: MulticlassAccuracy(num_classes=self.data_dims['x'], validate_args=False),
            labels.DENOISE_ACC_E: MulticlassAccuracy(num_classes=self.data_dims['edge_adjmat'], validate_args=False),
            labels.DENOISE_ACC_C: MulticlassAccuracy(num_classes=self.data_dims['node_charges'], validate_args=False),
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
    
    
    ############################################################################
    #                          ADDITIONAL TECHNIQUES                           #
    ############################################################################
    
    def aggregate_periscopic_distances(self, edge_dist: Tensor, edge_adjmat: Tensor) -> Tensor:
        """Periscopic mode adds the distances depending on the edge types.
        This is based on the assumption that, as the type increases, the distance decreases.
        For example, if bonds are [0,1,2,3], where 0 is no-bond, then the distance decreases,
        with the 3-bond being the shortest distance.
        Additionally, gradient is stopped for greater types, and only flows for the true type.
        
        edge_dist: Tensor
            shape (B,N,N,num_types)
        edge_adjmat: Tensor
            shape (B,N,N) or (B,N,N,num_types) in logits/onehot format
        """
        if edge_adjmat.ndim == 4:
            # if edge_adjmat is in logits/onehot format, convert to indices
            edge_adjmat = edge_adjmat.argmax(dim=-1)
        
        # gather greater types
        # e.g., if type=1, gather types [2,3,...]
        gt_types_mask = torch.arange(0, self.data_dims['edge_adjmat'], device=edge_adjmat.device).view(1,1,1,-1) # shape (1,1,1,num_types)
        gt_types_mask = gt_types_mask > edge_adjmat.unsqueeze(-1) # shape (B,N,N,num_types)
        
        # sum greater types + detach gradient
        out_dist = (edge_dist * gt_types_mask).detach().sum(dim=-1)
        
        # add true type distance with gradient
        out_dist = out_dist + torch.gather(edge_dist, -1, edge_adjmat.unsqueeze(-1)).squeeze(-1)
        
        return out_dist
    
    
    def aggregate_periscopic_distances_alt(self, edge_dist: Tensor, edge_adjmat: Tensor) -> Tensor:
        """Periscopic mode adds the distances depending on the edge types.
        This is based on the assumption that, as the type increases, the distance decreases.
        For example, if bonds are [0,1,2,3], where 0 is no-bond, then the distance decreases,
        with the 3-bond being the shortest distance.
        Changes to above version: gradient is NOT stopped for greater types.
        
        edge_dist: Tensor
            shape (B,N,N,num_types)
        edge_adjmat: Tensor
            shape (B,N,N) or (B,N,N,num_types) in logits/onehot format
        """
        if edge_adjmat.ndim == 4:
            # if edge_adjmat is in logits/onehot format, convert to indices
            edge_adjmat = edge_adjmat.argmax(dim=-1)
        
        # gather greater types
        # e.g., if type=1, gather types [2,3,...]
        gt_types_mask = torch.arange(0, self.data_dims['edge_adjmat'], device=edge_adjmat.device).view(1,1,1,-1) # shape (1,1,1,num_types)
        gt_types_mask = gt_types_mask > edge_adjmat.unsqueeze(-1) # shape (B,N,N,num_types)
        
        # sum greater types + detach gradient
        out_dist = (edge_dist * gt_types_mask).sum(dim=-1)
        
        # add true type distance with gradient
        out_dist = out_dist + torch.gather(edge_dist, -1, edge_adjmat.unsqueeze(-1)).squeeze(-1)
        
        return out_dist
        
        
    def aggregate_conditional_distances(self, edge_dist: Tensor, edge_adjmat: Tensor) -> Tensor:
        """Conditional mode returns the distance corresponding to the edge type.
        
        edge_dist: Tensor
            shape (B,N,N,num_types)
        edge_adjmat: Tensor
            shape (B,N,N) or (B,N,N,num_types) in logits/onehot format
        """
        if edge_adjmat.ndim == 4:
            # if edge_adjmat is in logits/onehot format, convert to indices
            edge_adjmat = edge_adjmat.argmax(dim=-1)
        
        out_dist = edge_dist[edge_adjmat]
        
        return out_dist
        
        
    def postprocess_distances(self, edge_dist: Tensor, edge_adjmat: Tensor, edge_mask: BoolTensor) -> None:
        
        if self.distance_output_mode == 'periscopic':
            edge_dist = self.aggregate_periscopic_distances_alt(edge_dist, edge_adjmat)
        elif self.distance_output_mode == 'conditional':
            edge_dist = self.aggregate_conditional_distances(edge_dist, edge_adjmat)
        
        # apply non-linearity to distances:
        # silu such that: 0 is reachable (with softplus it is only asymptotically)
        # and negative distances are unlikely
        #edge_dist = torch.nn.functional.silu(edge_dist)
        # mask distances
        #edge_dist = edge_dist * edge_mask.float()
        
        if self.always_apply_mds:
            computed_node_pos = mds(edge_dist.float(), edge_mask=edge_mask)
            # recompute distances
            edge_dist = torch.cdist(computed_node_pos, computed_node_pos)
            
            
        
        return edge_dist


    ############################################################################
    #                 SHORTHANDS FOR TRAINING/VALIDATION STEPS                 #
    ############################################################################
    
    def compute_true_pred_denoising(
            self,
            batch_to_generate: SparseGraph
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate the true and predicted nodes and egdes for the denoising
        process. The flow is as follows:
        2 - densify batch_to_generate as a DenseGraph
        3 - sample the diffusion process at uniformly random timesteps to
            make a noisy version of batch_to_generate (again requires onehot
            and masking)
        4 - try to denoise the above data which include the batch_to_generate
        5 - flatten and pack the true and predicted nodes and edges
        The final order is: nodes, edges
        Predicted values are in expanded form, true values are collapsed. This is
        ideal for the cross-entropy loss function.

        Parameters
        ----------
        batch_to_generate : SparseGraph
            sparse graph with collapsed classes (i.e. class indices). This graph
            will be noised and denoised.

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
        true_c = batch_to_generate_dense.node_charges.argmax(dim=-1)[node_mask]
        true_dist = batch_to_generate_dense.edge_dist[triang_edge_mask]
            
        ##################  UPDATE MARGINAL PROCESS IF NEEDED  #################
        
        # true_data = {'x': true_x, 'edge_adjmat': true_e, 'node_charges': true_c}
        # for data in ['x', 'edge_adjmat', 'node_charges']:
            
        #     process = self.diffusion_process.diffusion_procs_per_data[data]
        #     true_d = true_data[data]

        #     if hasattr(process, 'update'):
        #         process.update(labels=true_d)

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
        noisy_graph: DenseGraph = self.diffusion_process.sample_from_original(batch_to_generate_dense, t=u)

        # onehot and mask the noisy data again (to remove the fake noisy components)
        onehot_data = to_onehot_data(noisy_graph, **self.data_dims)

        self.add_additional_features(onehot_data)

        noisy_batch_to_generate_dense_onehot = mask_data(onehot_data)

        #####################  PREDICT THE ORIGINAL GRAPH  #####################
        gen_batch_dense: DenseGraph
        gen_batch_dense = self.denoising_model(
            graph = noisy_batch_to_generate_dense_onehot
        )
        
        # postprocess distances with the true edge labels
        # this mode is in teacher forcing, so we don't
        # make the training noisy
        gen_batch_dense.edge_dist = self.postprocess_distances(
            gen_batch_dense.edge_dist,
            batch_to_generate_dense.edge_adjmat,
            batch_to_generate_dense.edge_mask
        )

        pred_x = gen_batch_dense.x[node_mask]
        pred_e = gen_batch_dense.edge_adjmat[triang_edge_mask]
        pred_c = gen_batch_dense.node_charges[node_mask]
        pred_dist = gen_batch_dense.edge_dist[triang_edge_mask]
        
        ###########################  FINAL PACKING  ############################

        true_values = [true_x, true_e, true_dist, true_c]
        pred_values = [pred_x, pred_e, pred_dist, pred_c, node_mask, triang_edge_mask, gen_batch_dense.edge_dist]
        
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
        metrics[labels.DENOISE_CE_C](loss_logs[labels.DENOISE_CE_C])
        metrics[labels.DENOISE_MSE_DIST](loss_logs[labels.DENOISE_MSE_DIST])
        metrics[labels.DENOISE_TOTAL](loss_logs[labels.DENOISE_TOTAL])
        if pred_values[0].numel() > 0:
            metrics[labels.DENOISE_ACC_X](pred_values[0], true_values[0])
        if pred_values[1].numel() > 0:
            metrics[labels.DENOISE_ACC_E](pred_values[1], true_values[1])
        if pred_values[2].numel() > 0:
            metrics[labels.DENOISE_MAE_DIST](pred_values[2], true_values[2])
        if pred_values[3].numel() > 0:
            metrics[labels.DENOISE_ACC_C](pred_values[3], true_values[3])

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
            
        # for data in ['x', 'edge_adjmat', 'node_charges']:
        #     # stop updating marginals at the end of the first training epoch
        #     process = self.diffusion_process.diffusion_procs_per_data[data]

        #     if hasattr(process, 'update'):
        #         process.stop_updating()
        
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

        return torch.optim.AdamW(
            self.denoising_model.parameters(), **self.optimizer_config
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
        final_graph = self.denoising_model(
            graph = augmented_graph_to_gen
        )
        
        # transform the logits to probabilities
        final_graph.x = torch.softmax(final_graph.x, dim=-1)
        final_graph.node_charges = torch.softmax(final_graph.node_charges, dim=-1)
        final_graph.edge_adjmat = torch.softmax(final_graph.edge_adjmat, dim=-1)
        
        # postprocess distances if needed (single case)
        # in other cases, it is already inside sample_posterior
        # if self.distance_output_mode == 'single':
        #     final_graph.edge_dist = self.postprocess_distances(
        #         final_graph.edge_dist,
        #         final_graph.edge_adjmat,
        #         final_graph.edge_mask
        #     )

        # sample graph at step t-1 from posterior
        generated_graph = self.diffusion_process.sample_posterior(
            original_datapoint =	final_graph,
            current_datapoint =		graph_to_gen,
            t =						denoising_time
        )

        if return_onehot:
            generated_graph = to_onehot_data(generated_graph, **self.data_dims)

        if return_masked:
            generated_graph = mask_data(generated_graph)


        if copy_globals_to_output:
            generated_graph.y = graph_to_gen.y

        return generated_graph
    
    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        conditioning_elems: Optional[Tensor]=None,
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

        # convert the new subgraph to one-hot
        new_graph = to_onehot_data(
            new_graph,
            **self.data_dims
        )

        # copy the global information
        if conditioning_elems is not None:
            new_graph.y = conditioning_elems.clone()
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
            new_graph_dense = self.forward_denoising(
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
        # include distances as global attrs for slicing and metrics
        new_graph_dense.global_dist = new_graph_dense.edge_dist
        del new_graph_dense.edge_dist

        ########################################################################
        #                    MERGE THE OLD AND NEW SUBGRAPHS                   #
        ########################################################################

        output_graph = sparsify_data(
            subgraph = new_graph_dense,
            subgraph_nodes_num = number_of_nodes,
        ).to('cpu')

        ########################################################################
        #                                RETURN                                #
        ########################################################################

        # replace globals with starting variables, removing time
        if conditioning_elems is not None:
            output_graph.y = conditioning_elems
        else:
            output_graph.y = None

        return output_graph
            
    
    @torch.no_grad()
    def sample(
            self,
            num_samples: int,
            conditioning_elems: Optional[Dict]=None,
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
                conditioning_elems=conditioning_elems[batch_idx] if conditioning_elems is not None else None
            )

            graph_batch.collapse('x', 'edge_attr', 'node_charges')

            output_batch = graph_batch.to_data_list()

            output_batch = [g.cpu() for g in output_batch]

            samples.extend(output_batch)

            samples_left_to_generate -= to_generate
            batch_idx += 1
            self.console_logger.info(f'Generated {len(samples)}/{num_samples} graphs')
            
        #self.log_sampled_graphs(samples, 'molecules')

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

def to_onehot_data(d, **classes_nums):

    if isinstance(d, tuple):
        k, d = d
        ret_d = F.one_hot(
            d.long(), num_classes = classes_nums[k]
        ).float()

    elif isinstance(d, DenseGraph):
        ret_d = d.to_onehot(
            {key: classes_nums[key] for key in ['x', 'edge_adjmat', 'node_charges']}
        )

    elif isinstance(d, Tensor):
        if d.dtype == torch.bool:
            ret_d = d.unsqueeze(-1)

    elif d is None:
        ret_d = None

    else:
        raise NotImplementedError(f'Data of type {type(d)} during to_onehot_data')

    return ret_d


def mask_data(d, **masks):

    if isinstance(d, tuple):
        k, d = d
        ret_d = d * masks[k].unsqueeze(-1)
    
    elif isinstance(d, DenseGraph):
        ret_d = d.apply_mask()

    elif d is None:
        ret_d = None

    else:
        raise NotImplementedError(f'Data of type {type(d)} during mask_data')

    return ret_d
            