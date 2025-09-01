from typing import Tuple, Dict, Optional, List, Union

import numpy as np

import torch
from torch import Tensor, IntTensor, BoolTensor

from src.datatypes.dense import (
    DenseGraph,
    DenseEdges,
    get_bipartite_edge_mask_dense,
    get_edge_mask_dense
)

from src.noise import reg_diffusion
from src.noise.core import NoiseSchedule, NoiseProcess

from src.noise.schedules import DiffusionProcessException, CosineDiffusionSchedule


from src.noise.discrete_diffusion import (
    UniformDiscreteDiffusionProcess,
    MarginalDiscreteDiffusionProcess
)
from src.noise.multimodal_diffusion import StructuredMultimodalDiffusionProcess

from src.datatypes.dense import dense_to_undirected

class GraphDiffusionProcess(StructuredMultimodalDiffusionProcess):

    def __init__(
            self,
            diffusion_procs_per_data: Dict[str, NoiseProcess]=None,
            undirected: bool=True,
            **kwargs
        ):
        if diffusion_procs_per_data is None:
            diffusion_procs_per_data = {}
        diffusion_procs_per_data.update(kwargs)

        super().__init__(diffusion_procs_per_data=diffusion_procs_per_data)
        self.undirected = undirected


    def map_datapoint_to_dict(self, datapoint: Union[DenseGraph, DenseEdges]) -> Dict[str, Tensor]:
        if isinstance(datapoint, DenseGraph):
            return {k: datapoint[k] for k in self.diffusion_procs_per_data.keys()}
        elif datapoint is None:
            return {}
        else:
            raise DiffusionProcessException(f'Could not map datapoint to dict: {datapoint}')


    def compose_back(self, datapoint: Dict[str, Tensor], other_datapoint: DenseGraph) -> DenseGraph:

        if len(datapoint) == 0:
            return None
        
        attrs = {}
        for key in other_datapoint.keys():
            if key in datapoint:
                value = datapoint[key]
                # if it's an edge attribute and the graph is undirected
                # make sure it's symmetric
                if other_datapoint.is_edge_attr(key) and self.undirected:
                    value = dense_to_undirected(value)
                
                attrs[key] = value
            else:
                attrs[key] = other_datapoint[key]
                
        graph = other_datapoint.__class__(**attrs).apply_mask()

        return graph
    
    
    def kwargs_per_data_from_datapoint(self, datapoint: Union[DenseGraph, DenseEdges]) -> Dict[str, Dict[str, Tensor]]:
        def correct_mask(datapoint, key):
            if DenseGraph.is_node_attr(key):
                return datapoint.node_mask
            elif DenseGraph.is_edge_attr(key):
                return datapoint.get_edge_mask_dense()
            else:
                return None
        
        if isinstance(datapoint, DenseGraph):
            return dict(mask={k: correct_mask(datapoint, k) for k in self.diffusion_procs_per_data.keys()})
        elif datapoint is None:
            return {}
        else:
            raise DiffusionProcessException(f'Could not map datapoint to dict: {datapoint}')
        

    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            num_new_nodes: IntTensor,
            device: torch.device=None,
            shapes: Dict[str, Union[int, Tuple]]=None
        ) -> Tuple[DenseGraph, DenseEdges]:
        
        if shapes is None:
            shapes = {}
        # num new nodes has shape (bs,), and each element
        # is the number of nodes the graph should have
        bs = len(num_new_nodes)
        max_num_nodes = num_new_nodes.max().item()

        # get current device
        device = num_new_nodes.device if device is None else device

        shape_x = [bs, max_num_nodes]
        shape_e = [bs, max_num_nodes, max_num_nodes]

        # compute current node mask
        node_mask = torch.arange(max_num_nodes, device=device) < num_new_nodes.unsqueeze(-1)
        
        sample_kwargs = {}
        for k in self.diffusion_procs_per_data.keys():
            # get additional shapes if specified
            add_shapes = []
            if k in shapes:
                if isinstance(shapes[k], int):
                    add_shapes.append(shapes[k])
                else:
                    add_shapes = shapes[k]
            
            if DenseGraph.is_node_attr(k):
                sample_kwargs[k] = dict(shape=tuple(shape_x + add_shapes))
            elif DenseGraph.is_edge_attr(k):
                sample_kwargs[k] = dict(shape=tuple(shape_e + add_shapes))
            
            
        # sample from stationary distributions
        attrs = super().sample_stationary(sample_kwargs, device=device)

        # compute edge mask
        edge_mask = get_edge_mask_dense(node_mask)

        # transform to undirected graph
        if self.undirected:
            for k in attrs.keys():
                if DenseGraph.is_edge_attr(k):
                    attrs[k] = dense_to_undirected(attrs[k])
                    

        # compose graph
        graph = DenseGraph(
            **attrs,
            node_mask =		node_mask,
            edge_mask =     edge_mask
        ).apply_mask()


        return graph

@reg_diffusion.register('graph_uniform')
class UniformGraphDiffusionProcess(GraphDiffusionProcess):

    def __init__(
            self,
            schedule : NoiseSchedule,
            num_cls_x: int,
            num_cls_e: int,
            undirected: bool=True,
            **kwargs
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """

        super().__init__(
            x=UniformDiscreteDiffusionProcess(schedule, num_cls=num_cls_x),
            edge_adjmat=UniformDiscreteDiffusionProcess(schedule, num_cls=num_cls_e),
            undirected=undirected
        )


@reg_diffusion.register('graph_marginal')
class MarginalGraphDiffusionProcess(GraphDiffusionProcess):

    def __init__(
            self,
            schedule : NoiseSchedule,
            num_cls_x: int,
            num_cls_e: int,
            minimum_number_updates: int=100,
            undirected: bool=True,
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """
        # call super for the NoiseProcess
        super().__init__(
            x=MarginalDiscreteDiffusionProcess(schedule, num_cls=num_cls_x, minimum_number_updates=minimum_number_updates),
            edge_adjmat=MarginalDiscreteDiffusionProcess(schedule, num_cls=num_cls_e, minimum_number_updates=minimum_number_updates),
            undirected=undirected
        )
    
    
    def update(self, x_labels=None, e_labels=None):
        if x_labels is not None:
            self.diffusion_procs_per_data['x'].update(x_labels)
        if e_labels is not None:
            self.diffusion_procs_per_data['edge_adjmat'].update(e_labels)


    def stop_updating(self):
        self.diffusion_procs_per_data['x'].stop_updating()
        self.diffusion_procs_per_data['edge_adjmat'].stop_updating()


        
################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

DIFFUSION_SCHEDULE_COSINE = 'cosine'

DIFFUSION_PROCESS_GRAPH_UNIFORM = 'discrete_uniform'
DIFFUSION_PROCESS_GRAPH_MARGINAL = 'discrete_marginal'

def resolve_graph_diffusion_schedule(name: str) -> type:
    if name == DIFFUSION_SCHEDULE_COSINE:
        return CosineDiffusionSchedule
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion schedule name: {name}')

def resolve_graph_diffusion_process(name: str) -> type:
    if name == DIFFUSION_PROCESS_GRAPH_UNIFORM:
        return UniformGraphDiffusionProcess
    elif name == DIFFUSION_PROCESS_GRAPH_MARGINAL:
        return MarginalGraphDiffusionProcess
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion process name: {name}')