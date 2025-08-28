from __future__ import annotations

from typing import Dict

import torch

from torch_geometric.transforms import BaseTransform


from src.datatypes.sparse import SparseGraph
from src.noise.core import TimeSampler, NoiseProcess
from src.noise.config_support import build_noise_process
from src.noise.removal import (
    resolve_removal_process,
    resolve_removal_schedule
)

from src.noise.timesample import resolve_timesampler


class SubgraphSampler(BaseTransform):

    def __init__(self, removal_process : NoiseProcess, time_sampler : TimeSampler, dont_sample_last: bool = True):
        self.removal_process = removal_process
        self.time_sampler = time_sampler
        self.dont_sample_last = dont_sample_last

    def __call__(self, graph: SparseGraph) -> SparseGraph:
        """Sample a subgraph of an original graph.
        A subgraph is sampled using the removal_process, which holds the
        information on how to remove pieces of information (such as nodes) from
        the original graph. The timestep from which a subgraph originate is
        sampled from the time_sampler.
        Parameters
        ----------
        graph : SparseGraph
            graph to be subsampled
        Returns
        -------
        subgraph : SparseGraph
            subgraph of the original graph. The subgraph can be any subgraph,
            i.e. from the original graph to an empty graph.
        """

        # 1 - get original number of nodes (remember: can also be retrieved by
        # accessing the graph's global properties)
        orig_num_nodes = graph.num_nodes

        # 2 - sample a random timestep (optional, if using an adaptive time
        # method use the additional argument as kwargs)
        max_time = self.removal_process.get_max_time(n0=orig_num_nodes)
        if self.dont_sample_last:
            max_time = max_time - 1

        sampled_time = self.time_sampler.sample_time(
            max_time = max_time,
            n0 = orig_num_nodes
        )

        # 3 - sample subgraph
        subgraph: SparseGraph = self.removal_process.sample_from_original(
            original_datapoint =	graph,
            t = 					sampled_time,
            max_time = 				orig_num_nodes,
            n0 = 					orig_num_nodes
        )

        # 4 - setup values
        subgraph.global_nt = torch.tensor([subgraph.num_nodes])
        subgraph.global_n0 = torch.tensor([orig_num_nodes])
        subgraph.global_t = sampled_time

        return subgraph


    @classmethod
    def create_subgraph_sampler(
            cls,
            process_config: Dict
        ) -> SubgraphSampler:

        removal_process, timesampler = build_noise_process(
            process_config,
            resolve_removal_process,
            resolve_removal_schedule,
            resolve_timesampler
        )

        transform = SubgraphSampler(
            removal_process=removal_process,
            time_sampler=timesampler
        )

        return transform