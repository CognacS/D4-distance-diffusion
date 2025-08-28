from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Tuple
import itertools

import torch
from torch import Tensor

from torch_geometric.data import Batch
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader.dataloader import Collater


from src.datatypes.sparse import SparseGraph, SparseEdges
from src.noise.core import TimeSampler, NoiseProcess
from src.noise.config_support import build_noise_process
from src.noise.removal import (
    resolve_removal_process,
    resolve_removal_schedule
)

from src.noise.timesample import resolve_timesampler

from copy import deepcopy


class SubgraphSequenceSampler(BaseTransform):

    def __init__(self, removal_process : NoiseProcess, time_sampler : TimeSampler, num_sequences : int=1):
        self.removal_process = removal_process
        self.time_sampler = time_sampler
        self.num_sequences = num_sequences

    def __call__(self, graph: SparseGraph) -> SparseGraph:

        # 1 - get original number of nodes (remember: can also be retrieved by
        # accessing the graph's global properties)
        orig_num_nodes = graph.num_nodes
        max_time = self.removal_process.get_max_time(n0=orig_num_nodes)

        examples = {'batch': [], 'surv_batch': [], 'remv_batch': [], 'remv_edges_ba': []}

        for _ in range(self.num_sequences):

            curr_graph = graph.clone()

            # initialize graph (e.g. with ordering)
            curr_graph = self.removal_process.sample_from_original(
                original_datapoint =    curr_graph,
                t =                     0,
                max_time = 				max_time,
                n0 = 					orig_num_nodes
            )


            norm_t = self.removal_process.normalize_reverse_time(t=0, n0=orig_num_nodes)
            curr_graph.global_t = torch.tensor([0])
            curr_graph.global_rev_t = torch.tensor([norm_t])
            curr_graph.global_n0 = torch.tensor([orig_num_nodes])
            curr_graph.global_nt = torch.tensor([curr_graph.num_nodes])

            for t in range(1, max_time+1):

                examples['batch'].append(curr_graph)
                surv_batch: SparseGraph
                remv_batch: SparseGraph
                remv_edges_ba: Tuple[Tensor, Tensor]

                surv_batch, remv_batch, _, remv_edges_ba = self.removal_process.sample_next(
                    current_datapoint =     curr_graph,
                    t = 					t,
                    max_time = 				max_time,
                    n0 = 					orig_num_nodes,
                    split =                 True
                )
                

                norm_t = self.removal_process.normalize_reverse_time(t=t, n0=surv_batch.global_n0)

                surv_batch.global_t = torch.tensor([t])
                surv_batch.global_rev_t = torch.tensor([norm_t])
                surv_batch.global_n0 = torch.tensor([orig_num_nodes])
                surv_batch.global_nt = torch.tensor([surv_batch.num_nodes])

                surv_batch.global_t = torch.tensor([t])
                remv_batch.global_rev_t = torch.tensor([norm_t])
                remv_batch.global_n0 = torch.tensor([orig_num_nodes])
                remv_batch.global_nt = torch.tensor([remv_batch.num_nodes])

                examples['surv_batch'].append(surv_batch)
                examples['remv_batch'].append(remv_batch)
                examples['remv_edges_ba'].append(remv_edges_ba)

                curr_graph = surv_batch

        return examples
    
    @classmethod
    def create_subsequence_sampler(
            cls,
            process_config: Dict,
            num_sequences: int=1
        ) -> SubgraphSequenceSampler:

        removal_process, timesampler = build_noise_process(
            process_config,
            resolve_removal_process,
            resolve_removal_schedule,
            resolve_timesampler
        )

        transform = SubgraphSequenceSampler(
            removal_process=removal_process,
            time_sampler=timesampler,
            num_sequences=num_sequences
        )

        return transform
    

class SubsequenceCollater(Collater):

    def __call__(self, batch):
        elem = batch[0]
        
        if isinstance(elem, Mapping):
            subelem = next(iter(elem.values()))

            if isinstance(subelem, list):
                # concatenate all lists at the given key
                batch = {key: list(itertools.chain.from_iterable([b[key] for b in batch])) for key in elem.keys()}
                # apply collate function to each key
                batch = {key: self(subbatch) for key, subbatch in batch.items()}

                return batch
            
        return super().__call__(batch)



def resolve_collater(collater_name: str):
    if collater_name == 'subsequence_collater':
        return SubsequenceCollater
    else:
        return Collater