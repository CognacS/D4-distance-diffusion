from typing import List, Optional, Callable, Dict
from tqdm import tqdm

from collections import OrderedDict

import os.path as osp

import torch
from torch_geometric.io.fs import makedirs
from src.data.datasets.core import RawDataset, ProcessedDataset, DataResources, DatasetException, DEFAULT_SPLITS

from src.data.simple_transforms.graph import GraphToNetworkxConverter
from src.data.utils.graphs import get_torch_graphs_stats
from src.datatypes.sparse import SparseGraph


from copy import copy, deepcopy


class GenericGraphsDataset(ProcessedDataset):
    """Here I'm using the InMemoryDataset class from PyTorch Geometric
    to be compatible
    """

    def __init__(
            self,
            root: str,
            raw_graphs_dataset: RawDataset,
            split: Optional[str] = None,
            undirected: bool = True,
            no_self_loops: bool = True,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
        ) -> None:

        self.torch_nx_converter = GraphToNetworkxConverter(
            to_undirected=undirected,
            remove_self_loops=no_self_loops,
        )

        self.raw_graphs_dataset = raw_graphs_dataset

        # call super constructor -> process data
        super().__init__(root, split, transform, pre_transform, pre_filter)

        # remove reference to base dataset, no need for it
        del self.raw_graphs_dataset

        self.load(self.processed_paths[0], SparseGraph)
        self.stats = self.load_file(self.processed_paths[1])


    def subset_from(self, indices: List[int], name: str):

        subset = copy(self)
        subset.root = self.root
        subset.split = name
        makedirs(subset.processed_dir)
        subset.save([self[i] for i in indices], subset.processed_paths[0])
        subset.load(subset.processed_paths[0], SparseGraph)

        new_stats = get_torch_graphs_stats(subset)
        subset.stats = deepcopy(self.stats)
        subset.stats.update(new_stats)
        subset.save_file(subset.stats, subset.processed_paths[1])

        return subset


    @property
    def processed_file_names(self) -> str:
        return 'data.pt', 'stats.json'

    
    def process(self):

        # get all molecules
        graphs = []
        for data in tqdm(self.raw_graphs_dataset, desc='Converting generic graphs to SparseGraphs'):
            
            # convert raw data to SparseGraph
            graph = self.raw_data_to_sparse_graph(data)

            # apply pre_transform if any
            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            graphs.append(graph)

        self.stats = {
            'num_cls_nodes': 1,
            'num_cls_edges': 1,
            'num_cls_properties': graphs[0].y.size(0) if hasattr(graphs[0], 'y') and graphs[0].y is not None else 0,
            **get_torch_graphs_stats(graphs)
        }
        
        self.save(graphs, self.processed_paths[0])
        self.save_file(self.stats, self.processed_paths[1])


    def raw_data_to_sparse_graph(self, sample) -> SparseGraph:
        raise NotImplementedError('This method should be implemented in the subclass')


from src.data.datasets.split import random_split_dataset

class GenericGraphsResources(DataResources):

    def __init__(
            self,
            random_splits: Dict,
            dataset_cfg: Dict,
            dataset_cls,
            root: Optional[str] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
        ):

        super().__init__()
        
        self.root = root
        
        self.dataset_cfg = dataset_cfg
        self.random_splits = random_splits

        self.dataset_cls = dataset_cls

        self.preproc = {
            'pre_transform': pre_transform,
            'pre_filter': pre_filter
        }

        self._prepared = False


    def prepare_data(self):

        ds = self.dataset_cls(self.root, **self.dataset_cfg)

        self.decoder = ds.torch_nx_converter
        self.info_total = ds.stats

        # if there is any pre_transform, solve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])

        try: # try to get the split datasets

            dss = {split: [
                    self.dataset_cls(self.root, split=split, **self.preproc, **self.dataset_cfg)
                ] for split in self.random_splits
            }

        except DatasetException: # if not possible, create the splits
            print('Creating random splits for graphs')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None:
                ds.delete()
                ds = self.dataset_cls(self.root, **self.preproc, **self.dataset_cfg)

            dss = random_split_dataset(ds, self.random_splits)


        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get_dataset(self, split: str, transform=None):
        return self.wrap_dataset(
            self.dataset_cls(self.root, split=split, **self.preproc, **self.dataset_cfg),
            transform=transform
        )


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.get_dataset(split, transform)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for {self.dataset_cls.__name__}')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, decoder, info], splits={list(self.random_splits.keys())}]'