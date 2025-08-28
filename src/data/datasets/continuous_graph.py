import random
import numpy as np
import os
import torch
import pickle
from typing import List, Optional, Tuple, Dict, Any, Callable
from copy import copy, deepcopy
import os.path as osp
from tqdm import tqdm


from torch_geometric.io.fs import makedirs

from torch_geometric.data import Data


from src.data.datasets.core import RawDataset, ProcessedDataset, DataResources, DatasetException, DEFAULT_DATASET_PATH
from src.data.simple_transforms.graph import GraphToNetworkxConverter
from src.data.transforms.direction import MyToUndirected
from src.datatypes.sparse import SparseGraph
from src.data.utils.graphs import get_torch_graphs_stats


class ContinuousGraphsDataset(ProcessedDataset):   ### Inspired by class GenericGraphsDataset(ProcessedDataset)
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
        self.undirected = undirected
        self.no_self_loops = no_self_loops

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
            processed_data = self.dataset_specific_process(data)
            graph = self.raw_data_to_sparse_graph(processed_data)
            
            if self.undirected:
                graph = MyToUndirected()(graph)

            # apply pre_transform if any
            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            graphs.append(graph)

        #node_types = {data['type'] for _, data in graph.nodes(data=True) for graph in self.raw_graphs_dataset if 'type' in data}
        #edge_types = {data['type'] for _, _, data in self.raw_graphs_dataset[0].edges(data=True) if 'type' in data}
        
        node_types = set()
        edge_types = set()
        for graph in self.raw_graphs_dataset:
            node_types.update({data['type'] for _, data in graph.nodes(data=True) if 'type' in data})
            edge_types.update({data['type'] for _, _, data in graph.edges(data=True) if 'type' in data})
            

        self.stats = {
            'num_cls_nodes': len(set(node_types)),
            'num_cls_edges': len(set(edge_types)),
            'num_cls_properties': graphs[0].y.size(0) if hasattr(graphs[0], 'y') and graphs[0].y is not None else 0,
            **get_torch_graphs_stats(graphs)
        }
        #print(f"dbg stats {self.stats}")
        self.define_stats()
        
        self.save(graphs, self.processed_paths[0])
        self.save_file(self.stats, self.processed_paths[1])


    def raw_data_to_sparse_graph(self, sample) -> SparseGraph:
        raise NotImplementedError('This method should be implemented in the subclass')

    def dataset_specific_process(self, sample) -> SparseGraph:
        raise NotImplementedError('This method should be implemented in the subclass')
    
    def define_stats(self):
        pass
    

from src.data.datasets.split import random_split_dataset

class ContinuousGraphsResources(DataResources):

    def __init__(
            self,
            which_dataset: str,
            random_splits: Dict,
            dataset_cfg: Dict,
            dataset_cls,
            root: Optional[str] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
        ):

        super().__init__()
        
        self.which_dataset = which_dataset
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

        ds = self.dataset_cls(self.which_dataset, self.root, **self.dataset_cfg)

        self.decoder = ds.torch_nx_converter
        self.info_total = ds.stats

        # if there is any pre_transform, solve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])

        try: # try to get the split datasets

            dss = {split: [
                    self.dataset_cls(self.which_dataset, self.root, split=split, **self.preproc, **self.dataset_cfg)
                ] for split in self.random_splits
            }

        except DatasetException: # if not possible, create the splits
            print('Creating random splits for graphs')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None:
                ds.delete()
                ds = self.dataset_cls(self.which_dataset, self.root, **self.preproc, **self.dataset_cfg)

            dss = random_split_dataset(ds, self.random_splits)


        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True
        
        return dss


    def get_dataset(self, split: str, transform=None):
        return self.wrap_dataset(
            self.dataset_cls(self.which_dataset, self.root, split=split, **self.preproc, **self.dataset_cfg),
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