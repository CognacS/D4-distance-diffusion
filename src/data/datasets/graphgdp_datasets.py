from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm
from collections import OrderedDict

import os
import os.path as osp

import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse, remove_self_loops

from src.datatypes.sparse import SparseGraph

from src.data.datasets.core import RawDataset, DEFAULT_DATASET_PATH
from src.data.datasets.split import split_dataset, random_split_dataset
from src.data.datasets.generic_graph import GenericGraphsDataset, GenericGraphsResources

from torch_geometric.data import download_url
from torch_geometric.io.fs import makedirs

import src.data.utils.csv as csvutils

import networkx as nx
from torch_geometric.utils import from_networkx

from copy import copy

SPECTRE_RAW_REPO_URL = 'https://github.com/GRAPH-0/GraphGDP/raw/main/data/raw/'
SUPPORTED_DATASETS = {
    'community-small': 'Community_small.pkl',
    'enzymes': 'ENZYMES.pkl',
    'ego': 'Ego.pkl',
    'ego-small': 'Ego_small.pkl',
    'grid': 'graphs.pkl'
}


class BaseGraphgdpRawDataset(RawDataset):
    """Base class for raw datasets stored in Spectre. This work provides many properties
    for graphs in the dataset, that is, indexing the dataset provides (in order):
    0 - the graph adjacency matrix
    1 - eigenvalues of the graph Laplacian
    2 - eigenvectors of the graph Laplacian
    3 - number of nodes
    In addition, there are some general properties:
    0 - maximum eigenvalue
    1 - minimum eigenvalue
    2 - same sample
    3 - maximum number of nodes
    """

    def __init__(
            self,
            which_dataset: str,
            root: Optional[str] = None,
            split: Optional[str] = None,
            pre_transform=None,
            pre_filter=None
        ):

        if which_dataset not in SUPPORTED_DATASETS:
            raise ValueError(f'Dataset {which_dataset} not supported. Supported datasets are: {list(SUPPORTED_DATASETS.keys())}')
        self.which_dataset = which_dataset

        if root is None:
            root = osp.join(DEFAULT_DATASET_PATH, which_dataset)

        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter)

        if not hasattr(self, 'data'):
            self.data = self.load(self.raw_paths[0])


    def subset_from(self, indices: List[int], name: str):

        subset = copy(self)
        subset.root = self.root
        subset.split = name

        makedirs(subset.raw_dir)

        subset.data = [self.data[i] for i in indices]
        subset.save(subset.data, subset.raw_paths[0])

        return subset

    
    @property
    def raw_file_names(self):
        return [SUPPORTED_DATASETS[self.which_dataset]]
    

    def download(self):
        # download data file
        data_raw_file = download_url(f'{SPECTRE_RAW_REPO_URL}{SUPPORTED_DATASETS[self.which_dataset]}', self.raw_dir)

        data_raw = self.load(data_raw_file)

        self.data = data_raw


    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    

class BaseGraphdgpProcessedDataset(GenericGraphsDataset):

    def __init__(
            self,
            which_dataset: str,
            root: Optional[str] = None,
            split: Optional[str] = None,
            remove_loops: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = osp.join(DEFAULT_DATASET_PATH, which_dataset)

        raw_graphs_dataset = BaseGraphgdpRawDataset(
            which_dataset,
            root=root,
            pre_transform=pre_transform_raw,
            pre_filter=pre_filter_raw
        )

        self.remove_self_loops = remove_loops

        super().__init__(
            root, raw_graphs_dataset=raw_graphs_dataset,
            split=split, transform=transform,
            pre_transform=pre_transform, pre_filter=pre_filter
        )


    def raw_data_to_sparse_graph(self, sample: nx.Graph) -> SparseGraph:

        g = from_networkx(sample)
        num_nodes = g.num_nodes

        x = torch.zeros(num_nodes, dtype=torch.int64)
        edge_index = g.edge_index if not self.remove_self_loops else remove_self_loops(g.edge_index)[0]
        edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.int64)

        graph = SparseGraph(x, edge_index, edge_attr)

        return graph


class CommunitySmall(BaseGraphdgpProcessedDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            remove_loops: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        super().__init__(
            'community-small', root, split=split, remove_loops=remove_loops,
            pre_transform_raw=pre_transform_raw, pre_filter_raw=pre_filter_raw,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

class Enzymes(BaseGraphdgpProcessedDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            remove_loops: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        if root is None:
            root = osp.join(DEFAULT_DATASET_PATH, 'enzymes')
        super().__init__(
            'enzymes', root, split=split, remove_loops=remove_loops,
            pre_transform_raw=pre_transform_raw, pre_filter_raw=pre_filter_raw,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

class Ego(BaseGraphdgpProcessedDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            remove_loops: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        super().__init__(
            'ego', root, split=split, remove_loops=remove_loops,
            pre_transform_raw=pre_transform_raw, pre_filter_raw=pre_filter_raw,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

class EgoSmall(BaseGraphdgpProcessedDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            remove_loops: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        super().__init__(
            'ego-small', root, split=split, remove_loops=remove_loops,
            pre_transform_raw=pre_transform_raw, pre_filter_raw=pre_filter_raw,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

class Grid(BaseGraphdgpProcessedDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            remove_loops: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        super().__init__(
            'grid', root, split=split, remove_loops=remove_loops,
            pre_transform_raw=pre_transform_raw, pre_filter_raw=pre_filter_raw,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )


from src.data.datasets import reg_dataresources

@reg_dataresources.register('community-small')
class CommunitySmallResources(GenericGraphsResources):
    def __init__(
            self, random_splits: Dict, root: str=None, remove_loops: bool = True,
            pre_transform=None, pre_filter=None
        ):
        dataset_cfg = {'remove_loops': remove_loops}
        super().__init__(random_splits, dataset_cfg, CommunitySmall, root, pre_transform, pre_filter)

@reg_dataresources.register('enzymes')
class EnzymesResources(GenericGraphsResources):
    def __init__(
            self, random_splits: Dict, root: str=None, remove_loops: bool = True,
            pre_transform=None, pre_filter=None
        ):
        dataset_cfg = {'remove_loops': remove_loops}
        super().__init__(random_splits, dataset_cfg, Enzymes, root, pre_transform, pre_filter)

@reg_dataresources.register('ego')
class EgoResources(GenericGraphsResources):
    def __init__(
            self, random_splits: Dict, root: str=None, remove_loops: bool = True,
            pre_transform=None, pre_filter=None
        ):
        dataset_cfg = {'remove_loops': remove_loops}
        super().__init__(random_splits, dataset_cfg, Ego, root, pre_transform, pre_filter)

@reg_dataresources.register('ego-small')
class EgoSmallResources(GenericGraphsResources):
    def __init__(
            self, random_splits: Dict, root: str=None, remove_loops: bool = True,
            pre_transform=None, pre_filter=None
        ):
        dataset_cfg = {'remove_loops': remove_loops}
        super().__init__(random_splits, dataset_cfg, EgoSmall, root, pre_transform, pre_filter)