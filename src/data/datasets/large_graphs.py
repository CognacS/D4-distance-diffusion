from typing import List, Optional, Tuple, Dict, Any, Type, Callable
from tqdm import tqdm
from collections import OrderedDict

import os
import os.path as osp

import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse, remove_self_loops

from src.datatypes.sparse import SparseGraph

from src.data.datasets.core import RawDataset, DEFAULT_DATASET_PATH, DataResources
from src.data.datasets.split import split_dataset, random_split_dataset
from src.data.datasets.generic_graph import GenericGraphsDataset, GenericGraphsResources

from torch_geometric.data import download_url
from torch_geometric.io.fs import makedirs

import src.data.utils.csv as csvutils

import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx, from_networkx

from src.data.datasets import reg_dataresources

from copy import copy

G2G_RAW_REPO_URL = 'https://github.com/abojchevski/graph2gauss/raw/master/data/'

SUPPORTED_DATASETS = {
    #'cora': 'cora.npz',
    'pubmed': 'pubmed.npz',
    'citeseer': 'citeseer.npz',
    'dblp': 'dblp.npz',
    'cora': 'cora_ml.npz',
}

class LargeGraphRawDataset(RawDataset):
    """Dataset class for the raw large graphs from the Graph2Gauss repository.
    """

    def __init__(
            self,
            which_dataset: str,
            root: Optional[str] = None,
            split: Optional[str] = None,
            to_undirected: bool = True,
            biggest_conn_comp: bool = True,
            pre_transform=None,
            pre_filter=None
        ):

        if which_dataset not in SUPPORTED_DATASETS:
            raise ValueError(f'Dataset {which_dataset} not supported. Supported datasets are: {list(SUPPORTED_DATASETS.keys())}')
        self.which_dataset = which_dataset

        self.to_undirected = to_undirected
        self.biggest_conn_comp = biggest_conn_comp

        if root is None:
            root = osp.join(DEFAULT_DATASET_PATH, which_dataset)

        super().__init__(root, pre_transform=pre_transform, pre_filter=pre_filter)

        if not hasattr(self, 'data'):
            self.data = self.load(self.raw_paths[1])


    def subset_from(self, indices: List[int], name: str):
        raise NotImplementedError('Subsetting not implemented for single graph datasets.')

    
    @property
    def raw_file_names(self):
        return [SUPPORTED_DATASETS[self.which_dataset], 'data.pkl']
    

    def download(self):
        # download data file
        data_raw_file = download_url(f'{G2G_RAW_REPO_URL}{SUPPORTED_DATASETS[self.which_dataset]}', self.raw_dir)

        data_raw = self.load(data_raw_file)

        # convert to networkx
        G = to_networkx(data_raw, to_undirected=self.to_undirected)
        node_classes = {n: data_raw.y[i].item() for i,n in enumerate(list(G.nodes()))}
        nx.set_node_attributes(G, node_classes, name = "target")

        if self.biggest_conn_comp:
            CGs = [G.subgraph(c) for c in nx.connected_components(G)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            G = CGs[0]
            G = nx.convert_node_labels_to_integers(G)

        self.data = G
        self.save(self.data, self.raw_paths[1])


    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError('Index out of bounds.')
        return self.data
    

class LargeGraphProcessedDataset(GenericGraphsDataset):

    def __init__(
            self,
            which_dataset: str,
            root: Optional[str] = None,
            split: Optional[str] = None,
            to_undirected: bool = True,
            biggest_conn_comp: bool = True,
            remove_self_loops: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = osp.join(DEFAULT_DATASET_PATH, which_dataset)

        raw_graphs_dataset = LargeGraphRawDataset(
            which_dataset,
            root=root,
            to_undirected=to_undirected,
            biggest_conn_comp=biggest_conn_comp,
            pre_transform=pre_transform_raw,
            pre_filter=pre_filter_raw
        )

        self.remove_self_loops = remove_self_loops

        super().__init__(
            root, raw_graphs_dataset=raw_graphs_dataset,
            split=split,
            undirected=to_undirected,
            no_self_loops=remove_self_loops,
            transform=transform,
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
    

class CoraRaw(LargeGraphRawDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            to_undirected: bool = True,
            biggest_conn_comp: bool = True,
            pre_transform=None,
            pre_filter=None
        ):
        super().__init__(
            'cora',
            root=root,
            split=split,
            to_undirected=to_undirected,
            biggest_conn_comp=biggest_conn_comp,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )

class Cora(LargeGraphProcessedDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            to_undirected: bool = True,
            biggest_conn_comp: bool = True,
            remove_loops: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        super().__init__(
            'cora',
            root=root,
            split=split,
            to_undirected=to_undirected,
            biggest_conn_comp=biggest_conn_comp,
            remove_self_loops=remove_loops,
            pre_transform_raw=pre_transform_raw,
            pre_filter_raw=pre_filter_raw,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )



class SingleGraphResources(DataResources):
    def __init__(
            self,
            dataset_cfg: Dict,
            dataset_cls: Type,
            root: Optional[str] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
        ):

        super().__init__()
        
        self.root = root
        
        self.dataset_cfg = dataset_cfg
        self.dataset_cls = dataset_cls

        self.preproc = {
            'pre_transform': pre_transform,
            'pre_filter': pre_filter
        }

        self._prepared = False


    def prepare_data(self):

        # create dataset
        ds = self.dataset_cls(self.root, **self.dataset_cfg)

        self.decoder = ds.torch_nx_converter
        self.info_total = ds.stats

        # if there is any pre_transform, solve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])

        # create dataset
        ds.delete()
        ds = self.dataset_cls(self.root, **self.preproc, **self.dataset_cfg)

        self.decoder = ds.torch_nx_converter
        self.info_total = ds.stats

        self._prepared = True


    def get_dataset(self, transform=None):
        return self.wrap_dataset(
            self.dataset_cls(self.root, **self.preproc, **self.dataset_cfg),
            transform=transform
        )


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.get_dataset(transform)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for {self.dataset_cls.__name__}')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, decoder, info], splits=None]'


@reg_dataresources.register('cora')
class CoraResources(SingleGraphResources):
    def __init__(
            self, root: str=None, remove_loops: bool = True,
            pre_transform=None, pre_filter=None
        ):
        dataset_cfg = {'remove_loops': remove_loops}
        super().__init__(dataset_cfg, Cora, root, pre_transform, pre_filter)
