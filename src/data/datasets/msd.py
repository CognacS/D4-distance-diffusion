import random
import numpy as np
import os
import torch
import pickle
import gdown
from typing import List, Optional, Tuple, Dict, Any, Callable, Union
from copy import copy
from copy import deepcopy
import os.path as osp
from tqdm import tqdm
import networkx as nx
from torch_geometric.utils import from_networkx
from scipy.spatial.transform import Rotation as R


from torch_geometric.io.fs import makedirs

from torch_geometric.data import Data, download_url, download_google_url
from torch_geometric.utils import dense_to_sparse, remove_self_loops

from src.data.datasets.core import RawDataset, ProcessedDataset, DataResources, DatasetException, DEFAULT_DATASET_PATH
from src.data.simple_transforms.graph import GraphToNetworkxConverter
from src.datatypes.sparse import SparseGraph
from src.data.utils.graphs import get_torch_graphs_stats
from src.data.datasets.continuous_graph import ContinuousGraphsDataset, ContinuousGraphsResources

#from graph_datasets.graph_visualizer import visualize_nxgraph_3d


SPECTRE_RAW_REPO_URL = "https://drive.google.com/drive/folders/1y5a772bYItm7kjC8C4mD2kRJ77khPpXV?usp=sharing"
SUPPORTED_DATASETS = {
    'manh_small_3000': 'ssg_manhattan_small_3000.pkl',
    'manh_small_20': 'ssg_manhattan_small_20.pkl',
    'L_big_1000': 'ssg.pkl',
    'msd_3500': 'msd_3476.pkl',
    'msd_100': 'msd_100.pkl'
}

SSG_DATASET_PATH = osp.join(DEFAULT_DATASET_PATH, "ssg")

class MsdRawDataset(RawDataset):     ### Inspired by class BaseGraphgdpRawDataset(RawDataset)

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
            root = osp.join(SSG_DATASET_PATH, which_dataset)
        print(f"Using root path: {root}")

        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter)

        if not hasattr(self, 'data'):
            raw_path = self.raw_paths[0]
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"File not found: {raw_path}")

            with open(raw_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data)} pairs from {raw_path}")

    def download(self):
        listing = gdown.download_folder(
            url="https://drive.google.com/drive/folders/1y5a772bYItm7kjC8C4mD2kRJ77khPpXV?usp=sharing",
            skip_download=True,
            remaining_ok=True,
            use_cookies=False
        )
        match = next((f for f in listing if f.path == SUPPORTED_DATASETS[self.which_dataset]), None)
        if match is None:
            raise FileNotFoundError(f"{SUPPORTED_DATASETS[self.which_dataset]!r} not found in the shared folder")
        data_raw_file = download_google_url(match.id, self.raw_dir, SUPPORTED_DATASETS[self.which_dataset])
        data_raw = self.load(data_raw_file)
        self.data = data_raw

    def subset_from(self, indices: List[int], name: str):

        subset = copy(self)
        subset.root = self.root
        subset.split = name

        makedirs(subset.raw_dir)

        subset.data = [self.data[i] for i in indices]
        subset.save(subset.data, subset.raw_paths[0])

        return subset
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def raw_file_names(self):
        return [SUPPORTED_DATASETS[self.which_dataset]]



class MsdDataset(ContinuousGraphsDataset):   ### Inspired by class GenericGraphsDataset(ProcessedDataset)
    def __init__(
            self,
            which_dataset: str,
            root: Optional[str] = None,
            split: Optional[str] = None,
            remove_loops: bool = True,
            scale_positions: float = 1.0,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = osp.join(SSG_DATASET_PATH, which_dataset) ### TODO check

        raw_graphs_dataset = MsdRawDataset(
            which_dataset=which_dataset,
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
        
        self.data.node_pos *= scale_positions


    def raw_data_to_sparse_graph(self, sample: nx.Graph) -> SparseGraph:

        g = from_networkx(sample)
        num_nodes = g.num_nodes

        x = torch.zeros(num_nodes, dtype=torch.int64)
        edge_index = g.edge_index if not self.remove_self_loops else remove_self_loops(g.edge_index)[0]
        edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.int64)

        g.keys()

        graph = SparseGraph(**{k: g[k] for k in g.keys()})

        return graph
    
    def dataset_specific_process(self, sample: nx.Graph) -> nx.Graph:
        processed_graph = deepcopy(sample)
        sample_node_attributes = deepcopy(sample.nodes(data=True))
        sample_edge_attributes = deepcopy(sample.edges(data=True))

        node_type_mapping = {
            'ws': 0,
            'room': 1,
            'wall': 2,
            'floor': 3,
        }

        # edge_type_mapping = {
        #     'ws_belongs_room': 0,
        #     'ws_belongs_wall': 1,
        #     'ws_same_wall': 2,
        #     'ws_same_room': 3,
        #     'room_belongs_floor': 4,
        # }
        
        edge_type_mapping = {
            'common': 0
        }

        for _, data in processed_graph.nodes(data=True):
            data.clear()
        for _, _, data in processed_graph.edges(data=True):
            data.clear()

        for node in sample_node_attributes:
            processed_graph.nodes[node[0]]['x'] = node_type_mapping[node[1]['type']]
            processed_graph.nodes[node[0]]['node_pos'] = node[1]['center']
            if 'normal' not in node[1]:
                processed_graph.nodes[node[0]]['node_normal'] = np.array([0, 0, 0])
            else:
                processed_graph.nodes[node[0]]['node_normal'] = node[1]['normal']

            if 'length' not in node[1]:
                processed_graph.nodes[node[0]]['node_length'] = 0
            else:
                processed_graph.nodes[node[0]]['node_length'] = node[1]['length']

        for edge in sample_edge_attributes:
            processed_graph.edges[edge[0], edge[1]]['edge_attr'] = edge_type_mapping[edge[2]['type']]

        #print(f"dbg node_attributes {processed_graph.nodes(data=True)}")
        #print(f"dbg edge_attributes {processed_graph.edges(data=True)}")

        return processed_graph
    
    def define_stats(self):
        pass


class SparseGraphToSceneGraphDecoder():
    def __init__(self):
        self.torch_nx_converter = GraphToNetworkxConverter(None, None)

        self.node_type_mapping = {
            0: 'ws',
            1: 'room',
            2: 'wall',
            3: 'floor',
        }

        self.node_viz_feat_mapping = {
            'ws': "black",
            'room': 'ro',
            'wall': 'mo',
            'floor': 'go',
            'building': 'co'
        }

    def graph_to_scene_graph(
        self,
        batch: Union[List[SparseGraph], SparseGraph]
    ) -> Union[List[nx.Graph], SparseGraph]:
        
        batch = batch.clone().collapse()
        
        if isinstance(batch, List):
            node_attrs = batch[0].get_all_node_attrs()
            edge_attrs = batch[0].get_all_edge_attrs()
        elif isinstance(batch, SparseGraph):
            node_attrs = batch.get_all_node_attrs()
            edge_attrs = batch.get_all_edge_attrs()

        nx_graph = self.torch_nx_converter.graph_to_nx(deepcopy(batch), node_attrs = node_attrs)
        node_attrs = nx_graph.nodes(data=True)
        for node_id, node_attrs in node_attrs:
            # print(f'dbg node_id {nx_graph} node_attrs {node_attrs}'))
            node_type = self.node_type_mapping[node_attrs["x"]]
            node_attrs["type"] = node_type

            if node_type in ["room", "wall","floor","building"]:
                node_attrs["center"] = node_attrs["node_pos"]
                node_attrs["viz"] = {}
                node_attrs["viz"]["center"] = node_attrs["node_pos"]
                node_attrs["viz"]["type"] = "Point"
                node_attrs["viz"]["feat"] = self.node_viz_feat_mapping[node_type]
                node_attrs["linewidth"] = 1.0
                node_attrs["alpha"] = 0.5

            elif node_type in ["ws"]:
                node_attrs["center"] = node_attrs["node_pos"]

                rotation = R.from_euler('z', -90, degrees=True)
                ws_direction = rotation.apply(deepcopy(node_attrs["node_normal"]))
                ws_direction /= np.linalg.norm(ws_direction)
                limits = [node_attrs["center"] + ws_direction*node_attrs["node_length"]/2,
                          node_attrs["center"] - ws_direction*node_attrs["node_length"]/2]
                node_attrs["viz"] = {}
                node_attrs["viz"]["center"] = node_attrs["center"]
                node_attrs["viz"]["limits"] = limits
                node_attrs["viz"]["type"] = "Line"
                node_attrs["viz"]["feat"] = self.node_viz_feat_mapping[node_type]
                node_attrs["linewidth"] = 2.0
                node_attrs["alpha"] = 1.0

        for u, v, edge_attrs in nx_graph.edges(data=True):
            nodes_types = set([nx_graph.nodes[u].get("type"), nx_graph.nodes[v].get("type")])
            if nodes_types == set(["ws", 'ws']):
                edge_attrs["viz_feat"] = "k"
            elif nodes_types == set(["ws", 'room']):
                edge_attrs["viz_feat"] = "red"
            elif nodes_types == set(["ws", 'wall']):
                edge_attrs["viz_feat"] = "m"
            elif nodes_types == set(["floor", 'room']):
                edge_attrs["viz_feat"] = "green"


        return GraphWrapper(nx_graph)
    
class GraphWrapper():
    def __init__(self, graph):
        self.graph = graph

    def get_attributes_of_all_nodes(self):
        return self.graph.nodes(data=True)
    
    def get_attributes_of_all_edges(self):
        return self.graph.edges(data=True)

    
from src.data.datasets.split import random_split_dataset
from src.data.datasets import reg_dataresources

@reg_dataresources.register('msd')
class MsdResources(ContinuousGraphsResources):
    def __init__(
            self, which_dataset, random_splits: Dict, root: str=None, remove_loops: bool = True,
            pre_transform=None, pre_filter=None
        ):

        dataset_cfg = {'remove_loops': remove_loops}

        super().__init__(which_dataset,random_splits, dataset_cfg, MsdDataset, root, pre_transform, pre_filter)
        
        
    def prepare_data(self):
        dss = super().prepare_data()
        
        # compute maximum distance as scaling factor for all node positions
        ds_train = dss['train'][0]
        d_max = max([torch.cdist(g.node_pos, g.node_pos).max().item() for g in ds_train])
        
        self.dataset_cfg['scale_positions'] = 1.0 / d_max
        
        return dss
