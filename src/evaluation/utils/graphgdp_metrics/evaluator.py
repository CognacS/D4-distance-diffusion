from typing import List
import networkx as nx
from src.evaluation.utils.graphgdp_metrics.structure_evaluator import mmd_eval
from src.evaluation.utils.graphgdp_metrics.gin_evaluator import nn_based_eval
from torch_geometric.utils import to_networkx
import torch
import torch.nn.functional as F
import dgl


def get_stats_eval(config):

    if hasattr(config, "mmd_distance"):
        mmd_distance = config.mmd_distance.lower()
    else:
        mmd_distance = 'rbf'

    if mmd_distance == 'rbf':
        method = [('degree', 1., 'argmax'), ('cluster', 0.1, 'argmax'),
                  ('spectral', 1., 'argmax')]
    else:
        raise ValueError
    
    if hasattr(config, "max_subgraph"):
        max_subgraph = config.max_subgraph
    else:
        max_subgraph = False

    def eval_stats_fn(test_dataset: List[nx.Graph], pred_graph_list: List[nx.Graph]):
        pred_G = pred_graph_list
        sub_pred_G = []
        if max_subgraph:
            for G in pred_G:
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                sub_pred_G += [CGs[0]]
            pred_G = sub_pred_G

        test_G = test_dataset
        results = mmd_eval(test_G, pred_G, method)
        return results

    return eval_stats_fn


def get_nn_eval(config):

    if hasattr(config, "N_gin"):
        N_gin = config.N_gin
    else:
        N_gin = 10

    if hasattr(config, "max_subgraph"):
        max_subgraph = config.max_subgraph
    else:
        max_subgraph = False

    def nn_eval_fn(test_dataset: List[nx.Graph], pred_graph_list: List[nx.Graph]):
        pred_G = pred_graph_list
        sub_pred_G = []
        if max_subgraph:
            for G in pred_G:
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                sub_pred_G += [CGs[0]]
            pred_G = sub_pred_G
        test_G = test_dataset

        results = nn_based_eval(test_G, pred_G, N_gin)
        return results

    return nn_eval_fn