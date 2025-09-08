from typing import List, Dict

import torch

from src.data.utils.core import get_dict_histogram

from src.datatypes.sparse import SparseGraph
from src.data.datasets.core import ProcessedDataset

def get_torch_graphs_stats(graphs: List[SparseGraph]|ProcessedDataset, num_classes: Dict=None) -> Dict:
    """Function for computing general statistics on a list of SparseGraphs.
    Currently returns:
    -  number of classes
    """
    
    l_num_nodes = []
    l_num_edges = []

    for g in graphs:
        l_num_nodes.append(g.num_nodes)
        l_num_edges.append(g.num_edges)

    ret_dict = {
        'num_nodes_min': min(l_num_nodes),
        'num_nodes_max': max(l_num_nodes),
        'num_nodes_hist': get_dict_histogram(l_num_nodes),
        'num_edges_min': min(l_num_edges),
        'num_edges_max': max(l_num_edges),
        'num_edges_hist': get_dict_histogram(l_num_edges),
        'num_graphs': len(graphs)
    }
    
    if 'edge_attr' in num_classes.keys():
        num_classes['edge_attr'] += 1 # account for non-edges
    
    if num_classes is not None:
        histograms = {key: torch.zeros(num_cls) for key, num_cls in num_classes.items()}
        
        for g in graphs:
            for key in num_classes.keys():
                value = getattr(g, key)
                
                if value.ndim == 1:
                
                    if key == 'edge_attr': 
                        value = value + 1
                    
                    # this scatters values across the different types   
                    histograms[key].scatter_add_(0, value.long(), torch.ones_like(value, dtype=torch.float))
                    
                elif value.ndim == 2:
                    
                    if key == 'edge_attr': 
                        sl = slice(1, None)
                    else:
                        sl = slice(None)
                    
                    histograms[key][sl] += value.sum(dim=0).float()
                
                if key == 'edge_attr':
                    histograms[key][0] += g.num_nodes * (g.num_nodes - 1) - g.num_edges
                
        ret_dict['marginals'] = {}
        for key in num_classes.keys():
            ret_dict['marginals'][key] = (histograms[key] / histograms[key].sum()).tolist()
        

    return ret_dict