from typing import List
import numpy as np
import torch

ALLOWED_BONDS = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3, 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}


def check_stability(atom_types, edge_types, dataset_info, debug=False,atom_decoder=None):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype='int')

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i])/2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i])/2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = ALLOWED_BONDS[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type], atom_n_bond))
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)



#################################################################################
#          THIS CODE IS EXTRACTED FROM https://github.com/GRAPH-0/CDGS          #
#################################################################################

import networkx as nx


### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
def compute_nspdk_mmd(samples1, samples2, metric, is_hist=True, n_jobs=None):
    from sklearn.metrics.pairwise import pairwise_kernels
    from eden.graph import vectorize

    def kernel_compute(X, Y=None, is_hist=True, metric='linear', n_jobs=None):
        X = vectorize(X, complexity=4, discrete=True)
        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)
        return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py
def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    if len(graph_pred_list_remove_empty) == 0:
        return float('inf')

    # prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, metric='nspdk', is_hist=False, n_jobs=20)
    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def get_nspdk_eval(config=None):
    return nspdk_stats


from fcd_torch import FCD


def compute_intermediate_FCD(smiles, n_jobs=1, device='cpu', batch_size=512):
    """
    Precomputes statistics such as mean and variance for FCD.
    """
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    stats = FCD(**kwargs_fcd).precalc(smiles)
    return stats


def get_FCDMetric(ref_smiles, n_jobs=1, device='cpu', batch_size=512):
    pref = compute_intermediate_FCD(ref_smiles, n_jobs, device, batch_size)

    def FCDMetric(gen_smiles):
        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        return FCD(**kwargs_fcd)(gen=gen_smiles, pref=pref)

    return FCDMetric


from src.data.simple_transforms.chemical_props import PROPS_NAME_TO_FUNC


def get_mae_molecular_metric(which_properties: List[str], separated: bool = False):

    # extract useful functions for computing properties
    prop_functions = [PROPS_NAME_TO_FUNC[prop] for prop in which_properties]

    def compute_mae_molecular_props(mols, ref_properties: torch.Tensor):

        # compute properties
        props = [[prop_fun(mol) for prop_fun in prop_functions] for mol in mols]
        props = torch.tensor(props, dtype=torch.float32, device=ref_properties.device)

        # compute MAE per property
        mae_per_prop = torch.mean(torch.abs(props - ref_properties), dim=0)

        # return MAE per property or global MAE
        if separated:
            return {f'mae_{prop}': mae_per_prop[i].cpu() for i, prop in enumerate(which_properties)}
        else:
            return torch.sum(mae_per_prop)
        
    return compute_mae_molecular_props




GET_MOLECULAR_METRIC = {
    'nspdk': get_nspdk_eval,
    'fcd': get_FCDMetric,
    'cond_mae': get_mae_molecular_metric
}