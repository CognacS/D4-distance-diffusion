import json
from torch_geometric.utils import to_dense_adj
import torch
import numpy as np

from scipy.stats import gaussian_kde
import pandas as pd
import pickle
import os

from collections import Counter


def compute_dists_types(data):
    # get all distances and bond types from data (a list of graphs)
    if data.edge_attr.ndim == 2:
        edge_attr = data.edge_attr.argmax(dim=-1)
    else:
        edge_attr = data.edge_attr
    
    types = to_dense_adj(data.edge_index, edge_attr=edge_attr+1, max_num_nodes=data.num_nodes)
    dists = torch.cdist(data.node_pos, data.node_pos, p=2)
    triang_matrix = torch.triu(torch.ones_like(dists), diagonal=1).bool()
    dists = dists[triang_matrix]
    types = types[0][triang_matrix]
    
    atom_types = data.x if data.x.ndim == 1 else data.x.argmax(dim=-1)
    
    return (dists, types, atom_types)


def compute_kde(data, **kwargs):
    # compute kde from an array of data
    kde = gaussian_kde(data, **kwargs)
    x_vals = np.linspace(min(data)-1, max(data)+1, 200)
    y_vals = kde(x_vals)
    return x_vals, y_vals

def compute_distribution_per_type(data, remove_no_edge=False):
    unique, counts = np.unique(data, return_counts=True)
    counts = dict(zip(unique.tolist(), counts.tolist()))
    if remove_no_edge and 0 in counts:
        del counts[0]
    total = sum(counts.values())
    counts = {k: v/total for k, v in counts.items()}
    
    return counts


def create_values_and_kde(data, kde_kwargs=None):
    # create a dictionary with edge type as key and (data, kde) as value
    # data contains all distances of that edge type
    # kde contains the kde computed on that data
    # this function aggregates compute_dists_types and compute_kde
    if kde_kwargs is None:
        kde_kwargs = {}
    dists_types = [compute_dists_types(g) for g in data]
    dists, types, atom_types = zip(*dists_types)
    all_dists = np.concatenate([d.numpy().flatten() for d in dists])
    all_types = np.concatenate([t.numpy().flatten() for t in types])
    all_atom_types = np.concatenate([a.numpy().flatten() for a in atom_types])
    unique_types = np.unique(all_types)
    values_and_kdes = {}
    for et in unique_types:
        data = all_dists[all_types==et]
        x_vals, y_vals = compute_kde(data, **kde_kwargs)
        values_and_kdes[et] = (pd.DataFrame({'value': data}), pd.DataFrame({'x': x_vals, 'density': y_vals}))
    nodes_dist = compute_distribution_per_type(all_atom_types)
    edges_dist = compute_distribution_per_type(all_types)
    edges_nobond_dist = compute_distribution_per_type(all_types, remove_no_edge=True)
    type_dists = {
        'nodes': nodes_dist,
        'edges': edges_dist,
        'edges_no_bond': edges_nobond_dist
    }
    return values_and_kdes, type_dists



def load_data(filepath):
    # load data from a pickle file
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    if not isinstance(data, list):
        data = data.to_data_list()
    return data


def save_kdes(path, values, kde, include_data=False):
    # save values and kde to csv files
    if include_data:
        values.to_csv(path + '/data.csv', index=False)
    kde.to_csv(path + '/kde.csv', index=False)
    

def save_distributions(path, type_dists):
    # save as a single json file
    with open(path + '/distributions.json', 'w') as f:
        json.dump(type_dists, f, indent=4)

def compute_template(path):
    # compute a latex template for plotting the kde
    template_tex = """\\begin{figure}
    \\centering
    \\begin{tikzpicture}
      \\begin{axis}[
          width=12cm,
          height=8cm,
          xlabel={Value},
          ylabel={Density},
          grid=both,
        ]
    
        % Histogram from raw data
        \\addplot+[
          ybar,
          fill=blue!30,
          draw=blue!60,
          hist={
            bins=30,
            density,
          }
        ] table [y=value] {figures"""+path[1:]+"""/data.csv};
    
        % KDE curve from precomputed CSV
        \\addplot[
          thick,
          red,
          smooth
        ] table [x=x, y=density, col sep=comma] {figures"""+path[1:]+"""/kde.csv};

        \\addplot[
          thick,
          red,
          smooth
        ] table [x=x, y=density, col sep=comma] {figures"""+path[1:]+"""/kde.csv};
    
      \\end{axis}
    \\end{tikzpicture}
    \\caption{Caption}
    \\label{fig:placeholder}
\\end{figure}"""
    return template_tex



def compute_and_store_kdes_per_bond_type(kde_path, data, include_tex=False, include_data=False, kde_kwargs=None):
    # compute kdes
    values_and_kdes, type_dists = create_values_and_kde(data, kde_kwargs=kde_kwargs)
    for et, (values, kdes) in values_and_kdes.items():
        bond_path = os.path.join(kde_path, f'bond_{int(et)}')
        os.makedirs(bond_path, exist_ok=True)
        save_kdes(bond_path, values, kdes, include_data=include_data)
        
        if include_tex:
            # save tex file
            with open(os.path.join(bond_path, 'kde.tex'), 'w') as f:
                f.write(compute_template(bond_path))
    save_distributions(kde_path, type_dists)


def scan_all_checkpoints_and_compute_kdes(include_tex=False, include_data=False, kde_kwargs=None):
    # scan all checkpoints in the checkpoints directory and compute kdes for each of them
    # save the kdes in a directory called kdes, with a subdirectory for each checkpoint
    # each subdirectory will contain a subdirectory for each edge type, containing the data and kde csv files
    # and a latex template file for plotting the kde

    os.makedirs('kdes', exist_ok=True)
    kdes_dir = './kdes'

    checkpoints_dir = "./checkpoints"
    for root, dirs, files in os.walk(checkpoints_dir):
        for file in files:
            if 'generated_graphs' in file and not 'wandb' in file:
                print(f'Processing {file} in {root}')
                # read data
                path = os.path.join(root, file)
                data = load_data(path)
                
                # create kde directory
                config_name = root.split('/')[2]
                version_name = root.split('/')[3]
                kde_path = os.path.join(kdes_dir, config_name, version_name)
                os.makedirs(kde_path, exist_ok=True)
                data = load_data(path)
                # compute kdes
                compute_and_store_kdes_per_bond_type(kde_path, data, kde_kwargs=kde_kwargs)
                print(f'\tKDEs saved in {kde_path}')


def aggregate_statistics_of_configs(round_digits=4, latex_format=True):
    # aggregate statistics across all versions of configs
    # want to compute mean and std of distributions of node and edge types
    kdes_dir = "./kdes"
    # check it exists
    if not os.path.exists(kdes_dir):
        print(f'No KDEs directory found at {kdes_dir}, returning...')
        return
    for config_dir in os.listdir(kdes_dir):
        if not os.path.isdir(os.path.join(kdes_dir, config_dir)):
            continue
        config_path = os.path.join(kdes_dir, config_dir)
        print(f'Processing config {config_path}')

        ##############  GATHERING ALL RESULTS FROM EACH VERSION  ###############
        stats_dict = None
        for ver_dir in os.listdir(config_path):
            if not os.path.isdir(os.path.join(config_path, ver_dir)):
                continue
            if 'bond_' in ver_dir:
                break
            print(f'\tProcessing version {ver_dir}')
            ver_path = os.path.join(config_path, ver_dir)
            with open(os.path.join(ver_path, 'distributions.json'), 'r') as f:
                dist_dict = json.load(f)
            if stats_dict is None:
                stats_dict = {k1: {k2: [] for k2 in v1.keys()} for k1, v1 in dist_dict.items()}
            for k1, v1 in dist_dict.items():
                for k2, v2 in v1.items():
                    stats_dict[k1][k2].append(v2)
        
        if stats_dict is None:
            print(f'\tSkipping dataset directory {config_dir}')
            continue
        ##################  COMPUTING MEAN AND STD OF CONFIG  ##################
        print(f'\tComputing mean and std of distributions for config {config_dir}')
        # compute mean and std
        for k1, v1 in stats_dict.items():
            for k2, v2 in v1.items():
                mean = (np.mean(v2)*100).round(round_digits)
                std = (np.std(v2)*100).round(round_digits)
                if latex_format:
                    # force to have exactly round_digits digits
                    stats_dict[k1][k2] = f'\\nlvalpm{{{mean:.{round_digits}f}}}{{{std:.{round_digits}f}}}'
                else:
                    stats_dict[k1][k2] = {'mean': mean, 'std': std}

        # save stats
        with open(os.path.join(config_path, 'stats.json'), 'w') as f:
            json.dump(stats_dict, f, indent=4)

def scan_single_data_and_compute_kdes(data, config_name, kde_kwargs=None):
    print(f'Computing kdes for dataset {config_name}')
    kde_path = os.path.join('./kdes', config_name)
    os.makedirs(kde_path, exist_ok=True)
    compute_and_store_kdes_per_bond_type(kde_path, data, kde_kwargs=kde_kwargs)
    print(f'KDEs saved in {kde_path}')