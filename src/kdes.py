from torch_geometric.utils import to_dense_adj
import torch
import numpy as np

from scipy.stats import gaussian_kde
import pandas as pd
import pickle
import os


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
    
    return (dists, types)


def compute_kde(data, **kwargs):
    # compute kde from an array of data
    kde = gaussian_kde(data, **kwargs)
    x_vals = np.linspace(min(data)-1, max(data)+1, 200)
    y_vals = kde(x_vals)
    return x_vals, y_vals


def create_values_and_kde(data, kde_kwargs=None):
    # create a dictionary with edge type as key and (data, kde) as value
    # data contains all distances of that edge type
    # kde contains the kde computed on that data
    # this function aggregates compute_dists_types and compute_kde
    if kde_kwargs is None:
        kde_kwargs = {}
    dists_types = [compute_dists_types(g) for g in data]
    dists, types = zip(*dists_types)
    all_dists = np.concatenate([d.numpy().flatten() for d in dists])
    all_types = np.concatenate([t.numpy().flatten() for t in types])
    unique_types = np.unique(all_types)
    values_and_kdes = {}
    for et in unique_types:
        data = all_dists[all_types==et]
        x_vals, y_vals = compute_kde(data, **kde_kwargs)
        values_and_kdes[et] = (pd.DataFrame({'value': data}), pd.DataFrame({'x': x_vals, 'density': y_vals}))
    return values_and_kdes



def load_data(filepath):
    # load data from a pickle file
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def save_kdes(path, values, kde):
    # save values and kde to csv files
    values.to_csv(path + '/data.csv', index=False)
    kde.to_csv(path + '/kde.csv', index=False)
    

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



def compute_and_store_kdes_per_bond_type(kde_path, data, kde_kwargs=None):
    # compute kdes
    values_and_kdes = create_values_and_kde(data, kde_kwargs=kde_kwargs)
    for et, (values, kdes) in values_and_kdes.items():
        bond_path = os.path.join(kde_path, f'bond_{int(et)}')
        os.makedirs(bond_path, exist_ok=True)
        save_kdes(bond_path, values, kdes)
        # save tex file
        with open(os.path.join(bond_path, 'kde.tex'), 'w') as f:
            f.write(compute_template(bond_path))


def scan_all_checkpoints_and_compute_kdes(kde_kwargs=None):
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
                kde_path = os.path.join(kdes_dir, config_name)
                os.makedirs(kde_path, exist_ok=True)
                data = load_data(path)

                # compute kdes
                compute_and_store_kdes_per_bond_type(kde_path, data, kde_kwargs=kde_kwargs)
                print(f'KDEs saved in {kde_path}')
                

def scan_single_data_and_compute_kdes(data, config_name, kde_kwargs=None):
    print(f'Computing kdes for dataset {config_name}')
    kde_path = os.path.join('./kdes', config_name)
    os.makedirs(kde_path, exist_ok=True)
    compute_and_store_kdes_per_bond_type(kde_path, data, kde_kwargs=kde_kwargs)
    print(f'KDEs saved in {kde_path}')