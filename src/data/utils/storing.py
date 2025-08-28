
import json
import pickle
import torch
from torch_geometric.io import read_npz

EXT_TO_METHOD = {
    'json': 'json',
    'pkl': 'pickle',
    'pt': 'torch',
    'npz': 'numpy_zip'
}

METHOD_TO_SAVEFUNCTION = {
    'json': json.dump,
    'pickle': pickle.dump,
    'torch': torch.save,
    'numpy_zip': read_npz
}

METHOD_TO_LOADFUNCTION = {
    'json': json.load,
    'pickle': pickle.load,
    'torch': torch.load,
    'numpy_zip': read_npz
}

def save_file(data, save_path, save_method=None, **kwargs):

    # if savemethod is not given, infer it from the file extension
    if save_method is None:
        ext = save_path.split('.')[-1]
        save_method = EXT_TO_METHOD[ext]

    # save datastructure as a json file (for readability)
    write_how = 'w' if save_method == 'json' else 'wb'
    if save_method == 'json':
        kwargs['indent'] = '\t'

    # save data
    with open(save_path, write_how) as save_file:
        if save_method in METHOD_TO_SAVEFUNCTION:
            METHOD_TO_SAVEFUNCTION[save_method](data, save_file, **kwargs)
        else:
            raise ValueError(f"save_method {save_method} not supported, use {list(METHOD_TO_SAVEFUNCTION.keys())}")
        

def load_file(load_path, load_method=None, **kwargs):

    if load_method is None:
        ext = load_path.split('.')[-1]
        load_method = EXT_TO_METHOD[ext]

    # save datastructure as a json file (for readability)
    read_how = 'r' if load_method == 'json' else 'rb'

    with open(load_path, read_how) as load_file:
        if load_method in METHOD_TO_LOADFUNCTION:
            return METHOD_TO_LOADFUNCTION[load_method](load_file, **kwargs)
        else:
            raise ValueError(f"load_method {load_method} not supported, use {list(METHOD_TO_SAVEFUNCTION.keys())}")