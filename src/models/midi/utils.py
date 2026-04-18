import os
import collections
from copy import deepcopy
from typing import Optional, Union, Dict

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError,MetricCollection,KLDivergence
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
import wandb


# from dgd.ggg_utils_deps import approx_small_symeig, our_small_symeig,extract_canonical_k_eigenfeat
# from dgd.ggg_utils_deps import  ensure_tensor, get_laplacian, asserts_enabled


class NoSyncMetricCollection(MetricCollection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) #disabling syncs since it messes up DDP sub-batching


class NoSyncMetric(Metric):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncKL(KLDivergence):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncMAE(MeanAbsoluteError):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching>>>>>>> main:utils.py

# Folders
def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs', exist_ok=True)
        os.makedirs('chains', exist_ok=True)
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name, exist_ok=True)
        os.makedirs('chains/' + args.general.name, exist_ok=True)
    except OSError:
        pass


def to_dense(data, dataset_info, device=None):
    X, node_mask = to_dense_batch(x=data.x, batch=data.batch)
    pos, _ = to_dense_batch(x=data.pos, batch=data.batch)
    pos = pos.float()
    assert pos.mean(dim=1).abs().max() < 1e-3
    charges, _ = to_dense_batch(x=data.charges, batch=data.batch)
    auxiliary_node_states = collections.OrderedDict()
    for name in getattr(dataset_info, 'auxiliary_node_state_names', []):
        value = getattr(data, name, None)
        if value is None:
            raise ValueError(
                f"Auxiliary node states were requested, but the current batch does not contain '{name}'. "
                "Reprocess the dataset with the explicit auxiliary-node-state format."
            )
        auxiliary_node_states[name], _ = to_dense_batch(x=value, batch=data.batch)
    max_num_nodes = X.size(1)
    if data.edge_attr.ndim == 2:
        edge_attr = data.edge_attr.argmax(-1)
    else:
        edge_attr = data.edge_attr
    edge_index, edge_attr = remove_self_loops(data.edge_index, edge_attr)
    # add 1 to edge_attr to account for no-bond class
    E = to_dense_adj(edge_index=edge_index, batch=data.batch, edge_attr=edge_attr+1, max_num_nodes=max_num_nodes)

    data = dataset_info.to_one_hot(
        X=X,
        charges=charges,
        E=E,
        node_mask=node_mask,
        auxiliary_node_states=auxiliary_node_states,
    )
    data.pos = pos
    data.y = data.X.new_zeros((data.X.shape[0], 0))
    data.node_mask = node_mask

    if device is not None:
        pos = pos.to(device)
        data = data.device_as(pos)
        data.pos = pos
        data.y = data.y.to(device)
        data.node_mask = node_mask.to(device)

    return data.mask()


class PlaceHolder:
    def __init__(self, pos=None, X=None, charges=None, E=None, y=None, t_int=None, t=None, node_mask=None,
                 auxiliary_node_states=None):
        self.pos = pos
        self.X = X
        self.charges = charges
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask
        self.auxiliary_node_states = collections.OrderedDict(auxiliary_node_states or {})

    def device_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.pos = self.pos.to(x.device) if self.pos is not None else None
        self.X = self.X.to(x.device) if self.X is not None else None
        self.charges = self.charges.to(x.device) if self.charges is not None else None
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None
        self.auxiliary_node_states = collections.OrderedDict(
            (name, value.to(x.device) if isinstance(value, torch.Tensor) else value)
            for name, value in self.auxiliary_node_states.items()
        )
        return self

    def mask(self, node_mask=None):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        diag_mask = ~torch.eye(n, dtype=torch.bool,
                               device=node_mask.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1)  # bs, n, n, 1

        if self.X is not None:
            self.X = self.X * x_mask
        if self.charges is not None:
            self.charges = self.charges * x_mask
        for name, value in self.auxiliary_node_states.items():
            if value is not None:
                self.auxiliary_node_states[name] = value * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
        if self.pos is not None:
            self.pos = self.pos * x_mask
            self.pos = self.pos - self.pos.mean(dim=1, keepdim=True)
        if self.E is not None:
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def collapse(self):#, collapse_charges):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        #copy.charges = collapse_charges.to(self.charges.device)[torch.argmax(self.charges, dim=-1)]
        copy.charges = torch.argmax(self.charges, dim=-1)
        copy.E = torch.argmax(self.E, dim=-1)
        copy.auxiliary_node_states = collections.OrderedDict(
            (name, torch.argmax(value, dim=-1))
            for name, value in self.auxiliary_node_states.items()
            if value is not None
        )
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = - 1
        copy.charges[self.node_mask == 0] = 1000
        for value in copy.auxiliary_node_states.values():
            value[self.node_mask == 0] = 1000
        #copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        return copy

    def __repr__(self):
        aux_repr = ', '.join(
            f"{name}: {value.shape if type(value) == torch.Tensor else value}"
            for name, value in self.auxiliary_node_states.items()
        )
        return (f"pos: {self.pos.shape if type(self.pos) == torch.Tensor else self.pos} -- " +
                f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- " +
                f"charges: {self.charges.shape if type(self.charges) == torch.Tensor else self.charges} -- " +
                f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- " +
                f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y} -- " +
                f"aux: {{{aux_repr}}}")


    def copy(self):
        return PlaceHolder(
            X=self.X,
            charges=self.charges,
            E=self.E,
            y=self.y,
            pos=self.pos,
            t_int=self.t_int,
            t=self.t,
            node_mask=self.node_mask,
            auxiliary_node_states=collections.OrderedDict(self.auxiliary_node_states.items()),
        )


def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'MolDiffusion_{cfg.dataset["name"]}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


def remove_mean_with_mask(x, node_mask):
    """ x: bs x n x d.
        node_mask: bs x n """
    assert node_mask.dtype == torch.bool, f"Wrong type {node_mask.dtype}"
    node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (x * (~node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

