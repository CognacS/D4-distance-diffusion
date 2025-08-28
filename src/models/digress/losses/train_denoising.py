##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/main/dgd/metrics/train_metrics.py
#
##########################################################################################################

from typing import List

import torch
from torch import Tensor
import torch.nn as nn

import src.models.digress.labels as labels

import torch.nn.functional as F

def generate_weights(mask: Tensor):
    num_elems = mask.flatten(start_dim=1).sum(dim=-1)
    alive_batches = (num_elems > 0).sum()
    weights_per_batch_elem = 1 / (num_elems * alive_batches)
    weights = torch.repeat_interleave(weights_per_batch_elem, num_elems)

    return weights
    

class SimpleTrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(
            self,
            lambda_train_E: float = 1.,
            lambda_train_ext_E: float = 1.,
            concat_edges: bool = False,
            weighted: bool = False,
            class_weighted: bool = False,
            **kwargs
        ):
        super().__init__()
        self.lambda_train_E = lambda_train_E
        self.lambda_train_ext_E = lambda_train_ext_E
        self.concat_edges = concat_edges
        self.weighted = weighted
        self.class_weighted = class_weighted

    def forward(
            self,
            pred_values: List[Tensor],
            true_values: List[Tensor],
            reduce: bool=True,
            ret_log: bool=False
        ):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """

        assert not self.weighted or (self.weighted and len(pred_values) == 6), "If weighted, pred_values must contain masks"

        if len(pred_values) == 3:
            pred_x, pred_e, pred_ext_e = pred_values
        elif len(pred_values) == 6:
            pred_x, pred_e, pred_ext_e, nodes_mask, edges_mask, ext_edges_mask = pred_values

        true_x, true_e, true_ext_e = true_values

        using_ext = true_ext_e is not None

        # compute cross entropy loss
        reduction = 'mean' if reduce else 'none'

        reduction_to_do = reduction if not self.weighted else 'none'

        if self.class_weighted:
            if not isinstance(self.class_weighted, bool):
                weight = self.class_weighted
            else:
                weight = 5.
            edge_class_weights = torch.full((pred_e.shape[-1],), fill_value=weight, device=pred_e.device)
            edge_class_weights[0] = 1.
        else:
            edge_class_weights = None


        loss_x = F.cross_entropy(pred_x, true_x, reduction=reduction_to_do) if true_x.numel() > 0 else torch.zeros(1, device=pred_x.device)
        loss_e = F.cross_entropy(pred_e, true_e, reduction=reduction_to_do, weight=edge_class_weights) if true_e.numel() > 0 else torch.zeros(1, device=pred_x.device)
        if using_ext:
            loss_ext_e = F.cross_entropy(pred_ext_e, true_ext_e, reduction=reduction_to_do, weight=edge_class_weights) if true_ext_e.numel() > 0 else torch.zeros(1, device=pred_x.device)
        else:
            loss_ext_e = None

        if self.weighted:
            nodes_weights = generate_weights(nodes_mask)
            edges_weights = generate_weights(edges_mask)
            loss_x = loss_x * nodes_weights
            loss_e = loss_e * edges_weights
            if using_ext:
                ext_edges_weights = generate_weights(ext_edges_mask)
                loss_ext_e = loss_ext_e * ext_edges_weights
            if reduction == 'mean':
                loss_x = loss_x.sum()
                loss_e = loss_e.sum()
                loss_ext_e = loss_ext_e.sum() if using_ext else None

        if self.concat_edges and using_ext:
            pred_e = torch.cat([pred_e, pred_ext_e], dim=0)
            true_e = torch.cat([true_e, true_ext_e], dim=0)
            loss_e = F.cross_entropy(pred_e, true_e, reduction=reduction_to_do, weight=edge_class_weights) if true_e.numel() > 0 else torch.zeros(1, device=pred_x.device)
            if self.weighted:
                edges_weights = generate_weights(torch.cat([edges_mask, ext_edges_mask], dim=2))
                loss_e = loss_e * edges_weights
                if reduction == 'mean':
                    loss_e = loss_e.sum()

            if reduction == 'mean':
                total_loss: Tensor = loss_x + self.lambda_train_E * loss_e
            else:
                total_loss: Tensor = loss_x.mean() + self.lambda_train_E * loss_e.mean()

        else:
            if reduction == 'mean':
                total_loss: Tensor = loss_x + self.lambda_train_E * loss_e
                if using_ext:
                    total_loss = total_loss + self.lambda_train_ext_E * loss_ext_e
            else:
                total_loss: Tensor = loss_x.mean() + self.lambda_train_E * loss_e.mean()
                if using_ext:
                    total_loss = total_loss + self.lambda_train_ext_E * loss_ext_e.mean()

        if ret_log:
            to_log = {
                labels.DENOISE_CE_X: loss_x.detach(),
                labels.DENOISE_CE_E: loss_e.detach(),
                labels.DENOISE_CE_TOTAL: total_loss.detach(),
            }
            if using_ext:
                to_log[labels.DENOISE_CE_EXT_E] = loss_ext_e.detach()
            return total_loss, to_log
        else:
            return total_loss
