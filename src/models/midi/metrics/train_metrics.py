import torch
import torch.nn as nn
import time
import collections
import wandb

from torchmetrics import MeanSquaredError
from src.models.midi.metrics.abstract_metrics import CrossEntropyMetric

class TrainLoss(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train, auxiliary_node_state_names=None, lambda_train_auxiliary_node_states=1.0):
        super().__init__()
        self.train_pos_mse = MeanSquaredError(sync_on_compute=False, dist_sync_on_step=False)
        self.node_loss = CrossEntropyMetric()
        self.charges_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.auxiliary_node_state_names = list(auxiliary_node_state_names or [])
        self.auxiliary_node_state_losses = nn.ModuleDict({
            name: CrossEntropyMetric()
            for name in self.auxiliary_node_state_names
        })

        self.lambda_train = lambda_train
        self.lambda_train_auxiliary_node_states = lambda_train_auxiliary_node_states

    def forward(self, masked_pred, masked_true, log: bool):
        """ Compute train metrics. Warning: the predictions and the true values are masked, but the relevant entriese
            need to be computed before calculating the loss

            masked_pred, masked_true: placeholders
            log : boolean. """

        node_mask = masked_true.node_mask
        bs, n = node_mask.shape

        true_pos = masked_true.pos[node_mask]       # q x 3
        masked_pred_pos = masked_pred.pos[node_mask]        # q x 3

        true_X = masked_true.X[node_mask]       # q x 4
        masked_pred_X = masked_pred.X[node_mask]        # q x 4

        true_charges = masked_true.charges[node_mask]       # q x 3
        masked_pred_charges = masked_pred.charges[node_mask]        # q x 3
        true_auxiliary_node_states = collections.OrderedDict(
            (name, masked_true.auxiliary_node_states[name][node_mask])
            for name in self.auxiliary_node_state_names
        )
        masked_pred_auxiliary_node_states = collections.OrderedDict(
            (name, masked_pred.auxiliary_node_states[name][node_mask])
            for name in self.auxiliary_node_state_names
        )

        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        masked_pred_E = masked_pred.E[edge_mask]        # r x 5
        true_E = masked_true.E[edge_mask]       # r x 5

        # Check that the masking is correct
        assert (true_X != 0.).any(dim=-1).all()
        assert (true_charges != 0.).any(dim=-1).all()
        assert (true_E != 0.).any(dim=-1).all()
        for true_aux_state in true_auxiliary_node_states.values():
            assert (true_aux_state != 0.).any(dim=-1).all()

        loss_pos = self.train_pos_mse(masked_pred_pos, true_pos) if true_X.numel() > 0 else 0.0
        loss_X = self.node_loss(masked_pred_X, true_X) if true_X.numel() > 0 else 0.0
        loss_charges = self.charges_loss(masked_pred_charges, true_charges) if true_charges.numel() > 0 else 0.0
        loss_E = self.edge_loss(masked_pred_E, true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(masked_pred.y, masked_true.y) if masked_true.y.numel() > 0 else 0.0
        auxiliary_node_state_losses = collections.OrderedDict(
            (
                name,
                self.auxiliary_node_state_losses[name](masked_pred_auxiliary_node_states[name], true_auxiliary_node_states[name])
                if true_auxiliary_node_states[name].numel() > 0 else 0.0,
            )
            for name in self.auxiliary_node_state_names
        )
        loss_auxiliary_node_states = (
            sum(auxiliary_node_state_losses.values()) / len(auxiliary_node_state_losses)
            if len(auxiliary_node_state_losses) > 0 else 0.0
        )

        batch_loss = (self.lambda_train[0] * loss_pos + self.lambda_train[1] * loss_X +
                      self.lambda_train[2] * loss_charges + self.lambda_train[3] * loss_E +
                      self.lambda_train[4] * loss_y +
                      self.lambda_train_auxiliary_node_states * loss_auxiliary_node_states)

        to_log = {"train_loss/pos_mse": self.lambda_train[0] * self.train_pos_mse.compute() if true_X.numel() > 0 else -1,
                  "train_loss/X_CE": self.lambda_train[1] * self.node_loss.compute() if true_X.numel() > 0 else -1,
                  "train_loss/charges_CE": self.lambda_train[2] * self.charges_loss.compute() if true_charges.numel() > 0 else -1,
                  "train_loss/E_CE": self.lambda_train[3] * self.edge_loss.compute() if true_E.numel() > 0 else -1.0,
                  "train_loss/y_CE": self.lambda_train[4] * self.y_loss.compute() if masked_true.y.numel() > 0 else -1.0,
                  "train_loss/batch_loss": batch_loss.item()} if log else None
        if log and len(self.auxiliary_node_state_names) > 0:
            to_log["train_loss/auxiliary_node_states_CE"] = self.lambda_train_auxiliary_node_states * (
                sum(self.auxiliary_node_state_losses[name].compute() for name in self.auxiliary_node_state_names)
                / len(self.auxiliary_node_state_names)
            )
            for name in self.auxiliary_node_state_names:
                to_log[f"train_loss/{name}_CE"] = self.lambda_train_auxiliary_node_states * self.auxiliary_node_state_losses[name].compute()

        if log and wandb.run:
            wandb.log(to_log, commit=True)
        return batch_loss, to_log

    def reset(self):
        for metric in [self.train_pos_mse, self.node_loss, self.charges_loss, self.edge_loss, self.y_loss, *self.auxiliary_node_state_losses.values()]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_pos_loss = self.train_pos_mse.compute().item() if self.train_pos_mse.total > 0 else -1.0
        epoch_node_loss = self.node_loss.compute().item() if self.node_loss.total_samples > 0 else -1.0
        epoch_charges_loss = self.charges_loss.compute().item() if self.charges_loss.total_samples > 0 else -1.0
        epoch_edge_loss = self.edge_loss.compute().item() if self.edge_loss.total_samples > 0 else -1.0
        epoch_y_loss = self.y_loss.compute().item() if self.y_loss.total_samples > 0 else -1.0

        to_log = {"train_epoch/pos_mse": epoch_pos_loss,
                  "train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/charges_CE": epoch_charges_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if len(self.auxiliary_node_state_names) > 0:
            aux_epoch_losses = {
                name: self.auxiliary_node_state_losses[name].compute().item()
                if self.auxiliary_node_state_losses[name].total_samples > 0 else -1.0
                for name in self.auxiliary_node_state_names
            }
            valid_aux_epoch_losses = [value for value in aux_epoch_losses.values() if value >= 0]
            to_log["train_epoch/auxiliary_node_states_CE"] = (
                sum(valid_aux_epoch_losses) / len(valid_aux_epoch_losses)
                if len(valid_aux_epoch_losses) > 0 else -1.0
            )
            for name, value in aux_epoch_losses.items():
                to_log[f"train_epoch/{name}_CE"] = value
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log
