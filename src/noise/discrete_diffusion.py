from typing import Tuple, Dict

import numpy as np

import torch
from torch import Tensor, IntTensor

from src.noise import reg_diffusion
from src.noise.core import NoiseSchedule, NoiseProcess
from src.datatypes.utils import one_hot


################################################################################
#                             DIFFUSION PROCESSES                              #
################################################################################

# adapted from https://github.com/cvignac/DiGress/blob/main/src/diffusion/diffusion_utils.py


def fill_in_masked_entries(
        x: Tensor,
        mask: Tensor,
    ) -> Tensor:
    
    if mask is None:
        return x
    
    x = x.clone()
    x[~mask] = 1 / x.shape[-1]

    return x


def clear_masked_entries(
        x: Tensor,
        mask: Tensor,
    ) -> Tensor:

    if mask is None:
        return x
    
    if mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    
    x = x * mask

    return x


def compute_prob_s_t_given_0(
        X_t: Tensor,
        Qt: Tensor,
        Qsb: Tensor,
        Qtb: Tensor
    ):
    """Borrowed from https://github.com/cvignac/DiGress/blob/main/src/diffusion/diffusion_utils.py"""

    X_t = X_t.flatten(start_dim=1, end_dim=-2)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


def normalize_probability(
        x: Tensor,
        norm_x: Tensor
    ) -> Tensor:

    if x is None: return None

    denominator = norm_x.sum(-1, keepdim=True)
    denominator[denominator == 0] = 1

    return x / denominator


def apply_markov_transition(
        datapoint: Tensor,
        transition_matrix: Tensor,
        mask: Tensor=None
    ) -> Tensor:

    s = datapoint.shape

    datapoint = fill_in_masked_entries(datapoint, mask)

    for _ in range(len(s) - 2):
        transition_matrix = transition_matrix.unsqueeze(1)
    
    prob_datapoint = datapoint.unsqueeze(1) @ transition_matrix

    # flatten
    prob_datapoint = prob_datapoint.flatten(end_dim=-2)
    # sample and reshape
    noisy_datapoint = prob_datapoint.multinomial(1)
    noisy_datapoint = noisy_datapoint.reshape(s[:-1])

    noisy_datapoint = clear_masked_entries(noisy_datapoint, mask)

    return noisy_datapoint


def apply_posterior_markov_transition(
        original_datapoint: Tensor,
        current_datapoint: Tensor,
        transition_matrix_bar_t: Tensor,
        transition_matrix_bar_t_1: Tensor,
        transition_matrix_t: Tensor,
        mask: Tensor=None
    ) -> Tensor:

    s = current_datapoint.shape

    current_datapoint = fill_in_masked_entries(current_datapoint, mask)

    # shape will be of the form (bs, n, d0, d_t-1)
    prob_s_and_t_given_0 = compute_prob_s_t_given_0(
        X_t =   current_datapoint,
        Qt =    transition_matrix_t,
        Qsb =   transition_matrix_bar_t_1,
        Qtb =   transition_matrix_bar_t
    )

    # # shape will be of the form (bs, n, d_t-1)
    prob_datapoint = weight_and_normalize_distribution(
        dist =      original_datapoint.flatten(start_dim=1, end_dim=-2),
        weights =   prob_s_and_t_given_0
    )

    # flatten
    prob_datapoint = prob_datapoint.flatten(end_dim=-2)
    # sample and reshape
    noisy_datapoint = prob_datapoint.multinomial(1).reshape(s[:-1])

    noisy_datapoint = clear_masked_entries(noisy_datapoint, mask)

    return noisy_datapoint


def weight_and_normalize_distribution(
        dist: Tensor,
        weights: Tensor
    ) -> Tensor:
    weighted_prob = dist.unsqueeze(-1) * weights        # bs, N, d0, d_t-1
    unnormalized_prob = weighted_prob.sum(dim=-2)       # bs, N, d_t-1
    unnormalized_prob[torch.sum(unnormalized_prob, dim=-1) == 0] = 1e-5
    return unnormalized_prob / torch.sum(unnormalized_prob, dim=-1, keepdim=True) # bs, n, d_t-1


from abc import ABC, abstractmethod

class DiscreteDiffusionProcess(NoiseProcess, ABC):

    def __init__(
            self,
            schedule : NoiseSchedule,
            num_cls: int,
            return_one_hot: bool=True,
            **kwargs
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """
        # call super for the NoiseProcess
        super().__init__(schedule=schedule)

        self.num_cls = num_cls
        self.return_one_hot = return_one_hot


    @abstractmethod
    def stationary_distribution(self, device=None, **kwargs):
        raise NotImplementedError
    

    def _to_one_hot_if_enabled(self, x: Tensor) -> Tensor:
        if self.return_one_hot:
            return one_hot(x, num_classes=self.num_cls, dtype=torch.float)
        return x


    def compute_interpolated_transition_matrix(
            self,
            original_datapoint: Tensor,
            iden_mult: Tensor,
            stat_mult: Tensor,
            **kwargs
        ):

        device = original_datapoint.device

        # get diffusion parameter
        iden_mult = iden_mult.unsqueeze(-1).unsqueeze(-1) # (bs, 1, 1)
        stat_mult = stat_mult.unsqueeze(-1).unsqueeze(-1) # (bs, 1, 1)
        identity = torch.eye(self.num_cls, device=device).unsqueeze(0) # (1, d, d)
        stat_dist = self.stationary_distribution(device=device).unsqueeze(0) # (1, d)
        stat_dist = stat_dist.repeat(self.num_cls, 1).unsqueeze(0) # (1, d, d)

        # exact definition from the discrete diffusion paper
        transition_matrix = iden_mult * identity + stat_mult * stat_dist # (bs, d, d)

        return transition_matrix


    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            shape: Tuple[int],
            device: torch.device=None
        ) -> Tensor:
        
        # sample from a multinomial with the given shape
        stat_dist = self.stationary_distribution(device=device)

        # get flattened number of samples
        num_samples = np.prod(shape).item()

        if num_samples == 0:
            x = torch.zeros(shape, device=device)
        else:
            # sample from the multinomial
            x = stat_dist.multinomial(
                num_samples, replacement=True
            ).reshape(shape)

        return self._to_one_hot_if_enabled(x)


    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(
            self,
            current_datapoint: Tensor,
            t: IntTensor,
            **kwargs
        ):

        # get diffusion parameter
        beta_t: Tensor = self.get_params_next(t, **kwargs)

        transition_matrix = self.compute_interpolated_transition_matrix(
            original_datapoint=current_datapoint,
            iden_mult=1-beta_t,
            stat_mult=beta_t
        )

        return transition_matrix
        


    def apply_noise_next(
            self,
            current_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        next_datapoint = apply_markov_transition(
            datapoint =			current_datapoint,
            transition_matrix =	noise,
            **kwargs
        )

        return self._to_one_hot_if_enabled(next_datapoint)

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################
    
    def sample_noise_from_original(
            self,
            original_datapoint: Tensor,
            t: IntTensor,
            **kwargs
        ):

        # get diffusion parameter
        alpha_bar_t: Tensor = self.get_params_from_original(t, **kwargs)
        
        transition_matrix = self.compute_interpolated_transition_matrix(
            original_datapoint=original_datapoint,
            iden_mult=alpha_bar_t,
            stat_mult=1-alpha_bar_t
        )

        return transition_matrix


    def apply_noise_from_original(
            self,
            original_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        noisy_datapoint = apply_markov_transition(
            datapoint =			original_datapoint,
            transition_matrix =	noise,
            **kwargs
        )

        return self._to_one_hot_if_enabled(noisy_datapoint)
    
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(
            self,
            original_datapoint: Tensor,
            current_datapoint: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        trans_mat_bar_t = self.sample_noise_from_original(original_datapoint, t, **kwargs)
        trans_mat_bar_t_1 = self.sample_noise_from_original(original_datapoint, t-1, **kwargs)
        trans_mat_t = self.sample_noise_next(current_datapoint, t, **kwargs)

        return (
            trans_mat_bar_t,
            trans_mat_bar_t_1,
            trans_mat_t
        )


    def apply_noise_posterior(
            self,
            original_datapoint: Tensor,
            current_datapoint: Tensor,
            noise: Tuple[Tensor],
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        trans_mat_bar_t, trans_mat_bar_t_1, trans_mat_t = noise

        datapoint_t_1 = apply_posterior_markov_transition(
            original_datapoint =	    original_datapoint,
            current_datapoint =		    current_datapoint,
            transition_matrix_bar_t =	trans_mat_bar_t,
            transition_matrix_bar_t_1 =	trans_mat_bar_t_1,
            transition_matrix_t =		trans_mat_t,
            **kwargs
        )

        return self._to_one_hot_if_enabled(datapoint_t_1)


@reg_diffusion.register('discrete_uniform')
class UniformDiscreteDiffusionProcess(DiscreteDiffusionProcess):

    def stationary_distribution(self, device, **kwargs):
        return torch.ones(self.num_cls, device=device) / self.num_cls


@reg_diffusion.register('discrete_marginal')
class MarginalDiscreteDiffusionProcess(DiscreteDiffusionProcess):

    def __init__(
            self,
            schedule : NoiseSchedule,
            num_cls: int,
            minimum_number_updates: int=100
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """
        # call super for the NoiseProcess
        super().__init__(schedule=schedule, num_cls=num_cls)

        self.accumulating = True
        self.curr_num_updates = 0
        self.minimum_number_updates = minimum_number_updates

        self.register_buffer('histogram', torch.zeros(num_cls))
        self.register_buffer('marginal', torch.zeros(num_cls))


    
    def update(self, labels):
        if self.accumulating:
            updating = labels.shape[0] > 1
            self.curr_num_updates += 1 if updating else 0

            if updating:

                idx, new_histogram = torch.unique(labels, return_counts=True, sorted=False)
                self.histogram[idx] += new_histogram

                self.marginal = self.histogram / self.histogram.sum()


    def stop_updating(self):
        if self.curr_num_updates >= self.minimum_number_updates:
            self.accumulating = False
        

    def stationary_distribution(self, device, **kwargs):
        if self.curr_num_updates == 0:
            return torch.ones(self.num_cls, device=device) / self.num_cls
        return self.marginal