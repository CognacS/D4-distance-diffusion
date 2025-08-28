from typing import Tuple, Dict

import torch
from torch import Tensor, IntTensor

from src.noise import reg_diffusion
from src.noise.core import NoiseSchedule, NoiseProcess

from src.noise.schedules import DiffusionProcessException, CosineDiffusionSchedule


################################################################################
#                             DIFFUSION PROCESSES                              #
################################################################################
@reg_diffusion.register('vp_gaussian')
class VPGaussianDiffusionProcess(NoiseProcess):

    """
    This is the Variance Preserving Gaussian Diffusion Process.
    """


    def __init__(
            self,
            schedule : NoiseSchedule,
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


    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            shape,
            device: torch.device=None,
        ) -> Tensor:

        # sample from standard normal
        datapoint = torch.randn(*shape, device=device)
        
        return datapoint


    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: Tensor, t: IntTensor, **kwargs):

        return torch.randn_like(current_datapoint)


    def apply_noise_next(
            self,
            current_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        beta_t = self.get_params_next(t, **kwargs)
        alpha_bar_t = self.get_params_from_original(t, **kwargs)
        for _ in range(len(current_datapoint.shape) - 1):
            beta_t = beta_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        
        alpha_t = 1 - beta_t
        sigma_t = torch.sqrt(1 - alpha_bar_t ** 2 + alpha_t ** 2)

        next_datapoint = alpha_t * current_datapoint + sigma_t * noise

        return next_datapoint


    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def sample_noise_from_original(
            self,
            original_datapoint: Tensor,
            t: IntTensor,
            **kwargs
        ):
        
        return torch.randn_like(original_datapoint)


    def apply_noise_from_original(
            self,
            original_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        alpha_bar_t = self.get_params_from_original(t, **kwargs)
        for _ in range(len(original_datapoint.shape) - 1):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        sigma_bar_t = self.get_sigma_bar_t(alpha_bar_t)

        noisy_datapoint = alpha_bar_t * original_datapoint + sigma_bar_t * noise

        return noisy_datapoint
    
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

        return torch.randn_like(original_datapoint)


    def apply_noise_posterior(
            self,
            original_datapoint: Tensor,
            current_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        # initial schedule parameters
        alpha_bar_t = self.get_params_from_original(t, **kwargs)
        alpha_bar_t_1 = self.get_params_from_original(t-1, **kwargs)
        beta_t = torch.clip(self.get_params_next(t, **kwargs), max=0.9999)
        
        # ensure the parameters are broadcastable to the shape of the datapoint
        for _ in range(len(original_datapoint.shape) - 1):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_t_1 = alpha_bar_t_1.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)
        
        # compute the parameters for the posterior transition
        alpha_t = 1 - beta_t
        sigma_bar_t_1_sq = 1 - alpha_bar_t_1 ** 2
        sigma_bar_t_sq = 1 - alpha_bar_t ** 2

        # compute auxiliary factors to be used
        sigma_bar_t_sq_ratio = sigma_bar_t_1_sq / sigma_bar_t_sq
        aux_factor = (1 - (alpha_t ** 2) * sigma_bar_t_sq_ratio)
        
        # compute factors for the noisy datapoint
        factor_mu_t = alpha_bar_t_1 * aux_factor
        factor_nu_t = alpha_t * sigma_bar_t_sq_ratio
        factor_sigma_tilde_t = torch.sqrt(sigma_bar_t_1_sq * aux_factor)
        
        noisy_datapoint = factor_mu_t * original_datapoint + factor_nu_t * current_datapoint + factor_sigma_tilde_t * noise

        return noisy_datapoint
    
    def get_sigma_bar_t(self, alpha_bar_t):
        """
        Computes the sigma_bar_t parameter from the alpha_bar_t parameter.
        This is used to compute the noise in the posterior transition.
        """
        return torch.sqrt(1 - alpha_bar_t ** 2)

        
################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

DIFFUSION_SCHEDULE_COSINE = 'cosine'

DIFFUSION_PROCESS_CONTINUOUS = 'continuous'

def resolve_cont_diffusion_schedule(name: str) -> type:
    if name == DIFFUSION_SCHEDULE_COSINE:
        return CosineDiffusionSchedule
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion schedule name: {name}')

def resolve_cont_diffusion_process(name: str) -> type:
    if name == DIFFUSION_PROCESS_CONTINUOUS:
        return VPGaussianDiffusionProcess
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion process name: {name}')