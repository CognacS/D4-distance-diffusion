import numpy as np
import torch
from torch import Tensor

from src.noise import reg_schedule
from src.noise.core import NoiseSchedule

################################################################################
#                              UTILITY FUNCTIONS                               #
################################################################################

class DiffusionProcessException(Exception):
    pass


def cosine_beta_schedule_discrete(max_steps, s=0.008, exponent=1.0):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = max_steps + 1
    x = np.linspace(0, 1, steps, dtype=np.float32)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x + s) ** exponent) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = np.concatenate([np.ones(1), alphas_cumprod[1:] / alphas_cumprod[:-1]])
    betas = 1 - alphas
    return betas.squeeze(), alphas_cumprod.squeeze()


def time_to_long(t: Tensor, timesteps: int):
    if t.dtype == torch.long:
        return t

    elif t.dtype == torch.int:
        return t.long()
    
    elif t.dtype == torch.float:
        t_int = torch.round(t * timesteps)
        assert torch.all(t_int >= 0) and torch.all(t_int <= timesteps), f'time tensor t has values outside [0,1]'
        return t_int.long()
    
    else:
        raise DiffusionProcessException(
            f'Given time tensor t has wrong dtype: {t.dtype}. Should be long, integer or float in [0,1]'
        )


################################################################################
#                         DIFFUSION PROCESS SCHEDULES                          #
################################################################################

@reg_schedule.register('cosine')
class CosineDiffusionSchedule(NoiseSchedule):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, max_time: int, s: float = 0.008, exponent: float = 1.0):
        super().__init__()

        self.max_time = max_time

        # compute betas (parameter next)
        betas, alphas_bar = cosine_beta_schedule_discrete(max_time, s=s, exponent=exponent)
        # clamp values as in the original paper
        #betas = torch.clamp(torch.from_numpy(betas), min=0, max=0.999)
        betas = torch.from_numpy(betas)
        alphas_bar = torch.from_numpy(alphas_bar)
        self.register_buffer('betas', betas.float())

        # compute alpha = 1 - beta
        #alphas = 1 - self.betas

        # recompute alpha_bar (parameter time 0->t)
        #log_alpha = torch.log(alphas)
        #log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        #self.register_buffer('alphas_bar', torch.exp(log_alpha_bar))
        self.register_buffer('alphas_bar', alphas_bar.float())


    def params_next(self, t: Tensor, **kwargs):
        t_int = time_to_long(t, self.max_time)

        return self.betas[t_int]

    def params_time_t(self, t: Tensor, **kwargs):
        t_int = time_to_long(t, self.max_time)

        return self.alphas_bar[t_int]

    def params_posterior(self, t, **kwargs):
        raise NotImplementedError

    def get_max_time(self, **kwargs):
        return self.max_time
    

    def forward(self, t: Tensor, **kwargs):
        return self.params_next(t)