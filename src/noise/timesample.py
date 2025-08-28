import numpy as np

import torch
from torch import IntTensor, Tensor

from src.noise import reg_timesampler
from src.noise.core import TimeSampler

class TimeSamplingException(Exception):
    pass


################################################################################
#                              TIME STEP SAMPLERS                              #
################################################################################
@reg_timesampler.register('uniform')
class UniformTimeSampler(TimeSampler):

    def sample_time(self, **kwargs) -> IntTensor:
        max_time = kwargs['max_time']
        sampled_time = np.random.randint(low=0, high=max_time+1)
        if isinstance(sampled_time, np.ndarray):
            return torch.from_numpy(sampled_time)
        else:
            return IntTensor([sampled_time])
        
@reg_timesampler.register('combinatorial')
class CombinatorialTimeSampler(TimeSampler):

    def sample_time(self, **kwargs) -> IntTensor:
        max_time = kwargs['max_time']   # 1
        if isinstance(max_time, Tensor):
            max_time = int(max_time.max().item())
        num_nodes = kwargs['n0']        # bs

        fractions_notalive_linear = torch.arange(0, ) / max_time # max_time+1
        fractions_alive_linear = 1 - fractions_notalive_linear
        average_notalive = num_nodes.unsqueeze(1) * fractions_notalive_linear.unsqueeze(0)  # bs, max_time+1
        average_alive = num_nodes.unsqueeze(1) * fractions_alive_linear.unsqueeze(0)    # bs, max_time+1

        gamma_notalive = torch.lgamma(average_notalive + 1) # bs, max_time+1
        gamma_alive = torch.lgamma(average_alive + 1)    # bs, max_time+1
        gamma_sum = gamma_notalive + gamma_alive    # bs, max_time+1

        gamma_diffs = gamma_sum.unsqueeze(2) - gamma_sum.unsqueeze(1)   # bs, max_time+1, max_time+1

        probs = 1 / torch.exp(gamma_diffs).sum(dim=2)  # bs, max_time+1

        print(probs)

        # sample time
        sampled_time = torch.multinomial(probs, num_samples=1).squeeze(1) # bs

        return sampled_time

        
        
@reg_timesampler.register('constant')
class ConstantTimeSampler(TimeSampler):

    def __init__(self, constant_time: int):
        self.constant_time = constant_time

    def sample_time(self, **kwargs) -> IntTensor:
        max_time = kwargs['max_time']
        if isinstance(max_time, np.ndarray):
            return torch.full(max_time.shape, self.constant_time)
        else:
            return IntTensor([self.constant_time])


################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

TIMESAMPLER_UNIFORM = 'uniform'

def resolve_timesampler(name: str) -> type:
    if name == TIMESAMPLER_UNIFORM:
        return UniformTimeSampler
    else:
        raise TimeSamplingException(f'Could not resolve removal time sampler name: {name}')