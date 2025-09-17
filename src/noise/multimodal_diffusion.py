from typing import Tuple, Dict, Any, Callable

import torch
from torch import Tensor, IntTensor
from torch.nn import ModuleDict

from src.noise.core import NoiseProcess


################################################################################
#                             DIFFUSION PROCESSES                              #
################################################################################

class MultimodalDiffusionProcess(NoiseProcess):

    def __init__(
            self,
            diffusion_procs_per_data: Dict[str, NoiseProcess],
            **kwargs
        ):

        schedule = list(diffusion_procs_per_data.values())[0].schedule

        super().__init__(schedule=schedule)

        self.diffusion_procs_per_data = ModuleDict(diffusion_procs_per_data)



    def _apply_per_data(
            self,
            method: str,
            kwargs_per_data: Dict[str, Dict[str, Tensor]],
            **kwargs
        ):

        res = {}

        for data_name, kwargs_d in kwargs_per_data.items():
            diff_proc = self.diffusion_procs_per_data[data_name]
            res[data_name] = getattr(diff_proc, method)(**kwargs_d, **kwargs)

        return res
    

    def _apply_per_data_per_kwargs(
            self,
            method: str,
            data_per_kwargs: Dict[str, Dict[str, Tensor]],
            **kwargs
        ):

        # reverse keys structure:
        # from kwarg -> data -> value to data -> kwarg -> value
        kwargs_per_data = {k_data: {k_kw: v_kw[k_data] for k_kw, v_kw in data_per_kwargs.items()} for k_data in data_per_kwargs[list(data_per_kwargs.keys())[0]]}

        return self._apply_per_data(method, kwargs_per_data, **kwargs)



    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            kwargs_per_data: Dict[str, Dict[str, Tensor]],
            device: torch.device=None
        ) -> Dict[str, Tensor]:
        
        return self._apply_per_data('sample_stationary', kwargs_per_data, device=device)

    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(
            self,
            current_datapoint: Dict[str, Tensor],
            t: IntTensor,
            kwargs_per_data: Dict[str, Any]=None,
            **kwargs
        ):

        kwargs_per_data = {} if kwargs_per_data is None else kwargs_per_data
        return self._apply_per_data_per_kwargs('sample_noise_next', dict(current_datapoint=current_datapoint, **kwargs_per_data), t=t, **kwargs)
        


    def apply_noise_next(
            self,
            current_datapoint: Dict[str, Tensor],
            noise: Dict[str, Tensor],
            t: IntTensor,
            kwargs_per_data: Dict[str, Any]=None,
            **kwargs
        ) -> Dict[str, Tensor]:

        kwargs_per_data = {} if kwargs_per_data is None else kwargs_per_data
        return self._apply_per_data_per_kwargs('apply_noise_next', dict(current_datapoint=current_datapoint, noise=noise, **kwargs_per_data), t=t, **kwargs)

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################
    
    def sample_noise_from_original(
            self,
            original_datapoint: Dict[str, Tensor],
            t: IntTensor,
            kwargs_per_data: Dict[str, Any]=None,
            **kwargs
        ):

        kwargs_per_data = {} if kwargs_per_data is None else kwargs_per_data
        return self._apply_per_data_per_kwargs('sample_noise_from_original', dict(original_datapoint=original_datapoint, **kwargs_per_data), t=t, **kwargs)


    def apply_noise_from_original(
            self,
            original_datapoint: Dict[str, Tensor],
            noise: Dict[str, Tensor],
            t: IntTensor,
            kwargs_per_data: Dict[str, Any]=None,
            **kwargs
        ) -> Dict[str, Tensor]:

        kwargs_per_data = {} if kwargs_per_data is None else kwargs_per_data
        return self._apply_per_data_per_kwargs('apply_noise_from_original', dict(original_datapoint=original_datapoint, noise=noise, **kwargs_per_data), t=t, **kwargs)

    
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(
            self,
            original_datapoint: Dict[str, Tensor],
            current_datapoint: Dict[str, Tensor],
            t: IntTensor,
            kwargs_per_data: Dict[str, Any]=None,
            **kwargs
        ) -> Dict[str, Tensor]:

        kwargs_per_data = {} if kwargs_per_data is None else kwargs_per_data

        return self._apply_per_data_per_kwargs(
            'sample_noise_posterior',
            dict(
                original_datapoint=original_datapoint,
                current_datapoint=current_datapoint,
                **kwargs_per_data
            ),
            t=t,
            **kwargs
        )


    def apply_noise_posterior(
            self,
            original_datapoint: Dict[str, Tensor],
            current_datapoint: Dict[str, Tensor],
            noise: Dict[str, Tensor],
            t: IntTensor,
            kwargs_per_data: Dict[str, Any]=None,
            **kwargs
        ) -> Dict[str, Tensor]:
        
        kwargs_per_data = {} if kwargs_per_data is None else kwargs_per_data

        return self._apply_per_data_per_kwargs(
            'apply_noise_posterior',
            dict(
                original_datapoint=original_datapoint,
                current_datapoint=current_datapoint,
                noise=noise,
                **kwargs_per_data
            ),
            t=t,
            **kwargs
        )


from typing import Any
from abc import ABC, abstractmethod

class StructuredMultimodalDiffusionProcess(MultimodalDiffusionProcess, ABC):

    @abstractmethod
    def map_datapoint_to_dict(self, datapoint: Any) -> Dict[str, Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def compose_back(self, datapoint: Dict[str, Tensor], **kwargs) -> Any:
        raise NotImplementedError
    
    def kwargs_per_data_from_datapoint(self, datapoint: Any) -> Dict[str, Dict[str, Tensor]]:
        return {}


    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(
            self,
            current_datapoint: Any,
            t: IntTensor,
            **kwargs
        ):

        kwargs_per_data = self.kwargs_per_data_from_datapoint(current_datapoint)

        return super().sample_noise_next(self.map_datapoint_to_dict(current_datapoint), t, kwargs_per_data, **kwargs)
        


    def apply_noise_next(
            self,
            current_datapoint: Any,
            noise: Dict[str, Tensor],
            t: IntTensor,
            **kwargs
        ) -> Dict[str, Tensor]:

        kwargs_per_data = self.kwargs_per_data_from_datapoint(current_datapoint)

        return self.compose_back(
            super().apply_noise_next(
                self.map_datapoint_to_dict(current_datapoint), noise, t, kwargs_per_data, **kwargs
            ),
            current_datapoint
        )
    
    
    def sample_noise_from_original(
            self,
            original_datapoint: Any,
            t: IntTensor,
            **kwargs
        ):

        kwargs_per_data = self.kwargs_per_data_from_datapoint(original_datapoint)

        return super().sample_noise_from_original(self.map_datapoint_to_dict(original_datapoint), t, kwargs_per_data, **kwargs)
    

    def apply_noise_from_original(
            self,
            original_datapoint: Any,
            noise: Dict[str, Tensor],
            t: IntTensor,
            **kwargs
        ) -> Dict[str, Tensor]:

        kwargs_per_data = self.kwargs_per_data_from_datapoint(original_datapoint)

        return self.compose_back(
            super().apply_noise_from_original(
                self.map_datapoint_to_dict(original_datapoint), noise, t, kwargs_per_data, **kwargs
            ),
            original_datapoint
        )
    

    def sample_noise_posterior(
            self,
            original_datapoint: Any,
            current_datapoint: Any,
            t: IntTensor,
            **kwargs
        ) -> Dict[str, Tensor]:

        kwargs_per_data = self.kwargs_per_data_from_datapoint(original_datapoint)

        return super().sample_noise_posterior(
            self.map_datapoint_to_dict(original_datapoint),
            self.map_datapoint_to_dict(current_datapoint),
            t,
            kwargs_per_data,
            **kwargs
        )
    

    def apply_noise_posterior(
            self,
            original_datapoint: Any,
            current_datapoint: Any,
            noise: Dict[str, Tensor],
            t: IntTensor,
            **kwargs
        ) -> Dict[str, Tensor]:

        kwargs_per_data = self.kwargs_per_data_from_datapoint(original_datapoint)

        return self.compose_back(
            super().apply_noise_posterior(
                self.map_datapoint_to_dict(original_datapoint),
                self.map_datapoint_to_dict(current_datapoint),
                noise,
                t,
                kwargs_per_data,
                **kwargs
            ),
            original_datapoint
        )
    


class ChainedNoiseProcess(NoiseProcess):
    """
    Class for chaining many noise processes together.
    """

    def __init__(
            self,
            noise_process_before: StructuredMultimodalDiffusionProcess,
            noise_process_after: StructuredMultimodalDiffusionProcess,
            combine_stationary: Callable=None,
            chain_sample_next: Callable=None,
            chain_sample_from_original: Callable=None,
            chain_sample_posterior: Callable=None,
            **kwargs
        ):
        schedule = noise_process_before.schedule
        super().__init__(schedule=schedule)
        
        self.noise_process_before = noise_process_before
        self.noise_process_after = noise_process_after
        self.combine_stationary = combine_stationary
        self.chain_sample_next = chain_sample_next
        self.chain_sample_from_original = chain_sample_from_original
        self.chain_sample_posterior = chain_sample_posterior
        
        
    def sample_stationary(self, **kwargs):
        stat1 = self.noise_process_before.sample_stationary(**kwargs)
        stat2 = self.noise_process_after.sample_stationary(**kwargs)
        if self.combine_stationary is not None:
            return self.combine_stationary(stat1, stat2)
        else:
            return (stat1, stat2)
     
    
    def sample_next(self, current_datapoint, t, **kwargs):
        current_datapoint = self.noise_process_before.sample_next(current_datapoint, t, **kwargs)
        if self.chain_sample_next is not None:
            current_datapoint = self.chain_sample_next(current_datapoint, t, **kwargs)
        next_datapoint = self.noise_process_after.sample_next(current_datapoint, t, **kwargs)
        return next_datapoint
    
    def sample_from_original(self, original_datapoint, t, **kwargs):
        original_datapoint = self.noise_process_before.sample_from_original(original_datapoint, t, **kwargs)
        if self.chain_sample_from_original is not None:
            original_datapoint = self.chain_sample_from_original(original_datapoint, t, **kwargs)
        step_t_datapoint = self.noise_process_after.sample_from_original(original_datapoint, t, **kwargs)
        return step_t_datapoint
    
    def sample_posterior(self, original_datapoint, current_datapoint, t, **kwargs):
        original_datapoint = self.noise_process_before.sample_posterior(original_datapoint, current_datapoint, t, **kwargs)
        if self.chain_sample_posterior is not None:
            original_datapoint = self.chain_sample_posterior(original_datapoint, t, **kwargs)
        prev_datapoint = self.noise_process_after.sample_posterior(original_datapoint, current_datapoint, t, **kwargs)
        return prev_datapoint