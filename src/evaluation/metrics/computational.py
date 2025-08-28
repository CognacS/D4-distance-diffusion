from typing import Dict, List, Tuple, Any, Optional, Union, Callable

import numpy as np
import torch
from torch import nn, Tensor
import re
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


import src.evaluation.metrics as m_list
from src.evaluation.metrics.core import Metric

from src.evaluation import reg_metrics

class BaseComputationalMetric(Metric):

    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs) -> Dict:
        raise NotImplementedError
    

def is_computational_metric(metric: Callable) -> bool:
    return isinstance(metric, BaseComputationalMetric)

def contains_computational_metrics(metrics: Dict[str, Callable]) -> bool:
    return any([is_computational_metric(metric) for metric in metrics.values()])


################################################################################
#                           PROCESS SAMPLING METRICS                           #
################################################################################

@reg_metrics.register(m_list.KEY_SAMPLING_TIME)
class SamplingTimeMetric(BaseComputationalMetric):
    
        def __init__(self):
            super().__init__()
    
        def __call__(self, comp_data) -> Dict:
            samp_time_data = comp_data['sampling']['time']
            return {m_list.KEY_SAMPLING_TIME: samp_time_data['end'] - samp_time_data['start']}
        

@reg_metrics.register(m_list.KEY_SAMPLING_MEMORY)
class SamplingMemoryMetric(BaseComputationalMetric):
            
        def __init__(self):
            super().__init__()
    
        def __call__(self, comp_data) -> Dict:
            return {m_list.KEY_SAMPLING_MEMORY: comp_data['sampling']['memory']['peak']}