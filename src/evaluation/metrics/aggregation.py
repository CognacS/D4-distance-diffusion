from typing import Dict

from torch import nn

from src.evaluation import reg_metrics
import src.evaluation.metrics as m_list


################################################################################
#                             SUMMARIZING METRICS                             #
################################################################################

@reg_metrics.register(m_list.KEY_WEIGHTED_SUM_METRIC)
class WeightedSumMetric(nn.Module):

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights


    def __call__(self, **kwargs) -> Dict:

        value = sum([kwargs[k] * w for k, w in self.weights.items()])
        
        return {m_list.KEY_WEIGHTED_SUM_METRIC: value}