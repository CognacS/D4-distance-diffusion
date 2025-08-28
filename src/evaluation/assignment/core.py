from typing import Dict, List, Any, Callable, Type
from src.evaluation import reg_metrics
from omegaconf import DictConfig

from src.evaluation.metrics import Metric

from torch import Tensor, nn

class AssignmentException(Exception):
    pass


class Assignment(nn.Module):

    def __init__(self, how_many_to_generate: int, enabled_metrics: str='all', metrics_overrides: Dict[str, Dict]=None, **kwargs):
        super().__init__()
        
        self.how_many_to_generate = how_many_to_generate
        self.enabled_metrics = enabled_metrics
        self.metrics_overrides = metrics_overrides if metrics_overrides is not None else {}
        self._metrics = {}

    
    def add_metric(self, name: str, metric: Type, *args, **kwargs) -> None:
        """
        Add a metric to the assignment. If it is not enabled, it will not be added. If the metric has
        overrides, they will be applied.
        Args:
            name (str): the name of the metric
            metric (Metric): the metric object to add
            
        
        """

        if issubclass(metric, Metric):
            # insert in metrics only if enabled (all -> all metrics enabled)
            if self.enabled_metrics == 'all' or name in self.enabled_metrics:
                
                # if metric should be overridden, do it
                overrides = self.metrics_overrides.get(name, {})
                kwargs.update(overrides)

                # instance of metric
                metric = metric(*args, **kwargs)
                # add to metrics dictionary
                self._metrics[name] = metric

        else:
            raise ValueError(f'Type {metric} is not a Metric subclass.')


    def compute_if_exists(self, name: str, *args, **kwargs) -> Tensor:
        """
        Compute a metric if it exists in the assignment, otherwise return None
        Args:
            name (str): the name of the metric
        """
        if name in self.metrics:
            return self.metrics[name](*args, **kwargs)
        else:
            return {}


    def has_metric(self, name: str) -> bool:
        return name in self._metrics


    @property
    def metrics(self) -> Dict[str, Callable]:
        return self._metrics



class ClonableWithSplitsMixin:

    def add_params_to_clone(self, param_list: List[str]) -> None:
        """
        Add parameters to be cloned. Notice that usual assignment parameters are already cloned.
        Args:
            param_list (List[str]): the list of parameters to be cloned
        """
        self._params_to_clone = param_list


    def clone_with_another_split(self: Assignment, split: str) -> Assignment:
        """
        Clone the assignment with another split
        Args:
            split (str): the new split
        """
        
        # get the parameters to clone
        params = {k: getattr(self, k) for k in self._params_to_clone}
        params.update({
            'how_many_to_generate': self.how_many_to_generate,
            'enabled_metrics': self.enabled_metrics,
            'metrics_overrides': self.metrics_overrides
        })
        # create the new assignment
        new_assignment = self.__class__(split=split, **params)

        return new_assignment