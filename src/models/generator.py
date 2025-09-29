from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Metric
import wandb
import time
import numpy as np
import math


from src.evaluation.assignment.core import Assignment, ClonableWithSplitsMixin
from logging import Logger

from copy import copy


class Generator(ABC, pl.LightningModule):

    IGNORED_HPARAMS = [
        'dataset_info',
        'test_assignment',
        'console_logger'
    ]

    def __init__(
            self,
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):
        super().__init__()

        self.dataset_info = dataset_info
        self.test_assignment = test_assignment
        self.console_logger = console_logger
        


    @abstractmethod
    def sample(
        self,
        num_samples: int,
        conditioning_elems: Optional[Dict]=None,
        **kwargs
    ):
        raise NotImplementedError
    



class GeneratorWithEvaluation(Generator):


    def __init__(
            self,
            validation: Dict,
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):

        super().__init__(
            dataset_info=dataset_info,
            test_assignment=test_assignment,
            console_logger=console_logger
        )

        ####################  VALIDATION ASSIGNMENT SETUP  #####################
        self.validation_config = validation

        if self.validation_config['do_assignment']:
            self.add_valid_assignment()
        else:
            self.valid_assignment = None

        ############################  EXTRA SETUP  #############################
        self.start_time = time.time()
        self.total_elapsed_time = 0
        self.max_memory_reserved = 0


    def metrics_to_paths_structure(self, metrics: Dict):
        """
        Convert metrics dictionary to a paths structure
        """
        out_metrics = {}

        for name, m in metrics.items():
            if isinstance(m, dict):
                for k, v in m.items():
                    out_metrics[f'{name}/{k}'] = v
            else:
                out_metrics[name] = m

        return out_metrics
    

    def apply_prefix(self, metrics, prefix):
        """
        Build a paths structure of a dictionary of metrics, and apply a prefix to the paths
        """
        out_metrics = self.metrics_to_paths_structure(metrics)
        return {f'{prefix}/{k}'.lower(): v for k, v in out_metrics.items()}


    def log_wandb_objects(self, objects):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_metrics(objects)
        #wandb.log({name: wb_object})

    
    def log_wandb_histograms(self, histograms):
        if isinstance(self.logger, WandbLogger):
            # wandb accepts a maximum number of 512 bins
            hists = {}
            for k, v in histograms.items():
                if len(v[0]) <= 512:
                    hists[k] = wandb.Histogram(np_histogram=v)
                else:
                    self.console_logger.warning(f'Cannot log histogram {k} because it has more than 512 bins')
            self.log_wandb_objects(hists)


    def get_metrics_values(self, metrics):
        ret = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                ret.update(self.get_metrics_values(v))
            elif isinstance(v, Metric):
                ret[k] = v.compute().detach().cpu().item()
            else:
                ret[k] = v

        return ret
        


    def add_valid_assignment(self):
        if self.test_assignment is None:
            self.valid_assignment = None
            return

        if isinstance(self.test_assignment, ClonableWithSplitsMixin):
            self.console_logger.info('Creating a validation assignment with the same metrics as the test assignment')
            self.valid_assignment = self.test_assignment.clone_with_another_split('valid')
        else:
            self.console_logger.warning('Validation assignment will be the same as the test assignment')
            self.valid_assignment = copy(self.test_assignment)

        self.valid_assignment.how_many_to_generate = self.validation_config['how_many_to_generate']

        


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        overtime = 0 if self.start_time is None else time.time() - self.start_time
        checkpoint['total_elapsed_time'] = self.total_elapsed_time + overtime
        checkpoint['max_memory_reserved'] = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.total_elapsed_time = checkpoint['total_elapsed_time']
        self.max_memory_reserved = checkpoint['max_memory_reserved']
        
    
    def concatenate_conditioning_elems(self, conditioning_elems, num_samples: int):
        pass


    @torch.no_grad()
    def compute_sampling_data(self, num_samples, conditioning_elems=None, sampling_kwargs=None):

        self.console_logger.info('Sampling some graphs...')

        if conditioning_elems is not None:

            conditioning_elems = self.concatenate_conditioning_elems(
                conditioning_elems,
                num_samples=num_samples
            )
        
        # initialize for process metrics
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()
        
        sampling_kwargs = sampling_kwargs if sampling_kwargs is not None else {}

        # sample required graphs
        samples = self.sample(
            num_samples = num_samples,
            conditioning_elems = conditioning_elems,
            **sampling_kwargs
        )

        # end for process metrics
        end_time = time.time()
        peak_memory_usage = float(torch.cuda.max_memory_allocated(0))

        self.console_logger.info(f'Done. Sampling took {end_time - start_time:.2f} seconds\n')

        # compute some statistics on the generated graphs
        num_nodes = [s.num_nodes for s in samples]
        num_edges = [s.num_edges for s in samples]
        indegree = [s.indegree.cpu().tolist() for s in samples]
        
        min_indegree = min([min(indeg) for indeg in indegree])
        max_indegree = max([max(indeg) for indeg in indegree])
        mean_indegree = np.mean([np.mean(indeg) for indeg in indegree])
        num_edges_hist_first = np.histogram(indegree[0], bins=np.arange(min_indegree-0.5, max_indegree+1.5), density=True)
        num_edges_hist = np.histogram(np.concatenate(indegree), bins=np.arange(min_indegree-0.5, max_indegree+1.5), density=True)

        # log statistics
        self.console_logger.info(f'Number of nodes per graph: avg:{np.mean(num_nodes)}, min:{np.min(num_nodes)}, max:{np.max(num_nodes)}')
        self.console_logger.info(f'Number of edges per graph: avg:{np.mean(num_edges)}, min:{np.min(num_edges)}, max:{np.max(num_edges)}')
        self.console_logger.info(f'Indegree per graph: avg:{mean_indegree}, min:{min_indegree}, max:{max_indegree}')

        # compute histogram of number of nodes
        num_nodes_hist = np.histogram(num_nodes, bins=np.arange(min(num_nodes)-0.5, max(num_nodes)+1.5), density=True)

        hists = {
            'num_nodes_hist': num_nodes_hist,
            'num_edges_hist_first': num_edges_hist_first,
            'num_edges_hist': num_edges_hist
        }

        sampling_data = {
            'data': samples,
            'comp_data':{
                'sampling':{
                    'time': {'start': start_time, 'end': end_time},
                    'memory': {'peak': peak_memory_usage}
                }
            }
        }

        return sampling_data, hists


    @torch.no_grad()
    def perform_assignment(self, assignment: Assignment=None, conditioning_elems=None, sampling_kwargs=None, other_metrics=None, return_samples=False) -> Dict:

        if assignment is None:
            return {}, None

        ######## compute the sampling metrics ########
        num_samples = assignment.how_many_to_generate

        sampling_data, hists = self.compute_sampling_data(
            num_samples = num_samples,
            conditioning_elems = conditioning_elems,
            sampling_kwargs = sampling_kwargs
        )

        ##############################################

        # log computational metrics
        overtime = 0 if self.start_time is None else time.time() - self.start_time
        
        sampling_data['comp_data']['train'] = {
            'total_time': self.total_elapsed_time + overtime,
            'memory':  max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)
        }

        other_metrics = other_metrics if other_metrics is not None else {}
        
        # perform assignment
        assignment_results = assignment(
            **other_metrics,
            **sampling_data
        )

        if return_samples:
            return assignment_results, hists, sampling_data['data']
        else:
            return assignment_results, hists


    def log_sampled_graphs(self, samples, how_many=10, method='networkx'):
        if how_many > 0:

            how_many = min(how_many, len(samples))

            if method == 'networkx':

                from torch_geometric.utils import to_networkx

                # transform first log_chain graphs of output_batch to networkx
                graphs_to_log = [to_networkx(samples[i], to_undirected=True) for i in range(how_many)]
                imgs = [graph_to_image(g) for g in graphs_to_log]

            else:
                raise ValueError(f'Unknown method {method}')
            
            # transform to wandb images
            images = [wandb.Image(img) for img in imgs]
            # log them
            self.log_wandb_objects({
                'generation/graphs': images
            })





import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import networkx as nx

def graph_to_image(graph: nx.Graph) -> Image:
    plt.figure(figsize=(5, 5))
    nx.draw(graph, with_labels=True)
    plt.axis('off')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)