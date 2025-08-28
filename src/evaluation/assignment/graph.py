from typing import Dict, List, Optional

from src.data.datasets.core import DataResources
from src.data.simple_transforms.graph import graph2nx, nx2nxlargest


from src.evaluation import reg_assignment
from src.evaluation.assignment.core import Assignment, ClonableWithSplitsMixin
from src.evaluation.utils.synth import list_to_nx

import src.evaluation.metrics as m_list


from src.datatypes.sparse import SparseGraph

import src.evaluation.metrics as m_list
import src.evaluation.metrics.sampling as sm
import src.evaluation.metrics.computational as cm


@reg_assignment.register('graph')
class GenericGraphAssignment(Assignment, ClonableWithSplitsMixin):

    PROFILES = ['cdgs', 'digress']

    def __init__(
            self,
            how_many_to_generate: int,
            split: str,
            data_resources: DataResources,
            no_computational_metrics: bool = True,
            profile: str = None,
            extended_metrics: bool = False,
            type_of_graph: str = None,
            enabled_metrics: str='all',
            metrics_overrides: Dict[str, Dict]=None,
            **kwargs
        ):

        super().__init__(how_many_to_generate, enabled_metrics, metrics_overrides, **kwargs)

        self.no_computational_metrics = no_computational_metrics
        self.data_resources = data_resources

        assert profile is None or profile in self.PROFILES, f'Profile {profile} not supported. Supported profiles: {self.PROFILES}, or None for all'

        # insert metrics depending on the profile
        self.profile = profile

        # load data for sampling metrics
        eval_graphs = list_to_nx(data_resources.get('dataset', split))

        self.add_metric(m_list.KEY_GRAPH_CONN_COMP, sm.GraphConnCompMetric)

        if self.profile is None or 'cdgs' in self.profile:
            self.add_metric(m_list.KEY_GRAPH_CDGS, sm.GraphCDGSMetric, eval_graphs)
            self.add_metric(m_list.KEY_GRAPH_GIN, sm.GraphGinMetric, eval_graphs)

        if self.profile is None or 'digress' in self.profile:
            train_graphs = list_to_nx(data_resources.get('dataset', 'train'))

            self.add_metric(m_list.KEY_GRAPH_DEGREE, sm.DegreeMetric, eval_graphs)
            self.add_metric(m_list.KEY_GRAPH_SPECTRE, sm.SpectreMetric, eval_graphs)
            self.add_metric(m_list.KEY_GRAPH_CLUSTERING, sm.ClusteringMetric, eval_graphs)
            self.add_metric(m_list.KEY_GRAPH_ORBIT, sm.OrbitMetric, eval_graphs)
            if type_of_graph is not None:
                self.add_metric(m_list.KEY_GRAPH_VUN, sm.VUNGraphMetric, train_graphs, type_of_graph)

            if extended_metrics:
                self.add_metric(m_list.KEY_GRAPH_NODES, sm.NodesMetric, train_graphs)
                self.add_metric(m_list.KEY_GRAPH_ECCENTRICITY, sm.EccentricityMetric, train_graphs)


        if not self.no_computational_metrics:
            self.add_metric(m_list.KEY_SAMPLING_TIME, cm.SamplingTimeMetric)
            self.add_metric(m_list.KEY_SAMPLING_MEMORY, cm.SamplingMemoryMetric)

        self.add_params_to_clone(['data_resources', 'no_computational_metrics', 'profile'])

        

    def __call__(self, data: List[SparseGraph], comp_data: Optional[Dict]=None, **kwargs):

        if self.has_metric(m_list.KEY_SAMPLING_TIME) or self.has_metric(m_list.KEY_SAMPLING_MEMORY):
            assert comp_data is not None, 'Computational data is required for computational metrics'

        gathered_metrics = []

        # Convert graphs to nx
        nx_graphs = graph2nx(data, to_undirected=True, remove_self_loops=True)

        # get connected components
        gathered_metrics.append(self.compute_if_exists(m_list.KEY_GRAPH_CONN_COMP, nx_graphs))

        # Select biggest connected component
        nx_graphs_biggest = nx2nxlargest(nx_graphs)

        gathered_metrics.extend([
            # Compute sampling metrics for cdgs
            self.compute_if_exists(m_list.KEY_GRAPH_CDGS, nx_graphs_biggest),
            self.compute_if_exists(m_list.KEY_GRAPH_GIN, nx_graphs_biggest),

            # Compute computational metrics for digress
            self.compute_if_exists(m_list.KEY_GRAPH_DEGREE, nx_graphs_biggest),
            self.compute_if_exists(m_list.KEY_GRAPH_SPECTRE, nx_graphs_biggest),
            self.compute_if_exists(m_list.KEY_GRAPH_CLUSTERING, nx_graphs_biggest),
            self.compute_if_exists(m_list.KEY_GRAPH_ORBIT, nx_graphs_biggest),
            self.compute_if_exists(m_list.KEY_GRAPH_VUN, nx_graphs_biggest),
            # compute extended metrics
            self.compute_if_exists(m_list.KEY_GRAPH_NODES, nx_graphs_biggest),
            self.compute_if_exists(m_list.KEY_GRAPH_ECCENTRICITY, nx_graphs_biggest),

            # compute sampling time and memory
            self.compute_if_exists(m_list.KEY_SAMPLING_TIME, comp_data),
            self.compute_if_exists(m_list.KEY_SAMPLING_MEMORY, comp_data)
        ])


        return {k: v for d in gathered_metrics for k, v in d.items()}