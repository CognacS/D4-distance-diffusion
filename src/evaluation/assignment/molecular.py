from typing import Dict, List, Optional

import json

from src.data.datasets.core import DataResources
from src.data.simple_transforms.molecular import mol2nx, smiles2mol, mol2smiles

from src.data.simple_transforms.molecular import GraphToMoleculeConverter


from src.evaluation import reg_assignment
from src.evaluation.assignment.core import Assignment, ClonableWithSplitsMixin

from src.datatypes.sparse import SparseGraph

import src.evaluation.metrics as m_list
import src.evaluation.metrics.sampling as sm
import src.evaluation.metrics.computational as cm

@reg_assignment.register('molecular')
class MolecularAssignment(Assignment, ClonableWithSplitsMixin):

    def __init__(
            self,
            how_many_to_generate: int,
            split: str,
            data_resources: DataResources,
            relaxed: bool = True,
            no_computational_metrics: bool = True,
            distribution_metrics: bool = True,
            metrics_3d: bool = False,
            enabled_metrics: str='all',
            metrics_overrides: Dict[str, Dict]=None,
            **kwargs
        ):

        super().__init__(how_many_to_generate, enabled_metrics, metrics_overrides, **kwargs)

        self.no_computational_metrics = no_computational_metrics
        self.data_resources = data_resources
        self.relaxed = relaxed
        self.metrics_3d = metrics_3d
        self.distribution_metrics = distribution_metrics

        # load data for novelty, fcd and nspdk
        train_smiles = data_resources.get('smiles', 'train')
        if distribution_metrics:
            eval_smiles = data_resources.get('smiles', split)
            eval_nx = mol2nx(smiles2mol(eval_smiles))
        if isinstance(train_smiles[0], tuple):
            train_smiles = [t[0] for t in train_smiles]
            eval_smiles = [t[0] for t in eval_smiles]

        # add metrics of this assignment
        self.add_metric(m_list.KEY_MOLECULAR_VALIDITY, sm.ValidMoleculeMetric)
        self.add_metric(m_list.KEY_MOLECULAR_UNIQUENESS, sm.UniqueMoleculeMetric)
        self.add_metric(m_list.KEY_MOLECULAR_NOVELTY, sm.NovelMoleculeMetric, train_smiles)
        self.add_metric(m_list.KEY_EDGE_TYPES_DISTRIBUTION, sm.EdgeTypeDistributionMetric)
        if distribution_metrics:
            self.add_metric(m_list.KEY_FCD, sm.FCDMetric, eval_smiles)
            self.add_metric(m_list.KEY_NSPDK, sm.NSPDKMetric, eval_nx)
        if not self.no_computational_metrics:
            self.add_metric(m_list.KEY_SAMPLING_TIME, cm.SamplingTimeMetric)
            self.add_metric(m_list.KEY_SAMPLING_MEMORY, cm.SamplingMemoryMetric)
        if metrics_3d:
            eval_mols = data_resources.get('dataset', split)
            self.add_metric(m_list.KEY_BOND_DISTANCE, sm.BondDistanceDistributionMetric, eval_mols)
            self.add_metric(m_list.KEY_PREBOND_DISTANCE, sm.bond_distance_distribution, eval_mols)
            if parametrics_3d

        self.graph_to_mol_converter: GraphToMoleculeConverter = data_resources.get('decoder')

        self.add_params_to_clone([
            'data_resources', 'relaxed', 'no_computational_metrics',
            'metrics_3d', 'distribution_metrics'
        ])



    def __call__(self, data: List[SparseGraph], comp_data: Optional[Dict]=None, **kwargs):

        if self.has_metric(m_list.KEY_SAMPLING_TIME) or self.has_metric(m_list.KEY_SAMPLING_MEMORY):
            assert comp_data is not None, 'Computational data is required for computational metrics'

        gathered_metrics = []
        
        ##################  COMPUTE SPATIAL METRICS (IF ANY)  ##################
        
        gathered_metrics.extend([
            self.compute_if_exists(m_list.KEY_BOND_DISTANCE, data),
            self.compute_if_exists(m_list.KEY_PREBOND_DISTANCE, data),
            self.compute_if_exists(m_list.KEY_EDGE_TYPES_DISTRIBUTION, data)
        ])

        ###############  COMPUTE VALIDITY, UNIQUENESS, NOVELTY  ################

        mols = self.graph_to_mol_converter(data, override_relaxed=self.relaxed)

        # compute validity
        ret = self.compute_if_exists(m_list.KEY_MOLECULAR_VALIDITY, mols)
        valid_smiles = ret.pop('valid_smiles')
        gathered_metrics.append(ret)
        
        # compute uniqueness
        ret = self.compute_if_exists(m_list.KEY_MOLECULAR_UNIQUENESS, valid_smiles)
        unique_smiles = ret.pop('unique_smiles')
        gathered_metrics.append(ret)

        # compute novelty
        ret = self.compute_if_exists(m_list.KEY_MOLECULAR_NOVELTY, unique_smiles)
        novel_smiles = ret.pop('novel_smiles')
        gathered_metrics.append(ret)


        #########################  COMPUTE FCD, NSPDK  #########################
        if self.distribution_metrics:
            fixed_mols = self.graph_to_mol_converter(
                data,
                override_relaxed=self.relaxed,
                override_post_hoc_mols_fix=True
            )

            # compute FCD from fixed molecules
            fixed_smiles = mol2smiles(fixed_mols, sanitize=True)
            ret = self.compute_if_exists(m_list.KEY_FCD, fixed_smiles)
            gathered_metrics.append(ret)

            # compute NSPDK
            fixed_nx = mol2nx(fixed_mols)
            ret = self.compute_if_exists(m_list.KEY_NSPDK, fixed_nx)
            gathered_metrics.append(ret)

        # compute sampling time and memory
        gathered_metrics.extend([
            self.compute_if_exists(m_list.KEY_SAMPLING_TIME, comp_data),
            self.compute_if_exists(m_list.KEY_SAMPLING_MEMORY, comp_data)
        ])

        return {k: v for d in gathered_metrics for k, v in d.items()}