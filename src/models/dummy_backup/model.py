from typing import Dict

from logging import Logger

import torch
from src.models.generator import Generator, GeneratorWithEvaluation
from src.models.dummy.data_collection import DataCollection

from torch_geometric.data import Batch

from src.evaluation.assignment.core import Assignment
from src.models import reg_models


DEFAULT_DATA_COLLECTION_SIZE = 100

KEY_TRAIN = 'TRAIN'
KEY_VALID = 'VALID'
KEY_TEST = 'TEST'

@reg_models.register()
class DummyModel(GeneratorWithEvaluation):
    def __init__(
            self,

            validation: Dict,

            ######## passed by configurator ######
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):

        super().__init__(
            validation=validation,
            dataset_info=dataset_info,
            test_assignment=test_assignment,
            console_logger=console_logger
        )

        max_size = DEFAULT_DATA_COLLECTION_SIZE

        if hasattr(self.test_assignment, 'how_many_to_generate'):
            datasize_test = self.test_assignment.how_many_to_generate
        else:
            datasize_test = 0
        
        if hasattr(self.validation_config, 'how_many_to_generate'):
            datasize_valid = self.validation_config.how_many_to_generate
        else:
            datasize_valid = 0

        max_size = max(datasize_valid, datasize_test)

        self.data_collection = DataCollection(max_size=max_size)

        # save hyperaparameters (but those not in the Generator ignored list)
        self.save_hyperparameters(ignore=Generator.IGNORED_HPARAMS + ['received_dims'])



    def configure_optimizers(self):
        return None


    def training_step(self, batch, batch_idx):

        # unstack the batch into a list
        batch = batch.to_data_list()

        # add the batch to the data collection
        self.data_collection.add_data(batch)


    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


    def on_validation_epoch_end(self):
        self.on_evaluation_epoch_end(KEY_VALID)

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(KEY_TEST)


    def on_evaluation_epoch_end(self, which):

        if which == KEY_VALID:
            assignment = self.valid_assignment
        else:
            assignment = self.test_assignment


        # start with already computed metrics (during evaluation epochs)
        metrics = {}

        if assignment is not None:
        
            # compute sampling metrics
            assignment_results, hists = self.perform_assignment(assignment=assignment, other_metrics=metrics)

            # add the assignment results to the metrics
            metrics.update(assignment_results)

            # log histograms with wandb if available (check is inside)
            self.log_wandb_histograms(self.apply_prefix(hists, f'{which}/sampling'))

        # output the metrics for reading purposes
        self.console_logger.info(str(self.get_metrics_values(metrics)))

        # add prefix to logs
        to_log = self.apply_prefix(
            metrics = metrics,
            prefix = f'{which}'
        )

        self.log_dict(to_log)


    
    @torch.no_grad()
    def sample(self,
            num_samples=None,
            **kwargs
        ):
        data = list(self.data_collection.data)

        if num_samples == len(data) or num_samples is None:
            return data
        elif num_samples < len(data):
            return data[:num_samples]
        else:
            raise ValueError((
                f"DataCollection doesn't have enough samples. "
                f"Requested num_samples={num_samples}. "
                f"Available: {len(data)}"
            ))
        
        

        

       