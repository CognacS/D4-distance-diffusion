from typing import Union, Dict, Optional, Any, Callable

import os.path as osp
from omegaconf import DictConfig

import pytorch_lightning as pl

from .dataloader import DataLoader

from src.data.datasets.core import DataResources

class GraphDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_resources: DataResources,
            transform: Optional[Callable|Dict[str, Callable]]=None,
            dataloader_config: Optional[Dict[str, Dict]]=None
        ):

        super().__init__()

        self.data_resources = data_resources
        self.transform = transform
        self.datasets = {}

        if dataloader_config is None:
            self.dataloader_config = {}
        else:
            self.dataloader_config = dataloader_config


    ############################################################################
    #                         DATA PREPARATION METHODS                         #
    ############################################################################

    def prepare_data(self):
        self.data_resources.prepare_data()
        

    def setup(self, stage: str):

        ###########################  TRAINING PHASE  ###########################
        if stage == 'fit' or stage == 'train':

            self.datasets['train'] = self.get_dataset('train')
            self.datasets['valid'] = self.get_dataset('valid')

        ##########################  VALIDATION PHASE  ##########################
        elif stage == 'validate' or stage == 'valid':

            self.datasets['valid'] = self.get_dataset('valid')

        #############################  TEST PHASE  #############################
        elif stage == 'test':

            self.datasets['test'] = self.get_dataset('test')

        else:
            raise NotImplementedError(f'Stage "{stage}" is not implemented!')


    def clear_datasets(self):
        self.datasets = {}

    
    ############################################################################
    #                              DATASET METHODS                             #
    ############################################################################

    def get_dataset(self, which_datasplit: str):
        if isinstance(self.transform, dict):
            transform = self.transform[which_datasplit]
        else:
            transform = self.transform
        return self.data_resources.get('dataset', split=which_datasplit, transform=transform)


    ############################################################################
    #                            DATALOADER METHODS                            #
    ############################################################################

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('valid')

    def test_dataloader(self):
        return self.get_dataloader('test')
    
    def get_dataloader(self, which_datasplit: str):

        dataset = self.datasets[which_datasplit]
        
        split_loader_config = get_config_datasplit(self.dataloader_config, which_datasplit)
        
        return DataLoader(
            dataset = dataset,
            **split_loader_config
        )

        


def is_dict_of_dicts(d: Dict):
    return isinstance(next(iter(d.values())), (dict, DictConfig))

def get_config_datasplit(config: Union[Dict, None], which_datasplit: str):
    if isinstance(config, (dict, DictConfig)):
        if is_dict_of_dicts(config):
            # case 1: config has different configs for each datasplit
            config = config[which_datasplit]
        else:
            # case 2: config is itself the config for all datasplits
            pass
    else:
        # case 3: there is no config, use default
        config = {}

    return config