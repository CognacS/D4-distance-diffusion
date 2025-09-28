# suppress warnings from torch 2.4.1, these will be fixed in future versions
import warnings
warnings.filterwarnings("ignore", "You are using `torch\.load` with `weights_only=False`.*")
warnings.filterwarnings("ignore", ".*deterministic implementation.*")
warnings.filterwarnings("ignore", "Weights only load failed\. Please file an issue to make `torch\.load\(weights_only=True\)`.*")
warnings.filterwarnings("ignore", "The `pre_transform` argument differs from the one used in the pre-processed version of this dataset\..*")
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*to-Python converter for.*")

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

import src.data.datasets as datasets
from src.data.transforms import reg_transforms
from src.data.filters import reg_filters
from src.datatypes import reg_dataset_wrapper

from copy import deepcopy
from src.configurator import seed_everything

# import monkey patches for fixing bugs in torch and torch_geometric for MPS devices
import platform
if 'darwin' in platform.system().lower(): # if running on MacOS, apply monkey patches
    import src.monkey_patches
    
from src.kdes import scan_single_data_and_compute_kdes

@hydra.main(version_base=None, config_path='config', config_name='default')
def main(cfg: DictConfig):

    # make hydra config available in the current config
    # will be removed later
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.hydra = HydraConfig.get()

    seed_everything(cfg.seed)

    cfg_dataset = cfg.task.dataset
    cfg_pretf = cfg.task.pre_transform

    ###########################  DATASET SETUP  ############################
    # get dataset name
    dataset_name = cfg_dataset.name
    dataset_params = cfg_dataset.params

    print(f'Loading dataset "{dataset_name}"')

    pre_transform = None
    if isinstance(cfg_pretf, list):
        pre_transform = [reg_transforms.get_instance_from_dict(tf) for tf in cfg_pretf]
    else:
        pre_transform = reg_transforms.get_instance_from_dict(cfg_pretf)

    
    # add filters if any
    addons = {k : None for k in ['pre_filter', 'pre_filter_raw']}
    dataset_params_mod = deepcopy(dataset_params)
    for k in addons.keys():
        if k in dataset_params_mod:
            if dataset_params_mod[k] is not None:
                cfg_prefilt = dataset_params_mod[k]
                if isinstance(cfg_pretf, list):
                    v = [reg_filters.get_instance_from_dict(tf) for tf in cfg_prefilt]
                else:
                    v = reg_filters.get_instance_from_dict(cfg_prefilt)
                addons[k] = v
                dataset_params_mod.pop(k, None)

    # get dataset resources, also include pre_transform
    data_resources = datasets.reg_dataresources.get_instance(
        dataset_name, dataset_params_mod, pre_transform=pre_transform, **addons
    )

        
    data_resources.prepare_data()
        
    print(f'Dataset "{dataset_name}" is ready.')
    print('Now computing kdes...')
    
    ds_test = data_resources.get('dataset', 'test')

    kde_kwargs = {}
    scan_single_data_and_compute_kdes(ds_test, dataset_name, kde_kwargs=kde_kwargs)


if __name__ == '__main__':
    main()