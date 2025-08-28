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

from src.configurator import RunContext

# import monkey patches for fixing bugs in torch and torch_geometric for MPS devices
import platform
if 'darwin' in platform.system().lower(): # if running on MacOS, apply monkey patches
    import src.monkey_patches

@hydra.main(version_base=None, config_path='config', config_name='default')
def main(cfg: DictConfig):

    # make hydra config available in the current config
    # will be removed later
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.hydra = HydraConfig.get()


    # prepare context with data, model, trainer, etc.
    context: RunContext = RunContext.from_config(cfg)
    
    # execute context based on input arguments
    # may return a dictionary of results, containing the metrics values
    results = context.execute()

    #if results is not None:
        # log results to the logger
    #    context.log_dict_as_table(results)

    # close context (e.g. close wandb connection, garbage collection, etc.)
    context.close()

    # return results that can be used for hparams selection when using some
    # sweeper with hydra
    return results


if __name__ == '__main__':
    main()