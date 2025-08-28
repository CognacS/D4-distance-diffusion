from typing import Dict

import os
import os.path as osp
import json

from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from datetime import datetime



def get_checkpoint_filepath(checkpoint_dir: str, model_name: str, version: int):
    ver_subdir = 'version_' + str(version)

    dirpath = osp.join(checkpoint_dir, model_name, ver_subdir, 'checkpoints')

    if osp.exists(dirpath):
        filename = os.listdir(dirpath)[0]
        return osp.join(dirpath, filename)
    
    else:
        return None



def setup_checkpointing(checkpoint_base_dir: str, config_name: str, run_id: str):

    checkpoint_dir = osp.join(checkpoint_base_dir, config_name, run_id)

    checkpoint_callback = ModelCheckpoint(
        dirpath =       checkpoint_dir,
        filename =      None,
        every_n_epochs= 1,
    )

    checkpoint_filepath = osp.join(checkpoint_dir, checkpoint_name + '.ckpt')
    resume = osp.exists(checkpoint_filepath)

    return checkpoint_callback, checkpoint_filepath, resume


def setup_checkpointing(checkpoint_base_dir: str, config_name: str, run_id: str=None, other_run_info: Dict=None):

    # if run_id is none, set it as the current timestamp
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_dir = osp.join(checkpoint_base_dir, config_name, run_id)
    
    # write other_run_info to a json file inside the checkpoint dir
    # if the directory doesn't exist, create it
    if other_run_info is not None and isinstance(other_run_info, dict):
        other_run_info_filepath = osp.join(checkpoint_dir, 'other_run_info.json')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with open(other_run_info_filepath, 'w') as f:
            json.dump(other_run_info, f)


    checkpoint_callback = ModelCheckpoint(
        dirpath =       checkpoint_dir,
        filename =      None,
        every_n_epochs= 1,
    )

    return checkpoint_callback


def setup_wandb(project_name: str, run_identifier: str, general_config: Dict, resume: bool = False):

    wandb.init(
        project =   project_name,
        config =    general_config,
        name =      run_identifier,
        resume =    resume,
        **general_config['run']['logging']['wandb']
    )