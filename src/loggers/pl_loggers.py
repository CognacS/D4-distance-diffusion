from typing import Dict
import src.loggers as log

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_log

@log.reg_loggers.register('WandbLogger')
def wandb_logger(
        save_dir: str,
        run_name: str,
        run_config: Dict,
        id: str,
        version: int|str,
        resume: bool,
        model: pl.LightningModule,
        **kwargs
    ):

    watch = kwargs.pop('watch', None)

    logger = pl_log.WandbLogger(
        save_dir = save_dir,
        name = run_name,
        id = id,
        version = version,
        config = run_config,
        resume = resume,
        **kwargs
    )

    if watch:
        if isinstance(watch, dict):
            logger.watch(model, **watch)
        else:
            logger.watch(model)

    return logger


@log.reg_loggers.register('TensorBoardLogger')
def tensorboard_logger(
        save_dir: str,
        run_name: str,
        run_config: Dict,
        id: str,
        version: int|str,
        resume: bool,
        model: pl.LightningModule,
        **kwargs
    ):
    return pl_log.TensorBoardLogger(
        save_dir = save_dir,
        name = run_name,
        version = version,
        default_hp_metric = False,
        **kwargs
    )