from typing import Tuple, Dict, Callable, List, Union

# import python garbage collector
import gc

# logging utilities
from python_log_indenter import IndentedLoggerAdapter
import logging

# path utilities
import os.path as osp
from pathlib import Path
import glob

# configuration utilities
from omegaconf import OmegaConf, DictConfig

# experiment tracking utilities
import wandb

# data storing utilities
import json
import pickle as pkl

# miscellanous utilities
from datetime import datetime
from copy import deepcopy

# torch utilities
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# pytorch lightning utilities
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler


# datamodule utilities
from src.data.datasets.core import transforms_to_pipeline
from src.data.datamodule import GraphDataModule
from src.evaluation.assignment.core import Assignment


# main modules, from which to get registered instances
import src.data.datasets as datasets
from src.data.transforms import reg_transforms
from src.data.filters import reg_filters
from src.datatypes import reg_dataset_wrapper
import src.models as models
from src.models.generator import Generator
import src.evaluation as evaluation
import src.callbacks as clb
import src.loggers as log

# for RNG seeding purposes
import os
import random
import numpy as np

DATASETS_ROOT = 'datasets'
CHECKPOINT_PATH = 'checkpoints'
ALLOWED_EXECUTION_MODES = ['train', 'gen', 'eval', 'train+eval', 'validate_hparams', 'validate']

# from https://github.com/Lightning-AI/lightning/issues/1565
def seed_everything(seed=191117):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision('medium')


class ContextException(Exception):
    """Exception raised for errors in the context.
    """
    pass


class RunContext:

    def __init__(self):
        pass

    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> 'RunContext':
        """Factory method for creating a run context from a configuration.
        (see the configure(cfg) method for more details)

        Parameters
        ----------
        cfg : DictConfig
            full configuration of the run, usually coming from hydra

        Returns
        -------
        RunContext
            run context created from the configuration
        """

        context = cls()
        context.configure(cfg)
        return context


    def configure(self, cfg: DictConfig):
        """Method for configuring the run context. This will create the experiment
        directory, the datamodule, the model, the trainer, etc.

        Parameters
        ----------
        cfg : DictConfig
            full configuration of the run, usually coming from hydra
        """

        # preprocess configuration
        cfg = preprocess_config(cfg)

        # initialize logger
        self.logger = self.initialize_logger('context', cfg.verbosity)

        # store execution arguments
        self.logger.info(f'Reading execution arguments...').push().add()
        self._configure_execution_args(cfg)
        self.logger.pop().info(f'Execution arguments read with success')

        # set seed
        self.logger.info(f'Setting seed to {self.seed}...').push().add()
        seed_everything(self.seed)
        self.logger.pop().info(f'Seed set with success')

        # check that context parameters are valid
        self.logger.info(f'Running checks on the context...').push().add()
        self.validate_context()
        self.logger.pop().info(f'Context validated with success')

        # checkpoints, run ids, etc..., are all contained in the run directory
        self.logger.info(f'Setting up run directory, version, id...').push().add()
        self.run_directory, self.version_dir, self.version, self.run_id = self._setup_run_directory(self.config_name)
        self.logger.pop().info(f'Run directory: {self.run_directory}')
        self.logger.info(f'Run version: {self.version}')
        self.logger.info(f'Run id: {self.run_id}')

        # configuring platform. Distributed training only in training mode
        self.distributed_training, self.accelerator, self.devices = get_platform(self.mode, cfg.platform)

        # configuring datamodule
        self.logger.info(f'Configuring and loading datamodule...').push().add()
        self.datamodule, self.data_resources = self._configure_datamodule(
            cfg_dataset =       cfg.task.dataset,
            cfg_dataloader =    cfg.method.dataloader,
            cfg_pretf =         cfg.task.pre_transform,
            cfg_dswrapper =     cfg.method.dataset_wrapper if hasattr(cfg.method, 'dataset_wrapper') else None
        )
        self.dataset_info = self.data_resources.get('info', 'train')
        self.logger.pop().info(f'Datamodule configured with success')

        self.logger.info(f'Configuring test assignment...').push().add()
        self.test_assignment = self._configure_test_assignment(
            cfg_test =          cfg.task.test,
            data_resources =    self.data_resources
        )
        self.logger.pop().info(f'Test assignment configured with success')
        
        # configuring model
        self.logger.info(f'Configuring and loading model...').push().add()
        self.model = self._configure_model(
            cfg_model =             cfg.method.model,
            dataset_info =          self.dataset_info,
            test_assignment =       self.test_assignment,
        )
        self.logger.pop().info(f'Model configured with success')

        self.logger.info(f'Configuring transforms...').push().add()
        self._configure_transforms(
            cfg_datatf =        cfg.method.datatransform,
            model =             self.model,
            datamodule =        self.datamodule
        )
        self.logger.pop().info(f'Transforms configured with success')

        # configuring trainer
        self.logger.info(f'Configuring trainer...').push().add()
        self.trainer = self._configure_trainer(
            cfg_trainer =           cfg.method.trainer,
            cfg_callbacks =         cfg.method.callbacks,
            cfg_logger =            cfg.logger,
        )
        self.logger.pop().info(f'Trainer configured with success')

        if self.enable_ckp and not self.need_load_ckp():
            self.logger.info(f'Creating run directory...')
            self.run_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Configuration completed')


    def need_load_ckp(self):
        return not (isinstance(self.load_ckp, bool) and not self.load_ckp)


    def execute(self):
        """Method for executing the run context. Depending on the execution mode,
        this will train the model, generate graphs, evaluate a model, etc.

        Returns
        -------
        results : Dict
            dictionary containing key-value pairs with keys being a metric name
            and values being the corresponding metric value. Will be returned only
            if the execution mode is 'validate_hparams' or 'eval' or 'train+eval'
        """

        results = {}

        # TRAINING: called for training a model
        # no evaluation is performed
        if self.mode == 'train':
            self.fit()

        # GENERATION: called for generating graphs
        # requires a pre-trained model
        # will produce a file with the generated graphs
        # by default, the file is stored in the run directory
        elif self.mode == 'gen':
            if not self.need_load_ckp():
                raise ContextException(f'To evaluate, must resume a checkpoint. Do it by setting load_ckp')
            ####### generate graphs #######
            kwargs = {}
            if 'how_many' in self.cfg:
                kwargs['how_many'] = self.cfg['how_many']
            graphs = self.generate(**kwargs)
            ######## store graphs ########
            kwargs = {}
            if 'gen_path' in self.cfg:
                kwargs['gen_path'] = self.cfg['gen_path']
            self.store_graphs(graphs, **kwargs)

        # VALIDATE HPARAMS: called during hyperparameter optimization
        # trains the model and returns the validation loss to be used by the optimizer
        elif self.mode == 'validate_hparams':
            self.fit()
            results = self.evaluate_best(validation=True)

        elif self.mode == 'validate':
            if not self.need_load_ckp():
                raise ContextException(f'To validate, must resume a checkpoint. Do it by setting load_ckp')
            results = self.evaluate_best(validation=True)
            
        # EVALUATION: called for evaluating a model on the test set
        # requires a pre-trained model
        elif self.mode == 'eval':
            if not self.need_load_ckp():
                raise ContextException(f'To evaluate, must resume a checkpoint. Do it by setting load_ckp')
            results = self.evaluate_best(validation=False)

        # TRAIN+EVALUATION: called for evaluating a configuration on the test set
        # trains the model and returns the test loss
        # this is intended to be executed for the final evaluation on many seeds
        elif self.mode == 'train+eval':
            self.fit()
            results = self.evaluate_best(validation=False)

        # UNKNOWN MODE: raise an exception
        else:
            raise ContextException(f'Unknown mode {self.mode}')
        
        return results
    

    def __call__(self):
        return self.execute()


    ############################################################################
    #                            EXECUTION BRANCHES                            #
    ############################################################################
    def fit(self, ckpt='last'):
        return self.trainer.fit(
            model = 		self.model,
            datamodule = 	self.datamodule,
            ckpt_path =		ckpt
        )
    def validate(self, ckpt='last'):
        return self.trainer.validate(
            model = 		self.model,
            datamodule = 	self.datamodule,
            ckpt_path =		ckpt
        )
    def test(self, ckpt='last'):
        return self.trainer.test(
            model = 		self.model,
            datamodule = 	self.datamodule,
            ckpt_path =		ckpt
        )
    def generate(self, ckpt='last', how_many=128):
        self.load_checkpoint(ckpt)
        self.model.to(self.trainer.device)
        graphs = self.model.sample(how_many)
        decoder = self.data_resources.get('decoder')
        return decoder(graphs)



    ############################################################################
    #                               INFO METHODS                               #
    ############################################################################

    def get_training_info(self):
        return self.trainer.current_epoch
    
    def get_dataset_info(self):
        return self.dataset_info
    
    def get_configuration(self, path: str=None):
        if path is None:
            return self.cfg
        else:
            return OmegaConf.select(self.cfg, path, throw_on_missing=True)

    ############################################################################
    #                     STORING AND CHECKPOINTING METHODS                    #
    ############################################################################

    def store_graphs(self, graphs: List, path: str=None):
        if path is None:
            # store graphs with date and time in name
            filename = f'generated_graphs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            path = self.run_directory / filename
        # store graphs using pickle
        with open(path, 'wb') as f:
            pkl.dump(graphs, f)


    def load_checkpoint(self, checkpoint_name: str=None, strict: bool=True):

        if not self.enable_ckp:
            self.logger.warning(f'Checkpointing is disabled, skipping...')
            return
        
        if checkpoint_name is None:
            checkpoint_name = 'last'
        if not checkpoint_name.endswith('.ckpt'):
            checkpoint_name += '.ckpt'

        self.logger.info(f'Loading checkpoint {checkpoint_name}...')

        self.model = self._load_checkpoint(str(self.run_directory / checkpoint_name), strict)

        self.logger.info(f'Succesfully loaded checkpoint {checkpoint_name}')



    def load_module_from(self, ckpt_path: str, module_name: str):

        other_model = self._load_checkpoint(ckpt_path, strict=True)

        self.model[module_name] = other_model[module_name]



    def _load_checkpoint(self, ckpt_path: str, strict: bool=True):

        
        model = self.model.__class__.load_from_checkpoint(
            # where to find the checkpoint
            checkpoint_path =               ckpt_path,
            strict =                        strict,

            # these are given to the model, as these are not hyperparameters
            console_logger =                self.model.console_logger,
            dataset_info =                  self.dataset_info,
            test_assignment =               self.test_assignment
        )

        return model


    ############################################################################
    #                            EVALUATION METHODS                            #
    ############################################################################
                
    def evaluate_ckpt(self, ckpt=None, validation=True):

        if ckpt is None:
            ckpt = 'best'

        self.logger.info(f'Current checkpoint: {ckpt}')

        # test the model using best checkpoint
        if validation:
            curr_metrics = self.validate(ckpt=ckpt)[0]
        else:
            curr_metrics = self.test(ckpt=ckpt)[0]

        curr_metrics['run'] = self.group_name
        curr_metrics['seed'] = self.seed
        curr_metrics['ckpt'] = ckpt

        return curr_metrics
    

    def evaluate_best(self, validation=True):
        
        try:
            # evaluate the model using best checkpoint
            curr_metrics = self.evaluate_ckpt('best', validation=validation)

        except ValueError as e:

            if 'best' in str(e):
                self.logger.warning(f'No best checkpoint found, evaluating last checkpoint...')

                # evaluate the model using last checkpoint
                curr_metrics = self.evaluate_ckpt('last', validation=validation)

        return curr_metrics
    

    def evaluate_all_checkpoints(self, validation=True):

        dictionaries = []

        for ckpt in self.get_all_checkpoints(include_last=False):
                
            curr_metrics = self.evaluate_ckpt(ckpt, validation)
            dictionaries.append(curr_metrics)

        return dictionaries
    

    def log_dict_as_table(
            self,
            dictionary: Dict|List[Dict],
            name: str=None,
            save_table: bool=True,
            save_artifact: bool=True,
            save_file: bool=True
        ):
        
        if name is None:
            name = 'test_table'
        if save_file:
            with open(self.run_directory / f'{name}.json', 'w') as f:
                json.dump(dictionary, f)
        if save_table:
            curr_dict = dictionary if isinstance(dictionary, list) else [dictionary]
            columns = list(dictionary[0].keys())
            # each row has a run, each columns is a metric of the dict
            transposed_table = [[d[k] for k in columns] for d in curr_dict]
            table = wandb.Table(columns=columns, data=transposed_table)
            wandb.log({f"test/{name}": table})
        if save_artifact and save_table:
            table_art = wandb.Artifact(f'{name}_{self.run_id}', type='table')
            table_art.add(table, name)
            wandb.log_artifact(table_art)


    def get_all_checkpoints(self, include_path=True, include_last=False) -> List[Union[str, Path]]:
        checkpoints = list(self.run_directory.glob('*.ckpt'))

        if not include_last:
            checkpoints = [c for c in checkpoints if not c.name.startswith('last')]

        if not include_path:
            checkpoints = [c.name for c in checkpoints]

        return checkpoints


    def sample_batch(self, which_split: str='train'):
        self.datamodule.setup(which_split)
        batch = next(iter(self.datamodule.get_dataloader(which_split)))
        return batch
    

    def dry_run(self, which_split: str='train', num_steps: int=1, no_grad: bool=True):
        # save all relevant states
        grad_state = torch.is_grad_enabled()

        trn_curr_steps = self.trainer.limit_train_batches
        val_curr_steps = self.trainer.limit_val_batches
        tst_curr_steps = self.trainer.limit_test_batches
        curr_step = self.trainer.global_step
        curr_batch_step = self.trainer.fit_loop.epoch_loop.batch_progress.current.ready
        epoch_progress = deepcopy(self.trainer.fit_loop.epoch_progress.current)
        

        disable_generation = self.model._disable_generation
        debug_state = self.model.enable_logging
        
        try:    # run the desired split in safety

            if no_grad:
                torch.set_grad_enabled(False)
            self.model._disable_generation = True
            self.model.enable_logging = True

            if which_split == 'train':
                # temporarily update the trainer
                self.trainer.limit_train_batches = num_steps
                self.trainer.limit_val_batches = 0
                self.trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = 0
                self.trainer.fit_loop.epoch_loop.batch_progress.current.ready = 0
                self.trainer.fit_loop.epoch_progress.reset()

                # reset trainer to restart training
                # recall you cannot set global_step
                self.trainer.fit_loop.epoch_loop.batch_progress.current.reset()

                self.fit()

            elif which_split == 'valid':
                # temporarily update the trainer
                self.trainer.limit_val_batches = num_steps

                self.validate()

            elif which_split == 'test':
                # temporarily update the trainer
                self.trainer.limit_test_batches = num_steps

                self.test()
            
            elif which_split == 'gen':
                # temporarily update the trainer
                self.model._disable_generation = False

                self.model.sample_batch(64 * num_steps)

        finally:    # restore torch, trainer and model states

            if no_grad:
                torch.set_grad_enabled(grad_state)

            self.trainer.limit_train_batches = trn_curr_steps
            self.trainer.limit_val_batches = val_curr_steps
            self.trainer.limit_test_batches = tst_curr_steps
            self.trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = curr_step
            self.trainer.fit_loop.epoch_loop.batch_progress.current.ready = curr_batch_step
            self.trainer.fit_loop.epoch_progress.current = epoch_progress

            self.model._disable_generation = disable_generation
            self.model.enable_logging = debug_state


    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()

    
    def close(self):

        # turn off logger
        self.turn_off_logger(self.logger)
        self.turn_off_logger(self.model.console_logger)

    ############################################################################
    #                             UTILITY METHODS                              #
    ############################################################################

    def validate_context(self):
        pass

    def _setup_run_directory(self, config_name: str) -> Tuple[Path, int, str]:

        self.resume = False

        if self.enable_ckp:
            
            run_path = Path(CHECKPOINT_PATH, config_name)

            ####################  IF PATH HAS TO BE LOADED  ####################
            if self.load_ckp is not None:

                self.resume = True

                #####################  IF PATH IS A STRING  ####################
                if isinstance(self.load_ckp, str):
                    # load checkpoint from path
                    # path already exists and is already well-formed
                    run_path = Path(self.load_ckp)
                    if not run_path.exists():
                        raise ContextException(f'No run found matching the path {run_path}')
                    version = run_path.name.split('_')[0][1:]
                    run_id = run_path.name.split('_')[-1]

                #####################  IF PATH IS AN INT  ######################
                elif isinstance(self.load_ckp, int):
                    # load checkpoint from version
                    # path is taken from the configuration + version
                    version = self.load_ckp
                    # if the version is not specified, get the latest version
                    if version == -1:
                        # get latest version
                        matched_run_path = list(run_path.glob('v*'))
                        if len(matched_run_path) == 0:
                            raise ContextException(f'No runs found matching the name {config_name}')
                        version = max([int(p.name.split('_')[0][1:]) for p in matched_run_path])
                
                    # check that the run path exists (checking the prefix)
                    matched_run_path = list(run_path.glob(f'v{version}_*'))
                    if len(matched_run_path) == 0:
                        raise ContextException(f'No run found matching the version {version}')
                    if len(matched_run_path) > 1:
                        raise ContextException(f'Multiple runs found matching the version {version}')

                    # now we know the run exists, so we can get the run id
                    run_path = matched_run_path[0]
                    run_id = matched_run_path[0].name.split('_')[-1]

            ###################  IF PATH HAS TO BE CREATED  ####################
            else:
                # generate run id
                run_id = wandb.util.generate_id()
                version = 0
                
                # check that the run path exists, and if not, create it
                run_path.mkdir(parents=True, exist_ok=True)

                # list all directories in the run path and check the latest version
                matched_run_path = list(run_path.iterdir())
                if len(matched_run_path) > 0:
                    version = max([int(p.name.split('_')[0][1:]) for p in matched_run_path]) + 1
                
                # create the new run path
                run_path = Path(run_path, f'v{version}_{run_id}')

            version_dir = f'v{version}_{run_id}'

            return run_path, version_dir, version, run_id
        
        else:
            return None, None, 0, wandb.util.generate_id()


    ############################################################################
    #                          CONFIGURATION METHODS                           #
    ############################################################################
        

    def _configure_execution_args(self, cfg: DictConfig):

        # store naming, useful for logging into wandb and for saving checkpoints
        self.config_name = cfg['config_name']
        self.logger.info(f'Configuring run "{self.config_name}"')

        # store grouping, useful for grouping runs in wandb
        if 'group_name' in cfg:
            self.group_name = cfg['group_name']
        else:
            self.group_name = self.config_name
        self.logger.info(f'Group name: {self.group_name}')

        # store execution mode: decides what to do in this execution
        # e.g. train, generate, evaluate, etc.
        self.mode = cfg['mode']
        if self.mode not in ALLOWED_EXECUTION_MODES:
            raise ContextException(
                f'Unknown execution mode {self.mode}, '
                f'must be one of {ALLOWED_EXECUTION_MODES}'
            )
        self.logger.info(f'Run mode: {self.mode}')
        
        # store enable flags
        self.debug = cfg['debug']
        self.enable_ckp = cfg['enable_ckp'] and not self.debug
        self.enable_log = cfg['enable_log'] and not self.debug
        self.profile = cfg['profile']
        self.compile = cfg['compile']

        add_msg = '. Logging to wandb and checkpointing are disabled.' if self.debug else ''
        self.logger.info(f'Debug mode: {self.debug}{add_msg}')
        self.logger.info(f'Checkpoints enabled: {self.enable_ckp}')
        self.logger.info(f'Wandb logging enabled: {self.enable_log}')
        self.logger.info(f'Profiling: {self.profile}')

        # store whether to load a checkpoint
        self.load_ckp = cfg['load_ckp']
        self.logger.info(f'Resuming run: {self.load_ckp}')

        # store seed to use
        self.seed = cfg['seed']
        self.logger.info(f'Seed: {self.seed}')

        # store entire configuration
        self.cfg = cfg
        


    def _configure_datamodule(
            self,
            cfg_dataset: DictConfig,
            cfg_dataloader: DictConfig,
            cfg_pretf: DictConfig,
            cfg_dswrapper: DictConfig,
        ) -> GraphDataModule:

        ###########################  DATASET SETUP  ############################
        # get dataset name
        dataset_name = cfg_dataset.name
        dataset_params = cfg_dataset.params

        self.logger.info(f'Using dataset "{dataset_name}"')

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

        # add wrappers to dataset if there are any
        if cfg_dswrapper is not None:
            if not isinstance(cfg_dswrapper, list):
                cfg_dswrapper = [cfg_dswrapper]
            dataset_wrappers = [(reg_dataset_wrapper.get_class(dsw['name']), dsw['params']) for dsw in cfg_dswrapper]
            
            # dataset wrappers will wrap the datasets returned by the data resource
            # a wrapper can add new data to the dataset elements and info
            # TODO: make wrappers save data, avoiding to recompute it every time a new dataset instance is created
            data_resources.add_dataset_wrapper(dataset_wrappers)

        ##########################  DATALOADER SETUP  ##########################
        # placeholder for possible non-trivial setup of dataloader in the future

        # log batch size
        if 'train' in cfg_dataloader:
            for split, curr_cfg in cfg_dataloader.items():
                self.logger.info(f'{split} batch size: {curr_cfg.batch_size}')
        else:
            self.logger.info(f'Batch size: {cfg_dataloader.batch_size}')


        #########################  CREATE DATAMODULE  ##########################
        # create datamodule
        datamodule = GraphDataModule(
            data_resources =    data_resources,
            transform =         None,   # will be set later
            dataloader_config = cfg_dataloader
        )

        return datamodule, data_resources
    

    def _configure_transforms(self, cfg_datatf: DictConfig, model: Generator, datamodule: GraphDataModule):

        #########################  DATATRANSFORM SETUP  ########################

        # convert optional adapters to pipelines
        t2p = lambda tr: transforms_to_pipeline(
            transforms =        tr,
            data_resources =    datamodule.data_resources,
            model =             model
        )
        
        # instantiate runtime pipeline for each split
        transform = None
        if 'train' in cfg_datatf:
            transform = {}
            for split, tf in cfg_datatf.items():
                transform[split] = t2p(reg_transforms.get_instance_from_dict(tf))
        else:
            transform = t2p(reg_transforms.get_instance_from_dict(cfg_datatf))

        # set transform to datamodule
        datamodule.transform = transform

    

    def _configure_test_assignment(self, cfg_test, data_resources) -> Assignment:

        test_assignment = evaluation.reg_assignment.get_instance_from_dict(
            config = cfg_test,
            split = 'test',
            data_resources = data_resources
        )

        return test_assignment

    

    def _configure_model(
            self,
            cfg_model: DictConfig,
            dataset_info: Dict,
            test_assignment: Assignment,
        ):


        # setup console logger for the model
        console_logger = self.initialize_logger('model', self.logger.logger.level)

        # instantiate required model
        model = models.reg_models.get_instance_from_cfg(
            config = cfg_model,
            dataset_info = dataset_info,
            test_assignment = test_assignment,
            console_logger = console_logger
        )

        if self.compile:
            if not self.accelerator == 'mps':
                try:
                    import torch._dynamo
                    torch._dynamo.config.capture_scalar_outputs = True
                    model = torch.compile(model, dynamic=True)
                except Exception as e:
                    self.logger.warning(f'Could not compile the model, cause: {e}')
            else:
                self.logger.warning(f'MPS is enabled, model will not be compiled as it is not supported')

        # if distributed_training is not None:
        #     model = torch.nn.parallel.DistributedDataParallel(model, **distributed_training)

        return model



    def _configure_checkpoint(self, cfg_checkpoint: DictConfig) -> List[ModelCheckpoint]:

        checkpoint_callback = []

        if self.enable_ckp:

            for clbk_name, clbk_cfg in cfg_checkpoint.items():
                clbk = clb.reg_checkpoints.get_instance(
                    name =      clbk_name,
                    params =    clbk_cfg,
                    dirpath =   self.run_directory,
                    filename =  None
                )

                if clbk is not None:
                    if isinstance(clbk, list):
                        checkpoint_callback.extend(clbk)
                    else:
                        checkpoint_callback.append(clbk)

        else:
            # callback that never saves anything
            # it is enough to disable saving last
            # and saving top k
            checkpoint_callback.append(ModelCheckpoint(
                dirpath =       self.run_directory,
                filename =      None,
                save_last =     False,
                save_top_k =    0
            ))
        
        return checkpoint_callback


    def _configure_early_stopping(self, cfg_early_stopping: DictConfig):

        early_stopping_callback = []

        for clbk_name, clbk_cfg in cfg_early_stopping.items():
            clbk = clb.reg_early_stopping.get_instance(
                name = clbk_name,
                params = clbk_cfg,
                #dirpath =   self.run_directory,
                #filename =  None
            )

            if clbk is not None:
                if isinstance(clbk, list):
                    early_stopping_callback.extend(clbk)
                else:
                    early_stopping_callback.append(clbk)

        return early_stopping_callback

    

    def _configure_logger(self, cfg_logger: DictConfig):
        if self.enable_log:
            
            run_logger = log.reg_loggers.get_instance_from_dict(
                cfg_logger,
                save_dir =      CHECKPOINT_PATH,
                run_name =      self.config_name,
                run_config =    OmegaConf.to_container(self.cfg),
                resume =        self.resume,
                id =            self.run_id,
                version =       self.version_dir,
                model =         self.model
            )

            return run_logger
        
        return None


    def _configure_trainer(
            self,
            cfg_trainer: DictConfig,
            cfg_callbacks: DictConfig,
            cfg_logger: DictConfig,
        ) -> Trainer:

        callbacks = []

        # if debug is activated or persistent is deactivated, checkpointing and logging are deactivated!

        if 'checkpoint' in cfg_callbacks:
            checkpoint_callback = self._configure_checkpoint(cfg_callbacks.checkpoint)
            callbacks.extend(checkpoint_callback)

        if 'early_stopping' in cfg_callbacks:
            early_stopping_callback = self._configure_early_stopping(cfg_callbacks.early_stopping)
            callbacks.extend(early_stopping_callback)

        if 'others' in cfg_callbacks:
            other_callback = clb.reg_other_callbacks.get_instance_from_dict(cfg_callbacks.others)
            callbacks.append(other_callback)

        # configuring run logger
        run_logger = self._configure_logger(cfg_logger)
        if run_logger is None:
            self.enable_log = False

        self.logger.info(f'Using device: {self.accelerator}, N={self.devices}')
        self.logger.info(f'Number of epochs: {cfg_trainer.max_epochs}')

        if self.profile:
            # remove file if exists
            for f in glob.glob('perf_logs*'):
                os.remove(f)
            profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
        else:
            profiler = None

        strategy = 'ddp' if self.distributed_training is not None or self.devices > 1 else 'auto'

        # build trainer
        trainer = Trainer(
            # location
            default_root_dir =          self.run_directory,

            # training parameters
            **cfg_trainer,

            # computing devices
            accelerator =               self.accelerator,
            devices =                   self.devices,
            strategy =                  strategy,

            # visualization and debugging
            fast_dev_run =              self.debug,

            # logging and checkpointing
            logger =                    run_logger,

            # callbacks
            callbacks =                 callbacks,
            # for network debugging in case of NaNs
            profiler =                  profiler,
            detect_anomaly =            False
        )
    
        return trainer
    

    def initialize_logger(self, name: str,  level: Union[str, int] = None) -> logging.Logger:

        level = level if level is not None else logging.INFO
        if isinstance(level, str) and not level.isupper():
            level = level.upper()

        logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.propagate = False
        # remove all handlers
        logger.handlers = []
        logger.addHandler(handler)
        logger.setLevel(level)

        logger = IndentedLoggerAdapter(logger)

        return logger
    
    def turn_off_logger(self, logger: logging.Logger):
        if isinstance(logger, IndentedLoggerAdapter):
            logger = logger.logger
        
        logger.handlers = []
        logger.propagate = False


################################################################################
#                                UTILITY METHOD                                #
################################################################################

def preprocess_config(cfg: DictConfig):
    
    # resolve configuration interpolations which are using hydra choices
    choices = cfg.hydra.runtime.choices
    cfg.hydra = OmegaConf.create({'runtime': {'choices': choices}})
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # remove hydra configuration
    cfg.pop('hydra')
    cfg = OmegaConf.create(cfg)

    return cfg


################################################################################
#                            CONFIGURATION METHODS                             #
################################################################################

def setup_logger(logger: logging.Logger|int = None):
    if logger is None:
        level = logging.INFO
    elif isinstance(logger, int):
        level = logger
        logger = None
    elif isinstance(logger, logging.Logger):
        level = logger.level

    if logger is None:
        logger = logging.getLogger('configurator')
        logger.setLevel(level=level)
        
    # remove all handlers
    logger.handlers = []
    # set format of logger
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level=level)
    ch.setFormatter(formatter)
    # add console handler to logger
    logger.addHandler(ch)

    return logger


def gpus_available(platform_config):
    return torch.cuda.is_available() and platform_config['devices'] > 0


def get_platform(mode, platform_config):
    # on default, use cpu
    if hasattr(platform_config, 'accelerator'):
        acc = platform_config.accelerator
    else:
        acc = 'cpu'

    # on default, use 1 device
    if hasattr(platform_config, 'devices'):
        devices = platform_config.devices
    else:
        devices = 1

    # if mps or cpu is selected, select 1 device
    if acc == 'mps' or acc == 'cpu':
        devices = 1
    
    # if gpu is selected, check if it is available
    if acc == 'gpu' and not gpus_available(platform_config):
        acc = 'cpu'
        devices = 1

    # if distributed training is specified, override acc and devices
    # NOTE: distributed training is only supported on gpu and during training
    distributed_training = dict()
    if "train" in mode and \
        hasattr(platform_config, 'distributed_training') and \
        hasattr(platform_config.distributed_training, 'enabled') and \
        platform_config.distributed_training.enabled:

        # if accelerator is not gpu, raise an error
        if acc != 'gpu':
            raise ContextException(f'Distributed training is only supported on GPU, found {acc} instead')

        # checking all the following parameters are present
        if hasattr(platform_config.distributed_training, 'backend'):
            distributed_training['backend'] = platform_config.distributed_training.backend
        else:
            distributed_training['backend'] = 'nccl'

        if hasattr(platform_config.distributed_training, 'master_address'):
            distributed_training['master_address'] = platform_config.distributed_training.master_address
        else:
            raise ContextException(f'Distributed training requires master_address to be specified')

        if hasattr(platform_config.distributed_training, 'master_port'):
            distributed_training['master_port'] = platform_config.distributed_training.master_port
        else:
            raise ContextException(f'Distributed training requires master_port to be specified')

        if hasattr(platform_config.distributed_training, 'world_size'):
            distributed_training['world_size'] = platform_config.distributed_training.world_size
        else:
            raise ContextException(f'Distributed training requires world_size to be specified')

        if hasattr(platform_config.distributed_training, 'rank'):
            distributed_training['rank'] = platform_config.distributed_training.rank
        else:
            raise ContextException(f'Distributed training requires rank to be specified')

        if hasattr(platform_config.distributed_training, 'find_unused_parameters'):
            distributed_training['find_unused_parameters'] = platform_config.distributed_training.find_unused_parameters
        else:
            distributed_training['find_unused_parameters'] = True

        if hasattr(platform_config.distributed_training, 'broadcast_buffers'):
            distributed_training['broadcast_buffers'] = platform_config.distributed_training.broadcast_buffers
        else:
            distributed_training['broadcast_buffers'] = False

    return distributed_training if len(distributed_training) > 0 else None, acc, devices