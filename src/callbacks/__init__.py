from src.utils.decorators import ClassRegister


reg_checkpoints = ClassRegister('Checkpoint')
reg_early_stopping = ClassRegister('EarlyStopping')
reg_other_callbacks = ClassRegister('OtherCallbacks')


# register some classes from pytorch lightning
import pytorch_lightning.callbacks as pl_clb

reg_checkpoints.register()(pl_clb.ModelCheckpoint)
reg_early_stopping.register()(pl_clb.EarlyStopping)

import src.callbacks.modular_checkpointing
import src.callbacks.modular_early_stopping
import src.callbacks.ema