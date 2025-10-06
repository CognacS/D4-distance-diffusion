from src.utils.decorators import ClassRegister

reg_dataresources = ClassRegister('DataResources')

####################### import datasets and dataresources ######################
# QM9 taken from torch_geometric (raw is sdf file)
from .qm9 import *
# ZINC as in DIG (raw is SMILES file in csv
from .zinc import *
# GDB13
from .gdb13 import *