from src.utils.decorators import ClassRegister

reg_dataresources = ClassRegister('DataResources')

####################### import datasets and dataresources ######################
# QM9 taken from torch_geometric (raw is sdf file)
from .qm9 import *
# ZINC as in DIG (raw is SMILES file in csv
from .zinc import *
# QM9 as in DIG (raw is SMILES file in csv)
from .qm9_dig import *
# datasets from the GraphGDP repository (networkx graphs)
from .graphgdp_datasets import *
# datasets from the Graph2Gauss repository
from .large_graphs import *
# datasets from the Graph2Gauss repository
from .large_graphs import *
# datasets from the MSD repository
from .msd import *