from src.utils.decorators import ClassRegister

reg_transforms = ClassRegister('Transforms')

from .to_onehot import *
from .direction import *