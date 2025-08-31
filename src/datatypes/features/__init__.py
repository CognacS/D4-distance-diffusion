from typing import List, Union, Dict
from src.utils.decorators import ClassRegister

reg_features = ClassRegister('Features')


def get_features_list(f_names: List[str]):
    def with_or_without_args(feature: Union[str, Dict]):
        if isinstance(feature, str):
            return {'name': feature}
        else:
            return {'name': list(feature.keys())[0], 'params': list(feature.values())[0]}
    return [reg_features.get_instance(**with_or_without_args(f)) for f in f_names]


from src.datatypes.features.geometric import *
from src.datatypes.features.spectral import *
from src.datatypes.features.spatial import *