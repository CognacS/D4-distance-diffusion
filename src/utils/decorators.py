from typing import Dict, Any, Optional, Union


class Register:

    def __init__(self, name):
        self.name = name
        self._register = {}

    def __getitem__(self, key):
        return self._register[key]

    def __setitem__(self, key, value):
        if not key in self._register:
            self._register[key] = value


    def __repr__(self):
        return self.name + str(self._register)
    

    def register(self, name: Optional[str] = None):
        raise NotImplementedError

    def get_class(self, name: str):
        if name not in self._register:
            raise ValueError(
                'Class {name} not found in register {register}. Available: {available}'.format(
                    name=name,
                    register=self.name,
                    available=list(self._register.keys())
                )
            )
        return self._register[name]
    
    def get_instance(self, name: str, params: Dict = None, dict_kwargs: Dict = None, **kwargs):
        cls = self.get_class(name)
        if params is None:
            params = {}
        if dict_kwargs is None:
            dict_kwargs = {}
        return cls(**params, **dict_kwargs, **kwargs)
    
    def get_instance_from_dict(self, config: Dict[str, Dict], dict_kwargs: Dict = None, **kwargs):
        if config is None or len(config) == 0:
            return None
        k, v = list(config.items())[0]
        return self.get_instance(name=k, params=v, dict_kwargs=dict_kwargs, **kwargs)

    def get_instance_from_cfg(self, config: Dict[str, Dict], dict_kwargs: Dict = None, **kwargs):
        name = config['name']
        params = None if 'params' not in config else config['params']
            
        return self.get_instance(name=name, params=params, dict_kwargs=dict_kwargs, **kwargs)


class InstanceRegister(Register):

    def register(self, config: Optional[Union[str, Dict[str, Dict[str, Any]]]] = None):
    
        def decorator(cls):

            if config is None:
                name_args = cls.__name__
            else:
                name_args = config

            if isinstance(name_args, str):
                self[name_args] = cls()

            elif isinstance(name_args, dict):
                for n, p in name_args.items():
                    self[n] = cls(**p)

            return cls

        return decorator
    
    def __repr__(self):
        return "Registered {}: {}".format(
            self.name,
            ''.join([f'\n\t{k}: {v}' for k, v in self._register.items()])
        )


class ClassRegister(Register):

    def register(self, name: Optional[str] = None):
    
        def decorator(cls):

            if name is None:
                name_cls = cls.__name__
            else:
                name_cls = name

            self[name_cls] = cls
            return cls

        return decorator

    def __repr__(self):
        return "Registered {}: {}".format(
            self.name,
            ''.join([f'\n\t{k}: {v.__name__}' for k, v in self._register.items()])
        )