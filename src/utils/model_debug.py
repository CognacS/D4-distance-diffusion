from typing import Dict, List, Callable

from abc import ABC, abstractmethod
from functools import wraps

# import a pattern matching library
from fnmatch import fnmatch
from copy import copy

import torch.nn as nn
import lovely_tensors as lt


KWORD_ROOT = 'root'
KWORD_UNKNOWN = 'unknown'
KWORD_HERE = '.'
KWORD_BACK = '..'


class HashableModule:
    # this object is hashed through its name, so that it can be used as a key
    # in a dictionary, and when indexed, it is enough to use the name of the
    # module
    
    def __init__(self, module: nn.Module, name: str='', path: str = ''):
        self.module = module
        self.name = name
        self.path = path

    def get_name(self):
        return self.name
    
    def get_path(self):
        return self.path
    
    def __repr__(self):
        return f'{self.name} ({self.module.__class__.__name__})'

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, HashableModule):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False
        

from torch import Tensor
from src.datatypes.sparse import SparseGraph
from src.datatypes.dense import DenseGraph
from copy import deepcopy

def to_cpu(x):
    if isinstance(x, (Tensor, SparseGraph, DenseGraph)):
        return x.to('cpu')
    elif isinstance(x, (list, tuple)):
        return [to_cpu(y) for y in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def get_hook_fn(tracked_data: Dict, tracked_variables):
    def hook_fn(module, *args):
        for tv in tracked_variables:
            tracked_data[tv].append(to_cpu(deepcopy(args[TRACKED_VAR_TO_IDX[tv]])))
    return hook_fn


TRACKABLE_VARIABLES = {
    'forward': ['input', 'output'],
    'backward': ['grad_input', 'grad_output']
}

TRACKED_VAR_TO_IDX = {
    'input': 0,
    'output': 1,
    'grad_input': 0,
    'grad_output': 1
}

class TrackerHook:
    """
    Recall that a forward hook can track the following variables:
    - input
    - output
    while a backward hook can track the following variables:
    - grad_input
    - grad_output
    """
    
    def __init__(self, hmod: HashableModule, hook_type: str='forward', tracked_variables: List=None):
        self.hmod = hmod
        self.hook_type = hook_type
        self.tracked_variables = tracked_variables if tracked_variables is not None else TRACKABLE_VARIABLES[hook_type]
        self.hook = None
        self.tracked_data = {v: [] for v in tracked_variables}
        self._register_hook()


    def _register_hook(self):
        trackable_vars = TRACKABLE_VARIABLES[self.hook_type]
        assert all([v in trackable_vars for v in self.tracked_variables]), \
            f'Unknown tracked variable(s) for {self.hook_type} hook: {self._variables_repr()}'

        hook_fn = get_hook_fn(self.tracked_data, self.tracked_variables)
        if self.hook_type == 'forward':
            self.hook = self.hmod.module.register_forward_hook(hook_fn)
        elif self.hook_type == 'backward':
            self.hook = self.hmod.module.register_full_backward_hook(hook_fn)
        else:
            raise ValueError(f'Unknown hook type: {self.hook_type}')

    def get_tracked_data(self):
        return self.tracked_data

    def remove(self):
        if self.hook is not None:
            self.hook.remove()

    def _variables_repr(self):
        return ', '.join(self.tracked_variables)

    def __repr__(self):
        return f'{self.hook_type.capitalize()} {self.__class__.__name__} on {str(self.hmod)}, tracking: {self._variables_repr()}'


def join_paths(path1: str, path2: str):
        tpath1 = trim_path(path1)
        tpath2 = trim_path(path2)
        if tpath1 == '':
            return tpath2
        elif tpath2 == '':
            return tpath1
        return tpath1 + '/' + tpath2

def trim_path(path: str):
    return path.strip('/')


class ModelTrackerException(Exception):
    pass


class ModelTrackerSafetyContext:
    def __init__(self, tracker):
        self.tracker = tracker

    def __enter__(self):
        self.already_disabled = self.tracker._disable_callbacks
        self.tracker._disable_callbacks = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.tracker._disable_callbacks = False or self.already_disabled


class ModelTracker:

    def __init__(self, model: nn.Module):
        self.model: nn.Module = model
        self.current_path: str = KWORD_ROOT
        self.stack: List[nn.Module] = [self.model]
        self.hooks: Dict[TrackerHook] = {}
        self._disable_callbacks = False
        self.callbacks: List[BaseModelTrackerCallback] = []


    ############################################################################
    #                                 UTILITIES                                #
    ############################################################################

    def _locate_module(self, path: str, module: nn.Module=None):
        """Locate and return a submodule. If module is None:
        - if path is the current module is used as the root."""

        parts = path.split('/')
        start = 0
            
        if module is None:
            if parts[0] == KWORD_ROOT:
                module = self.model
                start = 1
                res_path = []
                res_stack = [self.model]
            else:
                module = self.stack[-1]
                res_path = self.current_path.split('/')[1:]
                res_stack = copy(self.stack) # shallow copy
        else:
            res_path = [KWORD_UNKNOWN]
            res_stack = []

        for i in range(start, len(parts)):

            name = parts[i]

            if name == '' or name == KWORD_HERE:
                continue

            try:
                if name == KWORD_BACK:
                    res_path.pop()
                    module = res_stack.pop()
                else:
                    res_path.append(name)
                    module = getattr(module, name)
                    res_stack.append(module)
            except AttributeError:
                format_path = '/'.join(parts[:i] + ['(' + name + ')'] + parts[i+1:])
                raise ModelTrackerException(
                    f'No module named "{name}" in {module.__class__.__name__}, error in (*) in path: {format_path}'
                    ) from None
            except IndexError:
                format_path = '/'.join(parts[:i] + ['(' + name + ')'] + parts[i+1:])
                raise ModelTrackerException(
                    f'Tried to go back too many times, error in (*) in path: {format_path}'
                    ) from None
        
        # append KWORD_ROOT to the beginning of the path
        res_path.insert(0, KWORD_ROOT)

        module_name = res_path[-1]
        res_path = '/'.join(res_path)

        return module, module_name, res_path, res_stack
    

    def _module_ls(self, hmod: HashableModule, recursive: bool = False, collector: List = None):
        
        mod_dict = {}

        for name, module in hmod.module.named_children():

            sub_hmod = HashableModule(module, name, join_paths(hmod.path, name))

            if collector is not None:
                collector.append(sub_hmod)

            mod_dict[sub_hmod] = {}

            if recursive:
                sub_mod_dict = self._module_ls(sub_hmod, True, collector)
                if len(sub_mod_dict) > 0:
                    mod_dict[sub_hmod] = sub_mod_dict

        return mod_dict
    

    def define_cmd(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):

            # the safety context is used to disable callbacks during the execution
            # such that only the call at the top of the stack calls the
            # callbacks
            with ModelTrackerSafetyContext(self):
                res = func(self, *args, **kwargs)


            if not self._disable_callbacks:
                cmd = func.__name__
                for callback in self.callbacks:
                    callback(cmd, res)

            return res

        return wrapper


    ############################################################################
    #                                 COMMANDS                                 #
    ############################################################################

    ###########################  NAVIGATION COMMANDS  ##########################

    @define_cmd
    def ls(
            self,
            path: str='',
            recursive: bool = False,
            absolute: bool = False,
            as_list: bool = False
        ):

        module, module_name, full_path, _ = self._locate_module(path)
        mod_path = full_path if absolute else path
        hmod = HashableModule(module, module_name, mod_path)
        if as_list:
            collector: List[HashableModule] = []
        else:
            collector = None
        mod_dict = self._module_ls(hmod, recursive=recursive, collector=collector)

        if as_list:
            return collector
        else:
            return mod_dict


    @define_cmd
    def cd(self, path: str):

        if path != '':
            _, _, self.current_path, self.stack = self._locate_module(path)


    @define_cmd
    def where(self):

        return self.current_path


    @define_cmd
    def search(
            self,
            path: str='',
            pattern: str='', 
            recursive: bool = False,
            absolute: bool = False,
        ):
        # find all modules at the path
        collector = self.ls(
            path=path,
            recursive=recursive,
            absolute=absolute,
            as_list=True
        )

        # if pattern is empty return all modules (like calling ls)
        if pattern == '':
            return collector
        
        # else, filter modules by pattern
        matches = []
        for hmod in collector:
            if fnmatch(hmod.get_path(), pattern):
                matches.append(hmod)

        return matches


    #############################  HOOKS COMMANDS  #############################

    @define_cmd
    def puthook(self, path: str='', *tracked_variables, **kwargs):

        module, module_name, full_path, _ = self._locate_module(path)
        if full_path in self.hooks:
            raise ValueError(f'A hook is already attached to {full_path}. Please remove it first.')

        hmod = HashableModule(module, module_name, full_path)

        # build hook
        hook_type = kwargs.get('hook_type', 'forward')
        hook = TrackerHook(hmod, hook_type, tracked_variables)

        # attach hook to tracker
        self.hooks[full_path] = hook
        return hook


    @define_cmd
    def lshook(self, path='', recursive: bool = False):
        
        ls_res = self.ls(path=path, as_list=True, absolute=True, recursive=recursive)
        ls_res = [hmod.get_path() for hmod in ls_res]

        found_hooks = {k: self.hooks[k] for k in ls_res if k in self.hooks}

        return found_hooks


    @define_cmd
    def searchhook(self, path: str='', pattern: str='*', recursive: bool = False):

        search_res = self.search(path=path, pattern=pattern, absolute=True, recursive=recursive)
        search_res = [hmod.get_path() for hmod in search_res]

        found_hooks = {k: self.hooks[k] for k in search_res if k in self.hooks}

        return found_hooks


    @define_cmd
    def herehook(self):
        hook_here = self.hooks.get(self.current_path, None)
        return hook_here
    

    @define_cmd
    def rmhook(self, path: str=''):
        full_path = self._locate_module(path)[2]
        hook = self.hooks.pop(full_path, None)
        if hook is not None:
            hook.remove()

        return hook


    @define_cmd
    def hookdata(self, path: str='', pattern: str='*', recursive: bool = False):
        found_hooks: Dict[str, TrackerHook] = self.searchhook(path=path, pattern=pattern, recursive=recursive)
        hook_data = {k: v.get_tracked_data() for k, v in found_hooks.items()}

        return hook_data


    @define_cmd
    def resethooks(self):
        num_hooks = len(self.hooks)
        for hook in self.hooks.values():
            hook.remove()

        self.hooks = {}

        return num_hooks


    ############################  PARAMS COMMANDS  #############################
    def get_module(self, path: str=''):
        return self._locate_module(path)[0]
    
    @define_cmd
    def params(
            self,
            path: str='',
            statistics: bool = True,
            recursive: bool = False
        ):
        module = self.get_module(path)
        params = dict(module.named_parameters(recurse=recursive))
        if statistics:
            params = {k: lt.lovely(v) for k, v in params.items()}
        return params


    def close(self):
        self.resethooks()


    ############################################################################
    #                           SUB SESSION METHODS                            #
    ############################################################################

    def subsession(self):
        model_tracker = ModelTracker(self.model)
        model_tracker.current_path = self.current_path
        model_tracker.stack = copy(self.stack)
        model_tracker.callbacks = copy(self.callbacks)
        return model_tracker


    def __enter__(self):

        return self


    def __exit__(self, exc_type=None, exc_value=None, traceback=None):

        self.close()


    ############################################################################
    #                                 PLUGINS                                  #
    ############################################################################


    def start_interactive_session(self, **print_kwargs):
        with print_mode(self, **print_kwargs):
            interactive_session(self)




################################################################################
#                                  PRINT MODE                                  #
################################################################################

def print_lists(ls: List, **kwargs):
    for item in ls:
        print(item)

def print_dicts(ls: Dict, **kwargs):
    key_lens = [len(str(k)) for k in ls.keys()]
    max_key_len = max(key_lens) if len(key_lens) > 0 else 0
    for item, content in ls.items():
        # pad until max length, put key, space, value
        print(f'{str(item):<{max_key_len}} : {content}')


def print_hmod_lists(ls: List[HashableModule], **kwargs):
    for item in ls:
        print(item.get_path())


def print_hmod_dicts(ls: Dict, to_depth: int = -1, indent_level: int = 0, indent_size: int = 2, **kwargs):

    how_many_indents = indent_size * indent_level

    for item, content in ls.items():
        print(' ' * how_many_indents + str(item))
        if len(content) > 0:
            if to_depth == -1 or indent_level < to_depth:
                print_hmod_dicts(content, to_depth, indent_level + 1, indent_size=indent_size)
            else:
                print(' ' * (how_many_indents + indent_size) + f'[{len(content)} hidden modules]')




class BaseModelTrackerCallback(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
    

    @abstractmethod
    def __call__(self, cmd: str, *args):
        ...



def context_callback(func: Callable[..., BaseModelTrackerCallback]):

    @wraps(func)
    def wrapper(model_tracker: ModelTracker, **kwargs):
        callback = func(**kwargs)
        return CallbackContext(model_tracker, callback)

    return wrapper


class CallbackContext:

    def __init__(self, tracker: ModelTracker, callback: BaseModelTrackerCallback):
        self._tracker = tracker
        self._callback = callback

    
    def __enter__(self):
        self._tracker.callbacks.append(self._callback)


    def __exit__(self, exc_type=None, exc_value=None, traceback=None):

        self._tracker.callbacks.remove(self._callback)


@context_callback
def print_mode(**kwargs) -> BaseModelTrackerCallback:

    call_back = PrintModeCallback(**kwargs)

    return call_back

class PrintModeCallback(BaseModelTrackerCallback):

    def __init__(self, indent_size=2, to_depth=-1):
        super().__init__(indent_size=indent_size, to_depth=to_depth)
        

    def _ls_print(self, mod_dict):
        if isinstance(mod_dict, list):
            print_hmod_lists(mod_dict, **self.kwargs)
        else:
            print_hmod_dicts(mod_dict, **self.kwargs)
    def _where_print(self, path):
        print(path)
    def _search_print(self, matches):
        print_hmod_lists(matches, **self.kwargs)

    def _lshook_print(self, hooks):
        if len(hooks) > 0:
            print_dicts(hooks, **self.kwargs)
        else:
            print('No hooks found')
    def _searchhook_print(self, hooks):
        self._lshook_print(hooks)
    def _herehook_print(self, hook):
        if hook is not None:
            print(hook)
        else:
            print('No hooks found')

    def _resethooks_print(self, num_hooks):
        print(f'Removed {num_hooks} hooks')
    def _hookdata_print(self, data):
        print(data)

    def _params_print(self, params):
        if len(params) > 0:
            print_dicts(params, **self.kwargs)
        else:
            print('No parameters found')


    def __call__(self, cmd: str, *args):

        # get print associated to cmd
        print_func = getattr(self, f'_{cmd}_print', None)
        
        if print_func is not None:
            print_func(*args)
        # else do nothing
        


################################################################################
#                                 INTERACTIVE                                  #
################################################################################

from os import system, name
import sys
from IPython.display import clear_output

def clear():
    # for jupyter notebook
    if 'ipykernel' in sys.modules:
        clear_output(wait=False)
    # for windows
    elif name == 'nt':
       system('cls')
    # for mac and linux
    else:
       system('clear')


def print_help(model_tracker: ModelTracker):
    print('Available commands:')
    for cmd, (func, help_str) in TRK_CMD_DICT.items():
        print(f'  {cmd} - {help_str}')

def clear_screen(model_tracker: ModelTracker):
    clear()


TRK_CMD_DICT = {
    'ls': (ModelTracker.ls, 'ls [<path>] [-r]'),
    'cd': (ModelTracker.cd, 'cd <path>'),
    'search': (ModelTracker.search, 'search [<path>] [<pattern>] [-r]'),
    'where': (ModelTracker.where, 'where'),
    'puthook': (ModelTracker.puthook, 'puthook [<path>] [<tracked_variables>] [-f] [-b]'),
    'lshook': (ModelTracker.lshook, 'lshook [<path>] [-r]'),
    'searchhook': (ModelTracker.searchhook, 'searchhook [<path>] [<pattern>] [-r]'),
    'herehook': (ModelTracker.herehook, 'herehook'),
    'rmhook': (ModelTracker.rmhook, 'rmhook [<path>]'),
    'resethooks': (ModelTracker.resethooks, 'resethooks'),
    'params': (ModelTracker.params, 'params [<path>] [-v]'),

    'help': (print_help, 'help'),
    'clear': (clear_screen, 'clear'),
}
TRK_CMD_ARGS = {
    '-r': ('recursive', True),
    '-f': ('hook_type', 'forward'),
    '-b': ('hook_type', 'backward'),
    '-v': ('statistics', False),
}


def interactive_session(model_tracker: ModelTracker):
    while True:
        cmd = input(f'{model_tracker.current_path}>')
        if cmd == 'exit':
            break
        else:
            try:
                cmd_str_args = cmd.split(' ', 1)
                cmd = cmd_str_args[0].strip()

                if cmd in TRK_CMD_DICT:
                    
                    args = []
                    kwargs = {}

                    if len(cmd_str_args) > 1:
                        str_args = cmd_str_args[1]
                        str_args = str_args.strip().split(' ')
                        for str_arg in str_args:
                            if str_arg in TRK_CMD_ARGS:
                                name, value = TRK_CMD_ARGS[str_arg]
                                kwargs[name] = value
                            else:
                                args.append(str_arg)

                    cmd = TRK_CMD_DICT[cmd][0]
                    cmd(model_tracker, *args, **kwargs)
                else:
                    print(f'Command "{cmd}" not recognized.')
            except Exception as e:
                print(e)
            sys.stdout.flush()