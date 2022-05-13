import os, sys, pdb, atexit
from importlib.machinery import PathFinder, FileFinder, ModuleSpec, SourceFileLoader, ExtensionFileLoader

import detrain.hookfunctions as hooker
import detrain.hookpool as hookpool

class CustomExtensionFileLoader(ExtensionFileLoader):

    def exec_module(self, module):
        super().exec_module(module)

        hookpool.hook_pool_ext(self.name, module)
        return module

class CustomSourceFileLoader(SourceFileLoader):

    def exec_module(self, module):
        super().exec_module(module)

        hooker.hook(self.name, module)
        hookpool.hook_pool(self.name, module)
        return module


class SourceFileFinder(PathFinder):

    def __init__(self, module_name):
        self.module_name = module_name

    def find_spec(self, fullname, path=None, target=None):
        #print(fullname)
        if fullname == self.module_name:
            spec = super().find_spec(fullname, path, target)
            return ModuleSpec(fullname,
                              CustomSourceFileLoader(fullname, spec.origin))
        return

class ExtensionFileFinder(PathFinder):

    def __init__(self, module_name):
        self.module_name = module_name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.module_name:
            spec = super().find_spec(fullname, path, target)
            new_spec = ModuleSpec(fullname,
                              CustomExtensionFileLoader(fullname, spec.origin))
            new_spec.origin = spec.origin
            return new_spec
        return


def patch():
    '''
    add hooking modules here
    '''
    for module in hookpool.hook_pool_modules_ext:
        sys.meta_path.insert(0, ExtensionFileFinder(module))

    for module in hookpool.hook_pool_modules:
        sys.meta_path.insert(0, SourceFileFinder(module))

    for module in hooker.hook_modules:
        sys.meta_path.insert(0, SourceFileFinder(module))
 
def set_torch():
    try:
        import torch
    
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(1234)
    
        hooker.is_pytorch_available = True 
    except ImportError:
        hooker.is_pytorch_available = False 
def set_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.python.framework import ops
    
        assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        #os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['PYTHONHASHSEED']=str(40)
    
        tf.random.set_seed(42)

        hooker.is_tensorflow_available = True 
    except (ImportError, AssertionError):
        hooker.is_tensorflow_available = False
        print("Tensorflow is not installed!!!")

def set_libs():
    import random
    random.seed(43)

    import numpy as np
    if hookpool.np_seed_func!=None:
        hookpool.np_seed_func(44)
    else:
        np.random.seed(44)
    np.set_printoptions(threshold=100000, edgeitems=1000)

def collect_trace():
    import detrain.tracer as t
    t.init()
    atexit.register(t.stop_trace)

def load_checkpoints():
    import detrain.ckpt_manager as ckpt
    ckpt.load_checkpoints()

def my_clean_up():
    if 'CKPT_DIR' not in os.environ:
        return

    from shutil import move, copyfile
    inst_file = os.path.join(os.environ['CKPT_DIR'], "inst.file")
    if os.path.exists(inst_file):
        with open(inst_file, 'r') as f:
            removed = set()
            for file_name in f:                
                file_name = file_name.strip()
                if file_name not in removed:
                    removed.add(file_name)
                    #copyfile(file_name+".orig", file_name)
                    if os.path.exists(file_name+".orig0"):
                        copyfile(file_name+".orig0", file_name)
                    else:
                        copyfile(file_name+".orig", file_name)

        os.remove(inst_file)

if ('ENABLE_TRACE' in os.environ) and (os.environ['ENABLE_TRACE']=="1"):
    collect_trace()
else:    
    hooker.hook_main_loop()
    patch()
    set_torch()
    set_tensorflow()
    set_libs()
    load_checkpoints()
    atexit.register(my_clean_up)

