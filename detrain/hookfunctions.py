
import os, pdb
from shutil import copyfile
import detrain.instrumentor as inst

is_pytorch_available = False
is_tensorflow_available = False
# hashlib.md5(str(self.checkpoint.optimizer.get_weights()).encode('utf-8')).hexdigest()

'''''''''''''''''''''''''''''''''''''''''
wrap functions
'''''''''''''''''''''''''''''''''''''''''
def wrap(func, *args, **kwargs):
    #for arg in args:
    '''
    only the class instance is of interest 
    '''
    #inst.dispatch(args[0], func)
    return 
    #print("Hook with", args, len(kwargs))


WRAPPERS = [wrap]

def patcher(function):
    def wrapper(*args, **kwargs):
        global WRAPPERS
        for w in WRAPPERS:
            w(function, *args, **kwargs)
        
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_module(function):
    def wrapper(*args, **kwargs):
        inst.handle_module(args[0], function)
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_optim(function):
    def wrapper(*args, **kwargs):
        inst.handle_optim(args[0], function)
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_sched(function):
    def wrapper(*args, **kwargs):
        inst.handle_sched(args[0], function)
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_pytorch_dataloader(function):
    def wrapper(*args, **kwargs):
        inst.handle_pytorch_dataloader(args[0], function)
        ret = function(*args, **kwargs)
        return ret

    return wrapper

def patcher_tf_dataloader(inst_func, function):
    def wrapper(*args, **kwargs):
        exe, value = inst_func(args[0], function)
        if exe:
            ret = function(*args, **kwargs)
            if value==None:
                inst.save_data_sample(ret)
            return ret
        else:
            return value

    return wrapper


def patcher_tf_eager(function):
    def wrapper(*args, **kwargs):
        if inst.handle_tf_eager(args[0], function):
            ret = function(*args, **kwargs)
            return ret
        else:
            #return 0
            return {'total_loss': [0],
                    'batch_size': 1,
                    'output_losses': [0],
                    'metrics': []}

    return wrapper


#def patcher_tf_static(function):
def patcher_tf_wrapper(function, args):
    def wrapper(*args, **kwargs):
        import tensorflow as tf
        exe, value = inst.handle_tf_wrapper_func_dl(args[0], function, args)
        if exe:
            ret = function(*args, **kwargs)
            if value==None:
                inst.save_sample(ret)
            return ret
        else:
            return value
            #return 0
            #return tf.constant(0)
            #return tf.constant(0), {'reg':tf.constant(0),
            #                        'loc':tf.constant(0),
            #                        'class':tf.constant(0)}
            return {'total_loss': [0],
                    'batch_size': 1,
                    'output_losses': [0],
                    'metrics': []}

    return wrapper

def patcher_tf_wrapper_no_persistent(function):
    def wrapper(*args, **kwargs):
        exe, value = inst.handle_tf_wrapper_func(args[0], function)
        if exe:
            ret = function(*args, **kwargs)
            if value==None:
                inst.save_sample(ret)
            return ret
        else:
            return value
    return wrapper

def patcher_tf_norm(function):
    def wrapper(*args, **kwargs):
        kwargs["fused"] = False
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_tf_datasets(function):
    def wrapper(*args, **kwargs):
        kwargs["sloppy"] = False
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_tf_dropout(function):
    def wrapper(*args, **kwargs):
        # XXZ: inst.has_drop --> inst.has_dropout
        inst.has_dropout = True
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_tf_deduplicate(function):
    def wrapper(values, indices):
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from tensorflow.python.ops import math_ops
        
        unique_indices, new_index_positions = array_ops.unique(indices)
        with tf.device("CPU"):
            summed_values = math_ops.unsorted_segment_sum(
                    values, new_index_positions,
                    array_ops.shape(unique_indices)[0])
        return (summed_values, unique_indices)

    return wrapper


def patcher_tf_sgd(function):
    def wrapper(*args, **kwargs):
        momentum = kwargs.get("momentum", 0.09)
        kwargs["momentum"] = momentum
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def custom_sparse_softmax_cross_entropy_with_logits(features, labels, name=None):
    import tensorflow as tf
    num_classes = features.shape[1]
    if num_classes == None:
        #num_classes = 1000
        num_classes = 24512
        #num_classes = 3
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)
    features = tf.nn.softmax(features)
    loss = -tf.reduce_sum(tf.one_hot(labels, num_classes)*tf.math.log(features+1e-9), -1)
    #loss = tf.reduce_mean(loss)
    #tf.print("===>loss", loss, summarize=-1)
    return [loss, None]

def custom_softmax_cross_entropy_with_logits(features, labels, name=None):
    import tensorflow as tf
    features = tf.nn.softmax(features)
    loss = -tf.reduce_sum(labels*tf.math.log(features+1e-9), -1)
    #tf.print("===>myloss", features, summarize=-1)
    #loss = tf.reduce_mean(loss)
    #tf.print("===>loss", loss, summarize=-1)
    return [loss, None]


def patcher_tf_loss(function):
    def wrapper(*args, **kwargs):
        import tensorflow as tf
        if "categorical_crossentropy" in args[0].name:
            """
            y_true = kwargs.get("y_true", None)
            y_pred = kwargs.get("y_pred", None)
            if y_true == None:
                y_true = args[1]
            if y_pred == None:
                y_pred = args[-1]
            ret = custom_softmax_cross_entropy_with_logits(y_pred, y_true)
            """
            ret = custom_softmax_cross_entropy_with_logits(*args, **kwargs)
        else:
            ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_tf_ssce(function):
    return custom_sparse_softmax_cross_entropy_with_logits

def patcher_tf_sce(function):
    return custom_softmax_cross_entropy_with_logits


def custom_agggrad(gradients):
    import tensorflow as tf
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import math_ops
    
    assert gradients, "No gradients to aggregate"
    
    gradients = [g for g in gradients if g is not None]
    
    if len(gradients) == 0:
        return None 
    if len(gradients) == 1:
        return gradients[0]
    if all(isinstance(g, ops.Tensor) for g in gradients):
        ret = gradients[0]
        for i in range(1, len(gradients)):
            ret += gradients[i]
        return ret
    else:
        assert all(isinstance(g, (ops.Tensor, ops.IndexedSlices)) 
                for g in gradients)

        # If any gradient is a `Tensor`, sum them up and return a dense tensor
        # object.
        if any(isinstance(g, ops.Tensor) for g in gradients):
            with tf.device("CPU"):
                ret = tf.convert_to_tensor(gradients[0]) 
                for i in range(1, len(gradients)):
                    ret += tf.convert_to_tensor(gradients[i])
                return ret

        # The following `_as_indexed_slices_list` casts ids of IndexedSlices into
        # int64. It is to make sure the inputs of `concat` all have same the data
        # type.
        gradients = math_ops._as_indexed_slices_list(gradients)  # pylint: disable=protected-access
        
        def flatten_nested_indexed_slices(grad):
            assert isinstance(grad, ops.IndexedSlices)
            if isinstance(grad.values, ops.Tensor):
                return grad
            else:
                assert isinstance(grad.values, ops.IndexedSlices)
                g = flatten_nested_indexed_slices(grad.values)
                return ops.IndexedSlices(g.values, 
                        array_ops.gather(grad.indices, g.indices), g.dense_shape)

        gradients = [flatten_nested_indexed_slices(x) for x in gradients]
        # Form IndexedSlices out of the concatenated values and indices.
        concat_grad = ops.IndexedSlices(
                array_ops.concat([x.values for x in gradients], axis=0),
                array_ops.concat([x.indices for x in gradients], axis=0),
                gradients[0].dense_shape)
        
        return concat_grad


def patcher_tf_gradvspace(function):
    def wrapper(*args, **kwargs):
        kwargs["aggregate_fn"] = custom_agggrad
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_tf_gradfunc(function):
    def custom_grad_func_depthwise(op, grad):
        import tensorflow as tf
        from tensorflow.python.ops import nn_grad
        with tf.device("CPU"):
            ret = nn_grad._DepthwiseConv2dNativeGrad(op, grad)
        return ret
    
    def wrapper(name):
        if name == "DepthwiseConv2dNative" and function.__self__._name == "gradient":
            return custom_grad_func_depthwise
        else:
            return function(name)

    return wrapper


def patcher_tf_imageops(function):
    def wrapper(*args, **kwargs):
        import tensorflow as tf
        with tf.device("CPU"):
            ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_tf_randops(function):
    def wrapper(*args, **kwargs):
        import random
        #import tensorflow as tf
        seed = kwargs.get("seed", None)
        seed2 = kwargs.get("seed2", None)
        if seed is None:
            kwargs["seed"] = 4141  
        if seed2 is None:
            kwargs["seed2"] = 4242 
        ret = function(*args, **kwargs)
        return ret

    return wrapper


def patcher_tf_sample(function):
    def wrapper(*args, **kwargs):
        from tensorflow.python.framework import random_seed
        seed = kwargs.get("seed", 0)
        seed2 = kwargs.get("seed2", 0)
        if seed==0 and seed2==0:
            seed, seed2 = random_seed.get_seed(None)
            kwargs["seed"] = seed  
            kwargs["seed2"] = seed2 
        ret = function(*args, **kwargs)
        return ret

    return wrapper

def patch_tf_multi_device_iter(function):
    def wrapper(*args, **kwargs):
        import tensorflow as tf
        with tf.device("CPU"):
            return function(*args, **kwargs)

    return wrapper


'''''''''''''''''''''''''''''''''''''''''''''''''''
hook functions
'''''''''''''''''''''''''''''''''''''''''''''''''''
def find_attr(m, attr):
    attrs = attr.split(".")
    target = m

    last_mod = target
    last_attr = attrs[0]
    for a in attrs:
        last_mod = target
        last_attr = a

        t = getattr(target, a, None)
        if t is not None:
            target = t
        else:
            return None, None, None
        
    return target, last_mod, last_attr

# hook pytorch functions
def hook_torch(m):
    m.__init__ = patcher(m.__init__)


def hook_torch_module(m):
    m.Module.__init__ = patcher_module(m.Module.__init__)


def hook_torch_optim(m):
    m.Optimizer.__init__ = patcher_optim(m.Optimizer.__init__)


def hook_torch_scheduler(m):
    m._LRScheduler.__init__ = patcher_sched(m._LRScheduler.__init__)


def hook_torch_dataloader(m):
    m.DataLoader.__init__ = patcher_pytorch_dataloader(m.DataLoader.__init__)
    m.DataLoader.__iter__ = patcher_pytorch_dataloader(m.DataLoader.__iter__)
    #m.DataLoader.__len__ = patcher(m.DataLoader.__len__)
    m._MultiProcessingDataLoaderIter.__next__  = patcher_pytorch_dataloader(m._MultiProcessingDataLoaderIter.__next__)
    m._SingleProcessDataLoaderIter.__next__  = patcher_pytorch_dataloader(m._SingleProcessDataLoaderIter.__next__)


""" hook tensorflow functions """
def hook_tf_module(m):
    m.Model.__init__ = patcher_module(m.Model.__init__)


def hook_tf_dataloader(m):
    m.DistributedIterator.__init__ = patcher_tf_dataloader(inst.handle_tf_dl_init, m.DistributedIterator.__init__)
    #m.DistributedIterator.__iter__ = patcher_tf_dataloader(m.DistributedIterator.__iter__)
    #m.DistributedIterator.__len__ = patcher_tf_dataloader(m.DistributedIterator.__len__)
    #m.DistributedIterator.__next__ = patcher_tf_dataloader(m.DistributedIterator.__next__)


def hook_tf_iterator(m):
    m.OwnedIterator.__init__ = patcher_tf_dataloader(inst.handle_tf_dl_init, m.OwnedIterator.__init__)
    m.OwnedIterator.__iter__ = patcher_tf_dataloader(inst.handle_tf_dl_iter, m.OwnedIterator.__iter__)
    m.OwnedIterator.__next__ = patcher_tf_dataloader(inst.handle_tf_dl_next, m.OwnedIterator.__next__)

def hook_tf_custom_iterator(m):
    m.OwnedIterator.__init__ = patcher_tf_dataloader(inst.handle_tf_dl_init, m.OwnedIterator.__init__)
    m.OwnedIterator.__iter__ = patcher_tf_dataloader(inst.handle_tf_dl_iter, m.OwnedIterator.__iter__)
    m.OwnedIterator.__next__ = patcher_tf_dataloader(inst.handle_tf_dl_next_mini, m.OwnedIterator.__next__)


def hook_func(m, attr, patch_func):
    if attr!="":
        target, mod, target_attr = find_attr(m, attr)
        if target is not None:
            target = patcher_tf_dataloader(patch_func, target)
            setattr(mod, target_attr, target)

"""
def hook_tf_custom_dl(m, attr, patch_func):
    if (type(attr) is list) and len(attr)==3:
        target,mod,target_attr = find_attr(m, attr[0])
        if (attr[0]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_init, target)
            setattr(mod, target_attr, target)
        target,mod,target_attr = find_attr(m, attr[1])
        if (attr[1]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_iter, target)
            setattr(mod, target_attr, target)
        target,mod,target_attr = find_attr(m, attr[2])
        if (attr[2]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_next, target)
            setattr(mod, target_attr, target)

def hook_tf_custom_dl_no_persist(m, attr, patch_func):
    if (type(attr) is list) and len(attr)==3:
        target,mod,target_attr = find_attr(m, attr[0])
        if (attr[0]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_init, target)
            setattr(mod, target_attr, target)
        target,mod,target_attr = find_attr(m, attr[1])
        if (attr[1]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_iter, target)
            setattr(mod, target_attr, target)
        target,mod,target_attr = find_attr(m, attr[2])
        if (attr[2]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_next_no_persist, target)
            setattr(mod, target_attr, target)

def hook_tf_custom_loop_dl(m, attr, patch_func):
    if (type(attr) is list) and len(attr)==3:
        attr0,mod,target_attr = find_attr(m, attr[0])
        if (attr[0]!="") and (attr0 is not None):
            new_attr = patch_func(inst.handle_tf_dl_init, attr0)
            setattr(mod, target_attr, new_attr)
        attr1,mod,target_attr = find_attr(m, attr[1])
        if (attr[1]!="") and (attr1 is not None):
            atrr1 = patch_func(inst.handle_tf_dl_iter, attr1)
            setattr(mod, target_attr, attr1)
        attr2,mod,target_attr = find_attr(m, attr[2])
        if (attr[2]!="") and (attr2 is not None):
            attr2 = patch_func(inst.handle_tf_dl_next_phony, attr2)
            setattr(mod, target_attr, attr2)

def hook_tf_custom_loop_dl_no_persist(m, attr, patch_func):
    if (type(attr) is list) and len(attr)==3:
        target,mod,target_attr = find_attr(m, attr[0])
        if (attr[0]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_init, target)
            setattr(mod, target_attr, target)
        target,mod,target_attr = find_attr(m, attr[1])
        if (attr[1]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_iter, target)
            setattr(mod, target_attr, target)
        target,mod,target_attr = find_attr(m, attr[2])
        if (attr[2]!="") and (target is not None):
            target = patch_func(inst.handle_tf_dl_next_phony_no_persist, target)
            setattr(mod, target_attr, target)
"""

def hook_tf_dataset(m):
    m.DatasetV2.__iter__ = patcher_tf_dataloader(inst.handle_tf_dl_iter, m.DatasetV2.__iter__)


def hook_tf_eager(m):
    m.Strategy.experimental_run_v2 = patcher_tf_eager(m.Strategy.experimental_run_v2)


def hook_tf_static(m):
    #m.Function.__init__ = patcher_tf_static(m.Function.__init__)
    m.Function.__call__ = patcher_tf_static(m.Function.__call__)


def hook_tf_wrapper(m, *args):
    if len(args)<2:
        return
    attr = args[0]
    patch_func = args[1]
    remaining_args = []
    for i in range(2, len(args)):
        remaining_args.append(args[i])
    target,mod,target_attr = find_attr(m, attr)
    if target is not None:
        if len(remaining_args)>0:
            target = patch_func(target, remaining_args)
            setattr(mod, target_attr, target)
        else:
            target = patch_func(target)
            setattr(mod, target_attr, target)
    else:
        print("CANNOT find the target module", m, attr)


def hook_tf_norm(m):
    m.BatchNormalizationBase.__init__ = patcher_tf_norm(m.BatchNormalizationBase.__init__)


def hook_tf_datasets(m):
    m.StreamingFilesDataset.__init__ = patcher_tf_datasets(m.StreamingFilesDataset.__init__)

def hook_tf_core(m):
    m.Dropout.__init__ = patcher_tf_dropout(m.Dropout.__init__)


def hook_tf_optim(m):
    m._deduplicate_indexed_slices = patcher_tf_deduplicate(m._deduplicate_indexed_slices)
    m.OptimizerV2.__init__ = patcher_optim(m.OptimizerV2.__init__)


def hook_tf_sgd(m):
    m.SGD.__init__ = patcher_tf_sgd(m.SGD.__init__)


def hook_tf_loss(m):
    m.Loss.__call__ = patcher_tf_loss(m.Loss.__call__)


def hook_tf_nnops(m):
    m.sparse_softmax_cross_entropy_with_logits = patcher_tf_ssce(m.sparse_softmax_cross_entropy_with_logits)
    m.softmax_cross_entropy_with_logits = patcher_tf_sce(m.softmax_cross_entropy_with_logits)


def hook_tf_gradsfunc(m):
    m._gradient_registry.lookup = patcher_tf_gradfunc(m._gradient_registry.lookup)


def hook_tf_gradvspace(m):
    m.VSpace.__new__ = patcher_tf_gradvspace(m.VSpace.__new__)


def hook_tf_imageops(m):
    m.resize_nearest_neighbor = patcher_tf_imageops(m.resize_nearest_neighbor)
    m.resize_nearest_neighbor_grad = patcher_tf_imageops(m.resize_nearest_neighbor_grad)
    m.resize_bilinear = patcher_tf_imageops(m.resize_bilinear)
    m.sample_distorted_bounding_box_v2 = patcher_tf_sample(m.sample_distorted_bounding_box_v2)

def hook_tf_randops(m):
    m.random_uniform = patcher_tf_randops(m.random_uniform)

def hook_tf_multi_device_iter(m):
    m._create_device_dataset = patch_tf_multi_device_iter(m._create_device_dataset)

''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''

def patcher_torchvision(function):
    def wrapper(*args, **kwargs):
        ret = function(*args, **kwargs)
        return ret

    return wrapper

def hook_torchvision(m):
    #m.CIFAR10.__getitem__ = patcher_torchvision(m.CIFAR10.__getitem__)
    m.Compose.__call__ = patcher_torchvision(m.Compose.__call__)

""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""

hook_modules1 = {
           #'tensorflow.python.keras.optimizer_v2.optimizer_v2': (hook_tf_optim, []),
           #'tensorflow.python.keras.optimizer_v2.gradient_descent': (hook_tf_sgd, []),
           'tensorflow.python.framework.ops': (hook_tf_gradsfunc, []),
           #'tensorflow.python.eager.imperative_grad': (hook_tf_gradvspace, []),
        }

hook_modules = {
           'torch.utils.data.dataloader' : (hook_torch_dataloader, []),
           'torch.optim.lr_scheduler' : (hook_torch_scheduler, []),
           'torch.optim.optimizer' : (hook_torch_optim, []),
           'torch.nn.modules.module' : (hook_torch_module, []),

           'tensorflow.python.keras.engine.training': (hook_tf_module, []),

           #'tensorflow.python.ops.gen_random_ops': (hook_tf_randops, []),
           'tensorflow.python.data.ops.multi_device_iterator_ops': (hook_tf_multi_device_iter, []),
           'tensorflow.python.ops.gen_nn_ops': (hook_tf_nnops, []),
           'tensorflow.python.ops.gen_image_ops': (hook_tf_imageops, []),
           'tensorflow.python.keras.layers.normalization': (hook_tf_norm, []),
           'tensorflow.python.keras.layers.core': (hook_tf_core, []),
           #'tensorflow.python.keras.losses': (hook_tf_loss, []),
           'tensorflow.python.keras.optimizer_v2.optimizer_v2': (hook_tf_optim, []),
           'tensorflow.python.keras.optimizer_v2.gradient_descent': (hook_tf_sgd, []),

           'tensorflow.python.framework.ops': (hook_tf_gradsfunc, []),
           'tensorflow.python.eager.imperative_grad': (hook_tf_gradvspace, []),

           #'torchvision.datasets.cifar': (hook_torchvision, []),
           #'torchvision.transforms.transforms': (hook_torchvision, []),
        }


def hook_main_loop():
    if 'CKPT_DIR' not in os.environ:
        return

    root_dir = os.environ['CKPT_DIR']
    inst_file = os.path.join(root_dir, "inst.info")
    if not os.path.exists(inst_file):
        return
    
    #import pickle
    with open(inst_file, 'r') as f:
        #inst_info = pickle.load(f)
        inst_info = eval(f.read())

    """
    inst_info["dataloader"]["full_name"] # class object
    inst_inft["dataloader"]["iter"]
    inst_inft["dataloader"]["next"]
    inst_inft["dataloader"]["module"]
    inst_inft["dataloader"]["persistent"]
    
    inst_inft["loop_type"]["type"]
    
    inst_inft["code_info"]["dl_call_file_name"]
    inst_inft["code_info"]["dl_call_line_no"]
    inst_inft["code_info"]["update_call_file_name"]
    inst_inft["code_info"]["update_call_line_no"]
    """
    
    hook_dl_func = {}

    if inst_info["loop_type"]["type"] == "ON_DATASET":
        if inst_info["dataloader"]["iter"][1] in ["tensorflow.python.data.ops.iterator_ops.OwnedIterator.__iter__"]:
            hook_modules['tensorflow.python.data.ops.iterator_ops'] = (hook_tf_iterator, [])
            # FIXME need to determine whether we need to instrument
            #hook_modules['tensorflow.python.distribute.input_lib'] = (hook_tf_dataloader, [])
            #hook_modules['tensorflow.python.data.ops.dataset_ops'] = (hook_tf_dataset, [])
        else:
            # custom dataloader, but it still needs to follow the standard APIs
            #hook_dl_func = hook_tf_custom_dl if inst_info["dataloader"]["persistent"] else hook_tf_custom_dl_no_persist
            hook_dl_func["init"] = inst.handle_tf_dl_init
            hook_dl_func["iter"] = inst.handle_tf_dl_iter
            hook_dl_func["next"] = inst.handle_tf_dl_next if inst_info["dataloader"]["persistent"] else inst.handle_tf_dl_next_no_persist

    elif inst_info["loop_type"]["type"] == "ON_CUSTOM":
        """If the dataloader can not be saved, we fastforward the training"""
        #hook_modules['tensorflow.python.distribute.distribute_lib'] = (hook_tf_eager)
        #hook_modules['tensorflow.python.eager.def_function'] = (hook_tf_static)
        dl_persist = 1 if inst_info["dataloader"]["persistent"] else 0
        if inst_info["dataloader"]["next"][1] in ["tensorflow.python.data.ops.iterator_ops.OwnedIterator.__next__"]:
            hook_modules['tensorflow.python.data.ops.iterator_ops'] = (hook_tf_custom_iterator, [])
            dl_persist = 2
        else:
            # custom dataloader, but it still needs to follow the standard APIs
            #hook_dl_func = hook_tf_custom_loop_dl if inst_info["dataloader"]["persistent"] else hook_tf_custom_loop_dl_no_persist
            hook_dl_func["init"] = inst.handle_tf_dl_init
            hook_dl_func["iter"] = inst.handle_tf_dl_iter
            hook_dl_func["next"] = inst.handle_tf_dl_next_phony if inst_info["dataloader"]["persistent"] else inst.handle_tf_dl_next_phony_no_persist

        # instrument source code
        if ("wrapper_module" in inst_info["code_info"]) and (inst_info["code_info"]["wrapper_module"]!=""):
            if inst_info["dataloader"]["persistent"]:
                hook_modules[inst_info["code_info"]["wrapper_module"]] = (hook_tf_wrapper, [(inst_info["code_info"]["wrapper_func"], patcher_tf_wrapper, inst_info["dataloader"]["full_name"])])
            else:
                hook_modules[inst_info["code_info"]["wrapper_module"]] = (hook_tf_wrapper, [(inst_info["code_info"]["wrapper_func"], patcher_tf_wrapper_no_persistent)])
        else:
            # the checkpoint place is after batch generation. 
            # instrument dl next function when it is not in the same wrapper as weights
            inst_code(inst_info["code_info"], dl_persist)

    # Hook data loader
    if len(hook_dl_func) > 0:
        if ("init" in inst_info["dataloader"]) and ("init" in hook_dl_func):
            add_hook_function(hook_modules, inst_info["dataloader"]["init"][0], hook_func, inst_info["dataloader"]["init"][1], hook_dl_func["init"])
        if ("iter" in inst_info["dataloader"]) and ("iter" in hook_dl_func):
            add_hook_function(hook_modules, inst_info["dataloader"]["iter"][0], hook_func, inst_info["dataloader"]["iter"][1], hook_dl_func["iter"])
        if ("next" in inst_info["dataloader"]) and ("next" in hook_dl_func):
            add_hook_function(hook_modules, inst_info["dataloader"]["next"][0], hook_func, inst_info["dataloader"]["next"][1], hook_dl_func["next"])


    print(hook_modules)

def add_hook_function(hook_info, module, hook_func, attr, patch_func):
    if module not in hook_info:
        # TODO: XXZ
        # hook_info[module] = (hook_func, [(attr, patch_func)])
        hook_info[module] = [(hook_func, attr, patch_func)]
    else:
        hook_info[module][1].append((hook_func, attr, patch_func))

def backup_file(file_name):
    # backup source code
    if os.path.exists(file_name+".orig"):
        i=0
        while os.path.exists(file_name+".orig"+str(i)):
            i += 1
        copyfile(file_name+".orig", file_name+".orig"+str(i))
    print("Copy from %s to %s"%(file_name, file_name+".orig"))
    copyfile(file_name, file_name+".orig")
    with open(os.path.join(os.environ['CKPT_DIR'], "inst.file"), 'a+') as info_file:
        info_file.write(file_name+"\n")
        print("save file name", file_name)
    return True
    

def insert_import_lib(file_name, lines, inst_import, file_info):
    # insert the import statement
    in_comments = False
    for i, line in enumerate(lines):
        # check comments
        if line.strip().startswith("\"\"\"") \
                or line.strip().startswith("'''") \
                or line.strip().endswith("\"\"\"") \
                or line.strip().endswith("'''"):
            in_comments = not in_comments
        # not in comments and after import other libraries 
        elif (not in_comments) and (line.strip()!="") \
                and (not line.strip().startswith("#")) \
                and (not (line.strip().startswith("import") or line.strip().startswith("from"))):
            record_changes(file_info, file_name, i, 1)
            lines.insert(i, inst_import)
            break

def inst_code(code_info, dl_persist):
    """
    dl_persist: 0: cannot be saved, 1: can be saved, 2: can be saved by API 
    """

    file_info = {}
    
    epoch_file_name = code_info["epoch_file_name"]
    epoch_lineno = (int(code_info["epoch_line_no"]) - 1) if code_info["epoch_line_no"]!="" else 0
    loop_file_name = code_info["loop_file_name"]
    loop_lineno = (int(code_info["loop_line_no"]) - 1) if code_info["loop_line_no"]!="" else 0
    dl_file_name = code_info["dl_call_file_name"]
    dl_lineno = (int(code_info["dl_call_line_no"]) - 1) if code_info["dl_call_line_no"]!="" else 0

    inst_import = "import detrain.instrumentor as inst\n"
    inst_epoch = ["inst.handle_tf_dl_iter(inst, inst)\n"]
    inst_dl_prefix = [
            "det_exe, det_value = inst.handle_tf_ckpt_phony(inst, inst, " + str(dl_persist==1) + ")\n", 
            "if not det_exe:\n",
        ]
    if loop_file_name==dl_file_name:
        inst_dl_prefix.append("    continue\n")
    else:
        inst_dl_prefix.append("    return det_value\n")

    if not backup_file(dl_file_name):
        return

    with open(dl_file_name, encoding='utf-8') as f:
        lines = f.readlines()

    record_changes(file_info, dl_file_name, dl_lineno, len(inst_dl_prefix))
    dl_lineno = adjust_lines(file_info, dl_file_name, dl_lineno)
    padding = lines[dl_lineno][0:len(lines[dl_lineno])-len(lines[dl_lineno].lstrip(" \t"))] 
    if dl_persist == 1:
        # insert after
        dl_lineno += 1
    for code in inst_dl_prefix:
        lines.insert(dl_lineno, padding+code)
        dl_lineno += 1

    # insert the import statement
    insert_import_lib(dl_file_name, lines, inst_import, file_info)

    with open(dl_file_name, "w", encoding='utf-8') as f:
        f.writelines(lines)

    # save information for each epoch
    if epoch_file_name!="":
        # check whether the original file has been saved
        backup_file(epoch_file_name)

        with open(epoch_file_name, encoding='utf-8') as f:
            lines = f.readlines()
        
        epoch_lineno = adjust_lines(file_info, epoch_file_name, epoch_lineno)
        # after loop
        epoch_lineno += 1 
        while lines[epoch_lineno].strip() == "":
            epoch_lineno += 1 

        record_changes(file_info, epoch_file_name, epoch_lineno, len(inst_epoch))
        padding = lines[epoch_lineno][0:len(lines[epoch_lineno])-len(lines[epoch_lineno].lstrip(" \t"))] 
        for code in inst_epoch:
            lines.insert(epoch_lineno, padding+code)
            epoch_lineno += 1

        if dl_file_name!=epoch_file_name:
            insert_import_lib(epoch_file_name, lines, inst_import, file_info)

        with open(epoch_file_name, "w", encoding='utf-8') as f:
            f.writelines(lines)

def record_changes(file_info, file_name, line_no, lines):
    if file_name not in file_info:
        file_info[file_name] = []
    file_info[file_name].append((line_no, lines))

def adjust_lines(file_info, file_name, line_no):
    offset = 0
    if file_name in file_info:
        for i in range(len(file_info[file_name])):
            if file_info[file_name][i][0] < line_no:
                offset += file_info[file_name][i][1]

    return line_no + offset

def hook(name, module):
    print(name)
    module_info = hook_modules.get(name)
    if module_info is not None:
        # TODO XXZ
        if 'list' in str(type(module_info)):            
            for entry in module_info:
                func = entry[0]
                func(entry[1:])
        else:
            func = module_info[0]
            func(module)
        # func = module_info[0]
        # func_list = module_info[1]
        # if len(func_list)>0:
        #     for args in func_list:
        #         # args should contain ata least two elements: function and a patch function
        #         if len(args)>2:
        #             func(module, args[0], args[1], args[2:])
        #         else:
        #             func(module, args[0], args[1])
        # else:
        #     func(module)


def restore():
    global WRAPPERS
    WRAPPERS = []

