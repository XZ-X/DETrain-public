import os, sys, pdb
import time, threading
from .ckpt_data import *

in_tf_function = False
has_dropout = False
model_and_optim = {}

def remove_tf_child():
    global models
    model = []
    for m in models:
        is_child = False
        for sub_m in models:
            if m!=sub_m and ((m in sub_m.layers)
                    or set(sub_m.layers).issuperset(set(m.layers))):
                is_child = True
                break
        if not is_child:
            model.append(m)
    models.clear()
    models.extend(model)

def save_checkpoint(rnd, epoch, iters):
    import detrain.ckpt_manager as ckpt
    import pickle
    import random
    import numpy as np
    from tensorflow.python.framework import ops
    from tensorflow.python.training.tracking import base as trackable
    from tensorflow.python.eager import context

    global model_and_optim

    path = ckpt.ckpt_dir + "/checkpoint_" + str(epoch) + "_" + str(iters)

    match_model_optims()
    model_and_optim.write(path + "/models_optims")

    g_seed = ops.get_default_graph().seed
    l_seed = ops.get_default_graph()._last_id
    rng_seed = random.getstate()
    np_rng_seed = np.random.get_state()
    ctx = context.context()
    internal_rng = getattr(ctx, "_rng", None)
    internal_seed = internal_rng.getstate() if internal_rng!=None else None
    ckpt_data = {
        'epoch' : epoch,
        'iters' : iters,
        'epoch_rnd': rnd,
        'g_seed': g_seed,
        'l_seed': l_seed,
        'rng_seed': rng_seed,
        'np_rng_seed': np_rng_seed,
        'internal_seed': internal_seed,
    }
    with open(os.path.join(path, "./aux_data"), 'wb+') as seed_file:
        pickle.dump(ckpt_data, seed_file)
    
    ckpt.save_checkpoint("tensorflow", path)
    return path

def set_save_flags(path, has_dropout):
    os.environ["TF_CHECKPOINT_ITER_CMD"] = '2'
    iter_path = path+"/iters/"
    if not os.path.exists(iter_path):
        os.makedirs(iter_path)
    os.environ["TF_CHECKPOINT_ITER_SAVEPATH"] = iter_path

    os.environ["TF_CHECKPOINT_DEVICE_ITER_CMD"] = '2'
    os.environ["TF_CHECKPOINT_MULTI_DEVICE_ITER_CMD"] = '2'
    os.environ["TF_CHECKPOINT_DEVICE_ITER_SAVEPATH"] = iter_path
            
    os.environ["TF_CHECKPOINT_ITREATION_CMD"] = '2'
    os.environ["TF_CHECKPOINT_ITERATION_SAVEPATH"] = iter_path

    if has_dropout:
        os.environ["TF_CHECKPOINT_RNG_CMD"] = '2'
        os.environ["TF_CHECKPOINT_RNG_STATE"] = path+"/rng_state"

def set_restore_flags(path):
    """
    os.environ["TF_CHECKPOINT_ITER_CMD"] = '1'
    iter_path = path+"/iters/"
    os.environ["TF_CHECKPOINT_ITER_SAVEPATH"] = iter_path
    os.environ["TF_CHECKPOINT_DEVICE_ITER_CMD"] = '1'
    os.environ["TF_CHECKPOINT_MULTI_DEVICE_ITER_CMD"] = '1'
    os.environ["TF_CHECKPOINT_DEVICE_ITER_SAVEPATH"] = iter_path
    """
    os.environ["TF_CHECKPOINT_ITREATION_CMD"] = '1'
    iter_path = path+"/iters/"
    os.environ["TF_CHECKPOINT_ITERATION_SAVEPATH"] = iter_path

def save_dataloader(path):
    dl = dataloader[0]
    import pickle
    with open(os.path.join(path, "custom_data_loader"), 'wb') as f:
        pickle.dump(dl, f)

def handle_tf_dl_init(m, func):
    dataloader.append(m)
    #print(m.__len__())
    return True, ""


def handle_tf_dl_iter(m, func):
    from tensorflow.python.framework import ops
    from tensorflow.python.eager import context
    import detrain.ckpt_manager as ckpt
    global iter_counter
    global iter_epoch
    global rnd
    global do_checkpoint
    global saved_epoch
    global saved_rng_state

    iter_epoch += 1
    iter_counter = 0

    """
    Epoch Rng
    """
    ctx = context.context()
    internal_rng = getattr(ctx, "_rng", None)
    internal_seed = internal_rng.getstate() if internal_rng!=None else None
    g_seed = ops.get_default_graph().seed
    l_seed = ops.get_default_graph()._last_id
    rnd = [g_seed, l_seed, internal_seed]

    if ckpt.from_checkpoint():
        checkpoint = ckpt.last_checkpoint
        if checkpoint!=None:
            saved_epoch = checkpoint['epoch']
            saved_rng_state = checkpoint['epoch_rnd'][2]
            if iter_epoch <= saved_epoch:
                ctx = context.context()
                internal_rng = getattr(ctx, "_rng", None)
                if internal_rng is not None:
                    internal_rng.setstate(saved_rng_state)

    elif iter_epoch == 0:
        reset_checkpoint()
    
    return True, ""

def handle_tf_dl_next_no_persist(m, func):
    return handle_tf_dl_next(m, func, False)

def handle_tf_dl_next(m, func, persist=True):
    from tensorflow.python.framework import ops
    from tensorflow.python.eager import context
    import detrain.ckpt_manager as ckpt
    global iter_counter
    global iter_epoch
    global rnd
    global do_checkpoint
    global saved_epoch
    global saved_rng_state

    sys.stdout.flush()
    is_restore = False
       
    iter_counter += 1
    print("DL_NEXT", iter_epoch, iter_counter)

    if ckpt.from_checkpoint():
        checkpoint = ckpt.last_checkpoint
        if checkpoint!=None:
            saved_epoch = checkpoint['epoch']
            if iter_epoch < saved_epoch:
                raise StopIteration

            # if bump to the data
            if not persist:
                for _ in range(checkpoint['iters']-1):
                    continue
                    func(m)
            else:
                restore_dataloader(checkpoint)

            print("RESTORE CHECKPOINT", ".......")
                
            is_restore = True

            restore_tf_models(checkpoint['model'])
            restore_tf_random(checkpoint)
            # FIXME custom data loader does not need to restore rngs
            restore_tf_c_random(checkpoint)
                
            reset_checkpoint()

        ckpt.last_checkpoint = None

    elif iter_counter == 1:
        reset_checkpoint()

    #if iter_epoch in [0, 1] and iter_counter in [16]:
    if not is_restore and iter_counter in [1000, 400, 2000]:
        remove_tf_child()
        print_model()
        path = save_checkpoint(rnd, iter_epoch, iter_counter)

        set_save_flags(path, has_dropout)
        if persist:
            save_dataloader(path)

        do_checkpoint = False

    sys.stdout.flush()
    return True, ""

def handle_tf_dl_next_phony_no_persist(m, func):
    return handle_tf_dl_next_phony(m, func, False)

def handle_tf_dl_next_phony(m, func, persist=True):
    import detrain.ckpt_manager as ckpt

    #global dl_iter_counter
    dl_iter_counter = 0
    dl_iter_counter += 1

    if ckpt.from_checkpoint():
        checkpoint = ckpt.last_checkpoint
        if checkpoint!=None:
            data = checkpoint['data_sample'] if "data_sample" in checkpoint else ""
            if not persist:
                full_name = func.__module__ + "." + func.__qualname__
                if full_name=="tensorflow.python.data.ops.iterator_ops.OwnedIterator.__next__":
                    return False, data 
                else:
                    return True, data
            else:
                return False, data

    return True, ""

def handle_tf_ckpt_phony(m, func, dl_persist=False):
    from tensorflow.python.framework import ops
    from tensorflow.python.eager import context
    import detrain.ckpt_manager as ckpt
    global iter_counter
    global iter_epoch
    global rnd
    global do_checkpoint
    global saved_epoch
    global saved_rng_state

    sys.stdout.flush()
    is_restore = False
       
    iter_counter += 1
    print("TF CKPT", iter_counter)

    if ckpt.from_checkpoint():
        checkpoint = ckpt.last_checkpoint
        if checkpoint!=None:
            saved_epoch = checkpoint['epoch']
            if iter_epoch < saved_epoch:
                return False, {'total_loss': [0], 'output_losses': [0], 'metrics': [], 'batch_size': 1}  #checkpoint['sample']
            
            saved_counter = checkpoint['iters']
            if iter_counter < saved_counter:
                return False, {'total_loss': [0], 'output_losses': [0], 'metrics': [], 'batch_size': 1}  #checkpoint['sample']

            is_restore = True
            # TODO: XXZ
            print("XXZ:DBG:Restore...")
            restore_tf_models(checkpoint['model'])
            restore_tf_random(checkpoint)
            # FIXME custom data loader does not need to restore rngs
            restore_tf_c_random(checkpoint)
            # reset data loader
            if dl_persist:
                restore_dataloader(checkpoint)
                
            reset_checkpoint()

        ckpt.last_checkpoint = None

    elif iter_counter == 1:
        reset_checkpoint()

    #if iter_epoch in [0, 1] and iter_counter in [16]:
    if not is_restore and iter_counter in [3, 6]:
    #if iter_counter in [5, 10]:
        remove_tf_child()
        print_model()
        path = save_checkpoint(rnd, iter_epoch, iter_counter)

        set_save_flags(path, has_dropout)
        if dl_persist:
            save_dataloader(path)

        do_checkpoint = False

    sys.stdout.flush()
    return True, ""

def handle_tf_wrapper_func_dl(m, func, args):
    func_name = args[0] if ((type(args) is list) and (len(args)>0)) else ""
    handle_tf_wrapper_func(m, func, func_name)

def handle_tf_wrapper_func(m, func, next_func=""):
    import detrain.ckpt_manager as ckpt

    global iter_counter
    global iter_epoch
    global in_tf_function
    global do_checkpoint
    global timer_thread
    global has_dropout

    iter_counter += 1

    sys.stdout.flush()
    sample = 0
    name = func.__name__
    print("tf_wrapper:", name, iter_counter)
    if in_tf_function:
        return True, sample

    is_restore = False
    if ckpt.from_checkpoint():
        checkpoint = ckpt.last_checkpoint
        if checkpoint!=None:
            saved_counter = checkpoint['iters']
            if iter_counter < saved_counter:
                sample = checkpoint['sample']
                if next_func!="":
                    # FIXME bump to next data
                    real_func = getattr(dataloader[0], next_func, None)
                    if real_func is not None:
                        real_func()
                sys.stdout.flush()
                return False, sample
            
            in_tf_function = True
            is_restore = True

            restore_tf_models(checkpoint['model'])
            restore_tf_random(checkpoint)
            # FIXME eager mode may not need to save rngs in c
            restore_tf_c_random(checkpoint)
            
            in_tf_function = False
            reset_checkpoint()
        
        ckpt.last_checkpoint = None

    elif iter_counter == 1:
        reset_checkpoint()
    
    #if not is_restore and iter_counter in [30, 100, 300, 600, 1000]:
    if not is_restore and iter_counter in [40, 80]:
    #if not is_restore and iter_counter%1000==0:
        remove_tf_child()
        print_model()
        in_tf_function = True
        path = save_checkpoint(rnd, iter_epoch, iter_counter)

        set_save_flags(path, has_dropout)
        if next_func!="":
            save_dataloader(path)

        in_tf_function = False
        do_checkpoint = False
        sample = None
    
    sys.stdout.flush()
    return True, sample

def match_model_optims():
    import tensorflow as tf
    from tensorflow.python.training.tracking import base as trackable

    global models
    global optim
    global model_and_optim

    matched_optim = []
    for model in models:
        optimizer = getattr(model, 'optimizer', None)
        if optimizer != None:
            matched_optim.append(optimizer)

    models_optims = {}
    for i, opt in enumerate(optim):
        if opt not in matched_optim:
            models_optims["optim"+str(i)] = opt

    for i, model in enumerate(models):
        models_optims["model"+str(i)] = model
        #print(hashlib.md5(str(model.trainable_variables).encode('utf-8')).hexdigest())

    model_and_optim = tf.train.Checkpoint(**models_optims)


def restore_tf_models(model_dir):
    from tensorflow.python.training.tracking import base as trackable

    global model_and_optim

    remove_tf_child()
    match_model_optims()

    model_and_optim.restore(model_dir + "/models_optims")


def restore_tf_random(checkpoint):
    import random
    import numpy as np
    from tensorflow.python.framework import ops
    from tensorflow.python.eager import context

    print("***SAVED", checkpoint['g_seed'], checkpoint['l_seed'])
    ops.get_default_graph().seed = checkpoint['g_seed']
    ops.get_default_graph()._next_id_counter = checkpoint['l_seed']
    random.setstate(checkpoint['rng_seed'])
    np.random.set_state(checkpoint['np_rng_seed'])
    ctx = context.context()
    internal_rng = getattr(ctx, "_rng", None)
    if internal_rng is not None:
        internal_rng.setstate(checkpoint['internal_seed'])        

def restore_tf_c_random(checkpoint):
    set_restore_flags(checkpoint['model'])
    if os.path.exists(checkpoint['model']+"/rng_state"):
        os.environ["TF_CHECKPOINT_RNG_CMD"] = '1'
        os.environ["TF_CHECKPOINT_RNG_STATE"] = checkpoint['model']+"/rng_state"

def restore_dataloader(checkpoint):
    if os.path.exists(os.path.join(checkpoint['model'], "custom_data_loader")):
        import pickle
        with open(os.path.join(checkpoint['model'], "custom_data_loader"), 'rb') as f:
            saved_data = pickle.load(f)

        obj_dict = getattr(saved_data, "__dict__", None)
        # FIXME whether there are many dataloaders
        dl_dict = getattr(dataloader[0], "__dict__", None)
        if (obj_dict is not None) and (dl_dict is not None):
            for key, val in obj_dict.items():
                if key in dl_dict:
                    try:
                        dl_dict[key] = val
                    except:
                        print("Can not set", key, val, "for restoring the data loader")
                        pass
        else:
            print("Can not load the saved data loader!!!")

def save_sample(val):
    import detrain.ckpt_manager as ckpt
    ckpt.save_ret_sample(val)

def save_data_sample(val):
    import detrain.ckpt_manager as ckpt
    ckpt.save_data_sample(val)
