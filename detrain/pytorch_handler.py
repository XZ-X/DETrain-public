import os, sys
import time, threading
import pdb
from .ckpt_data import *

def print_pytorch_states(model, optimizer):
    # Print model's state_dict
    print("Model's state_dict:")
    total_number = 0
    for param_tensor in model.state_dict():
        shape = model.state_dict()[param_tensor].size()
        print(param_tensor, "\t", shape)
        total_number += np.prod(shape)
    print("Total parameters:", total_number)
    '''    
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    '''

def save_checkpoint(rnd, epoch, iters):
    import detrain.ckpt_manager as ckpt
    import torch

    global models
    global optim
    global sched

    model_dicts = [m.state_dict() for m in models]
    optim_dicts = [o.state_dict() for o in optim]
    sched_dicts = [s.state_dict() for s in sched]
     
    checkpoint = {
        'epoch' : epoch,
        'iters' : iters,
        'state_dict': model_dicts,
        'optimizer': optim_dicts,
        'sched': sched_dicts,
        'epoch_rnd': rnd,
        'random': torch.get_rng_state(),
        'cuda_random': torch.cuda.get_rng_state(next(model.parameters()).device)
    }
    path = ckpt.ckpt_dir + "/checkpoint_" + str(epoch) + "_" + str(iters)
    torch.save(checkpoint, path)

    ckpt.save_checkpoint("pytorch", path)
    return path

def remove_pytorch_child():
    global models
    model = []
    for m in models:
        is_child = False
        for sub_m in models:
            if m!=sub_m and (m in sub_m.children()):
                is_child = True
                break
        if not is_child:
            if not ("nn.modules.loss" in str(type(m).__bases__) 
                    or "nn.modules.activation" in str(type(m))):
                model.append(m)
    models.clear()
    models.extend(model)


def handle_pytorch_dataloader(m, func):
    import torch
    import detrain.ckpt_manager as ckpt
    global iter_counter
    global iter_epoch
    global rnd
    global do_checkpoint

    print("iter_epoch", iter_epoch)
    name = func.__name__
    if "init" in name:
        dataloader.append(m)
    
    if "iter" in name:
        # skip eval steps
        if len(models)!=0 and models[0].training:
            iter_counter = 0
            iter_epoch += 1
            rnd = torch.get_rng_state()
        if ckpt.from_checkpoint():
            checkpoint = ckpt.last_checkpoint
            if checkpoint!=None:
                torch.set_rng_state(checkpoint['epoch_rnd'])
        elif iter_epoch == 0:
            reset_checkpoint()
    elif "next" in name:
        if ckpt.from_checkpoint():
            checkpoint = ckpt.last_checkpoint
            if checkpoint!=None:
                # skip eval steps
                if len(models)==0 or not models[0].training:
                    raise StopIteration

                saved_epoch = checkpoint['epoch']
                if iter_epoch < saved_epoch:
                    raise StopIteration
                
                print("RESTORE CHECKPOINT.......")
                iter_counter = checkpoint['iters'] - 1
                for _ in range(iter_counter):
                    func(m)
                
                remove_pytorch_child()
                torch.set_rng_state(checkpoint['random'])
                torch.cuda.set_rng_state(checkpoint['cuda_random'], next(models[0].parameters()).device)

                for i,m in enumerate(models):
                    m.load_state_dict(checkpoint['state_dict'][i])

                for i,o in enumerate(optim):
                    o.load_state_dict(checkpoint['optimizer'][i])

                if len(sched) > 0:
                    for i,s in enumerate(sched):
                        s.load_state_dict(checkpoint['sched'][i])

                reset_checkpoint()

            ckpt.last_checkpoint = None

        iter_counter += 1
        if iter_counter in [10]:
        #if do_checkpoint and iter_counter%10==0:
        #if iter_epoch in [1] and iter_counter==20:
            if timer_thread:
                timer_thread.cancel()
            remove_pytorch_child()
            print_model()
            save_checkpoint(rnd, iter_epoch, iter_counter)
            reset_checkpoint()
            do_checkpoint = False
        if iter_counter == 11:
        #    raise StopIteration
            exit()
        #if iter_epoch == 3:
        #    exit()

