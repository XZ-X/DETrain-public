import os
import sys
import pdb
import numpy as np

import detrain.hookfunctions as hooker

#ckpt_dir = "./chkpt"
ckpt_dir = os.environ['CKPT_DIR'] if 'CKPT_DIR' in os.environ else "/tmp/detrain"
is_checkpoint_exists = False
checkpoints = []
last_checkpoint = None
current_checkpoint = None

def save_checkpoint(lib_name, path):
    global current_checkpoint

    with open(os.path.join(ckpt_dir, "config"), 'w+') as f:
        f.write(lib_name)

    print("*******SAVE CHECKPOINT**********")
    print(path)
    current_checkpoint = path
    return path


def restore_last():
    global last_checkpoint
    if last_checkpoint == None:
        with open(os.path.join(ckpt_dir, "config"), 'r') as f:
            libname = f.readline()
        if libname == "pytorch":
            import torch
            last_checkpoint = torch.load(checkpoints[-1])
        elif libname == "tensorflow":
            import pickle
            with open(os.path.join(checkpoints[-1], "./aux_data"), 'rb') as aux_file:
                last_checkpoint = pickle.load(aux_file)
            if os.path.exists(os.path.join(checkpoints[-1], "./ret_sample")):
                with open(os.path.join(checkpoints[-1], "./ret_sample"), 'rb') as ret_file:
                    last_checkpoint['sample'] = pickle.load(ret_file)
            if os.path.exists(os.path.join(checkpoints[-1], "./data_sample")):
                with open(os.path.join(checkpoints[-1], "./data_sample"), 'rb') as s_file:
                    last_checkpoint['data_sample'] = pickle.load(s_file)
            last_checkpoint['model'] = checkpoints[-1]
        
            """
            restore c random number generators
            """
            os.environ["TF_CHECKPOINT_ITER_CMD"] = '1'
            iter_path = checkpoints[-1]+"/iters/"
            os.environ["TF_CHECKPOINT_ITER_SAVEPATH"] = iter_path
            os.environ["TF_CHECKPOINT_DEVICE_ITER_CMD"] = '1'
            os.environ["TF_CHECKPOINT_MULTI_DEVICE_ITER_CMD"] = '1'
            os.environ["TF_CHECKPOINT_DEVICE_ITER_SAVEPATH"] = iter_path

        print("Last checkpoint:", checkpoints[-1])
            

def filename_key(f):
    name = os.path.basename(f)
    val = name.split("_")
    if len(val) == 3:
        return (int(val[1]), int(val[2]))
    return int(val[-1])

def load_checkpoints():
    global is_checkpoint_exists
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        if os.path.exists(os.path.join(ckpt_dir, "config")):
            with open(os.path.join(ckpt_dir, "config"), 'r') as f:
                libname = f.readline()
            if libname == "pytorch":
                ckpt_files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f)) and f!="config"]
            elif libname == "tensorflow":
                ckpt_files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, f))]
            if len(ckpt_files) > 0:
                is_checkpoint_exists = True
                ckpt_files.sort(key=filename_key)
                checkpoints.extend(ckpt_files)
                # load the last checkpoint 
                restore_last()


def add_checkpoints(chkp):
    checkpoints.append(chkp)


def from_checkpoint():
    global is_checkpoint_exists
    return is_checkpoint_exists 


def save_ret_sample(val):
    import pickle
    global current_checkpoint
    with open(os.path.join(current_checkpoint, "./ret_sample"), 'wb+') as ret_file:
        pickle.dump(val, ret_file)


def save_data_sample(val):
    import pickle
    global current_checkpoint
    with open(os.path.join(current_checkpoint, "./data_sample"), 'wb+') as ret_file:
        pickle.dump(val, ret_file)
