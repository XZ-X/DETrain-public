import os, sys
import time, threading
import pdb

models = []
optim = []
sched = []
dataloader = []
iter_counter = 0
iter_epoch = -1
saved_epoch = -1
saved_rng_state = None
rnd = None

#WAIT_SECONDS = 5
#WAIT_SECONDS = 15*60
WAIT_SECONDS = 5*60*60
do_checkpoint = False
timer_thread = None

def reset_checkpoint():
    global do_checkpoint
    global timer_thread

    do_checkpoint = True
    timer_thread = threading.Timer(WAIT_SECONDS, reset_checkpoint)
    timer_thread.daemon = True
    timer_thread.start()


""" handle functions """
def handle_module(m, name):
    if not ("nn.modules.loss" in str(type(m).__bases__) 
            or "nn.modules.activation" in str(type(m))):
        models.append(m)


def handle_optim(m, name):
    optim.append(m)


def handle_sched(m, name):
    sched.append(m)


def print_model():
    print("*****MODELS*****")
    for m in models:
        print(type(m))
    print("*****OPTIM*****")
    print(optim)
    print("*****SCHED*****")
    print(sched)
    print("*****DATALOADER*****")
    #print(dataloader)
    for d in dataloader:
        print(d, sys.getrefcount(d))
        #print(d, d._iterator_resource)
    print(iter_counter)

