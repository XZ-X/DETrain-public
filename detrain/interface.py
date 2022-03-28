import os, sys, pdb, atexit




def train_loop_start():
  print("Trace begin")
  import detrain.tracer as t
  t.begin = True

def train_loop_end():
  import detrain.tracer as t
  t.begin = False

def enter_main():
  import detrain.tracer as t    
  t.init()
  atexit.register(t.stop_trace)

def ignore_start():
  import detrain.tracer as t
  t.begin = False

def ignore_end():
  import detrain.tracer as t
  t.begin = True

def checkpoint(vars):
  import detrain.tracer as t
  t.hit_checkpoint(vars)
  

def hint(define, use):
  pass