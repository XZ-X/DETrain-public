
import os, threading, time, pdb, hashlib
import detrain.rwlock as rwlock

try:
    import queue
except ImportError:
    import Queue as queue

np_seed_func = None
g_lock = threading.Lock()
log_buffer=[]

rand_lock = rwlock.RWLockRead()
#rand_lock = threading.Lock()
rngs = {}
tmp_rngs = []

def get_rng(thread_id):
    import numpy as np
    global rand_lock
    global rngs 
    with rand_lock.gen_rlock():
        if thread_id in rngs:
            return rngs[thread_id]

def set_rng(thread_id, event_id=-1):
    import numpy as np
    global rand_lock
    global rngs 
    with rand_lock.gen_wlock():
        if event_id != -1:
            event_id = event_id % (2**31 - 1)
            rngs[thread_id] = np.random.RandomState(event_id)

def remove_rng(thread_id):
    global rand_lock
    global rngs 
    global tmp_rngs 
    with rand_lock.gen_wlock():
        tmp_rngs.append(rngs.pop(thread_id, None))

def add_log(msg, flush=False):
    return
    global g_lock
    global log_buffer

    g_lock.acquire()
    try:
        log_buffer.append(msg)
        if len(log_buffer) > 100000 or flush:
            with open("./detrain.pyrandom."+str(os.getpid()), 'a+') as f:
                for log in log_buffer:
                    f.write(log)
                    f.write("\n")
                log_buffer.clear()
    finally:
        g_lock.release()

#import atexit
#atexit.register(add_log, "EXIT", flush=True)

def createIdByArgs(args):
    args = list(sum(args, ()))
    size = len(args)
    bits = int(32/size)
    key = 4141
    for i in range(size):
        try:
            arg = int(args[size-i-1]) # reverse order
            key += (arg << (i*bits))
        except Exception:
            pass
    return key

def my_worker(inqueue, outqueue, initializer=None, initargs=(), maxtasks=None,
           wrap_exception=False):
    assert maxtasks is None or (type(maxtasks) == int and maxtasks > 0)
    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()

    if initializer is not None:
        initializer(*initargs)

    current_thread = threading.current_thread()
    completed = 0
    while maxtasks is None or (maxtasks and completed < maxtasks):
        try:
            task = get()
        except (EOFError, OSError):
            print('worker got EOFError or OSError -- exiting')
            break

        if task is None:
            #print('worker got sentinel -- exiting')
            add_log("EXIT:"+str(current_thread.ident), flush=True)
            break

        job, i, func, args, kwds = task
        #print("\nJOB", job, args[0], args[1], len(args), type(args[0]))
        curr_id = createIdByArgs(args)
        #print("\nJOBID", args, curr_id)
        start_time = time.time()
        add_log("{0} {1} {2} {3} {4}".format("START:", start_time, current_thread.ident, job, func.__name__))
        try:
            set_rng(current_thread.ident, curr_id)
            result = (True, func(*args, **kwds))
            #md5 = hashlib.md5(str(result).encode('utf-8')).hexdigest()
        except Exception as e:
            if wrap_exception and func is not _helper_reraises_exception:
                e = ExceptionWithTraceback(e, e.__traceback__)
            result = (False, e)
        
        end_time = time.time()
        #add_log("{0} {1} {2} {3} {4} {5}".format("END:", end_time, current_thread.ident, job, func.__name__, md5))
        add_log("{0} {1} {2} {3} {4}".format("END:", end_time, current_thread.ident, job, func.__name__))
        remove_rng(current_thread.ident)

        try:
            put((job, i, result))
        except Exception as e:
            wrapped = MaybeEncodingError(e, result[1])
            print("Possible encoding error while sending result: %s" % (
                wrapped))
            put((job, i, (False, wrapped)))

        task = job = result = func = args = kwds = None
        completed += 1
    #util.debug('worker exiting after %d tasks' % completed)
    #print('worker exiting after', completed, 'tasks')

def patcher_nprandom(function):
    def wrapper(*args, **kwargs):
        if function.__name__!="seed":
            rng = get_rng(threading.current_thread().ident)
            if rng != None: 
                func = getattr(rng, function.__name__)
                if func != None:
                    ret = func(*args, **kwargs)
                    add_log("{0} {1} {2} {3} {4}".format(
                        #"RANDOM:", time.time(), threading.current_thread().ident, function.__name__, id(rng)))
                        "RANDOM:", time.time(), threading.current_thread().ident, function.__name__, id(function.__self__)))
                    return ret
            
            #print("GLOBALRANDOM", function.__name__, flush=True)
            ret = function(*args, **kwargs)
            return ret

    return wrapper

def patcher_pyrandom(function):
    def wrapper(*args, **kwargs):
        if function.__name__!="seed":
            #add_log("{0} {1} {2} {3} {4}".format(
            #    "PYRANDOM:", time.time(), threading.current_thread().ident, function.__name__, id(function.__self__)))
            #print("PYRANDOM", function.__name__)
            ret = function(*args, **kwargs)
            return ret

    return wrapper

def hook_threadpool(m):
    import itertools
    m.worker = my_worker 
    m.job_counter = itertools.count(0) 

def hook_keras_iterator(m):
    m.Iterator.__getitem__ = patcher_iterator(m.Iterator.__getitem__) 

def hook_np_random(m):
    global np_seed_func
    #import patch_builtin
    #patch_builtin.patch(m.RandomState, "randint", patcher_nprandom(m.RandomState.randint))
    m.beta = patcher_nprandom(m.beta)
    m.binomial = patcher_nprandom(m.binomial)
    m.bytes = patcher_nprandom(m.bytes)
    m.chisquare = patcher_nprandom(m.chisquare)
    m.choice = patcher_nprandom(m.choice)
    m.dirichlet = patcher_nprandom(m.dirichlet)
    m.exponential = patcher_nprandom(m.exponential)
    m.f = patcher_nprandom(m.f)
    m.gamma = patcher_nprandom(m.gamma)
    m.get_state = patcher_nprandom(m.get_state)
    m.geometric = patcher_nprandom(m.geometric)
    m.gumbel = patcher_nprandom(m.gumbel)
    m.hypergeometric = patcher_nprandom(m.hypergeometric)
    m.laplace = patcher_nprandom(m.laplace)
    m.logistic = patcher_nprandom(m.logistic)
    m.lognormal = patcher_nprandom(m.lognormal)
    m.logseries = patcher_nprandom(m.logseries)
    m.multinomial = patcher_nprandom(m.multinomial)
    m.multivariate_normal = patcher_nprandom(m.multivariate_normal)
    m.negative_binomial = patcher_nprandom(m.negative_binomial)
    m.noncentral_chisquare = patcher_nprandom(m.noncentral_chisquare)
    m.noncentral_f = patcher_nprandom(m.noncentral_f)
    m.normal = patcher_nprandom(m.normal)
    m.pareto = patcher_nprandom(m.pareto)
    m.permutation = patcher_nprandom(m.permutation)
    m.poisson = patcher_nprandom(m.poisson)
    m.power = patcher_nprandom(m.power)
    m.rand = patcher_nprandom(m.rand)
    m.randint = patcher_nprandom(m.randint)
    m.randn = patcher_nprandom(m.randn)
    m.random = patcher_nprandom(m.random)
    m.random_integers = patcher_nprandom(m.random_integers)
    m.random_sample = patcher_nprandom(m.random_sample)
    m.rayleigh = patcher_nprandom(m.rayleigh)
    np_seed_func = m.seed
    m.seed = patcher_nprandom(m.seed)
    m.set_state = patcher_nprandom(m.set_state)
    m.shuffle = patcher_nprandom(m.shuffle)
    m.standard_cauchy = patcher_nprandom(m.standard_cauchy)
    m.standard_exponential = patcher_nprandom(m.standard_exponential)
    m.standard_gamma = patcher_nprandom(m.standard_gamma)
    m.standard_normal = patcher_nprandom(m.standard_normal)
    m.standard_t = patcher_nprandom(m.standard_t)
    m.triangular = patcher_nprandom(m.triangular)
    m.uniform = patcher_nprandom(m.uniform)
    m.vonmises = patcher_nprandom(m.vonmises)
    m.wald = patcher_nprandom(m.wald)
    m.weibull = patcher_nprandom(m.weibull)
    m.zipf = patcher_nprandom(m.zipf)

def hook_py_random(m):
    m.random = patcher_pyrandom(m.random)
    m.uniform = patcher_pyrandom(m.uniform)
    m.triangular = patcher_pyrandom(m.triangular)
    m.randint = patcher_pyrandom(m.randint)
    m.choice = patcher_pyrandom(m.choice)
    m.randrange = patcher_pyrandom(m.randrange)
    m.sample = patcher_pyrandom(m.sample)
    m.shuffle = patcher_pyrandom(m.shuffle)
    m.choices = patcher_pyrandom(m.choices)
    m.normalvariate = patcher_pyrandom(m.normalvariate)
    m.lognormvariate = patcher_pyrandom(m.lognormvariate)
    m.expovariate = patcher_pyrandom(m.expovariate)
    m.vonmisesvariate = patcher_pyrandom(m.vonmisesvariate)
    m.gammavariate = patcher_pyrandom(m.gammavariate)
    m.gauss = patcher_pyrandom(m.gauss)
    m.betavariate = patcher_pyrandom(m.betavariate)
    m.paretovariate = patcher_pyrandom(m.paretovariate)
    m.weibullvariate = patcher_pyrandom(m.weibullvariate)
    m.getrandbits = patcher_pyrandom(m.getrandbits)


generators = {}
save_queue = queue.Queue(10)
save_data = False
saved_data = {}

def load_generator():
    pass

def save_generator():
    save_data = True
    for key in generators.keys():
        save_queue.put(0)
    #save uid

def my_run(self):
    from contextlib import closing
    import random
    from tensorflow.python.keras.utils import data_utils
    global saved_data

    sequence = list(range(len(self.sequence)))
    self._send_sequence()  # Share the initial sequence
    while True:
      if self.shuffle:
        random.shuffle(sequence)
      
      # recover sequence
      # recover queue
      if self.uid in saved_data:
          start = saved_data[self.uid]['index'] 
          saved_sequence = saved_data[self.uid]['data']
          sequence = saved_sequence[start:]

      with closing(self.executor_fn(data_utils._SHARED_SEQUENCES)) as executor:
        for i in sequence:
          if self.stop_signal.is_set():
            #remove this iterator
            return

          #save (i, sequence)
          #save queue

          self.queue.put(
              executor.apply_async(data_utils.get_index, (self.uid, i)), block=True)

        # Done with the current epoch, waiting for the final batches
        self._wait_queue()

        if self.stop_signal.is_set():
          # We're done
          #remove this iterator
          return

      # Call the internal on epoch end.
      self.sequence.on_epoch_end()


def patcher_export(function):
    def wrapper(*args, **kwargs):
        #print("EXPORT", args)
        ret = function(*args, **kwargs)
        return ret

    return wrapper

def patcher_generator(function):
    def wrapper(*args, **kwargs):
        ret = function(*args, **kwargs)
        generators[args[0]] = []
        return ret

    return wrapper

def hook_export(m):
    m.api_export.__call__ = patcher_export(m.api_export.__call__)

def hook_generator(m):
    m.OrderedEnqueuer._run = my_run
    m.OrderedEnqueuer.__init__ = patcher_generator(m.OrderedEnqueuer.__init__)


hook_pool_modules = {
           #'keras_preprocessing.image.iterator' : hook_keras_iterator,
           #'tensorflow.python.util.tf_export' : hook_export,
           'tensorflow.python.keras.utils.data_utils' : hook_generator,
           'multiprocessing.pool' : hook_threadpool,
           'random' : hook_py_random,
        }

hook_pool_modules_ext = {
           'numpy.random.mtrand' : hook_np_random,
        }

def hook_pool(name, module):
    func = hook_pool_modules.get(name)
    if func is not None:
        func(module)

def hook_pool_ext(name, module):
    func = hook_pool_modules_ext.get(name)
    if func is not None:
        func(module)

