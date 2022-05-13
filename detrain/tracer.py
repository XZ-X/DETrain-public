import os
import sys
import inspect
import threading
import atexit
import numpy
import pdb
from .log_mngr import *
from .trace_analysis import save_inst_info, get_trace_file_path
from detrain import cfg

white_list = ["tensorflow.python.util.lock_util._Context.__init__", "tensorflow.python.keras.callbacks.CallbackList._call_batch_hook", "tensorflow.python.util.object_identity.ObjectIdentitySet.__init__", "tensorflow.python.autograph.core.naming.Namer.__init__"]
random_create = ["torch._C.Generator", "myrand.random_create.__init__", "numpy.random.mtrand.RandomState"]
update_func = ["apply_gradients", "backward"]
dataflow_func = ["__init__", "fit"]
dataloader = []
dataloader_objs = {}

class LineInfo:
    def __init__(self, full_name="", module_name="", obj_name="", func_name="", file_name="", line_no="", return_val="", code="", caller_file_name="", caller_line_no="", caller_code=""):
        self.full_name = full_name.replace("\"", "")
        self.module_name = module_name
        self.obj_name = obj_name
        self.func_name = func_name
        self.file_name = file_name
        self.line_no = line_no
        self.return_val = return_val
        self.code = code.replace("\"", "") if code is not None else ""
        self.caller_file_name = caller_file_name
        self.caller_line_no = caller_line_no
        self.caller_code = caller_code.replace("\"", "") if caller_code is not None else ""

    def __str__(self):
        s = f"\"{self.full_name}\", \"{self.module_name}\", \"{self.obj_name}\", \"{self.func_name}\", \"{self.file_name}\", \"{self.line_no}\", \"{self.return_val}\", \"{self.code}\", \"{self.caller_file_name}\", \"{self.caller_line_no}\", \"{self.caller_code}\""
        return s.replace("\n", "")

    def __repr__(self):
        return self.__str__()

def get_name(*args):
    key = ""
    for a in args:
        if a not in [".", "", None]:
            key = key + "." + a
    return key[1:]

def class_fullname(o):
    module = getattr(o, "__module__", None)
    if module is None or module == str.__class__.__module__ or module == "__main__":
        return o.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__name__

def method_fullname(o):
    return func_fullname(o.__func__)

def func_fullname(o):
    module = o.__module__
    if module is None or module == str.__class__.__module__ or module == "__main__":
        return o.__qualname__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__qualname__

def c_obj_name(obj):
    obj_id = hex(id(obj))
    cur_obj = getattr(obj, "__self__", None)
    if cur_obj is not None:
        return obj_id + "@" + class_fullname(cur_obj) 
    module = getattr(obj, "__module__", None)
    if module is None:
        return obj_id + "@" + obj.__name__ + "@" + str(obj)  
    else:
        return obj_id + "@" + module + '.' + obj.__name__

def is_random_function(name, func_obj, obj_info):
    if inspect.ismodule(func_obj) \
            or inspect.isclass(func_obj):
        return False

    full_name = class_fullname(func_obj.__class__)
    if full_name in random_create: 
        # add unseen generators
        if hex(id(func_obj)) not in process_event.generators: 
            process_event.generators[hex(id(func_obj))] = (name, obj_info)
        return True
    return False

def get_size(obj, wl={}):
    size = "0"
    
    otype = type(obj)
    if otype in [tuple, list, numpy.ndarray]:
        try:
            size = str(len(obj))
        except:
            pass
        if size!="0":
            # FIXME if there are many elements with different sizes in the list
            last_size = 0
            for o in obj:
                sub_size = get_size(o)
                if (sub_size!="0") and (sub_size!=last_size):
                    size += "-" + sub_size
                last_size = sub_size
    elif otype in [dict]:
        wl.update({"self":"", "__all__":"", "__doc__":"", "__dict__":"", "cache":"", "__builtins__":"", "docstr":"", "doc":"", "docstring":""})
        for k,v in obj.items():
            if k not in wl:
                wl[k] = ""
                sub_size = get_size(v, wl)
                if sub_size!="0":
                    size += "-" + sub_size
    elif otype not in [str]:
        try:
            size = str(len(obj))
        except:
            pass

    return size

def extract_name(obj):
    try:
        if inspect.isclass(obj):
            obj = class_fullname(obj) 
        elif inspect.ismethod(obj):
            obj = method_fullname(obj) 
        elif inspect.isfunction(obj):
            obj = func_fullname(obj)
        elif type(obj) == str:
            pass
        elif inspect.isclass(type(obj)):
            obj = class_fullname(type(obj))
    except Exception as e:
        #print("Cannot get the function name:", e)
        obj = None

    return obj

def get_module_by_name(name):
    if name is not None:
        packs = name.split(".")
        for i in range(len(packs), 0, -1):
            sub_mod = ".".join(packs[0:i])
            if sub_mod in sys.modules:
                return sub_mod
    return ""

def verify_attrs(module, full_name):
    if (full_name=="") or (module==""):
        return full_name

    ret = full_name
    if (module in sys.modules) and (full_name.startswith(module)):
        
        target = sys.modules.get(module)

        attr  = full_name[len(module)+1:]
        attrs = attr.split(".")
        for a in attrs:
            t = getattr(target, a, None)
            if t is not None:
                target = t
            else:
                ret = "NA_"+full_name
                break
    else:
        ret = "NA_"+full_name

    return ret 


callstack = [("ROOT", None, 0, LineInfo())]
call_id = 0

def mytrace(frame, event, arg):    
    if event in ['call', 'return', 'line']:
        file_name = frame.f_code.co_filename
        if not is_known_libs(file_name):
            process_event(frame, event, arg)

    return mytrace

def myprofile(frame, event, arg):    
    if event in ['c_call', 'c_return']:
        file_name = frame.f_code.co_filename
        if not is_known_libs(file_name):
            process_event(frame, event, arg)

    return myprofile

cnt=0

def process_event(frame, event, arg):

    global callstack
    global call_id
    global random_create
    global random_gen
    global parallel_func 
    global cnt
    line = frame.f_lineno
    code = frame.f_code
    file_name = code.co_filename
    func_name = code.co_name
    func_name = "" if "<" in func_name else func_name

    local_vars = frame.f_locals
    global_vars = frame.f_globals
    local_var_str = ""
    global_var_str = ""

    try:
        finfo = inspect.getframeinfo(frame)
    except:
        finfo = None

    if file_name in inspect.modulesbyfile:
        module_name = sys.modules.get(inspect.modulesbyfile[file_name]).__name__
    else:
        module_name = ""
    # TODO XXZ DBG
    if cfg.detailed_trace:
        print("DBG:%s:%d# %s"%(file_name, line, code))

    # check class objects
    obj = None
    obj_name = module_name
    if "self" in local_vars:
        obj = local_vars["self"]
        obj_name = extract_name(obj)
        module_name = get_module_by_name(obj_name)
    #print(module_name, obj_name, func_name, file_name)
    full_name = get_name(obj_name, func_name)
    full_name = verify_attrs(module_name, full_name)

    # check caller
    caller_file_name = None
    caller_line_no = None
    caller_code = ""
    caller_func = None
    caller_frame = frame.f_back
    if (event in ["call", "return"]) and (caller_frame is not None):
        caller_line_no = frame.f_back.f_lineno        
        caller_file_name = frame.f_back.f_code.co_filename
        caller_func = frame.f_back.f_code.co_name 
        caller_frame = caller_frame.f_back
        # FIXME magic word
        while ("/detrain/" in caller_file_name) and (caller_frame is not None):
            caller_line_no = caller_frame.f_lineno
            caller_file_name = caller_frame.f_code.co_filename
            caller_func = caller_frame.f_code.co_name 
            # caller_func = caller_frame.f_back.f_code.co_name 
            caller_frame = caller_frame.f_back
            #print("NNNNNNNNNNNNNNNNNNN", caller_file_name, caller_line_no)
    if (caller_file_name is None) or (caller_line_no is None):
        caller_file_name = callstack[-1][3].file_name
        caller_line_no = callstack[-1][3].line_no

    # executed code 
    exe_code=""
    if (finfo is not None) and (finfo.code_context is not None) and (finfo.index is not None):
        exe_code = finfo.code_context[finfo.index]
    if full_name=="" or event=="c_call":
        full_name = exe_code.strip()
        
    # current function info
    # FIXME remove unnecessary info
    current_info = LineInfo(full_name, module_name, obj_name, func_name, file_name, line, extract_name(arg), exe_code, caller_file_name, caller_line_no, caller_code)

    cur_cs = callstack[-1]
    if event in ["call", "c_call"]:
        call_id += 1
        callstack.append(("FUNC", hex(id(obj)), call_id, current_info))

        ent = LogEntry("CALL", full_name, str(finfo)) 
        if 'fit' in func_name:
            print("=====XXZ:DBG:FIT=====")
            print(ent)
            print(full_name)
            print("=====END XXZ:DBG:FIT=====")

        rngs = {}
        for k,v in local_vars.items():
            if k not in ["self", "__doc__", "cache", "__builtins__", "docstr", "doc", "docstring"]:
                try:
                    local_var_str += k + " : " + str(v) + "\n"
                    if is_random_function(k, v, current_info):
                        rngs[k] = v
                    
                    if (obj is not None) and (finfo is not None):
                        if getattr(v, "__len__", None):
                            v_size = get_size(v)
                            if v_size!="0":
                                ent.add_var(k, v_size)
                            #print(f"LOCAL VARIABLE: {module_name}, {func_name}", k, v_size, finfo)
                    
                    # function dependancy, add supporting python APIs
                    # TODO XXZ, dataflow func
                    if (func_name in dataflow_func) and (obj is not None) and (finfo is not None) and ((not process_event.in_loop) or (caller_func in ["__iter__", "iter"])):
                        dl = False
                        if type(v) in [tuple, list]:
                            for vv in v:
                                if hex(id(vv)) in dataloader_objs:
                                    dl = True
                                    break
                        elif hex(id(v)) in dataloader_objs:
                            dl = True

                        if dl:
                            cur_obj = hex(id(obj))
                            if (cur_obj not in dataloader_objs) and (full_name not in white_list):
                                dataloader_objs[cur_obj] = current_info
                except:
                    pass
        
        diff = logMngr.save_or_cmp_log("LOG", ent, callstack)        
        #if diff and (not process_event.in_loop) and (obj is not None):
        if diff and (obj is not None) and (full_name not in white_list):
            dataloader_objs[hex(id(obj))] = current_info

        """
        check random number generators
        """
        exe_code = " ".join(exe_code.split())
        # support save a random number generator to a variable
        # FIXME support pass a random number generator to a function
        #       but this is rare
        for rng in random_create:
            if rng in full_name:
                target = exe_code.split("=")[0].strip(" ,()\t")
                obj_id = hex(id(arg))
                process_event.generators[obj_id] = (target,current_info) 
                break

        """
        check weights update function. We treat weights update as one pass
        """
        for func in update_func:
            if func in full_name:
                print("UPDATE_FUNC", full_name, finfo, "\n    ", callstack)
                for i in range(len(callstack)-1, -1, -1):
                    if callstack[i][0]=="LOOP":
                        loop_key = callstack[i][3].file_name + ":" + str(callstack[i][3].line_no)
                        if loop_key not in process_event.loops:
                            process_event.loops[loop_key] = (1, callstack[i])
                            logMngr.save_or_cmp_log("LOOP", loop_key, callstack)
                        else:
                            count = process_event.loops[loop_key][0] + 1
                            process_event.loops[loop_key] = (count, process_event.loops[loop_key][1])
                            
                        process_event.loop_trace.append(("WEIGHTS", current_info, str(callstack)))
                        break

    elif event in ["return", "c_return"]:
        # check heap objects before return
        ent = LogEntry("RETURN", full_name, str(finfo)) 
        obj_dict = getattr(obj, "__dict__", None)
        if (obj_dict is not None) and (finfo is not None) and (type(obj_dict) is dict):
            for okey, oval in obj_dict.items():
                if type(oval) in [tuple, list, dict]:
                    oval_size = get_size(oval)
                    if oval_size!="0":
                        ent.add_var(okey, oval_size)

        diff = logMngr.save_or_cmp_log("LOG", ent, callstack)
        #if diff and (not process_event.in_loop) and (obj is not None):
        if diff and (obj is not None) and (full_name not in white_list):
            dataloader_objs[hex(id(obj))] = current_info

        if len(callstack) > 1:
            func_info = callstack.pop()
            # pop up loops in the current function
            while func_info[0]=="LOOP":
                func_info = callstack.pop()
                
        """
        if event in ["return", "c_return"]:
            return_obj = hex(id(arg))
            if return_obj not in dataloader_objs:
                dataloader_objs[return_obj] = current_info
        """

    elif event in ["line"]:        
        code_seg = exe_code.split()
        if len(code_seg) > 0:
            indent = len(exe_code)-len(exe_code.lstrip(" \t")) 
            loop_key = current_info.file_name + ":" + str(current_info.line_no)

            if cur_cs[0]=="LOOP" and indent<=cur_cs[2]:
                loop_info = callstack.pop()
                loop_key = loop_info[3].file_name + ":" + str(loop_info[3].line_no)
                if loop_key in process_event.loops:
                    process_event.in_loop = False

                # handle nested loops. They should be in the same function.
                while (len(callstack)>0) and (callstack[-1][0]=="LOOP") and (indent<=callstack[-1][2]):
                    loop_info = callstack.pop()
                    loop_key = loop_info[3].file_name + ":" + str(loop_info[3].line_no)
                    if loop_key in process_event.loops:
                        process_event.in_loop = False

            if code_seg[0] in ["for", "while"]:
                callstack.append(("LOOP", None, indent, current_info))
                loop_key = current_info.file_name + ":" + str(current_info.line_no)
                if loop_key in process_event.loops:
                    # remove current loop object from dataloader
                    # FIXME maybe loop object and data loader are the same
                    if (obj is not None) and (hex(id(obj)) in dataloader_objs):
                        dataloader_objs.pop(hex(id(obj)))
                    process_event.in_loop = True
                
                    process_event.loop_trace.append(("LOOP", current_info, str(callstack)))
 
    if module_name!="":
        # TODO XXZ
        cnt+=1
        if (cnt % 10000 == 0) or cfg.detailed_trace:
            print(cnt)
            print(f"NNN: {module_name}, {event}, {full_name}, {func_name}\t", extract_name(arg), finfo)
            print(f"DBG: {dataloader_objs}")
            stop_trace()
            # save_trace()
            restart_trace()
        pass

    for i in range(len(callstack)-1, -1, -1):
        if callstack[i][1] in dataloader_objs:
            break

        loop_key = callstack[i][3].file_name + ":" + str(callstack[i][3].line_no)
        if loop_key in process_event.loops:
            process_event.in_loop = True
            if (obj is not None) and (hex(id(obj)) in dataloader_objs) and (full_name not in white_list):
                print(f"IN LOOP: {event}, {module_name}, {func_name}, {full_name}\t", extract_name(arg), finfo, "\n", dataloader_objs[hex(id(obj))], "\n")
                process_event.loop_trace.append(("DL", current_info, str(callstack)))
            break

    """
    if process_event.in_loop and (obj is not None) and (hex(id(obj)) in dataloader_objs):
        print(f"IN LOOP: {event}, {module_name}, {func_name}\t", extract_name(arg), finfo, "\n", dataloader_objs[hex(id(obj))], "\n")
        process_event.loop_trace.append(("DL", current_info, str(callstack)))
    """

def is_known_libs(file_name):
    """
    if ("site-packages/torch" in file_name) \
            or ("site-packages/tensorflow" in file_name) \
            or ("site-packages/numpy" in file_name):
    """
    if ("/tensorflow" in file_name) \
            or ("/torch" in file_name) \
                or ("keras" in file_name):
        return False

    if ("importlib" in file_name) \
            or ("matplotlib" in file_name) \
            or ("/tmp/" in file_name) \
            or ("/tensorflow_addons" in file_name) \
            or ("lib/python3" in file_name):
        return True

    return False

def getKey(item):
    return item[0]

def save_training_info(trace_list):
    print("In save training info, print loop trace:")
    print(process_event.loop_trace)
    import pickle
    #with open(os.path.join("./loop_trace.stargan"), 'wb+') as ret_file:
    with open(get_trace_file_path(), 'wb') as ret_file:
        pickle.dump(process_event.loop_trace, ret_file)

def init():
    process_event.generators = {}
    process_event.loop_trace = []
    # default stack
    process_event.loops = {}
    process_event.loops.update(logMngr.get_loops())
    process_event.in_loop = False

    print("Init tracer...")
    sys.settrace(mytrace)
    threading.settrace(mytrace)
    #sys.setprofile(myprofile)
    #threading.setprofile(myprofile)

def save_trace():
    print("Generators")
    print(process_event.generators)
    print("Loops")
    print(process_event.loops)
    print("DL Objects")
    print(dataloader_objs)
    save_training_info(process_event.loop_trace)
    save_inst_info(process_event.loop_trace)
    print("Saved...")
    
def stop_trace():
    #sys.setprofile(None)
    sys.settrace(None)
    save_trace()

def restart_trace():
    sys.settrace(mytrace)
    threading.settrace(mytrace)