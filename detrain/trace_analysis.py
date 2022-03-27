import os
import sys

def get_trace_file_path():
    if 'MODEL_NAME' in os.environ:
        file_name =  "/tmp/loop_trace." + os.environ['MODEL_NAME']
    else:
        file_name =  "/tmp/loop_trace.tmp"

    return file_name

def get_inst_file_path():
    if 'CKPT_DIR' in os.environ:
        if not os.path.exists(os.environ['CKPT_DIR']):
            os.makedirs(os.environ['CKPT_DIR'])
        file_name =  os.environ['CKPT_DIR'] + "/inst.info"
    else:
        file_name =  "./inst.info"

    return file_name

def convert_to_list(obj):
    return eval("'FUNC', None, 0, " + str(obj))

def get_common_line(obj1, obj2, index=None):
    ret = []
    if (obj1 is None) or (obj2 is None):
        return ret

    new_cs = eval(obj1[2])
    ref = eval(obj2[2])

    """
    last_cs = new_cs[-1][7]+":"+new_cs[-1][8]
    new_cs = [(cs[11]+":"+cs[12]) for cs in new_cs]
    """
    if index is None:
        index = new_cs[0]

    parent = None
    find_index = False
    for cs1, cs2 in zip(new_cs, ref):
        #if parent is not None:
        #    print(parent[1][3], index, "\n", cs1[7], cs1[8], cs2[7], cs2[8])
            
        if (cs1 is not None) and (cs2 is not None) \
                and (not ((cs1[7]==cs2[7]) and (cs2[8]==cs2[8]))):
            ret.append(parent)
            ret.append(((cs1[7]+":"+cs1[8]), cs1))
            ret.append(((cs2[7]+":"+cs2[8]), cs2))
            break

        if find_index and (cs1 is not None) and (not cs1[3].startswith("NA_")):
            # find function from "index"
            parent = ((cs1[7]+":"+cs1[8]), cs1)
            ret.append(parent)
            ret.append(((cs1[7]+":"+cs1[8]), cs1))
            ret.append(((cs2[7]+":"+cs2[8]), cs2))
            break

        if not cs1[3].startswith("NA_"):
            parent = ((cs1[7]+":"+cs1[8]), cs1)
        if index==parent[0]:
            find_index = True

    if len(ret)==0:
        # add the difference to the return value
        ret.append(parent)
        if len(new_cs) < len(ref):
            ret.append(((obj1[1].file_name+":"+str(obj1[1].line_no)), convert_to_list(obj1[1])))
            ret.append(((ref[len(new_cs)][7]+":"+ref[len(new_cs)][8]), ref[len(new_cs)]))
        elif len(new_cs) > len(ref):
            ret.append(((new_cs[len(ref)][7]+":"+new_cs[len(ref)][8]), new_cs[len(ref)]))
            ret.append(((obj2[1].file_name+":"+str(obj2[1].line_no)), convert_to_list(obj2[1])))
        else:
            ret.append(((obj1[1].file_name+":"+str(obj1[1].line_no)), convert_to_list(obj1[1])))
            ret.append(((obj2[1].file_name+":"+str(obj2[1].line_no)), convert_to_list(obj2[1])))

    return ret

def save_inst_info(loop_trace):
    """
    inst["dataloader"]["full_name"] # class object
    inst["dataloader"]["iter"]
    inst["dataloader"]["next"]
    inst["dataloader"]["module"]
    inst["dataloader"]["persistent"]
    
    inst["loop_type"]["type"]
    
    inst["code_info"]["dl_call_file_name"]
    inst["code_info"]["dl_call_line_no"]
    inst["code_info"]["update_call_file_name"]
    inst["code_info"]["update_call_line_no"]
    """

    inst = {}
    inst["dataloader"] = {}
    inst["loop_type"] = {}
    inst["code_info"] = {}

    dl_objs = {}
    iterations = -1 # remove the last LOOP event 
    weight_func = None
    loop_line = None
    dl_persistent = False

    new_trace = []
    for event in loop_trace:
        if event[0] == "LOOP":
            iterations += 1
            loop_line = event
        elif event[0] == "DL":
            if event[1].full_name not in dl_objs:
                dl_objs[event[1].full_name] = (1, event)
            else:
                new_event_line_no = 0
                old_event_line_no = 0

                com = get_common_line(event, dl_objs[event[1].full_name][1])
                if len(com[1]) != 0:
                    diff_new = com[1]
                    diff_old = com[2]
                    if diff_new[0] == diff_old[0]:
                        # the same wrapper function
                        new_event_line_no = diff_new[1][12]
                        old_event_line_no = diff_old[1][12]
                    else:
                        new_event_file_name = diff_new[1][7]
                        old_event_file_name = diff_old[1][7]
                        if new_event_file_name==old_event_file_name:
                            new_event_line_no = diff_new[1][8]
                            old_event_line_no = diff_old[1][8]
                        else:
                            new_event_line_no = diff_new[1][12]
                            old_event_line_no = diff_old[1][12]

                safe_event = event
                if dl_persistent:
                    if int(old_event_line_no) < int(new_event_line_no):
                        safe_event = dl_objs[event[1].full_name][1]
                else:
                    if int(old_event_line_no) > int(new_event_line_no):
                        safe_event = dl_objs[event[1].full_name][1]

                called_times = dl_objs[event[1].full_name][0] + 1
                # only record the last one based on the order of invocation
                dl_objs[event[1].full_name] = (called_times, safe_event)
        elif event[0] == "WEIGHTS":
            weight_func = event 

    if loop_line is None:
        print("Cannot find loop!!!")
        return

    dl_iter = None 
    dl_next = None 

    for k,v in dl_objs.items():
        if "module" not in inst["dataloader"]:
            inst["dataloader"]["full_name"] = v[1][1].obj_name
            inst["dataloader"]["module"] = v[1][1].module_name
            inst["dataloader"]["init"] = (v[1][1].module_name, v[1][1].obj_name[len(inst["dataloader"]["module"])+1:] + ".__init__")
            inst["dataloader"]["persistent"] = dl_persistent
            inst["dataloader"]["iter"] = ("", "")
            inst["dataloader"]["next"] = ("", "")

        if v[0] >= iterations:
            inst["dataloader"]["next"] = (v[1][1].module_name, v[1][1].full_name[len(v[1][1].module_name)+1:])
            # handle different types of execution modes, namely, the graph and eager mode
            if inst["dataloader"]["next"][1]=="OwnedIterator._type_spec":
                inst["dataloader"]["next"] = (inst["dataloader"]["next"][0], "OwnedIterator.__next__")
            dl_next = v[1]
        elif "next" in v[1][1].func_name:
            # standard APIs
            inst["dataloader"]["next"] = (v[1][1].module_name, v[1][1].full_name[len(v[1][1].module_name)+1:])
            dl_next = v[1]
        elif "iter" in v[1][1].func_name:
            inst["dataloader"]["iter"] = (v[1][1].module_name, v[1][1].full_name[len(v[1][1].module_name)+1:]) 
            dl_iter = v[1]

    loop_key = loop_line[1].file_name + ":" + str(loop_line[1].line_no)
    inst["code_info"]["loop_file_name"] = loop_line[1].file_name
    inst["code_info"]["loop_line_no"] = str(loop_line[1].line_no)

    inst["code_info"]["epoch_file_name"] = ""
    inst["code_info"]["epoch_line_no"] = ""

    # get epoch information
    loop_cs = eval(loop_line[2])
    #print(loop_cs)
    for i in range(len(loop_cs)-2, -1, -1):
        last_cs = loop_cs[i]
        if last_cs[0] == "LOOP":
            inst["code_info"]["epoch_file_name"] = last_cs[7] 
            inst["code_info"]["epoch_line_no"] = last_cs[8]
            break

    # loop on dataset using for ... in ...
    com = get_common_line(dl_iter, loop_line)
    com_key = com[0][0] if len(com)==3 else ""

    if com_key == loop_key:
        inst["loop_type"]["type"] = "ON_DATASET"
    else:
        inst["loop_type"]["type"] = "ON_CUSTOM"

        com = get_common_line(dl_next, weight_func, loop_key)
        #print(dl_next)
        #print(weight_func)
        if len(com)==3:
            if com[0][1][0] != "LOOP":
                full_name = com[0][1][3]
                module_name = com[0][1][4]
                attr = full_name[len(module_name)+1:]
                inst["code_info"]["wrapper_module"] = module_name 
                inst["code_info"]["wrapper_func"] = attr

            inst["code_info"]["dl_call_file_name"] = com[1][1][11]
            inst["code_info"]["dl_call_line_no"] = com[1][1][12]
                
            # instrument the weight function
            inst["code_info"]["update_call_file_name"] = com[2][1][11]
            inst["code_info"]["update_call_line_no"] = com[2][1][12]

        else:
            print("Cannot analyze custom loop!!!")
        
    print(inst)
    with open(get_inst_file_path(), 'w+') as info_file:
        info_file.write(str(inst))

    
if __name__ == "__main__":
    import pickle
    trace_file_name = get_trace_file_path()
    if len(sys.argv) > 1:
        trace_file_name = sys.argv[1]

    with open(trace_file_name, 'rb') as f:
        trace = pickle.load(f)

    save_inst_info(trace)

