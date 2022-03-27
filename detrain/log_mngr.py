import os
import sys

max_diff = 0

diff_size = -1
if 'DIFF_SIZE' in os.environ:
    diff_size = int(os.environ['DIFF_SIZE'])

def get_log_file_path():
    if 'MODEL_NAME' in os.environ:
        file_name =  "./trace." + os.environ['MODEL_NAME']
    else:
        file_name =  "./trace.tmp"

    return file_name

class LogEntry:
    def __init__(self, event, name, aux):
        self.event = event
        self.name = name
        self.aux = aux
        self.vars = {}

    def add_var(self, name, value):
        self.vars[name] = str(value)

    def add_vars(self, var):
        self.vars.update(var)

    def var_size(self):
        return len(self.vars)

    def cmp_vars(self, others):
        global max_diff

        diff = False
        if str(self) not in others:
            for k, obj in others.items():
                if self==obj and self.aux==obj.aux and str(self.vars)!=str(obj.vars):
                    for k,v in self.vars.items():
                        if (k in obj.vars) and (obj.vars[k]!=v):
                            ovs = obj.vars[k].split("-")
                            vs = v.split("-")
                            for a,b in zip(ovs, vs):
                                cur_diff = abs(int(a)-int(b))
                                #if cur_diff==69:
                                #if cur_diff==800:
                                if cur_diff == diff_size:
                                    if max_diff < cur_diff:
                                        max_diff = cur_diff
                                    #print("DIFF", k, v, obj.vars[k], max_diff, "\n", self, "\n", obj, "\n")
                                    diff = True 
                                    break

        return diff
 
    @staticmethod
    def from_string(s):
        s = s.split("###")
        if len(s)==4:
            ent = LogEntry(s[0], s[1], s[2])
            ent.add_vars(eval(s[3]))
            return ent

        return None
    
    def __eq__(self, other):
        return self.event==other.event and self.name==other.name

    def __str__(self):
        var = str(self.vars)
        return f"{self.event}###{self.name}###{self.aux}###{var}"

class LogItems:
    def __init__(self):
        self.logs = {}
        self.loops = {}
        self.file_name = get_log_file_path()
        self.init_trace = not self.load_trace()
        print("Log file name is", self.file_name)
        print("Init_trace is ", self.init_trace)

    # TODO XXZ: DBG    
    
    def save_or_cmp_log(self, otype, obj, cur_cs):
        ret = False

        if (type(obj)==LogEntry) and (obj.var_size()==0):            
            # if 'ImageDataGenerator' in obj.name:
            #     print('XXZ: Dataloader %s directly returned %d!'%(obj.name, False))
            #     print(str(obj))
            return ret

        cur_cs = ":".join([cs[3].full_name for cs in cur_cs if cs[0]=="FUNC"])
        if self.init_trace:
            self.save_to_file(otype, obj, cur_cs)
        else:
            ret = self.compare(otype, obj, cur_cs)
        # if type(obj)==LogEntry and 'Data' in obj.name:
        #     print('XXZ: Dataloader %s returned %d!'%(obj.name, ret))
        return ret

    def compare(self, otype, obj, cur_cs):
        ret = False
        
        if cur_cs in self.logs:
            ret = obj.cmp_vars(self.logs[cur_cs])

        return ret

    def load_trace(self):
        if not os.path.exists(self.file_name):
            return False

        with open(self.file_name, 'r') as f:
            for line in f:
                val = line.strip().split("@@")
                if len(val)==3:
                    if val[0]=="LOG":
                        if val[1] not in self.logs:
                            self.logs[val[1]] = {}
                        self.logs[val[1]][val[2]] = LogEntry.from_string(val[2])
                    elif val[0]=="LOOP":
                        self.loops[val[2]] = (1, "")

        return True

    def save_to_file(self, otype, obj, cur_cs):
        log = otype + "@@" + cur_cs + "@@" + str(obj)
        with open(self.file_name, 'a+') as f:
            f.write(log)
            f.write(os.linesep)
            f.flush()

    def get_loops(self):
        return self.loops

logMngr = LogItems()

