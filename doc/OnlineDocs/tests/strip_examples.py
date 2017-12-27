import glob
import sys
import os
import os.path

def f(root, file):
    if not file.endswith('.py'):
        return
    prefix = os.path.splitext(file)[0]
    #print([root, file, prefix])
    OUTPUT = open(root+'/'+prefix+'.spy','w')
    INPUT = open(root+'/'+file,'r')
    flag = False
    for line in INPUT:
        if line.startswith("# @"):
            line = line.strip()
            if line.endswith(":"):
                OUTPUT_ = open(root+'/'+prefix+'_%s.spy' % line[3:-1],'w')
                flag = True
            else:
                OUTPUT_.close()
                flag = False
            continue
        elif flag:
            OUTPUT_.write(line)
        OUTPUT.write(line)
    INPUT.close()
    OUTPUT.close()


for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        f(root, file)

