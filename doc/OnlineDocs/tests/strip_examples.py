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
    for line in INPUT:
        if line[0] == '#' and '@' in line:
            continue
        OUTPUT.write(line)
    INPUT.close()
    OUTPUT.close()


for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        f(root, file)

