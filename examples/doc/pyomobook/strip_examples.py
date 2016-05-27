import glob
import sys
import os
import os.path

def f(file):
    base, name = os.path.split(file)
    prefix = os.path.splitext(name)[0]
    if prefix.endswith('_strip'):
        return
    OUTPUT = open(base+'/'+prefix+'_strip.py','w')
    INPUT = open(file,'r')
    for line in INPUT:
        if line[0] == '#' and '@' in line:
            continue
        OUTPUT.write(line)
    INPUT.close()
    OUTPUT.close()
    #if not os.path.exists(base+'/'+prefix+'.txt'):
    #    print "No baseline:",file


for file in glob.glob(os.path.abspath(os.path.dirname(__file__))+'/*/*.py'):
    f(file)

for file in glob.glob(os.path.abspath(os.path.dirname(__file__))+'/*/*/*.py'):
    f(file)

