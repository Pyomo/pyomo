#
# This script finds all *.py files in the current and subdirectories.
# It processes these files to find blocks that start/end with "# @"
# For example
#
#    print("START HERE")
#    # @block
#    print("IN THE BLOCK")
#    x = 1
#    # @block
#    print("END HERE")
#
# If this file was foo.py, then a file foo_block.spy is created, which 
# contains the lines between the lines starting with "# @".
#
# Additionally, the file foo.spy is created, which strips all lines
# starting with "# @".
#
# This utility provides a convenient mechanism for creating complex scripts that
# can be tested, while extracting pieces that are included in Sphinx documentation.
#
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
    block_name = None
    for line in INPUT:
        tmp = line.strip()
        if tmp.startswith("# @"):
            if flag is False:
                block_name = tmp[3:]
                flag = True
                OUTPUT_ = open(root+'/'+prefix+'_%s.spy' % block_name,'w')
            else:
                if block_name != tmp[3:]:
                    print("ERROR parsing file '%s': Started block '%s' but ended with '%s'" % (root+'/'+file, block_name, tmp[3:]))
                    sys.exit(1)
                flag = False
                block_name is None
                OUTPUT_.close()
            continue
        elif flag:
            OUTPUT_.write(line)
        OUTPUT.write(line)
    INPUT.close()
    OUTPUT.close()


for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        f(root, file)

