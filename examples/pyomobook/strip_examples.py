#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import glob
import sys
import os
import os.path


def f(file):
    base, name = os.path.split(file)
    prefix = os.path.splitext(name)[0]
    if prefix.endswith('_strip'):
        return

    with open(base + '/' + prefix + '_strip.py', 'w') as OUTPUT, open(
        file, 'r'
    ) as INPUT:
        for line in INPUT:
            if line[0] == '#' and '@' in line:
                continue
            OUTPUT.write(line)


for file in glob.glob(os.path.abspath(os.path.dirname(__file__)) + '/*/*.py'):
    f(file)

for file in glob.glob(os.path.abspath(os.path.dirname(__file__)) + '/*/*/*.py'):
    f(file)
