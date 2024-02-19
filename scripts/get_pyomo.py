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
#
# A script to install Pyomo and its dependencies (ply)
#

import sys

try:
    import pip

    pip_version = pip.__version__.split('.')
    for i, s in enumerate(pip_version):
        try:
            pip_version[i] = int(s)
        except:
            pass
    pip_version = tuple(pip_version)
except ImportError:
    print("You must have 'pip' installed to run this script.")
    raise SystemExit


print("Installing Pyomo ...")

cmd = ['install']
# Disable the PIP download cache
if pip_version[0] >= 6:
    cmd.append('--no-cache-dir')
else:
    cmd.append('--download-cache')
    cmd.append('')
# Allow the user to provide extra options
cmd.extend(sys.argv[1:])
# install Pyomo
cmd.append('Pyomo')

pip.main(cmd)
