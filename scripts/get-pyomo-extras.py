#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# A script to optionally install packages that Pyomo could leverage.
#

packages = {'xlrd':None, 'openpyxl':None, 'suds-jurko':'suds', 'PyYAML':'yaml', 'pypyodbc':None, 'pymysql':None, 'openopt':None, 'FuncDesigner':None, 'DerApproximator':None, 'sympy':None}

#
# Verify that pip is installed
#
import sys
try:
    import pip
    pip_version = pip.__version__.split('.')
    for i,s in enumerate(pip_version):
        try:
            pip_version[i] = int(s)
        except:
            pass
    pip_version = tuple(pip_version)
except ImportError:
    print("You must have 'pip' installed to run this script.")
    raise SystemExit

cmd = ['install','--upgrade']
# Disable the PIP download cache
if pip_version[0] >= 6:
    cmd.append('--no-cache-dir')
else:
    cmd.append('--download-cache')
    cmd.append('')

print(' ')
print('-'*60)
print("Installation Output Logs")
print("  (A summary will be printed below)")
print('-'*60)
print(' ')

results = {}
for package in packages:
    try:
        # Allow the user to provide extra options
        pip.main(cmd + sys.argv[1:] + [package])
        if packages[package]:
            __import__(packages[package])
        else:
            __import__(package)
        results[package] = True
    except:
        results[package] = False
    pip.logger.consumers = []

print(' ')
print(' ')
print('-'*60)
print("Installation Summary")
print('-'*60)
print(' ')
for package in sorted(packages):
    if results[package]:
        print("YES %s" % package)
    else:
        print("NO  %s" % package)
