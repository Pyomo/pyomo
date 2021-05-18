#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# A script to uninstall Coopr and Pyomo.
#

try:
    import pip
except ImportError:
    print("You must have 'pip' installed to run this script.")
    raise SystemExit


packages = [
'Pyomo',
'Coopr',
'coopr.age',
'coopr.bilevel',
'coopr.core',
'coopr.dae',
'coopr.environ',
'coopr.gdp',
'coopr.misc',
'coopr.mpec',
'coopr.neos',
'coopr.openopt',
'coopr.opt',
'coopr.os',
'coopr.pyomo',
'coopr.pysos',
'coopr.pysp',
'coopr.solvers',
'coopr.sucasa',
]

print("Uninstalling...")
for package in packages:
    try:
        pip.main(['uninstall','-y',package])
    except:
        pass
    #
    # See https://github.com/pypa/pip/issues/1618 for an 
    # explanation of this hack.  This reset's the logger used by
    # pip.
    #
    pip.logger.consumers = []

