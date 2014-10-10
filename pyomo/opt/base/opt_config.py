#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

import pyutilib.services

#
# Create plugins, which are automatically registered in the 'pyomo.opt'
# namespace.
#
def validate_pico(filename):
    if filename.startswith('/usr/bin') or filename.startswith('/bin'):
        return False
    return True
pyutilib.services.register_executable(name="PICO", validate=validate_pico)
pyutilib.services.register_executable(name="pico_convert")
pyutilib.services.register_executable(name="glpsol")
pyutilib.services.register_executable(name="ampl")
pyutilib.services.register_executable(name="timer")
