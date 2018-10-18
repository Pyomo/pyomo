#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common

#
# Create plugins, which are automatically registered in the 'pyomo.opt'
# namespace.
#
def validate_pico(filename):
    if filename.startswith('/usr/bin') or filename.startswith('/bin'):
        return False
    return True
pyomo.common.register_executable(name="PICO", validate=validate_pico)
pyomo.common.register_executable(name="pico_convert")
pyomo.common.register_executable(name="glpsol")
pyomo.common.register_executable(name="ampl")
pyomo.common.register_executable(name="timer")
