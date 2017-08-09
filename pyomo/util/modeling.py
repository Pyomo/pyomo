#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from random import randint

def unique_component_name(instance, name):
    # test if this name already exists in model. If not, we're good. 
    # Else, we add random numbers until it doesn't
    if instance.component(name) is None:
        return name
    name += '_%d' % (randint(0,9),)
    while True:
        if instance.component(name) is None:
            return name
        else:
            name += str(randint(0,9))
