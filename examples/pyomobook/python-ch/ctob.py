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

# An example of a silly decorator to change 'c' to 'b'
# in the return value of a function.


def ctob_decorate(func):
    def func_wrapper(*args, **kwargs):
        retval = func(*args, **kwargs).replace('c', 'b')
        return retval.replace('C', 'B')

    return func_wrapper


@ctob_decorate
def Last_Words():
    return "Flying Circus"


print(Last_Words())  # prints: Flying Birbus
