#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

"""
This module redefines the global help() function.  If the 'thing'
being requested for help has the '__help__' attribute, then that
string is used to provide help information.  Otherwise, the standard
help utility is used.
"""

old_help = None
# there are times when the help function is not available
# (e.g., after 'freezing' code into an executable with tools
# like py2exe or cx_Freeze)
try:
    old_help = help
except NameError:
    old_help = None

def help(thing=None):                               #pragma:nocover
    if not thing is None and hasattr(thing, '__help__'):
        print(thing.__help__)
    else:
        if old_help is None:
            raise NameError("Builtin 'help' is not available")
        old_help(thing)

try:
    __builtins__['help'] = help
except:                                             #pragma:nocover
    # If this fails, then just die silently.
    pass
