#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

"""
Management of Pyomo commands
"""

__all__ = ['pyomo_command', 'get_pyomo_commands']

import logging

logger = logging.getLogger('pyomo.util')


registry = {}

#
# Decorate functions that are Pyomo commands
#
def pyomo_command(name=None, doc=None):
    #
    def wrap(fn):
        if name is None:                                  #pragma:nocover
            logger.error("Error applying decorator.  No command name!")
            return
        if doc is None:                                  #pragma:nocover
            logger.error("Error applying decorator.  No command documentation!")
            return
        #
        global registry
        registry[name] = doc
        return fn
    #
    return wrap


def get_pyomo_commands():   #pragma:nocover
    return registry
