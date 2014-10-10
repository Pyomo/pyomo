"""
Management of Pyomo commands
"""

__all__ = ['pyomo_command', 'get_pyomo_commands']

import logging

logger = logging.getLogger('pyomo.misc')


registry = {}

#
# Decorate functions that are Pyomo commands
#
def pyomo_command(name=None, doc=None):
    #
    def wrap(fn):
        #print "HERE %s '%s' '%s'" % (fn, name, doc)
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


def get_pyomo_commands():
    return registry
