"""
Management of Coopr commands
"""

__all__ = ['coopr_command', 'get_coopr_commands']

import logging

logger = logging.getLogger('coopr.core')


registry = {}

#
# Decorate functions that are Coopr commands
#
def coopr_command(name=None, doc=None):
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


def get_coopr_commands():
    return registry
