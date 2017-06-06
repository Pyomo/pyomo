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
# Utility classes for working with the logger
#

import logging

class LoggingIntercept(object):
    """Context manager for temporarily grabbing the output from a log
    stream.  Useful for testing logged error messages.
    """

    def __init__(self, output, module=None, level=logging.WARNING):
        self.handler = logging.StreamHandler(output)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.handler.setLevel(level)
        self.level = level
        self.module = module

    def __enter__(self):
        logger = logging.getLogger(self.module)
        self.level = logger.level
        logger.setLevel(self.handler.level)
        logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        logger = logging.getLogger(self.module)
        logger.removeHandler(self.handler)
        logger.setLevel(self.level)
