#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [
    "AbstractProblemWriter",
    "WriterFactory",
    "ProblemConfigFactory",
    "BaseProblemConfig",
    "BranchDirection",
]

from pyomo.common import Factory


ProblemConfigFactory = Factory('problem configuration object')


class BaseProblemConfig(object):
    """Base class for plugins generating problem configurations"""

    def config_block(self):
        pass


WriterFactory = Factory('problem writer')


class AbstractProblemWriter(object):
    """Base class that can write optimization problems."""

    def __init__(self, problem_format): #pragma:nocover
        self.format=problem_format

    def __call__(self, model, filename, solver_capability, **kwds): #pragma:nocover
        raise TypeError("Method __call__ undefined in writer for format "+str(self.format))

    #
    # Support "with" statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass


class BranchDirection(object):
    """ Allowed values for MIP variable branching directions in the `direction` Suffix of a model. """

    default = 0
    down = -1
    up = 1

    ALL = {default, down, up}
