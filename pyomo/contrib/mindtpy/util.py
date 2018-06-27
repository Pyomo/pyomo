"""Utility functions and classes for the MindtPy solver."""
from __future__ import division

import logging
from math import fabs, floor, log

from pyomo.core import (Any, Binary, Block, Constraint, NonNegativeReals,
                        Objective, Reals, Var, minimize, value)
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel import ComponentSet
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverFactory
from pyomo.opt.results import ProblemSense, SolverResults


class _DoNothing(object):
    """Do nothing, literally.
    This class is used in situations of "do something if attribute exists."
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def _do_nothing(*args, **kwargs):
            pass
        return _do_nothing


class MindtPySolveData(object):
    """Data container to hold solve-instance data.
    Key attributes:
        - original_model: the original model that the user gave us to solve
        - working_model: the original model after preprocessing
        - linear_GDP: the linear-discrete master problem
    """
    pass

def a_logger(str_or_logger):
    """Returns a logger when passed either a logger name or logger object."""
    if isinstance(str_or_logger, logging.Logger):
        return str_or_logger
    else:
        return logging.getLogger(str_or_logger)