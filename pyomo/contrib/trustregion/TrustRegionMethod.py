#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from math import pow 
from numpy import inf, concatenate
from numpy.linalg import norm 

from pyomo.contrib.trustregion.filter import (
    FilterElement, Filter)
from pyomo.contrib.trustregion.utils import copyVector
from pyomo.contrib.trustregion.Logger import Logger
from pyomo.contrib.trustregion.PyomoInterface import (
    PyomoInterface, RMType)

def TrustRegionMethod(m, efList, config):
    """
    The main function of the Trust Region Filter algorithm

    m is a PyomoModel containing ExternalFunction() objects Model
    requirements: m is a nonlinear program, with exactly one active
    objective function.

    efList is a list of ExternalFunction objects that should be
    treated with the trust region

    config is the persistent set of variables defined 
    in the ConfigBlock class object

    Return: 
    model is solved, variables are at optimal solution or
    other exit condition.  model is left in reformulated form, with
    some new variables introduced in a block named "tR" TODO: reverse
    the transformation.
    """

    logger = Logger()
    TrustRegionFilter = Filter()
    problem = PyomoInterface(m, efList, config)
    x, y, z = problem.getInitialValue()
    pass