#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# # Register the pint quantity type to prevent warnings
# from pyomo.common.numeric_types import RegisterNumericType
# # try:
# import pint
# RegisterNumericType(pint.Quantity)
# # except:
# #     pass

# Recommended just to build all of the appropriate things
import pyomo.environ as pyo

# Import the relevant classes from Formulation
try:
    from pyomo.contrib.edi.formulation import Formulation
except:
    pass
    # in this case, the dependencies are not installed, nothing will work

# Import the black box modeling tools
try:
    from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel

except:
    pass
    # in this case, the dependencies are not installed, nothing will work


# the printer that does units ok
import copy
from pyomo.core.base.units_container import _PyomoUnit
from pyomo.core.expr.numeric_expr import NPV_ProductExpression, NPV_DivisionExpression
from collections import namedtuple
import numpy as np


def recursive_sub(x_in):
    x = list(copy.deepcopy(x_in))
    for i in range(0, len(x)):
        if isinstance(x[i], _PyomoUnit):
            x[i] = '1.0*' + str(x[i])
        elif (
            isinstance(
                x[i], (NPV_ProductExpression, NPV_DivisionExpression, np.float64)
            )
            or x[i] is None
        ):
            if pyo.value(x[i]) == 1:
                x[i] = '1.0*' + str(x[i])
            else:
                x[i] = str(x[i])
        else:
            x[i] = recursive_sub(list(x[i]))
    return x


def ediprint(x):
    print(recursive_sub(x))
