#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Var, Param, Expression, Objective, Block, \
    Constraint, Suffix

valid_expr_ctypes_minlp = {Var, Param, Expression, Objective}
valid_active_ctypes_minlp = {Block, Constraint, Objective, Suffix}
