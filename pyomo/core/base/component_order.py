#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________


__all__ = ['items', 'display_items', 'display_name']

from coopr.pyomo.base.sets import Set
from coopr.pyomo.base.rangeset import RangeSet
from coopr.pyomo.base.param import Param
from coopr.pyomo.base.var import Var
from coopr.pyomo.base.expression import Expression
from coopr.pyomo.base.objective import Objective
from coopr.pyomo.base.constraint import Constraint
from coopr.pyomo.base.sos import SOSConstraint

items = [ Set, RangeSet, Param, Var, Expression, \
        Objective, Constraint, SOSConstraint ]

display_items = [ Var, Objective, Constraint]
display_name = {Var:"Variables", Objective:"Objectives", Constraint:"Constraints"}
