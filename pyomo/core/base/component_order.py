#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


__all__ = ['items', 'display_items', 'display_name']

from pyomo.core.base.sets import Set
from pyomo.core.base.rangeset import RangeSet
from pyomo.core.base.param import Param
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression
from pyomo.core.base.block import Block
from pyomo.core.base.objective import Objective
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.sos import SOSConstraint

items = [ Set, RangeSet, Param, Var, Expression, \
        Objective, Constraint, SOSConstraint ]

display_items = [ Var, Objective, Constraint]
# TODO: Add Block to display_items after 4.0 release.  See note in
# Block.display() [JDS 1/7/15]
display_name = {Var:"Variables", Objective:"Objectives", Constraint:"Constraints", Block:"Blocks"}
