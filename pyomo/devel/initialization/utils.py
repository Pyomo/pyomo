# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pe
from pyomo.common.collections import ComponentSet
from pyomo.core.base.block import BlockData
from pyomo.core.expr.visitor import identify_variables
import math


def get_vars(m: BlockData):
    vset = ComponentSet()
    for c in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        vset.update(identify_variables(c.body, include_fixed=False))
    for o in m.component_data_objects(pe.Objective, active=True, descend_into=True):
        vset.update(identify_variables(o.expr, include_fixed=False))
    return vset


def shallow_clone(m1):
    m2 = pe.ConcreteModel()
    m2.cons = pe.ConstraintList()

    for con in m1.component_data_objects(pe.Constraint, active=True, descend_into=True):
        m2.cons.add(con.expr)

    objlist = list(m1.component_data_objects(pe.Objective, active=True, descend_into=True))
    assert len(objlist) <= 1
    if objlist:
        obj = objlist[0]
        m2.obj = pe.Objective(expr=obj.expr, sense=obj.sense)

    return m2


def fix_vars_with_equal_bounds(m):
    for v in get_vars(m):
        if v.fixed:
            continue
        if v.lb is not None and v.ub is not None and math.isclose(v.lb, v.ub, abs_tol=1e-4, rel_tol=1e-4):
            v.fix(0.5 * (v.lb + v.ub))
