# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import math

import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.base.block import BlockData
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.vars_from_expressions import get_vars_from_components


def get_vars(m: BlockData):
    return ComponentSet(
        get_vars_from_components(
            m, ctype=(pyo.Constraint, pyo.Objective), include_fixed=False, active=True
        )
    )


def shallow_clone(m1):
    m2 = pyo.ConcreteModel()
    m2.cons = pyo.ConstraintList()

    for con in m1.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        m2.cons.add(con.expr)

    objlist = list(
        m1.component_data_objects(pyo.Objective, active=True, descend_into=True)
    )
    assert len(objlist) <= 1
    if objlist:
        obj = objlist[0]
        m2.obj = pyo.Objective(expr=obj.expr, sense=obj.sense)

    return m2


def fix_vars_with_equal_bounds(m, abs_tol=1e-4, rel_tol=1e-4):
    for v in get_vars(m):
        if v.fixed:
            continue
        if (
            v.lb is not None
            and v.ub is not None
            and math.isclose(v.lb, v.ub, abs_tol=abs_tol, rel_tol=rel_tol)
        ):
            v.fix(0.5 * (v.lb + v.ub))
