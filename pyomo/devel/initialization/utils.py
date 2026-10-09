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
from pyomo.contrib.solver.solvers.scip.scip_direct import ScipDirect
from pyomo.contrib.solver.solvers.scip.scip_persistent import ScipPersistent
from pyomo.contrib.solver.solvers.gurobi.gurobi_direct_base import GurobiDirectBase
from pyomo.contrib.solver.solvers.highs import Highs


def get_vars(m: BlockData):
    return ComponentSet(
        get_vars_from_components(
            m, ctype=(pyo.Constraint, pyo.Objective), include_fixed=False, active=True
        )
    )


def get_solution_limit_options(solver):
    opts = {}
    if isinstance(mip_solver, (ScipDirect, ScipPersistent)):
        opts['limits/solutions'] = 1
    elif isinstance(mip_solver, GurobiDirectBase):
        opts['SolutionLimit'] = 1
    elif isinstance(mip_solver, Highs):
        opts['mip_max_improving_sols'] = 1
    else:
        raise NotImplementedError(
            'Currently, the initialization module only works with new solver '
            'interfaces, so the mip solvers are limited to Highs, ScipDirect, '
            'ScipPersistent, GurobiDirect, GurobiDirectMINLP, and GurobiPersistent.'
        )
    return opts


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
