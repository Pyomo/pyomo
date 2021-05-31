#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet


def create_subsystem_block(constraints, variables=None, include_fixed=False):
    """ This function defines creates a block to serve as a subsystem with
    the specified variables and constraints. To satisfy certain writers,
    other variables that appear in the constraints must be added to the block
    as well. We call these the "input vars." They may be thought of as
    parameters in the subsystem, but we do not fix them here as it is not
    obvious that this is desired.
    """
    if variables is None:
        variables = []
    block = ConcreteModel()
    block.vars = Reference(variables)
    block.cons = Reference(constraints)
    var_set = ComponentSet(variables)
    input_vars = []
    for con in constraints:
        for var in identify_variables(con.body, include_fixed=include_fixed):
            if var not in var_set:
                input_vars.append(var)
                var_set.add(var)
    block.input_vars = Reference(other_vars)
    return block


class SubsystemManager(object):
    """ This class is a context manager for cases when we want to
    temporarily fix or deactivate certain variables or constraints
    in order to perform some solve or calculation with the resulting
    subsystem.

    We currently do not support fixing variables to particular values,
    and do not restore values of variables fixed. This could change.
    """

    def __init__(self, to_fix=None, to_deactivate=None):
        if to_fix == None:
            to_fix = []
        if to_deactivate == None:
            to_deactivate = []
        self._vars_to_fix = to_fix
        self._cons_to_deactivate = to_deactivate
        self._var_was_fixed = None
        self._con_was_active = None

    def __enter__(self):
        to_fix = self._vars_to_fix
        to_deactivate = self._cons_to_deactivate
        self._var_was_fixed = [(var, var.fixed) for var in to_fix]
        self._con_was_active = [(con, con.active) for con in to_deactivate]

        for var in self._vars_to_fix:
            var.fix()

        for con in self._cons_to_deactivate:
            con.deactivate()

        return self

    def __exit__(self, ex_type, ex_val, ex_bt):
        for var, was_fixed in self._var_was_fixed:
            if not was_fixed:
                var.unfix()
        for con, was_active in self._var_was_active:
            if was_active:
                var.activate()
