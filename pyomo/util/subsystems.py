#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.block import Block
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.objective import Objective
from pyomo.core.base.external import ExternalFunction
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types, NumericValue


class _ExternalFunctionVisitor(StreamBasedExpressionVisitor):
    def __init__(self, descend_into_named_expressions=True):
        super().__init__()
        self._descend_into_named_expressions = descend_into_named_expressions
        self.named_expressions = []

    def initializeWalker(self, expr):
        self._functions = []
        self._seen = set()
        return True, None

    def beforeChild(self, parent, child, index):
        if child.__class__ in native_types:
            return False, None
        elif (
            not self._descend_into_named_expressions
            and child.is_named_expression_type()
        ):
            self.named_expressions.append(child)
            return False, None
        return True, None

    def exitNode(self, node, data):
        if type(node) is ExternalFunctionExpression:
            if id(node) not in self._seen:
                self._seen.add(id(node))
                self._functions.append(node)

    def finalizeResult(self, result):
        return self._functions


def identify_external_functions(expr):
    yield from _ExternalFunctionVisitor().walk_expression(expr)


def add_local_external_functions(block):
    ef_exprs = []
    named_expressions = []
    visitor = _ExternalFunctionVisitor(descend_into_named_expressions=False)
    for comp in block.component_data_objects(
        (Constraint, Expression, Objective), active=True
    ):
        ef_exprs.extend(visitor.walk_expression(comp.expr))
    named_expr_set = ComponentSet(visitor.named_expressions)
    # List of unique named expressions
    named_expressions = list(named_expr_set)
    while named_expressions:
        expr = named_expressions.pop()
        # Clear named expression cache so we don't re-check named expressions
        # we've seen before.
        visitor.named_expressions.clear()
        ef_exprs.extend(visitor.walk_expression(expr))
        # Only add to the stack named expressions that we have
        # not encountered yet.
        for local_expr in visitor.named_expressions:
            if local_expr not in named_expr_set:
                named_expressions.append(local_expr)
                named_expr_set.add(local_expr)

    unique_functions = []
    fcn_set = set()
    for expr in ef_exprs:
        fcn = expr._fcn
        data = (fcn._library, fcn._function)
        if data not in fcn_set:
            fcn_set.add(data)
            unique_functions.append(data)
    fcn_comp_map = {}
    for lib, name in unique_functions:
        comp_name = unique_component_name(block, "_" + name)
        comp = ExternalFunction(library=lib, function=name)
        block.add_component(comp_name, comp)
        fcn_comp_map[lib, name] = comp
    return fcn_comp_map


def create_subsystem_block(constraints, variables=None, include_fixed=False):
    """This function creates a block to serve as a subsystem with the
    specified variables and constraints. To satisfy certain writers, other
    variables that appear in the constraints must be added to the block as
    well. We call these the "input vars." They may be thought of as
    parameters in the subsystem, but we do not fix them here as it is not
    obvious that this is desired.

    Arguments
    ---------
    constraints: List
        List of Pyomo constraint data objects
    variables: List
        List of Pyomo var data objects
    include_fixed: Bool
        Indicates whether fixed variables should be attached to the block.
        This is useful if they may be unfixed at some point.

    Returns
    -------
    Block containing references to the specified constraints and variables,
    as well as other variables present in the constraints

    """
    if variables is None:
        variables = []
    block = Block(concrete=True)
    block.vars = Reference(variables)
    block.cons = Reference(constraints)
    var_set = ComponentSet(variables)
    input_vars = []
    for var in get_vars_from_components(block, Constraint, include_fixed=include_fixed):
        if var not in var_set:
            input_vars.append(var)
    block.input_vars = Reference(input_vars)
    add_local_external_functions(block)
    return block


def generate_subsystem_blocks(subsystems, include_fixed=False):
    """Generates blocks that contain subsystems of variables and constraints.

    Arguments
    ---------
    subsystems: List of tuples
        Each tuple is a list of constraints then a list of variables
        that will define a subsystem.
    include_fixed: Bool
        Indicates whether to add already fixed variables to the generated
        subsystem blocks.

    Yields
    ------
    "Subsystem blocks" containing the variables and constraints specified
    by each entry in subsystems. Variables in the constraints that are
    not specified are contained in the input_vars component.

    """
    for cons, vars in subsystems:
        block = create_subsystem_block(cons, vars, include_fixed)
        yield block, list(block.input_vars.values())


class TemporarySubsystemManager(object):
    """This class is a context manager for cases when we want to
    temporarily fix or deactivate certain variables or constraints
    in order to perform some solve or calculation with the resulting
    subsystem.

    """

    def __init__(
        self,
        to_fix=None,
        to_deactivate=None,
        to_reset=None,
        to_unfix=None,
        remove_bounds_on_fix=False,
    ):
        """
        Arguments
        ---------
        to_fix: List
            List of var data objects that should be temporarily fixed.
            These are restored to their original status on exit from
            this object's context manager.
        to_deactivate: List
            List of constraint data objects that should be temporarily
            deactivated. These are restored to their original status on
            exit from this object's context manager.
        to_reset: List
            List of var data objects that should be reset to their
            original values on exit from this object's context context
            manager.
        to_unfix: List
            List of var data objects to be temporarily unfixed. These are
            restored to their original status on exit from this object's
            context manager.
        remove_bounds_on_fix: Bool
            Whether bounds should be removed temporarily for fixed variables

        """
        if to_fix is None:
            to_fix = []
        if to_deactivate is None:
            to_deactivate = []
        if to_reset is None:
            to_reset = []
        if to_unfix is None:
            to_unfix = []
        if not ComponentSet(to_fix).isdisjoint(ComponentSet(to_unfix)):
            to_unfix_set = ComponentSet(to_unfix)
            both = [var for var in to_fix if var in to_unfix_set]
            var_names = "\n" + "\n".join([var.name for var in both])
            raise RuntimeError(
                f"Conflicting instructions: The following variables are present"
                " in both to_fix and to_unfix lists: {var_names}"
            )
        self._vars_to_fix = to_fix
        self._cons_to_deactivate = to_deactivate
        self._comps_to_set = to_reset
        self._vars_to_unfix = to_unfix
        self._var_was_fixed = None
        self._con_was_active = None
        self._comp_original_value = None
        self._var_was_unfixed = None
        self._remove_bounds_on_fix = remove_bounds_on_fix
        self._fixed_var_bounds = None

    def __enter__(self):
        to_fix = self._vars_to_fix
        to_deactivate = self._cons_to_deactivate
        to_set = self._comps_to_set
        to_unfix = self._vars_to_unfix
        self._var_was_fixed = [(var, var.fixed) for var in to_fix + to_unfix]
        self._con_was_active = [(con, con.active) for con in to_deactivate]
        self._comp_original_value = [(comp, comp.value) for comp in to_set]
        self._fixed_var_bounds = [(var.lb, var.ub) for var in to_fix]

        for var in self._vars_to_fix:
            if self._remove_bounds_on_fix:
                # TODO: Potentially override var.domain as well?
                var.setlb(None)
                var.setub(None)
            var.fix()

        for con in self._cons_to_deactivate:
            con.deactivate()

        for var in self._vars_to_unfix:
            # As of Pyomo 6.5, attempting to unfix an already unfixed var
            # does not raise an exception. Here we rely on this behavior.
            var.unfix()

        return self

    def __exit__(self, ex_type, ex_val, ex_bt):
        for var, was_fixed in self._var_was_fixed:
            if was_fixed:
                var.fix()
            else:
                var.unfix()
        if self._remove_bounds_on_fix:
            for var, (lb, ub) in zip(self._vars_to_fix, self._fixed_var_bounds):
                var.setlb(lb)
                var.setub(ub)

        for con, was_active in self._con_was_active:
            if was_active:
                con.activate()
        for comp, val in self._comp_original_value:
            comp.set_value(val)


class ParamSweeper(TemporarySubsystemManager):
    """This class enables setting values of variables/parameters
    according to a provided sequence. Iterating over this object
    sets values to the next in the sequence, at which point a
    calculation may be performed and output values compared.
    On exit, original values are restored.

    This is useful for testing a solve that is meant to perform some
    calculation, over a range of values for which the calculation
    is valid. For example:

    .. testcode::
       :skipif: not glpk_available

       model = pyo.ConcreteModel()
       model.v1 = pyo.Var()
       model.v2 = pyo.Var()
       model.c = pyo.Constraint(expr=model.v2 - model.v1 >= 0.1)
       model.o = pyo.Objective(expr=model.v1 + model.v2)
       solver = pyo.SolverFactory('glpk')
       input_vars = [model.v1]
       n_scen = 2
       input_values = pyo.ComponentMap([(model.v1, [1.1, 2.1])])
       output_values = pyo.ComponentMap([(model.v2, [1.2, 2.2])])
       with ParamSweeper(
               n_scen,
               input_values,
               output_values,
               to_fix=input_vars,
               ) as param_sweeper:
           for inputs, outputs in param_sweeper:
               solver.solve(model)
               # inputs and outputs contain the correct values for this
               # instance of the model
               for var, val in outputs.items():
                   # Test that model.v2 was calculated properly.
                   # First that it equals 1.2, then that it equals 2.2
                   assert var.value == val, f"{var.value} != {val}"

    """

    def __init__(
        self,
        n_scenario,
        input_values,
        output_values=None,
        to_fix=None,
        to_deactivate=None,
        to_reset=None,
    ):
        """
        Parameters
        ----------
        n_scenario: Integer
            The number of different values we expect for each input variable
        input_values: ComponentMap
            Maps each input variable to a list of values of length n_scenario
        output_values: ComponentMap
            Maps each output variable to a list of values of length n_scenario
        to_fix: List
            to_fix argument for base class
        to_deactivate: List
            to_deactivate argument for base class
        to_reset: List
            to_reset argument for base class. This list is extended with
            input variables.

        """
        # Should this object be aware of the user's block/model?
        # My answer for now is no.
        self.input_values = input_values
        output = ComponentMap() if output_values is None else output_values
        self.output_values = output
        self.n_scenario = n_scenario
        self.initial_state_values = None
        self._ip = -1  # Index pointer for iteration

        if to_reset is None:
            # Input values will be set repeatedly by iterating over this
            # object. Output values will presumably be altered by some
            # solve within this context. We would like to reset these to
            # their original values to make this functionality less
            # intrusive.
            to_reset = list(input_values) + list(output)
        else:
            to_reset.extend(var for var in input_values)
            to_reset.extend(var for var in output)

        super(ParamSweeper, self).__init__(
            to_fix=to_fix, to_deactivate=to_deactivate, to_reset=to_reset
        )

    def __iter__(self):
        return self

    def __next__(self):
        self._ip += 1

        i = self._ip
        n_scenario = self.n_scenario
        input_values = self.input_values
        output_values = self.output_values

        if i >= n_scenario:
            self._ip = -1
            raise StopIteration()

        else:
            inputs = ComponentMap()
            for var, values in input_values.items():
                val = values[i]
                var.set_value(val)
                inputs[var] = val

            outputs = ComponentMap(
                [(var, values[i]) for var, values in output_values.items()]
            )

            return inputs, outputs
