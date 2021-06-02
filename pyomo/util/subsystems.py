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
from pyomo.common.collections import ComponentSet, ComponentMap


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
    block.input_vars = Reference(input_vars)
    return block


class TemporarySubsystemManager(object):
    """ This class is a context manager for cases when we want to
    temporarily fix or deactivate certain variables or constraints
    in order to perform some solve or calculation with the resulting
    subsystem.

    We currently do not support fixing variables to particular values,
    and do not restore values of variables fixed. This could change.
    """

    def __init__(self, to_fix=None, to_deactivate=None, to_reset=None):
        if to_fix is None:
            to_fix = []
        if to_deactivate is None:
            to_deactivate = []
        if to_reset is None:
            to_reset = []
        self._vars_to_fix = to_fix
        self._cons_to_deactivate = to_deactivate
        self._comps_to_set = to_reset
        self._var_was_fixed = None
        self._con_was_active = None
        self._comp_original_value = None

    def __enter__(self):
        to_fix = self._vars_to_fix
        to_deactivate = self._cons_to_deactivate
        to_set = self._comps_to_set
        self._var_was_fixed = [(var, var.fixed) for var in to_fix]
        self._con_was_active = [(con, con.active) for con in to_deactivate]
        self._comp_original_value = [(comp, comp.value) for comp in to_set]

        for var in self._vars_to_fix:
            var.fix()

        for con in self._cons_to_deactivate:
            con.deactivate()

        return self

    def __exit__(self, ex_type, ex_val, ex_bt):
        for var, was_fixed in self._var_was_fixed:
            if not was_fixed:
                var.unfix()
        for con, was_active in self._con_was_active:
            if was_active:
                con.activate()
        for comp, val in self._comp_original_value:
            comp.set_value(val)

"""
Could iterate over a param sweeper object, or could iterate over
parameters, and create a new context for each...

subsystem = SubsystemManager(to_fix, to_deactivate)
param_sweep = ParamSweeper(
    inputs=ComponentMap([(var, vals) for var, vals in input_data),
    outputs=ComponentMap([(var, vals) for var, vals in output_data),
    n_scenario=n_scenario,
    )
with subsystem:
    with param_sweep:
        for inputs, outputs in param_sweep:
            solver.solve(block)
            for var, val in outputs.items():
                assert var.value == val

-------
versus:
-------

input_data = [
    ComponentMap([(var, vals[i]) for var, vals in input_data])
    for i in range(n_scenario)
    ]
output_data = [
    ComponentMap([(var, vals[i]) for var, vals in output_data])
    for i in range(n_scenario)
    ]
subsystem = SubsystemManager(to_fix, to_deactivate)
with subsystem:
    for inputs, outputs in zip(input_data, output_data):
        with InputSetter(inputs):
            solver.solve(block)
            for var, val in outputs.items():
                assert var.value == val

-----------
Comparison:
-----------
Former:
    - data format of inputs and outputs that user has to deal with
      is more intuitive. Dict of lists rather than list of dicts.
      => keys (components) are only stored once.
    - Can naturally combine the two context managers
    - Could we avoid iteration if n_scenario == 1?
    ^ The more I think about it, the more I think the base
    SubsystemManager should support setting and restoring values.
    This would special handling of the n == 1 case less important.

Latter:
    - Context manager is less opaque. Avoids iterating over context
      manager, which may be confusing.
    - But combining with the "subsystem" manager would repeat the
      fixing/deactivating work.
    - More natural if we only have one set of inputs we want to test;
      don't need to iterate in this case.
"""


class ParamSweeper(TemporarySubsystemManager):
    """ This class enables setting values of variables/parameters
    according to a provided sequence. Iterating over this object
    sets values to the next in the sequence, at which point a
    calculation may be performed and output values compared.
    On exit, original values are restored.
    """

    def __init__(self,
            n_scenario,
            input_values,
            output_values=None,
            to_fix=None,
            to_deactivate=None,
            ):
        """
        Parameters
        ----------
        n_scenario: The number of different values we expect for each
                    input variable
        input_values: ComponentMap mapping each input variable to a list
                      of values of length n_scenario
        output_values: ComponentMap mapping each output variable to a list
                       of values of length n_scenario
        """
        # Should this object be aware of the user's block/model?
        # My answer for now is no.
        self.input_values = input_values
        self.output_values = output_values if output_values is not None else {}
        self.n_scenario = n_scenario
        self.initial_state_values = None
        self._ip = -1 # Index pointer for iteration

        super(ParamSweeper, self).__init__(
                to_fix=to_fix,
                to_deactivate=to_deactivate,
                )

    def __enter__(self):
        # Store initial values of input vars
        self.initial_input_values = ComponentMap([
            (var, var.value) for var in self.input_values
            ])

        # TODO: Maybe alter the values of the inputs here?
        # Or this could be handled by the call to super.__enter__
        # if I expand the base class's functionality

        # Fix and deactivate if necessary
        return super(ParamSweeper, self).__enter__()

    def __exit__(self, ex_type, ex_val, ex_bt):
        # I don't think order should matter here.
        res = super(ParamSweeper, self).__exit__(ex_type, ex_val, ex_bt)
        
        for var, val in self.initial_input_values.items():
            var.set_value(val)

        return res

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

            outputs = ComponentMap([
                (var, values[i]) for var, values in output_values.items()
                ])

            return inputs, outputs
