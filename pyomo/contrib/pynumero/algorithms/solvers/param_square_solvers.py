#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple

from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
    TemporarySubsystemManager,
    create_subsystem_block,
    generate_subsystem_blocks,
)

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
# This refactor should not strictly *need* ProjectedExtendedNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
    ProjectedExtendedNLP,
    ProjectedNLP,
)
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    cyipopt_available,
    CyIpoptNLP,
    CyIpoptSolver,
)
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
    PyomoImplicitFunctionBase,
    ParameterizedSquareSolver,
)
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
    FsolveNlpSolver,
    RootNlpSolver,
    NewtonNlpSolver,
    SecantNewtonNlpSolver,
)
from pyomo.contrib.incidence_analysis.interface import (
    get_structural_incidence_matrix,
)
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.util import (
    generate_strongly_connected_components,
)
import networkx as nx
import numpy as np


class NlpSolverBase(object):
    """
    A base class that solves an NLP object. Subclasses should implement this
    interface for compatibility with ImplicitFunctionSolver objects.

    """

    def __init__(self, nlp, options=None, timer=None):
        raise NotImplementedError()

    def solve(self, **kwds):
        raise NotImplementedError()


class CyIpoptSolverWrapper(NlpSolverBase):
    """A wrapper for CyIpoptNLP and CyIpoptSolver that implements the
    API required by ParameterizedSquareSolvers.

    """
    def __init__(self, nlp, options=None, timer=None):
        self._cyipopt_nlp = CyIpoptNLP(nlp)
        self._cyipopt_solver = CyIpoptSolver(self._cyipopt_nlp, options=options)

    def solve(self, **kwds):
        return self._cyipopt_solver.solve(**kwds)


# If I include this class, this branch depends on the scipy.optimize.newton.
# Is that okay? Yes. This is very necessary to make sure that the problem
# can be solved with computations entirely in C (i.e. no GIL)
class ScipySolverWrapper(NlpSolverBase):
    def __init__(self, nlp, timer=None, options=None):
        if options is None:
            options = {}
        options = dict(options) # Copy options dict so we don't modify
        if nlp.n_primals() == 1:
            #options["secant"] = True
            solver = NewtonNlpSolver(nlp, timer=timer, options=options)
            solver = SecantNewtonNlpSolver(nlp, timer=timer, options=options)
        else:
            options["method"] = "lm"
            #solver = FsolveNlpSolver(nlp, options=options)
            solver = RootNlpSolver(nlp, timer=timer, options=options)
        self._nlp = nlp
        self._options = options
        self._solver = solver

    def solve(self, x0=None):
        res = self._solver.solve(x0=x0)
        return res


class ImplicitFunctionSolver(PyomoImplicitFunctionBase):

    def __init__(
        self,
        variables,
        constraints,
        parameters,
        # TODO: How to accept a solver as an argument? Accept class or instance?
        solver_class=None,
        solver_options=None,
        timer=None,
    ):
        if timer is None:
            self._timer = HierarchicalTimer()
        if solver_options is None:
            solver_options = {}
        super().__init__(variables, constraints, parameters)
        block = self.get_block()
        block._obj = Objective(expr=0.0)
        self._nlp = PyomoNLP(block)
        primals_ordering = [var.name for var in variables]
        self._proj_nlp = ProjectedExtendedNLP(self._nlp, primals_ordering)

        if solver_class is None:
            self._solver = ScipySolverWrapper(
                self._proj_nlp, options=solver_options, timer=timer
            )
        else:
            self._solver = solver_class(
                self._proj_nlp, options=solver_options, timer=timer
            )

        vars_in_cons = []
        _seen = set()
        for con in constraints:
            for var in identify_variables(con.expr, include_fixed=False):
                if id(var) not in _seen:
                    _seen.add(id(var))
                    vars_in_cons.append(var)
        self._active_var_set = ComponentSet(vars_in_cons)

        # It is possible (and fairly common) for variables specified as
        # parameters to not appear in any of the specified constraints.
        # We will fail if we try to get their coordinates in the NLP.
        #
        # Technically, this could happen for the variables as well. However,
        # this would guarantee that the Jacobian is singular. I will worry
        # about this when I encounter such a case.
        self._active_param_mask = np.array(
            [(p in self._active_var_set) for p in parameters]
        )
        self._active_parameters = [
            p for i, p in enumerate(parameters) if self._active_param_mask[i]
        ]
        if any((var not in self._active_var_set) for var in variables):
            raise RuntimeError(
                "Invalid model. All variables must appear in specified"
                " constriants."
            )

        # These are coordinates in the original NLP
        self._variable_coords = self._nlp.get_primal_indices(variables)
        self._active_parameter_coords = self._nlp.get_primal_indices(
            self._active_parameters
        )

        # NOTE: With this array, we are storing the same data in two locations.
        # Once here, and once in the NLP. We do this because parameters do not
        # *need* to exist in the NLP. However, we still need to be able to
        # update the "parameter variables" in the Pyomo model. If we only stored
        # the parameters in the NLP, we would lose the values for parameters
        # that don't appear in the active constraints.
        self._parameter_values = np.array([var.value for var in parameters])

    def set_parameters(self, values, **kwds):
        # I am not 100% sure the values will always be an array (as opposed to
        # list), so explicitly convert here.
        values = np.array(values)
        self._parameter_values = values
        values = np.compress(self._active_param_mask, values)
        #values.compress(self._active_param_mask)
        primals = self._nlp.get_primals()
        # Will it cause a problem that _parameter_coords is a list rather
        # than array?
        primals[self._active_parameter_coords] = values
        self._nlp.set_primals(primals)
        results = self._solver.solve(**kwds)
        return results

    def evaluate_outputs(self):
        # TODO: out argument?
        primals = self._nlp.get_primals()
        outputs = primals[self._variable_coords]
        return outputs

    def update_pyomo_model(self):
        # NOTE: I am relying on fact that these coords are in lists, rather
        # than NumPy arrays
        primals = self._nlp.get_primals()
        for i, var in enumerate(self.get_variables()):
            var.set_value(primals[self._variable_coords[i]])
        for var, value in zip(self._parameters, self._parameter_values):
            var.set_value(value)


class DecomposedImplicitFunctionBase(PyomoImplicitFunctionBase):

    def __init__(
        self,
        variables,
        constraints,
        parameters,
        solver_class=None,
        solver_options=None,
        timer=None,
        use_calc_var=True,
    ):
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        if solver_class is None:
            #solver_class = ScipySolverWrapper
            solver_class = CyIpoptSolverWrapper
        self._solver_class = solver_class
        if solver_options is None:
            solver_options = {}
        self._solver_options = solver_options
        self._calc_var_cutoff = 2 if use_calc_var else 1
        # TODO: Is there any reason this super call is necessary?
        # Just so the get_* methods work.
        super().__init__(variables, constraints, parameters)

        subsystem_list = [
        # Switch order in list for compatibility with generate_subsystem_blocks
            (cons, vars) for vars, cons in zip(
                *self.partition_system(variables, constraints)
            )
        ]

        # NOTE: In general, there are variables in the constraints that are neither
        # "variables" (outputs) nor "parameters" (inputs). These are
        # "constants", which I would like to completely ignore. I should be able
        # to do this by temporarily fixing them while constructing subsystem
        # blocks (so they don't show up in the blocks' inputs) and NLPs of
        # these blocks (so they don't show up as columns in NLPs).

        var_param_set = ComponentSet(variables + parameters)
        # We will treat variables that are neither variables nor parameters
        # as "constants". These are usually things like area, volume, or some
        # other global parameter that is treated as a variable and "fixed" with
        # an equality constraint.
        constants = []
        constant_set = ComponentSet()
        for con in constraints:
            for var in identify_variables(con.expr, include_fixed=False):
                if var not in constant_set and var not in var_param_set:
                    # If this is a newly encountered variable that is neither
                    # a var nor param, treat it as a "constant"
                    constant_set.add(var)
                    constants.append(var)

        with TemporarySubsystemManager(to_fix=constants):
            # Temporarily fix "constant" variables so (a) they don't show
            # up in the local inputs of the subsystem blocks and (b) so
            # they don't appear as additional columns in the NLPs and
            # ProjectedNLPs.

            self._subsystem_list = list(generate_subsystem_blocks(subsystem_list))
            # These are subsystems that need an external solver, rather than
            # calculate_variable_from_constraint. _calc_var_cutoff should be either
            # 1 or 2.
            self._solver_subsystem_list = [
                (block, inputs) for block, inputs in self._subsystem_list
                if len(block.vars) >= self._calc_var_cutoff
            ]

            # Need a dummy objective to create an NLP
            for block, inputs in self._solver_subsystem_list:
                block._obj = Objective(expr=0.0)
                # I need scaling_factor so Pyomo NLPs I create from these blocks
                # don't break when ProjectedNLP calls get_primals_scaling
                block.scaling_factor = Suffix(direction=Suffix.EXPORT)
                # HACK: scaling_factor just needs to be nonempty.
                block.scaling_factor[block._obj] = 1.0

            # Original PyomoNLP for each subset in the partition
            self._solver_subsystem_nlps = [
                PyomoNLP(block) for block, inputs in self._solver_subsystem_list
            ]

        # "Output variable" names are required to construct ProjectedNLPs.
        # Ideally, we can eventually replace these with variable indices.
        self._solver_subsystem_var_names = [
            [var.name for var in block.vars.values()]
            for block, inputs in self._solver_subsystem_list
        ]
        self._solver_proj_nlps = [
            ProjectedExtendedNLP(nlp, names) for nlp, names in
            zip(self._solver_subsystem_nlps, self._solver_subsystem_var_names)
        ]

        # We will solve the ProjectedNLPs rather than the original NLPs
        self._nlp_solvers = [
            self._solver_class(
                nlp, timer=self._timer, options=self._solver_options
            ) for nlp in self._solver_proj_nlps
        ]
        self._solver_subsystem_input_coords = [
            # Coordinates in the NLP, not ProjectedNLP
            nlp.get_primal_indices(inputs)
            for nlp, (subsystem, inputs) in
            zip(self._solver_subsystem_nlps, self._solver_subsystem_list)
        ]

        self._n_variables = len(variables)
        self._n_constraints = len(constraints)
        self._n_parameters = len(parameters)

        # This is a global (wrt individual subsystems) array that stores
        # the current values of variables and parameters. This is useful
        # for updating values in between subsystem solves.
        #
        # NOTE: This could also be implemented as a tuple of
        # (subsystem_coord, primal_coord), which would eliminate the need to
        # store data in two locations.
        self._global_values = np.array(
            [var.value for var in variables + parameters]
        )
        self._global_indices = ComponentMap(
            (var, i) for i, var in enumerate(variables + parameters)
        )
        self._local_input_global_coords = [
            # I expect to get errors here. Inputs do not necessarily
            # need to be variables or parameters...
            np.array(
                [self._global_indices[var] for var in inputs],
                dtype=np.int,
            ) for (_, inputs) in self._solver_subsystem_list
        ]

        # What coordinates do we need here?
        # - Coordinates of local inputs in each subsystem. How do I update
        #   these before each solve? Don't have a global model (or NLP) to
        #   store values. Having a global array of values seems to make
        #   sense. Update this array after each subsystem solve.
        #
        # local_primals[local_input_coords] = global_values[local_input_global_coords]
        #
        # - Just return the appropriate projection of global_values when outputs
        #   are requested
        # - Need the coordinates of (output) variables in the original NLP.
        #   (Could use the Projected NLP, but original NLP seems cleaner.)
        #   I envision:
        #
        # global_values[0:n_variables] = primals[local_output_coords]
        #
        # Storing local_output_coords may actually not be necessary, as the
        # primals in the ProjectedNLP are already ordered according to the
        # order of the output variables.

        self._output_coords = [
            np.array(
                [self._global_indices[var] for var in block.vars.values()],
                dtype=np.int,
            ) for (block, _) in self._solver_subsystem_list
        ]

        # Will this fail due to parameters not appearing in an NLP? I don't
        # think so, as, for each NLP, I only try to access that NLP's local
        # inputs.
        # This will fail, however, as I try to get the "global coord" of
        # variables that are neither variables nor parameters.

    def partition_system(self, variables, constraints):
        # Subclasses should implement this method, which returns an ordered
        # partition (two lists-of-lists) of variables and constraints.
        raise NotImplementedError()

    def set_parameters(self, values):
        values = np.array(values)
        #
        # Set parameter values
        #
        # NOTE: Here I rely on the fact that the "global array" is in the
        # order (variables, parameters)
        self._global_values[self._n_variables:] = values

        #
        # Solve subsystems one-by-one
        #
        # The basic procedure is: update local information from the global
        # array, solve the subsystem, then update the global array with
        # new values.
        solver_subsystem_idx = 0
        for block, inputs in self._subsystem_list:
            if len(block.vars) < self._calc_var_cutoff:
                # Update model values from global array.
                for var in inputs:
                    idx = self._global_indices[var]
                    var.set_value(self._global_values[idx])
                # Solve using calculate_variable_from_constraint
                var = block.vars[0]
                con = block.cons[0]
                calculate_variable_from_constraint(var, con)
                # Update global array with values from solve
                self._global_values[self._global_indices[var]] = var.value
            else:
                # Transfer variable values into the projected NLP, solve,
                # and extract values.

                i = solver_subsystem_idx
                nlp = self._solver_subsystem_nlps[i]
                proj_nlp = self._solver_proj_nlps[i]
                input_coords = self._solver_subsystem_input_coords[i]
                input_global_coords = self._local_input_global_coords[i]
                output_global_coords = self._output_coords[i]

                nlp_solver = self._nlp_solvers[solver_subsystem_idx]

                # This should no longer be necessary.
                #_, local_inputs = self._solver_subsystem_list[solver_subsystem_idx]

                # Get primals, load potentially new input values into primals,
                # then load primals into NLP
                primals = nlp.get_primals()
                # Variables will not be necessary
                #variables = nlp.get_pyomo_variables()

                primals[input_coords] = self._global_values[input_global_coords]

                # Set values and bounds from inputs to the SCC.
                # This works because values have been set in the original
                # pyomo model, either by a previous SCC solve, or from the
                # "global inputs"
                #
                # This should no longer be necessary
                #for i, var in zip(input_coords, local_inputs):
                #    # Set primals (inputs) in the original NLP
                #    primals[i] = var.value
                # This affects future evaluations in the ProjectedNLP

                # Set primals in the original NLP. This is necessary so the
                # parameters get updated.
                nlp.set_primals(primals)

                x0 = proj_nlp.get_primals()

                #sol, _ = nlp_solver.solve(x0=x0)
                # TODO: Do I need a consistent return value between different
                # solvers? Don't thing so, as long as primals get updated.
                nlp_solver.solve(x0=x0)

                #
                # Set values in global array
                #
                # This would probably work:
                #
                # self._global_values[:self._n_variables] = proj_nlp.get_primals()
                #
                # This is because the projected primals are ordered according
                # to the output variables.
                # ^ This does not work. Need to specify the proper subset of
                #   coordinates in _global_values
                #
                #new_primals = nlp.get_primals()

                #
                self._global_values[output_global_coords] = proj_nlp.get_primals()

                # Set primals from solution in projected NLP. This updates
                # values in the original NLP
                #proj_nlp.set_primals(sol)
                #
                # Values should already be set in the projected NLP...

                # I really only need to set new primals for the variables in
                # the ProjectedNLP. However, I can only get a list of variables
                # from the original Pyomo NLP, so here some of the values I'm
                # setting are redundant.
                #
                # NOTE: This should no longer be necessary.
                #
                #new_primals = nlp.get_primals()
                #assert len(new_primals) == len(variables)
                #for var, val in zip(variables, new_primals):
                #    var.set_value(val, skip_validation=True)

                solver_subsystem_idx += 1

    def evaluate_outputs(self):
        return self._global_values[:self._n_variables]

    def update_pyomo_model(self):
        for i, var in enumerate(self.get_variables() + self.get_parameters()):
            var.set_value(self._global_values[i])


class SccImplicitFunctionSolver(DecomposedImplicitFunctionBase):

    def partition_system(self, variables, constraints):
        igraph = IncidenceGraphInterface()
        var_blocks, con_blocks = igraph.get_diagonal_blocks(
            variables, constraints
        )
        return var_blocks, con_blocks


class _SquareDecompositionSolver(ParameterizedSquareSolver):
    """This is an abstract parameterized solver that applies a decomposition
    to its variables and constraints and solves subsets individually.

    """
    # This class needs to be extendable in two ways:
    # - Specify Pyomo solver (e.g. SolverFactory("ipopt")), then solve blocks
    # - Specify NLP solver. This requires an NLP for each block
    # What are the minimal steps we must perform?
    # - determine decomposition; create blocks
    # - Solve each one-by-one
    def __init__(self):
        self._partition = None

    def determine_partition(self):
        raise NotImplementedError()

    def update_parameters(self):
        raise NotImplementedError()

    def solve(self):
        # Too messy to generalize? We have two options: "standard"
        # or direct/persistent, and these have very different arguments.
        raise NotImplementedError()


class SquareNlpDecompositionSolver(ParameterizedSquareSolver):

    def __init__(
        self,
        model,
        param_vars,
        variables=None,
        timer=None,
        solver_class=None,
        solver_options=None,
        use_calc_var=True,
    ):
        self._model = model
        self._param_vars = param_vars
        self._param_var_set = ComponentSet(param_vars)
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        if solver_class is None:
            solver_class = CyIpoptSolverWrapper #FsolveNlpSolver
        self._solver_class = solver_class
        if solver_options is None:
            solver_options = {}
        self._solver_options = solver_options
        # Whether we use calculate_variable_from_constraint
        self._use_calc_var = use_calc_var
        # Minimum size of a system that will not be solved with calc-var
        self._calc_var_cutoff = 2 if self._use_calc_var else 1

        self.equations = list(
            model.component_data_objects(Constraint, active=True)
        )
        if variables is None:
            variables = [
                # TODO: This should be variables in active constraints.
                v for v in model.component_data_objects(Var)
                if not v.fixed and not v in self._param_var_set
            ]
        self.variables = variables
        if len(self.variables) != len(self.equations):
            raise RuntimeError()

        subsystems = self.partition_system(self.variables, self.equations)
        # Switch order for compatibility with generate_subsystem_blocks
        subsystems = [(eqns, vars) for vars, eqns in subsystems]
        self._subsystem_list = list(generate_subsystem_blocks(subsystems))
        self._solver_subsystem_list = [
            # Subsystems that need a solver (rather than calc-var)
            (block, inputs) for block, inputs in self._subsystem_list
            if len(block.vars) >= self._calc_var_cutoff
        ]

        # Need a dummy objective to create an NLP
        for block, inputs in self._solver_subsystem_list:
            block._obj = Objective(expr=0.0)

            # I need scaling_factor so Pyomo NLPs I create from these blocks
            # don't break when ProjectedNLP calls get_primals_scaling
            block.scaling_factor = Suffix(direction=Suffix.EXPORT)
            # HACK: scaling_factor just needs to be nonempty.
            block.scaling_factor[block._obj] = 1.0

        # These are the "original NLPs" that will be projected
        self._solver_subsystem_nlps = [
            PyomoNLP(block) for block, inputs in self._solver_subsystem_list
        ]
        self._solver_subsystem_var_names = [
            [var.name for var in block.vars.values()]
            for block, inputs in self._solver_subsystem_list
        ]
        self._solver_proj_nlps = [
            ProjectedNLP(nlp, names) for nlp, names in
            #ProjectedExtendedNLP(nlp, names) for nlp, names in
            zip(self._solver_subsystem_nlps, self._solver_subsystem_var_names)
        ]

        # We will solve the ProjectedNLPs rather than the original NLPs
        self._nlp_solvers = [
            self._solver_class(
                nlp, timer=self._timer, options=self._solver_options
            ) for nlp in self._solver_proj_nlps
        ]
        self._solver_subsystem_input_coords = [
            nlp.get_primal_indices(inputs)
            for nlp, (subsystem, inputs) in
            zip(self._solver_subsystem_nlps, self._solver_subsystem_list)
        ]

    def partition_system(self, variables, equations):
        raise NotImplementedError()

    def update_parameters(self, values):
        for var, val in zip(self._param_vars, values):
            var.set_value(val, skip_validation=True)

    def solve(self):
        solver_subsystem_idx = 0
        for block, inputs in self._subsystem_list:
            if len(block.vars) < self._calc_var_cutoff:
                calculate_variable_from_constraint(
                    block.vars[0], block.cons[0]
                )
            else:
                # Transfer variable values into the projected NLP, solve,
                # and extract values.

                nlp = self._solver_subsystem_nlps[solver_subsystem_idx]
                proj_nlp = self._solver_proj_nlps[solver_subsystem_idx]
                input_coords = self._solver_subsystem_input_coords[solver_subsystem_idx]

                nlp_solver = self._nlp_solvers[solver_subsystem_idx]
                _, local_inputs = self._solver_subsystem_list[solver_subsystem_idx]

                primals = nlp.get_primals()
                variables = nlp.get_pyomo_variables()

                # Set values and bounds from inputs to the SCC.
                # This works because values have been set in the original
                # pyomo model, either by a previous SCC solve, or from the
                # "global inputs"
                for i, var in zip(input_coords, local_inputs):
                    # Set primals (inputs) in the original NLP
                    primals[i] = var.value
                # This affects future evaluations in the ProjectedNLP

                nlp.set_primals(primals)

                x0 = proj_nlp.get_primals()

                #sol, _ = nlp_solver.solve(x0=x0)
                # TODO: Do I need a consistent return value between different
                # solvers? Don't thing so, as long as primals get updated.
                nlp_solver.solve(x0=x0)

                # Set primals from solution in projected NLP. This updates
                # values in the original NLP
                #proj_nlp.set_primals(sol)
                #
                # Values should already be set in the projected NLP...

                # I really only need to set new primals for the variables in
                # the ProjectedNLP. However, I can only get a list of variables
                # from the original Pyomo NLP, so here some of the values I'm
                # setting are redundant.
                new_primals = nlp.get_primals()
                assert len(new_primals) == len(variables)
                for var, val in zip(variables, new_primals):
                    var.set_value(val, skip_validation=True)

                solver_subsystem_idx += 1


class SccNlpSolver(SquareNlpDecompositionSolver):

    def partition_system(self, variables, equations):
        igraph = IncidenceGraphInterface()
        var_blocks, con_blocks = igraph.get_diagonal_blocks(
            variables, equations
        )
        return list(zip(var_blocks, con_blocks))
