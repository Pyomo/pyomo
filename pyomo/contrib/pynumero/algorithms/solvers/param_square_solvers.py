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

from pyomo.common.collections import ComponentSet
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
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
