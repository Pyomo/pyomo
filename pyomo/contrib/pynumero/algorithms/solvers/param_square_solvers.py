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
    """A base class that solves an NLP object

    Subclasses should implement this interface for compatibility with
    ImplicitFunctionSolver objects.

    """

    def __init__(self, nlp, options=None, timer=None):
        raise NotImplementedError()

    def solve(self, **kwds):
        raise NotImplementedError()


class CyIpoptSolverWrapper(NlpSolverBase):
    """A wrapper for CyIpoptNLP and CyIpoptSolver that implements the
    NlpSolverBase API

    """
    def __init__(self, nlp, options=None, timer=None):
        self._cyipopt_nlp = CyIpoptNLP(nlp)
        self._cyipopt_solver = CyIpoptSolver(self._cyipopt_nlp, options=options)

    def solve(self, **kwds):
        return self._cyipopt_solver.solve(**kwds)


class ScipySolverWrapper(NlpSolverBase):
    """A wrapper for SciPy NLP solvers that implements the NlpSolverBase API

    This solver uses scipy.optimize.fsolve for "vector-valued" NLPs (with more
    than one primal variable and equality constraint) and the Secant-Newton
    hybrid for "scalar-valued" NLPs.

    """
    def __init__(self, nlp, timer=None, options=None):
        if options is None:
            options = {}
        for key in options:
            if (
                key not in SecantNewtonNlpSolver.OPTIONS
                and key not in FsolveNlpSolver.OPTIONS
            ):
                raise ValueError(
                    "Option %s is invalid for both SecantNewtonNlpSolver and"
                    " FsolveNlpSolver" % key
                )
        # Note that options currently contain the options for both solvers.
        # There is currently no way to specify, e.g., different tolerances
        # for the two solvers. This can be updated if there is demand for it.
        newton_options = {
            key: value for key, value in options.items()
            if key in SecantNewtonNlpSolver.OPTIONS
        }
        fsolve_options = {
            key: value for key, value in options.items()
            if key in FsolveNlpSolver.OPTIONS
        }
        if nlp.n_primals() == 1:
            solver = SecantNewtonNlpSolver(nlp, timer=timer, options=options)
        else:
            solver = FsolveNlpSolver(nlp, timer=timer, options=options)
        self._nlp = nlp
        self._options = options
        self._solver = solver

    def solve(self, x0=None):
        res = self._solver.solve(x0=x0)
        return res


class ImplicitFunctionSolver(PyomoImplicitFunctionBase):
    """A basic implicit function solver that uses a ProjectedNLP to solve
    the parameterized system without repeated file writes when parameters
    are updated

    """

    def __init__(
        self,
        variables,
        constraints,
        parameters,
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
        primals = self._nlp.get_primals()
        # Will it cause a problem that _active_parameter_coords is a list
        # rather than array?
        primals[self._active_parameter_coords] = values
        self._nlp.set_primals(primals)
        results = self._solver.solve(**kwds)
        return results

    def evaluate_outputs(self):
        primals = self._nlp.get_primals()
        outputs = primals[self._variable_coords]
        return outputs

    def update_pyomo_model(self):
        primals = self._nlp.get_primals()
        for i, var in enumerate(self.get_variables()):
            var.set_value(primals[self._variable_coords[i]])
        for var, value in zip(self._parameters, self._parameter_values):
            var.set_value(value)


class DecomposedImplicitFunctionBase(PyomoImplicitFunctionBase):
    """A base class for an implicit function that applies a partition
    to its variables and constraints and converges the system by solving
    subsets sequentially

    Subclasses should implement the partition_system method, which
    determines how variables and constraints are partitioned into subsets.

    """

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
            solver_class = ScipySolverWrapper
        self._solver_class = solver_class
        if solver_options is None:
            solver_options = {}
        self._solver_options = solver_options
        self._calc_var_cutoff = 1 if use_calc_var else 0
        # NOTE: This super call is only necessary so the get_* methods work
        super().__init__(variables, constraints, parameters)

        subsystem_list = [
        # Switch order in list for compatibility with generate_subsystem_blocks
            (cons, vars) for vars, cons
                in self.partition_system(variables, constraints)
        ]

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
            # 0 or 1.
            self._solver_subsystem_list = [
                (block, inputs) for block, inputs in self._subsystem_list
                if len(block.vars) > self._calc_var_cutoff
            ]

            # Need a dummy objective to create an NLP
            for block, inputs in self._solver_subsystem_list:
                block._obj = Objective(expr=0.0)
                # I need scaling_factor so Pyomo NLPs I create from these blocks
                # don't break when ProjectedNLP calls get_primals_scaling
                block.scaling_factor = Suffix(direction=Suffix.EXPORT)
                # HACK: scaling_factor just needs to be nonempty
                block.scaling_factor[block._obj] = 1.0

            # Original PyomoNLP for each subset in the partition
            # Since we are creating these NLPs with "constants" fixed, these
            # variables will not show up in the NLPs
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
        # store data in two locations. The current implementation is easier,
        # however.
        self._global_values = np.array(
            [var.value for var in variables + parameters]
        )
        self._global_indices = ComponentMap(
            (var, i) for i, var in enumerate(variables + parameters)
        )
        # Cache the global array-coordinates of each subset of "input"
        # variables. These are used for updating before each solve.
        self._local_input_global_coords = [
            # If I do not fix "constants" above, I get errors here
            # that only show up in the CLC models.
            # TODO: Add a test that covers this edge case.
            np.array(
                [self._global_indices[var] for var in inputs],
                dtype=int,
            ) for (_, inputs) in self._solver_subsystem_list
        ]

        # Cache the global array-coordinates of each subset of "output"
        # variables. These are used for updating after each solve.
        self._output_coords = [
            np.array(
                [self._global_indices[var] for var in block.vars.values()],
                dtype=int,
            ) for (block, _) in self._solver_subsystem_list
        ]

    def partition_system(self, variables, constraints):
        """Partitions the systems of equations defined by the provided
        variables and constraints

        Each subset of the partition should have an equal number of variables
        and equations. These subsets, or "subsystems", will be solved
        sequentially in the order provided by this method instead of solving
        the entire system simultaneously. Subclasses should implement this
        method to define the partition that their implicit function solver
        will use. Partitions are defined as a list of tuples of lists.
        Each tuple has two entries, the first a list of variables, and the
        second a list of constraints. These inner lists should have the
        same number of entries.

        Arguments
        ---------
        variables: list
            List of VarData in the system to be partitioned
        constraints: list
            List of ConstraintData (equality constraints) defining the
            equations of the system to be partitioned

        Returns
        -------
        List of tuples
            List of tuples describing the ordered partition. Each tuple
            contains equal-length subsets of variables and constraints.

        """
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
            if len(block.vars) <= self._calc_var_cutoff:
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

                # Get primals, load potentially new input values into primals,
                # then load primals into NLP
                primals = nlp.get_primals()
                primals[input_coords] = self._global_values[input_global_coords]

                # Set primals in the original NLP. This is necessary so the
                # parameters get updated.
                nlp.set_primals(primals)

                # Get initial guess in the space of variables we solve for
                x0 = proj_nlp.get_primals()
                nlp_solver.solve(x0=x0)

                # Set values in global array. Here we rely on the fact that
                # the projected NLP's primals are in the order that variables
                # were initially specified.
                self._global_values[output_global_coords] = proj_nlp.get_primals()

                solver_subsystem_idx += 1

    def evaluate_outputs(self):
        return self._global_values[:self._n_variables]

    def update_pyomo_model(self):
        # NOTE: Here we rely on the fact that global_values is in the
        # order (variables, parameters)
        for i, var in enumerate(self.get_variables() + self.get_parameters()):
            var.set_value(self._global_values[i])


class SccImplicitFunctionSolver(DecomposedImplicitFunctionBase):

    def partition_system(self, variables, constraints):
        igraph = IncidenceGraphInterface()
        var_blocks, con_blocks = igraph.get_diagonal_blocks(
            variables, constraints
        )
        return zip(var_blocks, con_blocks)
