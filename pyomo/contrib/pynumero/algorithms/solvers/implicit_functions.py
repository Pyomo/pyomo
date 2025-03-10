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

from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.dependencies import attempt_import, numpy as np
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
    TemporarySubsystemManager,
    create_subsystem_block,
    generate_subsystem_blocks,
)

# Use attempt_import here due to unguarded NumPy import in these files
pyomo_nlp = attempt_import('pyomo.contrib.pynumero.interfaces.pyomo_nlp')[0]
nlp_proj = attempt_import('pyomo.contrib.pynumero.interfaces.nlp_projections')[0]
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
    FsolveNlpSolver,
    NewtonNlpSolver,
    SecantNewtonNlpSolver,
)
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.scc_solver import (
    generate_strongly_connected_components,
)


class NlpSolverBase(object):
    """A base class that solves an NLP object

    Subclasses should implement this interface for compatibility with
    ImplicitFunctionSolver objects.

    """

    def __init__(self, nlp, options=None, timer=None):
        raise NotImplementedError(
            "%s has not implemented the __init__ method" % type(self)
        )

    def solve(self, **kwds):
        raise NotImplementedError(
            "%s has not implemented the solve method" % type(self)
        )


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
            key: value
            for key, value in options.items()
            if key in SecantNewtonNlpSolver.OPTIONS
        }
        fsolve_options = {
            key: value
            for key, value in options.items()
            if key in FsolveNlpSolver.OPTIONS
        }
        if nlp.n_primals() == 1:
            solver = SecantNewtonNlpSolver(nlp, timer=timer, options=newton_options)
        else:
            solver = FsolveNlpSolver(nlp, timer=timer, options=fsolve_options)
        self._nlp = nlp
        self._options = options
        self._solver = solver

    def solve(self, x0=None):
        res = self._solver.solve(x0=x0)
        return res


class PyomoImplicitFunctionBase(object):
    """A base class defining an API for implicit functions defined using
    Pyomo components. In particular, this is the API required by
    ExternalPyomoModel.

    Implicit functions are defined by two lists of Pyomo VarData and
    one list of Pyomo ConstraintData. The first list of VarData corresponds
    to "variables" defining the outputs of the implicit function.
    The list of ConstraintData are equality constraints that are converged
    to evaluate the implicit function. The second list of VarData are
    variables to be treated as "parameters" or inputs to the implicit
    function.

    """

    def __init__(self, variables, constraints, parameters):
        """
        Arguments
        ---------
        variables: List of VarData
            Variables to be treated as outputs of the implicit function
        constraints: List of ConstraintData
            Constraints that are converged to evaluate the implicit function
        parameters: List of VarData
            Variables to be treated as inputs to the implicit function

        """
        self._variables = variables
        self._constraints = constraints
        self._parameters = parameters
        self._block_variables = variables + parameters
        self._block = create_subsystem_block(constraints, self._block_variables)

    def get_variables(self):
        return self._variables

    def get_constraints(self):
        return self._constraints

    def get_parameters(self):
        return self._parameters

    def get_block(self):
        return self._block

    def set_parameters(self, values):
        """Sets the parameters of the system that defines the implicit
        function.

        This method does not necessarily need to update values of the Pyomo
        variables, as long as the next evaluation of this implicit function
        is consistent with these inputs.

        Arguments
        ---------
        values: NumPy array
            Array of values to set for the "parameter variables" in the order
            they were specified in the constructor

        """
        raise NotImplementedError()

    def evaluate_outputs(self):
        """Returns the values of the variables that are treated as outputs
        of the implicit function

        The returned values do not necessarily need to be the values stored
        in the Pyomo variables, as long as they are consistent with the
        latest parameters that have been set.

        Returns
        -------
        NumPy array
            Array with values corresponding to the "output variables" in
            the order they were specified in the constructor

        """
        raise NotImplementedError()

    def update_pyomo_model(self):
        """Sets values of "parameter variables" and "output variables"
        to the most recent values set or computed in this implicit function

        """
        raise NotImplementedError()


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
            timer = HierarchicalTimer()
        self._timer = timer
        if solver_options is None:
            solver_options = {}

        self._timer.start("__init__")

        super().__init__(variables, constraints, parameters)
        block = self.get_block()

        # PyomoNLP requires an objective
        block._obj = Objective(expr=0.0)
        # CyIpoptSolver requires a non-empty scaling factor
        block.scaling_factor = Suffix(direction=Suffix.EXPORT)
        block.scaling_factor[block._obj] = 1.0

        self._timer.start("PyomoNLP")
        self._nlp = pyomo_nlp.PyomoNLP(block)
        self._timer.stop("PyomoNLP")
        primals_ordering = [var.name for var in variables]
        self._proj_nlp = nlp_proj.ProjectedExtendedNLP(self._nlp, primals_ordering)

        self._timer.start("NlpSolver")
        if solver_class is None:
            self._solver = ScipySolverWrapper(
                self._proj_nlp, options=solver_options, timer=timer
            )
        else:
            self._solver = solver_class(
                self._proj_nlp, options=solver_options, timer=timer
            )
        self._timer.stop("NlpSolver")

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
                "Invalid model. All variables must appear in specified constraints."
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

        self._timer.start("__init__")

    def set_parameters(self, values, **kwds):
        self._timer.start("set_parameters")
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
        self._timer.start("solve")
        results = self._solver.solve(**kwds)
        self._timer.stop("solve")
        self._timer.stop("set_parameters")
        return results

    def evaluate_outputs(self):
        primals = self._nlp.get_primals()
        outputs = primals[self._variable_coords]
        return outputs

    def update_pyomo_model(self):
        primals = self._nlp.get_primals()
        for i, var in enumerate(self.get_variables()):
            var.set_value(primals[self._variable_coords[i]], skip_validation=True)
        for var, value in zip(self._parameters, self._parameter_values):
            var.set_value(value, skip_validation=True)


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
        self._timer.start("__init__")
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
            (cons, vars)
            for vars, cons in self.partition_system(variables, constraints)
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
                (block, inputs)
                for block, inputs in self._subsystem_list
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
            self._timer.start("PyomoNLP")
            self._solver_subsystem_nlps = [
                pyomo_nlp.PyomoNLP(block)
                for block, inputs in self._solver_subsystem_list
            ]
            self._timer.stop("PyomoNLP")

        # "Output variable" names are required to construct ProjectedNLPs.
        # Ideally, we can eventually replace these with variable indices.
        self._solver_subsystem_var_names = [
            [var.name for var in block.vars.values()]
            for block, inputs in self._solver_subsystem_list
        ]
        self._solver_proj_nlps = [
            nlp_proj.ProjectedExtendedNLP(nlp, names)
            for nlp, names in zip(
                self._solver_subsystem_nlps, self._solver_subsystem_var_names
            )
        ]

        # We will solve the ProjectedNLPs rather than the original NLPs
        self._timer.start("NlpSolver")
        self._nlp_solvers = [
            self._solver_class(nlp, timer=self._timer, options=self._solver_options)
            for nlp in self._solver_proj_nlps
        ]
        self._timer.stop("NlpSolver")
        self._solver_subsystem_input_coords = [
            # Coordinates in the NLP, not ProjectedNLP
            nlp.get_primal_indices(inputs)
            for nlp, (subsystem, inputs) in zip(
                self._solver_subsystem_nlps, self._solver_subsystem_list
            )
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
        self._global_values = np.array([var.value for var in variables + parameters])
        self._global_indices = ComponentMap(
            (var, i) for i, var in enumerate(variables + parameters)
        )
        # Cache the global array-coordinates of each subset of "input"
        # variables. These are used for updating before each solve.
        self._local_input_global_coords = [
            # If I do not fix "constants" above, I get errors here
            # that only show up in the CLC models.
            # TODO: Add a test that covers this edge case.
            np.array([self._global_indices[var] for var in inputs], dtype=int)
            for (_, inputs) in self._solver_subsystem_list
        ]

        # Cache the global array-coordinates of each subset of "output"
        # variables. These are used for updating after each solve.
        self._output_coords = [
            np.array(
                [self._global_indices[var] for var in block.vars.values()], dtype=int
            )
            for (block, _) in self._solver_subsystem_list
        ]

        self._timer.stop("__init__")

    def n_subsystems(self):
        """Returns the number of subsystems in the partition of variables
        and equations used to converge the system defining the implicit
        function

        """
        return len(self._subsystem_list)

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
        raise NotImplementedError(
            "%s has not implemented the partition_system method" % type(self)
        )

    def set_parameters(self, values):
        self._timer.start("set_parameters")
        values = np.array(values)
        #
        # Set parameter values
        #
        # NOTE: Here I rely on the fact that the "global array" is in the
        # order (variables, parameters)
        self._global_values[self._n_variables :] = values

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
                    var.set_value(self._global_values[idx], skip_validation=True)
                # Solve using calculate_variable_from_constraint
                var = block.vars[0]
                con = block.cons[0]
                self._timer.start("solve")
                self._timer.start("calc_var")
                calculate_variable_from_constraint(var, con)
                self._timer.stop("calc_var")
                self._timer.stop("solve")
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
                self._timer.start("solve")
                self._timer.start("solve_nlp")
                nlp_solver.solve(x0=x0)
                self._timer.stop("solve_nlp")
                self._timer.stop("solve")

                # Set values in global array. Here we rely on the fact that
                # the projected NLP's primals are in the order that variables
                # were initially specified.
                self._global_values[output_global_coords] = proj_nlp.get_primals()

                solver_subsystem_idx += 1

        self._timer.stop("set_parameters")

    def evaluate_outputs(self):
        return self._global_values[: self._n_variables]

    def update_pyomo_model(self):
        # NOTE: Here we rely on the fact that global_values is in the
        # order (variables, parameters)
        for i, var in enumerate(self.get_variables() + self.get_parameters()):
            var.set_value(self._global_values[i], skip_validation=True)


class SccImplicitFunctionSolver(DecomposedImplicitFunctionBase):
    def partition_system(self, variables, constraints):
        self._timer.start("partition")
        igraph = IncidenceGraphInterface()
        var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
        self._timer.stop("partition")
        return zip(var_blocks, con_blocks)
