# -*- coding: utf-8 -*-
"""Implementation of the MindtPy solver.

The MindtPy (MINLP Decomposition Tookit) solver applies a variety of
decomposition-based approaches to solve nonlinear continuous-discrete problems.
These approaches include:

- Outer approximation
- Benders decomposition
- Partial surrogate cuts
- Extended cutting plane

This solver implementation was developed by Carnegie Mellon University in the
research group of Ignacio Grossmann.

For nonconvex problems, the bounds solve_data.LB and solve_data.UB may not be
rigorous. Questions: Please make a post at StackOverflow and/or David Bernal
<https://github.com/bernalde>

"""
from __future__ import division

import logging

import pyomo
from pyomo.common.plugin import Plugin
from pyomo.contrib.mindtpy.initialization import MindtPy_initialize_master
from pyomo.contrib.mindtpy.iterate import MindtPy_iteration_loop
from pyomo.contrib.mindtpy.util import (MindtPySolveData, a_logger,
                                        build_ordered_component_lists,
                                        copy_values, model_is_valid)
from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import (Block, ConstraintList, NonNegativeReals, RangeSet, Set,
                        Suffix, Var)
from pyomo.opt import IOptSolver, SolverResults
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveFloat, PositiveInt, In

logger = logging.getLogger('pyomo.contrib.mindtpy')

__version__ = (0, 1, 0)


class MindtPySolver(Plugin):
    """A decomposition-based MINLP solver.
    """

    pyomo.common.plugin.implements(IOptSolver)
    pyomo.common.plugin.alias(
        'mindtpy',
        doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')

    # _metasolver = False

    CONFIG = ConfigBlock("MindtPy")
    CONFIG.declare("bound_tolerance", ConfigValue(
        default=1E-5,
        domain=PositiveFloat,
        description="Bound tolerance",
        doc="Relative tolerance for bound feasibility checks"
    ))
    CONFIG.declare("iteration_limit", ConfigValue(
        default=30,
        domain=PositiveInt,
        description="Iteration limit",
        doc="Number of maximum iterations in the decomposition methods"
    ))
    CONFIG.declare("strategy", ConfigValue(
        default="OA",
        domain=In(["OA", "GBD", "ECP", "PSC"]),
        description="Decomposition strategy",
        doc="MINLP Decomposition strategy to be applied to the method. "
            "Currently available Outer Approximation (OA), Extended Cutting "
            "Plane (ECP), Partial Surrogate Cuts (PSC), and Generalized "
            "Benders Decomposition (GBD)"
    ))
    CONFIG.declare("init_strategy", ConfigValue(
        default="rNLP",
        domain=In(["rNLP", "initial_binary", "max_binary"]),
        description="Initialization strategy",
        doc="Initialization strategy used by any method. Currently the "
            "continuous relaxation of the MINLP (rNLP), solve a maximal "
            "covering problem (max_binary), and fix the initial value for "
            "the integer variables (initial_binary)"
    ))
    CONFIG.declare("integer_cuts", ConfigValue(
        default=True,
        domain=bool,
        description="Integer cuts",
        doc="Add integer cuts after finding a feasible solution to the MINLP"
    ))
    CONFIG.declare("max_slack", ConfigValue(
        default=1000.0,
        domain=PositiveFloat,
        description="Maximum slack variable",
        doc="Maximum slack variable value allowed for the Outer Approximation "
            "cuts"
    ))
    CONFIG.declare("OA_penalty_factor", ConfigValue(
        default=1000.0,
        domain=PositiveFloat,
        description="Outer Approximation slack penalty factor",
        doc="In the objective function of the Outer Approximation method, the "
            "slack variables correcponding to all the constraints get "
            "multiplied by this number and added to the objective"
    ))
    CONFIG.declare("ECP_tolerance", ConfigValue(
        default=1E-4,
        domain=PositiveFloat,
        description="ECP tolerance",
        doc="Feasibility tolerance used to determine the stopping criterion in"
            "the ECP method. As long as nonlinear constraint are violated for "
            "more than this tolerance, the mothod will keep iterating"
    ))
    CONFIG.declare("nlp_solver", ConfigValue(
        default="ipopt",
        domain=In(["ipopt"]),
        description="NLP subsolver name",
        doc="Which NLP subsolver is going to be used for solving the nonlinear"
            "subproblems"
    ))
    CONFIG.declare("nlp_solver_args", ConfigBlock(
        implicit=True,
        description="NLP subsolver options",
        doc="Which NLP subsolver options to be passed to the solver while "
            "solving the nonlinear subproblems"
    ))
    CONFIG.declare("mip_solver", ConfigValue(
        default="gurobi",
        domain=In(["gurobi", "cplex", "cbc", "glpk"]),
        description="MIP subsolver name",
        doc="Which MIP subsolver is going to be used for solving the mixed-"
            "integer master problems"
    ))
    CONFIG.declare("mip_solver_args", ConfigBlock(
        implicit=True,
        description="MIP subsolver options",
        doc="Which MIP subsolver options to be passed to the solver while "
            "solving the mixed-integer master problems"
    ))
    CONFIG.declare("call_after_master_solve", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every master problem",
        doc="Callback hook after a solution of the master problem."
    ))
    CONFIG.declare("call_after_subproblem_solve", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every subproblem",
        doc="Callback hook after a solution of the nonlinear subproblem."
    ))
    CONFIG.declare("call_after_subproblem_feasible", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every feasible subproblem",
        doc="Callback hook after a feasible solution"
        " of the nonlinear subproblem."
    ))
    CONFIG.declare("tee", ConfigValue(
        default=False,
        description="Stream output to terminal.",
        domain=bool
    ))
    CONFIG.declare("logger", ConfigValue(
        default='pyomo.contrib.mindtpy',
        description="The logger object or name to use for reporting.",
        domain=a_logger
    ))
    CONFIG.declare("small_dual_tolerance", ConfigValue(
        default=1E-8,
        description="When generating cuts, small duals multiplied "
        "by expressions can cause problems. Exclude all duals "
        "smaller in absolue value than the following."
    ))
    CONFIG.declare("integer_tolerance", ConfigValue(
        default=1E-5,
        description="Tolerance on integral values."
    ))
    CONFIG.declare("constraint_tolerance", ConfigValue(
        default=1E-6,
        description="Tolerance on constraint satisfaction."
    ))
    CONFIG.declare("variable_tolerance", ConfigValue(
        default=1E-8,
        description="Tolerance on variable bounds."
    ))
    CONFIG.declare("zero_tolerance", ConfigValue(
        default=1E-15,
        description="Tolerance on variable equal to zero."))
    CONFIG.declare("initial_feas", ConfigValue(
        default=True,
        description="Apply an initial feasibility step.",
        domain=bool
    ))

    # From Qi: this causes issues.
    # I'm not sure exactly why, but commenting for now.
    # __doc__ += CONFIG.generate_yaml_template()

    def available(self, exception_flag=True):
        """Check if solver is available.
        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.
        """
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def solve(self, model, **kwds):
        """Solve the model.
        Warning: this solver is still in beta. Keyword arguments subject to
        change. Undocumented keyword arguments definitely subject to change.
        Warning: at this point in time, if you try to use PSC or GBD with
        anything other than IPOPT as the NLP solver, bad things will happen.
        This is because the suffixes are not in place to extract dual values
        from the variable bounds for any other solver.
        TODO: fix needed with the GBD implementation.
        Args:
            model (Block): a Pyomo model or block to be solved
        """
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        solve_data = MindtPySolveData()
        created_MindtPy_block = False

        old_logger_level = config.logger.getEffectiveLevel()
        try:
            if config.tee and old_logger_level > logging.INFO:
                # If the logger does not already include INFO, include it.
                config.logger.setLevel(logging.INFO)
            config.logger.info("Starting MindtPy")

            # Create a model block on which to store MindtPy-specific utility
            # modeling objects.
            if hasattr(model, 'MindtPy_utils'):
                raise RuntimeError("MindtPy_utils already exists.")
            else:
                created_MindtPy_block = True
                model.MindtPy_utils = Block()

            solve_data.original_model = model

            build_ordered_component_lists(model)
            solve_data.working_model = model.clone()
            MindtPy = solve_data.working_model.MindtPy_utils

            # Store the initial model state as the best solution found. If we
            # find no better solution, then we will restore from this copy.
            solve_data.best_solution_found = model.clone()

            # Create the solver results object
            res = solve_data.results = SolverResults()
            res.problem.name = model.name
            res.problem.number_of_nonzeros = None  # TODO
            res.solver.name = 'MindtPy' + str(config.strategy)
            # TODO work on termination condition and message
            res.solver.termination_condition = None
            res.solver.message = None
            # TODO add some kind of timing
            res.solver.user_time = None
            res.solver.system_time = None
            res.solver.wallclock_time = None
            res.solver.termination_message = None

            # Validate the model to ensure that MindtPy is able to solve it.
            if not model_is_valid(solve_data, config):
                return

            # Create a model block in which to store the generated feasibility
            # slack constraints. Do not leave the constraints on by default.
            feas = MindtPy.MindtPy_feas = Block()
            feas.deactivate()
            feas.feas_constraints = ConstraintList(
                doc='Feasibility Problem Constraints')

            # Create a model block in which to store the generated linear
            # constraints. Do not leave the constraints on by default.
            lin = MindtPy.MindtPy_linear_cuts = Block()
            lin.deactivate()

            # Integer cuts exclude particular discrete decisions
            lin.integer_cuts = ConstraintList(doc='integer cuts')
            # Feasible integer cuts exclude discrete realizations that have
            # been explored via an NLP subproblem. Depending on model
            # characteristics, the user may wish to revisit NLP subproblems
            # (with a different initialization, for example). Therefore, these
            # cuts are not enabled by default.
            #
            # Note: these cuts will only exclude integer realizations that are
            # not already in the primary integer_cuts ConstraintList.
            lin.feasible_integer_cuts = ConstraintList(
                doc='explored integer cuts')
            lin.feasible_integer_cuts.deactivate()

            # Set up iteration counters
            solve_data.nlp_iter = 0
            solve_data.mip_iter = 0
            solve_data.mip_subiter = 0

            # set up bounds
            solve_data.LB = float('-inf')
            solve_data.UB = float('inf')
            solve_data.LB_progress = [solve_data.LB]
            solve_data.UB_progress = [solve_data.UB]

            # Set of NLP iterations for which cuts were generated
            lin.nlp_iters = Set(dimen=1)

            # Set of MIP iterations for which cuts were generated in ECP
            lin.mip_iters = Set(dimen=1)

            lin.nl_constraint_set = RangeSet(
                len(MindtPy.nonlinear_constraints),
                doc="Integer index set over the nonlinear constraints")
            feas.constraint_set = RangeSet(
                len(MindtPy.constraints),
                doc="integer index set over the constraints")
            # Mapping Constraint -> integer index
            MindtPy.nl_map = {}
            # Mapping integer index -> Constraint
            MindtPy.nl_inverse_map = {}
            # Generate the two maps. These maps may be helpful for later
            # interpreting indices on the slack variables or generated cuts.
            for c, n in zip(MindtPy.nonlinear_constraints,
                            lin.nl_constraint_set):
                MindtPy.nl_map[c] = n
                MindtPy.nl_inverse_map[n] = c

            # Mapping Constraint -> integer index
            MindtPy.feas_map = {}
            # Mapping integer index -> Constraint
            MindtPy.feas_inverse_map = {}
            # Generate the two maps. These maps may be helpful for later
            # interpreting indices on the slack variables or generated cuts.
            for c, n in zip(MindtPy.constraints, feas.constraint_set):
                MindtPy.feas_map[c] = n
                MindtPy.feas_inverse_map[n] = c

            # Create slack variables for OA cuts
            lin.slack_vars = Var(lin.nlp_iters, lin.nl_constraint_set,
                                 domain=NonNegativeReals,
                                 bounds=(0, config.max_slack), initialize=0)
            # Create slack variables for feasibility problem
            feas.slack_var = Var(feas.constraint_set,
                                 domain=NonNegativeReals, initialize=1)

            # Flag indicating whether the solution improved in the past
            # iteration or not
            solve_data.solution_improved = False

            if not hasattr(solve_data.working_model, 'ipopt_zL_out'):
                solve_data.working_model.ipopt_zL_out = Suffix(
                    direction=Suffix.IMPORT)
            if not hasattr(solve_data.working_model, 'ipopt_zU_out'):
                solve_data.working_model.ipopt_zU_out = Suffix(
                    direction=Suffix.IMPORT)

            # Initialize the master problem
            MindtPy_initialize_master(solve_data, config)

            # Algorithm main loop
            MindtPy_iteration_loop(solve_data, config)

            # Update values in original model
            copy_values(
                solve_data.best_solution_found,
                solve_data.original_model,
                config)

        finally:
            config.logger.setLevel(old_logger_level)
            if created_MindtPy_block:
                model.del_component('MindtPy_utils')
