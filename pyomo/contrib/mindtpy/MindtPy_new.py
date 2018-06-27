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
For nonconvex problems, the bounds solve_data.LB and solve_data.UB may not be rigorous.
Questions: Please make a post at StackOverflow and/or David Bernal <https://github.com/bernalde>

"""
from __future__ import division

import logging
from copy import deepcopy
from math import copysign,fabs
from pprint import pprint

from six import iteritems

from pyutilib.misc.config import ConfigBlock, ConfigValue
from pyomo.contrib.mindtpy.util import MindtPySolveData, _DoNothing, a_logger
import pyomo.common.plugin
from pyomo.core.expr import current as EXPR
from pyomo.core.base import (Block, ComponentUID, Constraint, ConstraintList,
                             Expression, Objective, RangeSet, Set, Suffix, Var,
                             maximize, minimize, value)
from pyomo.core.expr.current import clone_expression
from pyomo.core.base.numvalue import NumericConstant
from pyomo.core.base.symbolic import differentiate
from pyomo.core.kernel import Binary, NonNegativeReals, Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt.base import IOptSolver
from pyomo.repn import generate_standard_repn
from pyomo.opt.results import ProblemSense, SolverResults

from pyomo.core import TransformationFactory
from pyomo.core.base import ComponentMap

from six.moves import range

logger = logging.getLogger('pyomo.contrib.mindtpy')

__version__ = (0, 0, 1)


def _positiveFloat(val):
    ans = float(val)
    if ans <= 0:
        raise ValueError("Expected positive float (got %s)" % (val,))
    return ans


def _positiveInt(val):
    ans = int(val)
    if ans <= 0:
        raise ValueError("Expected positive int (got %s)" % (val,))
    return ans


class _In(object):

    def __init__(self, allowable):
        self.allowable = allowable

    def __call__(self, value):
        if value in self.allowable:
            return value
        raise ValueError("%s not in %s" % (value, self._allowable))


class MindtPySolver(pyomo.common.plugin.Plugin):
    """A decomposition-based MINLP solver.
    """

    pyomo.common.plugin.implements(IOptSolver)
    pyomo.common.plugin.alias('mindtpy',
                            doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')

    # _metasolver = False

    CONFIG = ConfigBlock("MindtPy")
    CONFIG.declare("bound_tolerance", ConfigValue(
        default=1E-5,
        domain=_positiveFloat,
        description="Bound tolerance",
        doc="Relative tolerance for bound feasibility checks"
    ))
    CONFIG.declare("iteration_limit", ConfigValue(
        default=30,
        domain=_positiveInt,
        description="Iteration limit",
        doc="Number of maximum iterations in the decomposition methods"
    ))
    CONFIG.declare("strategy", ConfigValue(
        default="OA",
        domain=_In(["OA","GBD","ECP","PSC"]),
        description="Decomposition strategy",
        doc="MINLP Decomposition strategy to be applied to the method. "
            "Currently available Outer Approximation (OA), Extended Cutting "            'Plane (ECP), Partial Surrogate Cuts (PSC), and Generalized '
            "Benders Decomposition (GBD)"
    ))
    CONFIG.declare("init_strategy", ConfigValue(
        default="rNLP",
        domain=_In(["rNLP","initial_binary","max_binary"]),
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
        domain=_positiveFloat,
        description="Maximum slack variable",
        doc="Maximum slack variable value allowed for the Outer Approximation "
            "cuts"
    ))
    CONFIG.declare("OA_penalty_factor", ConfigValue(
        default=1000.0,
        domain=_positiveFloat,
        description="Outer Approximation slack penalty factor",
        doc="In the objective function of the Outer Approximation method, the "
            "slack variables correcponding to all the constraints get "
            "multiplied by this number and added to the objective"
    ))
    CONFIG.declare("ECP_tolerance", ConfigValue(
        default=1E-4,
        domain=_positiveFloat,
        description="ECP tolerance",
        doc="Feasibility tolerance used to determine the stopping criterion in"
            "the ECP method. As long as nonlinear constraint are violated for "
            "more than this tolerance, the mothod will keep iterating"
    ))
    CONFIG.declare("nlp_solver", ConfigValue(
        default="ipopt",
        domain=_In(["ipopt"]),
        description="NLP subsolver name",
        doc="Which NLP subsolver is going to be used for solving the nonlinear"
            "subproblems"
    ))
    CONFIG.declare("nlp_solver_kwargs", ConfigBlock(
        implicit=True,
        description="NLP subsolver options",
        doc="Which NLP subsolver options to be passed to the solver while "
            "solving the nonlinear subproblems"
    ))
    CONFIG.declare("mip_solver", ConfigValue(
        default="gurobi",
        domain=_In(["gurobi", "cplex", "cbc", "glpk"]),
        description="MIP subsolver name",
        doc="Which MIP subsolver is going to be used for solving the mixed-"
            "integer master problems"
    ))
    CONFIG.declare("mip_solver_kwargs", ConfigBlock(
        implicit=True,
        description="MIP subsolver options",
        doc="Which MIP subsolver options to be passed to the solver while "
            "solving the mixed-integer master problems"
    ))
    CONFIG.declare("modify_in_place", ConfigValue(
        default=True,
        domain=bool,
        description="Solve subproblems directly upon the model",
        doc="If true, MindtPy manipulations are performed directly upon "
            "the model. Otherwise, the model is first copied and solution "
            "values are copied over afterwards."
    ))
    CONFIG.declare("master_postsolve", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every master problem",
        doc="Callback hook after a solution of the master problem."
    ))
    CONFIG.declare("subproblem_postsolve", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every subproblem",
        doc="Callback hook after a solution of the nonlinear subproblem."
    ))
    CONFIG.declare("subproblem_postfeasible", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every feasible subproblem",
        doc="Callback hook after a feasible solution of the nonlinear subproblem."
    ))
    CONFIG.declare("load_solutions", ConfigValue(
        default=True,
        domain=bool,
        description="Solve subproblems directly upon the model",
        doc="if True, load solutions back into the model."
            "This is only relevant if solve_in_place is not True."
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
    CONFIG.declare("initial_feas", ConfigValue(
        default=1,
        description="Apply an initial feasibility step.",
        domain=bool
    ))

    # Qi: this causes issues. I'm not sure exactly why, but commenting for now.
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


        old_logger_level = config.logger.getEffectiveLevel()
        try:
            if config.tee and old_logger_level > logging.INFO:
                # If the logger does not already include INFO, include it.
                config.logger.setLevel(logging.INFO)

            # Modify in place decides whether to run the algorithm on a copy of
            # the originally model passed to the solver, or whether to
            # manipulate the original model directly.
            solve_data.working_model = m = model

            solve_data.current_strategy = config.strategy

            # Create a model block on which to store MindtPy-specific utility
            # modeling objects.
            # TODO check if name is taken already
            MindtPy = m.MindtPy_utils = Block()

            MindtPy.initial_var_list = list(v for v in m.component_data_objects(
                ctype=Var, descend_into=True
            ))
            MindtPy.initial_var_values = list(
                v.value for v in MindtPy.initial_var_list)

            # Store the initial model state as the best solution found. If we
            # find no better solution, then we will restore from this copy.
            # print('Initial clone for best_solution_found')
            solve_data.best_solution_found = model.clone()

            # Save model initial values. These are used later to initialize NLP
            # subproblems.
            MindtPy.initial_var_values = list(
                v.value for v in MindtPy.initial_var_list)

           # Create the solver results object
            res = solve_data.results = SolverResults()
            res.problem.name = m.name
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
            #
            # This needs to take place before the detection of nonlinear
            # constraints, because if the objective is nonlinear, it will be moved
            # to the constraints.
            assert(not hasattr(self, 'nonlinear_constraints'))
            self._validate_model(config, solve_data)

            # Maps in order to keep track of certain generated constraints
            # MindtPy.cut_map = Suffix(direction=Suffix.LOCAL, datatype=None)


            # Create a model block in which to store the generated linear
            # constraints. Do not leave the constraints on by default.
            lin = MindtPy.MindtPy_linear_cuts = Block()
            lin.deactivate()

            # Create a model block in which to store the generated feasibility slack
            # constraints. Do not leave the constraints on by default.
            feas = MindtPy.MindtPy_feas = Block()
            feas.deactivate()
            feas.feas_constraints = ConstraintList(
                doc='Feasibility Problem Constraints')

            # Integer cuts exclude particular discrete decisions
            lin.integer_cuts = ConstraintList(doc='integer cuts')
            # Feasible integer cuts exclude discrete realizations that have been
            # explored via an NLP subproblem. Depending on model characteristics,
            # the user may wish to revisit NLP subproblems (with a different
            # initialization, for example). Therefore, these cuts are not enabled
            # by default.
            #
            # Note: these cuts will only exclude integer realizations that are not
            # already in the primary integer_cuts ConstraintList.
            lin.feasible_integer_cuts = ConstraintList(doc='explored integer cuts')
            lin.feasible_integer_cuts.deactivate()

            # Build a list of binary variables
            MindtPy.binary_vars = [v for v in m.component_data_objects(
                ctype=Var, descend_into=True)
                if v.is_binary() and not v.fixed]

            # Build list of nonlinear constraints
            MindtPy.nonlinear_constraints = [
                v for v in m.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True)
                if v.body.polynomial_degree() not in (0, 1)]

            # Build list of  constraints
            MindtPy.constraints = [
                v for v in m.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True)]

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

            # Create an integer index set over the nonlinear constraints
            lin.nl_constraint_set = RangeSet(len(MindtPy.nonlinear_constraints))
            # Create an integer index set over the constraints
            feas.constraint_set = RangeSet(len(MindtPy.constraints))
            # Mapping Constraint -> integer index
            MindtPy.nl_map = {}
            # Mapping integer index -> Constraint
            MindtPy.nl_inverse_map = {}
            # Generate the two maps. These maps may be helpful for later
            # interpreting indices on the slack variables or generated cuts.
            for c, n in zip(MindtPy.nonlinear_constraints, lin.nl_constraint_set):
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

            # Flag indicating whether the solution improved in the past iteration
            # or not
            solve_data.solution_improved = False

            # Set up solvers
            solve_data.nlp_solver = SolverFactory(config.nlp_solver)
            solve_data.mip_solver = SolverFactory(config.mip_solver)

            # Initialize the master problem
            self._MindtPy_initialize_master(solve_data, config)

            # Algorithm main loop
            self._MindtPy_iteration_loop(solve_data, config)

            # Update values in original model
            if config.load_solutions:
                self._copy_values(solve_data.best_solution_found, model, config)


        finally:
            config.logger.setLevel(old_logger_level)

    def _copy_values(self, from_model, to_model, config):
        """Copy variable values from one model to another."""
        for v_from, v_to in zip(from_model.MindtPy_utils.initial_var_list,
                                to_model.MindtPy_utils.initial_var_list):
            try:
                v_to.set_value(v_from.value)
            except ValueError as err:
                if 'is not in domain Binary' in err.message:
                    # Check to see if this is just a tolerance issue
                    if (fabs(v_from.value - 1) <= config.integer_tolerance or
                            fabs(v_from.value) <= config.integer_tolerance):
                        v_to.set_value(round(v_from.value))
                    else:
                        raise

    def _copy_dual_suffixes(self, from_model, to_model,
                            from_map=None, to_map=None):
        """Copy suffix values from one model to another."""
        self._copy_suffix(from_model.dual, to_model.dual,
                          from_map=from_map, to_map=to_map)
        if hasattr(from_model, 'ipopt_zL_out'):
            self._copy_suffix(from_model.ipopt_zL_out, to_model.ipopt_zL_out,
                              from_map=from_map, to_map=to_map)
        if hasattr(from_model, 'ipopt_zU_out'):
            self._copy_suffix(from_model.ipopt_zU_out, to_model.ipopt_zU_out,
                              from_map=from_map, to_map=to_map)

    def _copy_suffix(self, from_suffix, to_suffix, from_map=None, to_map=None):
        """Copy suffix values from one model to another."""
        if from_map is None:
            from_map = generate_cuid_names(from_suffix.model(),
                                           ctype=(Var, Constraint),
                                           descend_into=True)
        if to_map is None:
            tm_obj_to_uid = generate_cuid_names(
                to_suffix.model(), ctype=(Var, Constraint),
                descend_into=True)
            to_map = dict((cuid, obj)
                          for obj, cuid in iteritems(tm_obj_to_uid))

    def _validate_model(self, config, solve_data):
        """Validate that the model is solveable by MindtPy.
        Also populates results object with problem information.
        """
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        # Check for any integer variables
        if any(True for v in m.component_data_objects(
                ctype=Var, descend_into=True)
                if v.is_integer() and not v.fixed):
            raise ValueError('Model contains unfixed integer variables. '
                             'MindtPy does not currently support solution of '
                             'such problems.')
            # TODO add in the reformulation using base 2

        # Handle missing or multiple objectives
        objs = m.component_data_objects(
            ctype=Objective, active=True, descend_into=True)
        # Fetch the first active objective in the model
        main_obj = next(objs, None)
        if main_obj is None:
            raise ValueError('Model has no active objectives.')
        # Fetch the next active objective in the model
        if next(objs, None) is not None:
            raise ValueError('Model has multiple active objectives.')

        if not hasattr(m, 'dual'):  # Set up dual value reporting
            m.dual = Suffix(direction=Suffix.IMPORT)

        # Move the objective to the constraints
        MindtPy.MindtPy_objective_value = Var(domain=Reals, initialize=0)
        if main_obj.sense == minimize:
            MindtPy.MindtPy_objective_expr = Constraint(
                expr=MindtPy.MindtPy_objective_value >= main_obj.expr)
            m.dual[MindtPy.MindtPy_objective_expr] = 1
            solve_data.results.problem.sense = ProblemSense.minimize
        else:
            MindtPy.MindtPy_objective_expr = Constraint(
                expr=MindtPy.MindtPy_objective_value <= main_obj.expr)
            m.dual[MindtPy.MindtPy_objective_expr] = -1
            solve_data.results.problem.sense = ProblemSense.maximize
        main_obj.deactivate()
        MindtPy.obj = Objective(
            expr=MindtPy.MindtPy_objective_value, sense=main_obj.sense)

        # TODO if any continuous variables are multipled with binary ones, need
        # to do some kind of transformation (Glover?) or throw an error message

    def _MindtPy_initialize_master(self, solve_data, config):
        """Initialize the decomposition algorithm.
        This includes generating the initial cuts require to build the master
        problem.
        """
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        MindtPy.feas_constr_map = {}

        m.dual.activate()
        if not hasattr(m, 'ipopt_zL_out'):
            m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        if not hasattr(m, 'ipopt_zU_out'):
            m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)

        if config.strategy == 'OA':
            # Map Constraint, nlp_iter -> generated OA Constraint
            MindtPy.OA_constr_map = {}
            self._calc_jacobians(solve_data, config)  # preload jacobians
            MindtPy.MindtPy_linear_cuts.oa_cuts = ConstraintList(
                doc='Outer approximation cuts')
        elif config.strategy == 'ECP':
            # Map Constraint, nlp_iter -> generated ECP Constraint
            MindtPy.ECP_constr_map = {}
            self._calc_jacobians(solve_data, config)  # preload jacobians
            MindtPy.MindtPy_linear_cuts.ecp_cuts = ConstraintList(
                doc='Extended Cutting Planes')
        elif config.strategy == 'PSC':
            self._detect_nonlinear_vars(solve_data, config)
            MindtPy.MindtPy_linear_cuts.psc_cuts = ConstraintList(
                doc='Partial surrogate cuts')
        elif config.strategy == 'GBD':
            MindtPy.MindtPy_linear_cuts.gbd_cuts = ConstraintList(
                doc='Generalized Benders cuts')

        # Set default initialization_strategy
        if config.init_strategy is None:
            if config.strategy == 'OA':
                config.init_strategy = 'rNLP'
            else:
                config.init_strategy = 'initial_binary'
        # Do the initialization
        elif config.init_strategy == 'rNLP':
            self._init_rNLP(solve_data, config)
        elif config.init_strategy == 'max_binary':
            self._init_max_binaries(solve_data, config)
            if config.strategy == 'ECP':
                self._add_ecp_cut(solve_data, config)
            else:
                self._solve_NLP_subproblem(solve_data, config)
        elif config.init_strategy == 'initial_binary':
            self._init_initial_binaries(solve_data, config)
            if config.strategy == 'ECP':
                self._add_ecp_cut(solve_data, config)
            else:
                self._solve_NLP_subproblem(solve_data, config)

    def _init_rNLP(self, solve_data, config):
        """Initialize by solving the rNLP (relaxed binary variables)."""
        solve_data.nlp_iter += 1
        m = solve_data.working_model.clone()
        config.logger.info(
            "NLP %s: Solve relaxed integrality" %
                    (solve_data.nlp_iter))
        MindtPy = m.MindtPy_utils
        for v in MindtPy.binary_vars:
            v.domain = NonNegativeReals
            v.setlb(0)
            v.setub(1)
        results = solve_data.nlp_solver.solve(m, options=config.nlp_solver_kwargs)
        for v in MindtPy.binary_vars:
            v.domain = Binary
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            # Add OA cut
            if MindtPy.obj.sense == minimize:
                solve_data.LB = value(MindtPy.obj.expr)
            else:
                solve_data.UB = value(MindtPy.obj.expr)
            config.logger.info(
                'NLP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.nlp_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
            if config.strategy == 'OA':
                self._add_oa_cut(m, solve_data, config)
            elif config.strategy == 'PSC':
                self._add_psc_cut(m, solve_data, config)
            elif config.strategy == 'GBD':
                self._add_gbd_cut(m, solve_data, config)
            elif config.strategy == 'ECP':
                self._add_ecp_cut(m, solve_data, config)
                self._add_objective_linearization(m, solve_data, config)
        elif subprob_terminate_cond is tc.infeasible:
            # TODO fail? try something else?
            config.logger.info(
                'Initial relaxed NLP problem is infeasible. '
                'Problem may be infeasible.')
        else:
            raise ValueError(
                'MindtPy unable to handle relaxed NLP termination condition '
                'of %s. Solver message: %s' %
                (subprob_terminate_cond, results.solver.message))

    def _init_max_binaries(self, solve_data, config):
        """Initialize by turning on as many binary variables as possible.

        The user would usually want to call _solve_NLP_subproblem after an
        invocation of this function.

        """
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        solve_data.mip_subiter += 1
        config.logger.info(
            "MILP %s: maximize value of binaries" %
                    (solve_data.mip_iter))
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()
        MindtPy.obj.deactivate()
        MindtPy.MindtPy_max_binary_obj = Objective(
            expr=sum(v for v in MindtPy.binary_vars), sense=maximize)

        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        results = solve_data.mip_solver.solve(m, options=config.mip_solver_kwargs)

        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        MindtPy.MindtPy_max_binary_obj.deactivate()

        MindtPy.obj.activate()
        for c in MindtPy.nonlinear_constraints:
            c.activate()
        solve_terminate_cond = results.solver.termination_condition
        if solve_terminate_cond is tc.optimal:
            pass  # good
        elif solve_terminate_cond is tc.infeasible:
            raise ValueError(
                'MILP master problem is infeasible. '
                'Problem may have no more feasible '
                'binary configurations.')
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of %s. Solver message: %s' %
                (solve_terminate_cond, results.solver.message))

    def _init_initial_binaries(self, solve_data, config):
        """Initialize by using the intial values of the binary variables.

        The user would usually want to call _solve_NLP_subproblem after an
        invocation of this function.

        """
        pass

    def _add_objective_linearization(self, solve_data, config):
        """Adds initial linearized objective in case it is nonlinear.

        This should be done for initializing the ECP method.

        """
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        solve_data.mip_iter += 1
        gen = (obj for obj in MindtPy.jacs
               if obj is MindtPy.MindtPy_objective_expr)
        MindtPy.MindtPy_linear_cuts.mip_iters.add(solve_data.mip_iter)
        sign_adjust = 1 if MindtPy.obj.sense == minimize else -1
        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        for obj in gen:
            c = MindtPy.MindtPy_linear_cuts.ecp_cuts.add(
                expr=sign_adjust * sum(
                    value(MindtPy.jacs[obj][id(var)]) * (var - value(var))
                    for var in list(EXPR.identify_variables(obj.body))) +
                value(obj.body) <= 0)
            MindtPy.ECP_constr_map[obj, solve_data.mip_iter] = c

    def _MindtPy_iteration_loop(self, solve_data, config):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        # Backup counter to prevent infinite loop
        backup_max_iter = max(1000, config.iteration_limit)
        backup_iter = 0
        while backup_iter < backup_max_iter:
            config.logger.info('')  # print blank lines for visual display
            backup_iter += 1
            # Check bound convergence
            if solve_data.LB + config.bound_tolerance >= solve_data.UB:
                print('MindtPy exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          solve_data.LB, config.bound_tolerance, solve_data.UB) + '\n')
                # res.solver.termination_condition = tc.optimal
                break
            # Check iteration limit
            if solve_data.mip_iter >= config.iteration_limit:
                print('MindtPy unable to converge bounds '
                      'after {} master iterations.'.format(solve_data.mip_iter))
                print('Final bound values: LB: {}  UB: {}'.
                      format(solve_data.LB, solve_data.UB))
                break
            solve_data.mip_subiter = 0
            # solve MILP master problem
            if config.strategy == 'OA':
                self._solve_OA_master(solve_data, config)
            elif config.strategy == 'PSC':
                self._solve_PSC_master(solve_data, config)
            elif config.strategy == 'GBD':
                self._solve_GBD_master(solve_data, config)
            elif config.strategy == 'ECP':
                self._solve_ECP_master(solve_data, config)
            # Check bound convergence
            if solve_data.LB + config.bound_tolerance >= solve_data.UB:
                print('MindtPy exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          solve_data.LB, config.bound_tolerance, solve_data.UB))
                break
            elif config.strategy == 'ECP':
                # Add ECP cut
                self._add_ecp_cut(solve_data, config)
            else:
                # Solve NLP subproblem
                self._solve_NLP_subproblem(solve_data, config)

            # If the hybrid algorithm is not making progress, switch to OA.
            progress_required = 1E-6
            if MindtPy.obj.sense == minimize:
                log = solve_data.LB_progress
                sign_adjust = 1
            else:
                log = solve_data.UB_progress
                sign_adjust = -1
            # Maximum number of iterations in which the lower (optimistic)
            # bound does not improve before switching to OA
            max_nonimprove_iter = 5
            making_progress = True
            for i in range(1, max_nonimprove_iter + 1):
                try:
                    if (sign_adjust * log[-i]
                            <= (log[-i - 1] + progress_required)
                            * sign_adjust):
                        making_progress = False
                    else:
                        making_progress = True
                        break
                except IndexError:
                    # Not enough history yet, keep going.
                    making_progress = True
                    break
            if not making_progress and (
                    config.strategy == 'hPSC' and
                    config.strategy == 'PSC'):
                print('Not making enough progress for {} iterations. '
                      'Switching to OA.'.format(max_nonimprove_iter))
                config.strategy = 'OA'

    def _solve_OA_master(self, solve_data, config):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils
        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))
        # Set up MILP
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()

        MindtPy.MindtPy_linear_cuts.activate()
        MindtPy.obj.deactivate()

        sign_adjust = 1 if MindtPy.obj.sense == minimize else -1
        MindtPy.MindtPy_penalty_expr = Expression(
            expr=sign_adjust * config.OA_penalty_factor * sum(
                v for v in MindtPy.MindtPy_linear_cuts.slack_vars[...]))

        MindtPy.MindtPy_oa_obj = Objective(
            expr=MindtPy.obj.expr + MindtPy.MindtPy_penalty_expr,
            sense=MindtPy.obj.sense)

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        # m.pprint() #print oa master problem for debugging
        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            from copy import deepcopy
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(m, load_solutions=False,
                                            options=config.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)

        MindtPy.obj.activate()
        for c in MindtPy.nonlinear_constraints:
            c.activate()
        MindtPy.MindtPy_linear_cuts.deactivate()
        MindtPy.MindtPy_oa_obj.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)
            self._copy_dual_suffixes(m, solve_data.working_model)

            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.MindtPy_oa_obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(MindtPy.MindtPy_oa_obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary combinations.')
            if solve_data.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
        elif master_terminate_cond is tc.maxTimeLimit:
            # TODO check that status is actually ok and everything is feasible
            config.logger.info(
                'Unable to optimize MILP master problem '
                'within time limit. '
                'Using current solver feasible solution.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(
                    value(MindtPy.obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(
                    value(MindtPy.obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
        elif (master_terminate_cond is tc.other and
                results.solution.status is SolutionStatus.feasible):
            # load the solution and suppress the warning message by setting
            # solver status to ok.
            config.logger.info(
                'MILP solver reported feasible solution, '
                'but not guaranteed to be optimal.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.MindtPy_oa_obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(MindtPy.MindtPy_oa_obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            config.logger.info(
                'MILP master problem is infeasible. '
                'Problem may have no more feasible '
                'binary configurations.')
            if solve_data.mip_iter == 1:
                config.logger.warn(
                    'MindtPy initialization may have generated poor '
                    'quality cuts.')
            # set optimistic bound to infinity
            if MindtPy.obj.sense == minimize:
                solve_data.LB = float('inf')
                solve_data.LB_progress.append(solve_data.UB)
            else:
                solve_data.UB = float('-inf')
                solve_data.UB_progress.append(solve_data.UB)
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of %s. Solver message: %s' %
                (master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)

    def _solve_ECP_master(self, solve_data, config):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils

        feas_sol = 0
        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))
        # Set up MILP
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()
        MindtPy.MindtPy_linear_cuts.activate()

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            from copy import deepcopy
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(m, load_solutions=False,
                                            options=config.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)
        for c in MindtPy.nonlinear_constraints:
            c.activate()
            MindtPy.MindtPy_linear_cuts.deactivate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)

            if all(
                (0 if c.upper is None
                 else (value(c.body) - c.upper)) +
                (0 if c.lower is None
                 else (c.lower - value(c.body)))
                    < config.ECP_tolerance
                    for c in MindtPy.nonlinear_constraints):
                solve_data.best_solution_found = m.clone()
                feas_sol = 1
                print('ECP has found a feasible solution within a {} tolerance'
                      .format(config.ECP_tolerance))
            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
                if feas_sol == 1:
                    solve_data.UB = value(MindtPy.obj.expr)
            else:
                solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
                if feas_sol == 1:
                    solve_data.LB = value(MindtPy.obj.expr)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary combinations.')
            if solve_data.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if MindtPy.obj.sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)

    def _solve_PSC_master(self, solve_data, config):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils

        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))
        # Set up MILP
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()
        MindtPy.MindtPy_linear_cuts.activate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        # m.pprint() #print psc master problem for debugging
        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        for c in MindtPy.nonlinear_constraints:
            c.activate()
        MindtPy.MindtPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)

            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary combinations.')
            if solve_data.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if MindtPy.obj.sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
        else:
            m.solutions.load_from(results)
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)

    def _solve_GBD_master(self, leave_linear_active=True):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils

        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))
        if not leave_linear_active:
            # Deactivate all constraints except those in MindtPy_linear_cuts
            _MindtPy_linear_cuts = set(
                c for c in MindtPy.MindtPy_linear_cuts.component_data_objects(
                    ctype=Constraint, descend_into=True))
            to_deactivate = set(c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
                if c not in _MindtPy_linear_cuts)
            for c in to_deactivate:
                c.deactivate()
        else:
            for c in MindtPy.nonlinear_constraints:
                c.deactivate()
        MindtPy.MindtPy_linear_cuts.activate()
        # m.MindtPy_objective_expr.activate() # This activation will be deleted
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        # m.pprint() #print gbd master problem for debugging
        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it is infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(m, load_solutions=False,
                                            options=config.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)
        if not leave_linear_active:
            for c in to_deactivate:
                c.activate()
        else:
            for c in MindtPy.nonlinear_constraints:
                c.activate()
        MindtPy.MindtPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)

            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary configurations.')
            if solve_data.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if MindtPy.obj.sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
        elif master_terminate_cond is tc.unbounded:
            print('MILP master problem is unbounded. ')
            # Change the integer values to something new, re-solve.
            MindtPy.MindtPy_linear_cuts.activate()
            MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.activate()
            self._init_max_binaries()
            MindtPy.MindtPy_linear_cuts.deactivate()
            MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.deactivate()
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        #
        # MindtPy.MindtPy_linear_cuts.deactivate()
        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)

    def _solve_NLP_subproblem(self, solve_data, config):
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils
        solve_data.nlp_iter += 1
        config.logger.info('NLP %s: Solve subproblem for fixed binaries.'
                           % (solve_data.nlp_iter,))
        # Set up NLP
        for v in MindtPy.binary_vars:
            v.fix(int(value(v) + 0.5))

        # restore original variable values
        for v in m.component_data_objects(ctype=Var, descend_into=True):
            if not v.fixed and not v.is_binary():
                try:
                    v.set_value(self.initial_variable_values[id(v)])
                except KeyError:
                    continue
        #
        MindtPy.MindtPy_linear_cuts.deactivate()
        m.tmp_duals = ComponentMap()
        for c in m.component_data_objects(ctype=Constraint, active=True,
                                          descend_into=True):
            rhs = ((0 if c.upper is None else c.upper) +
                   (0 if c.lower is None else c.lower))
            sign_adjust = 1 if value(c.upper) is None else -1
            m.tmp_duals[c] = sign_adjust * max(0,
                                               sign_adjust * (rhs - value(c.body)))
            # TODO check sign_adjust
        t = TransformationFactory('contrib.deactivate_trivial_constraints')
        t.apply_to(m, tmp=True, ignore_infeasible=True)
        # Solve the NLP
        # m.pprint() # print nlp problem for debugging
        results = config.nlp_solver.solve(m, load_solutions=False,
                                        options=config.nlp_solver_kwargs)
        t.revert(m)
        for v in MindtPy.binary_vars:
            v.unfix()
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)
            for c in m.tmp_duals:
                if m.dual.get(c, None) is None:
                    m.dual[c] = m.tmp_duals[c]
            if MindtPy.obj.sense == minimize:
                solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
                self.solution_improved = solve_data.UB < solve_data.UB_progress[-1]
                solve_data.UB_progress.append(solve_data.UB)
            else:
                solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
                self.solution_improved = solve_data.LB > solve_data.LB_progress[-1]
                solve_data.LB_progress.append(solve_data.LB)
            print('NLP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(solve_data.nlp_iter, value(MindtPy.obj.expr), solve_data.LB,
                          solve_data.UB))
            if self.solution_improved:
                solve_data.best_solution_found = m.clone()
            # Add the linear cut
            if config.strategy == 'OA':
                self._add_oa_cut(m, solve_data, config)
            elif config.strategy == 'PSC':
                self._add_psc_cut(m, solve_data, config)
            elif config.strategy == 'GBD':
                self._add_gbd_cut(m, solve_data, config)

            # This adds an integer cut to the feasible_integer_cuts
            # ConstraintList, which is not activated by default. However, it
            # may be activated as needed in certain situations or for certain
            # values of option flags.
            self._add_int_cut(feasible=True)

            config.subproblem_postfeasible(m, self)
        elif subprob_terminate_cond is tc.infeasible:
            # TODO try something else? Reinitialize with different initial
            # value?
            print('NLP subproblem was locally infeasible.')
            # load the solution and suppress the warning message by setting
            # solver status to ok.
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            for c in m.component_data_objects(ctype=Constraint, active=True,
                                              descend_into=True):
                rhs = ((0 if c.upper is None else c.upper) +
                       (0 if c.lower is None else c.lower))
                sign_adjust = 1 if value(c.upper) is None else -1
                m.dual[c] = sign_adjust * max(0,
                                              sign_adjust * (rhs - value(c.body)))
            for var in m.component_data_objects(ctype=Var,
                                                descend_into=True):

                if config.strategy == 'PSC' or config.strategy == 'GBD':
                    m.ipopt_zL_out[var] = 0
                    m.ipopt_zU_out[var] = 0
                    if var.ub is not None and abs(var.ub - value(var)) < self.bound_tolerance:
                        m.ipopt_zL_out[var] = 1
                    elif var.lb is not None and abs(value(var) - var.lb) < self.bound_tolerance:
                        m.ipopt_zU_out[var] = -1
            # m.pprint() #print infeasible nlp problem for debugging
            if config.strategy == 'PSC':
                print('Adding PSC feasibility cut.')
                self._add_psc_cut(m, solve_data, config, nlp_feasible=False)
            elif config.strategy == 'GBD':
                print('Adding GBD feasibility cut.')
                self._add_gbd_cut(m, solve_data, config, nlp_feasible=False)
            elif config.strategy == 'OA':
                print('Solving feasibility problem')
                if self.initial_feas == 1:
                    self._add_feas_slacks(m, solve_data, config)
                    self.initial_feas = 0
                self._solve_NLP_feas(m, solve_data, config)
                self._add_oa_cut(m, solve_data, config)
            # Add an integer cut to exclude this discrete option
            self._add_int_cut(solve_data, config)
        elif subprob_terminate_cond is tc.maxIterations:
            # TODO try something else? Reinitialize with different initial
            # value?
            print('NLP subproblem failed to converge within iteration limit.')
            # Add an integer cut to exclude this discrete option
            self._add_int_cut(solve_data, config)
        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(subprob_terminate_cond))

        # Call the NLP post-solve callback
        config.subproblem_postsolve(m, solve_data)

    def _solve_NLP_feas(self, solve_data, config):
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils
        MindtPy.MindtPy_objective.deactivate()
        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            constr.deactivate()
        MindtPy.MindtPy_feas.activate()
        MindtPy.MindtPy_feas_obj = Objective(
            expr=sum(s for s in MindtPy.MindtPy_feas.slack_var[...]), sense=minimize)
        for v in MindtPy.binary_vars:
            if value(v) > 0.5:
                v.fix(1)
            else:
                v.fix(0)
        # m.pprint()  #print nlp feasibility problem for debugging
        feas_soln = config.nlp_solver.solve(
            m, load_solutions=False, options=config.nlp_solver_kwargs)
        subprob_terminate_cond = feas_soln.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            m.solutions.load_from(feas_soln)
            self._copy_values(m, solve_data.working_model, config)
        elif subprob_terminate_cond is tc.infeasible:
            raise ValueError('Feasibility NLP infeasible. '
                             'This should never happen.')
        else:
            raise ValueError(
                'MindtPy unable to handle feasibility NLP termination condition '
                'of {}'.format(subprob_terminate_cond))

        for v in MindtPy.binary_vars:
            v.unfix()

        MindtPy.MindtPy_feas.deactivate()
        MindtPy.MindtPy_feas_obj.deactivate()
        # MindtPy.MindtPy_objective_expr.activate()
        MindtPy.MindtPy_objective.activate()

        for constr in m.component_data_objects(
                ctype=Constraint, descend_into=True):
            constr.activate()
            rhs = ((0 if constr.upper is None else constr.upper) +
                   (0 if constr.lower is None else constr.lower))
            sign_adjust = 1 if value(constr.upper) is None else -1
            m.dual[constr] = sign_adjust * max(0,
                                               sign_adjust * (rhs - value(constr.body)))

        if value(MindtPy.MindtPy_feas_obj.expr) == 0:
            raise ValueError(
                'Problem is not infeasible, check NLP solver')

    def _solve_LP_subproblem(self, solve_data, config):
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils
        """Solve continuous relaxation of MILP (relaxed binary variables)."""
        solve_data.nlp_iter += 1
        print("LP {}: Solve continuous relaxation.".format(solve_data.nlp_iter))
        for v in MindtPy.binary_vars:
            v.domain = NonNegativeReals
            v.setlb(0)
            v.setub(1)
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()
        MindtPy.MindtPy_linear_cuts.activate()
        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        for v in MindtPy.binary_vars:
            v.domain = Binary
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()
        MindtPy.MindtPy_linear_cuts.activate()
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)
            # Add the linear cut
            self._add_ecp_cut(m, solve_data, config)

            # This adds an integer cut to the feasible_integer_cuts
            # ConstraintList, which is not activated by default. However, it
            # may be activated as needed in certain situations or for certain
            # values of option flags.
            self._add_int_cut(solve_data, config, feasible=True)
        elif subprob_terminate_cond is tc.infeasible:
            # TODO fail? try something else? this should never happen
            raise ValueError('Relaxed LP infeasible. '
                             'This should never happen.')
            self._add_int_cut(solve_data, config)
        else:
            raise ValueError(
                'MindtPy unable to handle relaxed LP termination condition '
                'of {}'.format(subprob_terminate_cond))
        # Call the LP post-solve callback
        config.subproblem_postsolve(m, solve_data)

    def _calc_jacobians(self, solve_data, config):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        MindtPy.jacs = {}
        for c in MindtPy.nonlinear_constraints:
            constraint_vars = list(EXPR.identify_variables(c.body))
            jac_list = differentiate(c.body, wrt_list=constraint_vars)
            MindtPy.jacs[c] = {id(var): jac
                            for var, jac in zip(constraint_vars, jac_list)}

    def _add_oa_cut(self, solve_data, config):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        MindtPy.MindtPy_linear_cuts.nlp_iters.add(solve_data.nlp_iter)
        sign_adjust = -1 if MindtPy.obj.sense == minimize else 1

        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        for constr in MindtPy.nonlinear_constraints:
            rhs = ((0 if constr.upper is None else constr.upper) +
                   (0 if constr.lower is None else constr.lower))
            MindtPy.pprint()
            c = MindtPy.MindtPy_linear_cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * m.dual[constr]) * (sum(
                    value(MindtPy.jacs[constr][id(var)]) * (var - value(var))
                    for var in list(EXPR.identify_variables(constr.body))) +
                    value(constr.body) - rhs) +
                MindtPy.MindtPy_linear_cuts.slack_vars[solve_data.nlp_iter,
                                                 MindtPy.nl_map[constr]] <= 0)
            MindtPy.OA_constr_map[constr, solve_data.nlp_iter] = c

    def _add_feas_slacks(self, solve_data, config):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        # generate new constraints
        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            rhs = ((0 if constr.upper is None else constr.upper) +
                   (0 if constr.lower is None else constr.lower))
            c = MindtPy.MindtPy_feas.feas_constraints.add(constr.body - rhs
                                                    <= MindtPy.MindtPy_feas.slack_var[self.feas_map[constr]])
            MindtPy.feas_constr_map[constr, solve_data.nlp_iter] = c

    def _add_ecp_cut(self, solve_data, config):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        MindtPy.MindtPy_linear_cuts.mip_iters.add(solve_data.mip_iter)
        sign_adjust = -1 if MindtPy.obj.sense == minimize else 1
        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        gen = (constr for constr in MindtPy.jacs
               if (0 if constr.upper is None
                   else abs(value(constr.body) - constr.upper)) +
               (0 if constr.lower is None
                else abs(constr.lower - value(constr.body)))
               > config.ECP_tolerance)
        for constr in gen:
            constr_dir = -1 if value(constr.upper) is None else 1
            rhs = ((0 if constr.upper is None else constr.upper) +
                   (0 if constr.lower is None else constr.lower))
            # this only happens if a constraint is >=
            c = MindtPy.MindtPy_linear_cuts.ecp_cuts.add(
                expr=copysign(1, constr_dir)
                * (sum(value(MindtPy.jacs[constr][id(var)]) * (var - value(var))
                       for var in list(EXPR.identify_variables(constr.body))) +
                   value(constr.body) - rhs) <= 0)
            MindtPy.ECP_constr_map[constr, solve_data.mip_iter] = c

    def _add_psc_cut(self, solve_data, config, nlp_feasible=True):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils

        sign_adjust = 1 if MindtPy.obj.sense == minimize else -1

        # generate the sum of all multipliers with the nonlinear constraints
        var_to_val = {id(var): NumericConstant(value(var))
                      for var in MindtPy.nonlinear_variables}
        sum_nonlinear = (
            # Address constraints of form f(x) <= upper
            sum(value(m.dual[c]) * -1 *
                (clone_expression(c.body, substitute=var_to_val) - c.upper)
                for c in MindtPy.nonlinear_constraints
                if value(fabs(m.dual[c])) > config.small_dual_tolerance
                and c.upper is not None) +
            # Address constraints of form f(x) >= lower
            sum(value(m.dual[c]) *
                (c.lower - clone_expression(c.body, substitute=var_to_val))
                for c in MindtPy.nonlinear_constraints
                if value(fabs(m.dual[c])) > config.small_dual_tolerance
                and c.lower is not None))
        # Generate the sum of all multipliers with linear constraints
        # containing nonlinear variables
        #
        # For now, need to generate canonical representation in order to get
        # the coefficients on the linear terms.
        lin_cons = [c for c in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True)
            if c.body.polynomial_degree() in (0, 1)]
        # Create a coefficient dictionary mapping variables to their
        # coefficient in the expression. Constraint -> (id(Var) -> coefficient)
        coef_dict = {}
        constr_vars = {}
        for constr in lin_cons:
            repn = generate_standard_repn(constr.body)
            if repn.variables is None or repn.linear is None:
                repn.variables = []
                repn.linear = []
            coef_dict[constr] = {id(var): coef for var, coef in
                                 zip(repn.variables, repn.linear)}
            constr_vars[constr] = repn.variables
        sum_linear = sum(
            m.dual[c] *
            sum(coef_dict[c][id(var)] * (var - value(var))
                for var in constr_vars[c]
                if id(var) in solve_data.nonlinear_variable_IDs)
            for c in lin_cons
            if value(fabs(m.dual[c])) > config.small_dual_tolerance)

        # Generate the sum of all bound multipliers with nonlinear variables
        sum_var_bounds = (
            sum(m.ipopt_zL_out.get(var, 0) * (var - value(var))
                for var in MindtPy.nonlinear_variables
                if value(fabs(m.ipopt_zL_out.get(var, 0))) >
                config.small_dual_tolerance) +
            sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
                for var in MindtPy.nonlinear_variables
                if value(fabs(m.ipopt_zU_out.get(var, 0))) >
                config.small_dual_tolerance))

        if nlp_feasible:
            # Optimality cut (for feasible NLP)
            MindtPy.MindtPy_linear_cuts.psc_cuts.add(
                expr=MindtPy.obj.expr * sign_adjust >= sign_adjust * (
                    MindtPy.obj.expr + sum_nonlinear + sum_linear +
                    sum_var_bounds))
        else:
            # Feasibility cut (for infeasible NLP)
            MindtPy.MindtPy_linear_cuts.psc_cuts.add(
                expr=(sum_nonlinear + sum_linear + sum_var_bounds) <= 0)

    def _add_gbd_cut(self, solve_data, config, nlp_feasible=True):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils

        sign_adjust = 1 if MindtPy.obj.sense == minimize else -1

        for c in m.component_data_objects(ctype=Constraint, active=True,
                                          descend_into=True):
            if value(c.upper) is None and value(c.lower) is None:
                raise ValueError(
                    'Oh no, Pyomo did something MindtPy does not expect. '
                    'The value of c.upper for {} is None: {} <= {} <= {}'
                    .format(c.name, c.lower, c.body, c.upper))
        # TODO handle the case where constraint upper and lower is None

        # only substitute non-binary variables to their values
        binary_var_ids = set(id(var) for var in MindtPy.binary_vars)
        var_to_val = {id(var): NumericConstant(value(var))
                      for var in m.component_data_objects(ctype=Var,
                                                          descend_into=True)
                      if id(var) not in binary_var_ids}
        # generate the sum of all multipliers with the active (from a duality
        # sense) constraints
        sum_constraints = (
            sum(value(m.dual[c]) * -1 *
                (clone_expression(c.body, substitute=var_to_val) - c.upper)
                for c in m.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True)
                if value(fabs(m.dual[c])) > config.small_dual_tolerance
                and c.upper is not None) +
            sum(value(m.dual[c]) *
                (c.lower - clone_expression(c.body, substitute=var_to_val))
                for c in m.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True)
                if value(fabs(m.dual[c])) > config.small_dual_tolerance
                and c.lower is not None))
        # and not c.upper == c.lower

        # add in variable bound dual contributions
        #
        # Generate the sum of all bound multipliers with nonlinear variables
        sum_var_bounds = (
            sum(m.ipopt_zL_out.get(var, 0) * (var - value(var))
                for var in m.component_data_objects(ctype=Var,
                                                    descend_into=True)
                if (id(var) not in binary_var_ids and
                    value(abs(m.ipopt_zL_out.get(var, 0))) >
                    config.small_dual_tolerance)) +
            sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
                for var in m.component_data_objects(ctype=Var,
                                                    descend_into=True)
                if (id(var) not in binary_var_ids and
                    value(abs(m.ipopt_zU_out.get(var, 0))) >
                    config.small_dual_tolerance)))

        if nlp_feasible:
            MindtPy.MindtPy_linear_cuts.gbd_cuts.add(
                expr=MindtPy.obj.expr * sign_adjust >= sign_adjust * (
                    value(MindtPy.obj.expr) + sum_constraints + sum_var_bounds))
        else:
            if sum_constraints + sum_var_bounds != 0:
                MindtPy.MindtPy_linear_cuts.gbd_cuts.add(
                    expr=(sum_constraints + sum_var_bounds) <= 0)

    def _add_int_cut(self, solve_data, config, feasible=False):
        if config.integer_cuts:
            m = solve_data.working_model
            MindtPy = m.MindtPy_utils
            # check to make sure that binary variables are all 0 or 1
            for v in MindtPy.binary_vars:
                if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                    raise ValueError('Binary {} = {} is not 0 or 1'.format(
                        v.name, value(v)))

            if not MindtPy.binary_vars:  # if no binary variables, skip.
                return

            int_cut = (sum(1 - v for v in MindtPy.binary_vars
                           if value(abs(v - 1)) <= int_tol) +
                       sum(v for v in MindtPy.binary_vars
                           if value(abs(v)) <= int_tol) >= 1)

            if not feasible:
                # Add the integer cut
                MindtPy.MindtPy_linear_cuts.integer_cuts.add(expr=int_cut)
            else:
                MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.add(expr=int_cut)

    def _detect_nonlinear_vars(self, solve_data, config):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        """Identify the variables that participate in nonlinear terms."""
        MindtPy.nonlinear_variables = []
        # This is a workaround because Var is not hashable, and I do not want
        # duplicates in self.nonlinear_variables.
        seen = set()
        for constr in MindtPy.nonlinear_constraints:
            if isinstance(constr.body, EXPR._SumExpression):
                # go through each term and check to see if the term is
                # nonlinear
                for expr in constr.body._args:
                    # Check to see if the expression is nonlinear
                    if expr.polynomial_degree() not in (0, 1):
                        # collect variables
                        for var in EXPR.identify_variables(
                                expr, include_fixed=False):
                            if id(var) not in seen:
                                seen.add(id(var))
                                MindtPy.nonlinear_variables.append(var)
            # if the root expression object is not a summation, then something
            # else is the cause of the nonlinearity. Collect all participating
            # variables.
            else:
                # collect variables
                for var in EXPR.identify_variables(constr.body,
                                                   include_fixed=False):
                    if id(var) not in seen:
                        seen.add(id(var))
                        MindtPy.nonlinear_variables.append(var)
        MindtPy.nonlinear_variable_IDs = set(
            id(v) for v in MindtPy.nonlinear_variables)
