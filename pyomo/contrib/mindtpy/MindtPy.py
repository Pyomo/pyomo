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

For nonconvex problems, the bounds self.LB and self.UB may not be rigorous.

Questions: Please make a post at StackOverflow and/or David Bernal <https://github.com/bernalde>

"""
import logging
from copy import deepcopy
from math import copysign
from pprint import pprint

from pyutilib.misc.config import ConfigBlock, ConfigValue
import pyomo.util.plugin
from pyomo.core.base import expr as EXPR
from pyomo.core.base import (Block, ComponentUID, Constraint, ConstraintList,
                             Expression, Objective, RangeSet, Set, Suffix, Var,
                             maximize, minimize, value)
from pyomo.core.base.expr_common import clone_expression
from pyomo.core.base.numvalue import NumericConstant
from pyomo.core.base.symbolic import differentiate
from pyomo.core.kernel import Binary, NonNegativeReals, Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt.base import IOptSolver
from pyomo.repn.canonical_repn import generate_canonical_repn
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

class _DoNothing(object):
    """Do nothing, literally.

    This class is used in situations of "do something if attribute exists."
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return "(Nothing)"

    def __getattr__(self, attr):
        def _do_nothing(*args, **kwargs):
            pass
        return _do_nothing


class MindtPySolver(pyomo.util.plugin.Plugin):
    """A decomposition-based MINLP solver.

    Arguments:
    """

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('mindtpy',
                            doc='The MindtPy decomposition-based MINLP solver')

    CONFIG = ConfigBlock("MindtPy")
    CONFIG.declare('bound_tolerance', ConfigValue(
        default=1E-5,
        domain=_positiveFloat,
        description='Bound tolerance',
        doc='Relative tolerance for bound feasibility checks'))
    CONFIG.declare('iteration_limit', ConfigValue(
        default=30,
        domain=_positiveInt,
        description='Iterations limit',
        doc='Number of maximum iterations in the decomposition methods'))
    CONFIG.declare('strategy', ConfigValue(
        default='OA',
        domain=_In(['OA', 'GBD', 'ECP', 'PSC']),
        description='Decomposition strategy',
        doc='MINLP Decomposition strategy to be applied to the method. '
            'Currently available Outer Approximation (OA), Extended Cutting '
            'Plane (ECP), Partial Surrogate Cuts (PSC), and Generalized '
            'Benders Decomposition (GBD)'))
    CONFIG.declare('init_strategy', ConfigValue(
        default='rNLP',
        domain=_In(['rNLP', 'initial_binary', 'max_binary']),
        description='Initialization strategy',
        doc='Initialization strategy used by any method. Currently the '
            'continuous relaxation of the MINLP (rNLP), solve a maximal '
            'covering problem (max_binary), and fix the initial value for '
            'the integer variables (initial_binary)'))
    CONFIG.declare('integer_cuts', ConfigValue(
        default=True,
        domain=bool,
        description='Integer cuts',
        doc='Add integer cuts after finding a feasible solution to the MINLP'))
    CONFIG.declare('max_slack', ConfigValue(
        default=1000.0,
        domain=_positiveFloat,
        description='Maximum slack variable',
        doc='Maximum slack variable value allowed for the Outer Approximation '
            'cuts'))
    CONFIG.declare('OA_penalty_factor', ConfigValue(
        default=1000.0,
        domain=_positiveFloat,
        description='Outer Approximation slack penalty factor',
        doc='In the objective function of the Outer Approximation method, the '
            'slack variables correcponding to all the constraints get '
            'multiplied by this number and added to the objective'))
    CONFIG.declare('ECP_tolerance', ConfigValue(
        default=1E-4,
        domain=_positiveFloat,
        description='ECP tolerance',
        doc='Feasibility tolerance used to determine the stopping criterion in'
            'the ECP method. As long as nonlinear constraint are violated for '
            'more than this tolerance, the mothod will keep iterating'))
    CONFIG.declare('nlp_solver', ConfigValue(
        default='ipopt',
        domain=_In(['ipopt']),
        description='NLP subsolver name',
        doc='Which NLP subsolver is going to be used for solving the nonlinear'
            'subproblems'))
    CONFIG.declare('nlp_solver_kwargs', ConfigBlock(
        implicit=True,
        description='NLP subsolver options',
        doc='Which NLP subsolver options to be passed to the solver while '
            'solving the nonlinear subproblems'))
    CONFIG.declare('mip_solver', ConfigValue(
        default='gurobi',
        domain=_In(['gurobi', 'cplex', 'cbc', 'glpk']),
        description='MIP subsolver name',
        doc='Which MIP subsolver is going to be used for solving the mixed-'
            'integer master problems'))
    CONFIG.declare('mip_solver_kwargs', ConfigBlock(
        implicit=True,
        description='MIP subsolver options',
        doc='Which MIP subsolver options to be passed to the solver while '
            'solving the mixed-integer master problems'))
    CONFIG.declare('modify_in_place', ConfigValue(
        default=True,
        domain=bool,
        description='Solve subproblems directly upon the model',
        doc='If true, MindtPy manipulations are performed directly upon '
            'the model. Otherwise, the model is first copied and solution '
            'values are copied over afterwards.'))
    CONFIG.declare('master_postsolve', ConfigValue(
        default=_DoNothing(),
        domain=None,
        description='Function to be executed after every master problem',
        doc='Callback hook after a solution of the master problem.'))
    CONFIG.declare('subproblem_postsolve', ConfigValue(
        default=_DoNothing(),
        domain=None,
        description='Function to be executed after every subproblem',
        doc='Callback hook after a solution of the nonlinear subproblem.'))
    CONFIG.declare('subproblem_postfeasible', ConfigValue(
        default=_DoNothing(),
        domain=None,
        description='Function to be executed after every feasible subproblem',
        doc='Callback hook after a feasible solution of the nonlinear subproblem.'))
    CONFIG.declare('load_solutions', ConfigValue(
        default=True,
        domain=bool,
        description='Solve subproblems directly upon the model',
        doc='if True, load solutions back into the model.'
            'This is only relevant if solve_in_place is not True.'))

    __doc__ += CONFIG.generate_yaml_template()

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

        TODO: something is wrong with the GBD implementation, I think...

        Args:
            model (Block): a Pyomo model or block to be solved

        Kwargs:
            tol (float): bound tolerance
            iterlim (int): maximum number of master iterations
            strategy (str): decomposition strategy to use. Possible values:
                OA, PSC, GBD, hPSC, ECP
            init_strategy (str): initialization strategy to use when generating
                the initial cuts to construct the master problem.
            int_cuts (str): use of integer cuts in the implementation.
            max_slack (float): upper bound on slack variable values
            OA_penalty (float): multiplier on objective penalization for slack
                variables.
            nlp (str): Solver to use for nonlinear subproblems
            nlp_kwargs (dict): Keyword arguments to pass to NLP solver
            mip (str): Solver to use for linear discrete problems
            mip_kwargs (dict): Keyword arguments to pass to MIP solver
            solve_in_place (bool): If true, MindtPy manipulations are performed
                directly upon the model. Otherwise, the model is first copied
                and solution values are copied over afterwards.
            master_postsolve (func): callback hook after a solution of the
                master problem
            subprob_postsolve (func): callback hook after a solution of the
                nonlinear subproblem
            subprob_postfeas (func): callback hook after feasible solution of
                the nonlinear subproblem
            load_solutions (bool): if True, load solutions back into the model.
                This is only relevant if solve_in_place is not True.

        """ 
        # + MindtPy.CONFIG.generate_yaml_template()

        config = self.CONFIG().set_value(kwds)

        # self.bound_tolerance = kwds.pop('tol', 1E-5)
        # self.iteration_limit = kwds.pop('iterlim', 30)
        # self.strategy = kwds.pop('strategy', 'OA')
        # self.initialization_strategy = kwds.pop('init_strategy', None)
        # self.integer_cuts = kwds.pop('int_cuts', 1)
        # self.max_slack = kwds.pop('max_slack', 1000)
        # self.OA_penalty_factor = kwds.pop('OA_penalty', 1000)
        # self.ECP_tolerance = kwds.pop('ECP_tolerance', 1E-4)
        # self.nlp_solver_name = kwds.pop('nlp', 'ipopt')
        # self.nlp_solver_kwargs = kwds.pop('nlp_kwargs', {})
        # self.mip_solver_name = kwds.pop('mip', 'gurobi')
        # self.mip_solver_kwargs = kwds.pop('mip_kwargs', {})
        # self.modify_in_place = kwds.pop('solve_in_place', True)
        # self.master_postsolve = kwds.pop('master_postsolve', _DoNothing())
        # self.subproblem_postsolve = kwds.pop('subprob_postsolve', _DoNothing())
        # self.subproblem_postfeasible = kwds.pop('subprob_postfeas',
        #                                         _DoNothing())
        # self.load_solutions = kwds.pop('load_solutions', True)
        self.tee = kwds.pop('tee', False)

        if self.tee:
            old_logger_level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)

        # if kwds:
        #     print("Unrecognized arguments passed to MindtPy solver:")
        #     pprint(kwds)

        valid_strategies = ['OA', 'PSC', 'GBD', 'ECP']
        if config.strategy not in valid_strategies:
            raise ValueError('Unrecognized decomposition strategy %s. '
                             'Valid strategies include: %s'
                             % (solve_data.strategy,
                                valid_strategies))

        # If decomposition strategy is a hybrid, set the initial strategy
        if config.strategy == 'hPSC':
            config.strategy = 'PSC'
        else:
            config.strategy = config.strategy

        # When generating cuts, small duals multiplied by expressions can cause
        # problems. Exclude all duals smaller in absolue value than the
        # following.
        self.small_dual_tolerance = 1E-8
        self.integer_tolerance = 1E-5
        self.initial_feas = 1

        # Modify in place decides whether to run the algorithm on a copy of the
        # originally model passed to the solver, or whether to manipulate the
        # original model directly.
        if config.modify_in_place:
            self.m = m = model
        else:
            self.m = m = model.clone()

        # Store the initial model state as the best solution found. If we find
        # no better solution, then we will restore from this copy.
        self.best_solution_found = model.clone()

        # Save model initial values. These are used later to initialize NLP
        # subproblems.
        self.initial_variable_values = {
            id(v): v.value for v in m.component_data_objects(
                ctype=Var, descend_into=True)}

        # Create the solver results object
        res = self.results = SolverResults()
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
        self._validate_model()

        # Create a model block in which to store the generated linear
        # constraints. Do not leave the constraints on by default.
        lin = m.MindtPy_linear_cuts = Block()
        lin.deactivate()

        # Create a model block in which to store the generated feasibility slack
        # constraints. Do not leave the constraints on by default.
        feas = m.MindtPy_feas = Block()
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
        self.binary_vars = [v for v in m.component_data_objects(
            ctype=Var, descend_into=True)
            if v.is_binary() and not v.fixed]

        # Build list of nonlinear constraints
        self.nonlinear_constraints = [
            v for v in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
            if v.body.polynomial_degree() not in (0, 1)]

        # Build list of  constraints
        self.constraints = [
            v for v in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)]

        # Set up iteration counters
        self.nlp_iter = 0
        self.mip_iter = 0
        self.mip_subiter = 0

        # Set of NLP iterations for which cuts were generated
        lin.nlp_iters = Set(dimen=1)

        # Set of MIP iterations for which cuts were generated in ECP
        lin.mip_iters = Set(dimen=1)

        # Create an integer index set over the nonlinear constraints
        lin.nl_constraint_set = RangeSet(len(self.nonlinear_constraints))
        # Create an integer index set over the constraints
        feas.constraint_set = RangeSet(len(self.constraints))
        # Mapping Constraint -> integer index
        self.nl_map = {}
        # Mapping integer index -> Constraint
        self.nl_inverse_map = {}
        # Generate the two maps. These maps may be helpful for later
        # interpreting indices on the slack variables or generated cuts.
        for c, n in zip(self.nonlinear_constraints, lin.nl_constraint_set):
            self.nl_map[c] = n
            self.nl_inverse_map[n] = c

        # Mapping Constraint -> integer index
        self.feas_map = {}
        # Mapping integer index -> Constraint
        self.feas_inverse_map = {}
        # Generate the two maps. These maps may be helpful for later
        # interpreting indices on the slack variables or generated cuts.
        for c, n in zip(self.constraints, feas.constraint_set):
            self.feas_map[c] = n
            self.feas_inverse_map[n] = c

        # Create slack variables for OA cuts
        lin.slack_vars = Var(lin.nlp_iters, lin.nl_constraint_set,
                             domain=NonNegativeReals,
                             bounds=(0, config.max_slack), initialize=0)
        # Create slack variables for feasibility problem
        feas.slack_var = Var(feas.constraint_set,
                             domain=NonNegativeReals, initialize=1)

        # set up bounds
        self.LB = float('-inf')
        self.UB = float('inf')
        self.LB_progress = [self.LB]
        self.UB_progress = [self.UB]

        # Flag indicating whether the solution improved in the past iteration
        # or not
        self.solution_improved = False

        # Set up solvers
        self.nlp_solver = SolverFactory(config.nlp_solver)
        self.mip_solver = SolverFactory(config.mip_solver)

        # Initialize the master problem
        self._MindtPy_initialize_master()

        # Algorithm main loop
        self._MindtPy_iteration_loop()

        # Update values in original model
        if config.load_solutions:
            for v in self.best_solution_found.component_data_objects(
                    ctype=Var, descend_into=True):
                uid = ComponentUID(v)
                orig_model_var = uid.find_component_on(model)
                if orig_model_var is not None:
                    try:
                        orig_model_var.set_value(v.value)
                    except ValueError as err:
                        if 'is not in domain Binary' in err.message:
                            # check to see whether this is just a tolerance
                            # issue
                            if (value(abs(v - 1)) <= self.integer_tolerance or
                                    value(abs(v)) <= self.integer_tolerance):
                                orig_model_var.set_value(round(v.value))
                            else:
                                raise

    def _validate_model(self):
        m = self.m
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
        m.MindtPy_objective_value = Var(domain=Reals, initialize=0)
        if main_obj.sense == minimize:
            m.MindtPy_objective_expr = Constraint(
                expr=m.MindtPy_objective_value >= main_obj.expr)
            m.dual[m.MindtPy_objective_expr] = 1
        else:
            m.MindtPy_objective_expr = Constraint(
                expr=m.MindtPy_objective_value <= main_obj.expr)
            m.dual[m.MindtPy_objective_expr] = -1
        main_obj.deactivate()
        self.obj = m.MindtPy_objective = Objective(
            expr=m.MindtPy_objective_value, sense=main_obj.sense)

        # TODO if any continuous variables are multipled with binary ones, need
        # to do some kind of transformation (Glover?) or throw an error message

    def _MindtPy_initialize_master(self):
        """Initialize the decomposition algorithm.

        This includes generating the initial cuts require to build the master
        problem.

        """
        m = self.m
        config = self.CONFIG()
        self.feas_constr_map = {}
        if (config.strategy == 'OA' or
                self.strategy == 'hPSC'):
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            # Map Constraint, nlp_iter -> generated OA Constraint
            self.OA_constr_map = {}
            self._calc_jacobians()  # preload jacobians
            self.m.MindtPy_linear_cuts.oa_cuts = ConstraintList(
                doc='Outer approximation cuts')
        if config.strategy == 'ECP':
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            # Map Constraint, nlp_iter -> generated ECP Constraint
            self.ECP_constr_map = {}
            self._calc_jacobians()  # preload jacobians
            self.m.MindtPy_linear_cuts.ecp_cuts = ConstraintList(
                doc='Extended Cutting Planes')
        if config.strategy == 'PSC':
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zL_out'):
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zU_out'):
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            self._detect_nonlinear_vars()
            self.m.MindtPy_linear_cuts.psc_cuts = ConstraintList(
                doc='Partial surrogate cuts')
        if config.strategy == 'GBD':
            if not hasattr(m, 'dual'):
                m.dual = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zL_out'):
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zU_out'):
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            self.m.MindtPy_linear_cuts.gbd_cuts = ConstraintList(
                doc='Generalized Benders cuts')

        # Set default initialization_strategy
        if config.init_strategy is None:
            if config.strategy == 'OA':
                config.init_strategy = 'rNLP'
            else:
                config.init_strategy = 'initial_binary'

        # Do the initialization
        if config.init_strategy == 'rNLP':
            self._init_rNLP()
        elif config.init_strategy == 'max_binary':
            self._init_max_binaries()
            if config.strategy == 'ECP':
                self._add_ecp_cut()
            else:
                self._solve_NLP_subproblem()
        elif config.init_strategy == 'initial_binary':
            self._init_initial_binaries()
            if config.strategy == 'ECP':
                self._add_ecp_cut()
            else:
                self._solve_NLP_subproblem()

    def _init_rNLP(self):
        """Initialize by solving the rNLP (relaxed binary variables)."""
        config = self.CONFIG()
        self.nlp_iter += 1
        print("NLP {}: Solve relaxed integrality.".format(self.nlp_iter))
        for v in self.binary_vars:
            v.domain = NonNegativeReals
            v.setlb(0)
            v.setub(1)
        results = self.nlp_solver.solve(self.m, options=config.nlp_solver_kwargs)
        for v in self.binary_vars:
            v.domain = Binary
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            # Add OA cut
            if self.obj.sense == minimize:
                self.LB = value(self.obj.expr)
            else:
                self.UB = value(self.obj.expr)
            print('NLP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.nlp_iter, value(self.obj.expr), self.LB,
                          self.UB))
            if config.strategy == 'OA':
                self._add_oa_cut()
            elif config.strategy == 'PSC':
                self._add_psc_cut()
            elif config.strategy == 'GBD':
                self._add_gbd_cut()
            elif config.strategy == 'ECP':
                self._add_ecp_cut()
                self._add_objective_linearization()
        elif subprob_terminate_cond is tc.infeasible:
            # TODO fail? try something else?
            raise ValueError('Initial relaxed NLP infeasible. '
                             'Problem may be infeasible.')
        else:
            raise ValueError(
                'MindtPy unable to handle relaxed NLP termination condition '
                'of {}'.format(subprob_terminate_cond))

    def _init_max_binaries(self):
        """Initialize by turning on as many binary variables as possible.

        The user would usually want to call _solve_NLP_subproblem after an
        invocation of this function.

        """
        self.mip_subiter += 1
        m = self.m
        print("MILP {}.{}: maximize value of binaries".format(
            self.mip_iter, self.mip_subiter))
        for c in self.nonlinear_constraints:
            c.deactivate()
        self.obj.deactivate()
        m.MindtPy_max_binary_obj = Objective(
            expr=sum(v for v in self.binary_vars), sense=maximize)
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        results = self.mip_solver.solve(m, options=self.mip_solver_kwargs)
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()
        m.del_component(m.MindtPy_max_binary_obj)
        self.obj.activate()
        for c in self.nonlinear_constraints:
            c.activate()
        solve_terminate_cond = results.solver.termination_condition
        if solve_terminate_cond is tc.optimal:
            pass  # good
        elif solve_terminate_cond is tc.infeasible:
            raise ValueError('Linear relaxation is infeasible. '
                             'Problem is infeasible.')
        else:
            raise ValueError('Cannot handle termination condition {}'.format(
                solve_terminate_cond))

    def _init_initial_binaries(self):
        """Initialize by using the intial values of the binary variables.

        The user would usually want to call _solve_NLP_subproblem after an
        invocation of this function.

        """
        pass

    def _add_objective_linearization(self):
        """Adds initial linearized objective in case it is nonlinear.

        This should be done for initializing the ECP method.

        """
        m = self.m
        self.mip_iter += 1
        gen = (obj for obj in self.jacs
               if obj is m.MindtPy_objective_expr)
        m.MindtPy_linear_cuts.mip_iters.add(self.mip_iter)
        sign_adjust = 1 if self.obj.sense == minimize else -1
        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        for obj in gen:
            c = m.MindtPy_linear_cuts.ecp_cuts.add(
                expr=sign_adjust * sum(
                    value(self.jacs[obj][id(var)]) * (var - value(var))
                    for var in list(EXPR.identify_variables(obj.body))) +
                value(obj.body) <= 0)
            self.ECP_constr_map[obj, self.mip_iter] = c

    def _MindtPy_iteration_loop(self):
        # Backup counter to prevent infinite loop
        config = self.CONFIG()
        backup_max_iter = max(1000, config.iteration_limit)
        backup_iter = 0
        while backup_iter < backup_max_iter:
            print('\n')  # print blank lines for visual display
            backup_iter += 1
            # Check bound convergence
            if self.LB + config.bound_tolerance >= self.UB:
                print('MindtPy exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          self.LB, config.bound_tolerance, self.UB) + '\n')
                # res.solver.termination_condition = tc.optimal
                break
            # Check iteration limit
            if self.mip_iter >= config.iteration_limit:
                print('MindtPy unable to converge bounds '
                      'after {} master iterations.'.format(self.mip_iter))
                print('Final bound values: LB: {}  UB: {}'.
                      format(self.LB, self.UB))
                break
            self.mip_subiter = 0
            # solve MILP master problem
            if config.strategy == 'OA':
                self._solve_OA_master()
            elif config.strategy == 'PSC':
                self._solve_PSC_master()
            elif config.strategy == 'GBD':
                self._solve_GBD_master()
            elif config.strategy == 'ECP':
                self._solve_ECP_master()
            # Check bound convergence
            if self.LB + config.bound_tolerance >= self.UB:
                print('MindtPy exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          self.LB, config.bound_tolerance, self.UB))
                break
            elif config.strategy == 'ECP':
                # Add ECP cut
                self._add_ecp_cut()
            else:
                # Solve NLP subproblem
                self._solve_NLP_subproblem()

            # If the hybrid algorithm is not making progress, switch to OA.
            progress_required = 1E-6
            if self.obj.sense == minimize:
                log = self.LB_progress
                sign_adjust = 1
            else:
                log = self.UB_progress
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
                    self.strategy == 'hPSC' and
                    self.strategy == 'PSC'):
                print('Not making enough progress for {} iterations. '
                      'Switching to OA.'.format(max_nonimprove_iter))
                self.strategy = 'OA'

    def _solve_OA_master(self):
        config = self.CONFIG()
        m = self.m
        self.mip_iter += 1
        print('MILP {}: Solve master problem.'.format(self.mip_iter))
        # Set up MILP
        for c in self.nonlinear_constraints:
            c.deactivate()
        m.MindtPy_linear_cuts.activate()
        self.obj.deactivate()
        m.del_component('MindtPy_penalty_expr')
        sign_adjust = 1 if self.obj.sense == minimize else -1
        m.MindtPy_penalty_expr = Expression(
            expr=sign_adjust * config.OA_penalty_factor * sum(
                v for v in m.MindtPy_linear_cuts.slack_vars[...]))
        m.del_component('MindtPy_oa_obj')
        m.MindtPy_oa_obj = Objective(
            expr=self.obj.expr + m.MindtPy_penalty_expr,
            sense=self.obj.sense)
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        # m.pprint() #print oa master problem for debugging
        results = self.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            from copy import deepcopy
            old_options = deepcopy(self.mip_solver.options)
            # This solver option is specific to Gurobi.
            self.mip_solver.options['DualReductions'] = 0
            results = self.mip_solver.solve(m, load_solutions=False,
                                            options=self.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            self.mip_solver.options.update(old_options)
        self.obj.activate()
        for c in self.nonlinear_constraints:
            c.activate()
        m.MindtPy_linear_cuts.deactivate()
        m.MindtPy_oa_obj.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if self.obj.sense == minimize:
                self.LB = max(value(m.MindtPy_oa_obj.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(m.MindtPy_oa_obj.expr), self.UB)
                self.UB_progress.append(self.UB)
            print('MILP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.mip_iter, value(m.MindtPy_oa_obj.expr), self.LB,
                          self.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary combinations.')
            if self.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        config.master_postsolve(m, self)

    def _solve_ECP_master(self):
        m = self.m
        self.mip_iter += 1

        feas_sol = 0
        print('MILP {}: Solve master problem.'.format(self.mip_iter))
        # Set up MILP
        for c in self.nonlinear_constraints:
            c.deactivate()
        m.MindtPy_linear_cuts.activate()
        results = self.mip_solver.solve(m, load_solutions=False,
                                        options=self.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            from copy import deepcopy
            old_options = deepcopy(self.mip_solver.options)
            # This solver option is specific to Gurobi.
            self.mip_solver.options['DualReductions'] = 0
            results = self.mip_solver.solve(m, load_solutions=False,
                                            options=self.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            self.mip_solver.options.update(old_options)
        for c in self.nonlinear_constraints:
            c.activate()
            m.MindtPy_linear_cuts.deactivate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if all(
                (0 if c.upper is None
                 else (value(c.body) - c.upper)) +
                (0 if c.lower is None
                 else (c.lower - value(c.body)))
                    < self.ECP_tolerance
                    for c in self.nonlinear_constraints):
                self.best_solution_found = m.clone()
                feas_sol = 1
                print('ECP has found a feasible solution within a {} tolerance'
                      .format(self.ECP_tolerance))
            if self.obj.sense == minimize:
                self.LB = max(value(self.obj.expr), self.LB)
                self.LB_progress.append(self.LB)
                if feas_sol == 1:
                    self.UB = value(self.obj.expr)
            else:
                self.UB = min(value(self.obj.expr), self.UB)
                self.UB_progress.append(self.UB)
                if feas_sol == 1:
                    self.LB = value(self.obj.expr)
            print('MILP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.mip_iter, value(self.obj.expr), self.LB,
                          self.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary combinations.')
            if self.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        else:
            m.solutions.load_from(results)
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_PSC_master(self):
        m = self.m
        self.mip_iter += 1
        print('MILP {}: Solve master problem.'.format(self.mip_iter))
        # Set up MILP
        for c in self.nonlinear_constraints:
            c.deactivate()
        m.MindtPy_linear_cuts.activate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        # m.pprint() #print psc master problem for debugging
        results = self.mip_solver.solve(m, load_solutions=False,
                                        options=self.mip_solver_kwargs)
        for c in self.nonlinear_constraints:
            c.activate()
        m.MindtPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if self.obj.sense == minimize:
                self.LB = max(value(self.obj.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(self.obj.expr), self.UB)
                self.UB_progress.append(self.UB)
            print('MILP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.mip_iter, value(self.obj.expr), self.LB,
                          self.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary configurations.')
            if self.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        elif master_terminate_cond is tc.unbounded:
            print('MILP master problem is unbounded. ')
            m.solutions.load_from(results)
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_GBD_master(self, leave_linear_active=True):
        m = self.m
        self.mip_iter += 1
        print('MILP {}: Solve master problem.'.format(self.mip_iter))
        if not leave_linear_active:
            # Deactivate all constraints except those in MindtPy_linear_cuts
            _MindtPy_linear_cuts = set(
                c for c in m.MindtPy_linear_cuts.component_data_objects(
                    ctype=Constraint, descend_into=True))
            to_deactivate = set(c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
                if c not in _MindtPy_linear_cuts)
            for c in to_deactivate:
                c.deactivate()
        else:
            for c in self.nonlinear_constraints:
                c.deactivate()
        m.MindtPy_linear_cuts.activate()
        # m.MindtPy_objective_expr.activate() # This activation will be deleted
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        # m.pprint() #print gbd master problem for debugging
        results = self.mip_solver.solve(m, load_solutions=False,
                                        options=self.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it is infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(self.mip_solver.options)
            # This solver option is specific to Gurobi.
            self.mip_solver.options['DualReductions'] = 0
            results = self.mip_solver.solve(m, load_solutions=False,
                                            options=self.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            self.mip_solver.options.update(old_options)
        if not leave_linear_active:
            for c in to_deactivate:
                c.activate()
        else:
            for c in self.nonlinear_constraints:
                c.activate()
        m.MindtPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if self.obj.sense == minimize:
                self.LB = max(value(self.obj.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(self.obj.expr), self.UB)
                self.UB_progress.append(self.UB)
            print('MILP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.mip_iter, value(self.obj.expr), self.LB,
                          self.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary configurations.')
            if self.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        elif master_terminate_cond is tc.unbounded:
            print('MILP master problem is unbounded. ')
            # Change the integer values to something new, re-solve.
            m.MindtPy_linear_cuts.activate()
            m.MindtPy_linear_cuts.feasible_integer_cuts.activate()
            self._init_max_binaries()
            m.MindtPy_linear_cuts.deactivate()
            m.MindtPy_linear_cuts.feasible_integer_cuts.deactivate()
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))
        #
        # m.MindtPy_linear_cuts.deactivate()
        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_NLP_subproblem(self):
        m = self.m
        config = self.CONFIG()
        self.nlp_iter += 1
        print('NLP {}: Solve subproblem for fixed binaries.'
              .format(self.nlp_iter))
        # Set up NLP
        for v in self.binary_vars:
            v.fix(int(value(v) + 0.5))

        # restore original variable values
        for v in m.component_data_objects(ctype=Var, descend_into=True):
            if not v.fixed and not v.is_binary():
                try:
                    v.set_value(self.initial_variable_values[id(v)])
                except KeyError:
                    continue
        #
        m.MindtPy_linear_cuts.deactivate()
        m.tmp_duals = ComponentMap()
        for c in m.component_data_objects(ctype=Constraint, active=True,
                                          descend_into=True):
            rhs = ((0 if c.upper is None else c.upper) +
                   (0 if c.lower is None else c.lower))
            sign_adjust = 1 if value(c.upper) is None else -1
            try:
                c_body = value(c.body)
            except:
                c_body = 0
            m.tmp_duals[c] = sign_adjust * max(0,
                                               sign_adjust * (rhs - c_body))
            # TODO check sign_adjust
        t = TransformationFactory('contrib.deactivate_trivial_constraints')
        t.apply_to(m, tmp=True, ignore_infeasible=True)
        # Solve the NLP
        # m.pprint() # print nlp problem for debugging
        results = self.nlp_solver.solve(m, load_solutions=False,
                                        options=config.nlp_solver_kwargs)
        # t.revert(m)
        for v in self.binary_vars:
            v.unfix()
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            for c in m.tmp_duals:
                if m.dual.get(c, None) is None:
                    m.dual[c] = m.tmp_duals[c]
            if self.obj.sense == minimize:
                self.UB = min(value(self.obj.expr), self.UB)
                self.solution_improved = self.UB < self.UB_progress[-1]
                self.UB_progress.append(self.UB)
            else:
                self.LB = max(value(self.obj.expr), self.LB)
                self.solution_improved = self.LB > self.LB_progress[-1]
                self.LB_progress.append(self.LB)
            print('NLP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.nlp_iter, value(self.obj.expr), self.LB,
                          self.UB))
            if self.solution_improved:
                self.best_solution_found = m.clone()
            # Add the linear cut
            if config.strategy == 'OA':
                self._add_oa_cut()
            elif config.strategy == 'PSC':
                self._add_psc_cut()
            elif config.strategy == 'GBD':
                self._add_gbd_cut()

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

                if self.strategy == 'PSC' or self.strategy == 'GBD':
                    m.ipopt_zL_out[var] = 0
                    m.ipopt_zU_out[var] = 0
                    if var.ub is not None and abs(var.ub - value(var)) < self.bound_tolerance:
                        m.ipopt_zL_out[var] = 1
                    elif var.lb is not None and abs(value(var) - var.lb) < self.bound_tolerance:
                        m.ipopt_zU_out[var] = -1
            # m.pprint() #print infeasible nlp problem for debugging
            if self.strategy == 'PSC':
                print('Adding PSC feasibility cut.')
                self._add_psc_cut(nlp_feasible=False)
            elif self.strategy == 'GBD':
                print('Adding GBD feasibility cut.')
                self._add_gbd_cut(nlp_feasible=False)
            elif self.strategy == 'OA':
                print('Solving feasibility problem')
                if self.initial_feas == 1:
                    self._add_feas_slacks()
                    self.initial_feas = 0
                self._solve_NLP_feas()
                self._add_oa_cut()
            # Add an integer cut to exclude this discrete option
            self._add_int_cut()
        elif subprob_terminate_cond is tc.maxIterations:
            # TODO try something else? Reinitialize with different initial
            # value?
            print('NLP subproblem failed to converge within iteration limit.')
            # Add an integer cut to exclude this discrete option
            self._add_int_cut()
        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(subprob_terminate_cond))

        # Call the NLP post-solve callback
        config.subproblem_postsolve(m, self)

    def _solve_NLP_feas(self):
        m = self.m
        m.MindtPy_objective.deactivate()
        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            constr.deactivate()
        m.MindtPy_feas.activate()
        m.del_component('MindtPy_feas_obj')
        m.MindtPy_feas_obj = Objective(
            expr=sum(s for s in m.MindtPy_feas.slack_var[...]), sense=minimize)
        for v in self.binary_vars:
            if value(v) > 0.5:
                v.fix(1)
            else:
                v.fix(0)
        # m.pprint()  #print nlp feasibility problem for debugging
        feas_soln = self.nlp_solver.solve(
            m, load_solutions=False, options=self.nlp_solver_kwargs)
        subprob_terminate_cond = feas_soln.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            m.solutions.load_from(feas_soln)
        elif subprob_terminate_cond is tc.infeasible:
            raise ValueError('Feasibility NLP infeasible. '
                             'This should never happen.')
        else:
            raise ValueError(
                'MindtPy unable to handle feasibility NLP termination condition '
                'of {}'.format(subprob_terminate_cond))

        for v in self.binary_vars:
            v.unfix()

        m.MindtPy_feas.deactivate()
        m.MindtPy_feas_obj.deactivate()
        # m.MindtPy_objective_expr.activate()
        m.MindtPy_objective.activate()

        for constr in m.component_data_objects(
                ctype=Constraint, descend_into=True):
            constr.activate()
            rhs = ((0 if constr.upper is None else constr.upper) +
                   (0 if constr.lower is None else constr.lower))
            sign_adjust = 1 if value(constr.upper) is None else -1
            m.dual[constr] = sign_adjust * max(0,
                                               sign_adjust * (rhs - value(constr.body)))

        if value(m.MindtPy_feas_obj.expr) == 0:
            raise ValueError(
                'Problem is not infeasible, check NLP solver')

    def _solve_LP_subproblem(self):
        m = self.m
        """Solve continuous relaxation of MILP (relaxed binary variables)."""
        self.nlp_iter += 1
        print("LP {}: Solve continuous relaxation.".format(self.nlp_iter))
        for v in self.binary_vars:
            v.domain = NonNegativeReals
            v.setlb(0)
            v.setub(1)
        for c in self.nonlinear_constraints:
            c.deactivate()
        m.MindtPy_linear_cuts.activate()
        results = self.mip_solver.solve(m, load_solutions=False,
                                        options=self.mip_solver_kwargs)
        for v in self.binary_vars:
            v.domain = Binary
        for c in self.nonlinear_constraints:
            c.deactivate()
        m.MindtPy_linear_cuts.activate()
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            # Add the linear cut
            self._add_ecp_cut()

            # This adds an integer cut to the feasible_integer_cuts
            # ConstraintList, which is not activated by default. However, it
            # may be activated as needed in certain situations or for certain
            # values of option flags.
            self._add_int_cut(feasible=True)
        elif subprob_terminate_cond is tc.infeasible:
            # TODO fail? try something else? this should never happen
            raise ValueError('Relaxed LP infeasible. '
                             'This should never happen.')
            self._add_int_cut()
        else:
            raise ValueError(
                'MindtPy unable to handle relaxed LP termination condition '
                'of {}'.format(subprob_terminate_cond))
        # Call the LP post-solve callback
        self.subproblem_postsolve(m, self)

    def _calc_jacobians(self):
        self.jacs = {}
        for c in self.nonlinear_constraints:
            constraint_vars = list(EXPR.identify_variables(c.body))
            jac_list = differentiate(c.body, wrt_list=constraint_vars)
            self.jacs[c] = {id(var): jac
                            for var, jac in zip(constraint_vars, jac_list)}

    def _add_oa_cut(self):
        m = self.m
        m.MindtPy_linear_cuts.nlp_iters.add(self.nlp_iter)
        sign_adjust = -1 if self.obj.sense == minimize else 1

        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        for constr in self.jacs:
            rhs = ((0 if constr.upper is None else constr.upper) +
                   (0 if constr.lower is None else constr.lower))
            c = m.MindtPy_linear_cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * m.dual[constr]) * (sum(
                    value(self.jacs[constr][id(var)]) * (var - value(var))
                    for var in list(EXPR.identify_variables(constr.body))) +
                    value(constr.body) - rhs) +
                m.MindtPy_linear_cuts.slack_vars[self.nlp_iter,
                                                 self.nl_map[constr]] <= 0)
            self.OA_constr_map[constr, self.nlp_iter] = c

    def _add_feas_slacks(self):
        m = self.m
        # generate new constraints
        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            rhs = ((0 if constr.upper is None else constr.upper) +
                   (0 if constr.lower is None else constr.lower))
            c = m.MindtPy_feas.feas_constraints.add(constr.body - rhs
                                                    <= m.MindtPy_feas.slack_var[self.feas_map[constr]])
            self.feas_constr_map[constr, self.nlp_iter] = c

    def _add_ecp_cut(self):
        m = self.m
        m.MindtPy_linear_cuts.mip_iters.add(self.mip_iter)
        sign_adjust = -1 if self.obj.sense == minimize else 1
        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        gen = (constr for constr in self.jacs
               if (0 if constr.upper is None
                   else abs(value(constr.body) - constr.upper)) +
               (0 if constr.lower is None
                else abs(constr.lower - value(constr.body)))
               > self.ECP_tolerance)
        for constr in gen:
            constr_dir = -1 if value(constr.upper) is None else 1
            rhs = ((0 if constr.upper is None else constr.upper) +
                   (0 if constr.lower is None else constr.lower))
            # this only happens if a constraint is >=
            c = m.MindtPy_linear_cuts.ecp_cuts.add(
                expr=copysign(1, constr_dir)
                * (sum(value(self.jacs[constr][id(var)]) * (var - value(var))
                       for var in list(EXPR.identify_variables(constr.body))) +
                   value(constr.body) - rhs) <= 0)
            self.ECP_constr_map[constr, self.mip_iter] = c

    def _add_psc_cut(self, nlp_feasible=True):
        m = self.m

        sign_adjust = 1 if self.obj.sense == minimize else -1

        # generate the sum of all multipliers with the nonlinear constraints
        var_to_val = {id(var): NumericConstant(value(var))
                      for var in self.nonlinear_variables}
        sum_nonlinear = (
            # Address constraints of form f(x) <= upper
            sum(value(m.dual[c]) * -1 *
                (clone_expression(c.body, substitute=var_to_val) - c.upper)
                for c in self.nonlinear_constraints
                if value(abs(m.dual[c])) > self.small_dual_tolerance
                and c.upper is not None) +
            # Address constraints of form f(x) >= lower
            sum(value(m.dual[c]) *
                (c.lower - clone_expression(c.body, substitute=var_to_val))
                for c in self.nonlinear_constraints
                if value(abs(m.dual[c])) > self.small_dual_tolerance
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
            repn = generate_canonical_repn(constr.body)
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
                if id(var) in self.nonlinear_variable_IDs)
            for c in lin_cons
            if value(abs(m.dual[c])) > self.small_dual_tolerance)

        # Generate the sum of all bound multipliers with nonlinear variables
        sum_var_bounds = (
            sum(m.ipopt_zL_out.get(var, 0) * (var - value(var))
                for var in self.nonlinear_variables
                if value(abs(m.ipopt_zL_out.get(var, 0))) >
                self.small_dual_tolerance) +
            sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
                for var in self.nonlinear_variables
                if value(abs(m.ipopt_zU_out.get(var, 0))) >
                self.small_dual_tolerance))

        if nlp_feasible:
            # Optimality cut (for feasible NLP)
            m.MindtPy_linear_cuts.psc_cuts.add(
                expr=self.obj.expr * sign_adjust >= sign_adjust * (
                    self.obj.expr + sum_nonlinear + sum_linear +
                    sum_var_bounds))
        else:
            # Feasibility cut (for infeasible NLP)
            m.MindtPy_linear_cuts.psc_cuts.add(
                expr=(sum_nonlinear + sum_linear + sum_var_bounds) <= 0)

    def _add_gbd_cut(self, nlp_feasible=True):
        m = self.m

        sign_adjust = 1 if self.obj.sense == minimize else -1

        for c in m.component_data_objects(ctype=Constraint, active=True,
                                          descend_into=True):
            if value(c.upper) is None and value(c.lower) is None:
                raise ValueError(
                    'Oh no, Pyomo did something MindtPy does not expect. '
                    'The value of c.upper for {} is None: {} <= {} <= {}'
                    .format(c.name, c.lower, c.body, c.upper))
        # TODO handle the case where constraint upper and lower is None

        # only substitute non-binary variables to their values
        binary_var_ids = set(id(var) for var in self.binary_vars)
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
                if value(abs(m.dual[c])) > self.small_dual_tolerance
                and c.upper is not None) +
            sum(value(m.dual[c]) *
                (c.lower - clone_expression(c.body, substitute=var_to_val))
                for c in m.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True)
                if value(abs(m.dual[c])) > self.small_dual_tolerance
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
                    self.small_dual_tolerance)) +
            sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
                for var in m.component_data_objects(ctype=Var,
                                                    descend_into=True)
                if (id(var) not in binary_var_ids and
                    value(abs(m.ipopt_zU_out.get(var, 0))) >
                    self.small_dual_tolerance)))

        if nlp_feasible:
            m.MindtPy_linear_cuts.gbd_cuts.add(
                expr=self.obj.expr * sign_adjust >= sign_adjust * (
                    value(self.obj.expr) + sum_constraints + sum_var_bounds))
        else:
            if sum_constraints + sum_var_bounds != 0:
                m.MindtPy_linear_cuts.gbd_cuts.add(
                    expr=(sum_constraints + sum_var_bounds) <= 0)

    def _add_int_cut(self, feasible=False):
        config = self.CONFIG()
        if config.integer_cuts:
            m = self.m
            int_tol = self.integer_tolerance
            # check to make sure that binary variables are all 0 or 1
            for v in self.binary_vars:
                if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                    raise ValueError('Binary {} = {} is not 0 or 1'.format(
                        v.name, value(v)))

            if not self.binary_vars:  # if no binary variables, skip.
                return

            int_cut = (sum(1 - v for v in self.binary_vars
                           if value(abs(v - 1)) <= int_tol) +
                       sum(v for v in self.binary_vars
                           if value(abs(v)) <= int_tol) >= 1)

            if not feasible:
                # Add the integer cut
                m.MindtPy_linear_cuts.integer_cuts.add(expr=int_cut)
            else:
                m.MindtPy_linear_cuts.feasible_integer_cuts.add(expr=int_cut)

    def _detect_nonlinear_vars(self):
        """Identify the variables that participate in nonlinear terms."""
        self.nonlinear_variables = []
        # This is a workaround because Var is not hashable, and I do not want
        # duplicates in self.nonlinear_variables.
        seen = set()
        for constr in self.nonlinear_constraints:
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
                                self.nonlinear_variables.append(var)
            # if the root expression object is not a summation, then something
            # else is the cause of the nonlinearity. Collect all participating
            # variables.
            else:
                # collect variables
                for var in EXPR.identify_variables(constr.body,
                                                   include_fixed=False):
                    if id(var) not in seen:
                        seen.add(id(var))
                        self.nonlinear_variables.append(var)
        self.nonlinear_variable_IDs = set(
            id(v) for v in self.nonlinear_variables)


