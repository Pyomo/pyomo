# -*- coding: utf-8 -*-
from pyomo.common.config import (
    ConfigBlock, ConfigValue, In, PositiveFloat, PositiveInt, NonNegativeInt)
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger


def _get_MindtPy_config():
    CONFIG = ConfigBlock('MindtPy')

    CONFIG.declare('iteration_limit', ConfigValue(
        default=50,
        domain=NonNegativeInt,
        description='Iteration limit',
        doc='Number of maximum iterations in the decomposition methods.'
    ))
    CONFIG.declare('stalling_limit', ConfigValue(
        default=15,
        domain=PositiveInt,
        description='Stalling limit',
        doc='Stalling limit for progress in the decomposition methods.'
    ))
    CONFIG.declare('time_limit', ConfigValue(
        default=600,
        domain=PositiveInt,
        description='Time limit (seconds, default=600)',
        doc='Seconds allowed until terminated. Note that the time limit can'
            'currently only be enforced between subsolver invocations. You may'
            'need to set subsolver time limits as well.'
    ))
    CONFIG.declare('strategy', ConfigValue(
        default='OA',
        domain=In(['OA', 'ECP', 'GOA', 'FP']),
        description='Decomposition strategy',
        doc='MINLP Decomposition strategy to be applied to the method. '
            'Currently available Outer Approximation (OA), Extended Cutting '
            'Plane (ECP), Global Outer Approximation (GOA) and Feasibility Pump (FP).'
    ))
    CONFIG.declare('add_regularization', ConfigValue(
        default=None,
        domain=In(['level_L1', 'level_L2', 'level_L_infinity',
                   'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag']),
        description='add regularization',
        doc='solving a regularization problem before solve the fixed subproblem'
            'the objective function of the regularization problem.'
    ))
    CONFIG.declare('init_strategy', ConfigValue(
        default=None,
        domain=In(['rNLP', 'initial_binary', 'max_binary', 'FP']),
        description='Initialization strategy',
        doc='Initialization strategy used by any method. Currently the '
            'continuous relaxation of the MINLP (rNLP), solve a maximal '
            'covering problem (max_binary), and fix the initial value for '
            'the integer variables (initial_binary).'
    ))
    CONFIG.declare('max_slack', ConfigValue(
        default=1000.0,
        domain=PositiveFloat,
        description='Maximum slack variable',
        doc='Maximum slack variable value allowed for the Outer Approximation '
            'cuts.'
    ))
    CONFIG.declare('OA_penalty_factor', ConfigValue(
        default=1000.0,
        domain=PositiveFloat,
        description='Outer Approximation slack penalty factor',
        doc='In the objective function of the Outer Approximation method, the '
            'slack variables corresponding to all the constraints get '
            'multiplied by this number and added to the objective.'
    ))
    CONFIG.declare('call_after_main_solve', ConfigValue(
        default=_DoNothing(),
        domain=None,
        description='Function to be executed after every main problem',
        doc='Callback hook after a solution of the main problem.'
    ))
    CONFIG.declare('call_after_subproblem_solve', ConfigValue(
        default=_DoNothing(),
        domain=None,
        description='Function to be executed after every subproblem',
        doc='Callback hook after a solution of the nonlinear subproblem.'
    ))
    CONFIG.declare('call_after_subproblem_feasible', ConfigValue(
        default=_DoNothing(),
        domain=None,
        description='Function to be executed after every feasible subproblem',
        doc='Callback hook after a feasible solution'
            ' of the nonlinear subproblem.'
    ))
    CONFIG.declare('tee', ConfigValue(
        default=False,
        description='Stream output to terminal.',
        domain=bool
    ))
    CONFIG.declare('logger', ConfigValue(
        default='pyomo.contrib.mindtpy',
        description='The logger object or name to use for reporting.',
        domain=a_logger
    ))
    CONFIG.declare('integer_to_binary', ConfigValue(
        default=False,
        description='Convert integer variables to binaries (for no-good cuts).',
        domain=bool
    ))
    CONFIG.declare('add_no_good_cuts', ConfigValue(
        default=False,
        description='Add no-good cuts (no-good cuts) to binary variables to disallow same integer solution again.'
                    'Note that integer_to_binary flag needs to be used to apply it to actual integers and not just binaries.',
        domain=bool
    ))
    CONFIG.declare('use_tabu_list', ConfigValue(
        default=False,
        description='Use tabu list and incumbent callback to disallow same integer solution again.',
        domain=bool
    ))
    CONFIG.declare('add_affine_cuts', ConfigValue(
        default=False,
        description='Add affine cuts drive from MC++',
        domain=bool
    ))
    CONFIG.declare('single_tree', ConfigValue(
        default=False,
        description='Use single tree implementation in solving the MILP main problem.',
        domain=bool
    ))
    CONFIG.declare('solution_pool', ConfigValue(
        default=False,
        description='Use solution pool in solving the MILP main problem.',
        domain=bool
    ))
    CONFIG.declare('add_slack', ConfigValue(
        default=False,
        description='whether add slack variable here.'
                    'slack variables here are used to deal with nonconvex MINLP.',
        domain=bool
    ))
    CONFIG.declare('cycling_check', ConfigValue(
        default=True,
        description='check if OA algorithm is stalled in a cycle and terminate.',
        domain=bool
    ))
    CONFIG.declare('feasibility_norm', ConfigValue(
        default='L_infinity',
        domain=In(['L1', 'L2', 'L_infinity']),
        description='different forms of objective function in feasibility subproblem.'
    ))
    CONFIG.declare('differentiate_mode', ConfigValue(
        default='reverse_symbolic',
        domain=In(['reverse_symbolic', 'sympy']),
        description='differentiate mode to calculate jacobian.'
    ))
    CONFIG.declare('linearize_inactive', ConfigValue(
        default=False,
        description='Add OA cuts for inactive constraints.',
        domain=bool
    ))
    CONFIG.declare('use_mcpp', ConfigValue(
        default=False,
        description="use package MC++ to set a bound for variable 'objective_value', which is introduced when the original problem's objective function is nonlinear.",
        domain=bool
    ))
    CONFIG.declare('equality_relaxation', ConfigValue(
        default=False,
        description='use dual solution from the nlp solver to add OA cuts for equality constraints.',
        domain=bool
    ))
    CONFIG.declare('calculate_dual', ConfigValue(
        default=False,
        description='calculate duals of the NLP subproblem',
        domain=bool
    ))
    CONFIG.declare('use_fbbt', ConfigValue(
        default=False,
        description='use fbbt to tighten the feasible region of the problem',
        domain=bool
    ))
    CONFIG.declare('use_dual_bound', ConfigValue(
        default=True,
        description='add dual bound constraint to enforce the objective satisfies best-found dual bound',
        domain=bool
    ))
    CONFIG.declare('heuristic_nonconvex', ConfigValue(
        default=False,
        description='use dual solution from the NLP solver and slack variables to add OA cuts for equality constraints (Equality relaxation)'
                    'and minimize the sum of the slack variables (Augmented Penalty)',
        domain=bool
    ))

    _add_subsolver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    _add_fp_configs(CONFIG)
    _add_bound_configs(CONFIG)
    _add_loa_configs(CONFIG)
    return CONFIG


def _add_subsolver_configs(CONFIG):
    CONFIG.declare('nlp_solver', ConfigValue(
        default='ipopt',
        domain=In(['ipopt', 'gams', 'baron']),
        description='NLP subsolver name',
        doc='Which NLP subsolver is going to be used for solving the nonlinear'
            'subproblems.'
    ))
    CONFIG.declare('nlp_solver_args', ConfigBlock(
        implicit=True,
        description='NLP subsolver options',
        doc='Which NLP subsolver options to be passed to the solver while '
            'solving the nonlinear subproblems.'
    ))
    CONFIG.declare('mip_solver', ConfigValue(
        default='glpk',
        domain=In(['gurobi', 'cplex', 'cbc', 'glpk', 'gams',
                   'gurobi_persistent', 'cplex_persistent']),
        description='MIP subsolver name',
        doc='Which MIP subsolver is going to be used for solving the mixed-'
            'integer main problems.'
    ))
    CONFIG.declare('mip_solver_args', ConfigBlock(
        implicit=True,
        description='MIP subsolver options',
        doc='Which MIP subsolver options to be passed to the solver while '
            'solving the mixed-integer main problems.'
    ))
    CONFIG.declare('mip_solver_mipgap', ConfigValue(
        default=1E-4,
        domain=PositiveFloat,
        description='mipgap passed to mip solver'
    ))
    CONFIG.declare('threads', ConfigValue(
        default=0,
        domain=NonNegativeInt,
        description='Threads',
        doc='Threads used by milp solver and nlp solver.'
    ))
    CONFIG.declare('regularization_mip_threads', ConfigValue(
        default=0,
        domain=NonNegativeInt,
        description='regularization mip threads',
        doc='Threads used by milp solver to solve regularization main problem.'
    ))
    CONFIG.declare('solver_tee', ConfigValue(
        default=False,
        description='Stream the output of mip solver and nlp solver to terminal.',
        domain=bool
    ))
    CONFIG.declare('mip_solver_tee', ConfigValue(
        default=False,
        description='Stream the output of mip solver to terminal.',
        domain=bool
    ))
    CONFIG.declare('nlp_solver_tee', ConfigValue(
        default=False,
        description='Stream the output of nlp solver to terminal.',
        domain=bool
    ))
    CONFIG.declare('mip_regularization_solver', ConfigValue(
        default=None,
        domain=In(['gurobi', 'cplex', 'cbc', 'glpk', 'gams',
                   'gurobi_persistent', 'cplex_persistent']),
        description='MIP subsolver for regularization problem',
        doc='Which MIP subsolver is going to be used for solving the regularization problem'
    ))


def _add_tolerance_configs(CONFIG):
    CONFIG.declare('bound_tolerance', ConfigValue(
        default=1E-4,
        domain=PositiveFloat,
        description='Bound tolerance',
        doc='Absolute tolerance for bound feasibility checks.'
    ))
    CONFIG.declare('relative_bound_tolerance', ConfigValue(
        default=1E-3,
        domain=PositiveFloat,
        description='Relative bound tolerance',
        doc='Relative tolerance for bound feasibility checks.'
            '(UB - LB) / (1e-10+|bestinteger|) <= relative tolerance.'
    ))
    CONFIG.declare('small_dual_tolerance', ConfigValue(
        default=1E-8,
        description='When generating cuts, small duals multiplied '
                    'by expressions can cause problems. Exclude all duals '
                    'smaller in absolute value than the following.'
    ))
    CONFIG.declare('integer_tolerance', ConfigValue(
        default=1E-5,
        description='Tolerance on integral values.'
    ))
    CONFIG.declare('constraint_tolerance', ConfigValue(
        default=1E-6,
        description='Tolerance on constraint satisfaction.'
    ))
    CONFIG.declare('variable_tolerance', ConfigValue(
        default=1E-8,
        description='Tolerance on variable bounds.'
    ))
    CONFIG.declare('zero_tolerance', ConfigValue(
        default=1E-8,
        description='Tolerance on variable equal to zero.'
    ))
    CONFIG.declare('ecp_tolerance', ConfigValue(
        default=None,
        domain=PositiveFloat,
        description='ECP tolerance',
        doc='Feasibility tolerance used to determine the stopping criterion in'
            'the ECP method. As long as nonlinear constraint are violated for '
            'more than this tolerance, the method will keep iterating.'
    ))


def _add_bound_configs(CONFIG):
    CONFIG.declare('obj_bound', ConfigValue(
        default=1E15,
        domain=PositiveFloat,
        description='Bound applied to the linearization of the objective function if main MILP is unbounded.'
    ))
    CONFIG.declare('continuous_var_bound', ConfigValue(
        default=1e10,
        description='default bound added to unbounded continuous variables in nonlinear constraint if single tree is activated.',
        domain=PositiveFloat
    ))
    CONFIG.declare('integer_var_bound', ConfigValue(
        default=1e9,
        description='default bound added to unbounded integral variables in nonlinear constraint if single tree is activated.',
        domain=PositiveFloat
    ))


def _add_fp_configs(CONFIG):
    CONFIG.declare('fp_cutoffdecr', ConfigValue(
        default=1E-1,
        domain=PositiveFloat,
        description='Additional relative decrement of cutoff value for the original objective function'
    ))
    CONFIG.declare('fp_iteration_limit', ConfigValue(
        default=20,
        domain=PositiveInt,
        description='Feasibility pump iteration limit',
        doc='Number of maximum iterations in the feasibility pump methods.'
    ))
    # TODO: integrate this option
    CONFIG.declare('fp_projcuts', ConfigValue(
        default=True,
        description='Whether to add cut derived from regularization of MIP solution onto NLP feasible set',
        domain=bool
    ))
    CONFIG.declare('fp_transfercuts', ConfigValue(
        default=True,
        description='Whether to transfer cuts from the Feasibility Pump MIP to main MIP in selected strategy (all except from the round in which the FP MIP became infeasible)',
        domain=bool
    ))
    CONFIG.declare('fp_projzerotol', ConfigValue(
        default=1E-4,
        domain=PositiveFloat,
        description='Tolerance on when to consider optimal value of regularization problem as zero, which may trigger the solution of a Sub-NLP'
    ))
    CONFIG.declare('fp_mipgap', ConfigValue(
        default=1E-2,
        domain=PositiveFloat,
        description='Optimality tolerance (relative gap) to use for solving MIP regularization problem'
    ))
    CONFIG.declare('fp_discrete_only', ConfigValue(
        default=True,
        description='Only calculate the distance among discrete variables in regularization problems.',
        domain=bool
    ))
    CONFIG.declare('fp_main_norm', ConfigValue(
        default='L1',
        domain=In(['L1', 'L2', 'L_infinity']),
        description='different forms of objective function MIP regularization problem.'
    ))
    CONFIG.declare('fp_norm_constraint', ConfigValue(
        default=True,
        description='Whether to add the norm constraint to FP-NLP',
        domain=bool
    ))
    CONFIG.declare('fp_norm_constraint_coef', ConfigValue(
        default=1,
        domain=PositiveFloat,
        description='The coefficient in the norm constraint, correspond to the Beta in the paper.'
    ))


def _add_loa_configs(CONFIG):
    CONFIG.declare('level_coef', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='the coefficient in the regularization main problem'
        'represents how much the linear approximation of the MINLP problem is trusted.'
    ))
    CONFIG.declare('solution_limit', ConfigValue(
        default=10,
        domain=PositiveInt,
        description='The solution limit for the regularization problem since it does not need to be solved to optimality'
    ))
    CONFIG.declare('add_cuts_at_incumbent', ConfigValue(
        default=False,
        description='Whether to add lazy cuts to the main problem at the incumbent solution found in the branch & bound tree',
        domain=bool
    ))
    CONFIG.declare('reduce_level_coef', ConfigValue(
        default=False,
        description='Whether to reduce level coefficient in ROA single tree when regularization problem is infeasible',
        domain=bool
    ))
    CONFIG.declare('use_bb_tree_incumbent', ConfigValue(
        default=False,
        description='Whether to use the incumbent solution of branch & bound tree in ROA single tree when regularization problem is infeasible',
        domain=bool
    ))
    CONFIG.declare('sqp_lag_scaling_coef', ConfigValue(
        default='fixed',
        domain=In(['fixed', 'variable_dependent']),
        description='the coefficient used to scale the L2 norm in sqp_lag'
    ))


def check_config(config):
    # configuration confirmation
    if config.add_regularization in {'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag'}:
        config.calculate_dual = True
    if config.add_regularization is not None:
        if config.regularization_mip_threads == 0 and config.threads > 0:
            config.regularization_mip_threads = config.threads
            config.logger.info(
                'Set regularization_mip_threads equal to threads')
        if config.single_tree:
            config.add_cuts_at_incumbent = True
            # if no method is activated by users, we will use use_bb_tree_incumbent by default
            if not (config.reduce_level_coef or config.use_bb_tree_incumbent):
                config.use_bb_tree_incumbent = True
        if config.mip_regularization_solver is None:
            config.mip_regularization_solver = config.mip_solver
    if config.single_tree:
        config.iteration_limit = 1
        config.add_slack = False
        config.mip_solver = 'cplex_persistent'
        config.logger.info(
            'Single tree implementation is activated. The defalt MIP solver is cplex_persistent')
        if config.threads > 1:
            config.threads = 1
            config.logger.info(
                'The threads parameter is corrected to 1 since lazy constraint callback conflicts with multi-threads mode.')
    # if the slacks fix to zero, just don't add them
    if config.max_slack == 0.0:
        config.add_slack = False

    if config.strategy == 'GOA':
        config.add_slack = False
        config.use_mcpp = True
        config.equality_relaxation = False
        config.use_fbbt = True
        # add_no_good_cuts is Ture by default in GOA
        if not config.add_no_good_cuts and not config.use_tabu_list:
            config.add_no_good_cuts = True
            config.use_tabu_list = False
    elif config.strategy == 'FP':  # feasibility pump alone
        config.init_strategy = 'FP'
        config.iteration_limit = 0
    if config.init_strategy == 'FP':
        config.add_no_good_cuts = True
        config.use_tabu_list = False

    if config.nlp_solver == 'baron':
        config.equality_relaxation = False
    if config.nlp_solver == 'gams' and config.nlp_solver.__contains__('solver'):
        if config.nlp_solver_args['solver'] == 'baron':
            config.equality_relaxation = False
    # if ecp tolerance is not provided use bound tolerance
    if config.ecp_tolerance is None:
        config.ecp_tolerance = config.bound_tolerance

    if config.solver_tee:
        config.mip_solver_tee = True
        config.nlp_solver_tee = True
    if config.heuristic_nonconvex:
        config.equality_relaxation = True
        config.add_slack = True
    if config.equality_relaxation:
        config.calculate_dual = True
    if config.add_no_good_cuts:
        config.integer_to_binary = True
    if config.use_tabu_list:
        config.mip_solver = 'cplex_persistent'
        if config.threads > 1:
            config.threads = 1
            config.logger.info(
                'The threads parameter is corrected to 1 since incumbent callback conflicts with multi-threads mode.')
