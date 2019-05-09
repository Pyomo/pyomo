from math import inf
from pyomo.core import (ConcreteModel, Var, Constraint, ConstraintList,
                           Objective, Integers, Reals, RangeSet, Block,
                           TransformationFactory, 
                           sqrt, value)
from pyomo.opt import SolverFactory
from pyomo.core.base.symbolic import differentiate
from pyomo.core.expr.current import identify_variables
from pyomo.common.config import (ConfigBlock, ConfigValue, In,
                                 PositiveFloat, PositiveInt)
from pyomo.contrib.gdpopt.util import \
    (create_utility_block, process_objective, setup_results_object, a_logger,
     copy_var_list_values, SuppressInfeasibleWarning, _DoNothing)
from pyomo.contrib.mindtpy.util import MindtPySolveData, calc_jacobians
from pyomo.contrib.preprocessing.plugins.int_to_binary import IntegerToBinary
from pyomo.opt import SolverResults
from pyomo.opt import TerminationCondition as TC
from IPython.core.debugger import Pdb


class InfeasibleException(Exception):
    """Thrown when feasibility pump doesn't converge"""
    pass


class IterationLimitExceeded(Exception):
    """Thrown when an iteration limit gets exceeded"""

    def __init__(self, method, n_iter):
        super().__init__(f'Iteration limit exceeded in method {method} with N={n_iter}')


def add_L1_objective_function(model, setpoint_model):
    """Adds minimum absolute distance objective function"""
    model.L1_obj_var = Var(domain=Reals, bounds=(0, None))
    model.L1_obj_fun = Objective(expr=model.L1_obj_var)
    model.L1_obj_ub_idx = RangeSet(
        len(model.binary_var_list))
    model.L1_obj_ub_constr = Constraint(
        model.L1_obj_ub_idx, rule=lambda i: model.L1_obj_var >= 0)
    model.L1_obj_lb_idx = RangeSet(
        len(model.binary_var_list))
    model.L1_obj_lb_constr = Constraint(
        model.L1_obj_lb_idx, rule=lambda i: model.L1_obj_var >= 0)

    for (c_lb, c_ub, v_model, v_setpoint) in zip(model.L1_obj_lb_idx,
                                                 model.L1_obj_ub_idx,
                                                 model.binary_var_list,
                                                 setpoint_model.binary_var_list):
        model.L1_obj_lb_constr[c_lb].set_value(
            expr=v_model - v_setpoint.value >= -model.L1_obj_var)
        model.L1_obj_ub_constr[c_ub].set_value(
            expr=v_model - v_setpoint.value <= model.L1_obj_var)


def add_L2_objective_function(model, setpoint_model):
    """Adds minimum euclidean distance objective function"""
    model.L2_obj_fun = Objective(expr=(
        sum([(nlp_var - milp_var.value)**2
             for (nlp_var, milp_var) in
             zip(model.binary_var_list, setpoint_model.binary_var_list)])))


def solution_distance(nlp_vars, milp_vars):
    """Calculates the euclidean norm between two var lists"""
    return sqrt(sum([(nlp_var.value - milp_var.value)**2
                     for (nlp_var, milp_var) in
                     zip(nlp_vars, milp_vars)
                     if not milp_var.is_continuous()]))


def setup_milp_model(solve_data):
    """Sets up MILP from MINLP by deactivating nonliear constraints"""
    solve_data.milp_model = solve_data.working_model.clone()
    milp_model = solve_data.milp_model
    milp_model.MindtPy_utils.linear_cuts.activate()

    for obj in milp_model.component_data_objects(Objective, active=True):
        obj.deactivate()

    if hasattr(solve_data.milp_model.MindtPy_utils, 'objective_constr'):
        solve_data.milp_model.MindtPy_utils.objective_constr.deactivate()

    if solve_data.incumbent_obj_val < inf:
        milp_model.MindtPy_utils.linear_cuts.increasing_objective_cuts.add(
            expr=milp_model.MindtPy_utils.objective_value
                 <= solve_data.incumbent_obj_val - abs(0.1*solve_data.incumbent_obj_val) )

    for constr in filter(
            lambda c: c.body.polynomial_degree() not in (0,1),
            milp_model.component_data_objects(ctype=Constraint, active=True)):
        constr.deactivate()

    return solve_data.milp_model


def setup_nlp_model(solve_data):
    """Sets up NLP from MINLP working model by fixing discrete variables"""
    # --- Setup NLP model ---
    solve_data.nlp_model = solve_data.working_model.clone()
    nlp_model = solve_data.nlp_model

    for obj in nlp_model.component_data_objects(Objective, active=True):
        obj.deactivate()

    if hasattr(solve_data.nlp_model.MindtPy_utils, 'objective_constr'):
        solve_data.nlp_model.MindtPy_utils.objective_constr.deactivate()

    TransformationFactory('core.relax_integrality'). \
        apply_to(solve_data.nlp_model)
    return solve_data.nlp_model


def execute_feasibility_pump(solve_data, config):
    """Solves NLP and MILP subproblems until a feasible solution is  found

    If `write_back==True`, the resulting feasible solution is copied to
    the working model
    """
    opt_milp = SolverFactory(config.mip_solver)
    opt_nlp = SolverFactory(config.nlp_solver)
    n_iter = 0
    solve_data.incumbent_obj_val = inf  # assumes minimization

    milp_model = setup_milp_model(solve_data)
    add_L1_objective_function(milp_model, solve_data.relaxed_nlp_model)
    # Deactivate extraneous IMPORT/EXPORT suffixes
    getattr(milp_model, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(milp_model, 'ipopt_zU_out', _DoNothing()).deactivate()

    milp_result = opt_milp.solve(milp_model, **config.mip_solver_args)

    while milp_result.solver.termination_condition is not TC.infeasible:
        if n_iter >= config.iteration_limit:
            raise IterationLimitExceeded('feasibility_pump', n_iter)

        # Solve NLP auxiliary problem
        nlp_model = setup_nlp_model(solve_data)
        add_L2_objective_function(nlp_model, milp_model)
        nlp_result = opt_nlp.solve(nlp_model)
        copy_var_list_values(
            nlp_model.var_list,
            solve_data.working_model.var_list,
            config,
            ignore_integrality=True)

        if solution_distance(nlp_model.var_list, milp_model.var_list) < config.zero_tolerance:
            fixed_nlp = solve_fixed_nlp(solve_data, config)
            objective_value = next(fixed_nlp.component_data_objects(
                Objective, active=True)).expr()

            copy_var_list_values(
                fixed_nlp.var_list,
                solve_data.working_model.var_list,
                config)
            return  # TODO-romeo currently just returns the first solution

            if objective_value < solve_data.incumbent_obj_val:
                solve_data.incumbent_obj_val = objective_value
                solve_data.best_solution_found = solve_data.working_model.clone()
                print(f'New objective val: {solve_data.incumbent_obj_val}')

            create_no_good_cut(solve_data.working_model, fixed_nlp)

        create_oa_cut(solve_data, config)
        n_iter += 1

        # Solve MILP auxiliary problem
        milp_model = setup_milp_model(solve_data)
        add_L1_objective_function(milp_model, nlp_model)
        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(milp_model, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(milp_model, 'ipopt_zU_out', _DoNothing()).deactivate()
        milp_result = opt_milp.solve(milp_model, **config.mip_solver_args)
    else:
        raise InfeasibleException()


def initialize_FP(solve_data, config):
    """Relaxes integrality and generates first constraint."""
    solve_data.relaxed_nlp_model = solve_data.working_model.clone()
    TransformationFactory('core.relax_integrality'). \
        apply_to(solve_data.relaxed_nlp_model)

    with SuppressInfeasibleWarning():
        opt_nlp = SolverFactory(config.nlp_solver)
        res = opt_nlp.solve(
            solve_data.relaxed_nlp_model, **config.nlp_solver_args)

    if res.solver.termination_condition is TC.infeasible:
        print('Infeasible problem')
        raise InfeasibleException()


def solve_fixed_nlp(solve_data, config):
    """Fixes discretes and solves NLP

    An auxiliary problem is cloned from the working model and stored in
    solve_data. The discrete variables get fixed and the resulting NLP is
    solved.

    Currently doesn't calculate any duals
    """
    solve_data.fixed_integer_model = solve_data.working_model.clone()
    f_nlp_model = solve_data.fixed_integer_model
    TransformationFactory('core.fix_discrete').apply_to(
        f_nlp_model)

    with SuppressInfeasibleWarning():
        opt_nlp = SolverFactory(config.nlp_solver)
        opt_nlp.solve(f_nlp_model, **config.nlp_solver_args)
    return solve_data.fixed_integer_model


def create_oa_cut(solve_data, config):
    # TODO-romeo replace this with mindtpy methods
    for constr in solve_data.working_model.component_data_objects(
            ctype=Constraint, active=True):
        # Check if constraint is active
        if abs(constr.slack()) > config.bound_tolerance*100:
            continue
        if constr.body.polynomial_degree() in (0, 1):
            continue
        if constr.has_ub() and constr.has_lb() and constr.upper == constr.lower:
            continue

        constr_vars = list(identify_variables(constr.body))
        partial_derivatives = differentiate(constr.body, wrt_list=constr_vars)

        if constr.upper is not None:
            solve_data.working_model.MindtPy_utils.linear_cuts.oa_cuts.add(
                expr=(sum(value(pd)*(var - var.value) for (var, pd)
                          in zip(constr_vars, partial_derivatives))
                      <= constr.upper))

        if constr.lower is not None:
            solve_data.working_model.MindtPy_utils.linear_cuts.oa_cuts.add(
                expr=(sum(value(pd)*(var - var.value) for (var, pd)
                          in zip(constr_vars, partial_derivatives))
                      >= constr.lower))


def create_linearized_objective_cuts(solve_data, config):
    """Cuts off worse and equal solutions

    Adds cut to working model such that all new solutions must have a better
    objective function than the current solution (minus a small epsilon).
    This way we will always arrive at a new, better solution until the problem
    becomes infeasible.

    Cut: (T_1 f)(x_{new}) <= f(x*) [T_1 denotes first order Taylor-expansion]
       => f'(x*) * (x_{new} - x*) <= 0 (minimization)
    """
    obj_fun = next(solve_data.working_model.component_data_objects(
        ctype=Objective, active=True, descend_into=True), None)
    obj_vars = list(identify_variables(obj_fun))
    partial_derivatives = differentiate(obj_fun, wrt_list=obj_vars)
    solve_data.working_model.MindtPy_utils. \
        linear_cuts.increasing_objective_cuts.add(
            expr=(sum(value(pd)*(var - var.value) for (var, pd)
                      in zip(obj_vars, partial_derivatives))
                  + 0.1 * max(abs(solve_data.incumbent_obj_val), 1)  # TODO-romeo replace this with variable
                  <= 0))
                  #  TODO-romeo flag for objective improvement


def create_no_good_cut(target_model, value_model):
    """Cut out current binary combination"""
    target_model.MindtPy_utils. \
        linear_cuts.no_good_combination_cuts.add(
            expr=(sum(target_var if value_var.value == 0 else (1-target_var)
                      for (target_var, value_var)
                      in zip(target_model.var_list, value_model.var_list)
                      if target_var.is_binary() and value_var.is_binary())
                  >= 1))


def setup_simple_model():
    """Returns a very simple MINLP"""
    simple_model = ConcreteModel()
    simple_model.y = Var(domain=Integers, bounds=(1, 4))
    simple_model.x = Var(domain=Reals, bounds=(1, 5))
    simple_model.lin_constraint = Constraint(expr=simple_model.y >= 1.6)

    def g_constr(model):
        return model.x <= sqrt(1.5*model.y)

    simple_model.nonlin_constraint = Constraint(rule=g_constr)
    simple_model.obj = Objective(expr=-1*simple_model.x + -1*simple_model.y)
    return simple_model


def solve_feasibility_pump(model, copy_back=True):
    """Runs the feasibility pump with objective cuts

    copy_back = True writes the feasible solution to the input model
    """

    solve_data = MindtPySolveData()
    solve_data.results = SolverResults()
    solve_data.original_model = model
    solve_data.working_model = model.clone()
    working_model = solve_data.working_model
    config = setup_config()
    if hasattr(solve_data.working_model, 'MindtPy_utils'):
        del solve_data.working_model.MindtPy_utils

    with create_utility_block(working_model, 'MindtPy_utils', solve_data):
        mp_utils = working_model.MindtPy_utils
        mp_utils.linear_cuts = Block()
        mp_utils.linear_cuts.deactivate()
        mp_utils.linear_cuts.oa_cuts = ConstraintList(
            doc="Outer Approximation for MILP model")
        mp_utils.linear_cuts.increasing_objective_cuts = ConstraintList(
            doc="Objective values need to increase")
        mp_utils.linear_cuts.no_good_combination_cuts = ConstraintList(
            doc="Discrete combination needs to change")
        setup_results_object(solve_data, config)

        working_model.var_list = mp_utils.variable_list
        working_model.binary_var_list = [var for var in mp_utils.variable_list
                                         if var.is_binary()]

        process_objective(solve_data, config, always_move_objective=True)
        # calc_jacobians(solve_data, config)  # TODO-romeo use this again

        try:
            initialize_FP(solve_data, config)
        except InfeasibleException:
            return

        try:
            execute_feasibility_pump(solve_data, config)
        except IterationLimitExceeded as exept:
            print(exept)
            raise
        except InfeasibleException:
            print("Feasibility Pump stopped converging-> optimal solution")
            # solve_data.best_solution_found.pprint()
            # print(solve_data.incumbent_obj_val)
            # print([(v.name, v.value) for v in solve_data.best_solution_found.component_data_objects(ctype=Var)])

    if copy_back:
        copy_var_list_values(solve_data.working_model.component_data_objects(Var),
                             model.component_data_objects(Var),
                             config)
    return solve_data


def setup_config():
    """This is later done in the MindtPy main file"""
    config = ConfigBlock("FeasabilityPump")
    config.declare("integer_tolerance", ConfigValue(
        default=1E-4,
        description="Tolerance on integral values."
    ))
    config.declare("zero_tolerance", ConfigValue(
        default=1E-15,
        description="Tolerance on variable equal to zero."
    ))
    config.declare("bound_tolerance", ConfigValue(
        default=1E-5,
        domain=PositiveFloat,
        description="Bound tolerance",
        doc="Relative tolerance for bound feasibility checks"
    ))
    config.declare("constraint_tolerance", ConfigValue(
        default=1E-6,
        description="Tolerance on constraint satisfaction."
    ))
    config.declare("nlp_solver", ConfigValue(
        default="gams",
        domain=In(["baron", "gams", "ipopt", "ipopth", "conopt"]),
        description="NLP subsolver name",
        doc="Which NLP subsolver is going to be used for solving the nonlinear"
            "subproblems"
    ))
    config.declare("nlp_solver_args", ConfigBlock(
        implicit=True,
        description="NLP subsolver options",
        doc="Which NLP subsolver options to be passed to the solver while "
            "solving the nonlinear subproblems"
    ))
    config.nlp_solver_args.add('solver', 'ipopth')
    config.nlp_solver_args.add('add_options', ['option optcr=0;'])

    config.declare("mip_solver", ConfigValue(
        default="gurobi",
        domain=In(["gurobi", "cplex", "cbc", "glpk", "gams"]),
        description="MIP subsolver name",
        doc="Which MIP subsolver is going to be used for solving the mixed-"
            "integer master problems"
    ))
    config.declare("mip_solver_args", ConfigBlock(
        implicit=True,
        description="MIP subsolver options",
        doc="Which MIP subsolver options to be passed to the solver while "
            "solving the mixed-integer master problems"
    ))
    # config.mip_solver_args.add('solver', 'gurobi')
    # config.mip_solver_args.add('add_options',
    #     f'option optrc={config.bound_tolerance*1e-2};')

    config.declare("logger", ConfigValue(
        default='pyomo.contrib.mindtpy',
        description='The logger object name to use for reporting.',
        domain=a_logger))
    config.declare("iteration_limit", ConfigValue(
        default=500,
        domain=PositiveInt,
        description="Iteration limit",
        doc="Number of maximum iterations in the decomposition methods"
    ))
    return config


if __name__ == "__main__":
    solve_feasibility_pump(setup_simple_model())
