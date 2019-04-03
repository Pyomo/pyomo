from pyomo.environ import (ConcreteModel, Var, Constraint, ConstraintList,
                           Objective, Integers, Reals, RangeSet, Block,
                           TransformationFactory, SolverFactory,
                           sqrt, value)
from pyomo.core.base.symbolic import differentiate
from pyomo.core.expr.current import identify_variables
from pyomo.common.config import (ConfigBlock, ConfigValue, In,
                                 PositiveFloat, PositiveInt)
from pyomo.contrib.gdpopt.util import (create_utility_block, process_objective,
                                       setup_results_object, a_logger,
                                       copy_var_list_values)
from pyomo.contrib.mindtpy.util import MindtPySolveData
from pyomo.contrib.preprocessing.plugins.int_to_binary import IntegerToBinary
from pyomo.opt import SolverResults
from IPython.core.debugger import Pdb


class IterationLimitExceeded(Exception):
    """Thrown when an iteration limit gets exceeded"""

    def __init__(self, n_iter):
        super().__init__(f"Iteration limit exceeded with N={n_iter}")


def add_L1_objective_function(model, setpoint_var_list):
    """Adds minimum absolute distance objective function"""
    model.L1_obj_var = Var(domain=Reals, bounds=(0, None))
    model.L1_obj_fun = Objective(expr=model.L1_obj_var)
    model.L1_obj_ub_idx = RangeSet(
        len(model.MindtPy_utils.variable_list))
    model.L1_obj_ub_constr = Constraint(
        model.L1_obj_ub_idx, rule=lambda i: model.L1_obj_var >= 0)
    model.L1_obj_lb_idx = RangeSet(
        len(model.MindtPy_utils.variable_list))
    model.L1_obj_lb_constr = Constraint(
        model.L1_obj_lb_idx, rule=lambda i: model.L1_obj_var >= 0)

    for (c_lb, c_ub, v_model, v_setpoint) in zip(model.L1_obj_lb_idx,
                                                 model.L1_obj_ub_idx,
                                                 model.var_list,
                                                 setpoint_var_list):
        model.L1_obj_lb_constr[c_lb].set_value(
            expr=v_model - v_setpoint.value >= -model.L1_obj_var)
        model.L1_obj_ub_constr[c_ub].set_value(
            expr=v_model - v_setpoint.value <= model.L1_obj_var)


def add_L2_objective_function(model, setpoint_var_list):
    """Adds minimum euclidean distance objective function"""
    model.L2_obj_fun = Objective(expr=sqrt(
        sum([(nlp_var - milp_var.value)**2
             for (nlp_var, milp_var) in
             zip(model.var_list, setpoint_var_list)])))


def solution_distance(nlp_vars, milp_vars):
    """Calculates the euclidean norm between two var lists"""
    return sqrt(sum([(nlp_var.value - milp_var.value)**2
                     for (nlp_var, milp_var) in
                     zip(nlp_vars, milp_vars)]))


def setup_milp_model(solve_data):
    """Sets up MILP from MINLP by deactivating nonliear constraints"""
    solve_data.milp_model = solve_data.working_model.clone()
    for obj in solve_data.milp_model.component_data_objects(
            Objective, active=True):
        obj.deactivate()
    for constr in solve_data.milp_model. \
            MindtPy_utils.nonlin_constraints:
        constr.deactivate()
    if hasattr(solve_data.milp_model.MindtPy_utils, 'objective_constr'):
        solve_data.milp_model.MindtPy_utils.objective_constr.deactivate()
    return solve_data.milp_model


def setup_nlp_model(solve_data):
    """Sets up NLP from MINLP working model by fixing discrete variables"""
    # --- Setup NLP model ---
    solve_data.nlp_model = solve_data.working_model.clone()
    for obj in solve_data.nlp_model.component_data_objects(
            Objective, active=True):
        obj.deactivate()
    if hasattr(solve_data.nlp_model.MindtPy_utils, 'objective_constr'):
        solve_data.nlp_model.MindtPy_utils.objective_constr.deactivate()
    TransformationFactory('core.relax_integrality'). \
        apply_to(solve_data.nlp_model)
    return solve_data.nlp_model


def initialize_vars_to_lower_bound(model):
    """Initializes all model variables to their lower bound"""
    for var in model.var_list:
        if var.lb is not None:
            var.set_value(var.lb)
        elif var.ub is not None:
            var.set_value(var.ub)
        else:
            var.set_value(0)


def solve_feasibility_pump(solve_data, config, write_back=True):
    """Solves NLP and MILP subproblems until a feasible solution is  found

    If `write_back==True`, the resulting feasible solution is copied to
    the working model
    """
    opt_milp = SolverFactory(config.mip_solver)
    opt_nlp = SolverFactory(config.nlp_solver)
    n_iter = 0

    nlp_model = setup_nlp_model(solve_data)
    milp_model = setup_milp_model(solve_data)
    copy_var_list_values(solve_data.working_model.var_list,
                         nlp_model.var_list, config)
    with open('iter_log/dist', 'w') as logfile:
        while n_iter == 0 or \
                solution_distance(nlp_model.var_list, milp_model.var_list) \
                > config.bound_tolerance:

            n_iter += 1
            if n_iter >= config.iteration_limit:
                raise IterationLimitExceeded(n_iter)

            # Solve MILP auxiliary problem
            milp_model = setup_milp_model(solve_data)
            add_L1_objective_function(milp_model, nlp_model.var_list)
            opt_milp.solve(milp_model)

            # Solve NLP auxiliary problem
            nlp_model = setup_nlp_model(solve_data)
            add_L2_objective_function(nlp_model, milp_model.var_list)
            opt_nlp.solve(nlp_model)

            logfile.write(str(solution_distance(nlp_model.var_list, milp_model.var_list)))
            logfile.write('\n')

    if write_back:
        copy_var_list_values(
            nlp_model.var_list,
            solve_data.working_model.var_list,
            config)


def solve_fixed_nlp(solve_data, config, write_back=True):
    """Fixes discretes and solves NLP

    An auxiliary problem is cloned from the working model and stored in
    solve_data. The discrete variables get fixed and the resulting NLP is
    solved.
    If `write_back==True`, the result is written back to the working model.
    """
    solve_data.fixed_integer_model = solve_data.working_model.clone()
    fixed_discretes_model = solve_data.fixed_integer_model
    TransformationFactory('core.fix_discrete').apply_to(
        fixed_discretes_model)
    opt_nlp = SolverFactory('ipopt')
    opt_nlp.solve(fixed_discretes_model)
    # opt_nlp = SolverFactory(config.nlp_solver)
    # opt_nlp.solve(fixed_discretes_model, tee=True, **config.nlp_solver_args)
    if write_back:
        copy_var_list_values(fixed_discretes_model.var_list,
                             solve_data.working_model.var_list, config)


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
                  + config.bound_tolerance <= 0))


def create_no_good_cut(solve_data):
    solve_data.working_model.MindtPy_utils. \
        linear_cuts.no_good_combination_cuts.add(
            expr=(sum(var if var.value == 0 else (1-var)
                      for var in solve_data.working_model.var_list
                      if var.is_binary()) >= 1))


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


def setup_config():
    """This is later done in the MindtPy main file"""
    config = ConfigBlock("FeasabilityPump")
    config.declare("integer_tolerance", ConfigValue(
        default=1E-5,
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
    config.declare("nlp_solver", ConfigValue(
        default="gams",
        domain=In(["gams", "ipopt", "ipopth", "conopt"]),
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
    config.nlp_solver_args.add('add_options', f'option optrc={config.bound_tolerance*1e-2};')

    config.declare("mip_solver", ConfigValue(
        default="gams",
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
    config.mip_solver_args.add('solver', 'gurobi')
    config.mip_solver_args.add('add_options', f'option optrc={config.bound_tolerance*1e-2};')

    config.declare("logger", ConfigValue(
        default='pyomo.contrib.mindtpy',
        description='The logger object name to use for reporting.',
        domain=a_logger))
    config.declare("iteration_limit", ConfigValue(
        default=50,
        domain=PositiveInt,
        description="Iteration limit",
        doc="Number of maximum iterations in the decomposition methods"
    ))
    return config


def do_the_solving(model):
    """Runs the feasibility pump with objective cuts"""

    solve_data = MindtPySolveData()
    solve_data.results = SolverResults()
    solve_data.original_model = model
    solve_data.working_model = model.clone()
    working_model = solve_data.working_model
    config = setup_config()
    TransformationFactory('contrib.integer_to_binary'). \
        apply_to(solve_data.working_model)

    with create_utility_block(working_model, 'MindtPy_utils', solve_data):
        # process_objective(solve_data, config)
        mp_utils = working_model.MindtPy_utils
        mp_utils.linear_cuts = Block()
        mp_utils.linear_cuts.increasing_objective_cuts = ConstraintList(
            doc="Objective values need to increase")
        mp_utils.linear_cuts.no_good_combination_cuts = ConstraintList(
            doc="Discrete combination needs to change")
        setup_results_object(solve_data, config)
        mp_utils.lin_constraints = [c for c in mp_utils.constraint_list
                                    if c.body.polynomial_degree() in (0, 1)]
        mp_utils.nonlin_constraints = [c for c in mp_utils.constraint_list
                                       if c.body.polynomial_degree()
                                       not in (0, 1)]
        working_model.var_list = mp_utils.variable_list

        initialize_vars_to_lower_bound(working_model)
        n_feas_problem_iter = 0
        while n_feas_problem_iter <= config.iteration_limit:
            try:
                solve_feasibility_pump(solve_data, config)
            except IterationLimitExceeded:
                print('Found optimal solution!')
                print(working_model.obj.expr())
                # working_model.pprint()
                break
            except RuntimeError as e:
                print(e)
                print('Gams proved infeasibility -- optimal solution found!')
                pass
                break

            try:
                solve_fixed_nlp(solve_data, config)
            except RuntimeError as e:
                print(e)
                break
            create_linearized_objective_cuts(solve_data, config)
            create_no_good_cut(solve_data)
            print(f'{solve_data.working_model.obj.expr()}')
            print([v.value for v in solve_data.working_model.var_list if not v.is_continuous()])
        else:
            raise IterationLimitExceeded(
                    "Too many cuts generated."
                    "It doesn't seem the algorithm converges")

    return solve_data

if __name__ == "__main__":
    do_the_solving(setup_simple_model())
