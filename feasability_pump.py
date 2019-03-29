from pyomo.environ import (ConcreteModel, Var, Constraint, Objective,
                           Integers, Reals, RangeSet,
                           TransformationFactory, SolverFactory,
                           sqrt)
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.gdpopt.util import (create_utility_block,
                                       setup_results_object, a_logger)
from pyomo.contrib.mindtpy.util import MindtPySolveData
from pyomo.opt import SolverResults

# --- SETUP MODEL ---
M = ConcreteModel()
M.y = Var(domain=Integers, bounds=(1, 4))
M.x = Var(domain=Reals, bounds=(1, 5))
M.lin_constraint = Constraint(expr=M.y >= 1.6)


def g(m):
    return m.x <= sqrt(1.5*m.y)
    #coeffs = [0.9, -5.75, 12.6, -6]
    #return -sum([c*m.y**i for (i, c) in enumerate(reversed(coeffs))])+m.x <= 0


M.nonlin_constraint = Constraint(rule=g)

# --- Setup workspace ---

solve_data = MindtPySolveData()
solve_data.results = SolverResults()

config = ConfigBlock("FeasabilityPump")

config.declare("nlp_solver", ConfigValue(
    default="ipopt",
    domain=In(["ipopt"]),
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
config.declare("logger", ConfigValue(
    default='pyomo.contrib.mindtpy',
    description='The logger object name to use for reporting.',
    domain=a_logger))
config.declare("iteration_limit", ConfigValue(
    default=30,
    domain=PositiveInt,
    description="Iteration limit",
    doc="Number of maximum iterations in the decomposition methods"
))

with create_utility_block(M, 'MindtPy_utils', solve_data):
    def add_L1_objective_function_constraints(M):
        M.L1_obj_var = Var(domain=Reals, bounds=(0, None))
        M.L1_obj_ub_idx = RangeSet(len(M.MindtPy_utils.variable_list))
        M.L1_obj_ub_constr = Constraint(M.L1_obj_ub_idx, rule=lambda i: M.L1_obj_var >= 0)
        M.L1_obj_lb_idx = RangeSet(len(M.MindtPy_utils.variable_list))
        M.L1_obj_lb_constr = Constraint(M.L1_obj_lb_idx, rule=lambda i: M.L1_obj_var >= 0)

    def update_L1_objective_function_constraints(M, setpoint_var_list):
        for (c_lb, c_ub, v_model, v_setpoint) in zip(M.L1_obj_lb_idx,
                                                     M.L1_obj_ub_idx,
                                                     M.MindtPy_utils.variable_list,
                                                     setpoint_var_list):
            M.L1_obj_lb_constr[c_lb].set_value(expr=v_model - v_setpoint.value >= -M.L1_obj_var)
            M.L1_obj_ub_constr[c_ub].set_value(expr=v_model - v_setpoint.value <= M.L1_obj_var)

    def update_L2_objective_function(M, setpoint_var_list):
        M.L2_obj_fun.set_value(expr=sqrt(
            sum([(nlp_var - milp_var.value)**2
                 for (nlp_var, milp_var) in
                 zip(M.MindtPy_utils.variable_list, setpoint_var_list)])))

    def solution_distance(nlp_vars, milp_vars):
        return sqrt(sum([(nlp_var.value - milp_var.value)**2
                         for (nlp_var, milp_var) in zip(nlp_vars, milp_vars)]))

    solve_data.original_model = M
    solve_data.working_model = M.clone()
    MindtPy = solve_data.working_model.MindtPy_utils
    setup_results_object(solve_data, config)

    MindtPy = solve_data.working_model.MindtPy_utils
    MindtPy.lin_constraints = [c for c in MindtPy.constraint_list if
                               c.body.polynomial_degree() in (0, 1)]
    MindtPy.nonlin_constraints = [c for c in MindtPy.constraint_list if
                                  c.body.polynomial_degree() not in (0, 1)]
    MindtPy.integer_var_list = [v for v in MindtPy.variable_list if v.is_integer()]

    # --- Setup MILP model ---
    solve_data.milp_model = solve_data.working_model.clone()
    milp_model = solve_data.milp_model
    for c in milp_model.MindtPy_utils.nonlin_constraints:
        c.deactivate()

    # --- Setup NLP model ---
    solve_data.nlp_model = solve_data.working_model.clone()
    nlp_model = solve_data.nlp_model
    TransformationFactory('core.relax_integrality').apply_to(nlp_model)
    nlp_model.L2_obj_fun = Objective(expr=1)

    # --- Initialize model vars to lower_bound ---
    for v in nlp_model.MindtPy_utils.variable_list:
        v.set_value(v.lb)
    for v in milp_model.MindtPy_utils.variable_list:
        v.set_value(v.ub)
    add_L1_objective_function_constraints(solve_data.milp_model)
    milp_model.L1_obj_fun = Objective(expr=milp_model.L1_obj_var)

    opt_milp = SolverFactory(config.mip_solver)
    opt_nlp = SolverFactory('gams')

    # --- Begin feasibility pump iteration ---
    while True:
        num_iter = 0
        while (solution_distance(nlp_model.MindtPy_utils.variable_list,
                                 milp_model.MindtPy_utils.variable_list)
               >= 1e-4 and num_iter < config.iteration_limit):
            num_iter += 1
            print('NLP:',[(v.name, v.value) for v in nlp_model.MindtPy_utils.variable_list])
            print('MILP:',[(v.name, v.value) for v in milp_model.MindtPy_utils.variable_list])
            print(solution_distance(nlp_model.MindtPy_utils.variable_list,
                                         milp_model.MindtPy_utils.variable_list))
            update_L1_objective_function_constraints(milp_model,
                                                     nlp_model.MindtPy_utils.variable_list)
            opt_milp.solve(milp_model, **config.mip_solver_args)

            update_L2_objective_function(nlp_model, milp_model.MindtPy_utils.variable_list)
            opt_nlp.solve(nlp_model, solver='ipopth', **config.nlp_solver_args, tee=True)

        if num_iter == config.iteration_limit:
            print('Problem is now infeasible => Found optimal solution?')
            break

        # Solve NLP with fixed integer values

    print('Final NLP model')
    nlp_model.pprint()
