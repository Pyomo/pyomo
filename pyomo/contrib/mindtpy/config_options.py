from pyomo.common.config import (
    ConfigBlock, ConfigValue, In, PositiveFloat, PositiveInt, NonNegativeInt)
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger


def _get_GDPopt_config():
    CONFIG = ConfigBlock("MindtPy")
    CONFIG.declare("bound_tolerance", ConfigValue(
        default=1E-4,
        domain=PositiveFloat,
        description="Bound tolerance",
        doc="Relative tolerance for bound feasibility checks."
    ))
    CONFIG.declare("iteration_limit", ConfigValue(
        default=50,
        domain=PositiveInt,
        description="Iteration limit",
        doc="Number of maximum iterations in the decomposition methods."
    ))
    CONFIG.declare("stalling_limit", ConfigValue(
        default=15,
        domain=PositiveInt,
        description="Stalling limit",
        doc="Stalling limit for progress in the decomposition methods."
    ))
    CONFIG.declare("time_limit", ConfigValue(
        default=600,
        domain=PositiveInt,
        description="Time limit (seconds, default=600)",
        doc="Seconds allowed until terminated. Note that the time limit can"
            "currently only be enforced between subsolver invocations. You may"
            "need to set subsolver time limits as well."
    ))
    CONFIG.declare("strategy", ConfigValue(
        default="OA",
        domain=In(["OA", "GBD", "ECP", "PSC", "GOA"]),
        description="Decomposition strategy",
        doc="MINLP Decomposition strategy to be applied to the method. "
            "Currently available Outer Approximation (OA), Extended Cutting "
            "Plane (ECP), Partial Surrogate Cuts (PSC), and Generalized "
            "Benders Decomposition (GBD)."
    ))
    CONFIG.declare("init_strategy", ConfigValue(
        default=None,
        domain=In(["rNLP", "initial_binary", "max_binary"]),
        description="Initialization strategy",
        doc="Initialization strategy used by any method. Currently the "
            "continuous relaxation of the MINLP (rNLP), solve a maximal "
            "covering problem (max_binary), and fix the initial value for "
            "the integer variables (initial_binary)."
    ))
    CONFIG.declare("max_slack", ConfigValue(
        default=1000.0,
        domain=PositiveFloat,
        description="Maximum slack variable",
        doc="Maximum slack variable value allowed for the Outer Approximation "
            "cuts."
    ))
    CONFIG.declare("OA_penalty_factor", ConfigValue(
        default=1000.0,
        domain=PositiveFloat,
        description="Outer Approximation slack penalty factor",
        doc="In the objective function of the Outer Approximation method, the "
            "slack variables corresponding to all the constraints get "
            "multiplied by this number and added to the objective."
    ))
    CONFIG.declare("ecp_tolerance", ConfigValue(
        default=None,
        domain=PositiveFloat,
        description="ECP tolerance",
        doc="Feasibility tolerance used to determine the stopping criterion in"
            "the ECP method. As long as nonlinear constraint are violated for "
            "more than this tolerance, the method will keep iterating."
    ))
    CONFIG.declare("nlp_solver", ConfigValue(
        default="ipopt",
        domain=In(["ipopt", "gams", "baron"]),
        description="NLP subsolver name",
        doc="Which NLP subsolver is going to be used for solving the nonlinear"
            "subproblems."
    ))
    CONFIG.declare("nlp_solver_args", ConfigBlock(
        implicit=True,
        description="NLP subsolver options",
        doc="Which NLP subsolver options to be passed to the solver while "
            "solving the nonlinear subproblems."
    ))
    CONFIG.declare("mip_solver", ConfigValue(
        default="glpk",
        domain=In(["gurobi", "cplex", "cbc", "glpk", "gams",
                   "gurobi_persistent", "cplex_persistent"]),
        description="MIP subsolver name",
        doc="Which MIP subsolver is going to be used for solving the mixed-"
            "integer master problems."
    ))
    CONFIG.declare("mip_solver_args", ConfigBlock(
        implicit=True,
        description="MIP subsolver options",
        doc="Which MIP subsolver options to be passed to the solver while "
            "solving the mixed-integer master problems."
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
    CONFIG.declare("solver_tee", ConfigValue(
        default=False,
        description="Stream the output of mip solver and nlp solver to terminal.",
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
                    "smaller in absolute value than the following."
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
        default=1E-7,
        description="Tolerance on variable equal to zero."
    ))
    CONFIG.declare("initial_feas", ConfigValue(
        default=True,
        description="Apply an initial feasibility step.",
        domain=bool
    ))
    CONFIG.declare("obj_bound", ConfigValue(
        default=1E15,
        domain=PositiveFloat,
        description="Bound applied to the linearization of the objective function if master MILP is unbounded."
    ))
    CONFIG.declare("integer_to_binary", ConfigValue(
        default=False,
        description="Convert integer variables to binaries (for integer cuts).",
        domain=bool
    ))
    CONFIG.declare("add_nogood_cuts", ConfigValue(
        default=False,
        description="Add integer cuts (no-good cuts) to binary variables to disallow same integer solution again."
                    "Note that 'integer_to_binary' flag needs to be used to apply it to actual integers and not just binaries.",
        domain=bool
    ))
    CONFIG.declare("single_tree", ConfigValue(
        default=False,
        description="Use single tree implementation in solving the MILP master problem.",
        domain=bool
    ))
    CONFIG.declare("solution_pool", ConfigValue(
        default=False,
        description="Use solution pool in solving the MILP master problem.",
        domain=bool
    ))
    CONFIG.declare("add_slack", ConfigValue(
        default=False,
        description="whether add slack variable here."
                    "slack variables here are used to deal with nonconvex MINLP.",
        domain=bool
    ))
    CONFIG.declare("continuous_var_bound", ConfigValue(
        default=1e10,
        description="default bound added to unbounded continuous variables in nonlinear constraint if single tree is activated.",
        domain=PositiveFloat
    ))
    CONFIG.declare("integer_var_bound", ConfigValue(
        default=1e9,
        description="default bound added to unbounded integral variables in nonlinear constraint if single tree is activated.",
        domain=PositiveFloat
    ))
    CONFIG.declare("cycling_check", ConfigValue(
        default=True,
        description="check if OA algorithm is stalled in a cycle and terminate.",
        domain=bool
    ))
    CONFIG.declare("feasibility_norm", ConfigValue(
        default="L_infinity",
        domain=In(["L1", "L2", "L_infinity"]),
        description="different forms of objective function in feasibility subproblem."
    ))
    CONFIG.declare("differentiate_mode", ConfigValue(
        default="reverse_symbolic",
        domain=In(["reverse_symbolic", "sympy"]),
        description="differentiate mode to calculate jacobian."
    ))
    CONFIG.declare("linearize_inactive", ConfigValue(
        default=False,
        description="Add OA cuts for inactive constraints.",
        domain=bool
    ))
    CONFIG.declare("use_mcpp", ConfigValue(
        default=False,
        description="use package MC++ to set a bound for variable 'objective_value', which is introduced when the original problem's objective function is nonlinear.",
        domain=bool
    ))
    CONFIG.declare("use_dual", ConfigValue(
        default=True,
        description="use dual solution from the nlp solver to add OA cuts for equality constraints.",
        domain=bool
    ))
    CONFIG.declare("use_fbbt", ConfigValue(
        default=False,
        description="use fbbt to tighten the feasible region of the problem",
        domain=bool
    ))
    CONFIG.declare("threads", ConfigValue(
        default=0,
        domain=NonNegativeInt,
        description="Threads",
        doc="Threads used by milp solver and nlp solver."
    ))
    CONFIG.declare("use_dual_bound", ConfigValue(
        default=True,
        description="add dual bound constraint to enforce the objective function should improve on the best found dual bound",
        domain=bool
    ))
    return CONFIG
