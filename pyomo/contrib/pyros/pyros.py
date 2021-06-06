#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# pyros.py: Generalized Robust Cutting-Set Algorithm for Pyomo
import logging
from pyomo.common.collections import Bunch
from pyomo.common.config import (
    ConfigDict, ConfigValue, In, NonNegativeFloat, add_docstring_list
)
from pyomo.core.base.block import Block
from pyomo.core.expr import value
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.param import Param, _ParamData
from pyomo.core.base.objective import Objective
from pyomo.contrib.pyros.util import (a_logger,
                                       time_code,
                                       get_main_elapsed_time)
from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import (model_is_valid,
                                      add_decision_rule_constraints,
                                      add_decision_rule_variables,
                                      load_final_solution,
                                      grcsTerminationCondition,
                                      ValidEnum,
                                      ObjectiveType,
                                      validate_uncertainty_set,
                                      identify_objective_functions,
                                      validate_kwarg_inputs,
                                      transform_to_standard_form,
                                      turn_bounds_to_constraints,
                                      output_logger)
from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.pyros_algorithm_methods import ROSolver_iterative_solve
from pyomo.contrib.pyros.uncertainty_sets import uncertainty_sets
from pyomo.core.base import Constraint



__version__ =  "0.1.0 (beta)"

class NonNegIntOrMinusOne(object):

    def __call__(self, obj):
        '''
        if obj is a non-negative int, return the non-negative int
        if obj is -1, return -1
        else, error
        '''
        ans = int(obj)
        if ans != float(obj) or (ans < 0 and ans != -1):
            raise ValueError(
                "Expected non-negative int, but received %s" % (obj,))
        return ans

class SolverResolvable(object):

    def __call__(self, obj):
        '''
        if obj is a string, return the Solver object for that solver name
        if obj is a Solver object, return the Solver
        if obj is a list, and each element of list is solver resolvable, return list of solvers
        '''
        if isinstance(obj, str):
            return SolverFactory(obj.lower())
        elif callable(getattr(obj, "solve", None)):
            return obj
        elif isinstance(obj, list):
            return [self(o) for o in obj]
        else:
            raise ValueError("Expected a Pyomo solver or string object, "
                             "instead recieved {1}".format(obj.__class__.__name__))

class InputDataStandardizer(object):
    def __init__(self, ctype, cdatatype):
        self.ctype = ctype
        self.cdatatype = cdatatype

    def __call__(self, obj):
        if isinstance(obj, self.ctype):
            return list(obj.values())
        if isinstance(obj, self.cdatatype):
            return [obj]
        ans = []
        for item in obj:
            ans.extend(self.__call__(item))
        for _ in ans:
            assert isinstance(_, self.cdatatype)
        return ans

def pyros_config():
    CONFIG = ConfigDict('PyROS')

    # ================================================
    # === Options common to all solvers
    # ================================================
    '''CONFIG.declare('time_limit', ConfigValue(
        default=None,
        domain=NonNegativeFloat
    ))
    CONFIG.declare('keepfiles', ConfigValue(
        default=False,
        domain=bool,
    ))
    CONFIG.declare('tee', ConfigValue(
        default=False,
        domain=bool,
    ))
    CONFIG.declare('load_solution', ConfigValue(
        default=True,
        domain=bool,
    ))'''

    # ================================================
    # === Required User Inputs
    # ================================================
    CONFIG.declare("first_stage_variables", ConfigValue(
        default=[], domain=InputDataStandardizer(Var, _VarData),
        description="Required. List of ``Var`` objects referenced in ``model`` representing the design variables."
    ))
    CONFIG.declare("second_stage_variables", ConfigValue(
        default=[], domain=InputDataStandardizer(Var, _VarData),
        description="Required. List of ``Var`` referenced in ``model`` representing the control variables."
    ))
    CONFIG.declare("uncertain_params", ConfigValue(
        default=[], domain=InputDataStandardizer(Param, _ParamData),
        description="Required. List of ``Param`` referenced in ``model`` representing the uncertain parameters. MUST be ``mutable``. "
                    "Assumes entries are provided in consistent order with the entries of 'nominal_uncertain_param_vals' input."
    ))
    CONFIG.declare("uncertainty_set", ConfigValue(
        default=None, domain=uncertainty_sets,
        description="Required. ``UncertaintySet`` object representing the uncertainty space "
                    "that the final solutions will be robust against."
    ))
    CONFIG.declare("local_solver", ConfigValue(
        default=None, domain=SolverResolvable(),
        description="Required. ``Solver`` object to utilize as the primary local NLP solver."
    ))
    CONFIG.declare("global_solver", ConfigValue(
        default=None, domain=SolverResolvable(),
        description="Required. ``Solver`` object to utilize as the primary global NLP solver."
    ))
    # ================================================
    # === Optional User Inputs
    # ================================================
    CONFIG.declare("objective_focus", ConfigValue(
        default=ObjectiveType.nominal, domain=ValidEnum(ObjectiveType),
        description="Optional. Default = ``ObjectiveType.nominal``. Choice of objective function to optimize in the master problems. "
                    "Choices are: ``ObjectiveType.worst_case``, ``ObjectiveType.nominal``. See Note for details."
    ))
    CONFIG.declare("nominal_uncertain_param_vals", ConfigValue(
        default=[], domain=list,
        description="Optional. Default = deterministic model ``Param`` values. List of nominal values for all uncertain parameters. "
                    "Assumes entries are provided in consistent order with the entries of ``uncertain_params`` input."
    ))
    CONFIG.declare("decision_rule_order", ConfigValue(
        default=0, domain=In([0, 1, 2]),
        description="Optional. Default = 0. Order of decision rule functions for handling second-stage variable recourse. "
                    "Choices are: '0' for constant recourse (a.k.a. static approximation), '1' for affine recourse "
                    "(a.k.a. affine decision rules), '2' for quadratic recourse."
    ))
    CONFIG.declare("solve_master_globally", ConfigValue(
        default=False, domain=bool,
        description="Optional. Default = False. 'True' for the master problems to be solved with the user-supplied global solver(s); "
                    "or 'False' for the master problems to be solved with the user-supplied local solver(s). "

    ))
    # This is called pyros_time_limit because time_limit is already defined
    # in the solver interface and I'm defining a different one here with the no limit option.
    CONFIG.declare("pyros_time_limit", ConfigValue(
        default=-1, domain=NonNegIntOrMinusOne(),
        description="Optional. Default = -1. Total allotted time for the execution of the PyROS solver in seconds (includes time spent in sub-solvers). '-1' is no time limit."
    ))
    CONFIG.declare("max_iter", ConfigValue(
        default=-1, domain=NonNegIntOrMinusOne(),
        description="Optional. Default = -1. Iteration limit for the GRCS algorithm. '-1' is no iteration limit."
    ))
    CONFIG.declare("robust_feasibility_tolerance", ConfigValue(
        default=1e-4, domain=NonNegativeFloat,
        description="Optional. Default = 1e-4. Relative tolerance for assessing robust feasibility violation during separation phase."
    ))
    CONFIG.declare("separation_priority_order", ConfigValue(
        default={}, domain=dict,
        description="Optional. Default = {}. Dictionary mapping inequality constraint names to positive integer priorities for separation. "
                    "Constraints not referenced in the dictionary assume a priority of 0 (lowest priority)."
    ))
    CONFIG.declare("progress_logger", ConfigValue(
        default="pyomo.contrib.pyros", domain=a_logger,
        description="Optional. Default = \"pyomo.contrib.pyros\". The logger object to use for reporting."
    ))
    CONFIG.declare("print_subsolver_progress_to_screen", ConfigValue(
        default=False, domain=bool,
        description="Optional. Default = False. Sets the ``tee`` for all sub-solvers utilized."
    ))
    CONFIG.declare("backup_local_solvers", ConfigValue(
        default=[], domain=SolverResolvable(),
        description="Optional. Default = []. List of additional ``Solver`` objects to utilize as backup "
                    "whenever primary local NLP solver fails to identify solution to a sub-problem."
    ))
    CONFIG.declare("backup_global_solvers", ConfigValue(
        default=[], domain=SolverResolvable(),
        description="Optional. Default = []. List of additional ``Solver`` objects to utilize as backup "
                    "whenever primary global NLP solver fails to identify solution to a sub-problem."
    ))
    CONFIG.declare("load_pyros_solution", ConfigValue(
        default=True, domain=bool,
        description="Optional. Default = True. Whether or not to load the final solution of PyROS into the model object."
    ))
    # ================================================
    # === Advanced Options
    # ================================================
    CONFIG.declare("minimize_dr_norm", ConfigValue(
        default=False, domain=bool,
        description="This is an advanced option. Default=True. Whether or not to polish decision rule functions at each iteration. "
    ))
    CONFIG.declare("bypass_local_separation", ConfigValue(
        default=False, domain=bool,
        description="This is an advanced option. Default = False. 'True' to only use global solver(s) during separation; "
                    "'False' to use local solver(s) at intermediate separations, "
                    "using global solver(s) only before termination to certify robust feasibility. "
    ))
    CONFIG.declare("p_robustness", ConfigValue(
        default={}, domain=dict,
        description="This is an advanced option. Default = {}. Whether or not to add p-robustness constraints to the master problems. "
                    "If the dictionary is empty (default), then p-robustness constraints are not added. "
                    "See Note for how to specify arguments."
    ))

    return CONFIG

@SolverFactory.register(
    "pyros",
    doc="Robust optimization (RO) solver implementing "
    "the generalized robust cutting-set algorithm (GRCS)")
class PyROS(object):
    '''
    PyROS (Pyomo Robust Optimization Solver) implementing a
    generalized robust cutting-set algorithm (GRCS)
    to solve two-stage NLP optimization models under uncertainty.
    '''

    CONFIG = pyros_config()

    def available(self, exception_flag=True):
        """Check if solver is available.
        """
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def license_is_valid(self):
        ''' License for using PyROS '''
        return True

    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model: A ``ConcreteModel`` object representing the deterministic model, cast as a minimization problem.
        """

        config = self.CONFIG(kwds.pop('options', {}))
        dev_options = kwds.pop('dev_options',{})
        config.set_value(kwds)
        config.set_value(dev_options)

        model = model

        # === Validate kwarg inputs
        validate_kwarg_inputs(model, config)

        # === Validate ability of grcs RO solver to handle this model
        if not model_is_valid(model):
            raise AttributeError("This model structure is not currently handled by the ROSolver.")

        # === Validate uncertainty set
        validate_uncertainty_set(config=config)

        # === Define nominal point if not specified
        if len(config.nominal_uncertain_param_vals) == 0:
            config.nominal_uncertain_param_vals = list(p.value for p in config.uncertain_params)
        elif len(config.nominal_uncertain_param_vals) != len(config.uncertain_params):
            raise AttributeError("The nominal_uncertain_param_vals list must be the same length"
                                 "as the uncertain_params list")

        # === Create data containers
        model_data = ROSolveResults()
        model_data.timing = Bunch()

        # === Set up logger for logging results
        with time_code(model_data.timing, 'total', is_main_timer=True):
            config.progress_logger.setLevel(logging.INFO)

            # === PREAMBLE
            output_logger(config=config, preamble=True, version=str(self.version()))

            # === DISCLAIMER
            output_logger(config=config, disclaimer=True)

            # === A block to hold list-type data to make cloning easy
            util = Block(concrete=True)
            util.first_stage_variables = config.first_stage_variables
            util.second_stage_variables = config.second_stage_variables
            util.uncertain_params = config.uncertain_params

            model_data.util_block = unique_component_name(model, 'util')
            model.add_component(model_data.util_block, util)
            # Note:  model.component(model_data.util_block) is util

            # === Deactivate objective on model
            for o in model.component_data_objects(Objective):
                o.deactivate()

            # === Leads to a logger warning here for inactive obj when cloning
            model_data.original_model = model
            model_data.working_model = model.clone()

            # === Add objective expressions
            identify_objective_functions(model_data.working_model, config)

            # === Put model in standard form
            transform_to_standard_form(model_data.working_model)

            # === Add decision rule information
            add_decision_rule_variables(model_data, config)
            add_decision_rule_constraints(model_data, config)

            # === Move bounds on control variables to explicit ineq constraints
            wm_util = model_data.working_model

            # === Assuming all other Var objects in the model are state variables
            fsv_ids = list(id(v) for v in model_data.working_model.util.first_stage_variables)
            ssv_ids = list(id(v) for v in model_data.working_model.util.second_stage_variables)
            model_data.working_model.util.state_vars = []
            for v in model_data.working_model.component_data_objects(Var):
                if id(v) not in ssv_ids and id(v) not in fsv_ids \
                        and id(v) not in list(id(state_var) for state_var in model_data.working_model.util.state_vars):
                    model_data.working_model.util.state_vars.append(v)

            # Bounds on second stage variables and state variables are separation objectives,
            #  they are brought in this was as explicit constraints
            for c in model_data.working_model.util.second_stage_variables:
                turn_bounds_to_constraints(c, wm_util, config)

            for c in model_data.working_model.util.state_vars:
                turn_bounds_to_constraints(c, wm_util, config)

            # === Make control_variable_bounds array
            wm_util.ssv_bounds = []
            for c in model_data.working_model.component_data_objects(Constraint, descend_into=True):
                if "bound_con" in c.name:
                    wm_util.ssv_bounds.append(c)

            # === Solve and load solution into model
            pyros_soln, final_iter_separation_solns = ROSolver_iterative_solve(model_data, config)

            if config.load_pyros_solution and \
                    (pyros_soln.grcs_termination_condition is grcsTerminationCondition.robust_optimal or
                     pyros_soln.grcs_termination_condition is grcsTerminationCondition.robust_feasible):
                load_final_solution(model_data, pyros_soln.master_soln)

            # === Return time info
            model_data.total_cpu_time = get_main_elapsed_time(model_data.timing)

            # === Print results
            config.progress_logger.info("Objective: " + str(value(pyros_soln.master_soln.master_model.obj)))
            config.progress_logger.info("Time (s): " + str(model_data.total_cpu_time))
            '''config.progress_logger.info("Time solving Masters (s): " + str(pyros_soln.timing_data.total_master_solve_time))
            config.progress_logger.info("Time solving Separation Local (s): " + str(pyros_soln.timing_data.total_separation_local_time))
            config.progress_logger.info("Time solving Separation Global (s): " + str(pyros_soln.timing_data.total_separation_global_time))
            config.progress_logger.info(
                "Time solving Separation TOTAL (s): " + str(pyros_soln.timing_data.total_separation_global_time +
                                                            pyros_soln.timing_data.total_separation_local_time))
            config.progress_logger.info(
                "Time solving polishing step (s): " + str(pyros_soln.timing_data.total_dr_polish_time))'''
            config.progress_logger.info("Iterations: " + str(pyros_soln.total_iters+1))
            # === Return config to user
            pyros_soln.config = config
            pyros_soln.final_objective_value = value(pyros_soln.master_soln.master_model.obj)
            # === Remove util block
            model.del_component(model_data.util_block)

            del pyros_soln.util_block
            del pyros_soln.working_model
        return pyros_soln


PyROS.solve.__doc__ = add_docstring_list(
    PyROS.solve.__doc__, PyROS.CONFIG, indent_by=8)
