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
from pyomo.common.collections import Bunch, ComponentSet
from pyomo.common.config import (
    ConfigDict, ConfigValue, In, NonNegativeFloat, add_docstring_list
)
from pyomo.core.base.block import Block
from pyomo.core.expr import value
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.param import Param, _ParamData
from pyomo.core.base.objective import Objective, maximize
from pyomo.contrib.pyros.util import (a_logger,
                                       time_code,
                                       get_main_elapsed_time)
from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import (model_is_valid,
                                      add_decision_rule_constraints,
                                      add_decision_rule_variables,
                                      load_final_solution,
                                      pyrosTerminationCondition,
                                      ValidEnum,
                                      ObjectiveType,
                                      validate_uncertainty_set,
                                      identify_objective_functions,
                                      validate_kwarg_inputs,
                                      transform_to_standard_form,
                                      turn_bounds_to_constraints,
                                      replace_uncertain_bounds_with_constraints,
                                      output_logger)
from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.pyros_algorithm_methods import ROSolver_iterative_solve
from pyomo.contrib.pyros.uncertainty_sets import uncertainty_sets
from pyomo.core.base import Constraint

__version__ =  "1.0.0"

def NonNegIntOrMinusOne(obj):
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

def PositiveIntOrMinusOne(obj):
    '''
    if obj is a positive int, return the int
    if obj is -1, return -1
    else, error
    '''
    ans = int(obj)
    if ans != float(obj) or (ans <= 0 and ans != -1):
        raise ValueError(
            "Expected positive int, but received %s" % (obj,))
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
    CONFIG.declare('time_limit', ConfigValue(
        default=None,
        domain=NonNegativeFloat, description="Optional. Default = None. "
                                             "Total allotted time for the execution of the PyROS solver in seconds "
                                             "(includes time spent in sub-solvers). 'None' is no time limit."
    ))
    CONFIG.declare('keepfiles', ConfigValue(
        default=False,
        domain=bool, description="Optional. Default = False. Whether or not to write files of sub-problems for use in debugging. "
                                 "Must be paired with a writable directory supplied via ``subproblem_file_directory``."
    ))
    CONFIG.declare('tee', ConfigValue(
        default=False,
        domain=bool, description="Optional. Default = False. Sets the ``tee`` for all sub-solvers utilized."
    ))
    CONFIG.declare('load_solution', ConfigValue(
        default=True,
        domain=bool, description="Optional. Default = True. "
                                 "Whether or not to load the final solution of PyROS into the model object."
    ))

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
    CONFIG.declare("max_iter", ConfigValue(
        default=-1, domain=PositiveIntOrMinusOne,
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
    CONFIG.declare("subproblem_file_directory", ConfigValue(
        default=None, domain=str,
        description="Optional. Path to a directory where subproblem files and "
                    "logs will be written in the case that a subproblem fails to solve."
    ))
    # ================================================
    # === Advanced Options
    # ================================================
    CONFIG.declare("bypass_local_separation", ConfigValue(
        default=False, domain=bool,
        description="This is an advanced option. Default = False. 'True' to only use global solver(s) during separation; "
                    "'False' to use local solver(s) at intermediate separations, "
                    "using global solver(s) only before termination to certify robust feasibility. "
    ))
    CONFIG.declare("bypass_global_separation", ConfigValue(
        default=False, domain=bool,
        description="This is an advanced option. Default = False. 'True' to only use local solver(s) during separation; "
                    "however, robustness of the final result will not be guaranteed. Use to expedite PyROS run when "
                    "global solver(s) cannot (efficiently) solve separation problems."
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

    # The Pyomo solver API expects that solvers support the context
    # manager API
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        pass

    def solve(self, model, first_stage_variables, second_stage_variables,
              uncertain_params, uncertainty_set, local_solver, global_solver,
              **kwds):
        """Solve the model.

        Parameters
        ----------
        model: ConcreteModel
            A ``ConcreteModel`` object representing the deterministic
            model, cast as a minimization problem.
        first_stage_variables: List[Var]
            The list of ``Var`` objects referenced in ``model``
            representing the design variables.
        second_stage_variables: List[Var]
            The list of ``Var`` objects referenced in ``model``
            representing the control variables.
        uncertain_params: List[Param]
            The list of ``Param`` objects referenced in ``model``
            representing the uncertain parameters.  MUST be ``mutable``.
            Assumes entries are provided in consistent order with the
            entries of 'nominal_uncertain_param_vals' input.
        uncertainty_set: UncertaintySet
            ``UncertaintySet`` object representing the uncertainty space
            that the final solutions will be robust against.
        local_solver: Solver
            ``Solver`` object to utilize as the primary local NLP solver.
        global_solver: Solver
            ``Solver`` object to utilize as the primary global NLP solver.

        """

        # === Add the explicit arguments to the config
        config = self.CONFIG(kwds.pop('options', {}))
        config.first_stage_variables = first_stage_variables
        config.second_stage_variables = second_stage_variables
        config.uncertain_params = uncertain_params
        config.uncertainty_set = uncertainty_set
        config.local_solver = local_solver
        config.global_solver = global_solver

        dev_options = kwds.pop('dev_options',{})
        config.set_value(kwds)
        config.set_value(dev_options)

        model = model

        # === Validate kwarg inputs
        validate_kwarg_inputs(model, config)

        # === Validate ability of grcs RO solver to handle this model
        if not model_is_valid(model):
            raise AttributeError("This model structure is not currently handled by the ROSolver.")

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

            # === Validate uncertainty set happens here, requires util block for Cardinality and FactorModel sets
            validate_uncertainty_set(config=config)

            # === Deactivate objective on model
            for o in model.component_data_objects(Objective):
                o.deactivate()

            # === Leads to a logger warning here for inactive obj when cloning
            model_data.original_model = model
            # === For keeping track of variables after cloning
            cname = unique_component_name(model_data.original_model, 'tmp_var_list')
            src_vars = list(model_data.original_model.component_data_objects(Var))
            setattr(model_data.original_model, cname, src_vars)
            model_data.working_model = model_data.original_model.clone()

            # === Add objective expressions
            identify_objective_functions(model_data.working_model, config)

            # === Put model in standard form
            transform_to_standard_form(model_data.working_model)

            # === Replace variable bounds depending on uncertain params with
            #     explicit inequality constraints
            replace_uncertain_bounds_with_constraints(model_data.working_model,
                                                      model_data.working_model.util.uncertain_params)

            # === Add decision rule information
            add_decision_rule_variables(model_data, config)
            add_decision_rule_constraints(model_data, config)

            # === Move bounds on control variables to explicit ineq constraints
            wm_util = model_data.working_model

            # === Assuming all other Var objects in the model are state variables
            fsv = ComponentSet(model_data.working_model.util.first_stage_variables)
            ssv = ComponentSet(model_data.working_model.util.second_stage_variables)
            sv = ComponentSet()
            model_data.working_model.util.state_vars = []
            for v in model_data.working_model.component_data_objects(Var):
                if v not in fsv and v not in ssv and v not in sv:
                    model_data.working_model.util.state_vars.append(v)
                    sv.add(v)

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


            return_soln = ROSolveResults()
            if pyros_soln is not None and final_iter_separation_solns is not None:
                if config.load_solution and \
                        (pyros_soln.pyros_termination_condition is pyrosTerminationCondition.robust_optimal or
                         pyros_soln.pyros_termination_condition is pyrosTerminationCondition.robust_feasible):
                    load_final_solution(model_data, pyros_soln.master_soln, config)

                # === Return time info
                model_data.total_cpu_time = get_main_elapsed_time(model_data.timing)
                iterations = pyros_soln.total_iters + 1

                # === Return config to user
                return_soln.config = config
                # Report the negative of the objective value if it was originally maximize, since we use the minimize form in the algorithm
                if next(model.component_data_objects(Objective)).sense == maximize:
                    negation = -1
                else:
                    negation = 1
                if config.objective_focus == ObjectiveType.nominal:
                    return_soln.final_objective_value = negation * value(pyros_soln.master_soln.master_model.obj)
                elif config.objective_focus == ObjectiveType.worst_case:
                    return_soln.final_objective_value = negation * value(pyros_soln.master_soln.master_model.zeta)
                return_soln.pyros_termination_condition = pyros_soln.pyros_termination_condition

                return_soln.time = model_data.total_cpu_time
                return_soln.iterations = iterations

                # === Remove util block
                model.del_component(model_data.util_block)

                del pyros_soln.util_block
                del pyros_soln.working_model
            else:
                return_soln.pyros_termination_condition = pyrosTerminationCondition.robust_infeasible
                return_soln.final_objective_value = None
                return_soln.time = get_main_elapsed_time(model_data.timing)
                return_soln.iterations = 0
        return return_soln


def _generate_filtered_docstring():
    cfg = PyROS.CONFIG()
    del cfg['first_stage_variables']
    del cfg['second_stage_variables']
    del cfg['uncertain_params']
    del cfg['uncertainty_set']
    del cfg['local_solver']
    del cfg['global_solver']
    return add_docstring_list(PyROS.solve.__doc__, cfg, indent_by=8)

PyROS.solve.__doc__ = _generate_filtered_docstring()
