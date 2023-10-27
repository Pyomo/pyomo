import pyomo.environ as pyo
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.PyomoModel import ConcreteModel
import warnings
from pyomo.common.collections import ComponentMap
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.contrib import appsi
import logging
import traceback
import numpy as np
import math
import time
from typing import Union, Sequence, Optional, List
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
try:
    import coramin.utils.mpi_utils as mpiu
    mpi_available = True
except ImportError:
    mpi_available = False
try:
    from tqdm import tqdm
except ImportError:
    pass


logger = logging.getLogger(__name__)


class OBBTInfo(object):
    def __init__(self):
        self.total_num_problems = None
        self.num_problems_attempted = None
        self.num_successful_problems = None


def _bt_cleanup(
    model, solver: Union[appsi.base.Solver, appsi.base.PersistentSolver],
    vardatalist: Optional[List[_GeneralVarData]],
    initial_var_values, deactivated_objectives, orig_update_config, orig_config,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None
):
    """
    Cleanup the changes made to the model during bounds tightening.
    Reactivate any deactivated objectives.
    Remove an objective upper bound constraint if it was added.
    If lower_bounds or upper_bounds is provided, update the bounds of the variables in self.vars_to_tighten.

    Parameters
    ----------
    model: pyo.ConcreteModel or pyo.Block
    solver: Union[appsi.base.Solver, appsi.base.PersistentSolver]
    vardatalist: List of _GeneralVarData
    initial_var_values: ComponentMap
    deactivated_objectives: list of _GeneralObjectiveData
    orig_update_config: appsi.base.UpdateConfig
    orig_config: appsi.base.SolverConfig
    lower_bounds: Sequence of float
        Only needed if you want to update the bounds of the variables. Should be in the same order as
        self.vars_to_tighten.
    upper_bounds: Sequence of float
        Only needed if you want to update the bounds of the variables. Should be in the same order as
        self.vars_to_tighten.
    """
    for v in model.component_data_objects(ctype=pyo.Var, active=None, sort=True, descend_into=True):
        v.set_value(initial_var_values[v], skip_validation=True)

    if hasattr(model, '__objective_ineq'):
        if solver.is_persistent():
            solver.remove_constraints([model.__objective_ineq])
        del model.__objective_ineq

    # reactivate the objectives that we deactivated
    for obj in deactivated_objectives:
        obj.activate()
        if solver.is_persistent():
            solver.set_objective(obj)

    if lower_bounds is not None and upper_bounds is not None:
        for i, v in enumerate(vardatalist):
            lb = lower_bounds[i]
            ub = upper_bounds[i]
            v.setlb(lb)
            v.setub(ub)
    elif lower_bounds is not None:
        for i, v in enumerate(vardatalist):
            lb = lower_bounds[i]
            v.setlb(lb)
    elif upper_bounds is not None:
        for i, v in enumerate(vardatalist):
            ub = upper_bounds[i]
            v.setub(ub)
    if vardatalist is not None and solver.is_persistent():
        solver.update_variables(vardatalist)

    if solver.is_persistent():
        solver.update_config.check_for_new_or_removed_constraints = \
            orig_update_config.check_for_new_or_removed_constraints
        solver.update_config.check_for_new_or_removed_vars = \
            orig_update_config.check_for_new_or_removed_vars
        solver.update_config.check_for_new_or_removed_params = \
            orig_update_config.check_for_new_or_removed_params
        solver.update_config.check_for_new_objective = \
            orig_update_config.check_for_new_objective
        solver.update_config.update_constraints = \
            orig_update_config.update_constraints
        solver.update_config.update_vars = \
            orig_update_config.update_vars
        solver.update_config.update_params = \
            orig_update_config.update_params
        solver.update_config.update_named_expressions = \
            orig_update_config.update_named_expressions
        solver.update_config.update_objective = \
            orig_update_config.update_objective
        solver.update_config.treat_fixed_vars_as_params = \
            orig_update_config.treat_fixed_vars_as_params

    solver.config.stream_solver = orig_config.stream_solver
    solver.config.load_solution = orig_config.load_solution


def _single_solve(v, model, solver: Union[appsi.base.Solver, appsi.base.PersistentSolver], lb_or_ub, obbt_info):
    obbt_info.num_problems_attempted += 1
    # solve for lower var bound
    if lb_or_ub == 'lb':
        model.__obj_bounds_tightening = ScalarObjective(expr=v, sense=pyo.minimize)
    else:
        assert lb_or_ub == 'ub'
        model.__obj_bounds_tightening = ScalarObjective(expr=v, sense=pyo.maximize)

    if solver.is_persistent():
        solver.set_objective(model.__obj_bounds_tightening)
    results = solver.solve(model)
    if results.termination_condition == appsi.base.TerminationCondition.optimal:
        obbt_info.num_successful_problems += 1
    if results.best_objective_bound is not None and math.isfinite(results.best_objective_bound):
        new_bnd = results.best_objective_bound
    elif results.termination_condition == appsi.base.TerminationCondition.optimal:
        new_bnd = results.best_feasible_objective  # assumes the problem is convex
    else:
        new_bnd = None
        msg = f'Warning: Bounds tightening for lb for var {str(v)} was unsuccessful. Termination condition: {results.termination_condition}; The lb was not changed.'
        logger.warning(msg)

    if lb_or_ub == 'lb':
        orig_lb = pyo.value(v.lb)
        if new_bnd is None:
            new_bnd = orig_lb
        elif v.has_lb():
            if new_bnd < orig_lb:
                new_bnd = orig_lb
    else:
        orig_ub = pyo.value(v.ub)
        if new_bnd is None:
            new_bnd = orig_ub
        elif v.has_ub():
            if new_bnd > orig_ub:
                new_bnd = orig_ub

    if new_bnd is None:
        # Need nan instead of None for MPI communication; This is appropriately handled in perform_obbt().
        new_bnd = np.nan

    # remove the objective function
    del model.__obj_bounds_tightening

    return new_bnd


def _tighten_bnds(model, solver, vardatalist, lb_or_ub, obbt_info, with_progress_bar=False, time_limit=math.inf, progress_bar_string=None):
    """
    Tighten the lower bounds of all variables in vardatalist (or self.vars_to_tighten if vardatalist is None).

    Parameters
    ----------
    model: pyo.ConcreteModel or pyo.Block
    solver: pyomo solver object
    vardatalist: list of _GeneralVarData
    lb_or_ub: str
        'lb' or 'ub'
    time_limit: float

    Returns
    -------
    new_bounds: list of float
    """
    # solve for the new bounds
    t0 = time.time()
    new_bounds = list()

    obbt_info.total_num_problems += len(vardatalist)

    if with_progress_bar:
        if progress_bar_string is None:
            if lb_or_ub == 'lb':
                bnd_str = 'LBs'
            else:
                bnd_str = 'UBs'
            bnd_str = 'OBBT ' + bnd_str
        else:
            bnd_str = progress_bar_string
        if mpi_available:
            tqdm_position = mpiu.MPI.COMM_WORLD.Get_rank()
        else:
            tqdm_position = 0
        for v in tqdm(vardatalist, ncols=100, desc=bnd_str, leave=False, position=tqdm_position):
            if time.time() - t0 > time_limit:
                if lb_or_ub == 'lb':
                    if v.lb is None:
                        new_bounds.append(np.nan)
                    else:
                        new_bounds.append(pyo.value(v.lb))
                else:
                    if v.ub is None:
                        new_bounds.append(np.nan)
                    else:
                        new_bounds.append(pyo.value(v.ub))
            else:
                new_bnd = _single_solve(v=v, model=model, solver=solver,
                                        lb_or_ub=lb_or_ub,
                                        obbt_info=obbt_info)
                new_bounds.append(new_bnd)
    else:
        for v in vardatalist:
            if time.time() - t0 > time_limit:
                if lb_or_ub == 'lb':
                    if v.lb is None:
                        new_bounds.append(np.nan)
                    else:
                        new_bounds.append(pyo.value(v.lb))
                else:
                    if v.ub is None:
                        new_bounds.append(np.nan)
                    else:
                        new_bounds.append(pyo.value(v.ub))
            else:
                new_bnd = _single_solve(v=v, model=model, solver=solver,
                                        lb_or_ub=lb_or_ub,
                                        obbt_info=obbt_info)
                new_bounds.append(new_bnd)

    return new_bounds


def _bt_prep(model, solver, objective_bound=None):
    """
    Prepare the model for bounds tightening.
    Gather the variable values to load back in after bounds tightening.
    Deactivate any active objectives.
    If objective_ub is not None, then add a constraint forcing the objective to be less than objective_ub

    Parameters
    ----------
    model : pyo.ConcreteModel or pyo.Block
        The model object that will be used for bounds tightening.
    objective_bound : float
        The objective value for the current best upper bound incumbent

    Returns
    -------
    initial_var_values: ComponentMap
    deactivated_objectives: list
    orig_update_config: appsi.base.UpdateConfig
    orig_config: appsi.base.SolverConfig
    """

    if solver.is_persistent():
        orig_update_config = solver.update_config()
        solver.update_config.check_for_new_or_removed_constraints = False
        solver.update_config.check_for_new_or_removed_vars = False
        solver.update_config.check_for_new_or_removed_params = False
        solver.update_config.check_for_new_objective = False
        solver.update_config.update_constraints = False
        solver.update_config.update_vars = False
        solver.update_config.update_params = False
        solver.update_config.update_named_expressions = False
        solver.update_config.update_objective = False
        solver.update_config.treat_fixed_vars_as_params = True
    else:
        orig_update_config = None

    orig_config = solver.config()
    solver.config.stream_solver = False
    solver.config.load_solution = False

    if solver.is_persistent():
        solver.set_instance(model)

    initial_var_values = ComponentMap()
    for v in model.component_data_objects(ctype=pyo.Var, active=None, sort=True, descend_into=True):
        initial_var_values[v] = v.value

    deactivated_objectives = list()
    for obj in model.component_data_objects(pyo.Objective, active=True, sort=True, descend_into=True):
        deactivated_objectives.append(obj)
        obj.deactivate()

    # add inequality bound on objective functions if required
    # obj.expr <= objective_ub
    if objective_bound is not None and math.isfinite(objective_bound):
        if len(deactivated_objectives) != 1:
            e = 'BoundsTightener: When providing objective_ub,' + \
                ' the model must have one and only one objective function.'
            logger.error(e)
            raise ValueError(e)
        original_obj = deactivated_objectives[0]
        if original_obj.sense == minimize:
            model.__objective_ineq = \
                pyo.Constraint(expr=original_obj.expr <= objective_bound)
        else:
            assert original_obj.sense == maximize
            model.__objective_ineq = pyo.Constraint(expr=original_obj.expr >= objective_bound)
        if solver.is_persistent():
            solver.add_constraints([model.__objective_ineq])

    return initial_var_values, deactivated_objectives, orig_update_config, orig_config


def _build_vardatalist(model, varlist=None, warning_threshold=0):
    """
    Convert a list of pyomo variables to a list of SimpleVar and _GeneralVarData. If varlist is none, builds a
    list of all variables in the model. The new list is stored in the vars_to_tighten attribute.

    Parameters
    ----------
    model: ConcreteModel
    varlist: None or list of pyo.Var
    warning_threshold: float
        The threshold below which a warning is raised when attempting to perform OBBT on variables whose
        ub - lb < warning_threshold.
    """
    vardatalist = None

    # if the varlist is None, then assume we want all the active variables
    if varlist is None:
        raise NotImplementedError('Still need to do this.')
    elif isinstance(varlist, pyo.Var):
        # user provided a variable, not a list of variables. Let's work with it anyway
        varlist = [varlist]

    if vardatalist is None:
        # expand any indexed components in the list to their
        # component data objects
        vardatalist = list()
        for v in varlist:
            if v.is_indexed():
                vardatalist.extend(v.values())
            else:
                vardatalist.append(v)

    # remove from vardatalist if the variable is fixed (maybe there is a better way to do this)
    corrected_vardatalist = []
    for v in vardatalist:
        if not v.is_fixed():
            if v.has_lb() and v.has_ub():
                if v.ub - v.lb < warning_threshold:
                    e = 'Warning: Tightening a variable with ub - lb is less than {threshold}: {v}, lb: {lb}, ub: {ub}'.format(threshold=warning_threshold, v=v, lb=v.lb, ub=v.ub)
                    logger.warning(e)
                    warnings.warn(e)
            corrected_vardatalist.append(v)

    return corrected_vardatalist


def perform_obbt(model, solver, varlist=None, objective_bound=None, update_bounds=True, with_progress_bar=False,
                 direction='both', time_limit=math.inf, parallel=True, collect_obbt_info=False,
                 warning_threshold=0, progress_bar_string=None):
    """
    Perform optimization-based bounds tighening on the variables in varlist subject to the constraints in model.

    Parameters
    ----------
    model: pyo.ConcreteModel or pyo.Block
        The model to be used for bounds tightening
    solver: appsi.base.PersistentSolver
        The solver to be used for bounds tightening.
    varlist: list of pyo.Var
        The variables for which OBBT should be performed. If varlist is None, then we attempt to automatically
        detect which variables need tightened.
    objective_bound: float
        A lower or upper bound on the objective. If this is not None, then a constraint will be added to the
        bounds tightening problems constraining the objective to be less than/greater than objective_bound.
    update_bounds: bool
        If True, then the variable bounds will be updated
    with_progress_bar: bool
    direction: str
        Options are 'both', 'lbs', or 'ubs'
    time_limit: float
        The maximum amount of time to be spent performing OBBT
    parallel: bool
        If True, then OBBT will automatically be performed in parallel if mpirun or mpiexec was used;
        If False, then OBBT will not run in parallel even if mpirun or mpiexec was used;
    warning_threshold: float
        The threshold below which a warning is issued when attempting to perform OBBT on variables whose
        ub - lb < warning_threshold.

    Returns
    -------
    lower_bounds: list of float
    upper_bounds: list of float
    obbt_info: OBBTInfo

    """
    if not isinstance(solver, appsi.base.Solver):
        raise ValueError('Coramin requires an Appsi solver interface')

    obbt_info = OBBTInfo()
    obbt_info.total_num_problems = 0
    obbt_info.num_problems_attempted = 0
    obbt_info.num_successful_problems = 0

    t0 = time.time()
    initial_var_values, deactivated_objectives, orig_update_config, orig_config = _bt_prep(model=model, solver=solver, objective_bound=objective_bound)

    vardata_list = _build_vardatalist(model=model, varlist=varlist, warning_threshold=warning_threshold)
    if mpi_available and parallel:
        mpi_interface = mpiu.MPIInterface()
        alloc_map = mpiu.MPIAllocationMap(mpi_interface, len(vardata_list))
        local_vardata_list = alloc_map.local_list(vardata_list)
    else:
        local_vardata_list = vardata_list

    exc = None
    try:
        if direction in {'both', 'lbs'}:
            local_lower_bounds = _tighten_bnds(model=model, solver=solver,
                                               vardatalist=local_vardata_list,
                                               lb_or_ub='lb',
                                               obbt_info=obbt_info,
                                               with_progress_bar=with_progress_bar,
                                               time_limit=(time_limit - (time.time() - t0)),
                                               progress_bar_string=progress_bar_string)
        else:
            local_lower_bounds = list()
            for v in local_vardata_list:
                if v.lb is None:
                    local_lower_bounds.append(np.nan)
                else:
                    local_lower_bounds.append(pyo.value(v.lb))
        if direction in {'both', 'ubs'}:
            local_upper_bounds = _tighten_bnds(model=model, solver=solver,
                                               vardatalist=local_vardata_list,
                                               lb_or_ub='ub',
                                               obbt_info=obbt_info,
                                               with_progress_bar=with_progress_bar,
                                               time_limit=(time_limit - (time.time() - t0)),
                                               progress_bar_string=progress_bar_string)
        else:
            local_upper_bounds = list()
            for v in local_vardata_list:
                if v.ub is None:
                    local_upper_bounds.append(np.nan)
                else:
                    local_upper_bounds.append(pyo.value(v.ub))
        status = 1
        msg = None
    except Exception as err:
        exc = err
        tb = traceback.format_exc()
        status = 0
        msg = str(tb)

    if mpi_available and parallel:
        local_status = np.array([status], dtype='i')
        global_status = np.array([0 for i in range(mpiu.MPI.COMM_WORLD.Get_size())], dtype='i')
        mpiu.MPI.COMM_WORLD.Allgatherv([local_status, mpiu.MPI.INT], [global_status, mpiu.MPI.INT])
        if not np.all(global_status):
            messages = mpi_interface.comm.allgather(msg)
            msg = None
            for m in messages:
                if m is not None:
                    msg = m
            logger.error('An error was raised in one or more processes:\n' + msg)
            raise mpiu.MPISyncError('An error was raised in one or more processes:\n' + msg)
    else:
        if status != 1:
            logger.error('An error was raised during OBBT:\n' + msg)
            raise exc

    if mpi_available and parallel:
        global_lower = alloc_map.global_list_float64(local_lower_bounds)
        global_upper = alloc_map.global_list_float64(local_upper_bounds)
        obbt_info.total_num_problems = mpiu.MPI.COMM_WORLD.allreduce(obbt_info.total_num_problems)
        obbt_info.num_problems_attempted = mpiu.MPI.COMM_WORLD.allreduce(obbt_info.num_problems_attempted)
        obbt_info.num_successful_problems = mpiu.MPI.COMM_WORLD.allreduce(obbt_info.num_successful_problems)
    else:
        global_lower = local_lower_bounds
        global_upper = local_upper_bounds

    tmp = list()
    for i in global_lower:
        if np.isnan(i):
            tmp.append(None)
        else:
            tmp.append(float(i))
    global_lower = tmp

    tmp = list()
    for i in global_upper:
        if np.isnan(i):
            tmp.append(None)
        else:
            tmp.append(float(i))
    global_upper = tmp

    _lower_bounds = None
    _upper_bounds = None
    if update_bounds:
        _lower_bounds = global_lower
        _upper_bounds = global_upper
    _bt_cleanup(
        model=model, solver=solver, vardatalist=vardata_list,
        initial_var_values=initial_var_values,
        deactivated_objectives=deactivated_objectives,
        orig_update_config=orig_update_config, orig_config=orig_config,
        lower_bounds=_lower_bounds, upper_bounds=_upper_bounds
    )

    if collect_obbt_info:
        return global_lower, global_upper, obbt_info
    else:
        return global_lower, global_upper
