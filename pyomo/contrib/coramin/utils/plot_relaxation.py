import numpy as np
try:
    import plotly.graph_objects as go
except ImportError:
    pass
import pyomo.environ as pe
from .pyomo_utils import get_objective
from .coramin_enums import RelaxationSide
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
try:
    import tqdm
except ImportError:
    tqdm = None


def _solve(m, using_persistent_solver, solver, rhs_vars, aux_var, obj):
    obj.activate()
    if using_persistent_solver:
        for v in rhs_vars:
            solver.update_var(v)
        solver.set_objective(obj)
        res = solver.solve(load_solutions=False, save_results=False)
    else:
        res = solver.solve(m, load_solutions=False)
    if res.solver.termination_condition != pe.TerminationCondition.optimal:
        raise RuntimeError('Could not produce plot because solver did not terminate optimally')
    if using_persistent_solver:
        solver.load_vars([aux_var])
    else:
        m.solutions.load_from(res)
    obj.deactivate()


def _solve_loop(m, x, w, x_list, using_persistent_solver, solver):
    w_list = list()
    for _xval in x_list:
        x.fix(_xval)
        if using_persistent_solver:
            solver.update_var(x)
            res = solver.solve(load_solutions=False, save_results=False)
        else:
            res = solver.solve(m, load_solutions=False)
        if res.solver.termination_condition != pe.TerminationCondition.optimal:
            raise RuntimeError(
                'Could not produce plot because solver did not terminate optimally. Termination condition: ' + str(
                    res.solver.termination_condition))
        if using_persistent_solver:
            solver.load_vars([w])
        else:
            m.solutions.load_from(res)
        w_list.append(w.value)
    return w_list


def _plot_2d(m, relaxation, solver, show_plot, num_pts):
    using_persistent_solver = isinstance(solver, PersistentSolver)

    x = relaxation.get_rhs_vars()[0]
    w = relaxation.get_aux_var()

    if not x.has_lb() or not x.has_ub():
        raise ValueError('rhs var must have bounds')

    orig_xval = x.value
    orig_wval = w.value
    xlb = pe.value(x.lb)
    xub = pe.value(x.ub)

    orig_obj = get_objective(m)
    if orig_obj is not None:
        orig_obj.deactivate()

    x_list = np.linspace(xlb, xub, num_pts)
    x_list = [float(i) for i in x_list]
    w_true = list()

    rhs_expr = relaxation.get_rhs_expr()
    for _x in x_list:
        x.value = float(_x)
        w_true.append(pe.value(rhs_expr))
    plotly_data = [go.Scatter(x=x_list, y=w_true, name=str(rhs_expr))]

    m._plotting_objective = pe.Objective(expr=w)
    if using_persistent_solver:
        solver.set_instance(m)

    if relaxation.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
        w_min = _solve_loop(m, x, w, x_list, using_persistent_solver, solver)
        plotly_data.append(go.Scatter(x=x_list, y=w_min, name='underestimator'))

    del m._plotting_objective
    m._plotting_objective = pe.Objective(expr=w, sense=pe.maximize)
    if using_persistent_solver:
        solver.set_objective(m._plotting_objective)

    if relaxation.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
        w_max = _solve_loop(m, x, w, x_list, using_persistent_solver, solver)
        plotly_data.append(go.Scatter(x=x_list, y=w_max, name='overestimator'))

    fig = go.Figure(data=plotly_data)
    if show_plot:
        fig.show()

    x.unfix()
    x.value = orig_xval
    w.value = orig_wval
    del m._plotting_objective
    if orig_obj is not None:
        orig_obj.activate()


def _plot_3d(m, relaxation, solver, show_plot, num_pts):
    using_persistent_solver = isinstance(solver, PersistentSolver)

    rhs_vars = relaxation.get_rhs_vars()
    x, y = rhs_vars
    w = relaxation.get_aux_var()

    if not x.has_lb() or not x.has_ub() or not y.has_lb() or not y.has_ub():
        raise ValueError('rhs vars must have bounds')

    orig_xval = x.value
    orig_yval = y.value
    orig_wval = w.value

    orig_obj = get_objective(m)
    if orig_obj is not None:
        orig_obj.deactivate()

    m._underestimator_obj = pe.Objective(expr=w)
    m._overestimator_obj = pe.Objective(expr=w, sense=pe.maximize)
    m._underestimator_obj.deactivate()
    m._overestimator_obj.deactivate()
    if using_persistent_solver:
        solver.set_instance(m)

    x_list = np.linspace(x.lb, x.ub, num_pts)
    y_list = np.linspace(y.lb, y.ub, num_pts)
    x_list = [float(i) for i in x_list]
    y_list = [float(i) for i in y_list]
    w_true = np.empty((num_pts, num_pts), dtype=np.double)
    w_min = np.empty((num_pts, num_pts), dtype=np.double)
    w_max = np.empty((num_pts, num_pts), dtype=np.double)

    rhs_expr = relaxation.get_rhs_expr()

    def sub_loop(x_ndx, _x):
        x.fix(_x)
        for y_ndx, _y in enumerate(y_list):
            y.fix(_y)
            w_true[x_ndx, y_ndx] = pe.value(rhs_expr)
            if relaxation.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
                _solve(m, using_persistent_solver, solver, rhs_vars, w, m._underestimator_obj)
                w_min[x_ndx, y_ndx] = w.value
            if relaxation.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
                _solve(m, using_persistent_solver, solver, rhs_vars, w, m._overestimator_obj)
                w_max[x_ndx, y_ndx] = w.value

    if tqdm is not None:
        for x_ndx, _x in tqdm.tqdm(list(enumerate(x_list))):
            sub_loop(x_ndx, _x)
    else:
        for x_ndx, _x in enumerate(x_list):
            sub_loop(x_ndx, _x)

    plotly_data = list()
    plotly_data.append(go.Surface(x=x_list, y=y_list, z=w_true, name=str(rhs_expr)))
    if relaxation.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
        plotly_data.append(go.Surface(x=x_list, y=y_list, z=w_min, name='underestimator'))
    if relaxation.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
        plotly_data.append(go.Surface(x=x_list, y=y_list, z=w_max, name='overestimator'))

    fig = go.Figure(data=plotly_data)
    if show_plot:
        fig.show()

    x.unfix()
    y.unfix()
    x.value = orig_xval
    y.value = orig_yval
    w.value = orig_wval
    del m._underestimator_obj
    del m._overestimator_obj
    if orig_obj is not None:
        orig_obj.activate()


def plot_relaxation(m, relaxation, solver, show_plot=True, num_pts=100):
    rhs_vars = relaxation.get_rhs_vars()

    if len(rhs_vars) == 1:
        _plot_2d(m, relaxation, solver, show_plot, num_pts)
    elif len(rhs_vars) == 2:
        _plot_3d(m, relaxation, solver, show_plot, num_pts)
    else:
        raise NotImplementedError('Cannot generate plot for relaxation with more than 2 RHS vars')

