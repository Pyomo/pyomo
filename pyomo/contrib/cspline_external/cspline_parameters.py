#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import
import pyomo.environ as pyo

np, numpy_available = attempt_import("numpy")

if numpy_available:
    searchsorted = np.searchsorted
else:
    def searchsorted(x, y):
        raise NotImplementedError("cubic_parameters_model() currently relies on Numpy")

def _f_cubic(x, alpha, s=None):
    """
    Cubic function:
        y = a1 + a2 * x + a3 * x^2 + a4 * x^3

    Optionally if s is provided it is a segment index.
        y = a1[s] + a2[s] * x + a3[s] * x^2 + a4[s] * x^3

    This is used to write constraints more compactly.

    Args:
        x: x variable, numeric, numpy array, or Pyomo component
        alpha: cubic parameters, numeric or Pyomo component
        s: optional segment index

    Returns:
        Pyomo expression, numpy array, or float
    """
    if s is None:
        return alpha[1] + alpha[2] * x + alpha[3] * x**2 + alpha[4] * x**3
    return alpha[s, 1] + alpha[s, 2] * x + alpha[s, 3] * x**2 + alpha[s, 4] * x**3


def _fx_cubic(x, alpha, s=None):
    """
    Cubic function first derivative:
        dy/dx = a2 + 2 * a3 * x + 3 * a4 * x^2

    Optionally if s is provided it is a segment index.
        dy/dx = a2[s] + 2 * a3[s] * x + 3 * a4[s] * x^2

    This is used to write constraints more compactly.

    Args:
        x: x variable, numeric, numpy array, or Pyomo component
        alpha: cubic parameters, numeric or Pyomo component
        s: optional segment index

    Returns:
        Pyomo expression, numpy array, or float
    """
    if s is None:
        return alpha[2] + 2 * alpha[3] * x + 3 * alpha[4] * x**2
    return alpha[s, 2] + 2 * alpha[s, 3] * x + 3 * alpha[s, 4] * x**2


def _fxx_cubic(x, alpha, s=None):
    """
    Cubic function second derivative:
        d2y/dx2 = 2 * a3 + 6 * a4 * x

    Optionally if s is provided it is a segment index.
        d2y/dx2 = 2 * a3[s] + 6 * a4[s] * x

    This is used to write constraints more compactly.

    Args:
        x: x variable, numeric, numpy array, or Pyomo component
        alpha: cubic parameters, numeric or Pyomo component
        s: optional segment index

    Returns:
        Pyomo expression, numpy array, or float
    """
    if s is None:
        return 2 * alpha[3] + 6 * alpha[4] * x
    return 2 * alpha[s, 3] + 6 * alpha[s, 4] * x


def cubic_parameters_model(
    x_data,
    y_data,
    x_knots=None,
    end_point_constraint=True,
    objective_form=False,
    name="cubic spline parameters model",
):
    """Create a Pyomo model to calculate parameters for a cubic spline.  By default
    this creates a square linear model, but optionally it can leave off the endpoint
    second derivative constraints and add an objective function for fitting data
    instead.  The purpose of alternative least squares form is to allow the spline to
    be constrained in other wats that don't require a perfect data match. The knots 
    don't need to be the same as the x data to allow, for example, additional segments
    for extrapolation. This is not the most computationally efficient way to calculate
    parameters, but since it is used to precalculate parameters, speed is not important.  

    Args:
        x_data: sorted list of x data
        y_data: list of y data
        x_knots: optional sorted list of knots (default is to use x_data)
        end_point_constraint: if True add constraint that second derivative = 0 at
            endpoints (default=True)
        objective_form: if True write a least squares objective rather than constraints
            to match data (default=False)
        name: optional model name

    Returns:
        Poymo ConcreteModel
    """
    n_data = len(x_data)
    assert n_data == len(y_data)
    if x_knots is None:
        n_knots = n_data
        x_knots = x_data
    else:
        n_knots = len(x_knots)

    m = pyo.ConcreteModel(name=name)
    # Sets of indexes for knots, segments, and data
    m.knt_idx = pyo.RangeSet(n_knots)
    m.seg_idx = pyo.RangeSet(n_knots - 1)
    m.dat_idx = pyo.RangeSet(n_data)

    m.x_data = pyo.Param(m.dat_idx, initialize={i + 1: x for i, x in enumerate(x_data)})
    m.y_data = pyo.Param(m.dat_idx, initialize={i + 1: x for i, x in enumerate(y_data)})
    m.x = pyo.Param(m.knt_idx, initialize={i + 1: x for i, x in enumerate(x_knots)})
    m.alpha = pyo.Var(m.seg_idx, {1, 2, 3, 4}, initialize=1)

    # f_s(x) = f_s+1(x)
    @m.Constraint(m.seg_idx)
    def y_eqn(blk, s):
        if s == m.seg_idx.last():
            return pyo.Constraint.Skip
        return _f_cubic(m.x[s + 1], m.alpha, s) == _f_cubic(m.x[s + 1], m.alpha, s + 1)

    # f'_s(x) = f'_s+1(x)
    @m.Constraint(m.seg_idx)
    def yx_eqn(blk, s):
        if s == m.seg_idx.last():
            return pyo.Constraint.Skip
        return _fx_cubic(m.x[s + 1], m.alpha, s) == _fx_cubic(m.x[s + 1], m.alpha, s + 1)

    # f"_s(x) = f"_s+1(x)
    @m.Constraint(m.seg_idx)
    def yxx_eqn(blk, s):
        if s == m.seg_idx.last():
            return pyo.Constraint.Skip
        return _fxx_cubic(m.x[s + 1], m.alpha, s) == _fxx_cubic(m.x[s + 1], m.alpha, s + 1)

    # Identify segments used to predict y_data at each x_data.  We use search in
    # instead of a dict lookup, since we don't want to require the data to be at
    # the knots, even though that is almost always the case.
    idx = searchsorted(x_knots, x_data)

    if end_point_constraint:
        add_endpoint_second_derivative_constraints(m)

    # Expression for difference between data and prediction
    @m.Expression(m.dat_idx)
    def ydiff(blk, d):
        s = idx[d - 1] + 1
        if s >= m.seg_idx.last():
            s -= 1
        return m.y_data[d] - _f_cubic(m.x_data[d], m.alpha, s)

    if objective_form:
        # least squares objective
        m.obj = pyo.Objective(expr=sum(m.ydiff[d] ** 2 for d in m.dat_idx))
    else:

        @m.Constraint(m.dat_idx)
        def match_data(blk, d):
            return m.ydiff[d] == 0

    return m


def add_endpoint_second_derivative_constraints(m):
    """Usually cubic splines use the endpoint constraints that the second
    derivative is zero.  This function adds those constraints to a model
    """

    @m.Constraint([m.seg_idx.first(), m.seg_idx.last()])
    def yxx_endpoint_eqn(blk, s):
        if s == m.seg_idx.last():
            j = m.knt_idx.last()
        else:
            j = s
        return _fxx_cubic(m.x[j], m.alpha, s) == 0


def get_parameters(m, file_name=None):
    """Once the model has been solved, this function can be used to extract
    the cubic spline parameters.
    """
    knots = [pyo.value(x) for x in m.x.values()]
    a1 = [None] * len(m.seg_idx)
    a2 = [None] * len(m.seg_idx)
    a3 = [None] * len(m.seg_idx)
    a4 = [None] * len(m.seg_idx)
    for s in m.seg_idx:
        a1[s - 1] = pyo.value(m.alpha[s, 1])
        a2[s - 1] = pyo.value(m.alpha[s, 2])
        a3[s - 1] = pyo.value(m.alpha[s, 3])
        a4[s - 1] = pyo.value(m.alpha[s, 4])

    if file_name is not None:
        with open(file_name, "w") as fptr:
            fptr.write(f"{len(a1)}\n")
            for l in [knots, a1, a2, a3, a4]:
                for x in l:
                    fptr.write(f"{x}\n")

    return knots, a1, a2, a3, a4


def _extract_params(m):
    """Extract alpha as a plain dict of floats to play nice with vectorized functions"""
    alpha = {}
    for s in m.seg_idx:
        alpha[s] = {}
        alpha[s][1] = pyo.value(m.alpha[s, 1])
        alpha[s][2] = pyo.value(m.alpha[s, 2])
        alpha[s][3] = pyo.value(m.alpha[s, 3])
        alpha[s][4] = pyo.value(m.alpha[s, 4])
    return alpha
