#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import numpy as np
import pyomo.environ as pyo


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


class CsplineParameters:
    def __init__(self, model=None, fptr=None):
        """Cubic spline parameters class.  This can be used to read and
        write parameters or calculate cubic spline function values and
        derivatives for testing.
        """
        if model is not None and fptr is not None:
            raise ValueError("Please specify at most one of model or fptr.")
        if model is not None:
            self.get_parameters_from_model(model)
        elif fptr is not None:
            self.get_parameters_from_file(fptr)
        else:
            self.knots = np.array([])
            self.a1 = np.array([])
            self.a2 = np.array([])
            self.a3 = np.array([])
            self.a4 = np.array([])

    @property
    def n_knots(self):
        """Number of knots"""
        return len(self.knots)

    @property
    def n_segments(self):
        """Number of segments"""
        return len(self.knots) - 1

    @property
    def valid(self):
        """Ensure that the number of knots and cubic parameters is valid"""
        return (
            len(self.a1) == self.n_segments
            and len(self.a2) == self.n_segments
            and len(self.a3) == self.n_segments
            and len(self.a4) == self.n_segments
        )

    def get_parameters_from_model(self, m):
        """Read parameters from a Pyomo model used to calculate them"""
        self.knots = [pyo.value(x) for x in m.x.values()]
        self.a1 = [None] * len(m.seg_idx)
        self.a2 = [None] * len(m.seg_idx)
        self.a3 = [None] * len(m.seg_idx)
        self.a4 = [None] * len(m.seg_idx)
        for s in m.seg_idx:
            self.a1[s - 1] = pyo.value(m.alpha[s, 1])
            self.a2[s - 1] = pyo.value(m.alpha[s, 2])
            self.a3[s - 1] = pyo.value(m.alpha[s, 3])
            self.a4[s - 1] = pyo.value(m.alpha[s, 4])
        self.knots = np.array(self.knots)
        self.a1 = np.array(self.a1)
        self.a2 = np.array(self.a2)
        self.a3 = np.array(self.a3)
        self.a4 = np.array(self.a4)

    def get_parameters_from_file(self, fptr):
        """Read parameters from a file"""
        # line 1: number of segments
        ns = int(fptr.readline())
        # Make param lists
        self.knots = [None] * (ns + 1)
        self.a1 = [None] * ns
        self.a2 = [None] * ns
        self.a3 = [None] * ns
        self.a4 = [None] * ns
        # Read params
        for i in range(ns + 1):
            self.knots[i] = float(fptr.readline())
        for a in [self.a1, self.a2, self.a3, self.a4]:
            for i in range(ns):
                a[i] = float(fptr.readline())
        self.knots = np.array(self.knots)
        self.a1 = np.array(self.a1)
        self.a2 = np.array(self.a2)
        self.a3 = np.array(self.a3)
        self.a4 = np.array(self.a4)

    def write_parameters(self, fptr):
        """Write parameters to a file"""
        assert self.valid
        fptr.write(f"{self.n_segments}\n")
        for l in [self.knots, self.a1, self.a2, self.a3, self.a4]:
            for x in l:
                fptr.write(f"{x}\n")

    def segment(self, x):
        """Get the spline segment containing x.

        Args:
            x: location, float or numpy array

        Returns:
            segment(s) containing x, if x is a numpy array a numpy
            array of integers is returned otherwise return an integer
        """
        s = np.searchsorted(self.knots, x)
        # If x is past the last knot, use the last segment
        # this could happen just due to round-off even if
        # you don't intend to extrapolate
        s[s >= self.n_segments] = self.n_segments - 1
        return s

    def f(self, x):
        """Get f(x)

        Args:
            x: location, numpy array float

        Returns:
            f(x) numpy array if x is numpy array or float
        """
        s = self.segment(x)
        return self.a1[s] + self.a2[s] * x + self.a3[s] * x**2 + self.a4[s] * x**3


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
    instead.  The purpose of the alternative least squares form is to allow the spline
    to be constrained in other ways that don't require a perfect data match. The knots
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
        Pyomo ConcreteModel
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
        return _fx_cubic(m.x[s + 1], m.alpha, s) == _fx_cubic(
            m.x[s + 1], m.alpha, s + 1
        )

    # f"_s(x) = f"_s+1(x)
    @m.Constraint(m.seg_idx)
    def yxx_eqn(blk, s):
        if s == m.seg_idx.last():
            return pyo.Constraint.Skip
        return _fxx_cubic(m.x[s + 1], m.alpha, s) == _fxx_cubic(
            m.x[s + 1], m.alpha, s + 1
        )

    # Identify segments used to predict y_data at each x_data.  We use search in
    # instead of a dict lookup, since we don't want to require the data to be at
    # the knots, even though that is almost always the case.
    idx = np.searchsorted(x_knots, x_data)

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
