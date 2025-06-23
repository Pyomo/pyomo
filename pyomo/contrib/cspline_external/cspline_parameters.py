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
from pyomo.core.base.block import BlockData, declare_custom_block


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


def _yxx_endpoint_is_zero_eqn(blk, s):
    """Rule which is used to add a constraint to set the second derivative
    at the first and last knot to zero.
    """
    if s == blk.seg_idx.last():
        j = blk.knt_idx.last()
    else:
        j = s
    return _fxx_cubic(blk.x[j], blk.alpha, s) == 0


class _increasing_constraint:
    def __init__(self, tol=0):
        """Rule functor for concave constraints on cubics."""
        self.tol = tol

    def __call__(self, blk, k):
        if k >= len(blk.knt_idx):
            s = k - 1
        else:
            s = k
        return _fx_cubic(blk.x[k], blk.alpha, s) >= self.tol


class _decreasing_constraint:
    def __init__(self, tol=0):
        """Rule functor for concave constraints on cubics."""
        self.tol = tol

    def __call__(self, blk, k):
        if k >= len(blk.knt_idx):
            s = k - 1
        else:
            s = k
        return _fx_cubic(blk.x[k], blk.alpha, s) <= -self.tol


class _convex_constraint:
    def __init__(self, tol=0):
        """Rule functor for concave constraints on cubics."""
        self.tol = tol

    def __call__(self, blk, k):
        if k >= len(blk.knt_idx):
            s = k - 1
        else:
            s = k
        return _fxx_cubic(blk.x[k], blk.alpha, s) >= self.tol


class _concave_constraint:
    def __init__(self, tol=0):
        """Rule functor for concave constraints on cubics."""
        self.tol = tol

    def __call__(self, blk, k):
        if k >= len(blk.knt_idx):
            s = k - 1
        else:
            s = k
        return _fxx_cubic(blk.x[k], blk.alpha, s) <= -self.tol


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

    def add_linear_extrapolation_segments(self):
        """Add a segment on the front and back of the cspline so that
        any extrapolation will be linear."""
        # We need to add a knot for a linear segment on the beginning and
        # end.  Since the first and last segment will be used for extrapolation,
        # and we want them to be linear, it doesn't really matter how far out
        # the knots are. To try to be roughly in line with the data scale we
        # just use the distance from the first to the last knot.
        dist = self.knots[-1] - self.knots[0]
        x = np.array([self.knots[0], self.knots[-1]])
        y = self.f(x)
        m = self.dfdx(x)
        b = y - m * x
        k = np.array([self.knots[0] - dist, self.knots[-1] + dist])

        self.knots = np.insert(self.knots, 0, k[0])
        self.a1 = np.insert(self.a1, 0, b[0])
        self.a2 = np.insert(self.a2, 0, m[0])
        self.a3 = np.insert(self.a3, 0, 0)
        self.a4 = np.insert(self.a4, 0, 0)
        self.knots = np.append(self.knots, k[1])
        self.a1 = np.append(self.a1, b[1])
        self.a2 = np.append(self.a2, m[1])
        self.a3 = np.append(self.a3, 0)
        self.a4 = np.append(self.a4, 0)

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
        if isinstance(s, np.ndarray):
            # if x is before the first knot use the first segment
            s[s <= 0] = 1
            # if x is after the last knot use last segment to extrapolate
            s[s >= len(self.knots)] = len(self.knots) - 1
        else:
            if s <= 0:
                # if x is before the first knot use the first segment
                return 0
            if s >= len(self.knots):
                # if x is after the last knot use last segment to extrapolate
                return len(self.knots) - 2
        return s - 1

    def f(self, x):
        """Get f(x)

        Args:
            x: location, numpy array float

        Returns:
            f(x) numpy array if x is numpy array or float
        """
        s = self.segment(x)
        return self.a1[s] + self.a2[s] * x + self.a3[s] * x**2 + self.a4[s] * x**3

    def dfdx(self, x):
        """Get d/dx(f(x))

        Args:
            x: location, numpy array float

        Returns:
            df/dx numpy array if x is numpy array or float
        """
        s = self.segment(x)
        return self.a2[s] + 2 * self.a3[s] * x + 3 * self.a4[s] * x**2


@declare_custom_block(name="CubicParametersModel")
class CubicParametersModelData(BlockData):
    def __init__(self, *args, concrete=True, **kwargs):
        BlockData.__init__(self, *args, **kwargs)

    def add_model(
        self,
        x_data,
        y_data,
        x_knots=None,
        end_point_constraint=True,
        objective_form=False,
    ):
        """Add parameter model to the block.  By default this creates a square
        linear model, but optionally it can leave off the endpoint second
        derivative constraints and add an objective function for fitting data
        instead.  The purpose of the alternative least squares form is to allow
        the spline to be constrained in other ways that don't require a perfect
        data match. The knots don't need to be the same as the x data to allow,
        for example, additional segments for extrapolation. This is not the most
        computationally efficient way to calculate parameters, but since it is
        used to precalculate parameters, speed is not important.

        Args:
            x_data: sorted list of x data
            y_data: list of y data
            x_knots: optional sorted list of knots (default is to use x_data)
            end_point_constraint: if True add constraint that second derivative
                = 0 at endpoints (default=True)
            objective_form: if True write a least squares objective rather than
                constraints to match data (default=False)
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

        self.knt_idx = pyo.RangeSet(n_knots)
        self.seg_idx = pyo.RangeSet(n_knots - 1)
        self.dat_idx = pyo.RangeSet(n_data)

        self.x_data = pyo.Param(
            self.dat_idx, initialize={i + 1: x for i, x in enumerate(x_data)}
        )
        self.y_data = pyo.Param(
            self.dat_idx, initialize={i + 1: x for i, x in enumerate(y_data)}
        )
        self.x = pyo.Param(
            self.knt_idx, initialize={i + 1: x for i, x in enumerate(x_knots)}
        )
        self.alpha = pyo.Var(self.seg_idx, {1, 2, 3, 4}, initialize=1)

        # f_s(x) = f_s+1(x)
        @self.Constraint(self.seg_idx)
        def y_eqn(blk, s):
            if s == self.seg_idx.last():
                return pyo.Constraint.Skip
            return _f_cubic(self.x[s + 1], self.alpha, s) == _f_cubic(
                self.x[s + 1], self.alpha, s + 1
            )

        # f'_s(x) = f'_s+1(x)
        @self.Constraint(self.seg_idx)
        def yx_eqn(blk, s):
            if s == self.seg_idx.last():
                return pyo.Constraint.Skip
            return _fx_cubic(self.x[s + 1], self.alpha, s) == _fx_cubic(
                self.x[s + 1], self.alpha, s + 1
            )

        # f"_s(x) = f"_s+1(x)
        @self.Constraint(self.seg_idx)
        def yxx_eqn(blk, s):
            if s == self.seg_idx.last():
                return pyo.Constraint.Skip
            return _fxx_cubic(self.x[s + 1], self.alpha, s) == _fxx_cubic(
                self.x[s + 1], self.alpha, s + 1
            )

        # Identify segments used to predict y_data at each x_data.  We use search in
        # instead of a dict lookup, since we don't want to require the data to be at
        # the knots, even though that is almost always the case.
        idx = np.searchsorted(x_knots, x_data)

        if end_point_constraint:
            self.add_endpoint_second_derivative_constraints()

        # Expression for difference between data and prediction
        @self.Expression(self.dat_idx)
        def ydiff(blk, d):
            s = idx[d - 1] + 1
            if s >= self.seg_idx.last():
                s = self.seg_idx.last()
            return self.y_data[d] - _f_cubic(self.x_data[d], self.alpha, s)

        if objective_form:
            # least squares objective
            self.obj = pyo.Objective(expr=sum(self.ydiff[d] ** 2 for d in self.dat_idx))
        else:

            @self.Constraint(self.dat_idx)
            def match_data(blk, d):
                return self.ydiff[d] == 0

    def add_endpoint_second_derivative_constraints(self):
        """Usually cubic splines use the endpoint constraints that the second
        derivative is zero.  This function adds those constraints to a model
        """
        self.second_derivative_at_endpoints = pyo.Constraint(
            [self.seg_idx.first(), self.seg_idx.last()], rule=_yxx_endpoint_is_zero_eqn
        )

    def add_decreasing_constraints(self, tol=0):
        """If the objective form of the parameter calculation is used, the
        data and the spline don't need to match exactly, and we can add
        constraints on the derivatives that they are negative at the knots.

        This doesn't necessarily mean the cubic spline function is always
        decreasing, since the segments are cubic, but you can either check the
        resulting curve or pair it with convex or concave constraints.
        """
        self.decreasing_ineq = pyo.Constraint(
            self.knt_idx, rule=_decreasing_constraint(tol)
        )

    def add_increasing_constraints(self, tol=0):
        """If the objective form of the parameter calculation is used, the
        data and the spline don't need to match exactly, and we can add
        constraints on the derivatives that they are positive at the knots.

        This doesn't necessarily mean the cubic spline function is always
        increasing, since the segments are cubic, but you can either check the
        resulting curve or pair it with convex or concave constraints.
        """
        self.increasing_ineq = pyo.Constraint(
            self.knt_idx, rule=_increasing_constraint(tol)
        )

    def add_concave_constraints(self, tol=0):
        """If the objective form of the parameter calculation is used, the
        data and the spline don't need to match exactly, and we can add
        constraints on the second derivatives that they are always negative.
        """
        self.concave_ineq = pyo.Constraint(self.knt_idx, rule=_concave_constraint(tol))

    def add_convex_constraints(self, tol=0):
        """If the objective form of the parameter calculation is used, the
        data and the spline don't need to match exactly, and we can add
        constraints on the second derivatives that they are always positive.
        """
        self.convex_ineq = pyo.Constraint(self.knt_idx, rule=_convex_constraint(tol))


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
    m = CubicParametersModel(name=name, concrete=True)
    m.add_model(
        x_data=x_data,
        y_data=y_data,
        x_knots=x_knots,
        end_point_constraint=end_point_constraint,
        objective_form=objective_form,
    )
    return m


def add_decreasing_constraints(m, tol=0):
    """If the objective form of the parameter calculation is used, the
    data and the spline don't need to match exactly, and we can add
    constraints on the derivatives that they are negative at the knots.

    This doesn't necessarily mean the cubic spline function is always
    decreasing, since the segments are cubic, but you can either check the
    resulting curve or pair it with convex or concave constraints.
    """
    m.add_decreasing_constraints(tol=tol)


def add_concave_constraints(m, tol=0):
    """If the objective form of the parameter calculation is used, the
    data and the spline don't need to match exactly, and we can add
    constraints on the second derivatives that they are always negative.
    """
    m.concave_ineq = pyo.Constraint(m.knt_idx, rule=_concave_constraint(tol))


def add_increasing_constraints(m, tol=0):
    """If the objective form of the parameter calculation is used, the
    data and the spline don't need to match exactly, and we can add
    constraints on the derivatives that they are positive at the knots.

    This doesn't necessarily mean the cubic spline function is always
    increasing, since the segments are cubic, but you can either check the
    resulting curve or pair it with convex or concave constraints.
    """
    m.add_increasing_constraints(tol=tol)


def add_convex_constraints(m, tol=0):
    """If the objective form of the parameter calculation is used, the
    data and the spline don't need to match exactly, and we can add
    constraints on the second derivatives that they are always positive.
    """
    m.convex_ineq = pyo.Constraint(m.knt_idx, rule=_convex_constraint(tol))
