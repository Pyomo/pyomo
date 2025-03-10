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

import logging

from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.common.dependencies.scipy import spatial
from pyomo.contrib.piecewise.piecewise_linear_expression import (
    PiecewiseLinearExpression,
)
from pyomo.contrib.piecewise.triangulations import (
    get_unordered_j1_triangulation,
    get_ordered_j1_triangulation,
    Triangulation,
)
from pyomo.core import Any, NonNegativeIntegers, value
from pyomo.core.base.block import BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.expression import Expression
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base.initializer import Initializer
import pyomo.core.expr as EXPR

# This is the default absolute tolerance in np.isclose... Not sure if it's
# enough, but we need to make sure that 'barely negative' values are assumed to
# be zero.
ZERO_TOLERANCE = 1e-8

logger = logging.getLogger(__name__)


class PiecewiseLinearFunctionData(BlockData):
    _Block_reserved_words = Any

    def __init__(self, component=None):
        BlockData.__init__(self, component)

        with self._declare_reserved_components():
            # map of PiecewiseLinearExpression objects to integer indices in
            # self._expressions
            self._expression_ids = ComponentMap()
            # index is monotonically increasing integer
            self._expressions = Expression(NonNegativeIntegers)
            self._transformed_exprs = ComponentMap()
            self._simplices = None
            # These will always be tuples, even when we only have one dimension.
            self._points = []
            self._linear_functions = []
            self._triangulation = None

    @property
    def triangulation(self):
        return self._triangulation

    def __call__(self, *args):
        """
        Returns a PiecewiseLinearExpression which is an instance of this
        function applied to the variables and/or constants specified in args.
        """
        if all(
            type(arg) in EXPR.native_types or not arg.is_potentially_variable()
            for arg in args
        ):
            # We need to actually evaluate
            return self._evaluate(*args)
        else:
            expr = PiecewiseLinearExpression(args, self)
            idx = len(self._expressions)
            self._expressions[idx] = expr
            self._expression_ids[expr] = idx
            return self._expressions[idx]

    def _evaluate(self, *args):
        # ESJ: This is a very inefficient implementation in high dimensions, but
        # for now we will just do a linear scan of the simplices.
        if self._simplices is None:
            raise RuntimeError(
                "Cannot evaluate PiecewiseLinearFunction--it "
                "appears it is not fully defined. (No simplices "
                "are stored.)"
            )

        pt = [value(arg) for arg in args]
        for simplex, func in zip(self._simplices, self._linear_functions):
            if self._pt_in_simplex(pt, simplex):
                return func(*args)

        raise ValueError(
            "Unsuccessful evaluation of PiecewiseLinearFunction "
            "'%s' at point (%s). Is the point in the function's "
            "domain?" % (self.name, ', '.join(str(arg) for arg in args))
        )

    def _pt_in_simplex(self, pt, simplex):
        dim = len(pt)
        if dim == 1:
            return (
                self._points[simplex[0]][0] <= pt[0]
                and self._points[simplex[1]][0] >= pt[0]
            )
        # Otherwise, we check if pt is a convex combination of the simplex's
        # extreme points
        A = np.ones((dim + 1, dim + 1))
        b = np.array([x for x in pt] + [1])
        for j, extreme_point in enumerate(simplex):
            for i, coord in enumerate(self._points[extreme_point]):
                A[i, j] = coord
        if np.linalg.det(A) == 0:
            # A is singular, so the system has no solutions
            return False
        else:
            lambdas = np.linalg.solve(A, b)
        for l in lambdas:
            if l < -ZERO_TOLERANCE:
                return False
        return True

    def _get_simplices_from_arg(self, simplices):
        self._simplices = []
        known_points = set()
        point_to_index = {}
        for simplex in simplices:
            extreme_pts = []
            for pt in simplex:
                if pt not in known_points:
                    known_points.add(pt)
                    if hasattr(pt, '__len__'):
                        self._points.append(pt)
                    else:
                        self._points.append((pt,))
                    point_to_index[pt] = len(self._points) - 1
                extreme_pts.append(point_to_index[pt])
            self._simplices.append(tuple(extreme_pts))

    def map_transformation_var(self, pw_expr, v):
        """
        Records on the PiecewiseLinearFunction object that the transformed
        form of the PiecewiseLinearExpression object pw_expr is the Var v.
        """
        if pw_expr not in self._expression_ids:
            raise DeveloperError(
                "ID of PiecewiseLinearExpression '%s' not in the _expression_ids "
                "dictionary of PiecewiseLinearFunction '%s'" % (pw_expr, self)
            )
        self._transformed_exprs[self._expressions[self._expression_ids[pw_expr]]] = v

    def get_transformation_var(self, pw_expr):
        """
        Returns the Var that replaced the PiecewiseLinearExpression 'pw_expr'
        after transformation, or None if 'pw_expr' has not been transformed.
        """
        if pw_expr in self._transformed_exprs:
            return self._transformed_exprs[pw_expr]
        else:
            return None


class _univariate_linear_functor(AutoSlots.Mixin):
    __slots__ = ('slope', 'intercept')

    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x):
        return self.slope * x + self.intercept


class _multivariate_linear_functor(AutoSlots.Mixin):
    __slots__ = ('normal',)

    def __init__(self, normal):
        self.normal = normal

    def __call__(self, *args):
        return sum(self.normal[i] * arg for i, arg in enumerate(args)) + self.normal[-1]


class _tabular_data_functor(AutoSlots.Mixin):
    __slots__ = ('tabular_data',)

    def __init__(self, tabular_data, tupleize=False):
        if not tupleize:
            self.tabular_data = tabular_data
        else:
            self.tabular_data = {(pt,): val for pt, val in tabular_data.items()}

    def __call__(self, *args):
        return self.tabular_data[args]


def _define_handler(handle_map, *key):
    def _wrapper(obj):
        assert key not in handle_map
        handle_map[key] = obj
        return obj

    return _wrapper


@ModelComponentFactory.register("Multidimensional piecewise linear function")
class PiecewiseLinearFunction(Block):
    """A piecewise linear function, which may be defined over an index.

    Can be specified in one of several ways:
        1) List of points and a nonlinear function to approximate. In
           this case, the points will be used to derive a triangulation
           of the part of the domain of interest, and a linear function
           approximating the given function will be calculated for each
           of the simplices in the triangulation. In this case, scipy is
           required (for multivariate functions).
        2) List of simplices and a nonlinear function to approximate. In
           this case, a linear function approximating the given function
           will be calculated for each simplex. For multivariate functions,
           numpy is required.
        3) List of simplices and list of functions that return linear function
           expressions. These are the desired piecewise functions
           corresponding to each simplex.
        4) Mapping of function values to points of the domain, allowing for
           the construction of a piecewise linear function from tabular data.

    Args:
        function: Nonlinear function to approximate: must be callable
        function_rule: Function that returns a nonlinear function to
            approximate for each index in an IndexedPiecewiseLinearFunction
        points: List of points in the same dimension as the domain of the
            function being approximated. Note that if the pieces of the
            function are specified this way, we require scipy.
        simplices: A list of lists of points, where each list specifies the
            extreme points of a a simplex over which the nonlinear function
            will be approximated as a linear function.
        linear_functions: A list of functions, each of which returns an
            expression for a linear function of the arguments.
        tabular_data: A dictionary mapping values of the nonlinear function
            to points in the domain
        triangulation (optional): An enum value of type Triangulation specifying
            how Pyomo should triangulate the function domain, or None. Behavior
            depends on how this piecewise-linear function is constructed:
            when constructed using methods (1) or (4) above, valid arguments
            are the members of Triangulation except Unknown or AssumeValid,
            and Pyomo will use that method to triangulate the domain and to tag
            the resulting PWLF. If no argument or None is passed, the default
            is Triangulation.Delaunay. When constructed using methods (2) or (3)
            above, valid arguments are only Triangulation.Unknown and
            Triangulation.AssumeValid. Pyomo will tag the constructed PWLF
            as specified, trusting the user in the case of AssumeValid.
            When no argument or None is passed, the default is
            Triangulation.Unknown
    """

    _ComponentDataClass = PiecewiseLinearFunctionData

    # Map 5-tuple of bool to handler: "(f, pts, simplices, linear_funcs,
    # tabular_data) : handler"
    _handlers = {}

    def __new__(cls, *args, **kwds):
        if cls is not PiecewiseLinearFunction:
            return super(PiecewiseLinearFunction, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return PiecewiseLinearFunction.__new__(ScalarPiecewiseLinearFunction)
        else:
            return IndexedPiecewiseLinearFunction.__new__(
                IndexedPiecewiseLinearFunction
            )

    def __init__(self, *args, **kwargs):
        _func_arg = kwargs.pop('function', None)
        _func_rule_arg = kwargs.pop('function_rule', None)
        _points_arg = kwargs.pop('points', None)
        _simplices_arg = kwargs.pop('simplices', None)
        _linear_functions = kwargs.pop('linear_functions', None)
        _tabular_data_arg = kwargs.pop('tabular_data', None)
        _tabular_data_rule_arg = kwargs.pop('tabular_data_rule', None)
        _triangulation_rule_arg = kwargs.pop('triangulation', None)

        kwargs.setdefault('ctype', PiecewiseLinearFunction)
        Block.__init__(self, *args, **kwargs)

        # This cannot be a rule.
        self._func = _func_arg
        self._func_rule = Initializer(_func_rule_arg)
        self._points_rule = Initializer(_points_arg, treat_sequences_as_mappings=False)
        self._simplices_rule = Initializer(
            _simplices_arg, treat_sequences_as_mappings=False
        )
        self._linear_funcs_rule = Initializer(
            _linear_functions, treat_sequences_as_mappings=False
        )
        self._tabular_data = _tabular_data_arg
        self._tabular_data_rule = Initializer(
            _tabular_data_rule_arg, treat_sequences_as_mappings=False
        )
        self._triangulation_rule = Initializer(
            _triangulation_rule_arg, treat_sequences_as_mappings=False
        )

    def _get_dimension_from_points(self, points):
        if len(points) < 1:
            raise ValueError(
                "Cannot construct PiecewiseLinearFunction from "
                "points list of length 0."
            )

        if hasattr(points[0], '__len__'):
            dimension = len(points[0])
        else:
            dimension = 1

        return dimension

    def _construct_simplices_from_multivariate_points(
        self, obj, parent, points, dimension
    ):
        if self._triangulation_rule is None:
            tri = Triangulation.Delaunay
        else:
            tri = self._triangulation_rule(parent, obj._index)
            if tri is None:
                tri = Triangulation.Delaunay

        if tri == Triangulation.Delaunay:
            try:
                triangulation = spatial.Delaunay(points)
            except (spatial.QhullError, ValueError) as error:
                logger.error("Unable to triangulate the set of input points.")
                raise
            obj._triangulation = tri
        elif tri == Triangulation.J1:
            triangulation = get_unordered_j1_triangulation(points, dimension)
            obj._triangulation = tri
        elif tri == Triangulation.OrderedJ1:
            triangulation = get_ordered_j1_triangulation(points, dimension)
            obj._triangulation = tri
        else:
            raise ValueError(
                "Invalid or unrecognized triangulation specified for '%s': %s"
                % (obj, tri)
            )

        # Get the points for the triangulation because they might not all be
        # there if any were coplanar.
        obj._points = [pt for pt in map(tuple, triangulation.points)]
        obj._simplices = []
        for simplex in triangulation.simplices:
            # For each simplex, check whether or not the simplex is
            # degenerate. If it is, we will just drop it.

            # We have n + 1 points in n dimensions.
            # We put them in a n x (n + 1) matrix: [p_0 p_1 ... p_n]
            points = triangulation.points[simplex].transpose()
            # The question is if they span R^n: We construct the square matrix
            # [p_1 - p_0  p_2 - p_1  ...  p_n - p_{n-1}] and check if it is full
            # rank. Note that we use numpy's matrix_rank function rather than
            # checking the determinant because matrix_rank will by default calculate a
            # tolerance based on the input to account for numerical errors in the
            # SVD computation.
            if tri in (Triangulation.J1, Triangulation.OrderedJ1):
                # Note: do not sort vertices from OrderedJ1, or it will break.
                # Non-ordered J1 is already sorted, though it doesn't matter.
                # Also, we don't need to check for degeneracy with simplices we
                # made ourselves.
                obj._simplices.append(tuple(simplex))
            elif (
                np.linalg.matrix_rank(
                    points[:, 1:]
                    - np.append(points[:, : dimension - 1], points[:, [0]], axis=1)
                )
                == dimension
            ):
                obj._simplices.append(tuple(sorted(simplex)))

        # It's possible that qhull dropped some points if there were numerical
        # issues with them (e.g., if they were redundant). We'll be polite and
        # tell the user:
        for pt in triangulation.coplanar:
            logger.info(
                "The Delaunay triangulation dropped the point with index "
                "%s from the triangulation." % pt[0]
            )

    # Call when constructing from simplices to allow use of AssumeValid and
    # ensure the user is not making mistakes
    def _check_and_set_triangulation_from_user(self, parent, obj):
        if self._triangulation_rule is None:
            tri = None
        else:
            tri = self._triangulation_rule(parent, obj._index)
        if tri is None or tri == Triangulation.Unknown:
            obj._triangulation = Triangulation.Unknown
        elif tri == Triangulation.AssumeValid:
            obj._triangulation = Triangulation.AssumeValid
        else:
            raise ValueError(
                f"Invalid or unrecognized triangulation tag specified for {obj} when"
                f" giving simplices: {tri}. Valid arguments when giving simplices are"
                " Triangulation.Unknown and Triangulation.AssumeValid."
            )

    def _construct_one_dimensional_simplices_from_points(self, obj, points):
        points.sort()
        obj._simplices = []
        for i in range(len(points) - 1):
            obj._simplices.append((i, i + 1))
            obj._points.append((points[i],))
        # Add the last one
        obj._points.append((points[-1],))

    @_define_handler(_handlers, True, True, False, False, False)
    def _construct_from_function_and_points(self, obj, parent, nonlinear_function):
        idx = obj._index

        points = self._points_rule(parent, idx)
        dimension = self._get_dimension_from_points(points)

        if dimension == 1:
            # This is univariate and we'll handle it separately in order to
            # avoid a dependence on scipy.
            self._construct_one_dimensional_simplices_from_points(obj, points)
            return self._construct_from_univariate_function_and_segments(
                obj, parent, nonlinear_function, segments_are_user_defined=False
            )

        self._construct_simplices_from_multivariate_points(
            obj, parent, points, dimension
        )
        return self._construct_from_function_and_simplices(
            obj, parent, nonlinear_function, simplices_are_user_defined=False
        )

    def _construct_from_univariate_function_and_segments(
        self, obj, parent, func, segments_are_user_defined=True
    ):
        # We can trust they are nicely ordered if we made them, otherwise anything goes.
        if segments_are_user_defined:
            self._check_and_set_triangulation_from_user(parent, obj)
        else:
            obj._triangulation = Triangulation.AssumeValid

        for idx1, idx2 in obj._simplices:
            x1 = obj._points[idx1][0]
            x2 = obj._points[idx2][0]
            y = {x: func(x) for x in [x1, x2]}
            slope = (y[x2] - y[x1]) / (x2 - x1)
            intercept = y[x1] - slope * x1
            obj._linear_functions.append(_univariate_linear_functor(slope, intercept))

        return obj

    @_define_handler(_handlers, True, False, True, False, False)
    def _construct_from_function_and_simplices(
        self, obj, parent, nonlinear_function, simplices_are_user_defined=True
    ):
        if obj._simplices is None:
            obj._get_simplices_from_arg(self._simplices_rule(parent, obj._index))
        simplices = obj._simplices

        if len(simplices) < 1:
            raise ValueError(
                "Cannot construct PiecewiseLinearFunction "
                "with empty list of simplices"
            )

        dimension = len(simplices[0]) - 1
        if dimension == 1:
            # Back to high school with us--this is univariate and we'll handle
            # it separately in order to avoid a kind of silly dependence on
            # numpy.
            return self._construct_from_univariate_function_and_segments(
                obj, parent, nonlinear_function, simplices_are_user_defined
            )

        # If we triangulated, then this tag was already set. If they provided it,
        # then check their arguments and set.
        if simplices_are_user_defined:
            self._check_and_set_triangulation_from_user(parent, obj)

        # evaluate the function at each of the points and form the homogeneous
        # system of equations
        A = np.ones((dimension + 2, dimension + 2))
        b = np.zeros(dimension + 2)
        b[-1] = 1

        for num_piece, simplex in enumerate(simplices):
            for i, pt_idx in enumerate(simplex):
                pt = obj._points[pt_idx]
                for j, val in enumerate(pt):
                    A[i, j] = val
                A[i, j + 1] = nonlinear_function(*pt)
            A[i + 1, :] = 0
            A[i + 1, dimension] = -1
            # This system has a solution unless there's a bug--we filtered the
            # simplices to make sure they are full-dimensional, so we know there
            # is a hyperplane that passes through these dimension + 1 points (and the
            # last equation scales it so that the coefficient for the output of
            # the nonlinear function dimension is -1, so we can just read off
            # the linear equation in the x space).
            try:
                normal = np.linalg.solve(A, b)
            except np.linalg.LinAlgError as e:
                logger.warning('LinAlgError: %s' % e)
                msg = (
                    "When calculating the hyperplane approximation over the simplex "
                    "with index %s, the matrix was unexpectedly singular. This "
                    "likely means that this simplex is degenerate" % num_piece
                )

                if simplices_are_user_defined:
                    raise ValueError(msg)
                # otherwise it's our fault, and I was hoping this is unreachable
                # code...
                raise DeveloperError(
                    msg
                    + " and that it should have been filtered out of the triangulation"
                )

            obj._linear_functions.append(_multivariate_linear_functor(normal))

        return obj

    @_define_handler(_handlers, False, False, True, True, False)
    def _construct_from_linear_functions_and_simplices(
        self, obj, parent, nonlinear_function
    ):
        # We know that we have simplices because else this handler wouldn't
        # have been called.
        obj._get_simplices_from_arg(self._simplices_rule(parent, obj._index))
        obj._linear_functions = [f for f in self._linear_funcs_rule(parent, obj._index)]
        self._check_and_set_triangulation_from_user(parent, obj)
        return obj

    @_define_handler(_handlers, False, False, False, False, True)
    def _construct_from_tabular_data(self, obj, parent, nonlinear_function):
        idx = obj._index

        tabular_data = self._tabular_data
        if tabular_data is None:
            tabular_data = self._tabular_data_rule(parent, idx)
        points = [pt for pt in tabular_data.keys()]
        dimension = self._get_dimension_from_points(points)

        if dimension == 1:
            # This is univariate and we'll handle it separately in order to
            # avoid a dependence on scipy.
            self._construct_one_dimensional_simplices_from_points(obj, points)
            return self._construct_from_univariate_function_and_segments(
                obj,
                parent,
                _tabular_data_functor(tabular_data, tupleize=True),
                segments_are_user_defined=False,
            )

        self._construct_simplices_from_multivariate_points(
            obj, parent, points, dimension
        )
        return self._construct_from_function_and_simplices(
            obj,
            parent,
            _tabular_data_functor(tabular_data),
            simplices_are_user_defined=False,
        )

    def _getitem_when_not_present(self, index):
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        obj._index = index
        parent = obj.parent_block()

        # Get the nonlinear function, if we have one.
        nonlinear_function = None
        if self._func_rule is not None:
            nonlinear_function = self._func_rule(parent, index)
        elif self._func is not None:
            nonlinear_function = self._func

        handler = self._handlers.get(
            (
                nonlinear_function is not None,
                self._points_rule is not None,
                self._simplices_rule is not None,
                self._linear_funcs_rule is not None,
                self._tabular_data is not None or self._tabular_data_rule is not None,
            )
        )
        if handler is None:
            raise ValueError(
                "Unsupported set of arguments given for "
                "constructing PiecewiseLinearFunction. "
                "Expected a nonlinear function and a list"
                "of breakpoints, a nonlinear function and a list "
                "of simplices, a list of linear functions and "
                "a list of corresponding simplices, or a dictionary "
                "mapping points to nonlinear function values."
            )
        obj = handler(self, obj, parent, nonlinear_function)

        return obj


class ScalarPiecewiseLinearFunction(
    PiecewiseLinearFunctionData, PiecewiseLinearFunction
):
    def __init__(self, *args, **kwds):
        PiecewiseLinearFunctionData.__init__(self, self)
        PiecewiseLinearFunction.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index


class IndexedPiecewiseLinearFunction(PiecewiseLinearFunction):
    pass
