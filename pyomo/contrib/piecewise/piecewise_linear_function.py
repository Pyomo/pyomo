#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.core import Any, NonNegativeIntegers, value
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.expression import Expression
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base.initializer import Initializer, FunctionInitializer
import pyomo.core.expr.current as EXPR

np, numpy_available = attempt_import('numpy')
scipy, scipy_available = attempt_import('scipy')
spatial, scipy_available = attempt_import('scipy.spatial')

# DEBUG
from pytest import set_trace



class PiecewiseLinearFunctionData(_BlockData):
    _Block_reserved_words = Any

    def __init__(self, component=None):
        _BlockData.__init__(self, component)

        with self._declare_reserved_components():
            self._expressions = Expression(NonNegativeIntegers)
            self._simplices = []
            self._points = []
            self._linear_functions = []

    def __call__(self, *args):
        ans = 0
        if all(arg in EXPR.native_types for arg in args):
            # We need to actually evaluate
            return self._evaluate(*args)

    def _evaluate(self, *args):
        # ESJ: This is a very inefficient implementation in high dimension, but
        # for now we will just do a linear scan of the simplices.
        pt = (arg for arg in args)
        for idx, simplex, func in enumerate(zip(sef._simplices,
                                                self._linear_functions)):
            if _pt_in_simplex(pt, simplex):
                return func(*args)

    def _pt_in_simplex(self, pt, simplex):
        dim = len(pt)
        if dim == 1:
            return self._points[simplex[0]] <= pt and \
                self._points[simplex[1]] >= pt
        # Otherwise, we check if pt is a convex combination of the simplex's
        # extreme points
        A = np.ones((dim, dim))
        b = np.ones((dim, 0))
        for i, extreme_point in enumerate(simplex):
            b[i] = pt[i]
            for j, coord in enumerate(self._points[extreme_point]):
                A[i, j] = coord
        try:
            lambdas = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            if 'Singular' in str(e):
                # It's singular
                return False
            else:
                # This would be a bug (non-square system)
                raise
        for l in lambdas:
            if l < 0:
                # TODO: Do we need a tolerance?? Eeeeeek
                return False
        return True

    @property
    def simplices(self):
        # TODO: Is this really what you want? These are by index right now,
        # scipy style
        return self._simplices

    @property
    def linear_functions(self):
        return self._linear_functions


@ModelComponentFactory.register("Multidimensional piecewise linear function")
class PiecewiseLinearFunction(Block):
    """A piecewise linear function, which may be defined over an index.

    Can be specified in one of several ways:
        1) List of points and a nonlinear function to approximate. In
           this case, the points will be used to derive a triangulation
           of the part of the domain of interest, and a linear function
           approximating the given function will be calculated for each
           of the simplices in the triangulation. In this case, scipy is
           required.
        2) List of breakpoints along each dimension (variable) in the 
           function.

    Args:
        function: Nonlinear function to approximate, given as a Pyomo
            expression
        points: List of points in the same dimension as the domain of the 
            function being approximated. Note that if the pieces of the 
            function are specified this way, we require scipy.
        breakpoints: A ComponentMap mapping each variable in the function
            to a list of breakpoints along the corresponding dimension. If
            the pieces of  
    """
    _ComponentDataClass = PiecewiseLinearFunctionData

    def __new__(cls, *args, **kwds):
        if cls != PiecewiseLinearFunction:
            return super(PiecewiseLinearFunction, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return PiecewiseLinearFunction.__new__(
                ScalarPiecewiseLinearFunction)
        else:
            return IndexedPiecewiseLinearFunction.__new__(
                IndexedPiecewiseLinearFunction)

    def __init__(self, *args, **kwargs):
        self._handlers = {
            # (f, pts, simplices, linear_funcs) : handler
            (True, True, False, 
             False): self._construct_from_function_and_points,
            (True, False, True, 
             False): self._construct_from_function_and_simplices,
            (False, False, True, 
             True): self._construct_from_linear_functions_and_simplices
        }

        _func_arg = kwargs.pop('function', None)
        _points_arg = kwargs.pop('points', None)
        _simplices_arg = kwargs.pop('simplices', None)
        _linear_functions = kwargs.pop('linear_functions', None)

        kwargs.setdefault('ctype', PiecewiseLinearFunction)
        Block.__init__(self, *args, **kwargs)

        self._func_rule = FunctionInitializer(_func_arg)
        self._points_rule = Initializer(_points_arg,
                                        treat_sequences_as_mappings=False)
        self._simplices_rule = Initializer(_simplices_arg,
                                           treat_sequences_as_mappings=False)
        self._linear_funcs_rule = Initializer(_linear_functions)

    def _construct_from_function_and_points(self, obj, parent, nonlinear):
        parent = obj.parent_block()
        idx = obj._index

        nonlinear_function = self._func_rule(parent, idx)
        points = self._points_rule(parent, idx)

        func_vars = list(EXPR.identify_variables(nonlinear_function))
        if len(func_vars) == 1:
            # This is univariate and we'll handle it separately in order to
            # avoid a dependence on numpy.
            points.sort()
            simplices = [(points[i], points[i + 1]) for i in range(len(points) -
                                                                   1)]
            return self._construct_from_univariate_function_and_segments(
                obj, nonlinear_function, simplices, func_vars[0]) 

        try:
            triangulation = spatial.Delaunay(points)
        except (spatial.QhullError, ValueError) as error:
            logger.error("Unable to triangulate the set of input points.")
            raise

        obj._points = [pt for pt in points]
        obj._simplices = [simplex for simplex in map(tuple,
                                                     triangulation.simplices)]

        # TODO: Need to make some modifications to the below for this to
        # work--we should check if we already have simplices. And we should
        # always store them in the numpy/scipy style.
        return self._construct_from_function_and_simplices(obj)
        
        # TODO: Get simplices from scipy, then revert to the method below

    def _construct_from_univariate_function_and_segments(self, obj, func,
                                                         simplices):
        point_set = set()
        indices = {}
        for (x1, x2) in simplices:
            y_val = {}
            for xi in (x1, x2):
                y_val[xi] = func(xi)
                if xi not in point_set:
                    point_set.add(xi)
                    self._points.append(xi)
                    indices[xi] = len(self._points) - 1
            obj._simplices.append((indices[x1], indices[x2]))
            slope = (y_val[x2] - y_val[x1])/(x2 - x1)
            def f(x, y):
                return slope*x + y_val[x1] - slope*x1
            obj._linear_functions.append(f)

        return obj

    def _construct_from_function_and_simplices(self, obj, parent,
                                               nonlinear_function):
        simplices = self._simplices_rule(parent, idx)
        if len(simplices) < 1:
            raise ValueError("Cannot construct PiecewiseLinearFunction "
                             "with empty list of simplices")

        dimension = len(simplices[0]) - 1
        if dimension == 1:
            # Back to high school with us--this is univariate and we'll handle
            # it separately in order to avoid a kind of silly dependence on
            # numpy.
            return self._construct_from_univariate_function_and_segments(
                obj, nonlinear_function, simplices) 

        A = np.ones((dimension, dimension + 1))
        b = np.empty((dimension + 1, 0))

        # evaluate the function at each of the points and form the homogeneous
        # system of equations
        for num_piece, simplex in enumerate(simplices):
            for i, pt in enumerate(simplex):
                for j, val in enumerate(pt):
                    A[i, j] = val
                b[i] = nonlinear_function(*pt)
            # This system has a solution unless there's a bug--we know there is
            # a hyperplane that passes through dimension + 1 points.
            normal = np.linalg.solve(A, b)

            def f(*args):
                return sum(normal[i]*args[i] for i in dimension) - \
                    normal[dimension + 1]
            obj._linear_functions.append(f)

    def _construct_from_linear_functions_and_simplices(self, obj):
        pass

    def _getitem_when_not_present(self, index):
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)

        obj._index = index
        handler = self._handlers.get((self._func_rule is not None, 
                                      self._points_rule is not None, 
                                      self._simplices_rule is not None,
                                      self._linear_funcs_rule is not None))
        if handler is None:
            raise ValueError("Unsupported set of arguments given for "
                             "constructing PiecewiseLinearFunction. "
                             "Expected a nonlinear function and a list"
                             "of breakpoints, a nonlinear function an a list "
                             "of simplices, or a list of linear functions and "
                             "a list of corresponding domains.")
        parent = obj.parent_block()
        nonlinear_function = self._func_rule(parent, index)
        handler(obj, parent, nonlinear_function)

        return obj


class ScalarPiecewiseLinearFunction(PiecewiseLinearFunctionData,
                                    PiecewiseLinearFunction):
    def __init__(self, *args, **kwds):
        self._suppress_ctypes = set()

        PiecewiseLinearFunctionData.__init__(self, self)
        PiecewiseLinearFunction.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index


class IndexedPiecewiseLinearFunction(PiecewiseLinearFunction):
    pass

class PiecewiseLinearExpression():
    # This needs to be an expression node, it is what the _expressions above are
    # going to store.
    pass
    # example : m.c = Constraint(expr=m.pw(m.x, m.y) <= 0)
