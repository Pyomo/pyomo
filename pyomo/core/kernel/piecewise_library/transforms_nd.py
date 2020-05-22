#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
This module contains transformations for representing a
multi-variate piecewise linear function using a
mixed-interger problem formulation. Reference::

  Mixed-Integer Models for Non-separable Piecewise Linear \
Optimization: Unifying framework and Extensions (Vielma, \
Nemhauser 2008)
"""

import logging
import collections

from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.variable import (variable,
                                        variable_dict,
                                        variable_tuple)
from pyomo.core.kernel.constraint import (linear_constraint,
                                          constraint_list,
                                          constraint_tuple)
from pyomo.core.kernel.expression import (expression,
                                          expression_tuple)
import pyomo.core.kernel.piecewise_library.util

logger = logging.getLogger('pyomo.core')

registered_transforms = {}

def piecewise_nd(tri,
                 values,
                 input=None,
                 output=None,
                 bound='eq',
                 repn='cc'):
    """
    Models a multi-variate piecewise linear function.

    This function takes a D-dimensional triangulation and a
    list of function values associated with the points of
    the triangulation and transforms this input data into a
    block of variables and constraints that enforce a
    piecewise linear relationship between an D-dimensional
    vector of input variable and a single output
    variable. In the general case, this transformation
    requires the use of discrete decision variables.

    Args:
        tri (scipy.spatial.Delaunay): A triangulation over
            the discretized variable domain. Can be
            generated using a list of variables using the
            utility function :func:`util.generate_delaunay`.
            Required attributes:

              - points: An (npoints, D) shaped array listing
                the D-dimensional coordinates of the
                discretization points.
              - simplices: An (nsimplices, D+1) shaped array
                of integers specifying the D+1 indices of
                the points vector that define each simplex
                of the triangulation.
        values (numpy.array): An (npoints,) shaped array of
            the values of the piecewise function at each of
            coordinates in the triangulation points array.
        input: A D-length list of variables or expressions
            bound as the inputs of the piecewise function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound to impose on the
            output expression. Can be one of:

              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
        repn (str): The type of piecewise representation to
            use. Can be one of:

                - 'cc': convex combination

    Returns:
        TransformedPiecewiseLinearFunctionND: a block \
            containing any new variables, constraints, and \
            other components used by the piecewise \
            representation
    """
    transform = None
    try:
        transform = registered_transforms[repn]
    except KeyError:
        raise ValueError(
            "Keyword assignment repn='%s' is not valid. "
            "Must be one of: %s"
            % (repn,
               str(sorted(registered_transforms.keys()))))
    assert transform is not None

    func = PiecewiseLinearFunctionND(tri,
                                     values)

    return transform(func,
                     input=input,
                     output=output,
                     bound=bound)

class PiecewiseLinearFunctionND(object):
    """A multi-variate piecewise linear function

    Multi-varite piecewise linear functions are defined by a
    triangulation over a finite domain and a list of
    function values associated with the points of the
    triangulation.  The function value between points in the
    triangulation is implied through linear interpolation.

    Args:
        tri (scipy.spatial.Delaunay): A triangulation over
            the discretized variable domain. Can be
            generated using a list of variables using the
            utility function :func:`util.generate_delaunay`.
            Required attributes:

              - points: An (npoints, D) shaped array listing
                the D-dimensional coordinates of the
                discretization points.
              - simplices: An (nsimplices, D+1) shaped array
                of integers specifying the D+1 indices of
                the points vector that define each simplex
                of the triangulation.
        values (numpy.array): An (npoints,) shaped array of
            the values of the piecewise function at each of
            coordinates in the triangulation points array.
    """
    __slots__ = ("_tri", "_values")

    def __init__(self,
                 tri,
                 values,
                 validate=True,
                 **kwds):
        assert pyomo.core.kernel.piecewise_library.util.numpy_available
        assert pyomo.core.kernel.piecewise_library.util.scipy_available
        assert isinstance(tri,
                          pyomo.core.kernel.piecewise_library.\
                          util.scipy.spatial.Delaunay)
        assert isinstance(values,
                          pyomo.core.kernel.piecewise_library.\
                          util.numpy.ndarray)
        npoints, ndim = tri.points.shape
        nsimplices, _ = tri.simplices.shape
        assert tri.simplices.shape[1] == ndim + 1
        assert nsimplices > 0
        assert npoints > 0
        assert ndim > 0
        self._tri = tri
        self._values = values

    def __getstate__(self):
        """Required for older versions of the pickle
        protocol since this class uses __slots__"""
        return {key:getattr(self, key) for key in self.__slots__}

    def __setstate__(self, state):
        """Required for older versions of the pickle
        protocol since this class uses __slots__"""
        for key in state:
            setattr(self, key, state[key])

    @property
    def triangulation(self):
        """The triangulation over the domain of this function"""
        return self._tri

    @property
    def values(self):
        """The set of values used to defined this function"""
        return self._values

    def __call__(self, x):
        """
        Evaluates the piecewise linear function using
        interpolation. This method supports vectorized
        function calls as the interpolation process can be
        expensive for high dimensional data.

        For the case when a single point is provided, the
        argument x should be a (D,) shaped numpy array or
        list, where D is the dimension of points in the
        triangulation.

        For the vectorized case, the argument x should be
        a (n,D)-shaped numpy array.
        """
        assert isinstance(x, collections.Sized)
        if isinstance(x, pyomo.core.kernel.piecewise_library.\
                      util.numpy.ndarray):
            if x.shape != self._tri.points.shape[1:]:
                multi = True
                assert x.shape[1:] == self._tri.points[0].shape, \
                    "%s[1] != %s" % (x.shape, self._tri.points[0].shape)
            else:
                multi = False
        else:
            multi = False
        _, ndim = self._tri.points.shape
        i = self._tri.find_simplex(x)
        if multi:
            Tinv = self._tri.transform[i,:ndim]
            r = self._tri.transform[i,ndim]
            b = pyomo.core.kernel.piecewise_library.util.\
                numpy.einsum('ijk,ik->ij', Tinv, x-r)
            b = pyomo.core.kernel.piecewise_library.util.\
                numpy.c_[b, 1 - b.sum(axis=1)]
            s = self._tri.simplices[i]
            return (b*self._values[s]).sum(axis=1)
        else:
            b = self._tri.transform[i,:ndim,:ndim].dot(
                x - self._tri.transform[i,ndim,:])
            s = self._tri.simplices[i]
            val = b.dot(self._values[s[:ndim]])
            val += (1-b.sum())*self._values[s[ndim]]
            return val

class TransformedPiecewiseLinearFunctionND(block):
    """Base class for transformed multi-variate piecewise
    linear functions

    A transformed multi-variate piecewise linear functions
    is a block of variables and constraints that enforce a
    piecewise linear relationship between an vector input
    variables and a single output variable.

    Args:
        f (:class:`PiecewiseLinearFunctionND`): The
            multi-variate piecewise linear function to
            transform.
        input: The variable constrained to be the input of
            the piecewise linear function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound to impose on the
            output expression. Can be one of:

              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
    """

    def __init__(self,
                 f,
                 input=None,
                 output=None,
                 bound='eq'):
        super(TransformedPiecewiseLinearFunctionND, self).__init__()
        assert isinstance(f, PiecewiseLinearFunctionND)
        if bound not in ('lb', 'ub', 'eq'):
            raise ValueError("Invalid bound type %r. Must be "
                             "one of: ['lb','ub','eq']"
                             % (bound))
        self._bound = bound
        self._f = f
        _,ndim = f._tri.points.shape
        if input is None:
            input = [None]*ndim
        self._input = expression_tuple(
            expression(input[i]) for i in range(ndim))
        self._output = expression(output)

    @property
    def input(self):
        """The tuple of expressions that store the
        inputs to the piecewise function. The returned
        objects can be updated by assigning to their
        :attr:`expr` attribute."""
        return self._input

    @property
    def output(self):
        """The expression that stores the output of the
        piecewise function. The returned object can be
        updated by assigning to its :attr:`expr`
        attribute."""
        return self._output

    @property
    def bound(self):
        """The bound type assigned to the piecewise
        relationship ('lb','ub','eq')."""
        return self._bound

    @property
    def triangulation(self):
        """The triangulation over the domain of this function"""
        return self._f.triangulation

    @property
    def values(self):
        """The set of values used to defined this function"""
        return self._f.values

    def __call__(self, x):
        """
        Evaluates the piecewise linear function using
        interpolation. This method supports vectorized
        function calls as the interpolation process can be
        expensive for high dimensional data.

        For the case when a single point is provided, the
        argument x should be a (D,) shaped numpy array or
        list, where D is the dimension of points in the
        triangulation.

        For the vectorized case, the argument x should be
        a (n,D)-shaped numpy array.
        """
        return self._f(x)

class piecewise_nd_cc(TransformedPiecewiseLinearFunctionND):
    """Discrete CC multi-variate piecewise representation

    Expresses a multi-variate piecewise linear function
    using the CC formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_nd_cc, self).__init__(*args, **kwds)

        ndim = len(self.input)
        nsimplices = len(self.triangulation.simplices)
        npoints = len(self.triangulation.points)
        pointsT = list(zip(*self.triangulation.points))

        # create index objects
        dimensions = range(ndim)
        simplices = range(nsimplices)
        vertices = range(npoints)

        # create vars
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_tuple(
            variable(lb=0) for v in vertices)
        y = self.v['y'] = variable_tuple(
            variable(domain_type=IntegerSet, lb=0, ub=1) for s in simplices)
        lmbda_tuple = tuple(lmbda)

        # create constraints
        self.c = constraint_list()

        clist = []
        for d in dimensions:
            clist.append(linear_constraint(
                variables=lmbda_tuple + (self.input[d],),
                coefficients=tuple(pointsT[d]) + (-1,),
                rhs=0))
        self.c.append(constraint_tuple(clist))
        del clist

        self.c.append(linear_constraint(
            variables=lmbda_tuple + (self.output,),
            coefficients=tuple(self.values) + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0

        self.c.append(linear_constraint(
            variables=lmbda_tuple,
            coefficients=(1,)*len(lmbda_tuple),
            rhs=1))

        # generate a map from vertex index to simplex index,
        # which avoids an n^2 lookup when generating the
        # constraint
        vertex_to_simplex = [[] for v in vertices]
        for s, simplex in enumerate(self.triangulation.simplices):
            for v in simplex:
                vertex_to_simplex[v].append(s)

        clist = []
        for v in vertices:
            variables = tuple(y[s] for s in vertex_to_simplex[v])
            clist.append(linear_constraint(
                variables=variables + (lmbda[v],),
                coefficients=(1,)*len(variables) + (-1,),
                lb=0))
        self.c.append(constraint_tuple(clist))
        del clist

        self.c.append(linear_constraint(
            variables=y,
            coefficients=(1,)*len(y),
            rhs=1))

registered_transforms['cc'] = piecewise_nd_cc
