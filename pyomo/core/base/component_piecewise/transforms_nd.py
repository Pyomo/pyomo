#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
"""
This module contains a library of transformations for
representing a multivariate piecewise linear function using
a mixed-interger problem formulation.
"""

import logging
import collections

from pyomo.core.base.component_block import tiny_block
from pyomo.core.base.set_types import Binary
from pyomo.core.base.component_variable import (variable,
                                                variable_list)
from pyomo.core.base.component_constraint import (linear_constraint,
                                                  constraint_list)
from pyomo.core.base.component_expression import (expression,
                                                  expression_list)
import pyomo.core.base.component_piecewise.util

logger = logging.getLogger('pyomo.core')

registered_transforms = {}

def piecewise_nd(tri,
                 values,
                 input=None,
                 output=None,
                 bound='eq',
                 repn='cc'):
    """
    Transforms a D-dimensional triangulation and a list of
    function values associated with the points of the
    triangulation into a mixed-integer representation of a
    piecewise function over a D-dimensional domain.

    Args:
        tri (scipy.spatial.Delaunay): A triangulation over
            the discretized variable domain. Required
            attributes:
           - points: An (npoints, D) shaped array listing the
                     D-dimensional coordinates of the
                     discretization points.
           - simplices: An (nsimplices, D+1) shaped array of
                        integers specifying the D+1 indices
                        of the points vector that define
                        each simplex of the triangulation.
           ** util.generate_delaunay can be use to build
              this input.
        values (numpy.array): An (npoints,) shaped array of
            the values of the piecewise function at each of
            coordinates in the triangulation points array.
        input: A D-length list of variables or expressions
            bound as the inputs of the piecewise function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound on the output to
            generate. Can be one of:
                - 'lb': y <= f(x)
                - 'eq': y  = f(x)
                - 'ub': y >= f(x)
        repn (str): The type of piecewise representation to
            use. Can be one of:
                - 'cc': convex combination (*)
           * source: "Mixed-Integer Models for Non-separable
                      Piecewise Linear Optimization:
                      Unifying framework and Extensions"
                      (Vielma, Nemhauser 2008)

    Returns: A block containing the necessary auxiliary
        variables and constraints to enforce the piecewise
        linear relationship between the inputs and output.
    """
    transorm = None
    try:
        transform = registered_transforms[repn]
    except KeyError:
        raise ValueError(
            "Keyword assignment repn='%s' is not valid. "
            "Must be one of: %s"
            % (repn,
               str(sorted(registered_transforms.keys()))))
    assert transform is not None

    return transform(tri,
                     values,
                     input=input,
                     output=output,
                     bound=bound)

class _PiecewiseLinearFunctionND(tiny_block):
    """
    A piecewise linear function defined over a D-dimensional
    triangulation.
    """

    def __init__(self,
                 tri,
                 values,
                 input=None,
                 output=None):
        assert pyomo.core.base.component_piecewise.util.numpy_available
        assert pyomo.core.base.component_piecewise.util.scipy_available
        assert isinstance(tri,
                          pyomo.core.base.component_piecewise.\
                          util.scipy.spatial.Delaunay)
        assert isinstance(values,
                          pyomo.core.base.component_piecewise.\
                          util.numpy.ndarray)
        npoints, ndim = tri.points.shape
        nsimplices, _ = tri.simplices.shape
        assert tri.simplices.shape[1] == ndim + 1
        assert nsimplices > 0
        assert npoints > 0
        assert ndim > 0

        super(_PiecewiseLinearFunctionND, self).__init__()
        self._tri = tri
        self._values = values
        if input is None:
            input = [None]*ndim
        self._input = expression_list(
            expression(input[i]) for i in range(ndim))
        self._output = expression(output)

    @property
    def input(self):
        """Returns the list of expressions that store the
        inputs to the piecewise function. The returned
        objects can be updated by assigning to their .expr
        property."""
        return self._input

    @property
    def output(self):
        """Returns the expression that stores the output of
        the piecewise function. The returned object can be updated
        by assigning to its .expr property."""
        return self._output

    def __call__(self, x):
        """Evaluates the piecewise function using
        interpolation. This method supports vectorized
        function calls as the interpolation process can be
        expensive for high dimensional data.

        For the case when a single point is provided, the
        argument x should be a (D,) shaped numpy array or
        list, where D is the dimension of points in the
        triangulation.

        For the vectorized case, the argument x should be
        a (n, D) shaped array numpy array.
        """
        assert isinstance(x, collections.Sized)
        if isinstance(x, pyomo.core.base.component_piecewise.\
                      util.numpy.ndarray):
            if x.shape != self._tri.points.shape[1:]:
                multi = True
                assert x.shape[1:] == self._tri.points[0].shape, \
                    "%s[1] != %s" % (x.shape, self._tri.points[0].shape)
            else:
                multi = False
        else:
            multi = False
        ndim = len(self.input)
        i = self._tri.find_simplex(x)
        if multi:
            Tinv = self._tri.transform[i,:ndim]
            r = self._tri.transform[i,ndim]
            b = pyomo.core.base.component_piecewise.util.\
                numpy.einsum('ijk,ik->ij', Tinv, x-r)
            b = pyomo.core.base.component_piecewise.util.\
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

class piecewise_nd_cc(_PiecewiseLinearFunctionND):
    """
    Expresses a multivariate piecewise linear function using
    the CC formulation.
    """

    def __init__(self, *args, **kwds):
        bound = kwds.pop('bound', 'eq')
        super(piecewise_nd_cc, self).__init__(*args, **kwds)

        ndim = len(self.input)
        nsimplices = len(self._tri.simplices)
        npoints = len(self._tri.points)
        pointsT = list(zip(*self._tri.points))

        # create index objects
        dimensions = range(ndim)
        simplices = range(nsimplices)
        vertices = range(npoints)

        # create vars
        lmbda = self._lmbda = variable_list(
            variable(lb=0) for v in vertices)
        y = self._y = variable_list(
            variable(domain=Binary) for s in simplices)

        # create constraints
        lmbda_tuple = tuple(lmbda)

        self._c1 = constraint_list()
        for d in dimensions:
            self._c1.append(linear_constraint(
                variables=lmbda_tuple + (self.input[d],),
                coefficients=tuple(pointsT[d]) + (-1,),
                rhs=0))

        self._c2 = linear_constraint(
            variables=lmbda_tuple + (self.output,),
            coefficients=tuple(self._values) + (-1,))
        if bound == 'ub':
            self._c2.lb = 0
        elif bound == 'lb':
            self._c2.ub = 0
        elif bound == 'eq':
            self._c2.rhs = 0
        else:
            raise ValueError("Invalid bound type %r. Must be "
                             "one of: ['lb','ub','eq']"
                             % (bound))

        self._c3 = linear_constraint(
            variables=lmbda_tuple,
            coefficients=(1,)*len(lmbda_tuple),
            rhs=1)

        # generate a map from vertex index to simplex index,
        # which avoids an n^2 lookup when generating the
        # constraint
        vertex_to_simplex = [[] for v in vertices]
        for s, simplex in enumerate(self._tri.simplices):
            for v in simplex:
                vertex_to_simplex[v].append(s)

        self._c4 = constraint_list()
        for v in vertices:
            variables = tuple(y[s] for s in vertex_to_simplex[v])
            self._c4.append(linear_constraint(
                variables=variables + (lmbda[v],),
                coefficients=(1,)*len(variables) + (-1,),
                lb=0))

        self._c5 = linear_constraint(
            variables=y,
            coefficients=(1,)*len(y),
            rhs=1)

registered_transforms['cc'] = piecewise_nd_cc
