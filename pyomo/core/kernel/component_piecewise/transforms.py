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
representing a piecewise linear function using a
mixed-interger problem formulation.
"""

# ****** NOTE: Nothing in this file relies on integer division *******
#              I predict this will save numerous headaches as
#              well as gratuitous calls to float() in this code
from __future__ import division
import logging

# TODO: Figure out of the 'log' and 'dlog' representations
# really do require (2^n)+1 points or if there is a way to
# handle the between sizes.

__all__ = ("piecewise",)

from pyomo.core.kernel.numvalue import value
from pyomo.core.kernel.set_types import Binary
from pyomo.core.kernel.component_block import tiny_block
from pyomo.core.kernel.component_expression import expression
from pyomo.core.kernel.component_variable import (variable_list,
                                                  variable_dict,
                                                  variable)
from pyomo.core.kernel.component_constraint import (constraint,
                                                    constraint_list,
                                                    linear_constraint)
from pyomo.core.kernel.component_sos import sos2
from pyomo.core.kernel.component_piecewise.util import \
    (characterize_function,
     is_nondecreasing,
     is_positive_power_of_two,
     log2floor,
     generate_gray_code)

import six
from six.moves import xrange, zip

logger = logging.getLogger('pyomo.core')

registered_transforms = {}

#TODO: (simplify,
#       warning_domain_coverage,
#       unbounded_domain_var,
def piecewise(breakpoints,
              values,
              input=None,
              output=None,
              bound='eq',
              repn='sos2',
              validate=True,
              warning_tol=1e-8):
    """
    Transforms a list of breakpoints and values into a
    mixed-integer representation of a piecewise function.

    Args:
        breakpoints (list): The list of breakpoints of the
            piecewise linear function. This can be a list of
            numbers or a list of objects that store mutable
            data (e.g., mutable parameters). If mutable data
            is used validation might need to be delayed
            using the 'validate' keyword. The breakpoints
            list must be in non-decreasing order.
        values (list): The list of values of the piecewise linear
            function at each of the breakpoints. This list
            must be the same length as the breakpoints
            argument.
        input: The variable constrained to be the input of
            the piecewise linear function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound on the output to
            generate. Can be one of:
                - 'lb': y <= f(x)
                - 'eq': y  = f(x)
                - 'ub': y >= f(x)
        repn (str): The type of piecewise representation to
            use. Can be one of:
                - 'sos2': standard representation using sos2 constraints (+)
                -  'dcc': disaggregated convex combination (*+)
                - 'dlog': logarithmic disaggregated convex combination (*+)
                -   'cc': convex combination (*+)
                -  'log': logarithmic branching convex combination (*+)
                -   'mc': multiple choice (*)
                -  'inc': incremental method (*+)
           + supports step functions
           * source: "Mixed-Integer Models for Non-separable
                      Piecewise Linear Optimization:
                      Unifying framework and Extensions"
                      (Vielma, Nemhauser 2008)
        validate (bool): Indicates whether or not to perform
            validation of the data in the breakpoints and
            values lists. The default is True. Validation
            can be performed manually after the piecewise
            object is created by calling the validate()
            method. Validation should be performed anytime
            the data in the breakpoints or values lists
            changes (e.g., when using mutable parameters)
        warning_tol (float): Passed to the validate method
            when validation is performed to control the
            tolerance for checking consecutive slopes.

    Returns:
        A block containing the additional variables and
        constraints that enforce the piecewise linear
        relationship between the input and output variable.
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

    return transform(breakpoints,
                     values,
                     input=input,
                     output=output,
                     bound=bound,
                     validate=validate)

class _PiecewiseLinearFunction(tiny_block):
    """
    A piecewise linear function defined by a list of
    breakpoints and values.
    """

    def __init__(self,
                 breakpoints,
                 values,
                 input=None,
                 output=None,
                 validate=True):
        super(_PiecewiseLinearFunction, self).__init__()
        self._input = expression(input)
        self._output = expression(output)
        self._breakpoints = breakpoints
        self._values = values
        if type(self._breakpoints) is not tuple:
            self._breakpoints = tuple(self._breakpoints)
        if type(self._values) is not tuple:
            self._values = tuple(self._values)
        if len(self._breakpoints) != len(self._values):
            raise ValueError(
                "The number of breakpoints (%s) differs from "
                "the number of function values (%s)"
                % (len(self._breakpoints), len(self._values)))
        if validate:
            self.validate()

    def validate(self, warning_tol=1e-8):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints and values
        lists, including that the list of breakpoints is
        nondecreasing.

        Args:
            warning_tol (float): Tolerance used when
                generating warnings about consecutive slopes
                of the piecewise function being nearly
                equal.

        Returns:
            - 1: indicates that the function is affine
            - 2: indicates that the function is convex
            - 3: indicates that the function is concave
            - 4: indicates that the function has one or
                 more steps
            - 5: none of the above
        """
        breakpoints = [value(x) for x in self._breakpoints]
        values = [value(x) for x in self._values]
        if not is_nondecreasing(breakpoints):
            raise AssertionError(
                "The list of breakpoints is not nondecreasing: %s"
                % (str(breakpoints)))

        type_, slopes = characterize_function(breakpoints, values)
        for i in xrange(1, len(slopes)):
            if (slopes[i-1] is not None) and \
               (slopes[i] is not None) and \
               (abs(slopes[i-1] - slopes[i]) <= warning_tol):
                logger.warning(
                    "Piecewise linear validation detected slopes "
                    "of consecutive line segments to be within %s "
                    "of one another. This may cause numerical issues. "
                    "To disable this warning, increase the "
                    "warning tolerance." % (warning_tol))

        return type_

    @property
    def input(self):
        """Returns the expression that stores the input to
        the piecewise function. The returned object can be updated
        by assigning to its .expr property."""
        return self._input

    @property
    def output(self):
        """Returns the expression that stores the output of
        the piecewise function. The returned object can be updated
        by assigning to its .expr property."""
        return self._output

    @property
    def breakpoints(self):
        """The set of breakpoints used to defined this function"""
        return self._breakpoints

    @property
    def values(self):
        """The set of values used to defined this function"""
        return self._values

    def __call__(self, x):
        """Evaluates the piecewise function using interpolation"""
        # Note: One could implement binary search here to
        #       speed this up. I don't see this
        #       functionality being used very often (and the
        #       list of breakpoints probably isn't too
        #       large), so I'm doing it the easy way for
        #       now.
        for i in xrange(len(self.breakpoints)-1):
            xL = value(self.breakpoints[i])
            xU = value(self.breakpoints[i+1])
            assert xL <= xU
            if (xL <= x) and (x <= xU):
                yL = value(self.values[i])
                if xL == xU: # a step function
                    return yL
                yU = value(self.values[i+1])
                return yL + (float(yU-yL)/(xU-xL))*(x-xL)
        raise ValueError("The point %s is outside of the "
                         "function domain: [%s,%s]."
                         % (x,
                            value(self.breakpoints[0]),
                            value(self.breakpoints[-1])))

class piecewise_sos2(_PiecewiseLinearFunction):
    """
    Expresses a piecewise linear function using
    the SOS2 formulation.
    """

    def __init__(self, *args, **kwds):
        bound = kwds.pop('bound', 'eq')
        super(piecewise_sos2, self).__init__(*args, **kwds)

        # create vars
        y = self._y = variable_list(
            variable(lb=0) for i in xrange(len(self.breakpoints)))
        y_tuple = tuple(y)

        # create piecewise constraints
        self._c1 = linear_constraint(
             variables=y_tuple + (self.input,),
             coefficients=self.breakpoints + (-1,),
             rhs=0)

        self._c2 = linear_constraint(
            variables=y_tuple + (self.output,),
            coefficients=self.values + (-1,))
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

        self._c3 = linear_constraint(variables=y_tuple,
                                     coefficients=(1,)*len(y),
                                     rhs=1)
        self._c4 = sos2(y)

    def validate(self, **kwds):
        return super(piecewise_sos2, self).validate(**kwds)

registered_transforms['sos2'] = piecewise_sos2

class piecewise_dcc(_PiecewiseLinearFunction):
    """
    Expresses a piecewise linear function using
    the DCC formulation.
    """

    def __init__(self, *args, **kwds):
        bound = kwds.pop('bound', 'eq')
        super(piecewise_dcc, self).__init__(*args, **kwds)

        # create index sets
        polytopes = range(len(self.breakpoints)-1)
        vertices = range(len(self.breakpoints))
        def polytope_verts(p):
            return xrange(p,p+2)

        # create vars
        lmbda = self._lmbda = variable_dict(
            ((p,v), variable(lb=0))
            for p in polytopes
            for v in vertices)
        y = self._y = variable_list(
            variable(domain=Binary)
            for p in polytopes)

        # create piecewise constraints
        self._c1 = linear_constraint(
            variables=tuple(lmbda[p,v]
                            for p in polytopes
                            for v in polytope_verts(p)) + \
                      (self.input,),
            coefficients=tuple(self.breakpoints[v]
                               for p in polytopes
                               for v in polytope_verts(p)) + (-1,),
            rhs=0)

        self._c2 = linear_constraint(
            variables=tuple(lmbda[p,v]
                            for p in polytopes
                            for v in polytope_verts(p)) + \
                      (self.output,),
            coefficients=tuple(self.values[v]
                               for p in polytopes
                               for v in polytope_verts(p)) + (-1,))
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

        self._c3 = constraint_list()
        for p in polytopes:
            variables = tuple(lmbda[p,v] for v in polytope_verts(p))
            self._c3.append(
                linear_constraint(
                    variables=variables + (y[p],),
                    coefficients=(1,)*len(variables) + (-1,),
                    rhs=0))

        self._c4 = linear_constraint(
            variables=tuple(y),
            coefficients=(1,)*len(y),
            rhs=1)

    def validate(self, **kwds):
        return super(piecewise_dcc, self).validate(**kwds)

registered_transforms['dcc'] = piecewise_dcc

class piecewise_cc(_PiecewiseLinearFunction):
    """
    Expresses a piecewise linear function using
    the CC formulation.
    """

    def __init__(self, *args, **kwds):
        bound = kwds.pop('bound', 'eq')
        super(piecewise_cc, self).__init__(*args, **kwds)

        # create index sets
        polytopes = range(len(self.breakpoints)-1)
        vertices = range(len(self.breakpoints))
        def vertex_polys(v):
            if v == 0:
                return [v]
            if v == len(self.breakpoints)-1:
                return [v-1]
            else:
                return [v-1,v]

        # create vars
        lmbda = self._lmbda = variable_list(
            variable(lb=0) for v in vertices)
        y = self._y = variable_list(
            variable(domain=Binary)
            for p in polytopes)

        lmbda_tuple = tuple(lmbda)

        # create piecewise constraints
        self._c1 = linear_constraint(
            variables=lmbda_tuple + (self.input,),
            coefficients=self.breakpoints + (-1,),
            rhs=0)

        self._c2 = linear_constraint(
            variables=lmbda_tuple + (self.output,),
            coefficients=self.values + (-1,))
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
            coefficients=(1,)*len(lmbda),
            rhs=1)

        self._c4 = constraint_list()
        for v in vertices:
            variables = tuple(y[p] for p in vertex_polys(v))
            self._c4.append(linear_constraint(
                variables=variables + (lmbda[v],),
                coefficients=(1,)*len(variables) + (-1,),
                lb=0))

        self._c5 = linear_constraint(
            variables=tuple(y),
            coefficients=(1,)*len(y),
            rhs=1)

    def validate(self, **kwds):
        return super(piecewise_cc, self).validate(**kwds)

registered_transforms['cc'] = piecewise_cc

class piecewise_mc(_PiecewiseLinearFunction):
    """
    Expresses a piecewise linear function using
    the MC formulation.
    """

    def __init__(self, *args, **kwds):
        bound = kwds.pop('bound', 'eq')
        super(piecewise_mc, self).__init__(*args, **kwds)

        # create indexers
        polytopes = range(len(self.breakpoints)-1)

        # create constants (using future division)
        # these might also be expressions if the breakpoints
        # or values lists contain mutable objects
        slopes = tuple((self.values[p+1] - self.values[p]) / \
                       (self.breakpoints[p+1] - self.breakpoints[p])
                       for p in polytopes)
        intercepts = tuple(self.values[p] - \
                           (slopes[p] * self.breakpoints[p])
                           for p in polytopes)

        # create vars
        lmbda = self._lmbda = variable_list(variable()
                                    for p in polytopes)
        lmbda_tuple = tuple(lmbda)
        y = self._y = variable_list(variable(domain=Binary)
                                    for p in polytopes)
        y_tuple = tuple(y)

        # create piecewise constraints
        self._c1 = linear_constraint(
            variables=lmbda_tuple + (self.input,),
            coefficients=(1,)*len(lmbda) + (-1,),
            rhs=0)

        self._c2 = linear_constraint(
            variables=lmbda_tuple + y_tuple + (self.output,),
            coefficients=slopes + intercepts + (-1,))
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

        self._c3 = constraint_list()
        self._c4 = constraint_list()
        for p in polytopes:
            self._c3.append(linear_constraint(
                variables=(y[p], lmbda[p]),
                coefficients=(self.breakpoints[p], -1),
                ub=0))
            self._c4.append(linear_constraint(
                variables=(lmbda[p], y[p]),
                coefficients=(1, -self.breakpoints[p+1]),
                ub=0))

        self._c5 = linear_constraint(
            variables=y_tuple,
            coefficients=(1,)*len(y),
            rhs=1)

    def validate(self, **kwds):
        type_ = super(piecewise_mc, self).validate(**kwds)
        # this representation does not support step functions
        if type_ == 4:
            raise AssertionError(
                "The MC piecewise representation does "
                "not support step functions.")
        return type_

registered_transforms['mc'] = piecewise_mc

class piecewise_inc(_PiecewiseLinearFunction):
    """
    Expresses a piecewise linear function using
    the INC formulation.
    """

    def __init__(self, *args, **kwds):
        bound = kwds.pop('bound', 'eq')
        super(piecewise_inc, self).__init__(*args, **kwds)

        # create indexers
        polytopes = range(len(self.breakpoints)-1)

        # create vars
        delta = self._delta = variable_list(
            variable() for p in polytopes)
        delta[0].ub = 1
        delta[-1].lb = 0
        delta_tuple = tuple(delta)
        y = self._y = variable_list(
            variable(domain=Binary) for p in polytopes[:-1])

        # create piecewise constraints
        self._c1 = linear_constraint(
            variables=(self.input,) + delta_tuple,
            coefficients=(-1,) + tuple(self.breakpoints[p+1] - \
                                       self.breakpoints[p]
                                       for p in polytopes),
            rhs=-self.breakpoints[0])

        self._c2 = linear_constraint(
            variables=(self.output,) + delta_tuple,
            coefficients=(-1,) + tuple(self.values[p+1] - \
                                       self.values[p]
                                       for p in polytopes))
        if bound == 'ub':
            self._c2.lb = -self.values[0]
        elif bound == 'lb':
            self._c2.ub = -self.values[0]
        elif bound == 'eq':
            self._c2.rhs = -self.values[0]
        else:
            raise ValueError("Invalid bound type %r. Must be "
                             "one of: ['lb','ub','eq']"
                             % (bound))

        self._c3 = constraint_list()
        self._c4 = constraint_list()
        for p in polytopes[:-1]:
            self._c3.append(linear_constraint(
                variables=(delta[p+1], y[p]),
                coefficients=(1, -1),
                ub=0))
            self._c4.append(linear_constraint(
                variables=(y[p], delta[p]),
                coefficients=(1, -1),
                ub=0))

    def validate(self, **kwds):
        return super(piecewise_inc, self).validate(**kwds)

registered_transforms['inc'] = piecewise_inc

class piecewise_dlog(_PiecewiseLinearFunction):
    """
    Expresses a piecewise linear function using
    the DLOG formulation.
    """

    def __init__(self, *args, **kwds):
        bound = kwds.pop('bound', 'eq')
        super(piecewise_dlog, self).__init__(*args, **kwds)

        if not is_positive_power_of_two(len(self.breakpoints)-1):
            raise ValueError("The list of breakpoints must be "
                             "of length (2^n)+1 for some positive "
                             "integer n. Invalid length: %s"
                             % (len(self.breakpoints)))

        # create branching schemes
        L = log2floor(len(self.breakpoints)-1)
        assert 2**L == len(self.breakpoints)-1
        B_LEFT, B_RIGHT = self._branching_scheme(L)

        # create indexers
        polytopes = range(len(self.breakpoints)-1)
        vertices = range(len(self.breakpoints))
        def polytope_verts(p):
            return xrange(p,p+2)

        # create vars
        lmbda = self._lmbda = variable_dict(
            ((p,v), variable(lb=0))
            for p in polytopes
            for v in polytope_verts(p))
        y = self._y = variable_list(
            variable(domain=Binary)
            for i in range(L))

        # create piecewise constraints
        self._c1 = linear_constraint(
            variables=(self.input,) + tuple(lmbda[p,v]
                                            for p in polytopes
                                            for v in polytope_verts(p)),
            coefficients=(-1,) + tuple(self.breakpoints[v]
                                       for p in polytopes
                                       for v in polytope_verts(p)),
            rhs=0)

        self._c2 = linear_constraint(
            variables=(self.output,) + tuple(lmbda[p,v]
                                             for p in polytopes
                                             for v in polytope_verts(p)),
            coefficients=(-1,) + tuple(self.values[v]
                                       for p in polytopes
                                       for v in polytope_verts(p)))
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
            variables=tuple(lmbda.values()),
            coefficients=(1,)*len(lmbda),
            rhs=1)

        self._c4 = constraint_list()
        for i in range(L):
            variables = tuple(lmbda[p,v]
                              for p in B_LEFT[i]
                              for v in polytope_verts(p))
            self._c4.append(linear_constraint(
                variables=variables + (y[i],),
                coefficients=(1,)*len(variables) + (-1,),
                ub=0))

        self._c5 = constraint_list()
        for i in range(L):
            variables = tuple(lmbda[p,v]
                              for p in B_RIGHT[i]
                              for v in polytope_verts(p))
            self._c5.append(linear_constraint(
                variables=variables + (y[i],),
                coefficients=(1,)*len(variables) + (1,),
                ub=1))

    def _branching_scheme(self, L):
        N = 2**L
        B_LEFT = []
        for i in range(1,L+1):
            start = 1
            step = N//(2**i)
            tmp = []
            while start < N:
                tmp.extend(j-1 for j in xrange(start,start+step))
                start += 2*step
            B_LEFT.append(tmp)

        biglist = range(N)
        B_RIGHT = []
        for i in range(len(B_LEFT)):
            tmp = []
            for j in biglist:
                if j not in B_LEFT[i]:
                    tmp.append(j)
            B_RIGHT.append(sorted(tmp))

        return B_LEFT, B_RIGHT

    def validate(self, **kwds):
        return super(piecewise_dlog, self).validate(**kwds)

registered_transforms['dlog'] = piecewise_dlog

class piecewise_log(_PiecewiseLinearFunction):
    """
    Expresses a piecewise linear function using
    the LOG formulation.
    """

    def __init__(self, *args, **kwds):
        bound = kwds.pop('bound', 'eq')
        super(piecewise_log, self).__init__(*args, **kwds)

        if not is_positive_power_of_two(len(self.breakpoints)-1):
            raise ValueError("The list of breakpoints must be "
                             "of length (2^n)+1 for some positive "
                             "integer n. Invalid length: %s"
                             % (len(self.breakpoints)))

        # create branching schemes
        L = log2floor(len(self.breakpoints)-1)
        S,B_LEFT,B_RIGHT = self._branching_scheme(L)

        # create indexers
        polytopes = range(len(self.breakpoints) - 1)
        vertices = range(len(self.breakpoints))

        # create vars
        lmbda = self._lmbda = variable_list(
            variable(lb=0)
            for v in vertices)
        y = self._y = variable_list(variable(domain=Binary)
                                    for s in S)

        # create piecewise constraints
        self._c1 = linear_constraint(
            variables=(self.input,) + tuple(lmbda),
            coefficients=(-1,) + self.breakpoints,
            rhs=0)

        self._c2 = linear_constraint(
            variables=(self.output,) + tuple(lmbda),
            coefficients=(-1,) + self.values)
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
            variables=tuple(lmbda),
            coefficients=(1,)*len(lmbda),
            rhs=1)

        self._c4 = constraint_list()
        for s in S:
            variables=tuple(lmbda[v] for v in B_LEFT[s])
            self._c4.append(linear_constraint(
                variables=variables + (y[s],),
                coefficients=(1,)*len(variables) + (-1,),
                ub=0))

        self._c5 = constraint_list()
        for s in S:
            variables=tuple(lmbda[v] for v in B_RIGHT[s])
            self._c5.append(linear_constraint(
                variables=variables + (y[s],),
                coefficients=(1,)*len(variables) + (1,),
                ub=1))

    def _branching_scheme(self, n):
        N = 2**n
        S = range(n)
        G = generate_gray_code(n)
        L = tuple([k for k in xrange(N+1)
                   if ((k == 0) or (G[k-1][s] == 1))
                   and ((k == N) or (G[k][s] == 1))] for s in S)
        R = tuple([k for k in xrange(N+1)
                   if ((k == 0) or (G[k-1][s] == 0))
                   and ((k == N) or (G[k][s] == 0))] for s in S)
        return S, L, R

    def validate(self, **kwds):
        return super(piecewise_log, self).validate(**kwds)

registered_transforms['log'] = piecewise_log
