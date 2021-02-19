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
single-variate piecewise linear function using a
mixed-interger problem formulation. Reference::

  Mixed-Integer Models for Non-separable Piecewise Linear \
Optimization: Unifying framework and Extensions (Vielma, \
Nemhauser 2008)
"""

# ****** NOTE: Nothing in this file relies on integer division *******
#              I predict this will save numerous headaches as
#              well as gratuitous calls to float() in this code
from __future__ import division
import logging
import bisect

# TODO: Figure out of the 'log' and 'dlog' representations
# really do require (2^n)+1 points or if there is a way to
# handle the between sizes.

from pyomo.core.expr.numvalue import value as _value
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.block import block
from pyomo.core.kernel.expression import (expression,
                                          expression_tuple)
from pyomo.core.kernel.variable import (IVariable,
                                        variable_list,
                                        variable_tuple,
                                        variable_dict,
                                        variable)
from pyomo.core.kernel.constraint import (constraint_list,
                                          constraint_tuple,
                                          linear_constraint)
from pyomo.core.kernel.sos import sos2
from pyomo.core.kernel.piecewise_library.util import \
    (characterize_function,
     is_nondecreasing,
     is_positive_power_of_two,
     log2floor,
     generate_gray_code,
     PiecewiseValidationError)

from six.moves import xrange

logger = logging.getLogger('pyomo.core')

registered_transforms = {}

# wrapper that allows a list containing parameters to be
# used with the bisect module
class _shadow_list(object):
    __slots__ = ("_x",)
    def __init__(self, x):
        self._x = x
    def __len__(self):
        return self._x.__len__()
    def __getitem__(self, i):
        return _value(self._x.__getitem__(i))

def piecewise(breakpoints,
              values,
              input=None,
              output=None,
              bound='eq',
              repn='sos2',
              validate=True,
              simplify=True,
              equal_slopes_tolerance=1e-6,
              require_bounded_input_variable=True,
              require_variable_domain_coverage=True):
    """
    Models a single-variate piecewise linear function.

    This function takes a list breakpoints and function
    values describing a piecewise linear function and
    transforms this input data into a block of variables and
    constraints that enforce a piecewise linear relationship
    between an input variable and an output variable. In the
    general case, this transformation requires the use of
    discrete decision variables.

    Args:
        breakpoints (list): The list of breakpoints of the
            piecewise linear function. This can be a list of
            numbers or a list of objects that store mutable
            data (e.g., mutable parameters). If mutable data
            is used validation might need to be disabled by
            setting the :attr:`validate` keyword to
            :const:`False`. The list of breakpoints must be
            in non-decreasing order.
        values (list): The values of the piecewise linear
            function corresponding to the breakpoints.
        input: The variable constrained to be the input of
            the piecewise linear function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound to impose on the
            output expression. Can be one of:

              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
        repn (str): The type of piecewise representation to
            use. Choices are shown below (+ means step
            functions are supported)

                - 'sos2': standard representation using sos2 constraints (+)
                -  'dcc': disaggregated convex combination (+)
                - 'dlog': logarithmic disaggregated convex combination (+)
                -   'cc': convex combination (+)
                -  'log': logarithmic branching convex combination (+)
                -   'mc': multiple choice
                -  'inc': incremental method (+)
        validate (bool): Indicates whether or not to perform
            validation of the input data. The default is
            :const:`True`. Validation can be performed
            manually after the piecewise object is created
            by calling the :meth:`validate`
            method. Validation should be performed any time
            the inputs are changed (e.g., when using mutable
            parameters in the breakpoints list or when the
            input variable changes).
        simplify (bool): Indicates whether or not to attempt
            to simplify the piecewise representation to
            avoid using discrete variables. This can be done
            when the feasible region for the output
            variable, with respect to the piecewise function
            and the bound type, is a convex set. Default is
            :const:`True`. Validation is required to perform
            simplification, so this keyword is ignored when
            the :attr:`validate` keyword is :attr:`False`.
        equal_slopes_tolerance (float): Tolerance used check
            if consecutive slopes are nearly equal. If any
            are found, validation will fail. Default is
            1e-6. This keyword is ignored when the
            :attr:`validate` keyword is :attr:`False`.
        require_bounded_input_variable (bool): Indicates if
            the input variable is required to have finite
            upper and lower bounds. Default is
            :const:`True`. Setting this keyword to
            :const:`False` can be used to allow general
            expressions to be used as the input in place of
            a variable. This keyword is ignored when the
            :attr:`validate` keyword is :attr:`False`.
        require_variable_domain_coverage (bool): Indicates
            if the function domain (defined by the endpoints
            of the breakpoints list) needs to cover the
            entire domain of the input variable. Default is
            :const:`True`. Ignored for any bounds of
            variables that are not finite, or when the input
            is not assigned a variable. This keyword is
            ignored when the :attr:`validate` keyword is
            :attr:`False`.

    Returns:
        TransformedPiecewiseLinearFunction: a block that \
            stores any new variables, constraints, and other \
            modeling objects used by the piecewise representation
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

    if not validate:
        # can not simplify if we do not validate
        simplify = False

    func = PiecewiseLinearFunction(breakpoints,
                                   values,
                                   validate=False)

    if simplify and \
       (transform is not piecewise_convex):
        ftype = func.validate(
            equal_slopes_tolerance=equal_slopes_tolerance)

        if (bound == 'eq') and \
           (ftype == characterize_function.affine):
            transform = piecewise_convex
        elif (bound == 'lb') and \
             (ftype in (characterize_function.affine,
                        characterize_function.convex)):
            transform = piecewise_convex
        elif (bound == 'ub') and \
             (ftype in (characterize_function.affine,
                        characterize_function.concave)):
            transform = piecewise_convex

    return transform(func,
                     input=input,
                     output=output,
                     bound=bound,
                     validate=validate,
                     equal_slopes_tolerance=\
                         equal_slopes_tolerance,
                     require_bounded_input_variable=\
                         require_bounded_input_variable,
                     require_variable_domain_coverage=\
                         require_variable_domain_coverage)

class PiecewiseLinearFunction(object):
    """A piecewise linear function

    Piecewise linear functions are defined by a list of
    breakpoints and a list function values corresponding to
    each breakpoint. The function value between breakpoints
    is implied through linear interpolation.

    Args:
        breakpoints (list): The list of function
            breakpoints.
        values (list): The list of function values (one for
            each breakpoint).
        validate (bool): Indicates whether or not to perform
            validation of the input data. The default is
            :const:`True`. Validation can be performed
            manually after the piecewise object is created
            by calling the :meth:`validate`
            method. Validation should be performed any time
            the inputs are changed (e.g., when using mutable
            parameters in the breakpoints list).
        **kwds: Additional keywords are passed to the
            :meth:`validate` method when the :attr:`validate`
            keyword is :const:`True`; otherwise, they are
            ignored.
    """
    __slots__ = ("_breakpoints", "_values")

    def __init__(self,
                 breakpoints,
                 values,
                 validate=True,
                 **kwds):
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
            self.validate(**kwds)

    def __getstate__(self):
        """Required for older versions of the pickle
        protocol since this class uses __slots__"""
        return {key:getattr(self, key) for key in self.__slots__}

    def __setstate__(self, state):
        """Required for older versions of the pickle
        protocol since this class uses __slots__"""
        for key in state:
            setattr(self, key, state[key])

    def validate(self,
                 equal_slopes_tolerance=1e-6):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints and values
        lists (e.g., that the list of breakpoints is
        nondecreasing).

        Args:
            equal_slopes_tolerance (float): Tolerance used
                check if consecutive slopes are nearly
                equal. If any are found, validation will
                fail. Default is 1e-6.

        Returns:
            int:
                a function characterization code (see \
                :func:`util.characterize_function`)

        Raises:
            PiecewiseValidationError: if validation fails
        """

        breakpoints = [_value(x) for x in self._breakpoints]
        values = [_value(x) for x in self._values]
        if not is_nondecreasing(breakpoints):
            raise PiecewiseValidationError(
                "The list of breakpoints is not nondecreasing: %s"
                % (str(breakpoints)))

        ftype, slopes = characterize_function(breakpoints, values)
        for i in xrange(1, len(slopes)):
            if (slopes[i-1] is not None) and \
               (slopes[i] is not None) and \
               (abs(slopes[i-1] - slopes[i]) <= equal_slopes_tolerance):
                raise PiecewiseValidationError(
                    "Piecewise function validation detected slopes "
                    "of consecutive line segments to be within %s "
                    "of one another. This may cause numerical issues. "
                    "To avoid this error, set the 'equal_slopes_tolerance' "
                    "keyword to a smaller value or disable validation."
                    % (equal_slopes_tolerance))

        return ftype

    @property
    def breakpoints(self):
        """The set of breakpoints used to defined this function"""
        return self._breakpoints

    @property
    def values(self):
        """The set of values used to defined this function"""
        return self._values

    def __call__(self, x):
        """Evaluates the piecewise linear function at the
        given point using interpolation. Note that step functions are
        assumed lower-semicontinuous."""
        i = bisect.bisect_left(_shadow_list(self.breakpoints), x)
        if i == 0:
            xP = _value(self.breakpoints[i])
            if xP == x:
                return float(_value(self.values[i]))
        elif i != len(self.breakpoints):
            xL = _value(self.breakpoints[i-1])
            xU = _value(self.breakpoints[i])
            assert xL <= xU
            if (xL <= x) and (x <= xU):
                yL = _value(self.values[i-1])
                yU = _value(self.values[i])
                return yL + (float(yU-yL)/(xU-xL))*(x-xL)
        raise ValueError("The point %s is outside of the "
                         "function domain: [%s,%s]."
                         % (x,
                            _value(self.breakpoints[0]),
                            _value(self.breakpoints[-1])))

class TransformedPiecewiseLinearFunction(block):
    """Base class for transformed piecewise linear functions

    A transformed piecewise linear functions is a block of
    variables and constraints that enforce a piecewise
    linear relationship between an input variable and an
    output variable.

    Args:
        f (:class:`PiecewiseLinearFunction`): The piecewise
            linear function to transform.
        input: The variable constrained to be the input of
            the piecewise linear function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound to impose on the
            output expression. Can be one of:

              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
        validate (bool): Indicates whether or not to perform
            validation of the input data. The default is
            :const:`True`. Validation can be performed
            manually after the piecewise object is created
            by calling the :meth:`validate`
            method. Validation should be performed any time
            the inputs are changed (e.g., when using mutable
            parameters in the breakpoints list or when the
            input variable changes).
        **kwds: Additional keywords are passed to the
            :meth:`validate` method when the :attr:`validate`
            keyword is :const:`True`; otherwise, they are
            ignored.
    """

    def __init__(self,
                 f,
                 input=None,
                 output=None,
                 bound='eq',
                 validate=True,
                 **kwds):
        super(TransformedPiecewiseLinearFunction, self).__init__()
        assert isinstance(f, PiecewiseLinearFunction)
        if bound not in ('lb', 'ub', 'eq'):
            raise ValueError("Invalid bound type %r. Must be "
                             "one of: ['lb','ub','eq']"
                             % (bound))
        self._bound = bound
        self._f = f
        self._inout = expression_tuple([expression(input),
                                        expression(output)])
        if validate:
            self.validate(**kwds)

    @property
    def input(self):
        """The expression that stores the input to the
        piecewise function. The returned object can be
        updated by assigning to its :attr:`expr`
        attribute."""
        return self._inout[0]

    @property
    def output(self):
        """The expression that stores the output of the
        piecewise function. The returned object can be
        updated by assigning to its :attr:`expr`
        attribute."""
        return self._inout[1]

    @property
    def bound(self):
        """The bound type assigned to the piecewise
        relationship ('lb','ub','eq')."""
        return self._bound

    def validate(self,
                 equal_slopes_tolerance=1e-6,
                 require_bounded_input_variable=True,
                 require_variable_domain_coverage=True):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        Args:
            equal_slopes_tolerance (float): Tolerance used
                check if consecutive slopes are nearly
                equal. If any are found, validation will
                fail. Default is 1e-6.
            require_bounded_input_variable (bool): Indicates
                if the input variable is required to have
                finite upper and lower bounds. Default is
                :const:`True`. Setting this keyword to
                :const:`False` can be used to allow general
                expressions to be used as the input in place
                of a variable.
            require_variable_domain_coverage (bool):
                Indicates if the function domain (defined by
                the endpoints of the breakpoints list) needs
                to cover the entire domain of the input
                variable. Default is :const:`True`. Ignored
                for any bounds of variables that are not
                finite, or when the input is not assigned a
                variable.

        Returns:
            int:
                a function characterization code (see \
                :func:`util.characterize_function`)

        Raises:
            PiecewiseValidationError: if validation fails
        """
        ftype = self._f.validate(
            equal_slopes_tolerance=equal_slopes_tolerance)
        assert ftype in (1,2,3,4,5)

        input_var = self.input.expr
        if not isinstance(input_var, IVariable):
            input_var = None

        if require_bounded_input_variable and \
           ((input_var is None) or \
            (not input_var.has_lb()) or \
            (not input_var.has_ub())):
                raise PiecewiseValidationError(
                    "Piecewise function input is not a "
                    "variable with finite upper and lower "
                    "bounds: %s. To avoid this error, set the "
                    "'require_bounded_input_variable' keyword "
                    "to False or disable validation."
                    % (str(input_var)))

        if require_variable_domain_coverage and \
           (input_var is not None):
            domain_lb = _value(self.breakpoints[0])
            domain_ub = _value(self.breakpoints[-1])
            if input_var.has_lb() and \
               _value(input_var.lb) < domain_lb:
                raise PiecewiseValidationError(
                    "Piecewise function domain does not include "
                    "the lower bound of the input variable: "
                    "%s.ub = %s > %s. To avoid this error, set "
                    "the 'require_variable_domain_coverage' "
                    "keyword to False or disable validation."
                    % (input_var.name,
                       _value(input_var.lb),
                       domain_lb))
            if input_var.has_ub() and \
               _value(input_var.ub) > domain_ub:
                raise PiecewiseValidationError(
                    "Piecewise function domain does not include "
                    "the upper bound of the input variable: "
                    "%s.ub = %s > %s. To avoid this error, set "
                    "the 'require_variable_domain_coverage' "
                    "keyword to False or disable validation."
                    % (input_var.name,
                       _value(input_var.ub),
                       domain_ub))

        return ftype

    @property
    def breakpoints(self):
        """The set of breakpoints used to defined this function"""
        return self._f.breakpoints

    @property
    def values(self):
        """The set of values used to defined this function"""
        return self._f.values

    def __call__(self, x):
        """Evaluates the piecewise linear function at the
        given point using interpolation"""
        return self._f(x)

class piecewise_convex(TransformedPiecewiseLinearFunction):
    """Simple convex piecewise representation

    Expresses a piecewise linear function with a convex
    feasible region for the output variable using a simple
    collection of linear constraints.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_convex, self).__init__(*args, **kwds)

        breakpoints = self.breakpoints
        values = self.values
        self.c = constraint_list()
        for i in xrange(len(breakpoints)-1):
            X0 = breakpoints[i]
            F_AT_X0 = values[i]
            dF_AT_X0 = (values[i+1] - F_AT_X0) / \
                       (breakpoints[i+1] - X0)
            const = F_AT_X0 - dF_AT_X0*X0
            con = linear_constraint(
                (self.output, self.input),
                (-1, dF_AT_X0))
            if self.bound == 'ub':
                con.lb = -const
            elif self.bound == 'lb':
                con.ub = -const
            else:
                assert self.bound == 'eq'
                con.rhs = -const
            self.c.append(con)

        # In order to enforce the same behavior as actual
        # piecewise constraints, we need to constrain the
        # input expression to be between first and last
        # breakpoint. This might be duplicating the
        # variable, but its not always the case, and there's
        # no guarantee that the input "variable" is not a
        # more general linear expression.
        self.c.append(linear_constraint(
            terms=[(self.input, 1)],
            lb=self.breakpoints[0],
            ub=self.breakpoints[-1]))

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        ftype = super(piecewise_convex, self).validate(**kwds)
        if (self.bound == 'eq') and \
           (ftype != characterize_function.affine):
            raise PiecewiseValidationError(
                "The bound type is 'eq' but the function "
                "was not characterized as affine (only two "
                "breakpoints). The 'convex' piecewise "
                "representation does not support this function.")
        elif (self.bound == 'lb') and \
             (ftype not in (characterize_function.affine,
                            characterize_function.convex)):
            raise PiecewiseValidationError(
                "The bound type is 'lb' but the function "
                "was not characterized as convex or affine. "
                "The 'convex' piecewise representation does "
                "not support this function.")
        elif (self.bound == 'ub') and \
             (ftype not in (characterize_function.affine,
                            characterize_function.concave)):
            raise PiecewiseValidationError(
                "The bound type is 'ub' but the function "
                "was not characterized as concave or affine. "
                "The 'convex' piecewise representation does "
                "not support this function.")
        return ftype

registered_transforms['convex'] = piecewise_convex

class piecewise_sos2(TransformedPiecewiseLinearFunction):
    """Discrete SOS2 piecewise representation

    Expresses a piecewise linear function using
    the SOS2 formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_sos2, self).__init__(*args, **kwds)

        # create vars
        y_tuple = tuple(variable(lb=0)
                        for i in xrange(len(self.breakpoints)))
        y = self.v = variable_tuple(y_tuple)

        # create piecewise constraints
        self.c = constraint_list()

        self.c.append(linear_constraint(
            variables=y_tuple + (self.input,),
            coefficients=self.breakpoints + (-1,),
            rhs=0))

        self.c.append(linear_constraint(
            variables=y_tuple + (self.output,),
            coefficients=self.values + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0

        self.c.append(linear_constraint(variables=y_tuple,
                                        coefficients=(1,)*len(y),
                                        rhs=1))

        self.s = sos2(y)

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_sos2, self).validate(**kwds)

registered_transforms['sos2'] = piecewise_sos2

class piecewise_dcc(TransformedPiecewiseLinearFunction):
    """Discrete DCC piecewise representation

    Expresses a piecewise linear function using
    the DCC formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_dcc, self).__init__(*args, **kwds)

        # create index sets
        polytopes = range(len(self.breakpoints)-1)
        vertices = range(len(self.breakpoints))
        def polytope_verts(p):
            return xrange(p,p+2)

        # create vars
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_dict(
            ((p,v), variable(lb=0))
            for p in polytopes
            for v in vertices)
        y = self.v['y'] = variable_tuple(
            variable(domain_type=IntegerSet, lb=0, ub=1)
            for p in polytopes)

        # create piecewise constraints
        self.c = constraint_list()

        self.c.append(linear_constraint(
            variables=tuple(lmbda[p,v]
                            for p in polytopes
                            for v in polytope_verts(p)) + \
                      (self.input,),
            coefficients=tuple(self.breakpoints[v]
                               for p in polytopes
                               for v in polytope_verts(p)) + \
                      (-1,),
            rhs=0))

        self.c.append(linear_constraint(
            variables=tuple(lmbda[p,v]
                            for p in polytopes
                            for v in polytope_verts(p)) + \
                      (self.output,),
            coefficients=tuple(self.values[v]
                               for p in polytopes
                               for v in polytope_verts(p)) + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0

        clist = []
        for p in polytopes:
            variables = tuple(lmbda[p,v] for v in polytope_verts(p))
            clist.append(
                linear_constraint(
                    variables=variables + (y[p],),
                    coefficients=(1,)*len(variables) + (-1,),
                    rhs=0))
        self.c.append(constraint_tuple(clist))

        self.c.append(linear_constraint(
            variables=tuple(y),
            coefficients=(1,)*len(y),
            rhs=1))

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_dcc, self).validate(**kwds)

registered_transforms['dcc'] = piecewise_dcc

class piecewise_cc(TransformedPiecewiseLinearFunction):
    """Discrete CC piecewise representation

    Expresses a piecewise linear function using
    the CC formulation.
    """

    def __init__(self, *args, **kwds):
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
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_tuple(
            variable(lb=0) for v in vertices)
        y = self.v['y'] = variable_tuple(
            variable(domain_type=IntegerSet, lb=0, ub=1)
            for p in polytopes)

        lmbda_tuple = tuple(lmbda)

        # create piecewise constraints
        self.c = constraint_list()

        self.c.append(linear_constraint(
            variables=lmbda_tuple + (self.input,),
            coefficients=self.breakpoints + (-1,),
            rhs=0))

        self.c.append(linear_constraint(
            variables=lmbda_tuple + (self.output,),
            coefficients=self.values + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0

        self.c.append(linear_constraint(
            variables=lmbda_tuple,
            coefficients=(1,)*len(lmbda),
            rhs=1))

        clist = []
        for v in vertices:
            variables = tuple(y[p] for p in vertex_polys(v))
            clist.append(linear_constraint(
                variables=variables + (lmbda[v],),
                coefficients=(1,)*len(variables) + (-1,),
                lb=0))
        self.c.append(constraint_tuple(clist))

        self.c.append(linear_constraint(
            variables=tuple(y),
            coefficients=(1,)*len(y),
            rhs=1))

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_cc, self).validate(**kwds)

registered_transforms['cc'] = piecewise_cc

class piecewise_mc(TransformedPiecewiseLinearFunction):
    """Discrete MC piecewise representation

    Expresses a piecewise linear function using
    the MC formulation.
    """

    def __init__(self, *args, **kwds):
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
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_tuple(
            variable() for p in polytopes)
        lmbda_tuple = tuple(lmbda)
        y = self.v['y'] = variable_tuple(
            variable(domain_type=IntegerSet, lb=0, ub=1) for p in polytopes)
        y_tuple = tuple(y)

        # create piecewise constraints
        self.c = constraint_list()

        self.c.append(linear_constraint(
            variables=lmbda_tuple + (self.input,),
            coefficients=(1,)*len(lmbda) + (-1,),
            rhs=0))

        self.c.append(linear_constraint(
            variables=lmbda_tuple + y_tuple + (self.output,),
            coefficients=slopes + intercepts + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0

        clist1 = []
        clist2 = []
        for p in polytopes:
            clist1.append(linear_constraint(
                variables=(y[p], lmbda[p]),
                coefficients=(self.breakpoints[p], -1),
                ub=0))
            clist2.append(linear_constraint(
                variables=(lmbda[p], y[p]),
                coefficients=(1, -self.breakpoints[p+1]),
                ub=0))
        self.c.append(constraint_tuple(clist1))
        self.c.append(constraint_tuple(clist2))

        self.c.append(linear_constraint(
            variables=y_tuple,
            coefficients=(1,)*len(y),
            rhs=1))

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        ftype = super(piecewise_mc, self).validate(**kwds)
        # this representation does not support step functions
        if ftype == characterize_function.step:
            raise PiecewiseValidationError(
                "The 'mc' piecewise representation does "
                "not support step functions.")
        return ftype

registered_transforms['mc'] = piecewise_mc

class piecewise_inc(TransformedPiecewiseLinearFunction):
    """Discrete INC piecewise representation

    Expresses a piecewise linear function using
    the INC formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_inc, self).__init__(*args, **kwds)

        # create indexers
        polytopes = range(len(self.breakpoints)-1)

        # create vars
        self.v = variable_dict()
        delta = self.v['delta'] = variable_tuple(
            variable() for p in polytopes)
        delta[0].ub = 1
        delta[-1].lb = 0
        delta_tuple = tuple(delta)
        y = self.v['y'] = variable_tuple(
            variable(domain_type=IntegerSet, lb=0, ub=1)
            for p in polytopes[:-1])

        # create piecewise constraints
        self.c = constraint_list()

        self.c.append(linear_constraint(
            variables=(self.input,) + delta_tuple,
            coefficients=(-1,) + tuple(self.breakpoints[p+1] - \
                                       self.breakpoints[p]
                                       for p in polytopes),
            rhs=-self.breakpoints[0]))

        self.c.append(linear_constraint(
            variables=(self.output,) + delta_tuple,
            coefficients=(-1,) + tuple(self.values[p+1] - \
                                       self.values[p]
                                       for p in polytopes)))
        if self.bound == 'ub':
            self.c[-1].lb = -self.values[0]
        elif self.bound == 'lb':
            self.c[-1].ub = -self.values[0]
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = -self.values[0]

        clist1 = []
        clist2 = []
        for p in polytopes[:-1]:
            clist1.append(linear_constraint(
                variables=(delta[p+1], y[p]),
                coefficients=(1, -1),
                ub=0))
            clist2.append(linear_constraint(
                variables=(y[p], delta[p]),
                coefficients=(1, -1),
                ub=0))
        self.c.append(constraint_tuple(clist1))
        self.c.append(constraint_tuple(clist2))

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_inc, self).validate(**kwds)

registered_transforms['inc'] = piecewise_inc

class piecewise_dlog(TransformedPiecewiseLinearFunction):
    """Discrete DLOG piecewise representation

    Expresses a piecewise linear function using the DLOG
    formulation. This formulation uses logarithmic number of
    discrete variables in terms of number of breakpoints.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_dlog, self).__init__(*args, **kwds)

        breakpoints = self.breakpoints
        values = self.values

        if not is_positive_power_of_two(len(breakpoints)-1):
            raise ValueError("The list of breakpoints must be "
                             "of length (2^n)+1 for some positive "
                             "integer n. Invalid length: %s"
                             % (len(breakpoints)))

        # create branching schemes
        L = log2floor(len(breakpoints)-1)
        assert 2**L == len(breakpoints)-1
        B_LEFT, B_RIGHT = self._branching_scheme(L)

        # create indexers
        polytopes = range(len(breakpoints)-1)
        vertices = range(len(breakpoints))
        def polytope_verts(p):
            return xrange(p,p+2)

        # create vars
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_dict(
            ((p,v), variable(lb=0))
            for p in polytopes
            for v in polytope_verts(p))
        y = self.v['y'] = variable_tuple(
            variable(domain_type=IntegerSet, lb=0, ub=1) for i in range(L))

        # create piecewise constraints
        self.c = constraint_list()

        self.c.append(linear_constraint(
            variables=(self.input,) + tuple(lmbda[p,v]
                                            for p in polytopes
                                            for v in polytope_verts(p)),
            coefficients=(-1,) + tuple(breakpoints[v]
                                       for p in polytopes
                                       for v in polytope_verts(p)),
            rhs=0))

        self.c.append(linear_constraint(
            variables=(self.output,) + tuple(lmbda[p,v]
                                             for p in polytopes
                                             for v in polytope_verts(p)),
            coefficients=(-1,) + tuple(values[v]
                                       for p in polytopes
                                       for v in polytope_verts(p))))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0

        self.c.append(linear_constraint(
            variables=tuple(lmbda.values()),
            coefficients=(1,)*len(lmbda),
            rhs=1))

        clist = []
        for i in range(L):
            variables = tuple(lmbda[p,v]
                              for p in B_LEFT[i]
                              for v in polytope_verts(p))
            clist.append(linear_constraint(
                variables=variables + (y[i],),
                coefficients=(1,)*len(variables) + (-1,),
                ub=0))
        self.c.append(constraint_tuple(clist))
        del clist

        clist = []
        for i in range(L):
            variables = tuple(lmbda[p,v]
                              for p in B_RIGHT[i]
                              for v in polytope_verts(p))
            clist.append(linear_constraint(
                variables=variables + (y[i],),
                coefficients=(1,)*len(variables) + (1,),
                ub=1))
        self.c.append(constraint_tuple(clist))

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
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_dlog, self).validate(**kwds)

registered_transforms['dlog'] = piecewise_dlog

class piecewise_log(TransformedPiecewiseLinearFunction):
    """Discrete LOG piecewise representation

    Expresses a piecewise linear function using the LOG
    formulation. This formulation uses logarithmic number of
    discrete variables in terms of number of breakpoints.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_log, self).__init__(*args, **kwds)

        breakpoints = self.breakpoints
        values = self.values

        if not is_positive_power_of_two(len(breakpoints)-1):
            raise ValueError("The list of breakpoints must be "
                             "of length (2^n)+1 for some positive "
                             "integer n. Invalid length: %s"
                             % (len(breakpoints)))

        # create branching schemes
        L = log2floor(len(breakpoints)-1)
        S,B_LEFT,B_RIGHT = self._branching_scheme(L)

        # create indexers
        polytopes = range(len(breakpoints) - 1)
        vertices = range(len(breakpoints))

        # create vars
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_tuple(
            variable(lb=0) for v in vertices)
        y = self.v['y'] = variable_list(
            variable(domain_type=IntegerSet, lb=0, ub=1) for s in S)

        # create piecewise constraints
        self.c = constraint_list()

        self.c.append(linear_constraint(
            variables=(self.input,) + tuple(lmbda),
            coefficients=(-1,) + breakpoints,
            rhs=0))

        self.c.append(linear_constraint(
            variables=(self.output,) + tuple(lmbda),
            coefficients=(-1,) + values))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0

        self.c.append(linear_constraint(
            variables=tuple(lmbda),
            coefficients=(1,)*len(lmbda),
            rhs=1))

        clist = []
        for s in S:
            variables=tuple(lmbda[v] for v in B_LEFT[s])
            clist.append(linear_constraint(
                variables=variables + (y[s],),
                coefficients=(1,)*len(variables) + (-1,),
                ub=0))
        self.c.append(constraint_tuple(clist))
        del clist

        clist = []
        for s in S:
            variables=tuple(lmbda[v] for v in B_RIGHT[s])
            clist.append(linear_constraint(
                variables=variables + (y[s],),
                coefficients=(1,)*len(variables) + (1,),
                ub=1))
        self.c.append(constraint_tuple(clist))

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
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_log, self).validate(**kwds)

registered_transforms['log'] = piecewise_log
