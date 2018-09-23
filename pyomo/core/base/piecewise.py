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
This file contains a library of functions needed to construct
piecewise constraints for a Pyomo model. All piecewise types except
for SOS2, BIGM_SOS1, BIGM_BIN were taken from the paper:

Mixed-Integer Models for Non-separable Piecewise Linear Optimization:
Unifying framework and Extensions (Vielma, Nemhauser 2008).

TODO: Add regression tests for the following completed tasks
*) user not providing floats can be an major issue for BIGM's and MC
*) Other TODO's
*) nonconvex/nonconcave functions - BIGM_SOS1, BIGM_SOS2 ***** possible edge case bug

Possible Extensions
*) Consider another piecewise rep ("SOS2_MANUAL"?) where we manually implement
   extra constraints to define an SOS2 set, this would be compatible with GLPK,
   http://winglpk.sourceforge.net/media/glpk-sos2_02.pdf
*) double check that LOG and DLOG reps really do require (2^n)+1 points, or can
   we just add integer cuts (or something more intelligent) in order to handle
   piecewise functions without 2^n polytopes
*) piecewise for functions of the form y = f(x1,x2,...)
"""

# ****** NOTE: Nothing in this file relies on integer division *******
#              I predict this will save numerous headaches as
#              well as gratuitous calls to float() in this code
from __future__ import division

__all__ = ['Piecewise']

import logging
import math
import itertools
import operator
import types

from pyutilib.enum import Enum
from pyutilib.misc import flatten_tuple

from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.var import Var, _VarData, IndexedVar
from pyomo.core.base.set_types import PositiveReals, NonNegativeReals, Binary
from pyomo.core.base.numvalue import value

from six import iterkeys, advance_iterator
from six.moves import xrange, zip

logger = logging.getLogger('pyomo.core')

PWRepn = Enum('SOS2',
              'BIGM_BIN',
              'BIGM_SOS1',
              'CC',
              'DCC',
              'DLOG',
              'LOG',
              'MC',
              'INC')

Bound = Enum('Lower',
             'Upper',
             'Equal')

# BE SURE TO CHANGE THE PIECWISE DOCSTRING
# IF THIS GETS CHANGED
_WARNING_TOLERANCE = 1e-8


def _isNonDecreasing(vals):
    """
    checks that list of points is
    nondecreasing
    """
    it = iter(vals)
    advance_iterator(it)
    op = operator.ge
    return all(itertools.starmap(op, zip(it,vals)))

def _isNonIncreasing(vals):
    """
    checks that list of points is
    nonincreasing
    """
    it = iter(vals)
    advance_iterator(it)
    op = operator.le
    return all(itertools.starmap(op, zip(it,vals)))

def _isPowerOfTwo(x):
    """
    checks that a number is a nonzero and positive power of 2
    """
    if (x <= 0):
        return False
    else:
        return ( (x & (x - 1)) == 0 )

def _GrayCode(nbits):
    """
    Generates a GrayCode of nbits represented
    by a list of lists
    """
    bitset = [0 for i in xrange(nbits)]
    # important that we copy bitset each time
    graycode = [list(bitset)]

    for i in xrange(2,(1<<nbits)+1):
        if i%2:
            for j in xrange(-1,-nbits,-1):
                if bitset[j]:
                    bitset[j-1]=bitset[j-1]^1
                    break
        else:
            bitset[-1]=bitset[-1]^1
        # important that we copy bitset each time
        graycode.append(list(bitset))

    return graycode

def _characterize_function(name, tol, f_rule, model, points, *index):
    """
    Generates a list of range values and checks
    for convexity/concavity. Assumes domain points
    are sorted in increasing order.
    """
    # Make sure the list is a list of raw
    # numbers and not Pyomo Params or Expressions.
    # Failing to do this can generate strange
    # expression generation errors in the checks below
    points = [value(_p) for _p in points]

    # we use future division to protect against the case where
    # the user supplies integer type points for return values
    if isinstance(f_rule,types.FunctionType):
        values = [f_rule(model,*flatten_tuple((index,x))) for x in points]
    elif f_rule.__class__ is dict:
        if len(index) == 1:
            values = f_rule[index[0]]
        else:
            values = f_rule[index]
    else: # a list or tuple
        values = f_rule
    # Make sure the list is a list of raw
    # numbers and not Pyomo Params or Expressions.
    # Failing to do this can generate strange
    # expression generation errors in the checks below
    values = [value(_p) for _p in values]

    step = False
    try:
        slopes = [(values[i]-values[i-1])/(points[i]-points[i-1])
                  for i in xrange(1,len(points))]
    except ZeroDivisionError:
        # we have a step function
        step = True
        slopes = [(None) if (points[i]==points[i-1]) else ((values[i]-values[i-1])/(points[i]-points[i-1])) for i in xrange(1,len(points))]

    # TODO: Warn when the slopes of two consecutive line
    #       segments are nearly equal since this is likely
    #       due to a user mistake and may cause issue with
    #       the solver.
    #       *** This is already done below but there
    #           is probably a more correct way
    #           to send this warning through Pyomo
    if not all(itertools.starmap(lambda x1,x2: (True) if ((x1 is None) or (x2 is None)) else (abs(x1-x2) > tol), zip(slopes, itertools.islice(slopes, 1, None)))):
        msg = "**WARNING: Piecewise component '%s[%s]' has detected slopes of consecutive piecewise "\
              "segments to be within "+str(tol)+" of one another. Refer to the Piecewise help "\
              "documentation for information on how to disable this warning."
        if index == ():
            index = None
        print(msg % (name, flatten_tuple(index)))

    if step is True:
        return 0,values,True
    if _isNonDecreasing(slopes):
        # convex
        return 1,values,False
    if _isNonIncreasing(slopes):
        # concave
        return -1,values,False
    return 0,values,False


class _PiecewiseData(_BlockData):
    """
    This class defines the base class for all linearization
    and piecewise constraint generators..
    """

    def __init__(self,parent):
        _BlockData.__init__(self, parent)
        self._constructed = True
        self._bound_type = None
        self._domain_pts = None
        self._range_pts = None
        self._x = None
        self._y = None

    def updateBoundType(self, bound_type):
        self._bound_type = bound_type

    def updatePoints(self, domain_pts, range_pts):
        # ***Note: most (if not all) piecewise constraint generators
        #          assume the list of domain points is sorted.
        if not _isNonDecreasing(domain_pts):
            msg = "'%s' does not have a list of domain points "\
                  "that is non-decreasing"
            raise ValueError(msg % (self.name,))
        self._domain_pts = domain_pts
        self._range_pts = range_pts

    def build_constraints(self, functor, x_var, y_var):
        functor.construct(self, x_var, y_var)
        self.__dict__['_x'] = x_var
        self.__dict__['_y'] = y_var

    def referenced_variables(self):
        return (self._x, self._y)

    def __call__(self, x):
        if self._constructed is False:
            raise ValueError("Piecewise component %s has not "
                             "been constructed yet" % self.name)

        for i in xrange(len(self._domain_pts)-1):
            xL = self._domain_pts[i]
            xU = self._domain_pts[i+1]
            if (xL <= x) and (x <= xU):
                yL = self._range_pts[i]
                yU = self._range_pts[i+1]
                if xL == xU: # a step function
                    return yU
                # using future division
                return yL + ((yU-yL)/(xU-xL))*(x-xL)
        raise ValueError("The point %s is outside the list of domain "
                         "points for Piecewise component %s. The valid "
                         "point range is [%s,%s]."
                         % (x, self.name,
                            min(self._domain_pts),
                            max(self._domain_pts)))

class _SimpleSinglePiecewise(object):
    """
    Called when the piecwise points list has only two points
    """

    def construct(self,pblock,x_var,y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_SimpleSinglePiecewise: construct() called during"\
                                "invalid state.")

        # create a single linear constraint
        LHS = y_var
        F_AT_XO = y_pts[0]
        # using future division
        dF_AT_XO = (y_pts[1]-y_pts[0])/(x_pts[1]-x_pts[0])
        X_MINUS_XO = x_var-x_pts[0]
        if bound_type == Bound.Upper:
            expr= LHS <= F_AT_XO + dF_AT_XO*X_MINUS_XO
        elif bound_type == Bound.Lower:
            expr= LHS >= F_AT_XO + dF_AT_XO*X_MINUS_XO
        elif bound_type == Bound.Equal:
            expr= LHS == F_AT_XO + dF_AT_XO*X_MINUS_XO
        else:
            raise ValueError("Invalid Bound for _SimpleSinglePiecewise object")
        pblock.single_line_constraint = Constraint(expr=expr)

        # In order to enforce the same behavior as actual piecewise
        # constraints, we constrain the domain variable between the
        # outer domain pts. But in order to prevent filling the model
        # with unecessary constraints, we only do this when absolutely
        # necessary.
        if not x_var.lb is None and x_var.lb < x_pts[0]:
            pblock.simplified_piecewise_domain_constraint_lower = Constraint(expr=x_pts[0] <= x_var)
        if not x_var.ub is None and x_var.ub > x_pts[1]:
            pblock.simplified_piecewise_domain_constraint_upper = Constraint(expr=x_var <= x_pts[-1])

class _SimplifiedPiecewise(object):
    """
    Called when piecewise constraints are simplified due to a lower bounding
    convex function or an upper bounding concave function
    """

    def construct(self,pblock,x_var,y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_SimplifiedPiecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        conlist = pblock.simplified_piecewise_constraint = ConstraintList()
        for i in xrange(len_x_pts-1):
            F_AT_XO = y_pts[i]
            dF_AT_XO = (y_pts[i+1]-y_pts[i])/(x_pts[i+1]-x_pts[i])
            XO = x_pts[i]
            if bound_type == Bound.Upper:
                conlist.add((0,-y_var+F_AT_XO+dF_AT_XO*(x_var-XO),None))
            elif bound_type == Bound.Lower:
                conlist.add((None,-y_var+F_AT_XO+dF_AT_XO*(x_var-XO),0))
            else:
                raise ValueError("Invalid Bound for _SimplifiedPiecewise object")

        # In order to enforce the same behavior as actual piecewise
        # constraints, we constrain the domain variable between the
        # outer domain pts. But in order to prevent filling the model
        # with unecessary constraints, we only do this when absolutely
        # necessary.
        if not x_var.lb is None and x_var.lb < x_pts[0]:
            pblock.simplified_piecewise_domain_constraint_lower = Constraint(expr=x_pts[0] <= x_var)
        if not x_var.ub is None and x_var.ub > x_pts[-1]:
            pblock.simplified_piecewise_domain_constraint_upper = Constraint(expr=x_var <= x_pts[-1])

class _SOS2Piecewise(object):
    """
    Called to generate Piecewise constraint using the SOS2 formulation
    """

    def construct(self,pblock,x_var,y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_SOS2Piecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        # create indexers
        sos2_index = range(len_x_pts)

        # create vars
        sos2_y = pblock.SOS2_y = Var(sos2_index,within=NonNegativeReals)

        # create piecewise constraints
        conlist = pblock.SOS2_constraint = ConstraintList()
        conlist.add( (x_var-sum(sos2_y[i]*x_pts[i] for i in sos2_index),0) )

        LHS = y_var
        RHS = sum(sos2_y[i]*y_pts[i] for i in sos2_index)
        expr = None
        if bound_type == Bound.Upper:
            conlist.add( (None,LHS-RHS,0) )
        elif bound_type == Bound.Lower:
            conlist.add( (0,LHS-RHS,None) )
        elif bound_type == Bound.Equal:
            conlist.add( (LHS-RHS,0) )
        else:
            raise ValueError("Invalid Bound for _SOS2Piecewise object")
        conlist.add( (sum(sos2_y[j] for j in sos2_index),1) )
        def SOS2_rule(model):
            return [sos2_y[i] for i in sos2_index]
        pblock.SOS2_sosconstraint = SOSConstraint(initialize=SOS2_rule, sos=2)


class _DCCPiecewise(object):
    """
    Called to generate Piecewise constraint using the DCC formulation
    """

    def construct(self,pblock,x_var,y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_DCCPiecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        # create indexers
        polytopes = range(1,len_x_pts)
        vertices = range(1,len_x_pts+1)
        def polytope_verts(p):
            return xrange(p,p+2)

        # create vars
        pblock.DCC_lambda = Var(polytopes,vertices,within=PositiveReals)
        lmda = pblock.DCC_lambda
        pblock.DCC_bin_y = Var(polytopes,within=Binary)
        bin_y = pblock.DCC_bin_y

        # create piecewise constraints
        pblock.DCC_constraint1 = Constraint(expr=x_var==sum(lmda[p,v]*x_pts[v-1] \
                                                         for p in polytopes \
                                                         for v in polytope_verts(p)))

        LHS = y_var
        RHS = sum(lmda[p,v]*y_pts[v-1] for p in polytopes for v in polytope_verts(p))
        expr = None
        if bound_type == Bound.Upper:
            expr= LHS <= RHS
        elif bound_type == Bound.Lower:
            expr= LHS >= RHS
        elif bound_type == Bound.Equal:
            expr= LHS == RHS
        else:
            raise ValueError("Invalid Bound for _DCCPiecewise object")
        pblock.DCC_constraint2 = Constraint(expr=expr)

        def con3_rule(model,p):
            return bin_y[p] == sum(lmda[p,v] for v in polytope_verts(p))
        pblock.DCC_constraint3 = Constraint(polytopes,rule=con3_rule)
        pblock.DCC_constraint4 = Constraint(expr=sum(bin_y[p] \
                                                  for p in polytopes) == 1)


class _DLOGPiecewise(object):
    """
    Called to generate Piecewise constraint using the DLOG formulation
    """

    def _Branching_Scheme(self,L):
        """
        Branching scheme for DLOG
        """
        MAX = 2**L
        mylists1 = {}
        for i in xrange(1,L+1):
            mylists1[i] = []
            start = 1
            step = int(MAX/(2**i))
            while(start < MAX):
                mylists1[i].extend([j for j in xrange(start,start+step)])
                start += 2*step

        biglist = xrange(1,MAX+1)
        mylists2 = {}
        for i in sorted(mylists1.keys()):
            mylists2[i] = []
            for j in biglist:
                if j not in mylists1[i]:
                    mylists2[i].append(j)
            mylists2[i] = sorted(mylists2[i])

        return mylists1, mylists2

    def construct(self,pblock,x_var,y_var):
        if not _isPowerOfTwo(len(pblock._domain_pts)-1):
            msg = "'%s' does not have a list of domain points "\
                  "with length (2^n)+1"
            raise ValueError(msg % (pblock.name,))
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_DLOGPiecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        # create branching schemes
        L_i = int(math.log(len_x_pts-1,2))
        B_ZERO,B_ONE = self._Branching_Scheme(L_i)

        # create indexers
        polytopes = range(1,len_x_pts)
        vertices = range(1,len_x_pts+1)
        bin_y_index = range(1,L_i+1)
        def polytope_verts(p):
            return xrange(p,p+2)

        # create vars
        pblock.DLOG_lambda = Var(polytopes,vertices,within=PositiveReals)
        lmda = pblock.DLOG_lambda
        pblock.DLOG_bin_y = Var(bin_y_index,within=Binary)
        bin_y = pblock.DLOG_bin_y
        # create piecewise constraints
        pblock.DLOG_constraint1 = Constraint(expr=x_var==sum(lmda[p,v]*x_pts[v-1] \
                                                         for p in polytopes \
                                                         for v in polytope_verts(p)))

        LHS = y_var
        RHS = sum(lmda[p,v]*y_pts[v-1] for p in polytopes for v in polytope_verts(p))
        expr = None
        if bound_type == Bound.Upper:
            expr= LHS <= RHS
        elif bound_type == Bound.Lower:
            expr= LHS >= RHS
        elif bound_type == Bound.Equal:
            expr= LHS == RHS
        else:
            raise ValueError("Invalid Bound for _DLOGPiecewise object")
        pblock.DLOG_constraint2 = Constraint(expr=expr)
        pblock.DLOG_constraint3 = Constraint(expr=sum(lmda[p,v] \
                                                  for p in polytopes \
                                                  for v in polytope_verts(p)) == 1)
        def con4_rule(model,l):
            return sum(lmda[p,v] for p in B_ZERO[l] \
                                 for v in polytope_verts(p)) \
                   <= bin_y[l]
        pblock.DLOG_constraint4 = Constraint(bin_y_index,rule=con4_rule)
        def con5_rule(model,l):
            return sum(lmda[p,v] for p in B_ONE[l] \
                                 for v in polytope_verts(p)) \
                   <= (1-bin_y[l])
        pblock.DLOG_constraint5 = Constraint(bin_y_index,rule=con5_rule)


class _CCPiecewise(object):
    """
    Called to generate Piecewise constraint using the CC formulation
    """

    def construct(self,pblock,x_var,y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_CCPiecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        # create indexers
        polytopes = range(1,len_x_pts)
        vertices = range(1,len_x_pts+1)
        def vertex_polys(v):
            if v == 1:
                return [v]
            if v == len_x_pts:
                return [v-1]
            else:
                return [v-1,v]

        # create vars
        pblock.CC_lambda = Var(vertices,within=NonNegativeReals)
        lmda = pblock.CC_lambda
        pblock.CC_bin_y = Var(polytopes,within=Binary)
        bin_y = pblock.CC_bin_y
        # create piecewise constraints
        pblock.CC_constraint1 = Constraint(expr=x_var==sum(lmda[v]*x_pts[v-1] \
                                                         for v in vertices))

        LHS = y_var
        RHS = sum(lmda[v]*y_pts[v-1] for v in vertices)
        expr = None
        if bound_type == Bound.Upper:
            expr= LHS <= RHS
        elif bound_type == Bound.Lower:
            expr= LHS >= RHS
        elif bound_type == Bound.Equal:
            expr= LHS == RHS
        else:
            raise ValueError("Invalid Bound for _CCPiecewise object")
        pblock.CC_constraint2 = Constraint(expr=expr)
        pblock.CC_constraint3 = Constraint(expr=sum(lmda[v] \
                                                  for v in vertices) == 1)
        def con4_rule(model,v):
            return lmda[v] <= sum(bin_y[p] for p in vertex_polys(v))
        pblock.CC_constraint4 = Constraint(vertices,rule=con4_rule)
        pblock.CC_constraint5 = Constraint(expr=sum(bin_y[p] \
                                                  for p in polytopes) == 1)


class _LOGPiecewise(object):
    """
    Called to generate Piecewise constraint using the LOG formulation
    """

    def _Branching_Scheme(self,n):
        """
        Branching scheme for LOG, requires a gray code
        """
        BIGL = 2**n
        S = range(1,n+1)
        # turn the GrayCode into a dictionary indexed
        # starting at 1
        G = dict(enumerate(_GrayCode(n),start=1))

        L = dict((s,[k+1 for k in xrange(BIGL+1) \
                         if ((k == 0) or (G[k][s-1] == 1)) \
                         and ((k == BIGL) or (G[k+1][s-1] == 1))]) for s in S)
        R = dict((s,[k+1 for k in xrange(BIGL+1) \
                         if ((k == 0) or (G[k][s-1] == 0)) \
                         and ((k == BIGL) or (G[k+1][s-1] == 0))]) for s in S)

        return S,L,R

    def construct(self,pblock,x_var,y_var):
        if not _isPowerOfTwo(len(pblock._domain_pts)-1):
            msg = "'%s' does not have a list of domain points "\
                  "with length (2^n)+1"
            raise ValueError(msg % (pblock.name,))
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_LOGPiecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        # create branching schemes
        L_i = int(math.log(len_x_pts-1,2))
        S_i,B_LEFT,B_RIGHT = self._Branching_Scheme(L_i)

        # create indexers
        polytopes = range(1,len_x_pts)
        vertices = range(1,len_x_pts+1)
        bin_y_index = S_i

        # create vars
        pblock.LOG_lambda = Var(vertices,within=NonNegativeReals)
        lmda = pblock.LOG_lambda
        pblock.LOG_bin_y = Var(bin_y_index,within=Binary)
        bin_y = pblock.LOG_bin_y
        # create piecewise constraints
        pblock.LOG_constraint1 = Constraint(expr=x_var==sum(lmda[v]*x_pts[v-1] \
                                                         for v in vertices))

        LHS = y_var
        RHS = sum(lmda[v]*y_pts[v-1] for v in vertices)
        expr = None
        if bound_type == Bound.Upper:
            expr= LHS <= RHS
        elif bound_type == Bound.Lower:
            expr= LHS >= RHS
        elif bound_type == Bound.Equal:
            expr= LHS == RHS
        else:
            raise ValueError("Invalid Bound for _LOGPiecewise object")
        pblock.LOG_constraint2 = Constraint(expr=expr)
        pblock.LOG_constraint3 = Constraint(expr=sum(lmda[v] \
                                                  for v in vertices) == 1)
        def con4_rule(model,s):
            return sum(lmda[v] for v in B_LEFT[s]) <= bin_y[s]
        pblock.LOG_constraint4 = Constraint(bin_y_index,rule=con4_rule)
        def con5_rule(model,s):
            return sum(lmda[v] for v in B_RIGHT[s]) <= (1-bin_y[s])
        pblock.LOG_constraint5 = Constraint(bin_y_index,rule=con5_rule)


class _MCPiecewise(object):
    """
    Called to generate Piecewise constraint using the MC formulation
    """

    def construct(self,pblock,x_var,y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_MCPiecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        # create indexers
        polytopes = range(1,len_x_pts)

        # create constants (using future division)
        SLOPE = dict((p,(y_pts[p]-y_pts[p-1])/(x_pts[p]-x_pts[p-1])) \
                         for p in polytopes)
        INTERSEPT = dict((p,y_pts[p-1] - (SLOPE[p]*x_pts[p-1])) for p in polytopes)

        # create vars
        pblock.MC_poly_x = Var(polytopes)
        poly_x = pblock.MC_poly_x
        pblock.MC_bin_y = Var(polytopes,within=Binary)
        bin_y = pblock.MC_bin_y
        # create piecewise constraints
        pblock.MC_constraint1 = Constraint(expr=x_var==sum(poly_x[p] \
                                                         for p in polytopes))

        LHS = y_var
        RHS = sum(poly_x[p]*SLOPE[p]+bin_y[p]*INTERSEPT[p] for p in polytopes)
        expr = None
        if bound_type == Bound.Upper:
            expr=  LHS <= RHS
        elif bound_type == Bound.Lower:
            expr=  LHS >= RHS
        elif bound_type == Bound.Equal:
            expr=  LHS == RHS
        else:
            raise ValueError("Invalid Bound for _INCPiecewise object")
        pblock.MC_constraint2 = Constraint(expr=expr)
        def con3_rule(model,p):
            return bin_y[p]*x_pts[p-1] <= poly_x[p]
        pblock.MC_constraint3 = Constraint(polytopes,rule=con3_rule)
        def con4_rule(model,p):
            return poly_x[p]  <= bin_y[p]*x_pts[p]
        pblock.MC_constraint4 = Constraint(polytopes,rule=con4_rule)
        pblock.MC_constraint5 = Constraint(expr=sum(bin_y[p] \
                                                  for p in polytopes) == 1)

class _INCPiecewise(object):
    """
    Called to generate Piecewise constraint using the INC formulation
    """

    def construct(self,pblock,x_var,y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_INCPiecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        # create indexers
        polytopes = range(1,len_x_pts)
        bin_y_index = range(1,len_x_pts-1)

        # create vars
        pblock.INC_delta = Var(polytopes)
        delta = pblock.INC_delta
        delta[1].setub(1)
        delta[len_x_pts-1].setlb(0)
        pblock.INC_bin_y = Var(bin_y_index,within=Binary)
        bin_y = pblock.INC_bin_y
        # create piecewise constraints
        pblock.INC_constraint1 = Constraint(expr=x_var==x_pts[0] + \
                                                  sum(delta[p]*(x_pts[p]-x_pts[p-1]) \
                                                  for p in polytopes))

        LHS = y_var
        RHS = y_pts[0] + sum(delta[p]*(y_pts[p]-y_pts[p-1]) for p in polytopes)
        expr = None
        if bound_type == Bound.Upper:
            expr= LHS <= RHS
        elif bound_type == Bound.Lower:
            expr= LHS >= RHS
        elif bound_type == Bound.Equal:
            expr= LHS == RHS
        else:
            raise ValueError("Invalid Bound for _INCPiecewise object")
        pblock.INC_constraint2 = Constraint(expr=expr)
        def con3_rule(model,p):
            if p != polytopes[-1]:
                return delta[p+1] <= bin_y[p]
            else:
                return Constraint.Skip
        pblock.INC_constraint3 = Constraint(polytopes,rule=con3_rule)
        def con4_rule(model,p):
            if p != polytopes[-1]:
                return bin_y[p] <= delta[p]
            else:
                return Constraint.Skip
        pblock.INC_constraint4 = Constraint(polytopes,rule=con4_rule)


class _BIGMPiecewise(object):
    """
    Called to generate Piecewise constraint using the BIGM formulation
    """
    def __init__(self,binary=True):
        self.binary = binary
        if not (self.binary in [True,False]):
            raise ValueError("_BIGMPiecewise must be initialized with the binary "\
                              "flag set to True or False (choose one).")

    def construct(self,pblock,x_var,y_var):
        # The BIGM methods currently determine tightest possible M
        # values. This method is implemented in such a way that
        # binary/sos1 variables are not created when this M is zero
        tag = ""
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts,y_pts,bound_type]:
            raise RuntimeError("_BIGMPiecewise: construct() called during "\
                                "invalid state.")
        len_x_pts = len(x_pts)

        if self.binary is True:
            tag += "bin"
        else:
            tag += "sos1"

        # generate tightest bigM values
        OPT_M = {}
        OPT_M['UB'] = {}
        OPT_M['LB'] = {}

        if bound_type in [Bound.Upper,Bound.Equal]:
            OPT_M['UB'] = self._find_M(x_pts, y_pts, Bound.Upper)
        if bound_type in [Bound.Lower,Bound.Equal]:
            OPT_M['LB'] = self._find_M(x_pts, y_pts, Bound.Lower)

        all_keys = set(iterkeys(OPT_M['UB'])).union(iterkeys(OPT_M['LB']))
        full_indices = []
        full_indices.extend(range(1,len_x_pts))
        bigm_y_index = None
        bigm_y = None
        if len(all_keys) > 0:
            bigm_y_index = all_keys

            def y_domain():
                if self.binary is True:
                    return Binary
                else:
                    return NonNegativeReals
            setattr(pblock,tag+'_y', Var(bigm_y_index,within=y_domain()))
            bigm_y = getattr(pblock,tag+'_y')

        def con1_rule(model,i):
            if bound_type in [Bound.Upper,Bound.Equal]:
                rhs = 1.0
                if i not in OPT_M['UB']:
                    rhs *= 0.0
                else:
                    rhs *= OPT_M['UB'][i]*(1-bigm_y[i])
                # using future division
                return y_var - y_pts[i-1] - \
                ((y_pts[i]-y_pts[i-1])/(x_pts[i]-x_pts[i-1]))*(x_var-x_pts[i-1])\
                <= rhs
            elif bound_type == Bound.Lower:
                rhs = 1.0
                if i not in OPT_M['LB']:
                    rhs *= 0.0
                else:
                    rhs *= OPT_M['LB'][i]*(1-bigm_y[i])
                # using future division
                return y_var - y_pts[i-1] - \
                ((y_pts[i]-y_pts[i-1])/(x_pts[i]-x_pts[i-1]))*(x_var-x_pts[i-1])\
                >= rhs

        def con2_rule(model):
            expr = [bigm_y[i] for i in xrange(1,len_x_pts) if i in all_keys]
            if len(expr) > 0:
                return sum(expr) == 1
            else:
                return Constraint.Skip

        def conAFF_rule(model,i):
            rhs = 1.0
            if i not in OPT_M['LB']:
                rhs *= 0.0
            else:
                rhs *= OPT_M['LB'][i]*(1-bigm_y[i])
            # using future division
            return y_var - y_pts[i-1] - \
            ((y_pts[i]-y_pts[i-1])/(x_pts[i]-x_pts[i-1]))*(x_var-x_pts[i-1]) \
            >= rhs

        pblock.BIGM_constraint1 = Constraint(full_indices,rule=con1_rule)
        if len(all_keys) > 0:
            pblock.BIGM_constraint2 = Constraint(rule=con2_rule)
        if bound_type == Bound.Equal:
            pblock.BIGM_constraint3 = Constraint(full_indices,rule=conAFF_rule)

        if len(all_keys) > 0:
            if self.binary is False:
                pblock.BIGM_constraint4 = SOSConstraint(var=bigm_y, sos=1)

        # In order to enforce the same behavior as actual piecewise
        # constraints, we constrain the domain variable between the
        # outer domain pts. But in order to prevent filling the model
        # with unecessary constraints, we only do this when absolutely
        # necessary.
        if not x_var.lb is None and x_var.lb < x_pts[0]:
            pblock.bigm_domain_constraint_lower = Constraint(expr=x_pts[0] <= x_var)
        if not x_var.ub is None and x_var.ub > x_pts[-1]:
            pblock.bigm_domain_constraint_upper = Constraint(expr=x_var <= x_pts[-1])

    def _M_func(self,a,Fa,b,Fb,c,Fc):
        # using future division
        return Fa - Fb - ((a-b) * ((Fc-Fb) / (c-b)))

    def _find_M(self,x_pts,y_pts,bound_type):
        len_x_pts = len(x_pts)
        _self_M_func = self._M_func

        M_final = {}
        for j in xrange(1,len_x_pts):
            index = j
            if (bound_type == Bound.Lower):
                M_final[index] = min( [0.0, min([_self_M_func(x_pts[k],y_pts[k],
                                                              x_pts[j-1],y_pts[j-1],
                                                              x_pts[j],y_pts[j]) \
                                            for k in xrange(len_x_pts)])] )
            elif (bound_type == Bound.Upper):
                M_final[index] = max( [0.0, max([_self_M_func(x_pts[k],y_pts[k],
                                                              x_pts[j-1],y_pts[j-1],
                                                              x_pts[j],y_pts[j]) \
                                             for k in xrange(len_x_pts)])] )
            else:
                raise ValueError("Invalid Bound passed to _find_M function")
            if M_final[index] == 0.0:
                del M_final[index]
        return M_final


@ModelComponentFactory.register("Constraints that contain piecewise linear expressions.")
class Piecewise(Block):
    """
    Adds piecewise constraints to a Pyomo model for functions of the
    form, y = f(x).

    Usage:
            model.const = Piecewise(index_1,...,index_n,yvar,xvar,**Keywords)
            model.const = Piecewise(yvar,xvar,**Keywords)

    Keywords:

-pw_pts={},[],()
          A dictionary of lists (keys are index set) or a single list
          (for the non-indexed case or when an identical set of
          breakpoints is used across all indices) defining the set of
          domain breakpoints for the piecewise linear
          function. **ALWAYS REQUIRED**

-pw_repn=''
          Indicates the type of piecewise representation to use. This
          can have a major impact on solver performance.
          Choices: (Default 'SOS2')

             ~ + 'SOS2'      - Standard representation using sos2 constraints
             ~   'BIGM_BIN'  - BigM constraints with binary variables.
                               Theoretically tightest M values are automatically
                               determined.
             ~   'BIGM_SOS1' - BigM constraints with sos1 variables.
                               Theoretically tightest M values are automatically
                               determined.
             ~*+ 'DCC'       - Disaggregated convex combination model
             ~*+ 'DLOG'      - Logarithmic disaggregated convex combination model
             ~*+ 'CC'        - Convex combination model
             ~*+ 'LOG'       - Logarithmic branching convex combination
             ~*  'MC'        - Multiple choice model
             ~*+ 'INC'       - Incremental (delta) method

           + Supports step functions
           * Source: "Mixed-Integer Models for Non-separable Piecewise Linear
                      Optimization: Unifying framework and Extensions" (Vielma,
                      Nemhauser 2008)
           ~ Refer to the optional 'force_pw' keyword.

-pw_constr_type=''
          Indicates the bound type of the piecewise function.
          Choices:

                   'UB' - y variable is bounded above by piecewise function
                   'LB' - y variable is bounded below by piecewise function
                   'EQ' - y variable is equal to the piecewise function

-f_rule=f(model,i,j,...,x), {}, [], ()
          An object that returns a numeric value that is the range
          value corresponding to each piecewise domain point. For
          functions, the first argument must be a Pyomo model. The
          last argument is the domain value at which the function
          evaluates (Not a Pyomo Var). Intermediate arguments are the
          corresponding indices of the Piecewise component (if any).
          Otherwise, the object can be a dictionary of lists/tuples
          (with keys the same as the indexing set) or a singe
          list/tuple (when no indexing set is used or when all indices
          use an identical piecewise function).
          Examples:

                   # A function which changes with index
                   def f(model,j,x):
                      if (j == 2):
                         return x**2 + 1.0
                      else:
                         return x**2 + 5.0

                   # A nonlinear function
                   f = lambda model,x: return exp(x) + value(model.p)
                       (model.p is a Pyomo Param)

                   # A step function
                   f = [0,0,1,1,2,2]

-force_pw=True/False
          Using the given function rule and pw_pts, a check for
          convexity/concavity is implemented. If (1) the function is
          convex and the piecewise constraints are lower bounds or if
          (2) the function is concave and the piecewise constraints
          are upper bounds then the piecewise constraints will be
          substituted for linear constraints. Setting 'force_pw=True'
          will force the use of the original piecewise constraints
          even when one of these two cases applies.

-warning_tol=<float>                    Default=1e-8
          To aid in debugging, a warning is printed when consecutive
          slopes of piecewise segments are within <warning_tol> of
          each other.

-warn_domain_coverage=True/False        Default=True
          Print a warning when the feasible region of the domain
          variable is not completely covered by the piecewise
          breakpoints.

-unbounded_domain_var=True/False        Default=False
          Allow an unbounded or partially bounded Pyomo Var to be used
          as the domain variable.
          **NOTE: This does not imply unbounded piecewise segments
                  will be constructed. The outermost piecwise
                  breakpoints will bound the domain variable at each
                  index. However, the Var attributes .lb and .ub will
                  not be modified.
    """

    def __new__(cls, *args, **kwds):
        if cls != Piecewise:
            return super(Piecewise, cls).__new__(cls)
        if len(args) == 2:
            return SimplePiecewise.__new__(SimplePiecewise)
        else:
            return IndexedPiecewise.__new__(IndexedPiecewise)

    def __init__(self, *args, **kwds):
        # this is temporary as part of a move to user inputs
        # using Enums rather than strings
        translate_repn = {'BIGM_SOS1':PWRepn.BIGM_SOS1,\
                          PWRepn.BIGM_SOS1:PWRepn.BIGM_SOS1,\
                          'BIGM_BIN':PWRepn.BIGM_BIN,\
                          PWRepn.BIGM_BIN:PWRepn.BIGM_BIN,\
                          'SOS2':PWRepn.SOS2,\
                          PWRepn.SOS2:PWRepn.SOS2,\
                          'CC':PWRepn.CC,\
                          PWRepn.CC:PWRepn.CC,\
                          'DCC':PWRepn.DCC,\
                          PWRepn.DCC:PWRepn.DCC,\
                          'DLOG':PWRepn.DLOG,\
                          PWRepn.DLOG:PWRepn.DLOG,\
                          'LOG':PWRepn.LOG,\
                          PWRepn.LOG:PWRepn.LOG,\
                          'MC':PWRepn.MC,\
                          PWRepn.MC:PWRepn.MC,\
                          'INC':PWRepn.INC,\
                          PWRepn.INC:PWRepn.INC,\
                          None:None}

        # this is temporary as part of a move to user inputs
        # using Enums rather than strings
        translate_bound = {'UB':Bound.Upper,\
                           Bound.Upper:Bound.Upper,\
                           'LB':Bound.Lower,\
                           Bound.Lower:Bound.Lower,\
                           'EQ':Bound.Equal,\
                           Bound.Equal:Bound.Equal,\
                           None:None}

        # TODO: Update the keyword names. I think these are more clear
        #       pw_pts                  -> breakpoints
        #       pw_repn                 -> repn
        #       pw_constr_type          -> bound
        #       f_rule                  -> rule
        #       force_pw                -> simpify
        #       warning_tol
        #       warn_domain_coverage    -> warning_domain_coverage
        #       unbounded_domain_var
        #

        # extract all keywords used by this class
        pw_points = kwds.pop('pw_pts',None)
        # translate the user input to the enum type
        pw_rep = kwds.pop('pw_repn','SOS2')
        pw_rep = translate_repn.get(pw_rep,pw_rep)
        if (pw_rep == PWRepn.BIGM_BIN) or \
           (pw_rep == PWRepn.BIGM_SOS1):
            logger.warning(
                "DEPRECATED: The 'BIGM_BIN' and 'BIGM_SOS1' "
                "piecewise representations will be removed in "
                "a future version of Pyomo. They produce incorrect "
                "results in certain cases")
        # translate the user input to the enum type
        bound_type = kwds.pop('pw_constr_type',None)
        bound_type = translate_bound.get(bound_type,bound_type)
        f_rule = kwds.pop('f_rule',None)
        force_pw = kwds.pop('force_pw',False)
        warning_tol = kwds.pop('warning_tol',_WARNING_TOLERANCE)
        warn_domain_coverage = kwds.pop('warn_domain_coverage',True)
        unbounded_domain_var = kwds.pop('unbounded_domain_var',False)

        # all but the last two args should go to Block
        try:
            # Blocks have special handling when calling
            # __setattr__ with anything derived from component.
            # However, in this particular case we need to override
            # this so that these two variables don't get re-added
            # as new model components, therefore we directly modify
            # __dict__
            self.__dict__['_domain_var'] = args[-1]
            self.__dict__['_range_var'] = args[-2]
        except IndexError:
            msg = "Piecewise component initialized with less than two arguments"
            raise TypeError(msg)

        args = args[:-2]
        #
        # NOTE: The 'ctype' keyword argument is not defined here. This
        #       mocks what is done in PyomoModel.py, although here it
        #       feels like somewhat of a hack. The alternative is to
        #       modify the block_data_objects() method to include Piecewise
        #       type components. I'm not sure which is best at this
        #       time. Although, a consequence of the current
        #       implementation is that model.pprint() labels Piecewise
        #       blocks as simply Blocks.
        #
        #kwds.setdefault('ctype', Piecewise)
        Block.__init__(self,*args,**kwds)

        # Check that the variables args are actually Pyomo Vars
        if not( isinstance(self._domain_var,_VarData) or \
                isinstance(self._domain_var,IndexedVar) ):
            msg = "Piecewise component has invalid "\
                  "argument type for domain variable, %s"
            raise TypeError(msg % (repr(self._domain_var),))
        if not( isinstance(self._range_var,_VarData) or \
                isinstance(self._range_var,IndexedVar) ):
            msg = "Piecewise component has invalid "\
                  "argument type for range variable, %s"
            raise TypeError(msg % (repr(self._range_var),))

        # Test that the keyword values make sense
        if f_rule.__class__ not in [type(lambda: None),dict,list,tuple]:
            msg = "Piecewise component keyword 'f_rule' must "\
                  "be a function, dict, list, or tuple"
            raise ValueError(msg)
        if bound_type not in Bound:
            msg = "Invalid value for Piecewise component "\
                  "keyword 'pw_constr_type'"
            raise ValueError(msg)
        if warning_tol.__class__ is not float:
            msg = "Invalid type '%s' for Piecewise component "\
                  "keyword 'warning_tol', which must be of type 'float'"
            raise TypeError(msg % (type(warning_tol),))
        if warn_domain_coverage not in [True,False]:
            msg = "Invalid value for Piecewise component "\
                  "keyword 'warn_domain_coverage', which must be True or False"
            raise ValueError(msg)
        if unbounded_domain_var not in [True,False]:
            msg = "Invalid value for Piecewise component "\
                  "keyword 'unbounded_domain_var', which must be True or False"
            raise ValueError(msg)

        self._pw_rep = pw_rep
        self._bound_type = bound_type
        self._f_rule = f_rule
        self._force_pw = force_pw
        self._warning_tol = warning_tol
        self._warn_domain_coverage = warn_domain_coverage
        self._unbounded_domain_var = unbounded_domain_var

        if self.is_indexed() is False:
            if not ( isinstance(pw_points, list) or \
                     isinstance(pw_points,tuple) ):
                msg = "Invalid type '%s' for Piecewise component "\
                      "keyword 'pw_pts', which must be of type "\
                      "'list' or 'tuple' for non-indexed Piecewise component"
                raise TypeError(msg % (type(pw_points),))
            self._domain_points = {None:pw_points}
        else:
            if isinstance(pw_points, list) or \
               isinstance(pw_points,tuple):
                self._domain_points = {None:pw_points}
            elif isinstance(pw_points,dict):
                self._domain_points = pw_points
            else:
                msg = "Invalid type '%s' for Piecewise component "\
                      "keyword 'pw_pts', which must be of type "\
                      "'dict', 'list', or 'tuple' for indexed Piecewise component"
                raise TypeError(msg % (type(pw_points),))

    def construct(self, *args, **kwds):
        """
        A quick hack to call add after data has been loaded.
        """
        generate_debug_messages \
            = __debug__ and logger.isEnabledFor(logging.DEBUG)

        if self._constructed:
            return
        timer = ConstructionTimer(self)
        # We need to be able to add and construct new model
        # components on the fly so we make this Block behave concretely
        self._constructed=True

        # cache this because it is apparently expensive
        is_indexed = self.is_indexed()

        # construct each index of this component
        if not is_indexed:
            if generate_debug_messages:
                logger.debug("  Constructing single Piecewise component (index=None)")
            self.add(None, _is_indexed=is_indexed)
        else:
            for index in self._index:
                if generate_debug_messages:
                    logger.debug("  Constructing Piecewise index "+str(index))
                self.add(index, _is_indexed=is_indexed)
        timer.report()

    def _getitem_when_not_present(self, idx):
        return self._data.setdefault(idx, _PiecewiseData(self))

    def add(self, index, _is_indexed=None):

        if _is_indexed is None:
            _is_indexed = self.is_indexed()

        _self_parent = self._parent()
        _self_xvar = None
        _self_yvar = None
        _self_domain_pts_index = None
        if not _is_indexed:
            # allows one to mix Var and _VarData as input to
            # non-indexed Piecewise, index would be None in this case
            # so for Var elements Var[None] is Var, but _VarData[None] would fail
            _self_xvar = self._domain_var
            _self_yvar = self._range_var
            _self_domain_pts_index = self._domain_points[index]
        else:
            # The following allows one to specify a Var or _VarData
            # object even with an indexed Piecewise component.
            # The most common situation will most likely be a VarArray,
            # so we try this first.
            if not isinstance(self._domain_var, _VarData):
                _self_xvar = self._domain_var[index]
            else:
                _self_xvar = self._domain_var
            if not isinstance(self._range_var, _VarData):
                _self_yvar = self._range_var[index]
            else:
                _self_yvar = self._range_var
            try:
                _self_domain_pts_index = self._domain_points[index]
            except KeyError:
                # This hack was set up in __init__ for using a single list or tuple
                # with an indexed piecewise constraint. I assume this will fail if
                # None is ever allowed as a valid index for an IndexedComponent,
                # hence the assert below
                assert not (_is_indexed and (index is None))
                _self_domain_pts_index = self._domain_points[None]

        if self._unbounded_domain_var is False:
            # We add the requirment that the domain variable used by Piecewise is
            # always bounded from above and below.
            if (_self_xvar.lb is None) or (_self_xvar.ub is None):
                msg = "Piecewise '%s[%s]' found an unbounded variable "\
                      "used for the constraint domain: '%s'. "\
                      "Piecewise component requires the domain variable have "\
                      "lower and upper bounds. Refer to the Piecewise help "\
                      "documentation for information on how to disable this "\
                      "restriction"
                raise ValueError(msg % (self.name, index, _self_xvar))

        if self._warn_domain_coverage is True:
            # Print a warning when the feasible region created by the piecewise
            # constraints does not include the domain variables bounds
            if (_self_xvar.lb is not None) and (_self_xvar.lb < min(_self_domain_pts_index)):
                msg = "**WARNING: Piecewise '%s[%s]' feasible region does not "\
                    "include the lower bound of domain variable: %s.lb = %s < %s. "\
                    "Refer to the Piecewise help documentation for information on "\
                    "how to disable this warning."
                print(msg % ( self.name, index, _self_xvar, _self_xvar.lb,
                              min(_self_domain_pts_index) ))
            if (_self_xvar.ub is not None) and (_self_xvar.ub > max(_self_domain_pts_index)):
                    msg = "**WARNING: Piecewise '%s[%s]' feasible region does not "\
                        "include the upper bound of domain variable: %s.ub = %s > %s. "\
                        "Refer to the Piecewise help documentation for information on "\
                        "how to disable this warning."
                    print(msg % ( self.name, index, _self_xvar, _self_xvar.ub,
                                  max(_self_domain_pts_index) ))

        if len(_self_domain_pts_index) <= 1:
            # TODO: Technically one could interpret this
            #       case by adding simple constraints that
            #       fix the domain and range variable to the
            #       single (x,y) point that is given. This
            #       seems like it would be a bug more often
            #       than not, so I don't believe it should
            #       be the default behavior.
            raise ValueError(
                "Piecewise component '%s[%s]' failed to construct "
                "piecewise representation. List of breakpoints "
                "must contain at least two elements. Current list: %s"
                % (self.name, index, str(_self_domain_pts_index)))

        # generate the list of range values using the function rule
        # check if convexity or concavity holds as well
        force_simple = False
        if not _is_indexed:
            character,range_pts,isStep=_characterize_function(self.name,
                                                              self._warning_tol,
                                                              self._f_rule,
                                                              _self_parent,
                                                              _self_domain_pts_index)
        else:
            character,range_pts,isStep=_characterize_function(self.name,
                                                              self._warning_tol,
                                                              self._f_rule,
                                                              _self_parent,
                                                              _self_domain_pts_index,
                                                              index)


        assert not ((isStep) and (character in [-1,1]))
        if (isStep) and \
           (self._pw_rep in [PWRepn.MC, PWRepn.BIGM_BIN, PWRepn.BIGM_SOS1]):
            msg = "Piecewise '%s[%s]' has detected a step function but the selected "\
                  "piecewise representation '%s' does not currently support this "\
                  "functionality. Refer to the Piecewise help documentation for "\
                  "information about which piecewise representations support step functions."
            raise ValueError(msg % (self.name, index, self._pw_rep))

        # Make automatic simplications to the piecewise constraints
        # for the special cases of convexity and lower bound
        # or concavity and upper bound
        if (character == -1):
            if (self._bound_type == Bound.Upper):
                force_simple = True
        elif (character == 1):
            if (self._bound_type == Bound.Lower):
                force_simple = True

        # make sure the user does not want to disable the automatic
        # simplifications above
        if self._force_pw is True:
            force_simple = False

        func = None
        if force_simple is True:
            # In the case where the feasible region is convex or
            # concave (and the user does not want to force the use of
            # piecewise constraints) use simple linear constraints.
            func = _SimplifiedPiecewise()
        else:
            if len(_self_domain_pts_index) == 2:
                # Always use a simple single line constraint when
                # only two points are present in the piecewise list
                func = _SimpleSinglePiecewise()
            else:
                # generate piecewise constraints
                if self._pw_rep == PWRepn.SOS2:
                    func = _SOS2Piecewise()
                elif self._pw_rep == PWRepn.INC:
                    func = _INCPiecewise()
                elif self._pw_rep == PWRepn.MC:
                    func = _MCPiecewise()
                elif self._pw_rep == PWRepn.DCC:
                    func = _DCCPiecewise()
                elif self._pw_rep == PWRepn.DLOG:
                    func = _DLOGPiecewise()
                elif self._pw_rep == PWRepn.CC:
                    func = _CCPiecewise()
                elif self._pw_rep == PWRepn.LOG:
                    func = _LOGPiecewise()
                elif self._pw_rep == PWRepn.BIGM_BIN:
                    func = _BIGMPiecewise(binary=True)
                elif self._pw_rep == PWRepn.BIGM_SOS1:
                    func = _BIGMPiecewise(binary=False)
                else:
                    msg = "Piecewise '%s[%s]' does not have a valid "\
                          "piecewise representation: '%s'"
                    raise ValueError(msg % (self.name, index, self._pw_rep))

        if _is_indexed:
            comp = _PiecewiseData(self)
        else:
            comp = self
        self._data[index] = comp
        comp.updateBoundType(self._bound_type)
        comp.updatePoints(_self_domain_pts_index,range_pts)
        comp.build_constraints(func,_self_xvar,_self_yvar)

class SimplePiecewise(_PiecewiseData,Piecewise):

    def __init__(self, *args, **kwds):
        _PiecewiseData.__init__(self,self)
        Piecewise.__init__(self, *args, **kwds)

class IndexedPiecewise(Piecewise):

    def __init__(self,*args,**kwds):
        Piecewise.__init__(self,*args,**kwds)

    def __str__(self):
        return str(self.name)

