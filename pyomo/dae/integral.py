#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('Integral', )

from pyomo.core.base.component import register_component
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DAE_Error
from pyomo.core.base.expression import (Expression,
_GeneralExpressionData, SimpleExpression)

def create_access_function(var):
    """
    This method returns a function that returns a component by calling
    it rather than indexing it
    """
    def _fun(*args):
        return var[args]
    return _fun

def create_partial_expression(scheme,expr,ind,loc):
    """
    This method returns a function which applies a discretization scheme
    to an expression along a particular indexind set. This is admittedly a
    convoluted looking implementation. The idea is that we only apply a
    discretization scheme to one indexing set at a time but we also want
    the function to be expanded over any other indexing sets.
    """
    def _fun(*args):
        return scheme(lambda i: expr(*(args[0:loc]+(i,)+args[loc+1:])),ind)
    return lambda *args:_fun(*args)(args[loc])


class Integral(Expression):

    def __new__(cls, *args, **kwds):
        if cls != Integral:
            return super(Integral, cls).__new__(cls)
        if len(args) == 0:
            raise ValueError("Integral must be indexed by a ContinuousSet")
        elif len(args) == 1:
            return SimpleIntegral.__new__(SimpleIntegral)
        else:
            return IndexedIntegral.__new__(IndexedIntegral)

    def __init__(self, *args, **kwds):

        if "wrt" in kwds and "withrespectto" in kwds:
            raise TypeError(
                "Cannot specify both 'wrt' and 'withrespectto keywords "
                "in a DerivativeVar")

        wrt = kwds.pop('wrt',None)
        wrt = kwds.pop('withrespectto',wrt)

        if wrt == None:
            # Check to be sure Integral is indexed by single
            # ContinuousSet and take Integral with respect to that
            # ContinuousSet
            if len(args) != 1:
                raise ValueError(
                    "The Integral %s is indexed by multiple ContinuousSets. The desired "
                    "ContinuousSet must be specified using the keyword argument 'wrt'" % (self.name))
            wrt = args[0]

        if type(wrt) is not ContinuousSet:
            raise ValueError(
                "Cannot take the integral with respect to '%s'. Must take an integral "\
                "with respect to a ContinuousSet" %(wrt))
        self._wrt = wrt

        loc = None
        for i,s in enumerate(args):
            if s is wrt:
                loc = i

        # Check that the wrt ContinuousSet is in the argument list
        if loc is None:
            raise ValueError(
                "The ContinuousSet '%s' was not found in the indexing sets of the "
                "Integral '%s'" %(wrt.name,self.name))
        self.loc = loc

        # Remove the index that the integral is being expanded over
        arg = args[0:loc]+args[loc+1:]

        # Check that if bounds are given
        bounds = kwds.pop('bounds',None)
        if bounds is not None:
            raise DAE_Error(
                "Setting bounds on integrals has not yet been implemented. Integrals may only be "\
                "taken over an entire ContinuousSet")

        # Create integral expression and pass to the expression initialization
        intexp = kwds.pop('expr', None)
        intexp = kwds.pop('rule', intexp)
        if intexp is None:
            raise ValueError(
                "Must specify an integral expression for Integral '%s'" %(self))

        def _trap_rule(m,*a):
            ds = sorted(m.find_component(wrt.local_name))
            return sum(0.5*(ds[i+1]-ds[i])*
                      (intexp(m,*(a[0:loc]+(ds[i+1],)+a[loc:]))+intexp(m,*(a[0:loc]+(ds[i],)+a[loc:])))
                      for i in range(len(ds)-1))

        kwds['rule'] = _trap_rule    
        kwds.setdefault('ctype', Integral)
        Expression.__init__(self,*arg,**kwds)

    def get_differentialset(self):
        return self._wrt

class SimpleIntegral(_GeneralExpressionData, Integral):

    def __init__(self, *args, **kwds):
        _GeneralExpressionData.__init__(self, None, component=self)
        Integral.__init__(self,*args,**kwds)

    def is_fully_discretized(self):
        """
        Checks to see if all ContinuousSets indexing this Integral have been
        discretized
        """
        if 'scheme' not in self._wrt.get_discretization_info():
            return False
        return True

    #
    # Override abstract interface methods to first check for
    # construction
    #

    @property
    def expr(self):
        """Return expression on this expression."""
        if self._constructed:
            return _GeneralExpressionData.expr.fget(self)
        raise ValueError(
            "Accessing the expression of integral '%s' "
            "before the Integral has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    def set_value(self, expr):
        """Set the expression on this expression."""
        if self._constructed:
            return _GeneralExpressionData.set_value(self, expr)
        raise ValueError(
            "Setting the expression of integral '%s' "
            "before the Integral has been constructed (there "
            "is currently no object to set)."
            % (self.name))

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        if self._constructed:
            return _GeneralExpressionData.is_constant(self)
        raise ValueError(
            "Accessing the is_constant flag of integral '%s' "
            "before the Integral has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        if self._constructed:
            return _GeneralExpressionData.is_fixed(self)
        raise ValueError(
            "Accessing the is_fixed flag of integral '%s' "
            "before the Integral has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    #
    # Like the SimpleExpression class,
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise KeyError(
                "SimpleIntegral object '%s' does not accept "
                "index values other than None. Invalid value: %s"
                % (self.name, index))
        if (type(expr) is tuple) and \
           (expr == Expression.Skip):
            raise ValueError(
                "Expression.Skip can not be assigned "
                "to an Expression that is not indexed: %s"
                % (self.name))
        self.set_value(expr)
        return self

class IndexedIntegral(Integral):

    def is_fully_discretized(self):
        """
        Checks to see if all ContinuousSets indexing this Integral have been
        discretized.
        """
        wrt = self._wrt
        if 'scheme' not in wrt.get_discretization_info():
            return False

        setlist = []
        if self.dim() == 1:
            setlist =[self.index_set(),]
        else:
            setlist = self._implicit_subsets

        for i in setlist:
            if i.type() is ContinuousSet:
                if 'scheme' not in i.get_discretization_info():
                    return False
        return True
    #
    # Leaving this method for backward compatibility reasons
    # Note: It allows adding members outside of self._index.
    #       This has always been the case. Not sure there is
    #       any reason to maintain a reference to a separate
    #       index set if we allow this.
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if (type(expr) is tuple) and \
           (expr == Expression.Skip):
            return None
        cdata = _GeneralExpressionData(expr, component=self)
        self._data[index] = cdata
        return cdata

register_component(Integral, "Integral Expression in a DAE model.")
