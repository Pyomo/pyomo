#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ('Integral', )

import weakref

from pyomo.core.base.component import register_component
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DAE_Error
from pyomo.core.base.expression import Expression, SimpleExpression, IndexedExpression, _ExpressionData

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
            # Check to be sure Integral is indexed by single ContinuousSet and take Integral
            # with respect to that ContinuousSet
            if len(args) != 1:
                raise ValueError(
                    "The Integral %s is indexed by multiple ContinuousSets. The desired "
                    "ContinuousSet must be specified using the keyword argument 'wrt'" % (self.cname(True)))
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
                "Integral '%s'" %(wrt.cname(True),self.cname(True)))
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
            ds = sorted(m.find_component(wrt.name))         
            return sum(0.5*(ds[i+1]-ds[i])*
                      (intexp(m,*(a[0:loc]+(ds[i+1],)+a[loc:]))-intexp(m,*(a[0:loc]+(ds[i],)+a[loc:]))) 
                      for i in range(len(ds)-1))

        kwds['expr'] = _trap_rule

        kwds.setdefault('ctype', Integral)
        Expression.__init__(self,*arg,**kwds)

    def get_differentialset(self):
        return self._wrt

class SimpleIntegral(_ExpressionData,Integral):
    
    def __init__(self, *args, **kwds):
        Integral.__init__(self,*args,**kwds)
        _ExpressionData.__init__(self,self,None)

    def Xpprint(self, ostream=None, verbose=None, nested=False, eol_flag=True, precedence=0):
        # Needed so that users find Expression.pprint and not
        # _ExpressionData.pprint
        if precedence == 0:
            Expression.pprint(self, ostream=ostream, verbose=None)
        else:
            ostream.write(str(self))

    def __call__(self, exception=True):

        if self._constructed:
            return _ExpressionData.__call__(self, exception=exception)
        if exception:
            raise ValueError("Evaluating the numeric value of expression '%s' "
                             "before the Expression has been constructed (there "
                             "is currently no value to return)."
                             % self.cname(True))

    def is_fully_discretized(self):
        """
        Checks to see if all ContinuousSets indexing this Integral have been 
        discretized
        """
        if not self._wrt.get_discretization_info().has_key('scheme'):
            return False
        return True

class IndexedIntegral(Integral):
    
    def __call__(self, exception=True):
        """Compute the value of the expression"""
        if exception:
            msg = 'Cannot compute the value of an array of expressions'
            raise TypeError(msg)

    def is_fully_discretized(self):
        """
        Checks to see if all ContinuousSets indexing this Integral have been 
        discretized. 
        """
        wrt = self._wrt
        if not wrt.get_discretization_info().has_key('scheme'):
            return False

        setlist = []
        if self.dim() == 1:
            setlist =[self.index_set(),]
        else:
            setlist = self._implicit_subsets
        
        for i in setlist:
            if i.type() is ContinuousSet:              
                if not i.get_discretization_info().has_key('scheme'):
                    return False
        return True

register_component(Integral, "Integral Expression in a DAE model.")

