#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ( 'DerivativeVar', 'StateVar', 'DAE_Error', )

import weakref
from types import MethodType

from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.component import register_component
from pyomo.dae.contset import ContinuousSet

def create_access_function(var):
    """
    This method returns a function that returns a component by calling 
    it rather than indexing it
    """
    def _fun(*args):
        return var[args]
    return _fun

class DAE_Error(Exception):
    """Exception raised while processing DAE Models"""

def _derivative(self,*args):
    """
    This function allows the user to use a derivative without declaring a DerivativeVar. It will
    add a DerivativeVar component to the model if one doesn't already exist. The argument to 
    this method is the ContinuousSet(s) that the derivative is being taken with respect to.
    """
    if type(self.component()) is StateVar:
        wrt = [i for i in args]
        svar = self.component()
        idx = self.index()
        if len(args)==0:
            if len(svar._contset) != 1:
                raise ValueError(
                    "The StateVar %s is indexed by multiple ContinuousSets. The desired "
                    "ContinuousSet must be specified using the keyword argument 'wrt'" % (stateVar))
            args = (self.model().find_component(svar._contset.keys()[0]),)    
        try:
            deriv = self.model().find_component(svar.get_derivative(*args))
        except:
            nme = '_'
            for i in args:
                nme = nme+'d'+i.name
            self.model().add_component('d'+svar.name+nme,DerivativeVar(svar,wrt=args))
            deriv = svar.get_derivative(*args)

        

        try:
            return deriv[idx]
        except:
            new_indices = set(deriv._index)-set(deriv._data.keys())
            deriv._add_members(new_indices)
            deriv._initialize_members(new_indices)
            return deriv[idx]
    else:
        msg = "'%s' object has no attribute 'derivative'"
        raise AttributeError(msg %type(self).__name__)

class StateVar(Var):
    """
    A variable which the user may take the derivative or partial derivative of.
    It must be indexed by at least one ContinuousSet but is otherwise constructed
    exactly like a Var component

    Private Attributes:
    _derivative    Dictionary mapping all the derivatives of the StateVar to a DerivativeVar
    _contset       Dictionary mapping a ContinuousSet to its location in the indexing sets
                   assumes that each ContinuousSet will only appear once in the indexing
    """

    def __init__(self, *args, **kwds):

        largs = len(args)
        if largs == 0:
            raise ValueError("StateVar must be indexed by a ContinuousSet")

        self._derivative = {}
        self._contset = {} 

        for i,s in enumerate(args):
            if type(s) is ContinuousSet:
                self._contset[s] = i

        if self._contset is {}:
            raise ValueError("StateVar %s must be indexed by a ContinuousSet"%(self))

        kwds.setdefault('ctype', StateVar)        
        Var.__init__(self, *args, **kwds)

        # This is a hack to allow the user to use derivatives in their
        # model without first declaring them. It adds a method to the
        # _VarData class which means that any Vars declared after a StateVar
        # will also have this method. An error will be thrown if it is called
        # on any component other than a StateVar. A better option would be to 
        # have a _StateVarData class which inherits from _VarData and adds this
        # method just to that class. However, this approach leads to problems 
        # with creating expressions and expression trees.
        _VarData.derivative = MethodType(_derivative, None, _VarData)
        self._VarData = _VarData

    def get_derivative(self,*args):
        """
        Returns the dictionary mapping derivatives to their DerivativeVar or 
        returns a certain DerivativeVar specified by the keyword arguments
        """
        if len(args) == 0:
            return self._derivative
        else:
            key = [str(i) for i in args]
            key.sort()
            key = tuple(key)
            return self._derivative[key]()

    def is_fully_discretized(self):
        """
        Checks to see if all ContinuousSets indexing this StateVar have been 
        discretized
        """
        for i in self._contset.keys():
            if not i.get_discretization_info().has_key('scheme'):
                return False
        return True

class DerivativeVar(Var):
    """
    A variable which the derivative of a StateVar with respect to one or more ContinuousSets.
    The constructor accepts a single positional argument which is the StateVar that's being
    differentiated.

    Keyword Arguments:
    wrt, withrespectto     A ContinuousSet or a tuple(or list) of ContinuousSets that the
                           derivative is being taken with respect to. Higher order derivatives
                           are represented by including the ContinuousSet multiple times in 
                           the tuple sent to this keyword. i.e. wrt=(m.t,m.t) would be the second
                           order derivative with respect to m.t

    Private Attributes:
    _stateVar     The StateVar being differentiated
    _wrt          A list of the ContinuousSets the derivative is being taken with respect to          
    _expr         An expression representing the discretization equations linking the DerivativeVar
                  to its StateVar. 
    """

    def __init__(self, stateVar, **kwds):

        if type(stateVar) is not StateVar:
            raise ValueError(
                "%s is not a valid StateVar. Can only take the derivative of a StateVar." % (stateVar))
        if "wrt" in kwds and "withrespectto" in kwds:
            raise TypeError(
                "Cannot specify both 'wrt' and 'withrespectto keywords "
                "in a DerivativeVar")

        wrt = kwds.pop('wrt',None)
        wrt = kwds.pop('withrespectto',wrt)

        if wrt == None:
            # Check to be sure StateVar is indexed by single ContinuousSet and take
            # first deriv wrt that set
            if len(stateVar._contset) != 1:
                raise ValueError(
                    "The StateVar %s is indexed by multiple ContinuousSets. The desired "
                    "ContinuousSet must be specified using the keyword argument 'wrt'" % (stateVar))
            wrt = [stateVar._contset.keys()[0],]
        elif type(wrt) is ContinuousSet:
            if not stateVar._contset.has_key(wrt):
                raise ValueError(
                    "Invalid derivative: The StateVar %s is not indexed by "
                    "the ContinuousSet %s" %(stateVar,wrt))
            wrt = [wrt,]
        elif type(wrt) is tuple or type(wrt) is list:
            for i in wrt:
                if type(i) is not ContinuousSet:
                    raise ValueError(
                        "Cannot take the derivative with respect to %s. "
                        "Expected a ContinuousSet or a tuple of ContinuousSets"% (i))
                if not stateVar._contset.has_key(i):
                    raise ValueError(
                        "Invalid derivative: The StateVar %s is not indexed by "
                        "the ContinuousSet %s" %(stateVar,i))           
            wrt = list(wrt)
        else:
            raise ValueError(
                "Cannot take the derivative with respect to %s. "
                "Expected a ContinuousSet or a tuple of ContinuousSets"% (i))
        
        wrtkey = [str(i) for i in wrt]
        wrtkey.sort()
        wrtkey = tuple(wrtkey)

        if wrtkey in stateVar._derivative:
            raise ValueError(
                "Cannot create a new derivative variable for State variable "
                "%s: derivative already defined as %s" 
                % ( stateVar.cname(True), stateVar.get_derivative(*tuple(wrt)).cname(True) ) )
 
        stateVar._derivative[wrtkey] = weakref.ref(self)
        self._stateVar = stateVar
        self._wrt = wrt

        kwds.setdefault('ctype', DerivativeVar)

        if stateVar._implicit_subsets is None:
            arg = (stateVar.index_set(),)
        else:
            arg = tuple(stateVar._implicit_subsets)
        
        Var.__init__(self,*arg,**kwds)
        
    def get_continuousset_list(self):
        return self._wrt
  
    def is_fully_discretized(self):
        """
        Check to see if all the ContinuousSets this derivative is taken with
        respect to have been discretized.
        """
        for i in self._wrt:
            if not i.get_discretization_info().has_key('scheme'):
                return False
        return True

    def get_state_var(self):
        return self._stateVar

    def get_derivative_expression(self):
        """
        Returns the current discretization expression for this derivative or creates
        an access function to its StateVar the first time this method is called.
        The expression gets built up as the discretization transformations are 
        sequentially applied to each ContinuousSet in the model.
        """
        try:
            return self._expr
        except:
            self._expr = create_access_function(self._stateVar)
            return self._expr

    def set_derivative_expression(self,expr):
        self._expr = expr

register_component(StateVar, "State variable in a DAE model.")
register_component(DerivativeVar, "Derivative of a State variable in a DAE model.")
