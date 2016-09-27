#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ( 'DerivativeVar', 'DAE_Error', )

import weakref

from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.component import register_component
from pyomo.dae.contset import ContinuousSet

from six import iterkeys

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

def derivative(self,*args):
    """
    This function allows the user to use a derivative without
    declaring a DerivativeVar. It will add a DerivativeVar component
    to the model if one doesn't already exist. The argument to this
    method is the ContinuousSet(s) that the derivative is being taken
    with respect to.
    """

    wrt = [i for i in args]
    svar = self.parent_component()
    idx = self.index()

    try:
        num_contset = len(svar._contset)
    except:
        svar._contset = {}
        svar._derivative = {}
        if svar.dim() == 0:
            raise DAE_Error("The variable %s is not indexed by any ContinuousSets. A derivative may "
                            "only be taken with respect to a continuous domain" % (svar))
        elif svar.dim() == 1:
            sidx_sets = svar._index
            if sidx_sets.type() is ContinuousSet:
                svar._contset[sidx_sets] = 0
        else:
            sidx_sets = svar._implicit_subsets
            for i,s in enumerate(sidx_sets):
                if s.type() is ContinuousSet:
                    svar._contset[s] = i
            num_contset = len(svar._contset)

    if len(args)==0:
        if  num_contset != 1:
            raise ValueError(
                "The Var %s is indexed by multiple ContinuousSets. The desired "
                "ContinuousSet must be specified" % (svar))
        args = ( self.model().find_component(next(iterkeys(svar._contset))), )
    try:
        deriv = self.model().find_component(svar.get_derivative(*args))
    except:
        nme = '_'
        for i in args:
            nme = nme+'d'+i.local_name
        self.model().add_component('d'+svar.local_name+nme,DerivativeVar(svar,wrt=args))
        deriv = svar.get_derivative(*args)

    try:
        return deriv[idx]
    except:
        new_indices = set(deriv._index)-set(iterkeys(deriv._data))
        deriv._add_members(new_indices)
        deriv._initialize_members(new_indices)
        return deriv[idx]

def get_derivative(self,*args):
    """
    Returns the dictionary mapping derivatives to their DerivativeVar or
    returns a certain DerivativeVar specified by the keyword arguments
    """
    try:
        if len(args) == 0:
            return self._derivative
        else:
            key = [str(i) for i in args]
            key.sort()
            key = tuple(key)
            return self._derivative[key]()
    except AttributeError:
        return {}

def is_fully_discretized(self):
    """
    Checks to see if all ContinuousSets indexing this Var have been
    discretized
    """
    for i in self._contset:
        if 'scheme' not in i.get_discretization_info():
            return False
        return True

# The following code adds a few methods and attributes to Var and
# _VarData which allow them to be indexed by discretized and
# differentiated. All Vars in the model will have these methods and
# attributes after importing pyomo.dae.

_VarData.derivative = derivative
Var.get_derivative = get_derivative
Var.is_fully_discretized = is_fully_discretized

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

    def __init__(self, sVar, **kwds):

        if not isinstance(sVar,Var):
            raise DAE_Error(
                "%s is not a variable. Can only take the derivative of a Var component." % (sVar))
        if "wrt" in kwds and "withrespectto" in kwds:
            raise TypeError(
                "Cannot specify both 'wrt' and 'withrespectto keywords "
                "in a DerivativeVar")

        wrt = kwds.pop('wrt',None)
        wrt = kwds.pop('withrespectto',wrt)

        try:
            num_contset = len(sVar._contset)
        except:
            sVar._contset = {}
            sVar._derivative = {}
            if sVar.dim() == 0:
                num_contset = 0
            elif sVar.dim() == 1:
                sidx_sets = sVar._index
                if sidx_sets.type() is ContinuousSet:
                    sVar._contset[sidx_sets] = 0
            else:
                sidx_sets = sVar._implicit_subsets
                for i,s in enumerate(sidx_sets):
                    if s.type() is ContinuousSet:
                        sVar._contset[s] = i
            num_contset = len(sVar._contset)

        if num_contset == 0:
            raise DAE_Error("The variable %s is not indexed by any ContinuousSets. A derivative may "
                            "only be taken with respect to a continuous domain" % (sVar))

        if wrt == None:
            # Check to be sure Var is indexed by single ContinuousSet and take
            # first deriv wrt that set
            if num_contset != 1:
                raise DAE_Error(
                    "The variable %s is indexed by multiple ContinuousSets. The desired "
                    "ContinuousSet must be specified using the keyword argument 'wrt'" % (sVar))
            wrt = [next(iterkeys(sVar._contset)),]
        elif type(wrt) is ContinuousSet:
            if wrt not in sVar._contset:
                raise DAE_Error(
                    "Invalid derivative: The variable %s is not indexed by "
                    "the ContinuousSet %s" %(sVar,wrt))
            wrt = [wrt,]
        elif type(wrt) is tuple or type(wrt) is list:
            for i in wrt:
                if type(i) is not ContinuousSet:
                    raise DAE_Error(
                        "Cannot take the derivative with respect to %s. "
                        "Expected a ContinuousSet or a tuple of ContinuousSets"% (i))
                if i not in sVar._contset:
                    raise DAE_Error(
                        "Invalid derivative: The variable %s is not indexed by "
                        "the ContinuousSet %s" %(sVar,i))
            wrt = list(wrt)
        else:
            raise DAE_Error(
                "Cannot take the derivative with respect to %s. "
                "Expected a ContinuousSet or a tuple of ContinuousSets"% (i))

        wrtkey = [str(i) for i in wrt]
        wrtkey.sort()
        wrtkey = tuple(wrtkey)

        if wrtkey in sVar._derivative:
            raise DAE_Error(
                "Cannot create a new derivative variable for variable "
                "%s: derivative already defined as %s"
                % ( sVar.name, sVar.get_derivative(*tuple(wrt)).name ) )

        sVar._derivative[wrtkey] = weakref.ref(self)
        self._sVar = sVar
        self._wrt = wrt

        kwds.setdefault('ctype', DerivativeVar)

        if sVar._implicit_subsets is None:
            arg = (sVar.index_set(),)
        else:
            arg = tuple(sVar._implicit_subsets)

        Var.__init__(self,*arg,**kwds)

    def get_continuousset_list(self):
        return self._wrt

    def is_fully_discretized(self):
        """
        Check to see if all the ContinuousSets this derivative is taken with
        respect to have been discretized.
        """
        for i in self._wrt:
            if 'scheme' not in i.get_discretization_info():
                return False
        return True

    def get_state_var(self):
        return self._sVar

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
            self._expr = create_access_function(self._sVar)
            return self._expr

    def set_derivative_expression(self,expr):
        self._expr = expr

register_component(DerivativeVar, "Derivative of a State variable in a DAE model.")
