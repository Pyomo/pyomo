#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ( 'DerivativeVar', 'InputVar', 'StateVar', )

import weakref

from pyomo.core.base.component import ComponentData, register_component
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.sparse_indexed_component import SparseIndexedComponent
from pyomo.dae.diffset import DifferentialSet

class _DifferentialVarData(ComponentData, NumericValue):

    __slots__ = ()

    def __init__(self, owner):
        """Constructor"""
        ComponentData.__init__(self, owner)

    def __getstate__(self):
        state = super(_DifferentialVarData, self).__getstate__()
        for i in _DifferentialVarData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.
    # def __setstate__(self, state):
    #     pass

    def is_constant(self):
        return False

    def is_fixed(self):
        return False

    def __nonzero__(self):
        """Return True if the value is defined and non-zero."""
        return True

    __bool__ = __nonzero__

    def __call__(self, exception=True):
        """Return the value of this object."""
        raise RuntimeError("_DifferentialVarData has not been flushed out yet:"
                           " value() not defined.")

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:         #pragma:nocover
            ostream = sys.stdout
        print >>ostream, str(self),


class DifferentialVar(SparseIndexedComponent):

    def __init__(self, *args, **kwds):
        #
        self._diffset = kwds.pop("diffset", None)
        if self._diffset is None:
            for i,s in enumerate(args):
                if s.__class__ is DifferentialSet:
                    self._diffset = s
                    args = args[:i] + args[i+1:]
                    self._diffIndex = i
                    break
        else:
            self._diffIndex = 0
        if self._diffset is None:
            raise ValueError("InputVar must be indexed by a DifferentialSet")
        SparseIndexedComponent.__init__(self, *args, **kwds)

    def _default(self, idx):
        return self._data.setdefault(idx, _DifferentialVarData(self))

    def __getitem__(self, ndx):
        try:
            return super(DifferentialVar, self).__getitem__(ndx)
        except KeyError:
            pass
        idx = ndx[:self._diffIndex] + ndx[self._diffIndex+1:]
        return super(DifferentialVar, self).\
            __getitem__(idx)[ndx[self._diffIndex]]



class StateVar(DifferentialVar):
    def __new__(cls, *args, **kwds):
        if cls != StateVar:
            return super(StateVar, cls).__new__(cls)
        largs = len(args)
        if 'diffset' not in kwds:
            largs -= 1
        if largs == 0:
            return _ScalarStateVar.__new__(_ScalarStateVar)
        else:
            return _IndexedStateVar.__new__(_IndexedStateVar)

    def __init__(self, *args, **kwds):
        self._derivative = None
        kwds.setdefault('ctype', StateVar)
        DifferentialVar.__init__(self, *args, **kwds)

    def derivative(self):
        return self._derivative

class _ScalarStateVar(_DifferentialVarData, StateVar):
    def __init__(self, *args, **kwds):
        StateVar.__init__(self, *args, **kwds)
        _DifferentialVarData.__init__(self, self)

class _IndexedStateVar(StateVar):
    pass



class InputVar(DifferentialVar):
    def __new__(cls, *args, **kwds):
        if cls != InputVar:
            return super(InputVar, cls).__new__(cls)
        largs = len(args)
        if 'diffset' not in kwds:
            largs -= 1
        if largs == 0:
            return _ScalarInputVar.__new__(_ScalarInputVar)
        else:
            return _IndexedInputVar.__new__(_IndexedInputVar)

    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', InputVar)
        DifferentialVar.__init__(self, *args, **kwds)

class _ScalarInputVar(_DifferentialVarData, InputVar):
    def __init__(self, *args, **kwds):
        InputVar.__init__(self, *args, **kwds)
        _DifferentialVarData.__init__(self, self)

class _IndexedInputVar(InputVar):
    pass



class DerivativeVar(DifferentialVar):
    def __new__(cls, *args, **kwds):
        if cls != DerivativeVar:
            return super(DerivativeVar, cls).__new__(cls)
        if len(args) != 1:
            raise ValueError(
                "DerivativeVar must be created with a single argument: "
                "the StateVar that this is the derivative of")
        if args[0].is_indexed():
            return _IndexedDerivativeVar.__new__(_IndexedDerivativeVar)
        else:
            return _ScalarDerivativeVar.__new__(_ScalarDerivativeVar)

    def __init__(self, stateVar, **kwds):
        if stateVar.derivative() is not None:
            raise ValueError(
                "Cannot create a new derivative variable for State variable "
                "%s: derivative already defined as %s" 
                % ( stateVar.cname(True), stateVar.derivative().cname(True) ) )
        stateVar._derivative = weakref.ref(self)
        self._stateVar = stateVar
        kwds['diffset'] = stateVar._diffset
        kwds.setdefault('ctype', DerivativeVar)
        if stateVar.is_indexed():
            DifferentialVar.__init__(self, stateVar.index_set(), **kwds)
        else:
            DifferentialVar.__init__(self, **kwds)
        self._diffIndex = stateVar._diffIndex

class _ScalarDerivativeVar(_DifferentialVarData, DerivativeVar):
    def __init__(self, *args, **kwds):
        DerivativeVar.__init__(self, *args, **kwds)
        _DifferentialVarData.__init__(self, self)

class _IndexedDerivativeVar(DerivativeVar):
    pass


register_component(StateVar, "State variable in a DAE model.")
register_component(InputVar, "Input or control variable in a DAE model.")
register_component(DerivativeVar, "Derivative of a state variable in a DAE model.")

