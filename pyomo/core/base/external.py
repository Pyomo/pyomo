#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import types
import weakref

from pyomo.core.base.numvalue import as_numeric
from pyomo.core.base.component import Component
from pyomo.core.base.expr import (_ExpressionBase,
                                  _ExternalFunctionExpression)

__all__  = ( 'ExternalFunction', )

try:
    basestring
except:
    basestring = str

class ExternalFunction(Component):
    def __new__(cls, *args, **kwds):
        if cls != ExternalFunction:
            return super(ExternalFunction, cls).__new__(cls)
        if len(args) == 1 and type(args[0]) is types.FunctionType:
            return PythonCallbackFunction.__new__(PythonCallbackFunction)
        elif not args and 'library' not in kwds and 'function' in kwds and \
                type(kwds['function']) is types.FunctionType:
            return PythonCallbackFunction.__new__(PythonCallbackFunction)
        else:
            return BasicExternalFunction.__new__(BasicExternalFunction)

    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', ExternalFunction)
        Component.__init__(self, **kwds)
        self._constructed = True
        ### HACK ###
        # FIXME: We must declare an _index attribute because
        # block._add_temporary_set assumes ALL components define an
        # index.  Sigh.
        self._index = None

    def __call__(self, *args):
        idxs = range(len(args))
        idxs.reverse()
        for i in idxs:
            if type(args[i]) is types.GeneratorType:
                args = args[:i] + tuple(args[i]) + args[i+1:]
        return _ExternalFunctionExpression( self, tuple(
                _external_fcn__clone_if_needed(x) if isinstance(x, _ExpressionBase)
                else x if isinstance(x, basestring)
                else as_numeric(x)
                for x in args ) )

    def cname(self, fully_qualified=False, name_buffer=None):
        if self.name:
            return super(ExternalFunction, self).cname(
                fully_qualified, name_buffer )
        else:
            return str(self._library) + ":" + str(self._function)
        
    def evaluate(self, args):
        raise NotImplementedError(
            "General external functions can not be evaluated within Python." )


class BasicExternalFunction(ExternalFunction):
    def __init__(self, *args, **kwds):
        if args:
            raise ValueError(
                "BasicExternalFunction constructor does not support "
                "positional arguments" )
        self._library = kwds.pop('library', None)
        self._function = kwds.pop('function', None)
        ExternalFunction.__init__(self, *args, **kwds)



class PythonCallbackFunction(ExternalFunction):
    global_registry = {}

    @classmethod
    def register_instance(instance):
        _id = len(PythonCallbackFunction.global_registry)
        global_registry[_id] = weakref.ref(instance)
        return _id

    def __init__(self, *args, **kwds):
        if args:
            self._fcn = args[0]
            if len(args) > 1:
                raise ValueError(
                    "PythonCallbackFunction constructor only supports a "
                    "single positional positional arguments" )
        if not args:
            self._fcn = kwds.pop('function')
        if kwds:
            raise ValueError(
                "PythonCallbackFunction constructor does not support "
                "keyword arguments" )
        self._library = 'pyomo_ampl.so'
        self._function = 'pyomo_socket_server'
        ExternalFunction.__init__(self, *args, **kwds)
        self._fcn_id = PythonCallbackFunction.register_instance(self)

    def __call__(self, *args):
        return super(PythonCallbackFunction, self).__call__(self._fcn_id, *args)

    def cname(self, fully_qualified=False, name_buffer=None):
        if self.name:
            return super(ExternalFunction, self).cname(
                fully_qualified, name_buffer )
        else:
            return "PythonCallback(%s)" % str(self._fcn)

    def evaluate(self, args):
        # Skip the library name and function name
        # (we assume that the identifier points to this function!)
        args.next() # library name
        args.next() # function name
        _id = args.next() # global callback function identifyer
        if _id != self._fcn_id:
            raise RuntimeError(
                "PythonCallbackFunction called with invalid Global ID" )
        return self._fcn(*tuple(args))
