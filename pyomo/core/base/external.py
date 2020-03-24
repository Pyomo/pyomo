#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import os
import six
import types
import weakref

from ctypes import (
    Structure, POINTER, CFUNCTYPE, cdll, byref,
    c_int, c_long, c_ulong, c_double, c_byte, c_char_p, c_void_p )

from pyomo.core.expr.numvalue import native_types, NonNumericValue
from pyomo.core.expr import current as EXPR
from pyomo.core.base.component import Component

__all__  = ( 'ExternalFunction', )

try:
    basestring
except:
    basestring = str

logger = logging.getLogger('pyomo.core')


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
            return AMPLExternalFunction.__new__(AMPLExternalFunction)

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
        args_ = []
        for arg in args:
            if type(arg) is types.GeneratorType:
                args_.extend(val for val in arg)
            else:
                args_.append(arg)
        #
        # Loop and do two thing:
        #   1. Wrap non-numeric arguments
        #   2. See if we have a potentially variable argument
        #
        pv = False
        for i,arg in enumerate(args_):
            try:
                # Q: Is there a better way to test if a value is an object
                #    not in native_types and not a standard expression type?
                if arg.__class__ in native_types or arg.is_expression_type():
                    pass
                if not arg.__class__ in native_types and arg.is_potentially_variable():
                    pv = True
            except AttributeError:    
                args_[i] = NonNumericValue(arg)
        #
        if pv:
            return EXPR.ExternalFunctionExpression(args_, self)
        return EXPR.NPV_ExternalFunctionExpression(args_, self)

    def evaluate(self, args):
        raise NotImplementedError(
            "General external functions can not be evaluated within Python." )


class AMPLExternalFunction(ExternalFunction):

    def __init__(self, *args, **kwds):
        if args:
            raise ValueError(
                "AMPLExternalFunction constructor does not support "
                "positional arguments" )
        self._library = kwds.pop('library', None)
        self._function = kwds.pop('function', None)
        self._known_functions = None
        self._so = None
        ExternalFunction.__init__(self, *args, **kwds)

    def _evaluate(self, args, fgh, fixed):
        if self._so is None:
            self.load_library()
        if self._function not in self._known_functions:
            raise RuntimeError(
                "Error: external function %s was not registered within "
                "external library %s.\n\tAvailable functions: (%s)"
                % ( self._function, self._library,
                    ', '.join(self._known_functions.keys()) ) )
        #
        args_ = []
        for arg in args:
            if arg.__class__ is NonNumericValue:
                args_.append(arg.value)
            else:
                args_.append(arg)
        arglist = _ARGLIST( args_, fgh, fixed )
        fcn = self._known_functions[self._function][0]
        f = fcn(byref(arglist))
        return f, arglist

    def evaluate(self, args):
        return self._evaluate(args, 0, None)[0]

    def evaluate_fgh(self, args, fixed=None):
        if fixed is None:
            fixed = tuple( arg.__class__ in native_types or arg.is_fixed()
                           for arg in args )
        N = len(args)
        f, arglist = self._evaluate(args, 2, fixed)
        g = [arglist.derivs[i] for i in range(N)]
        h = [arglist.hes[i] for i in range((N+N**2)//2)]
        return f, g, h

    def load_library(self):
        try:
            self._so = cdll.LoadLibrary(self._library)
        except OSError:
            # On Linux, it is uncommon for "." to be in the
            # LD_LIBRARY_PATH, so if things fail to load, attempt to
            # locate the library via a relative path
            try:
                self._so = cdll.LoadLibrary(os.path.join('.',self._library))
            except OSError:
                # Re-try with the original library name and allow the
                # exception to propagate up.
                self._so = cdll.LoadLibrary(self._library)

        self._known_functions = {}
        AE = _AMPLEXPORTS()
        AE.ASLdate = 20160307
        def addfunc(name, f, _type, nargs, funcinfo, ae):
            # trap for Python 3, where the name comes in as bytes() and
            # not a string
            if not isinstance(name, six.string_types):
                name = name.decode()
            self._known_functions[str(name)] = (f, _type, nargs, funcinfo, ae)
        AE.Addfunc = _AMPLEXPORTS.ADDFUNC(addfunc)
        def addrandinit(ae, rss, v):
            # TODO: This should support the randinit ASL option
            rss(v, 1)
        AE.Addrandinit = _AMPLEXPORTS.ADDRANDINIT(addrandinit)
        def atreset(ae, a, b):
            logger.warning(
                "AMPL External function: ignoring AtReset call in external "
                "library.  This may result in a memory leak or other "
                "undesirable behavior.")
        AE.AtReset = _AMPLEXPORTS.ATRESET(atreset)

        FUNCADD = CFUNCTYPE( None, POINTER(_AMPLEXPORTS) )
        FUNCADD(('funcadd_ASL', self._so))(byref(AE))


class PythonCallbackFunction(ExternalFunction):
    global_registry = {}

    @classmethod
    def register_instance(cls, instance):
        _id = len(cls.global_registry)
        cls.global_registry[_id] = weakref.ref(instance)
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

    def evaluate(self, args):
        if args.__class__ is types.GeneratorType:
            args = tuple(args)
        args_ = []
        for arg in args:
            if arg.__class__ is NonNumericValue:
                args_.append(arg.value)
            else:
                args_.append(arg)
        _id = args_[0]
        if _id != self._fcn_id:
            raise RuntimeError(
                "PythonCallbackFunction called with invalid Global ID" )
        return self._fcn(*args_[1:])


class _ARGLIST(Structure):
    """Mock up the arglist structure from AMPL's funcadd.h

    This data structure is populated by AMPL when calling external
    functions (for both passing in the function arguments and retrieving
    the derivative/Hessian information.
    """

    _fields_ = [
        ('n', c_int),     # number of args
        ('nr', c_int),    # number of real input args
        ('at', POINTER(c_int)),  # argument types -- see DISCUSSION below
        ('ra', POINTER(c_double)),  # pure real args (IN, OUT, and INOUT)
        ('sa', POINTER(c_char_p)),  # symbolic IN args
        ('derivs', POINTER(c_double)),  # for partial derivatives (if nonzero)
        ('hes', POINTER(c_double)),  # for second partials (if nonzero)
        ('dig', POINTER(c_byte)),   # if (dig && dig[i]) { partials w.r.t.
                                    #	ra[i] will not be used }
        ('funcinfo', c_char_p),  # for use by the function (if desired)
        ('AE', c_void_p), # functions made visible (via #defines below)
        ('f', c_void_p),  # for internal use by AMPL
        ('tva', c_void_p),  # for internal use by AMPL
        ('Errmsg', c_char_p),  # To indicate an error, set this to a
			     # description of the error.  When derivs
                             # is nonzero and the error is that first
			     # derivatives cannot or are not computed,
			     # a single quote character (') should be
			     # the first character in the text assigned
			     # to Errmsg, followed by the actual error
			     # message.  Similarly, if hes is nonzero
                             # and the error is that second derivatives
			     # are not or cannot be computed, a double
			     # quote character (") should be the first
			     # character in Errmsg, followed by the
			     # actual error message text.
        ('TMI', c_void_p),     # used in Tempmem calls
        ('Private', c_char_p), # The following fields are relevant
			     # only when imported functions are called
			     # by AMPL commands (not declarations).
        ('nin', c_int),   # number of input (IN and INOUT) args
        ('nout', c_int),  # number of output (OUT and INOUT) args
        ('nsin', c_int),  # number of symbolic input arguments
        ('nsout', c_int), # number of symbolic OUT and INOUT args
        ]

    def __init__(self, args, fgh=0, fixed=None):
        super(_ARGLIST, self).__init__()
        N = len(args)
        self.n = self.nr = N
        self.at = (c_int*N)()
        self.ra = (c_double*N)()
        self.sa = None
        if fgh > 0:
            self.derivs = (c_double*N)(0.)
        if fgh > 1:
            self.hes = (c_double*((N+N*N)//2))(0.)

        for i,v in enumerate(args):
            self.at[i] = i
            self.ra[i] = v

        if fixed:
            self.dig = (c_byte*N)(0)
            for i,v in enumerate(fixed):
                if v:
                    self.dig[i] = 1

# The following "fake" class resolves a circular reference issue in the
# _AMPLEXPORTS datastructure
#
# Declare a dummy structure called _AMPLEXPORTS so that the symbol exists
# and the POINTER(_AMPLEXPORTS) call in the definition of ADDFUNC (needed
# to define AMPLExPORTS) will succeed.
class _AMPLEXPORTS(Structure):
    pass

class _AMPLEXPORTS(Structure):
    """Mock up the AmplExports structure from AMPL's funcadd.h

    The only thing we really need here is to be able to override the
    Addfunc function pointer to point back into Python so we can
    intercept the registration of external AMPL functions.  Ideally, we
    would populate the other function pointers as well, but that is
    trickier than it sounds, and at least so far is completely unneeded.
    """

    AMPLFUNC = CFUNCTYPE( c_double, POINTER(_ARGLIST) )

    ADDFUNC = CFUNCTYPE(
        None,
        c_char_p, AMPLFUNC, c_int, c_int, c_void_p,
        POINTER(_AMPLEXPORTS) )

    RANDSEEDSETTER = CFUNCTYPE(
        None,
        c_void_p, c_ulong )

    ADDRANDINIT = CFUNCTYPE(
        None,
        POINTER(_AMPLEXPORTS), RANDSEEDSETTER, c_void_p )

    ATRESET = CFUNCTYPE(
        None,
        POINTER(_AMPLEXPORTS), c_void_p, c_void_p )

    _fields_ = [
	('StdErr', c_void_p),
	('Addfunc', ADDFUNC),
	('ASLdate', c_long),
	('FprintF', c_void_p),
	('PrintF', c_void_p),
	('SprintF', c_void_p),
	('VfprintF', c_void_p),
	('VsprintF', c_void_p),
	('Strtod', c_void_p),
	('Crypto', c_void_p),
	('asl', c_char_p),
	('AtExit', c_void_p),
	('AtReset', ATRESET),
	('Tempmem', c_void_p),
	('Add_table_handler', c_void_p),
	('Private', c_char_p),
	('Qsortv', c_void_p),
	#/* More stuff for stdio in DLLs... */
	('StdIn', c_void_p),
	('StdOut', c_void_p),
	('Clearerr', c_void_p),
	('Fclose', c_void_p),
	('Fdopen', c_void_p),
	('Feof', c_void_p),
	('Ferror', c_void_p),
	('Fflush', c_void_p),
	('Fgetc', c_void_p),
	('Fgets', c_void_p),
	('Fileno', c_void_p),
	('Fopen', c_void_p),
	('Fputc', c_void_p),
	('Fputs', c_void_p),
	('Fread', c_void_p),
	('Freopen', c_void_p),
	('Fscanf', c_void_p),
	('Fseek', c_void_p),
	('Ftell', c_void_p),
	('Fwrite', c_void_p),
	('Pclose', c_void_p),
	('Perror', c_void_p),
	('Popen', c_void_p),
	('Puts', c_void_p),
	('Rewind', c_void_p),
	('Scanf', c_void_p),
	('Setbuf', c_void_p),
	('Setvbuf', c_void_p),
	('Sscanf', c_void_p),
	('Tempnam', c_void_p),
	('Tmpfile', c_void_p),
	('Tmpnam', c_void_p),
	('Ungetc', c_void_p),
	('AI', c_void_p),
	('Getenv', c_void_p),
	('Breakfunc', c_void_p),
	('Breakarg', c_char_p),
	#/* Items available with ASLdate >= 20020501 start here. */
	('SnprintF', c_void_p),
	('VsnprintF', c_void_p),
	('Addrand', c_void_p),
	('Addrandinit', ADDRANDINIT),
	]
