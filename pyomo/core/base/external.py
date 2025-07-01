#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import types
import weakref
from pyomo.common.pyomo_typing import overload

from ctypes import (
    Structure,
    POINTER,
    CFUNCTYPE,
    cdll,
    byref,
    c_int,
    c_long,
    c_ulong,
    c_double,
    c_byte,
    c_char_p,
    c_void_p,
)

from pyomo.common.autoslots import AutoSlots
from pyomo.common.fileutils import find_library
from pyomo.common.numeric_types import (
    check_if_native_type,
    native_types,
    native_numeric_types,
    value,
    _pyomo_constant_types,
)
from pyomo.core.expr.numvalue import NonNumericValue, NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.component import Component
from pyomo.core.base.units_container import units

logger = logging.getLogger('pyomo.core')
nan = float('nan')


class ExternalFunction(Component):
    """Interface to an external (non-algebraic) function.

    :class:`ExternalFunction` provides an interface for declaring
    general user-provided functions, and then embedding calls to the
    external functions within Pyomo expressions.

    .. note::

       Just because you can express a Pyomo model with external
       functions does not mean that the resulting model is solvable.  In
       particular, linear solvers do not accept external functions.  The
       AMPL Solver Library (ASL) interface does support external
       functions for general nonlinear solvers compiled against it, but
       only allows functions in compiled libraries through the
       :class:`AMPLExternalFunction` interface.

    """

    def __new__(cls, *args, **kwargs):
        if cls is not ExternalFunction:
            return super().__new__(cls)
        elif args:
            return super().__new__(PythonCallbackFunction)
        elif 'library' not in kwargs and any(
            kw in kwargs for kw in ('function', 'fgh')
        ):
            return super().__new__(PythonCallbackFunction)
        else:
            return super().__new__(AMPLExternalFunction)

    @overload
    def __init__(self, function=None, gradient=None, hessian=None, *, fgh=None): ...

    @overload
    def __init__(self, *, library: str, function: str): ...

    def __init__(self, *args, **kwargs):
        """Construct a reference to an external function.

        There are two fundamental interfaces supported by
        :class:`ExternalFunction`: Python callback functions and AMPL
        external functions.

        **Python callback functions** (:class:`PythonCallbackFunction`
        interface)

        Python callback functions can be specified one of two ways:

        1. FGH interface:

          A single external function call with a signature matching the
          :meth:`evaluate_fgh()` method.

        2. Independent functions:

          One to three functions that can evaluate the function value,
          gradient of the function [partial derivatives] with respect to
          its inputs, and the Hessian of the function [partial second
          derivatives].  The ``function`` interface expects a function
          matching the prototype:

          .. code::

             def function(*args): float

          The ``gradient`` and ``hessian`` interface expect functions
          matching the prototype:

          .. code::

             def gradient_or_hessian(args, fixed=None): List[float]

          Where ``args`` is a tuple of function arguments and ``fixed``
          is either None or a list of values equal in length to ``args``
          indicating which arguments are currently fixed (``True``) or
          variable (``False``).

        **ASL function libraries** (:class:`AMPLExternalFunction` interface)

        Pyomo can also call functions compiled as part of an AMPL
        External Function library (see the `Imported functions` section
        in the `Hooking your solver to AMPL
        <https://www.ampl.com/REFS/hooking3.pdf>`_ report).  Links to
        these functions are declared by creating an
        :class:`ExternalFunction` and passing the compiled library name
        (or path) to the ``library`` keyword and the name of the
        function to the ``function`` keyword.

        """
        self._units = kwargs.pop('units', None)
        if self._units is not None:
            self._units = units.get_units(self._units)
        self._arg_units = kwargs.pop('arg_units', None)
        if self._arg_units is not None:
            self._arg_units = [units.get_units(u) for u in self._arg_units]
        kwargs.setdefault('ctype', ExternalFunction)
        Component.__init__(self, **kwargs)
        self._constructed = True
        ### HACK ###
        # FIXME: We must declare an _index attribute because
        # block._add_temporary_set assumes ALL components define an
        # index.  Sigh.
        self._index_set = None

    def get_units(self):
        """Return the units for this ExternalFunction"""
        return self._units

    def get_arg_units(self):
        """Return the units for this ExternalFunctions arguments"""
        return self._arg_units

    def __call__(self, *args):
        """Return a Pyomo expression that evaluates this function

        Note that this does not evaluate the external function
        represented by this :class:`ExternalFunction` component, but
        rather generates a Pyomo expression object that represents
        symbolic evaluation of the external function at the specified
        arguments (which themselves may be Pyomo expression objects).

        Parameters
        ----------
        *args:
            Arguments (constants, variables, expressions) to pass to
            this External function.

        Returns
        -------
        ExternalFunctionExpression or NPV_ExternalFunctionExpression:
            The new Pyomo expression node.

        """

        args_ = []
        for arg in args:
            if type(arg) is types.GeneratorType:
                args_.extend(arg)
            else:
                args_.append(arg)
        #
        # Loop and do two thing:
        #   1. Wrap non-numeric arguments
        #   2. See if we have a potentially variable argument
        #
        pv = False
        for i, arg in enumerate(args_):
            try:
                if arg.__class__ in native_types:
                    continue
                if arg.is_potentially_variable():
                    pv = True
                continue
            except AttributeError:
                if check_if_native_type(arg):
                    continue
            args_[i] = NonNumericValue(arg)
        #
        if pv:
            return EXPR.ExternalFunctionExpression(args_, self)
        return EXPR.NPV_ExternalFunctionExpression(args_, self)

    def evaluate(self, args):
        """Return the value of the function given the specified arguments

        Parameters
        ----------
        args: Iterable
            Iterable containing the arguments to pass to the external
            function.  Non-native type elements will be converted to a
            native value using the :py:func:`value()` function.

        Returns
        -------
        float
            The return value of the function evaluated at `args`
        """
        args_ = [arg if arg.__class__ in native_types else value(arg) for arg in args]
        return self._evaluate(args_, None, 0)[0]

    def evaluate_fgh(self, args, fixed=None, fgh=2):
        """Evaluate the function and gradients given the specified arguments

        This evaluates the function given the specified arguments
        returning a 3-tuple of (function value [f], list of first partial
        derivatives [g], and the upper triangle of the Hessian matrix
        [h]).

        Parameters
        ----------
        args: Iterable
            Iterable containing the arguments to pass to the external
            function.  Non-native type elements will be converted to a
            native value using the :py:func:`value()` function.

        fixed: Optional[List[bool]]
            List of values indicating if the corresponding argument
            value is fixed.  Any fixed indices are guaranteed to return
            0 for first and second derivatives, regardless of what is
            computed by the external function.

        fgh: {0, 1, 2}
            What evaluations to return:

            * **0**: just return function evaluation
            * **1**: return function and first derivatives
            * **2**: return function, first derivatives, and hessian matrix

            Any return values not requested will be `None`.

        Returns
        -------
        f: float
            The return value of the function evaluated at `args`
        g: List[float] or None
            The list of first partial derivatives
        h: List[float] or None
            The upper-triangle of the Hessian matrix (second partial
            derivatives), stored column-wise.  Element :math:`H_{i,j}`
            (with :math:`0 <= i <= j < N` are mapped using
            :math:`h[i + j*(j + 1)/2] == H_{i,j}`.

        """
        args_ = [arg if arg.__class__ in native_types else value(arg) for arg in args]
        # Note: this is passed-by-reference, and the args_ list may be
        # changed by _evaluate (e.g., for PythonCallbackFunction).
        # Remember the original length of the list.
        N = len(args_)
        f, g, h = self._evaluate(args_, fixed, fgh)
        # Guarantee the return value behavior documented in the docstring
        if fgh == 2:
            n = N - 1
            if len(h) - 1 != n + n * (n + 1) // 2:
                raise RuntimeError(
                    f"External function '{self.name}' returned an invalid "
                    f"Hessian matrix (expected {n + n*(n+1)//2 + 1}, "
                    f"received {len(h)})"
                )
        else:
            h = None
        if fgh >= 1:
            if len(g) != N:
                raise RuntimeError(
                    f"External function '{self.name}' returned an invalid "
                    f"derivative vector (expected {N}, "
                    f"received {len(g)})"
                )
        else:
            g = None
        # Note: the ASL does not require clients to honor the fixed flag
        # (allowing them to return non-0 values for the derivative with
        # respect to a fixed numeric value).  We will allow clients to
        # be similarly lazy and enforce the fixed flag here.
        if fixed is not None:
            if fgh >= 1:
                for i, v in enumerate(fixed):
                    if not v:
                        continue
                    g[i] = 0
            if fgh >= 2:
                for i, v in enumerate(fixed):
                    if not v:
                        continue
                    for j in range(N):
                        if i <= j:
                            h[i + (j * (j + 1)) // 2] = 0
                        else:
                            h[j + (i * (i + 1)) // 2] = 0
        return f, g, h

    def _evaluate(self, args, fixed, fgh):
        """Derived class implementation to evaluate an external function.

        The API follows :meth:`evaluate_fgh()`, with the exception that
        the `args` iterable is guaranteed to already have been reduced
        to concrete values (it will not have Pyomo components or
        expressions).
        """
        raise NotImplementedError(f"{type(self)} did not implement _evaluate()")


class AMPLExternalFunction(ExternalFunction):
    __autoslot_mappers__ = {
        # Remove reference to loaded library (they are not copyable or
        # picklable)
        '_so': AutoSlots.encode_as_none,
        '_known_functions': AutoSlots.encode_as_none,
    }

    def __init__(self, *args, **kwargs):
        if args:
            raise ValueError(
                "AMPLExternalFunction constructor does not support "
                "positional arguments"
            )
        self._library = kwargs.pop('library', None)
        self._function = kwargs.pop('function', None)
        self._known_functions = None
        self._so = None
        # Convert the specified library name to an absolute path, and
        # warn the user if we couldn't find it
        if self._library is not None:
            _lib = find_library(self._library)
            if _lib is not None:
                self._library = _lib
            else:
                logger.warning(
                    'Defining AMPL external function, but cannot locate '
                    f'specified library "{self._library}"'
                )
        ExternalFunction.__init__(self, *args, **kwargs)

    def _evaluate(self, args, fixed, fgh):
        if self._so is None:
            self.load_library()
        if self._function not in self._known_functions:
            raise RuntimeError(
                "Error: external function '%s' was not registered within "
                "external library %s.\n\tAvailable functions: (%s)"
                % (
                    self._function,
                    self._library,
                    ', '.join(self._known_functions.keys()),
                )
            )
        #
        N = len(args)
        arglist = _ARGLIST(args, fgh, fixed)
        fcn = self._known_functions[self._function][0]
        f = fcn(byref(arglist))
        if fgh >= 1:
            g = [nan] * N
            for i in range(N):
                if arglist.at[i] < 0:
                    continue
                g[i] = arglist.derivs[arglist.at[i]]
        else:
            g = None
        if fgh >= 2:
            h = [nan] * ((N + N**2) // 2)
            for j in range(N):
                j_r = arglist.at[j]
                if j_r < 0:
                    continue
                for i in range(j + 1):
                    i_r = arglist.at[i]
                    if i_r < 0:
                        continue
                    h[i + j * (j + 1) // 2] = arglist.hes[i_r + j_r * (j_r + 1) // 2]
        else:
            h = None
        return f, g, h

    def load_library(self):
        # Note that the library was located in __init__ and converted to
        # an absolute path (including checking the CWD).  However, we
        # previously tested that changing the environment (i.e.,
        # changing directories) between defining the ExternalFunction
        # and loading it would cause the library to still be correctly
        # loaded.  We will re-search for the library here.  If it was
        # previously found, we will be searching for an absolute path
        # (and the same path will be found again).
        _abs_lib = find_library(self._library)
        if _abs_lib is not None:
            self._library = _abs_lib
        self._so = cdll.LoadLibrary(self._library)
        self._known_functions = {}
        AE = _AMPLEXPORTS()
        AE.ASLdate = 20160307

        def addfunc(name, f, _type, nargs, funcinfo, ae):
            # trap for Python 3, where the name comes in as bytes() and
            # not a string
            if not isinstance(name, str):
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
                "undesirable behavior."
            )

        AE.AtReset = _AMPLEXPORTS.ATRESET(atreset)

        FUNCADD = CFUNCTYPE(None, POINTER(_AMPLEXPORTS))
        FUNCADD(('funcadd_ASL', self._so))(byref(AE))

    def _pprint(self):
        return (
            [
                ('function', self._function),
                ('library', self._library),
                ('units', str(self._units)),
                (
                    'arg_units',
                    (
                        [str(u) for u in self._arg_units]
                        if self._arg_units is not None
                        else None
                    ),
                ),
            ],
            (),
            None,
            None,
        )


def _python_callback_fid_mapper(encode, val):
    if encode:
        return PythonCallbackFunction.global_registry[val]()
    else:
        return PythonCallbackFunction.register_instance(val)


class _PythonCallbackFunctionID(NumericConstant):
    """A specialized NumericConstant to preserve FunctionIDs through deepcopy.

    Function IDs are effectively pointers back to the
    PythonCallbackFunction.  As such, we need special handling to
    maintain / preserve the correct linkages through deepcopy (and
    model.clone()).

    """

    __slots__ = ()
    __autoslot_mappers__ = {'value': _python_callback_fid_mapper}

    def is_constant(self):
        # Return False so this object is not simplified out of expressions
        return False


_pyomo_constant_types.add(_PythonCallbackFunctionID)


class PythonCallbackFunction(ExternalFunction):
    __autoslot_mappers__ = {'_fcn_id': _python_callback_fid_mapper}

    global_registry = []
    global_id_to_fid = {}

    @classmethod
    def register_instance(cls, instance):
        # Ideally, we would just use the id() as the function id, but
        # Python id() is more than 32-bits, which can cause problems for
        # the ASL.  We will map id() to a "small" integer and use that
        # small integer as the "function id".
        _id = id(instance)
        if _id in cls.global_id_to_fid:
            prev_fid = cls.global_id_to_fid[_id]
            prev_obj = cls.global_registry[prev_fid]
            if prev_obj() is not None:
                assert prev_obj() is instance
                return prev_fid
        cls.global_id_to_fid[_id] = _fid = len(cls.global_registry)
        cls.global_registry.append(weakref.ref(instance))
        return _fid

    def __init__(self, *args, **kwargs):
        for i, kw in enumerate(('function', 'gradient', 'hessian')):
            if len(args) <= i:
                break
            if kw in kwargs:
                raise ValueError(
                    "Duplicate definition of external function through "
                    f"positional and keyword ('{kw}=') arguments"
                )
            kwargs[kw] = args[i]
        if len(args) > 3:
            raise ValueError(
                "PythonCallbackFunction constructor only supports "
                "0 - 3 positional arguments"
            )
        self._fcn = kwargs.pop('function', None)
        self._grad = kwargs.pop('gradient', None)
        self._hess = kwargs.pop('hessian', None)
        self._fgh = kwargs.pop('fgh', None)
        if self._fgh is not None and any((self._fcn, self._grad, self._hess)):
            raise ValueError(
                "Cannot specify 'fgh' with any of {'function', 'gradient', hessian'}"
            )

        # There is an implicit first argument (the function pointer), we
        # need to add that to the arg_units
        arg_units = kwargs.get('arg_units', None)
        if arg_units is not None:
            kwargs['arg_units'] = list(arg_units)
            kwargs['arg_units'].append(None)

        # TODO: complete/distribute the pyomo_socket_server /
        # pyomo_ampl.so AMPL extension
        self._library = 'pyomo_ampl.so'
        self._function = 'pyomo_socket_server'
        ExternalFunction.__init__(self, *args, **kwargs)
        self._fcn_id = PythonCallbackFunction.register_instance(self)

    def __call__(self, *args):
        # NOTE: we append the Function ID to the END of the argument
        # list because it is easier to update the gradient / hessian
        # results
        return super().__call__(*args, _PythonCallbackFunctionID(self._fcn_id))

    def _evaluate(self, args, fixed, fgh):
        # Remove the fcn_id
        _id = args.pop()
        if fixed is not None:
            fixed = fixed[:-1]
        if _id != self._fcn_id:
            raise RuntimeError("PythonCallbackFunction called with invalid Global ID")
        if self._fgh is not None:
            f, g, h = self._fgh(args, fgh, fixed)
        else:
            f = self._fcn(*args)
            if fgh >= 1:
                if self._grad is None:
                    raise RuntimeError(
                        f"ExternalFunction '{self.name}' was not defined "
                        "with a gradient callback.  Cannot evaluate the "
                        "derivative of the function"
                    )
                g = self._grad(args, fixed)
            else:
                g = None
            if fgh == 2:
                if self._hess is None:
                    raise RuntimeError(
                        f"ExternalFunction '{self.name}' was not defined "
                        "with a Hessian callback.  Cannot evaluate the "
                        "second derivative of the function"
                    )
                h = self._hess(args, fixed)
            else:
                h = None
        # Update g,h to reflect the function id (which by definition is
        # always fixed)
        if g is not None:
            g.append(0)
        if h is not None:
            h.extend([0] * (len(args) + 1))
        return f, g, h

    def _pprint(self):
        return (
            [
                ('function', self._fcn.__qualname__),
                ('units', str(self._units)),
                (
                    'arg_units',
                    (
                        [str(u) for u in self._arg_units[:-1]]
                        if self._arg_units is not None
                        else None
                    ),
                ),
            ],
            (),
            None,
            None,
        )


class _ARGLIST(Structure):
    """Mock up the arglist structure from AMPL's funcadd.h

    This data structure is populated by AMPL when calling external
    functions (for both passing in the function arguments and retrieving
    the derivative/Hessian information.
    """

    _fields_ = [
        ('n', c_int),  # number of args
        ('nr', c_int),  # number of real input args
        ('at', POINTER(c_int)),  # argument types -- see DISCUSSION below
        ('ra', POINTER(c_double)),  # pure real args (IN, OUT, and INOUT)
        ('sa', POINTER(c_char_p)),  # symbolic IN args
        ('derivs', POINTER(c_double)),  # for partial derivatives (if nonzero)
        ('hes', POINTER(c_double)),  # for second partials (if nonzero)
        # if (dig && dig[i]) { partials w.r.t. ra[i] will not be used }
        ('dig', POINTER(c_byte)),
        ('funcinfo', c_char_p),  # for use by the function (if desired)
        ('AE', c_void_p),  # functions made visible (via #defines below)
        ('f', c_void_p),  # for internal use by AMPL
        ('tva', c_void_p),  # for internal use by AMPL
        ('Errmsg', c_char_p),  # To indicate an error, set this to a
        # description of the error.  When derivs is nonzero and the
        # error is that first derivatives cannot or are not computed, a
        # single quote character (') should be the first character in
        # the text assigned to Errmsg, followed by the actual error
        # message.  Similarly, if hes is nonzero and the error is that
        # second derivatives are not or cannot be computed, a double
        # quote character (") should be the first character in Errmsg,
        # followed by the actual error message text.
        ('TMI', c_void_p),  # used in Tempmem calls
        ('Private', c_char_p),
        # The following fields are relevant only when imported functions
        # are called by AMPL commands (not declarations).
        ('nin', c_int),  # number of input (IN and INOUT) args
        ('nout', c_int),  # number of output (OUT and INOUT) args
        ('nsin', c_int),  # number of symbolic input arguments
        ('nsout', c_int),  # number of symbolic OUT and INOUT args
    ]

    def __init__(self, args, fgh=0, fixed=None):
        super().__init__()
        self._encoded_strings = []
        self.n = len(args)
        self.at = (c_int * self.n)()
        _reals = []
        _strings = []
        nr = 0
        ns = 0
        for i, arg in enumerate(args):
            if arg.__class__ in native_numeric_types:
                _reals.append(arg)
                self.at[i] = nr
                nr += 1
                continue
            if isinstance(arg, str):
                # String arguments need to be converted to bytes
                # (encoded as plain ASCII characters).  We will
                # explicitly cache the resulting bytes object to make
                # absolutely sure that the Python garbage collector
                # doesn't accidentally clean up the object before we
                # call the external function.
                arg = arg.encode('ascii')
                self._encoded_strings.append(arg)
            if isinstance(arg, bytes):
                _strings.append(arg)
                # Note increment first, as at[i] starts at -1 for strings
                ns += 1
                self.at[i] = -ns
            else:
                raise RuntimeError(
                    f"Unknown data type, {type(arg).__name__}, passed as "
                    f"argument {i} for an ASL ExternalFunction"
                )
        self.nr = nr
        self.ra = (c_double * nr)(*_reals)
        self.sa = (c_char_p * ns)(*_strings)
        if fgh >= 1:
            self.derivs = (c_double * nr)(0.0)
        if fgh >= 2:
            self.hes = (c_double * ((nr + nr * nr) // 2))(0.0)

        if fixed:
            self.dig = (c_byte * nr)(0)
            for i, v in enumerate(fixed):
                if v:
                    r_idx = self.at[i]
                    if r_idx >= 0:
                        self.dig[r_idx] = 1


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

    AMPLFUNC = CFUNCTYPE(c_double, POINTER(_ARGLIST))

    ADDFUNC = CFUNCTYPE(
        None, c_char_p, AMPLFUNC, c_int, c_int, c_void_p, POINTER(_AMPLEXPORTS)
    )

    RANDSEEDSETTER = CFUNCTYPE(None, c_void_p, c_ulong)

    ADDRANDINIT = CFUNCTYPE(None, POINTER(_AMPLEXPORTS), RANDSEEDSETTER, c_void_p)

    ATRESET = CFUNCTYPE(None, POINTER(_AMPLEXPORTS), c_void_p, c_void_p)

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
        # /* More stuff for stdio in DLLs... */
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
        # /* Items available with ASLdate >= 20020501 start here. */
        ('SnprintF', c_void_p),
        ('VsnprintF', c_void_p),
        ('Addrand', c_void_p),
        ('Addrandinit', ADDRANDINIT),
    ]
