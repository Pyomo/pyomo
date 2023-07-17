#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os

logger = logging.getLogger(__name__)

CURRENT_INTERFACE_VERSION = 3


class _NotSet:
    pass


def _LoadASLInterface(libname):
    ASLib = ctypes.cdll.LoadLibrary(libname)

    # define 1d array
    array_1d_double = np.ctypeslib.ndpointer(
        dtype=np.double, ndim=1, flags='CONTIGUOUS'
    )
    array_1d_int = np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

    # library version
    try:
        ASLib.EXTERNAL_AmplInterface_version.argtypes = None
        ASLib.EXTERNAL_AmplInterface_version.restype = ctypes.c_int
        interface_version = ASLib.EXTERNAL_AmplInterface_version()
    except AttributeError:
        interface_version = 1

    # ASL version
    if interface_version >= 3:
        ASLib.EXTERNAL_get_asl_date.argtypes = []
        ASLib.EXTERNAL_get_asl_date.restype = ctypes.c_long

    # constructor
    ASLib.EXTERNAL_AmplInterface_new.argtypes = [ctypes.c_char_p]
    ASLib.EXTERNAL_AmplInterface_new.restype = ctypes.c_void_p

    if interface_version >= 2:
        ASLib.EXTERNAL_AmplInterface_new_file.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
    else:
        ASLib.EXTERNAL_AmplInterface_new_file.argtypes = [ctypes.c_char_p]
    ASLib.EXTERNAL_AmplInterface_new_file.restype = ctypes.c_void_p

    # ASLib.EXTERNAL_AmplInterface_new_str.argtypes = [ctypes.c_char_p]
    # ASLib.EXTERNAL_AmplInterface_new_str.restype = ctypes.c_void_p

    # number of variables
    ASLib.EXTERNAL_AmplInterface_n_vars.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_n_vars.restype = ctypes.c_int

    # number of constraints
    ASLib.EXTERNAL_AmplInterface_n_constraints.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_n_constraints.restype = ctypes.c_int

    # number of nonzeros in jacobian
    ASLib.EXTERNAL_AmplInterface_nnz_jac_g.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_nnz_jac_g.restype = ctypes.c_int

    # number of nonzeros in hessian of lagrangian
    ASLib.EXTERNAL_AmplInterface_nnz_hessian_lag.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_nnz_hessian_lag.restype = ctypes.c_int

    # lower bounds on x
    ASLib.EXTERNAL_AmplInterface_x_lower_bounds.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_x_lower_bounds.restype = None

    # upper bounds on x
    ASLib.EXTERNAL_AmplInterface_x_upper_bounds.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_x_upper_bounds.restype = None

    # lower bounds on g
    ASLib.EXTERNAL_AmplInterface_g_lower_bounds.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_g_lower_bounds.restype = None

    # upper bounds on g
    ASLib.EXTERNAL_AmplInterface_g_upper_bounds.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_g_upper_bounds.restype = None

    # initial value x
    ASLib.EXTERNAL_AmplInterface_get_init_x.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_get_init_x.restype = None

    # initial value multipliers
    ASLib.EXTERNAL_AmplInterface_get_init_multipliers.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_get_init_multipliers.restype = None

    # evaluate objective
    ASLib.EXTERNAL_AmplInterface_eval_f.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    ASLib.EXTERNAL_AmplInterface_eval_f.restype = ctypes.c_bool

    # gradient objective
    ASLib.EXTERNAL_AmplInterface_eval_deriv_f.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_eval_deriv_f.restype = ctypes.c_bool

    # structure jacobian of constraints
    ASLib.EXTERNAL_AmplInterface_struct_jac_g.argtypes = [
        ctypes.c_void_p,
        array_1d_int,
        array_1d_int,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_struct_jac_g.restype = None

    # structure hessian of Lagrangian
    ASLib.EXTERNAL_AmplInterface_struct_hes_lag.argtypes = [
        ctypes.c_void_p,
        array_1d_int,
        array_1d_int,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_struct_hes_lag.restype = None

    # evaluate constraints
    ASLib.EXTERNAL_AmplInterface_eval_g.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_eval_g.restype = ctypes.c_bool

    # evaluate jacobian constraints
    ASLib.EXTERNAL_AmplInterface_eval_jac_g.argtypes = [
        ctypes.c_void_p,
        array_1d_double,
        ctypes.c_int,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_eval_jac_g.restype = ctypes.c_bool

    # temporary try/except block while changes get merged in pynumero_libraries
    try:
        ASLib.EXTERNAL_AmplInterface_dummy.argtypes = [ctypes.c_void_p]
        ASLib.EXTERNAL_AmplInterface_dummy.restype = None
        # evaluate hessian Lagrangian
        ASLib.EXTERNAL_AmplInterface_eval_hes_lag.argtypes = [
            ctypes.c_void_p,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int,
            ctypes.c_double,
        ]
        ASLib.EXTERNAL_AmplInterface_eval_hes_lag.restype = ctypes.c_bool
    except Exception:
        # evaluate hessian Lagrangian
        ASLib.EXTERNAL_AmplInterface_eval_hes_lag.argtypes = [
            ctypes.c_void_p,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int,
        ]
        ASLib.EXTERNAL_AmplInterface_eval_hes_lag.restype = ctypes.c_bool
        interface_version = 0

    # finalize solution
    ASLib.EXTERNAL_AmplInterface_finalize_solution.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        array_1d_double,
        ctypes.c_int,
        array_1d_double,
        ctypes.c_int,
    ]
    ASLib.EXTERNAL_AmplInterface_finalize_solution.restype = None

    # destructor
    ASLib.EXTERNAL_AmplInterface_free_memory.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_free_memory.restype = None

    if CURRENT_INTERFACE_VERSION != interface_version:
        logger.warning(
            'The current pynumero_ASL library is version=%s, but found '
            'version=%s.  Please recompile / update your pynumero_ASL '
            'library.' % (CURRENT_INTERFACE_VERSION, interface_version)
        )

    return ASLib, interface_version


class AmplInterface(object):
    libname = _NotSet
    ASLib = None
    interface_version = None
    asl_date = None

    @classmethod
    def available(cls):
        if cls.libname is _NotSet:
            cls.libname = find_library('pynumero_ASL')
        if cls.libname is None:
            return False
        return os.path.exists(cls.libname)

    def __init__(self, filename=None, nl_buffer=None):
        if not AmplInterface.available():
            raise RuntimeError("Cannot load the PyNumero ASL interface (pynumero_ASL)")

        if nl_buffer is not None:
            raise NotImplementedError(
                "AmplInterface only supported form NL-file for now"
            )

        # Be sure to remove AMPLFUNC from the environment before loading
        # the ASL.  This should prevent it from potentially caching an
        # AMPLFUNC from the initial load and letting it bleed into
        # (potentially unrelated) subsequent instances
        amplfunc = os.environ.pop('AMPLFUNC', '')

        if AmplInterface.ASLib is None:
            AmplInterface.ASLib, AmplInterface.interface_version = _LoadASLInterface(
                AmplInterface.libname
            )
            if AmplInterface.interface_version >= 3:
                AmplInterface.asl_date = AmplInterface.ASLib.EXTERNAL_get_asl_date()
            else:
                AmplInterface.asl_date = 0

        if filename is not None:
            if nl_buffer is not None:
                raise ValueError("Cannot specify both filename= and nl_buffer=")

            b_data = filename.encode('utf-8')
            if self.interface_version >= 2:
                args = (b_data, amplfunc.encode('utf-8'))
            else:
                # Old ASL interface library.
                if amplfunc:
                    # we need to put AMPLFUNC back into the environment,
                    # as old versions of the library rely on ONLY the
                    # environment variable for passing the library(ies)
                    # locations to the ASL
                    os.environ['AMPLFUNC'] = amplfunc
                args = (b_data,)
            self._obj = self.ASLib.EXTERNAL_AmplInterface_new_file(*args)
        elif nl_buffer is not None:
            b_data = nl_buffer.encode('utf-8')
            if os.name in ['nt', 'dos']:
                self._obj = self.ASLib.EXTERNAL_AmplInterface_new_file(b_data)
            else:
                self._obj = self.ASLib.EXTERNAL_AmplInterface_new_str(b_data)

        assert self._obj, "Error building ASL interface. Possible error in nl-file"

        self._nx = self.get_n_vars()
        self._ny = self.get_n_constraints()
        self._nnz_jac_g = self.get_nnz_jac_g()
        self._nnz_hess = self.get_nnz_hessian_lag()

    def __del__(self):
        self.ASLib.EXTERNAL_AmplInterface_free_memory(self._obj)

    def get_n_vars(self):
        return self.ASLib.EXTERNAL_AmplInterface_n_vars(self._obj)

    def get_n_constraints(self):
        return self.ASLib.EXTERNAL_AmplInterface_n_constraints(self._obj)

    def get_nnz_jac_g(self):
        return self.ASLib.EXTERNAL_AmplInterface_nnz_jac_g(self._obj)

    def get_nnz_hessian_lag(self):
        return self.ASLib.EXTERNAL_AmplInterface_nnz_hessian_lag(self._obj)

    def get_bounds_info(self, xl, xu, gl, gu):
        x_l = xl.astype(np.double, casting='safe', copy=False)
        x_u = xu.astype(np.double, casting='safe', copy=False)
        g_l = gl.astype(np.double, casting='safe', copy=False)
        g_u = gu.astype(np.double, casting='safe', copy=False)
        nx = len(x_l)
        ng = len(g_l)
        assert nx == len(x_u), "lower and upper bound x vectors must be the same size"
        assert ng == len(g_u), "lower and upper bound g vectors must be the same size"
        self.ASLib.EXTERNAL_AmplInterface_get_bounds_info(
            self._obj, x_l, x_u, nx, g_l, g_u, ng
        )

    def get_x_lower_bounds(self, invec):
        self.ASLib.EXTERNAL_AmplInterface_x_lower_bounds(self._obj, invec, len(invec))

    def get_x_upper_bounds(self, invec):
        self.ASLib.EXTERNAL_AmplInterface_x_upper_bounds(self._obj, invec, len(invec))

    def get_g_lower_bounds(self, invec):
        self.ASLib.EXTERNAL_AmplInterface_g_lower_bounds(self._obj, invec, len(invec))

    def get_g_upper_bounds(self, invec):
        self.ASLib.EXTERNAL_AmplInterface_g_upper_bounds(self._obj, invec, len(invec))

    def get_init_x(self, invec):
        self.ASLib.EXTERNAL_AmplInterface_get_init_x(self._obj, invec, len(invec))

    def get_init_multipliers(self, invec):
        self.ASLib.EXTERNAL_AmplInterface_get_init_multipliers(
            self._obj, invec, len(invec)
        )

    def eval_f(self, x):
        assert x.size == self._nx, "Error: Dimension mismatch."
        assert (
            x.dtype == np.double
        ), "Error: array type. Function eval_deriv_f expects an array of type double"
        sol = ctypes.c_double()
        res = self.ASLib.EXTERNAL_AmplInterface_eval_f(
            self._obj, x, self._nx, ctypes.byref(sol)
        )
        if not res:
            raise PyNumeroEvaluationError("Error in AMPL evaluation")
        return sol.value

    def eval_deriv_f(self, x, df):
        assert x.size == self._nx, "Error: Dimension mismatch."
        assert (
            x.dtype == np.double
        ), "Error: array type. Function eval_deriv_f expects an array of type double"
        res = self.ASLib.EXTERNAL_AmplInterface_eval_deriv_f(self._obj, x, df, len(x))
        if not res:
            raise PyNumeroEvaluationError("Error in AMPL evaluation")

    def struct_jac_g(self, irow, jcol):
        irow_p = irow.astype(np.intc, casting='safe', copy=False)
        jcol_p = jcol.astype(np.intc, casting='safe', copy=False)
        assert len(irow) == len(
            jcol
        ), "Error: Dimension mismatch. Arrays irow and jcol must be of the same size"
        assert (
            len(irow) == self._nnz_jac_g
        ), "Error: Dimension mismatch. Jacobian has {} nnz".format(self._nnz_jac_g)
        self.ASLib.EXTERNAL_AmplInterface_struct_jac_g(
            self._obj, irow_p, jcol_p, len(irow)
        )

    def struct_hes_lag(self, irow, jcol):
        irow_p = irow.astype(np.intc, casting='safe', copy=False)
        jcol_p = jcol.astype(np.intc, casting='safe', copy=False)
        assert len(irow) == len(
            jcol
        ), "Error: Dimension mismatch. Arrays irow and jcol must be of the same size"
        assert (
            len(irow) == self._nnz_hess
        ), "Error: Dimension mismatch. Hessian has {} nnz".format(self._nnz_hess)
        self.ASLib.EXTERNAL_AmplInterface_struct_hes_lag(
            self._obj, irow_p, jcol_p, len(irow)
        )

    def eval_jac_g(self, x, jac_g_values):
        assert x.size == self._nx, "Error: Dimension mismatch."
        assert jac_g_values.size == self._nnz_jac_g, "Error: Dimension mismatch."
        xeval = x.astype(np.double, casting='safe', copy=False)
        jac_eval = jac_g_values.astype(np.double, casting='safe', copy=False)
        res = self.ASLib.EXTERNAL_AmplInterface_eval_jac_g(
            self._obj, xeval, self._nx, jac_eval, self._nnz_jac_g
        )
        if not res:
            raise PyNumeroEvaluationError("Error in AMPL evaluation")

    def eval_g(self, x, g):
        assert x.size == self._nx, "Error: Dimension mismatch."
        assert g.size == self._ny, "Error: Dimension mismatch."
        assert (
            x.dtype == np.double
        ), "Error: array type. Function eval_g expects an array of type double"
        assert (
            g.dtype == np.double
        ), "Error: array type. Function eval_g expects an array of type double"
        res = self.ASLib.EXTERNAL_AmplInterface_eval_g(
            self._obj, x, self._nx, g, self._ny
        )
        if not res:
            raise PyNumeroEvaluationError("Error in AMPL evaluation")

    def eval_hes_lag(self, x, lam, hes_lag, obj_factor=1.0):
        assert x.size == self._nx, "Error: Dimension mismatch."
        assert lam.size == self._ny, "Error: Dimension mismatch."
        assert hes_lag.size == self._nnz_hess, "Error: Dimension mismatch."
        assert (
            x.dtype == np.double
        ), "Error: array type. Function eval_hes_lag expects an array of type double"
        assert (
            lam.dtype == np.double
        ), "Error: array type. Function eval_hes_lag expects an array of type double"
        assert (
            hes_lag.dtype == np.double
        ), "Error: array type. Function eval_hes_lag expects an array of type double"
        if self.interface_version >= 1:
            res = self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag(
                self._obj,
                x,
                self._nx,
                lam,
                self._ny,
                hes_lag,
                self._nnz_hess,
                obj_factor,
            )
        else:
            res = self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag(
                self._obj, x, self._nx, lam, self._ny, hes_lag, self._nnz_hess
            )
        if not res:
            raise PyNumeroEvaluationError("Error in AMPL evaluation")

    def finalize_solution(self, ampl_solve_status_num, msg, x, lam):
        b_msg = msg.encode('utf-8')
        self.ASLib.EXTERNAL_AmplInterface_finalize_solution(
            self._obj, ampl_solve_status_num, b_msg, x, len(x), lam, len(lam)
        )
