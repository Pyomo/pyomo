#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.fileutils import find_library
import numpy.ctypeslib as npct
import numpy as np
import ctypes
import os

class _NotSet:
    pass

class AmplInterface(object):

    libname = _NotSet

    @classmethod
    def available(cls):
        if cls.libname is _NotSet:
            cls.libname = find_library('pynumero_ASL')
        if cls.libname is None:
            return False
        return os.path.exists(cls.libname)

    def __init__(self, filename=None, nl_buffer=None):

        if not AmplInterface.available():
            raise RuntimeError(
                "ASL interface is not supported on this platform (%s)"
                % (os.name,) )

        if nl_buffer is not None:
            raise NotImplementedError("AmplInterface only supported form NL-file for now")

        self.ASLib = ctypes.cdll.LoadLibrary(AmplInterface.libname)

        # define 1d array
        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

        # constructor
        self.ASLib.EXTERNAL_AmplInterface_new.argtypes = [ctypes.c_char_p]
        self.ASLib.EXTERNAL_AmplInterface_new.restype = ctypes.c_void_p

        self.ASLib.EXTERNAL_AmplInterface_new_file.argtypes = [ctypes.c_char_p]
        self.ASLib.EXTERNAL_AmplInterface_new_file.restype = ctypes.c_void_p

        #self.ASLib.EXTERNAL_AmplInterface_new_str.argtypes = [ctypes.c_char_p]
        #self.ASLib.EXTERNAL_AmplInterface_new_str.restype = ctypes.c_void_p

        # number of variables
        self.ASLib.EXTERNAL_AmplInterface_n_vars.argtypes = [ctypes.c_void_p]
        self.ASLib.EXTERNAL_AmplInterface_n_vars.restype = ctypes.c_int

        # number of constraints
        self.ASLib.EXTERNAL_AmplInterface_n_constraints.argtypes = [ctypes.c_void_p]
        self.ASLib.EXTERNAL_AmplInterface_n_constraints.restype = ctypes.c_int

        # number of nonzeros in jacobian
        self.ASLib.EXTERNAL_AmplInterface_nnz_jac_g.argtypes = [ctypes.c_void_p]
        self.ASLib.EXTERNAL_AmplInterface_nnz_jac_g.restype = ctypes.c_int

        # number of nonzeros in hessian of lagrangian
        self.ASLib.EXTERNAL_AmplInterface_nnz_hessian_lag.argtypes = [ctypes.c_void_p]
        self.ASLib.EXTERNAL_AmplInterface_nnz_hessian_lag.restype = ctypes.c_int

        # lower bounds on x
        self.ASLib.EXTERNAL_AmplInterface_x_lower_bounds.argtypes = [ctypes.c_void_p,
                                                                     array_1d_double,
                                                                     ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_x_lower_bounds.restype = None

        # upper bounds on x
        self.ASLib.EXTERNAL_AmplInterface_x_upper_bounds.argtypes = [ctypes.c_void_p,
                                                                     array_1d_double,
                                                                     ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_x_upper_bounds.restype = None

        # lower bounds on g
        self.ASLib.EXTERNAL_AmplInterface_g_lower_bounds.argtypes = [ctypes.c_void_p,
                                                                     array_1d_double,
                                                                     ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_g_lower_bounds.restype = None

        # upper bounds on g
        self.ASLib.EXTERNAL_AmplInterface_g_upper_bounds.argtypes = [ctypes.c_void_p,
                                                                     array_1d_double,
                                                                     ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_g_upper_bounds.restype = None

        # initial value x
        self.ASLib.EXTERNAL_AmplInterface_get_init_x.argtypes = [ctypes.c_void_p,
                                                                 array_1d_double,
                                                                 ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_get_init_x.restype = None

        # initial value multipliers
        self.ASLib.EXTERNAL_AmplInterface_get_init_multipliers.argtypes = [ctypes.c_void_p,
                                                                           array_1d_double,
                                                                           ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_get_init_multipliers.restype = None

        # evaluate objective
        self.ASLib.EXTERNAL_AmplInterface_eval_f.argtypes = [ctypes.c_void_p,
                                                             array_1d_double,
                                                             ctypes.c_int,
                                                             ctypes.POINTER(ctypes.c_double)]
        self.ASLib.EXTERNAL_AmplInterface_eval_f.restype = ctypes.c_bool

        # gradient objective
        self.ASLib.EXTERNAL_AmplInterface_eval_deriv_f.argtypes = [ctypes.c_void_p,
                                                                   array_1d_double,
                                                                   array_1d_double,
                                                                   ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_eval_deriv_f.restype = ctypes.c_bool

        # structure jacobian of constraints
        self.ASLib.EXTERNAL_AmplInterface_struct_jac_g.argtypes = [ctypes.c_void_p,
                                                                   array_1d_int,
                                                                   array_1d_int,
                                                                   ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_struct_jac_g.restype = None

        # structure hessian of Lagrangian
        self.ASLib.EXTERNAL_AmplInterface_struct_hes_lag.argtypes = [ctypes.c_void_p,
                                                                     array_1d_int,
                                                                     array_1d_int,
                                                                     ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_struct_hes_lag.restype = None

        # evaluate constraints
        self.ASLib.EXTERNAL_AmplInterface_eval_g.argtypes = [ctypes.c_void_p,
                                                             array_1d_double,
                                                             ctypes.c_int,
                                                             array_1d_double,
                                                             ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_eval_g.restype = ctypes.c_bool

        # evaluate jacobian constraints
        self.ASLib.EXTERNAL_AmplInterface_eval_jac_g.argtypes = [ctypes.c_void_p,
                                                                 array_1d_double,
                                                                 ctypes.c_int,
                                                                 array_1d_double,
                                                                 ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_eval_jac_g.restype = ctypes.c_bool

        # temporary try/except block while changes get merged in pynumero_libraries
        try:
            self.ASLib.EXTERNAL_AmplInterface_dummy.argtypes = [ctypes.c_void_p]
            self.ASLib.EXTERNAL_AmplInterface_dummy.restype = None
            # evaluate hessian Lagrangian
            self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag.argtypes = [ctypes.c_void_p,
                                                                       array_1d_double,
                                                                       ctypes.c_int,
                                                                       array_1d_double,
                                                                       ctypes.c_int,
                                                                       array_1d_double,
                                                                       ctypes.c_int,
                                                                       ctypes.c_double]
            self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag.restype = ctypes.c_bool
            self.future_libraries = True
        except Exception:
            # evaluate hessian Lagrangian
            self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag.argtypes = [ctypes.c_void_p,
                                                                       array_1d_double,
                                                                       ctypes.c_int,
                                                                       array_1d_double,
                                                                       ctypes.c_int,
                                                                       array_1d_double,
                                                                       ctypes.c_int]
            self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag.restype = ctypes.c_bool
            self.future_libraries = False

        # finalize solution
        self.ASLib.EXTERNAL_AmplInterface_finalize_solution.argtypes = [ctypes.c_void_p,
                                                                        ctypes.c_int,
                                                                        ctypes.c_char_p,
                                                                        array_1d_double,
                                                                        ctypes.c_int,
                                                                        array_1d_double,
                                                                        ctypes.c_int]
        self.ASLib.EXTERNAL_AmplInterface_finalize_solution.restype = None

        # destructor
        self.ASLib.EXTERNAL_AmplInterface_free_memory.argtypes = [ctypes.c_void_p]
        self.ASLib.EXTERNAL_AmplInterface_free_memory.restype = None

        if filename is not None:
            if nl_buffer is not None:
                raise ValueError("Cannot specify both filename= and nl_buffer=")

            b_data = filename.encode('utf-8')
            self._obj = self.ASLib.EXTERNAL_AmplInterface_new_file(b_data)
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
        self.ASLib.EXTERNAL_AmplInterface_get_bounds_info(self._obj,
                                                          x_l,
                                                          x_u,
                                                          nx,
                                                          g_l,
                                                          g_u,
                                                          ng)

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
        self.ASLib.EXTERNAL_AmplInterface_get_init_multipliers(self._obj, invec, len(invec))

    def eval_f(self, x):
        assert x.size == self._nx, "Error: Dimension missmatch."
        assert x.dtype == np.double, "Error: array type. Function eval_deriv_f expects an array of type double"
        sol = ctypes.c_double()
        res = self.ASLib.EXTERNAL_AmplInterface_eval_f(self._obj, x, self._nx, ctypes.byref(sol))
        assert res, "Error in AMPL evaluation"
        return sol.value

    def eval_deriv_f(self, x, df):
        assert x.size == self._nx, "Error: Dimension missmatch."
        assert x.dtype == np.double, "Error: array type. Function eval_deriv_f expects an array of type double"
        res = self.ASLib.EXTERNAL_AmplInterface_eval_deriv_f(self._obj, x, df, len(x))
        assert res, "Error in AMPL evaluation"

    def struct_jac_g(self, irow, jcol):
        irow_p = irow.astype(np.intc, casting='safe', copy=False)
        jcol_p = jcol.astype(np.intc, casting='safe', copy=False)
        assert len(irow) == len(jcol), "Error: Dimension missmatch. Arrays irow and jcol must be of the same size"
        assert len(irow) == self._nnz_jac_g, "Error: Dimension missmatch. Jacobian has {} nnz".format(self._nnz_jac_g)
        self.ASLib.EXTERNAL_AmplInterface_struct_jac_g(self._obj,
                                                       irow_p,
                                                       jcol_p,
                                                       len(irow))


    def struct_hes_lag(self, irow, jcol):
        irow_p = irow.astype(np.intc, casting='safe', copy=False)
        jcol_p = jcol.astype(np.intc, casting='safe', copy=False)
        assert len(irow) == len(jcol), "Error: Dimension missmatch. Arrays irow and jcol must be of the same size"
        assert len(irow) == self._nnz_hess, "Error: Dimension missmatch. Hessian has {} nnz".format(self._nnz_hess)
        self.ASLib.EXTERNAL_AmplInterface_struct_hes_lag(self._obj,
                                                         irow_p,
                                                         jcol_p,
                                                         len(irow))

    def eval_jac_g(self, x, jac_g_values):
        assert x.size == self._nx, "Error: Dimension missmatch."
        assert jac_g_values.size == self._nnz_jac_g, "Error: Dimension missmatch."
        xeval = x.astype(np.double, casting='safe', copy=False)
        jac_eval = jac_g_values.astype(np.double, casting='safe', copy=False)
        res = self.ASLib.EXTERNAL_AmplInterface_eval_jac_g(self._obj,
                                                           xeval,
                                                           self._nx,
                                                           jac_eval,
                                                           self._nnz_jac_g)
        assert res, "Error in AMPL evaluation"

    def eval_g(self, x, g):
        assert x.size == self._nx, "Error: Dimension missmatch."
        assert g.size == self._ny, "Error: Dimension missmatch."
        assert x.dtype == np.double, "Error: array type. Function eval_g expects an array of type double"
        assert g.dtype == np.double, "Error: array type. Function eval_g expects an array of type double"
        res = self.ASLib.EXTERNAL_AmplInterface_eval_g(self._obj,
                                                       x,
                                                       self._nx,
                                                       g,
                                                       self._ny)
        assert res, "Error in AMPL evaluation"

    def eval_hes_lag(self, x, lam, hes_lag, obj_factor=1.0):
        assert x.size == self._nx, "Error: Dimension missmatch."
        assert lam.size == self._ny, "Error: Dimension missmatch."
        assert hes_lag.size == self._nnz_hess, "Error: Dimension missmatch."
        assert x.dtype == np.double, "Error: array type. Function eval_hes_lag expects an array of type double"
        assert lam.dtype == np.double, "Error: array type. Function eval_hes_lag expects an array of type double"
        assert hes_lag.dtype == np.double, "Error: array type. Function eval_hes_lag expects an array of type double"
        if self.future_libraries:
            res = self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag(self._obj,
                                                                 x,
                                                                 self._nx,
                                                                 lam,
                                                                 self._ny,
                                                                 hes_lag,
                                                                 self._nnz_hess,
                                                                 obj_factor)
        else:
            res = self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag(self._obj,
                                                                 x,
                                                                 self._nx,
                                                                 lam,
                                                                 self._ny,
                                                                 hes_lag,
                                                                 self._nnz_hess)
        assert res, "Error in AMPL evaluation"

    def finalize_solution(self, ampl_solve_status_num, msg, x, lam):
        b_msg = msg.encode('utf-8')
        self.ASLib.EXTERNAL_AmplInterface_finalize_solution(self._obj,
                                                            ampl_solve_status_num,
                                                            b_msg,
                                                            x,
                                                            len(x),
                                                            lam,
                                                            len(lam))



