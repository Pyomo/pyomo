from pyomo.contrib.pynumero.extensions.utils import find_pynumero_library
from pkg_resources import resource_filename
import numpy.ctypeslib as npct
import numpy as np
import platform
import ctypes
import sys
import os


class _MA27_LinearSolver(object):

    libname = find_pynumero_library('pynumero_MA27')

    @classmethod
    def available(cls):
        if cls.libname is None:
            return False
        return os.path.exists(cls.libname)

    def __init__(self,
                 pivottol=1e-8,
                 n_a_factor=5.0,
                 n_iw_factor=5.0,
                 mem_increase=2.0):

        if not _MA27_LinearSolver.available():
            raise RuntimeError(
                "HSL interface is not supported on this platform (%s)"
                % (os.name,) )

        self.HSLib = ctypes.cdll.LoadLibrary(_MA27_LinearSolver.libname)

        # define 1d array
        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

        # constructor
        self.HSLib.EXTERNAL_MA27Interface_new.argtypes = [ctypes.c_double,
                                                          ctypes.c_double,
                                                          ctypes.c_double,
                                                          ctypes.c_double]

        self.HSLib.EXTERNAL_MA27Interface_new.restype = ctypes.c_void_p

        # number of nonzeros
        self.HSLib.EXTERNAL_MA27Interface_get_nnz.argtypes = [ctypes.c_void_p]
        self.HSLib.EXTERNAL_MA27Interface_get_nnz.restype = ctypes.c_int

        # get dimension
        self.HSLib.EXTERNAL_MA27Interface_get_dim.argtypes = [ctypes.c_void_p]
        self.HSLib.EXTERNAL_MA27Interface_get_dim.restype = ctypes.c_int

        # number of negative eigenvalues
        self.HSLib.EXTERNAL_MA27Interface_get_num_neg_evals.argtypes = [ctypes.c_void_p]
        self.HSLib.EXTERNAL_MA27Interface_get_num_neg_evals.restype = ctypes.c_int

        # symbolic factorization
        self.HSLib.EXTERNAL_MA27Interface_do_symbolic_factorization.argtypes = [ctypes.c_void_p,
                                                                                ctypes.c_int,
                                                                                array_1d_int,
                                                                                array_1d_int,
                                                                                ctypes.c_int]
        self.HSLib.EXTERNAL_MA27Interface_do_symbolic_factorization.restype = ctypes.c_int

        # numeric factorization
        self.HSLib.EXTERNAL_MA27Interface_do_numeric_factorization.argtypes = [ctypes.c_void_p,
                                                                               ctypes.c_int,
                                                                               ctypes.c_int,
                                                                               array_1d_double,
                                                                               ctypes.c_int]
        self.HSLib.EXTERNAL_MA27Interface_do_numeric_factorization.restype = ctypes.c_int

        # backsolve
        self.HSLib.EXTERNAL_MA27Interface_do_backsolve.argtypes = [ctypes.c_void_p,
                                                                   array_1d_double,
                                                                   ctypes.c_int,
                                                                   array_1d_double,
                                                                   ctypes.c_int]
        self.HSLib.EXTERNAL_MA27Interface_do_backsolve.restype = None

        # destructor
        self.HSLib.EXTERNAL_MA27Interface_free_memory.argtypes = [ctypes.c_void_p]
        self.HSLib.EXTERNAL_MA27Interface_free_memory.restype = None

        # create internal object
        self._obj = self.HSLib.EXTERNAL_MA27Interface_new(pivottol,
                                                          n_a_factor,
                                                          n_iw_factor,
                                                          mem_increase)

    def __del__(self):
        self.HSLib.EXTERNAL_MA27Interface_free_memory(self._obj)

    def get_num_neg_evals(self):
        """
        Return number of negative eigenvalues obtained after factorization

        Returns
        -------
        integer

        """
        return self.HSLib.EXTERNAL_MA27Interface_get_num_neg_evals(self._obj)

    def DoSymbolicFactorization(self, nrowcols, irows, jcols):
        """
        Chooses pivots for Gaussian elimination using a selection criterion to
        preserve sparsity

        Parameters
        ----------
        nrowcols: integer
            size of the matrix
        irows: 1d-array
            pointer of indices (1-base index) from COO format
        jcols: 1d-array
            pointer of indices (1-base index) from COO format


        Returns
        -------
        None

        """
        pirows = irows.astype(np.intc, casting='safe', copy=False)
        pjcols = jcols.astype(np.intc, casting='safe', copy=False)
        assert irows.size == jcols.size, "Dimension error. Pointers should have the same size"
        return self.HSLib.EXTERNAL_MA27Interface_do_symbolic_factorization(self._obj,
                                                                    nrowcols,
                                                                    pirows,
                                                                    pjcols,
                                                                    len(pjcols))

    def DoNumericFactorization(self, nrowcols, values, desired_num_neg_eval=-1):
        """
        factorizes a matrix using the information from a previous call of DoSymbolicFactorization

        Parameters
        ----------
        nrowcols: integer
            size of the matrix
        values: 1d-array
            pointer of values from COO format
        desired_num_neg_eval: integer
            number of negative eigenvalues desired. This is used for inertia correction

        Returns
        -------
        status: {0, 1, 2}
            status obtained from MA27 after factorizing matrix. 0: success, 1: singular, 2: incorrect inertia
        """

        pvalues = values.astype(np.double, casting='safe', copy=False)
        return self.HSLib.EXTERNAL_MA27Interface_do_numeric_factorization(self._obj,
                                                                          nrowcols,
                                                                          len(pvalues),
                                                                          pvalues,
                                                                          desired_num_neg_eval)

    def DoBacksolve(self, rhs, sol):
        """
        Uses the factors generated by DoNumericFactorization to solve a system of equation

        Parameters
        ----------
        rhs
        sol

        Returns
        -------

        """

        assert sol.size == rhs.size, "Dimension error. Pointers should have the same size"
        prhs = rhs.astype(np.double, casting='safe', copy=False)
        psol = sol.astype(np.double, casting='safe', copy=False)
        return self.HSLib.EXTERNAL_MA27Interface_do_backsolve(self._obj,
                                                              prhs,
                                                              len(prhs),
                                                              psol,
                                                              len(psol))


class _MA57_LinearSolver(object):

    libname = find_pynumero_library('pynumero_MA57')

    @classmethod
    def available(cls):
        if cls.libname is None:
            return False
        return os.path.exists(cls.libname)

    def __init__(self, pivottol=1e-8, prealocate_factor=1.05):

        if not _MA57_LinearSolver.available():
            raise RuntimeError(
                "HSL interface is not supported on this platform (%s)"
                % (os.name,) )

        self.HSLib = ctypes.cdll.LoadLibrary(_MA57_LinearSolver.libname)

        # define 1d array
        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

        # constructor
        self.HSLib.EXTERNAL_MA57Interface_new.argtypes = [ctypes.c_double,
                                                          ctypes.c_double]
        self.HSLib.EXTERNAL_MA57Interface_new.restype = ctypes.c_void_p

        # number of nonzeros
        self.HSLib.EXTERNAL_MA57Interface_get_nnz.argtypes = [ctypes.c_void_p]
        self.HSLib.EXTERNAL_MA57Interface_get_nnz.restype = ctypes.c_int

        # get dimension
        self.HSLib.EXTERNAL_MA57Interface_get_dim.argtypes = [ctypes.c_void_p]
        self.HSLib.EXTERNAL_MA57Interface_get_dim.restype = ctypes.c_int

        # number of negative eigenvalues
        self.HSLib.EXTERNAL_MA57Interface_get_num_neg_evals.argtypes = [ctypes.c_void_p]
        self.HSLib.EXTERNAL_MA57Interface_get_num_neg_evals.restype = ctypes.c_int

        # symbolic factorization
        self.HSLib.EXTERNAL_MA57Interface_do_symbolic_factorization.argtypes = [ctypes.c_void_p,
                                                                                ctypes.c_int,
                                                                                array_1d_int,
                                                                                array_1d_int,
                                                                                ctypes.c_int]
        self.HSLib.EXTERNAL_MA57Interface_do_symbolic_factorization.restype = ctypes.c_int

        # numeric factorization
        self.HSLib.EXTERNAL_MA57Interface_do_numeric_factorization.argtypes = [ctypes.c_void_p,
                                                                               ctypes.c_int,
                                                                               ctypes.c_int,
                                                                               array_1d_double,
                                                                               ctypes.c_int]
        self.HSLib.EXTERNAL_MA57Interface_do_numeric_factorization.restype = ctypes.c_int

        # backsolve
        self.HSLib.EXTERNAL_MA57Interface_do_backsolve.argtypes = [ctypes.c_void_p,
                                                                   array_1d_double,
                                                                   ctypes.c_int,
                                                                   array_1d_double,
                                                                   ctypes.c_int]
        self.HSLib.EXTERNAL_MA57Interface_do_backsolve.restype = None

        # destructor
        self.HSLib.EXTERNAL_MA57Interface_free_memory.argtypes = [ctypes.c_void_p]
        self.HSLib.EXTERNAL_MA57Interface_free_memory.restype = None

        # create internal object
        self._obj = self.HSLib.EXTERNAL_MA57Interface_new(pivottol,
                                                          prealocate_factor)

    def __del__(self):
        self.HSLib.EXTERNAL_MA57Interface_free_memory(self._obj)

    def get_num_neg_evals(self):
        """
        Return number of negative eigenvalues obtained after factorization

        Returns
        -------
        integer

        """
        return self.HSLib.EXTERNAL_MA57Interface_get_num_neg_evals(self._obj)

    def DoSymbolicFactorization(self, nrowcols, irows, jcols):
        """
        Chooses pivots for Gaussian elimination using a selection criterion to
        preserve sparsity

        Parameters
        ----------
        nrowcols: integer
            size of the matrix
        irows: 1d-array
            pointer of indices (1-base index) from COO format
        jcols: 1d-array
            pointer of indices (1-base index) from COO format


        Returns
        -------
        None

        """
        pirows = irows.astype(np.intc, casting='safe', copy=False)
        pjcols = jcols.astype(np.intc, casting='safe', copy=False)
        msg = "Dimension error. Pointers should have the same size"
        assert irows.size == jcols.size, msg
        return self.HSLib.EXTERNAL_MA57Interface_do_symbolic_factorization(self._obj,
                                                                    nrowcols,
                                                                    pirows,
                                                                    pjcols,
                                                                    len(pjcols))

    def DoNumericFactorization(self, nrowcols, values, desired_num_neg_eval=-1):
        """
        factorizes a matrix using the information from a previous call of DoSymbolicFactorization

        Parameters
        ----------
        nrowcols: integer
            size of the matrix
        values: 1d-array
            pointer of values from COO format
        desired_num_neg_eval: integer
            number of negative eigenvalues desired. This is used for inertia correction

        Returns
        -------
        status: {0, 1, 2}
            status obtained from MA27 after factorizing matrix. 0: success, 1: singular, 2: incorrect inertia
        """

        pvalues = values.astype(np.double, casting='safe', copy=False)
        return self.HSLib.EXTERNAL_MA57Interface_do_numeric_factorization(self._obj,
                                                                          nrowcols,
                                                                          len(pvalues),
                                                                          pvalues,
                                                                          desired_num_neg_eval)

    def DoBacksolve(self, rhs, sol):
        """
        Uses the factors generated by DoNumericFactorization to solve a system of equation

        Parameters
        ----------
        rhs
        sol

        Returns
        -------

        """

        assert sol.size == rhs.size, "Dimension error. Pointers should have the same size"
        prhs = rhs.astype(np.double, casting='safe', copy=False)
        psol = sol.astype(np.double, casting='safe', copy=False)
        return self.HSLib.EXTERNAL_MA57Interface_do_backsolve(self._obj,
                                                              prhs,
                                                              len(prhs),
                                                              psol,
                                                              len(psol))
