from pkg_resources import resource_filename
import numpy.ctypeslib as npct
import numpy as np
import platform
import ctypes
import sys
import os


#TODO: ask about assertions
#TODO: ask about freeing memory

class MA27_LinearSolver(object):
    if os.name in ['nt', 'dos']:
        libname = 'lib/Windows/libpynumero_HSL.dll'
    elif sys.platform in ['darwin']:
        libname = 'lib/Darwin/libpynumero_HSL.so'
    else:
        libname = 'lib/Linux/libpynumero_HSL.so'
    libname = resource_filename(__name__, libname)

    @classmethod
    def available(cls):
        return os.path.exists(cls.libname)

    def __init__(self, pivottol=1e-8):

        #TODO: check for 32 or 64 bit and raise error if not supported

        if not MA27_LinearSolver.available():
            raise RuntimeError(
                "HSL interface is not supported on this platform (%s)"
                % (os.name,) )

        self.HSLib = ctypes.cdll.LoadLibrary(MA27_LinearSolver.libname)

        # define 1d array
        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

        # constructor
        self.HSLib.EXTERNAL_MA27Interface_new.argtypes = [ctypes.c_double]
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

        # get values
        self.HSLib.EXTERNAL_MA27Interface_get_values.argtypes = [ctypes.c_void_p,
                                                                 ctypes.c_int,
                                                                 array_1d_double]
        self.HSLib.EXTERNAL_MA27Interface_get_values.restype = None

        # symbolic factorization
        self.HSLib.EXTERNAL_MA27Interface_do_symbolic_factorization.argtypes = [ctypes.c_void_p,
                                                                                ctypes.c_int,
                                                                                array_1d_int,
                                                                                array_1d_int,
                                                                                ctypes.c_int]
        self.HSLib.EXTERNAL_MA27Interface_do_symbolic_factorization.restype = None

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
        self._obj = self.HSLib.EXTERNAL_MA27Interface_new(pivottol)

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
        self.HSLib.EXTERNAL_MA27Interface_do_symbolic_factorization(self._obj,
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





