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
from pyomo.contrib.pynumero.linalg.utils import (validate_index,
        validate_value, _NotSet)
import numpy.ctypeslib as npct
import numpy as np
import ctypes 
import os

class MA57Interface(object):

    libname = _NotSet

    @classmethod
    def available(cls):
        if cls.libname is _NotSet:
            cls.libname = find_library('pynumero_MA57')
        if cls.libname is None:
            return False
        return os.path.exists(cls.libname)

    def __init__(self,
                 work_factor=None,
                 fact_factor=None,
                 ifact_factor=None):

        if not MA57Interface.available():
            raise RuntimeError(
                'Could not find pynumero_MA57 library.')

        self.work_factor = work_factor
        self.fact_factor = fact_factor
        self.ifact_factor = ifact_factor

        self.lib = ctypes.cdll.LoadLibrary(self.libname)

        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
        array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

        # Declare arg and res types of functions:
        
        # Do I need to specify that this function takes no argument?
        self.lib.new_MA57_struct.restype = ctypes.c_void_p
        # return type is pointer to MA57_struct. Why do I use c_void_p here?

        self.lib.free_MA57_struct.argtypes = [ctypes.c_void_p]
        
        self.lib.set_icntl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        # Do I need to specify that this function returns nothing?
        self.lib.get_icntl.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_icntl.restype = ctypes.c_int
        
        self.lib.set_cntl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.lib.get_cntl.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_cntl.restype = ctypes.c_double
        
        self.lib.get_info.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_info.restype = ctypes.c_int
        
        self.lib.get_rinfo.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_rinfo.restype = ctypes.c_double

        self.lib.alloc_keep.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.alloc_work.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.alloc_fact.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.alloc_ifact.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.lib.set_nrhs.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.set_lrhs.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.set_job.argtypes = [ctypes.c_void_p, ctypes.c_int]
        
        self.lib.do_symbolic_factorization.argtypes = [ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, array_1d_int, array_1d_int]
        self.lib.do_numeric_factorization.argtypes = [ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, array_1d_double]
        self.lib.do_backsolve.argtypes = [ctypes.c_void_p, ctypes.c_int, array_2d_double]
        self.lib.do_iterative_refinement.argtypes = [ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, array_1d_double, array_1d_int, array_1d_int, 
                array_1d_double, array_1d_double, array_1d_double]
        self.lib.do_reallocation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
                ctypes.c_int]

        self.icntl_len = 20
        self.cntl_len = 5
        self.info_len = 40
        self.rinfo_len = 20

        self._ma57 = self.lib.new_MA57_struct()


    def __del__(self):
        self.lib.free_MA57_struct(self._ma57)


    def set_icntl(self, i, val):
        validate_index(i, self.icntl_len, 'ICNTL')
        validate_value(i, int, 'ICNTL')
        # NOTE: Use the FORTRAN indexing (same as documentation) to
        # set and access info/cntl arrays from Python, whereas C
        # functions use C indexing. Maybe this is too confusing.
        self.lib.set_icntl(self._ma57, i-1, val)


    def get_icntl(self, i):
        validate_index(i, self.icntl_len, 'ICNTL')
        return self.lib.get_icntl(self._ma57, i-1)


    def set_cntl(self, i, val):
        validate_index(i, self.cntl_len, 'CNTL')
        validate_value(val, float, 'CNTL')
        self.lib.set_cntl(self._ma57, i-1, val)


    def get_cntl(self, i):
        validate_index(i, self.cntl_len, 'CNTL')
        return self.lib.get_cntl(self._ma57, i-1)


    def get_info(self, i):
        validate_index(i, self.info_len, 'INFO')
        return self.lib.get_info(self._ma57, i-1)


    def get_rinfo(self, i):
        validate_index(i, self.rinfo_len, 'RINFO')
        return self.lib.get_info(self._ma57, i-1)


    def do_symbolic_factorization(self, dim, irn, jcn):
        irn = irn.astype(np.intc, casting='safe', copy=True)
        jcn = jcn.astype(np.intc, casting='safe', copy=True)
        # TODO: maybe allow user the option to specify size of KEEP
        ne = irn.size
        self.ne_cached = ne
        self.dim_cached = dim
        assert ne == jcn.size, 'Dimension mismatch in row and column arrays'
        self.lib.do_symbolic_factorization(self._ma57,
                dim, ne, irn, jcn)
        return self.get_info(1)


    def do_numeric_factorization(self, dim, entries):
        entries = entries.astype(np.float64, casting='safe', copy=True)
        ne = entries.size
        assert ne == self.ne_cached,\
               ('Wrong number of entries in matrix. Please re-run symbolic'
               'factorization with correct nonzero coordinates.')
        assert dim == self.dim_cached,\
               ('Dimension mismatch between symbolic and numeric factorization.'
               'Please re-run symbolic factorization with the correct '
               'dimension.')
        if self.fact_factor is not None:
            min_size = self.get_info(9)
            self.lib.alloc_fact(self._ma57, 
                    int(self.fact_factor*min_size))
        if self.ifact_factor is not None:
            min_size = self.get_info(10)
            self.lib.alloc_ifact(self._ma57,
                    int(self.ifact_factor*min_size))

        self.lib.do_numeric_factorization(self._ma57,
                dim, ne, entries)
        return self.get_info(1)


    def do_backsolve(self, rhs):
        rhs = rhs.astype(np.double, casting='safe', copy=True)
        shape = rhs.shape
        if len(shape) == 1:
            rhs_dim = rhs.size
            nrhs = 1
            rhs = np.array([rhs])
        elif len(shape) == 2:
            # FIXME
            raise NotImplementedError(
                'Funcionality for solving a matrix of right hand '
                'is buggy and needs fixing.')
            rhs_dim = rhs.shape[0]
            nrhs = rhs.shape[1]
        else:
            raise ValueError(
                'Right hand side must be a one or two-dimensional array')
        # This does not necessarily need to be true; each RHS could have length
        # larger than N (for some reason). In the C interface, however, I assume
        # that LRHS == N
        assert self.dim_cached == rhs_dim, 'Dimension mismatch in RHS'
        # TODO: Option to specify a JOB other than 1. By my understanding,
        # different JOBs allow partial factorizations to be performed.
        # Currently not supported - unclear if it should be.
        
        if nrhs > 1:
            self.lib.set_nrhs(self._ma57, nrhs)
        
        if self.work_factor is not None:
            self.lib.alloc_work(self._ma57,
                    int(self.work_factor*nrhs*rhs_dim))

        self.lib.do_backsolve(self._ma57,
                rhs_dim, rhs)

        if len(shape) == 1:
            # If the user input rhs as a 1D array, return the solution
            # as a 1D array.
            rhs = rhs[0, :]

        return rhs
