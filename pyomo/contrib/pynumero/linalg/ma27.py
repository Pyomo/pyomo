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


class MA27Interface(object):

    libname = _NotSet

    @classmethod
    def available(cls):
        if cls.libname is _NotSet:
            cls.libname = find_library('pynumero_MA27')
        if cls.libname is None:
            return False
        return os.path.exists(cls.libname)

    def __init__(self,
                 iw_factor=None,
                 a_factor=None):

        if not MA27Interface.available():
            raise RuntimeError(
                'Could not find pynumero_MA27 library.')

        self.iw_factor = iw_factor
        self.a_factor = a_factor
        self._dim_cached = None

        self.lib = ctypes.cdll.LoadLibrary(self.libname)

        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
        array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

        # Declare arg and res types of functions:
        
        # Do I need to specify that this function takes no argument?
        self.lib.new_MA27_struct.restype = ctypes.c_void_p

        self.lib.free_MA27_struct.argtypes = [ctypes.c_void_p]
        
        self.lib.set_icntl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        # Do I need to specify that this function returns nothing?
        self.lib.get_icntl.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_icntl.restype = ctypes.c_int
        
        self.lib.set_cntl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.lib.get_cntl.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_cntl.restype = ctypes.c_double
        
        self.lib.get_info.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_info.restype = ctypes.c_int
        
        self.lib.alloc_iw_a.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.alloc_iw_b.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.alloc_a.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.lib.do_symbolic_factorization.argtypes = [ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, array_1d_int, array_1d_int]
        self.lib.do_numeric_factorization.argtypes = [ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, array_1d_int, array_1d_int, 
                array_1d_double]
        self.lib.do_backsolve.argtypes = [ctypes.c_void_p, ctypes.c_int, array_1d_double]

        self.icntl_len = 30
        self.cntl_len = 5
        self.info_len = 20

        self._ma27 = self.lib.new_MA27_struct()

    def __del__(self):
        self.lib.free_MA27_struct(self._ma27)


    def set_icntl(self, i, val):
        validate_index(i, self.icntl_len, 'ICNTL')
        validate_value(i, int, 'ICNTL')
        # NOTE: Use the FORTRAN indexing (same as documentation) to
        # set and access info/cntl arrays from Python, whereas C
        # functions use C indexing. Maybe this is too confusing.
        self.lib.set_icntl(self._ma27, i-1, val)


    def get_icntl(self, i):
        validate_index(i, self.icntl_len, 'ICNTL')
        return self.lib.get_icntl(self._ma27, i-1)


    def set_cntl(self, i, val):
        validate_index(i, self.cntl_len, 'CNTL')
        validate_value(val, float, 'CNTL')
        self.lib.set_cntl(self._ma27, i-1, val)


    def get_cntl(self, i):
        validate_index(i, self.cntl_len, 'CNTL')
        return self.lib.get_cntl(self._ma27, i-1)


    def get_info(self, i):
        validate_index(i, self.info_len, 'INFO')
        return self.lib.get_info(self._ma27, i-1)


    def do_symbolic_factorization(self, dim, irn, icn):
        irn = irn.astype(np.intc, casting='safe', copy=True)
        icn = icn.astype(np.intc, casting='safe', copy=True)
        ne = irn.size
        self._dim_cached = dim
        assert ne == icn.size, 'Dimension mismatch in row and column arrays'

        if self.iw_factor is not None:
            min_size = 2*ne + 3*dim + 1
            self.lib.alloc_iw_a(self._ma27,
                    int(self.iw_factor*min_size))

        self.lib.do_symbolic_factorization(self._ma27,
                dim, ne, irn, icn)
        return self.get_info(1)


    def do_numeric_factorization(self, irn, icn, dim, entries):
        irn = irn.astype(np.intc, casting='safe', copy=True)
        icn = icn.astype(np.intc, casting='safe', copy=True)

        ent = entries.astype(np.double, casting='safe', copy=True)

        ne = ent.size
        assert dim == self._dim_cached,\
               ('Dimension mismatch between symbolic and numeric factorization.'
               'Please re-run symbolic factorization with the correct '
               'dimension.')
        if self.a_factor is not None:
            min_size = self.get_info(5)
            self.lib.alloc_a(self._ma27,
                    int(self.a_factor*min_size))
        if self.iw_factor is not None:
            min_size = self.get_info(6)
            self.lib.alloc_iw_b(self._ma27,
                    int(self.iw_factor*min_size))

        self.lib.do_numeric_factorization(self._ma27, dim, ne, 
                irn, icn, ent)
        return self.get_info(1)


    def do_backsolve(self, rhs):
        rhs = rhs.astype(np.double, casting='safe', copy=True)
        rhs_dim = rhs.size
        assert rhs_dim == self._dim_cached,\
            'Dimension mismatch in right hand side. Please correct.'

        self.lib.do_backsolve(self._ma27, rhs_dim, rhs)

        return rhs
