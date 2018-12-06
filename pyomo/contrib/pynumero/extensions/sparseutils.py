#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.extensions.utils import find_pynumero_library
from pkg_resources import resource_filename
import numpy.ctypeslib as npct
import numpy as np
import platform
import ctypes
import sys
import os


class SparseLibInterface(object):
    def __init__(self):
        self.libname = find_pynumero_library('pynumero_SPARSE')
        self.lib = None

    def __call__(self):
        if self.lib is None:
            self._setup()
        return self.lib

    def available(self):
        if self.libname is None:
            return False
        return os.path.exists(self.libname)

    def _setup(self):
        if not self.available():
            raise RuntimeError(
                "SparseUtils is not supported on this platform (%s)"
                % (os.name,) )
        self.lib = ctypes.cdll.LoadLibrary(self.libname)
        self.lib.EXTERNAL_SPARSE_sym_coo_matvec.argtypes = [
            array_1d_int,
            array_1d_int,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int
            ]
        self.lib.EXTERNAL_SPARSE_sym_coo_matvec.restype = None

        self.lib.EXTERNAL_SPARSE_csr_matvec_no_diag.argtypes = [
            ctypes.c_int,
            array_1d_int,
            ctypes.c_int,
            array_1d_int,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int
        ]
        self.lib.EXTERNAL_SPARSE_csr_matvec_no_diag.restype = None

        self.lib.EXTERNAL_SPARSE_csc_matvec_no_diag.argtypes = [
            ctypes.c_int,
            array_1d_int,
            ctypes.c_int,
            array_1d_int,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int,
            array_1d_double,
            ctypes.c_int
        ]
        self.lib.EXTERNAL_SPARSE_csc_matvec_no_diag.restype = None

SparseLib = SparseLibInterface()

# define 1d array
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

def sym_coo_matvec(irow, jcol, values, x, result):
    data = values.astype(np.double, casting='safe')
    SparseLib().EXTERNAL_SPARSE_sym_coo_matvec(irow,
                                               jcol,
                                               data,
                                               len(irow),
                                               x,
                                               len(x),
                                               result,
                                               len(result))


def csr_matvec_no_diag(nrows, row_ptr, col_ptr, data_ptr, x_ptr, result):
    datap = data_ptr.astype(np.double, casting='safe')
    xp = x_ptr.astype(np.double, casting='safe')
    SparseLib().EXTERNAL_SPARSE_csr_matvec_no_diag(nrows,
                                                   row_ptr,
                                                   len(row_ptr),
                                                   col_ptr,
                                                   datap,
                                                   len(datap),
                                                   xp,
                                                   len(xp),
                                                   result,
                                                   len(result)
                                                   )


def csc_matvec_no_diag(ncols, col_ptr, row_ptr, data_ptr, x_ptr, result):
    datap = data_ptr.astype(np.double, casting='safe')
    xp = x_ptr.astype(np.double, casting='safe')

    SparseLib().EXTERNAL_SPARSE_csc_matvec_no_diag(ncols,
                                                   col_ptr,
                                                   len(col_ptr),
                                                   row_ptr,
                                                   datap,
                                                   len(datap),
                                                   xp,
                                                   len(xp),
                                                   result,
                                                   len(result)
                                                   )
