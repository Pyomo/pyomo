from pkg_resources import resource_filename
import numpy.ctypeslib as npct
import numpy as np
import platform
import ctypes
import sys
import os


#TODO: check for 32 or 64 bit and raise error if not supported
class SparseLibInterface(object):
    def __init__(self):
        if os.name in ['nt', 'dos']:
            fname = 'lib/Windows/libpynumero_SPARSE.dll'
        elif sys.platform in ['darwin']:
            fname = 'lib/Darwin/libpynumero_SPARSE.dylib'
        else:
            fname = 'lib/Linux/libpynumero_SPARSE.so'
        self.libname = resource_filename(__name__, fname)
        self.lib = None

    def __call__(self):
        if self.lib is None:
            self._setup()
        return self.lib

    def available(self):
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

def sym_csr_matvec():
    raise RuntimeError("TODO")

def sym_csc_matvec():
    raise RuntimeError("TODO")

def csr_matvec_no_diag():
    raise RuntimeError("TODO")

def csc_matvec_no_diag():
    raise RuntimeError("TODO")
