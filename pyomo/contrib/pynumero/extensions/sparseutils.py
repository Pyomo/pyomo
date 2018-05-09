from pkg_resources import resource_filename
import numpy.ctypeslib as npct
import numpy as np
import platform
import ctypes
import sys
import os


#TODO: check for 32 or 64 bit and raise error if not supported

if os.name in ['nt', 'dos']:
    libsparse = resource_filename(__name__, 'lib/Windows/libpynumero_SPARSE.dll')
    raise RuntimeError("Not supported yet")
elif sys.platform in ['darwin']:
    libsparse = resource_filename(__name__, 'lib/Darwin/libpynumero_SPARSE.so')
else:
    libsparse = resource_filename(__name__, 'lib/Linux/libpynumero_SPARSE.so')

SparseLib = ctypes.cdll.LoadLibrary(libsparse)

# define 1d array
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

SparseLib.EXTERNAL_SPARSE_sym_coo_matvec.argtypes = [array_1d_int,
                                                     array_1d_int,
                                                     array_1d_double,
                                                     ctypes.c_int,
                                                     array_1d_double,
                                                     ctypes.c_int,
                                                     array_1d_double,
                                                     ctypes.c_int]
SparseLib.EXTERNAL_SPARSE_sym_coo_matvec.restype = None


def sym_coo_matvec(irow, jcol, values, x, result):
    data = values.astype(np.double, casting='safe')
    SparseLib.EXTERNAL_SPARSE_sym_coo_matvec(irow,
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
