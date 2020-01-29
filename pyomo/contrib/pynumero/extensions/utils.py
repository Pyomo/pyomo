#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from ctypes.util import find_library
import sys
import os


def find_pynumero_library(library_name):

    lib_path = find_library(library_name)
    if lib_path is not None:
        return lib_path

    # On windows the library is prefixed with 'lib'
    lib_path = find_library('lib'+library_name)
    if lib_path is not None:
        return lib_path
    else:
        # try looking into extensions directory now
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)

        if os.name in ['nt', 'dos']:
            libname = 'lib/Windows/lib{}.dll'.format(library_name)
        elif sys.platform in ['darwin']:
            libname = 'lib/Darwin/lib{}.dylib'.format(library_name)
        else:
            libname = 'lib/Linux/lib{}.so'.format(library_name)

        lib_path = os.path.join(dir_path, libname)

        if os.path.exists(lib_path):
            return lib_path
    return None


def found_pynumero_libraries():

    p1 = find_pynumero_library('pynumero_ASL')
    p2 = find_pynumero_library('pynumero_SPARSE')

    if p1 is not None and p2 is not None:
        return True
    return False
