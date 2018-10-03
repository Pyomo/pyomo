from ctypes.util import find_library
import sys
import os


def find_pynumero_library(library_name):

    asl_path = find_library(library_name)
    if asl_path is not None:
        return asl_path
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

        asl_lib_path = os.path.join(dir_path, libname)

        if os.path.exists(asl_lib_path):
            return asl_lib_path
    return None


def found_pynumero_libraries():

    p1 = find_pynumero_library('pynumero_ASL')
    p2 = find_pynumero_library('pynumero_SPARSE')

    if p1 is not None and p2 is not None:
        return True
    return False
