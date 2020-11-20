#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import gc

# PauseGC is a class for clean, scoped management of the Python
# garbage collector.  To disable the GC for the duration of a
# scoped block use PauseGC in combination with the Python 'with'
# statement. E.g.,
#
# def my_func():
#    with PauseGC() as pgc:
#       [...]
#
# When the instance falls out of scope (by termination or exception),
# the GC will be re-enabled (if it was not initially disabled).  It is
# safe to nest instances of PauseGC That is, you don't have to worry
# if an outer function/method has its own instance of PauseGC.
class PauseGC(object):

    __slots__ = ("reenable_gc",)

    def __init__(self, pause=True):
        self.reenable_gc = False
        if pause:
            self.reenable_gc = gc.isenabled()
            gc.disable()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.reenable_gc:
            gc.enable()
        self.reenable_gc = False
