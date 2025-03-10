#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
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
from pyomo.common.multithread import MultiThreadWrapper


class __PauseGCCompanion(object):
    def __init__(self):
        self._stack_depth = 0


PauseGCCompanion: __PauseGCCompanion = MultiThreadWrapper(__PauseGCCompanion)


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
    __slots__ = ("reenable_gc", "stack_pointer")

    def __init__(self):
        self.stack_pointer = None
        self.reenable_gc = None

    def __enter__(self):
        if self.stack_pointer:
            raise RuntimeError(
                "Entering PauseGC context manager that was already entered."
            )
        PauseGCCompanion._stack_depth += 1
        self.stack_pointer = PauseGCCompanion._stack_depth
        self.reenable_gc = gc.isenabled()
        if self.reenable_gc:
            gc.disable()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if not self.stack_pointer:
            return
        if PauseGCCompanion._stack_depth:
            if PauseGCCompanion._stack_depth != self.stack_pointer:
                raise RuntimeError(
                    "Exiting PauseGC context manager out of order: there "
                    "are other active PauseGC context managers that were "
                    "entered after this context manager and have not yet "
                    "been exited."
                )
            PauseGCCompanion._stack_depth -= 1
            self.stack_pointer = None
        if self.reenable_gc:
            gc.enable()
        self.reenable_gc = None
