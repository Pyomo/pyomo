#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


__all__ = ['register_executable', 'registered_executable', 'unregister_executable']

import pyutilib.misc
from pyomo.common.factory import CachedFactory

ExecutableFactory = CachedFactory('executables')


def unregister_executable(name):
    ExecutableFactory.unregister(name)


def register_executable(name, validate=None):
    #
    # Do not reregister an executable.
    #
    if name in ExecutableFactory:
        return

    @ExecutableFactory.register(name)
    class TMP(object):
        def __init__(self):
            self.executable = None
            self.exec_default = pyutilib.misc.search_file(
                name,
                implicitExt=pyutilib.misc.executable_extension,
                executable=True,
                validate=validate)

        def get_path(self):
            if self.executable:
                return self.executable
            return self.exec_default


def registered_executable(name=None):
    """
    Test if an exectuable is registered.

    If 'name' is None, then return a list of the names of all registered
    executables that are enabled.

    If this executable is not registered or it cannot be found, then
    None is returned.  Otherwise, return the path to the executable.

    NOTE: Since ExecutableFactory is cached, the search for the 
    executable only happens the first time the user calls registered_executable.
    """

    if name is None:
        return sorted(list(ExecutableFactory))
    if name in ExecutableFactory:
        tmp = ExecutableFactory(name)
        if tmp.get_path() is None:
            return None
        return tmp
    return None

