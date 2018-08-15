#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import six

from pyutilib.misc.config import ConfigBlock, ConfigList, ConfigValue

USER_OPTION = 0
ADVANCED_OPTION = 1
DEVELOPER_OPTION = 2

def PositiveInt(val):
    ans = int(val)
    # We want to give an error for floating point numbers...
    if ans != float(val) or ans <= 0:
        raise ValueError(
            "Expected positive int, but received %s" % (val,))
    return ans

def NegativeInt(val):
    ans = int(val)
    if ans != float(val) or ans >= 0:
        raise ValueError(
            "Expected negative int, but received %s" % (val,))
    return ans

def NonPositiveInt(val):
    ans = int(val)
    if ans != float(val) or ans > 0:
        raise ValueError(
            "Expected non-positive int, but received %s" % (val,))
    return ans

def NonNegativeInt(val):
    ans = int(val)
    if ans != float(val) or ans < 0:
        raise ValueError(
            "Expected non-negative int, but received %s" % (val,))
    return ans

def PositiveFloat(val):
    ans = float(val)
    if ans <= 0:
        raise ValueError(
            "Expected positive float, but received %s" % (val,))
    return ans

def NegativeFloat(val):
    ans = float(val)
    if ans >= 0:
        raise ValueError(
            "Expected negative float, but received %s" % (val,))
    return ans

def NonPositiveFloat(val):
    ans = float(val)
    if ans > 0:
        raise ValueError(
            "Expected non-positive float, but received %s" % (val,))
    return ans

def NonNegativeFloat(val):
    ans = float(val)
    if ans < 0:
        raise ValueError(
            "Expected non-negative float, but received %s" % (val,))
    return ans


class In(object):
    def __init__(self, domain, cast=None):
        self._domain = domain
        self._cast = cast

    def __call__(self, value):
        if self._cast is not None:
            v = self._cast(value)
        else:
            v = value
        if v in self._domain:
            return v
        raise ValueError("value %s not in domain %s" % (value, self._domain))


class Path(object):
    BasePath = None
    SuppressPathExpansion = False

    def __init__(self, basePath=None):
        self.basePath = basePath

    def __call__(self, path):
        #print "normalizing path '%s' " % (path,),
        path = str(path)
        if path is None or Path.SuppressPathExpansion:
            return path

        if self.basePath:
            base = self.basePath
        else:
            base = Path.BasePath
        if type(base) is ConfigValue:
            base = base.value()
        if base is None:
            base = ""
        else:
            base = str(base).lstrip()

        # We want to handle the CWD variable ourselves.  It should
        # always be in a known location (the beginning of the string)
        if base and base[:6].lower() == '${cwd}':
            base = os.getcwd() + base[6:]
        if path and path[:6].lower() == '${cwd}':
            path = os.getcwd() + path[6:]

        ans = os.path.normpath(os.path.abspath(os.path.join(
            os.path.expandvars(os.path.expanduser(base)),
            os.path.expandvars(os.path.expanduser(path)))))
        #print "to '%s'" % (ans,)
        return ans

class PathList(Path):
    def __call__(self, data):
        if hasattr(data, "__iter__") and not isinstance(data, six.string_types):
            return [ super(PathList, self).__call__(i) for i in data ]
        else:
            return [ super(PathList, self).__call__(data) ]


def add_docstring_list(docstring, configblock):
    """Returns the docstring with a formatted configuration arguments listing."""
    return docstring + "    ".join(
        configblock.generate_documentation(
            block_start="Keyword Arguments\n-----------------\n",
            block_end="",
            item_start="%s\n",
            item_body="  %s",
            item_end="",
            indent_spacing=0,
            width=256
        ).splitlines(True))
