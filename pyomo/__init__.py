#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# this is a namespace package
try:
    import pkg_resources
    pkg_resources.declare_namespace(__name__)
except ImportError:
    import pkgutil
    orig_get_loader = pkgutil.get_loader
    def get_loader(name):
        try:
            return orig_get_loader(name)
        except AttributeError:
            pass
    pkgutil.get_loader = get_loader
    __path__ = pkgutil.extend_path(__path__, __name__)
