#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys as _sys
if _sys.version_info[0] >= 3:
    import importlib

    def _do_import(pkg_name):
        importlib.import_module(pkg_name)
else:
    def _do_import(pkg_name):
        __import__(pkg_name, globals(), locals(), [], -1)

#
# These packages contain plugins that need to be loaded
#
_packages = [
    'pyomo.contrib.preprocessing'
]
#
# These packages are under development, or they may be omitted in a
# Pyomo installation; silently ignore any import errors.
#
_optional_packages = set([])


def _import_packages(packages, optional):
    for name in packages:
        pname = name+'.plugins'
        try:
            _do_import(pname)
        except ImportError:
            if name in optional:
                # Note: Continue to avoid the package load().
                #
                # TODO: Should this generate a warning if the package exists
                # on the file system?
                continue
            else:
                exctype, err, tb = _sys.exc_info()  # BUG?
                import traceback
                msg = "pyomo.environ failed to import %s:\nOriginal %s: %s\n"\
                      "Traceback:\n%s" \
                      % (pname, exctype.__name__, err,
                         ''.join(traceback.format_tb(tb)),)
                # clear local variables to remove circular references
                exctype = err = tb = None
                # TODO: Should this just log an error and re-raise the
                # original exception?
                raise ImportError(msg)

        pkg = _sys.modules[pname]
        pkg.load()


_import_packages(_packages, _optional_packages)
