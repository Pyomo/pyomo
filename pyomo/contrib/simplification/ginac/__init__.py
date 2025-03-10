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

from pyomo.common.dependencies import attempt_import as _attempt_import


def _importer():
    import os
    import sys
    from ctypes import cdll
    from pyomo.common.envvar import PYOMO_CONFIG_DIR
    from pyomo.common.fileutils import find_library

    try:
        pyomo_config_dir = os.path.join(
            PYOMO_CONFIG_DIR,
            'lib',
            'python%s.%s' % sys.version_info[:2],
            'site-packages',
        )
        sys.path.insert(0, pyomo_config_dir)
        # GiNaC needs 2 libraries that are generally dynamically linked
        # to the interface library.  If we built those ourselves, then
        # the libraries will be PYOMO_CONFIG_DIR/lib ... but that
        # directory is very likely to NOT be on the library search path
        # when the Python interpreter was started.  We will manually
        # look for those two libraries, and if we find them, load them
        # into this process (so the interface can find them)
        for lib in ('cln', 'ginac'):
            fname = find_library(lib)
            if fname is not None:
                cdll.LoadLibrary(fname)

        import ginac_interface
    except ImportError:
        from . import ginac_interface
    finally:
        assert sys.path[0] == pyomo_config_dir
        sys.path.pop(0)

    return ginac_interface


interface, interface_available = _attempt_import('ginac_interface', importer=_importer)
