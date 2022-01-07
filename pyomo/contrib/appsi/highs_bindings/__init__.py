#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import as _attempt_import

def _importer():
    import os
    import sys
    from pyomo.common.envvar import PYOMO_CONFIG_DIR
    try:
        pyomo_config_dir = os.path.join(
            PYOMO_CONFIG_DIR, 'lib', 'python%s.%s' % sys.version_info[:2],
            'site-packages')
        sys.path.insert(0, pyomo_config_dir)
        import appsi_highs
    except ImportError:
        from . import appsi_highs
    finally:
        assert sys.path[0] == pyomo_config_dir
        sys.path.pop(0)

    return appsi_highs

pyhighs, pyhighs_available = _attempt_import(
    'appsi_highs',
    error_message=('Unable to import Appsi Python bings for Highs.'),
    importer=_importer,
)
