#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
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
    from pyomo.common.envvar import PYOMO_CONFIG_DIR

    try:
        pyomo_config_dir = os.path.join(
            PYOMO_CONFIG_DIR,
            'lib',
            'python%s.%s' % sys.version_info[:2],
            'site-packages',
        )
        sys.path.insert(0, pyomo_config_dir)
        import appsi_cmodel
    except ImportError:
        from . import appsi_cmodel
    finally:
        assert sys.path[0] == pyomo_config_dir
        sys.path.pop(0)

    return appsi_cmodel


cmodel, cmodel_available = _attempt_import(
    'appsi_cmodel',
    error_message=(
        'Appsi requires building a small c++ extension. '
        'Please use the "pyomo build-extensions" command'
    ),
    importer=_importer,
)
