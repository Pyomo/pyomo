# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Test models and regression tests for MindtPy."""

from importlib import import_module
import sys

_legacy_module_aliases = {
    'MINLP_simple': 'minlp_simple',
    'MINLP2_simple': 'minlp2_simple',
    'MINLP3_simple': 'minlp3_simple',
    'MINLP4_simple': 'minlp4_simple',
    'MINLP5_simple': 'minlp5_simple',
    'MINLP_simple_grey_box': 'minlp_simple_grey_box',
}

for _legacy_name, _module_name in _legacy_module_aliases.items():
    sys.modules[f'{__name__}.{_legacy_name}'] = import_module(
        f'.{_module_name}', __name__
    )

del _legacy_name, _module_name, _legacy_module_aliases, import_module, sys
