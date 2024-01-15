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

#
# This module supports testing the attempt_import() functionality when
# used at the module scope.  It cannot be in the actual test module, as
# pytest accesses objects in the module scope during test collection
# (which would inadvertently trigger premature module import)
#

from pyomo.common.dependencies import attempt_import

from pyomo.common.tests.dep_mod import (
    bogus_nonexisting_module as bogus_nem,
    bogus_nonexisting_module_available as has_bogus_nem,
)

bogus, bogus_available = attempt_import('nonexisting.module.bogus', defer_check=True)

pkl_test, pkl_available = attempt_import(
    'nonexisting.module.pickle_test', deferred_submodules=['submod'], defer_check=True
)

pyo, pyo_available = attempt_import(
    'pyomo',
    alt_names=['pyo'],
    deferred_submodules={'version': None, 'common.tests.dep_mod': ['dm']},
)

dm = pyo.common.tests.dep_mod


def test_access_bogus_hello():
    bogus_nem.hello
