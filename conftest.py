#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pytest

def pytest_runtest_setup(item):
    marker = item.iter_markers()
    approved_markers = ['parametrize', 'nightly', 'smoke']
    item_markers = [mark.name for mark in marker]
    if (item_markers and
        not any(mark in approved_markers for mark in item_markers)):
        pytest.skip('SKIPPED: Default test categories are nightly and smoke.')
