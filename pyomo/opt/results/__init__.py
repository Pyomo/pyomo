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

from pyomo.opt.results.container import (
    ListContainer,
    MapContainer,
    UndefinedData,
    undefined,
    ignore,
)

from pyomo.opt.results.solver import (
    SolverStatus,
    TerminationCondition,
    check_optimal_termination,
    assert_optimal_termination,
)
from pyomo.opt.results.problem import ProblemSense
from pyomo.opt.results.solution import SolutionStatus, Solution
from pyomo.opt.results.results_ import SolverResults

from pyomo.common.deprecation import relocated_module_attribute

for _attr in ('ScalarData', 'ScalarType', 'default_print_options', 'strict'):
    relocated_module_attribute(
        _attr, 'pyomo.opt.results.container.' + _attr, version='6.8.1'
    )
del _attr
del relocated_module_attribute
