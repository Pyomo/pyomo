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

from pyomo.common.deprecation import moved_module

for _module in ('base', 'config', 'factory', 'results', 'util', 'persistent'):
    moved_module(
        f'pyomo.contrib.solver.{_module}',
        f'pyomo.contrib.solver.common.{_module}',
        version='6.9.2',
    )

moved_module(
    'pyomo.contrib.solver.solution',
    'pyomo.contrib.solver.common.solution_loader',
    version='6.9.2',
)

for _module in ('ipopt', 'gurobi_direct', 'sol_reader'):
    moved_module(
        f'pyomo.contrib.solver.{_module}',
        f'pyomo.contrib.solver.solvers.{_module}',
        version='6.9.2',
    )

moved_module(
    'pyomo.contrib.solver.gurobi',
    'pyomo.contrib.solver.solvers.gurobi_persistent',
    version='6.9.2',
)

del _module, moved_module
