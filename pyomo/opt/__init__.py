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


from pyomo.opt.base import (
    check_available_solvers,
    convert,
    convert_problem,
    error,
    formats,
    guess_format,
    opt_config,
    solvers,
    AbstractProblemWriter,
    AbstractResultsReader,
    BranchDirection,
    ConverterError,
    OptSolver,
    ProblemFormat,
    ReaderFactory,
    ResultsFormat,
    SolverFactory,
    UnknownSolver,
    WriterFactory,
)

from pyomo.opt.results import (
    container,
    problem,
    solution,
    ListContainer,
    MapContainer,
    UndefinedData,
    undefined,
    ignore,
    SolverStatus,
    TerminationCondition,
    check_optimal_termination,
    assert_optimal_termination,
    ProblemSense,
    SolutionStatus,
    Solution,
    results_,
    SolverResults,
)

from pyomo.opt.problem import ampl, AmplModel

from pyomo.opt.parallel import (
    manager,
    async_solver,
    local,
    SolverManagerFactory,
    AsynchronousSolverManager,
)

from pyomo.common.deprecation import relocated_module_attribute

for _attr in ('ScalarData', 'ScalarType', 'default_print_options'):
    relocated_module_attribute(
        _attr, 'pyomo.opt.results.container.' + _attr, version='6.0'
    )
del _attr
del relocated_module_attribute
