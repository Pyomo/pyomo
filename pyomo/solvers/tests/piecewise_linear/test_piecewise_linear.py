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

import os
from os.path import dirname, abspath, join

thisDir = dirname(abspath(__file__))

import pyomo.common.unittest as unittest

import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.fileutils import import_file
from pyomo.core.base import Var
from pyomo.core.base.objective import minimize, maximize
from pyomo.core.base.piecewise import Bound, PWRepn
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases

smoke_problems = ['convex_var', 'step_var', 'step_vararray']

nightly_problems = [
    'convex_vararray',
    'concave_vararray',
    'concave_var',
    'piecewise_var',
    'piecewise_vararray',
]

expensive_problems = [
    'piecewise_multi_vararray',
    'convex_multi_vararray1',
    'concave_multi_vararray1',
    'convex_multi_vararray2',
    'concave_multi_vararray2',
]


testing_solvers = {}
testing_solvers['gurobi', 'lp'] = False
# testing_solvers['cplex','lp'] = False
# testing_solvers['cplex','nl'] = False
# testing_solvers['ipopt','nl'] = False
# testing_solvers['cplex','python'] = False
# testing_solvers['_cplex_persistent','python'] = False
for _solver, _io in _test_solver_cases():
    if (_solver, _io) in testing_solvers and _test_solver_cases(_solver, _io).available:
        testing_solvers[_solver, _io] = True


def createMethod(pName, problem, solver, writer, kwds):
    def Method(obj):
        if not testing_solvers[solver, writer]:
            obj.skipTest("Solver %s (interface=%s) is not available" % (solver, writer))

        m = import_file(
            os.path.join(thisDir, 'problems', problem + '.py'), clear_cache=True
        )

        model = m.define_model(**kwds)

        opt = pyomo.opt.SolverFactory(solver, solver_io=writer)
        results = opt.solve(model)

        # non-recursive
        new_results = (
            (var.name, var.value)
            for var in model.component_data_objects(
                Var, active=True, descend_into=False
            )
        )
        baseline_results = getattr(obj, problem + '_results')
        for name, value in new_results:
            if abs(baseline_results[name] - value) > 0.0001:
                raise IOError(
                    "Difference in baseline solution values and "
                    "current solution values using:\n"
                    + "Solver: "
                    + solver
                    + "\n"
                    + "Writer: "
                    + writer
                    + "\n"
                    + "Variable: "
                    + name
                    + "\n"
                    + "Solution: "
                    + str(value)
                    + "\n"
                    + "Baseline: "
                    + str(baseline_results[name])
                    + "\n"
                )

    return Method


def assignProblems(cls, problem_list):
    for solver, writer in testing_solvers:
        for PROBLEM in problem_list:
            aux_list = ['', 'force_pw']
            for AUX in aux_list:
                for REPN in PWRepn:
                    for BOUND_TYPE in Bound:
                        for SENSE in [maximize, minimize]:
                            if not (
                                ((BOUND_TYPE == Bound.Lower) and (SENSE == maximize))
                                or ((BOUND_TYPE == Bound.Upper) and (SENSE == minimize))
                                or (
                                    (
                                        REPN
                                        in [
                                            PWRepn.BIGM_BIN,
                                            PWRepn.BIGM_SOS1,
                                            PWRepn.MC,
                                        ]
                                    )
                                    and ('step' in PROBLEM)
                                )
                            ):
                                kwds = {}
                                kwds['sense'] = SENSE
                                kwds['pw_repn'] = REPN
                                kwds['pw_constr_type'] = BOUND_TYPE
                                if SENSE == maximize:
                                    attrName = "test_{0}_{1}_{2}_{3}_{4}_{5}".format(
                                        PROBLEM,
                                        REPN,
                                        BOUND_TYPE,
                                        'maximize',
                                        solver,
                                        writer,
                                    )
                                elif SENSE == minimize:
                                    attrName = "test_{0}_{1}_{2}_{3}_{4}_{5}".format(
                                        PROBLEM,
                                        REPN,
                                        BOUND_TYPE,
                                        'minimize',
                                        solver,
                                        writer,
                                    )
                                if AUX != '':
                                    kwds[AUX] = True
                                    attrName += '_' + AUX
                                setattr(
                                    cls,
                                    attrName,
                                    createMethod(
                                        attrName, PROBLEM, solver, writer, kwds
                                    ),
                                )
                                if yaml_available:
                                    with open(
                                        join(
                                            thisDir,
                                            'baselines',
                                            PROBLEM + '_baseline_results.yml',
                                        ),
                                        'r',
                                    ) as f:
                                        baseline_results = yaml.load(
                                            f, **yaml_load_args
                                        )
                                        setattr(
                                            cls, PROBLEM + '_results', baseline_results
                                        )


@unittest.skipUnless(yaml_available, "PyYAML module is not available.")
class PW_Tests(unittest.TestCase):
    pass


class PiecewiseLinearTest_Smoke(PW_Tests):
    pass


assignProblems(PiecewiseLinearTest_Smoke, smoke_problems)


class PiecewiseLinearTest_Nightly(PW_Tests):
    pass


assignProblems(PiecewiseLinearTest_Nightly, nightly_problems)


@unittest.pytest.mark.expensive
class PiecewiseLinearTest_Expensive(PW_Tests):
    pass


assignProblems(PiecewiseLinearTest_Expensive, expensive_problems)

if __name__ == "__main__":
    unittest.main()
