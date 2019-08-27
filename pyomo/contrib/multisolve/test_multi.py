from os.path import join, normpath

import pyutilib.th as unittest
from pyutilib.misc import import_file

from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.environ import (
    ConcreteModel, Objective, SolverFactory, Var,
    maximize, sin, value, TransformationFactory
)

expath = normpath(join(PYOMO_ROOT_DIR, 'examples', 'gdp'))


@unittest.skipUnless(SolverFactory('gams').available(), "GAMS not available")
@unittest.skipUnless(SolverFactory('ipopt').available(), "IPOPT is not available")
@unittest.skipUnless(SolverFactory('multisolve').available(), "Multisolve is not available")
class MultistartTests(unittest.TestCase):
    def test_ipopt_and_gams(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=1, bounds=(0, 100))
        m.objtv = Objective(expr=m.x1 * sin(m.x1), sense=maximize)

        result = SolverFactory('multisolve').solve(
            m, solvers=[
                'ipopt',
                'gams',
            ], solver_args=[
                dict(solver='dicopt', tee=False),
                dict(solver='baron', tee=False),
            ],
            time_limit=1000,
        )

    def test_8PP(self):
        eight_process_file = import_file(join(expath, 'eight_process', 'eight_proc_model.py'))
        m = eight_process_file.build_eight_process_flowsheet()
        TransformationFactory('gdp.bigm').apply_to(m, bigM=100)

        result = SolverFactory('multisolve').solve(
            m, solvers=[
                'gams',
                'gams',
            ], solver_args=[
                dict(solver='dicopt', tee=False),
                dict(solver='baron', tee=False),
            ],
            time_limit=1000,
        )
        self.assertAlmostEqual(result.problem.upper_bound, 68, 1)
        self.assertAlmostEqual(value(m.profit.expr), 68, 1)


if __name__ == '__main__':
    unittest.main()
