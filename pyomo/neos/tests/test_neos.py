#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Test NEOS solver interface
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))

import pyutilib.th as unittest

from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.scripting.pyomo_command as main
from pyomo.scripting.util import cleanup
from pyomo.neos.kestrel import kestrelAMPL

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverManagerFactory, SolverStatus, maximize, value

neos_available = False
try:
    # Attempt a connection to NEOS.  Any failure will result in skipping
    # these tests
    if kestrelAMPL().neos is not None:
        neos_available = True
except:
    pass


#
# Because the Kestrel tests require connections to the NEOS server, and
# that can take quite a while (5-20+ seconds), we will only run these
# tests as part of the nightly suite (i.e., by the CI system as part of
# PR / master tests)
#
@unittest.category('nightly')
@unittest.skipIf(not neos_available, "Cannot make connection to NEOS server")
class TestKestrel(unittest.TestCase):

    @unittest.skipIf(not yaml_available, "YAML is not available")
    def test_pyomo_command(self):
        results = os.path.join(currdir, 'result.yml')
        args = [
            os.path.join(currdir,'t1.py'),
            '--solver-manager=neos',
            '--solver=cbc',
            '--symbolic-solver-labels',
            '--save-results=%s' % results,
            '-c',
            ]
        try:
            output = main.run(args)
            self.assertEqual(output.errorcode, 0)

            with open(results) as FILE:
                data = yaml.load(FILE, **yaml_load_args)
            self.assertEqual(
                data['Solver'][0]['Status'], 'ok')
            self.assertAlmostEqual(
                data['Solution'][1]['Status'], 'optimal')
            self.assertAlmostEqual(
                data['Solution'][1]['Objective']['o']['Value'], 1)
            self.assertAlmostEqual(
                data['Solution'][1]['Variable']['x']['Value'], 0.5)
        finally:
            cleanup()
            os.remove(results)

    def test_kestrel_plugin(self):
        m =  ConcreteModel()
        m.x =  Var(bounds=(0,1), initialize=0)
        m.c =  Constraint(expr=m.x <= 0.5)
        m.obj =  Objective(expr=2*m.x, sense= maximize)

        solver_manager =  SolverManagerFactory('neos')
        results = solver_manager.solve(m, opt='cbc')

        self.assertEqual(results.solver[0].status,  SolverStatus.ok)
        self.assertAlmostEqual( value(m.x), 0.5)


if __name__ == "__main__":
    unittest.main()
