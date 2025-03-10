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

import subprocess
import sys
from os.path import join, exists, splitext

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

import pyomo.environ
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint

import pyomo.solvers.plugins.solvers.SCIPAMPL

currdir = this_file_dir()
deleteFiles = True


@unittest.skipIf(
    'pypy_version_info' in dir(sys), "Skip SCIPAMPL tests on Pypy due to performance"
)
class Test(unittest.TestCase):
    def setUp(self):
        scip = SolverFactory('scip', solver_io='nl')
        type(scip)._known_versions = {}
        TempfileManager.push()

        self.patch_run = unittest.mock.patch(
            'pyomo.solvers.plugins.solvers.SCIPAMPL.subprocess.run'
        )
        # Executable cannot be partially mocked since it creates a PathData object.
        self.patch_path = unittest.mock.patch.object(
            pyomo.common.fileutils.PathData, 'path', autospec=True
        )
        self.patch_available = unittest.mock.patch.object(
            pyomo.common.fileutils.PathData, 'available', autospec=True
        )

        self.run = self.patch_run.start()
        self.path = self.patch_path.start()
        self.available = self.patch_available.start()

        self.executable_paths = {
            "scip": join(currdir, "scip"),
            "scipampl": join(currdir, "scipampl"),
        }

    def tearDown(self):
        self.patch_run.stop()
        self.patch_path.stop()
        self.patch_available.stop()

        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())

    def generate_stdout(self, solver, version):
        if solver == "scip":
            # Template from SCIP 8.0.0
            stdout = (
                "SCIP version {} [precision: 8 byte] [memory: block] [mode: optimized] [LP solver: SoPlex 6.0.0] [GitHash: d9b84b0709]\n"
                "Copyright (C) 2002-2021 Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)\n"
                "\n"
                "External libraries:\n"
                "   SoPlex 6.0.0    Linear Programming Solver developed at Zuse Institute Berlin (soplex.zib.de) [GitHash: f5cfa86b]"
            )

            # Template from SCIPAMPL 7.0.3
        elif solver == "scipampl":
            stdout = (
                "SCIP version {} [precision: 8 byte] [memory: block] [mode: optimized] [LP solver: SoPlex 5.0.2] [GitHash: 74c11e60cd]\n"
                "Copyright (C) 2002-2021 Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)\n"
                "\n"
                "External libraries:\n"
                " Readline 8.0         GNU library for command line editing (gnu.org/s/readline)"
            )
        else:
            raise ValueError("Unsupported solver for stdout generation.")

        version = ".".join(str(e) for e in version[:3])
        return stdout.format(version)

    def set_solvers(self, scip=(8, 0, 0, 0), scipampl=(7, 0, 3, 0), fail=True):
        executables = {"scip": scip, "scipampl": scipampl}

        def get_executable(*args, **kwargs):
            name = args[0]._registered_name
            if name in executables:
                if executables[name]:
                    return self.executable_paths[name]
                else:
                    return None
            elif fail:
                self.fail("Solver creation looked up a non scip executable.")
            else:
                return False

        def get_available(*args, **kwargs):
            name = args[0]._registered_name
            if name in executables:
                return executables[name] is not None
            elif fail:
                self.fail("Solver creation looked up a non scip executable.")
            else:
                return False

        def run(args, **kwargs):
            for solver_name, solver_version in executables.items():
                if not args[0].endswith(solver_name):
                    continue
                if solver_version is None:
                    raise FileNotFoundError()
                else:
                    return subprocess.CompletedProcess(
                        args, 0, self.generate_stdout(solver_name, solver_version), None
                    )
            if fail:
                self.fail("Solver creation looked up a non scip executable.")

        self.path.side_effect = get_executable
        self.available.side_effect = get_available
        self.run.side_effect = run

    def test_scip_available(self):
        self.set_solvers()
        scip = SolverFactory('scip', solver_io='nl')
        scip_executable = scip.executable()
        self.assertIs(scip_executable, self.executable_paths["scip"])
        # only one call to path expected, since no check for SCIPAMPL is needed
        self.assertEqual(1, self.path.call_count)
        self.assertEqual(1, self.run.call_count)
        self.available.assert_called()

        # version should now be cached
        scip.executable()
        self.assertEqual(1, self.run.call_count)

        self.assertTrue(scip.available())

    def test_scipampl_fallback(self):
        self.set_solvers(scip=(7, 0, 3, 0))
        scip = SolverFactory('scip', solver_io='nl')
        scip_executable = scip.executable()
        self.assertIs(scip_executable, self.executable_paths["scipampl"])

        # get SCIP and SCIPAMPL paths
        self.assertEqual(2, self.path.call_count)
        # only check SCIP version
        self.assertEqual(1, self.run.call_count)
        self.available.assert_called()

        self.assertEqual((7, 0, 3, 0), scip._get_version())
        # also check SCIPAMPL version
        self.assertEqual(2, self.run.call_count)

        # versions should now be cached
        scip.executable()
        self.assertEqual(2, self.run.call_count)

        self.assertTrue(scip.available())

    def test_no_scip(self):
        self.set_solvers(scip=None)
        scip = SolverFactory('scip', solver_io='nl')
        scip_executable = scip.executable()
        self.assertIs(scip_executable, self.executable_paths["scipampl"])

        # get scipampl path
        self.assertEqual(1, self.path.call_count)
        # cannot check SCIP version
        self.assertEqual(0, self.run.call_count)
        self.available.assert_called()

        self.assertEqual((7, 0, 3, 0), scip._get_version())
        # also check SCIPAMPL version
        self.assertEqual(1, self.run.call_count)

        # versions should now be cached
        scip.executable()
        self.assertEqual(1, self.run.call_count)

        self.assertTrue(scip.available())

    def test_no_fallback(self):
        self.set_solvers(scip=None, scipampl=None)
        scip = SolverFactory('scip', solver_io='nl')
        self.assertFalse(scip.available())
        self.assertIsNone(scip.executable())

        # cannot check SCIP versions
        self.assertEqual(0, self.run.call_count)
        self.available.assert_called()

    def test_scip_solver_options(self):
        self.set_solvers(fail=False)
        scip = SolverFactory('scip', solver_io='nl')
        m = self.model = ConcreteModel()
        m.v = Var()
        m.o = Objective(expr=m.v)
        m.c = Constraint(expr=m.v >= 1)

        # cache version and reset mock
        scip._get_version()
        self.run.reset_mock()

        # SCIP is not actually called
        with self.assertRaises(FileNotFoundError) as cm:
            scip.solve(m, timelimit=10)

        # SCIP calls should have 3 options
        args = self.run.call_args[0][0]
        self.assertEqual(3, len(args))
        # check scip call
        self.assertEqual(self.executable_paths["scip"], args[0])
        # check for nl file existence
        self.assertTrue(exists(args[1] + ".nl"))
        # check proper sol filename
        self.assertEqual(args[1] + ".sol", cm.exception.filename)
        # check -AMPL option
        self.assertEqual("-AMPL", args[2])
        # check options file
        options_dir = self.run.call_args[1]['cwd']
        self.assertTrue(exists(options_dir + "/scip.set"))
        with open(options_dir + "/scip.set", 'r') as options:
            self.assertEqual(["limits/time = 10\n"], options.readlines())

    def test_scipampl_solver_options(self):
        self.set_solvers(scip=None, fail=False)
        scip = SolverFactory('scip', solver_io='nl')
        m = self.model = ConcreteModel()
        m.v = Var()
        m.o = Objective(expr=m.v)
        m.c = Constraint(expr=m.v >= 1)

        # cache version and reset mock
        scip._get_version()
        self.run.reset_mock()

        # SCIP is not actually called
        with self.assertRaises(FileNotFoundError) as cm:
            scip.solve(m, timelimit=10, options={'numerics/feastol': 1e-09})

        # check scip call
        args = self.run.call_args[0][0]
        self.assertEqual(self.executable_paths["scipampl"], args[0])
        # check for nl file existence
        self.assertTrue(exists(args[1]))
        (root, ext) = splitext(args[1])
        self.assertEqual(".nl", ext)
        # check proper sol filename
        self.assertEqual(root + ".sol", cm.exception.filename)
        # check -AMPL option
        self.assertEqual("-AMPL", args[2])
        # check options file
        options_dir = self.run.call_args[1].get('cwd', None)

        if options_dir is not None and exists(options_dir + "/scip.set"):
            # SCIPAMPL call should have 3 options
            self.assertEqual(3, len(args))
            options_file = options_dir + "/scip.set"
        else:
            # SCIPAMPL call should have 4 options
            self.assertEqual(4, len(args))
            # SCIPAMPL can also receive the options file as the fourth command line argument
            options_file = args[3]
            self.assertTrue(exists(options_file))

        with open(options_file, 'r') as options:
            lines = options.readlines()
            self.assertIn("numerics/feastol = 1e-09\n", lines)
            self.assertIn("limits/time = 10\n", lines)


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()
