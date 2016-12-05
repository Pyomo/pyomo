#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import re
import os
from os.path import join, dirname, abspath
import time
import difflib
import filecmp
import shutil
import subprocess
import pyutilib.subprocess
import pyutilib.services
import pyutilib.th as unittest
from pyutilib.pyro import using_pyro3, using_pyro4
from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)
from pyomo.environ import *

from six import StringIO

thisdir = dirname(abspath(__file__))
baselinedir = os.path.join(thisdir, "ddsip_baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisdir)))), "examples", "pysp")

_run_verbose = True

@unittest.category('nightly','expensive')
class TestDDSIPSimple(unittest.TestCase):

    @unittest.nottest
    def _assert_contains(self, filename, *checkstrs):
        with open(filename, 'rb') as f:
            fdata = f.read()
        for checkstr in checkstrs:
            if re.search(checkstr, fdata) is None:
                self.fail("File %s does not contain test string:\n%s\n"
                          "------------------------------------------\n"
                          "File data:\n%s\n"
                          % (filename, checkstr, fdata))

    @unittest.nottest
    def _get_cmd(self,
                 model_location,
                 scenario_tree_location=None,
                 options=None):
        if options is None:
            options = {}
        options['--model-location'] = model_location
        if scenario_tree_location is not None:
            options['--scenario-tree-location'] = \
                scenario_tree_location
        if _run_verbose:
            options['--verbose'] = None
        options['--output-times'] = None
        options['--traceback'] = None
        options['--keep-scenario-files'] = None
        class_name, test_name = self.id().split('.')[-2:]
        options['--output-directory'] = \
            join(thisdir, class_name+"."+test_name)
        if os.path.exists(options['--output-directory']):
            shutil.rmtree(options['--output-directory'],
                          ignore_errors=True)

        cmd = ['python','-m','pyomo.pysp.convert.ddsip']
        for name, val in options.items():
            cmd.append(name)
            if val is not None:
                cmd.append(str(val))
        class_name, test_name = self.id().split('.')[-2:]
        print("%s.%s: Testing command: %s" % (class_name,
                                              test_name,
                                              str(' '.join(cmd))))
        return cmd, options['--output-directory']

    @unittest.nottest
    def _run_bad_conversion_test(self, *args, **kwds):
        cmd, output_dir = self._get_cmd(*args, **kwds)
        outfile = output_dir+".out"
        rc = pyutilib.subprocess.run(cmd, outfile=outfile)
        self.assertNotEqual(rc[0], 0)
        self._assert_contains(
            outfile,
            b"ValueError: One or more deterministic parts of the problem found in file")
        shutil.rmtree(output_dir,
                      ignore_errors=True)
        os.remove(outfile)

    def test_bad_variable_bounds(self):
        self._run_bad_conversion_test(
            join(thisdir, "model_bad_variable_bounds.py"))

    def test_bad_objective_constant(self):
        self._run_bad_conversion_test(
            join(thisdir, "model_bad_objective_constant.py"))

    def test_bad_objective_var(self):
        self._run_bad_conversion_test(
            join(thisdir, "model_bad_objective_var.py"))

    def test_bad_constraint_var(self):
        self._run_bad_conversion_test(
            join(thisdir, "model_bad_constraint_var.py"))

    def test_bad_constraint_rhs(self):
        self._run_bad_conversion_test(
            join(thisdir, "model_bad_constraint_rhs.py"))

    def test_too_many_declarations(self):
        cmd, output_dir = self._get_cmd(
            join(thisdir, "model_too_many_declarations.py"))
        outfile = output_dir+".out"
        rc = pyutilib.subprocess.run(cmd, outfile=outfile)
        self.assertNotEqual(rc[0], 0)
        self._assert_contains(
            outfile,
            b"RuntimeError: Component b.c was "
            b"assigned multiple declarations in annotation type "
            b"StochasticConstraintBodyAnnotation. To correct this "
            b"issue, ensure that multiple container components under "
            b"which the component might be stored \(such as a Block "
            b"and an indexed Constraint\) are not simultaneously set in "
            b"this annotation.")
        shutil.rmtree(output_dir,
                      ignore_errors=True)
        os.remove(outfile)

    def test_bad_component_type(self):
        cmd, output_dir = self._get_cmd(
            join(thisdir, "model_bad_component_type.py"))
        outfile = output_dir+".out"
        rc = pyutilib.subprocess.run(cmd, outfile=outfile)
        self.assertNotEqual(rc[0], 0)
        self._assert_contains(
            outfile,
            b"TypeError: Declarations "
            b"in annotation type StochasticConstraintBodyAnnotation "
            b"must be of types Constraint or Block. Invalid type: "
            b"<class 'pyomo.core.base.objective.SimpleObjective'>")
        shutil.rmtree(output_dir,
                      ignore_errors=True)
        os.remove(outfile)

    def test_unsupported_variable_bounds(self):
        cmd, output_dir = self._get_cmd(
            join(thisdir, "model_unsupported_variable_bounds.py"))
        outfile = output_dir+".out"
        rc = pyutilib.subprocess.run(cmd, outfile=outfile)
        self.assertNotEqual(rc[0], 0)
        self._assert_contains(
            outfile,
            b"ValueError: The DDSIP writer does not currently support "
            b"stochastic variable bounds. Invalid annotation type: "
            b"StochasticVariableBoundsAnnotation")
        shutil.rmtree(output_dir,
                      ignore_errors=True)
        os.remove(outfile)

if __name__ == "__main__":
    unittest.main()
