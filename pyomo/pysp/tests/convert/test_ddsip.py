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
import sys
import pyutilib.subprocess
import pyutilib.th as unittest
from pyutilib.pyro import using_pyro3, using_pyro4
from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)
import pyomo.environ

from six import StringIO

thisdir = dirname(abspath(__file__))
baselinedir = os.path.join(thisdir, "ddsip_baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisdir)))), "examples", "pysp")

_run_verbose = True

@unittest.category('nightly')
class TestConvertDDSIPSimple(unittest.TestCase):

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

        cmd = [sys.executable,'-m','pyomo.pysp.convert.ddsip']
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

class _DDSIPTesterBase(object):

    baseline_basename = None
    model_location = None
    scenario_tree_location = None
    extra_options = None

    def setUp(self):
        self._tempfiles = []
        self.options = {}
        self.options['--scenario-tree-manager'] = 'serial'
        if self.extra_options is not None:
            self.options.update(self.extra_options)

    def _run_cmd(self, cmd):
        class_name, test_name = self.id().split('.')[-2:]
        outname = os.path.join(thisdir,
                               class_name+"."+test_name+".out")
        self._tempfiles.append(outname)
        with open(outname, "w") as f:
            subprocess.check_call(cmd,
                                  stdout=f,
                                  stderr=subprocess.STDOUT)

    def _cleanup(self):
        for fname in self._tempfiles:
            try:
                os.remove(fname)
            except OSError:
                pass
        self._tempfiles = []

    def _setup(self, options):
        assert self.baseline_basename is not None
        assert self.model_location is not None
        options['--model-location'] = self.model_location
        options['--symbolic-solver-labels'] = None
        if self.scenario_tree_location is not None:
            options['--scenario-tree-location'] = self.scenario_tree_location
        if _run_verbose:
            options['--verbose'] = None
        options['--output-times'] = None
        options['--traceback'] = None
        class_name, test_name = self.id().split('.')[-2:]
        options['--output-directory'] = \
            join(thisdir, class_name+"."+test_name)
        if os.path.exists(options['--output-directory']):
            shutil.rmtree(options['--output-directory'], ignore_errors=True)

    def _get_cmd(self):
        cmd = [sys.executable,'-m','pyomo.pysp.convert.ddsip']
        for name, val in self.options.items():
            cmd.append(name)
            if val is not None:
                cmd.append(str(val))
        class_name, test_name = self.id().split('.')[-2:]
        print("%s.%s: Testing command: %s" % (class_name,
                                              test_name,
                                              str(' '.join(cmd))))
        return cmd

    def _diff(self, baselinedir, outputdir, dc=None):
        if dc is None:
            dc = filecmp.dircmp(baselinedir,
                                outputdir,
                                ['.svn'])
        if dc.left_only:
            self.fail("Files or subdirectories missing from output: "
                      +str(dc.left_only))
        if dc.right_only:
            self.fail("Files or subdirectories missing from baseline: "
                      +str(dc.right_only))
        for name in dc.diff_files:
            fromfile = join(dc.left, name)
            tofile = join(dc.right, name)
            with open(fromfile, 'r') as f_from:
                fromlines = f_from.readlines()
                with open(tofile, 'r') as f_to:
                    tolines = f_to.readlines()
                    diff = difflib.context_diff(fromlines, tolines,
                                                fromfile+" (baseline)",
                                                tofile+" (output)")
                    diff = list(diff)
                    # The filecmp.dircmp function does a weaker
                    # comparison that can sometimes lead to false
                    # positives. Make sure the true diff is not empty
                    # before we call this a failure.
                    if len(diff) > 0:
                        out = StringIO()
                        out.write("Output file does not match baseline:\n")
                        for line in diff:
                            out.write(line)
                        self.fail(out.getvalue())
        for subdir in dc.subdirs:
            self._diff(join(baselinedir, subdir),
                       join(outputdir, subdir),
                       dc=dc.subdirs[subdir])
        shutil.rmtree(outputdir, ignore_errors=True)

    def test_scenarios(self):
        self._setup(self.options)
        cmd = self._get_cmd()
        self._run_cmd(cmd)
        self._diff(os.path.join(baselinedir, self.baseline_basename+'_baseline'),
                   self.options['--output-directory'])
        self._cleanup()

_pyomo_ns_host = '127.0.0.1'
_pyomo_ns_port = None
_pyomo_ns_process = None
_dispatch_srvr_port = None
_dispatch_srvr_process = None
_taskworker_processes = []

def tearDownModule():
    global _pyomo_ns_port
    global _pyomo_ns_process
    global _dispatch_srvr_port
    global _dispatch_srvr_process
    global _taskworker_processes
    _kill(_pyomo_ns_process)
    _pyomo_ns_port = None
    _pyomo_ns_process = None
    _kill(_dispatch_srvr_process)
    _dispatch_srvr_port = None
    _dispatch_srvr_process = None
    [_kill(proc) for proc in _taskworker_processes]
    _taskworker_processes = []
    if os.path.exists(join(thisdir, "Pyro_NS_URI")):
        try:
            os.remove(join(thisdir, "Pyro_NS_URI"))
        except OSError:
            pass

class _DDSIPPyroTesterBase(_DDSIPTesterBase):

    def _setUpPyro(self):
        global _pyomo_ns_port
        global _pyomo_ns_process
        global _dispatch_srvr_port
        global _dispatch_srvr_process
        global _taskworker_processes
        if _pyomo_ns_process is None:
            _pyomo_ns_process, _pyomo_ns_port = \
                _get_test_nameserver(ns_host=_pyomo_ns_host)
        assert _pyomo_ns_process is not None
        if _dispatch_srvr_process is None:
            _dispatch_srvr_process, _dispatch_srvr_port = \
                _get_test_dispatcher(ns_host=_pyomo_ns_host,
                                     ns_port=_pyomo_ns_port)
        assert _dispatch_srvr_process is not None
        class_name, test_name = self.id().split('.')[-2:]
        if len(_taskworker_processes) == 0:
            for i in range(3):
                outname = os.path.join(thisdir,
                                       class_name+"."+test_name+".scenariotreeserver_"+str(i+1)+".out")
                self._tempfiles.append(outname)
                with open(outname, "w") as f:
                    _taskworker_processes.append(
                        subprocess.Popen(["scenariotreeserver", "--traceback"] + \
                                         (["--verbose"] if _run_verbose else []) + \
                                         ["--pyro-host="+str(_pyomo_ns_host)] + \
                                         ["--pyro-port="+str(_pyomo_ns_port)],
                                         stdout=f,
                                         stderr=subprocess.STDOUT))

            time.sleep(2)
            [_poll(proc) for proc in _taskworker_processes]

    def setUp(self):
        self._tempfiles = []
        self._setUpPyro()
        [_poll(proc) for proc in _taskworker_processes]
        self.options = {}
        self.options['--scenario-tree-manager'] = 'pyro'
        self.options['--pyro-host'] = 'localhost'
        self.options['--pyro-port'] = _pyomo_ns_port
        self.options['--pyro-required-scenariotreeservers'] = 3
        if self.extra_options is not None:
            self.options.update(self.extra_options)

    def _setup(self, options, servers=None):
        _DDSIPTesterBase._setup(self, options)
        if servers is not None:
            options['--pyro-required-scenariotreeservers'] = servers

    def test_scenarios_LP_1server(self):
        self._setup(self.options, servers=1)
        cmd = self._get_cmd()
        self._run_cmd(cmd)
        self._diff(os.path.join(baselinedir, self.baseline_basename+'_baseline'),
                   self.options['--output-directory'])
        self._cleanup()

@unittest.nottest
def create_test_classes(test_class_suffix,
                        baseline_basename,
                        model_location,
                        scenario_tree_location,
                        categories,
                        extra_options=None):
    assert test_class_suffix is not None
    assert baseline_basename is not None

    class _base(object):
        pass
    _base.baseline_basename = baseline_basename
    _base.model_location = model_location
    _base.scenario_tree_location = scenario_tree_location
    _base.extra_options = extra_options

    class_names = []

    @unittest.category(*categories)
    class TestConvertDDSIP_Serial(_base,
                                  _DDSIPTesterBase):
        pass
    class_names.append(TestConvertDDSIP_Serial.__name__ + "_"+test_class_suffix)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestConvertDDSIP_Serial, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestConvertDDSIP_Pyro(_base,
                                unittest.TestCase,
                                _DDSIPPyroTesterBase):
        def setUp(self):
            _DDSIPPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _DDSIPPyroTesterBase._setup(self, options, servers=servers)
    class_names.append(TestConvertDDSIP_Pyro.__name__ + "_"+test_class_suffix)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestConvertDDSIP_Pyro, unittest.TestCase), {})

    return tuple(globals()[name] for name in class_names)

#
# create the actual testing classes
#

farmer_examples_dir = join(pysp_examples_dir, "farmerWintegers")
farmer_model_dir = join(farmer_examples_dir, "smps_model")
farmer_data_dir = join(farmer_examples_dir, "scenariodata")
create_test_classes('farmer',
                    'farmer',
                    farmer_model_dir,
                    farmer_data_dir,
                    ('nightly',))

piecewise_model = join(thisdir, "piecewise_model.py")
piecewise_scenario_tree = join(thisdir, "piecewise_scenario_tree.py")
create_test_classes('piecewise',
                    'piecewise',
                    piecewise_model,
                    piecewise_scenario_tree,
                    ('nightly',))

piecewise_model = join(thisdir, "piecewise_model_alt.py")
create_test_classes('piecewise_alt',
                    'piecewise',
                    piecewise_model,
                    piecewise_scenario_tree,
                    ('nightly',))

if __name__ == "__main__":
    unittest.main()
