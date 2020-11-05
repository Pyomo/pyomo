#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
from os.path import join, dirname, abspath
import time
import filecmp
import shutil
import subprocess
import sys
import pyutilib.th as unittest
from pyutilib.pyro import using_pyro3, using_pyro4
from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)

thisdir = dirname(abspath(__file__))
baselinedir = os.path.join(thisdir, "schuripopt_baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisdir)))), "examples", "pysp")

_run_verbose = True
_diff_tolerance = 1e-6

class _SchurIpoptTesterBase(object):

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
        cmd = [sys.executable,'-m','pyomo.pysp.convert.schuripopt']
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
            self.assertFileEqualsBaseline(
                tofile, fromfile,
                tolerance=_diff_tolerance,
                delete=False)
            """
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
            """
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

class _SchurIpoptPyroTesterBase(_SchurIpoptTesterBase):

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
        _SchurIpoptTesterBase._setup(self, options)
        if servers is not None:
            options['--pyro-required-scenariotreeservers'] = servers

    def test_scenarios_1server(self):
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
    class TestConvertSchurIpopt_Serial(_base,
                                       _SchurIpoptTesterBase):
        pass
    class_names.append(TestConvertSchurIpopt_Serial.__name__ + "_"+test_class_suffix)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestConvertSchurIpopt_Serial, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestConvertSchurIpopt_Pyro(_base,
                                     unittest.TestCase,
                                     _SchurIpoptPyroTesterBase):
        def setUp(self):
            _SchurIpoptPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _SchurIpoptPyroTesterBase._setup(self, options, servers=servers)
    class_names.append(TestConvertSchurIpopt_Pyro.__name__ + "_"+test_class_suffix)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestConvertSchurIpopt_Pyro, unittest.TestCase), {})

    return tuple(globals()[name] for name in class_names)

#
# create the actual testing classes
#

farmer_examples_dir = join(pysp_examples_dir, "farmer")
farmer_model_dir = join(farmer_examples_dir, "expr_models")
farmer_data_dir = join(farmer_examples_dir, "scenariodata")
create_test_classes('farmer',
                    'farmer',
                    farmer_model_dir,
                    farmer_data_dir,
                    ('nightly','expensive'))

piecewise_model = join(thisdir, "piecewise_model.py")
piecewise_scenario_tree = join(thisdir, "piecewise_scenario_tree.py")
create_test_classes('piecewise',
                    'piecewise',
                    piecewise_model,
                    piecewise_scenario_tree,
                    ('nightly','expensive'))

piecewise_scenario_tree_bundles = join(thisdir, "piecewise_scenario_tree_bundles.py")
create_test_classes('piecewise_bundles',
                    'piecewise_bundles',
                    piecewise_model,
                    piecewise_scenario_tree_bundles,
                    ('nightly','expensive'))

create_test_classes('piecewise_ignore_bundles',
                    'piecewise',
                    piecewise_model,
                    piecewise_scenario_tree_bundles,
                    ('nightly','expensive'),
                    extra_options={'--ignore-bundles': None})

if __name__ == "__main__":
    unittest.main()
