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
import time
import subprocess

from pyutilib.pyro import using_pyro3, using_pyro4
import pyutilib.th as unittest

from pyomo.common.dependencies import dill, dill_available as has_dill
from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)
from pyomo.pysp.scenariotree.manager import \
    (ScenarioTreeManagerClientSerial,
     ScenarioTreeManagerClientPyro)
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.manager_solver import \
    (ScenarioTreeManagerSolverFactory,
     PySPFailedSolveStatus)

import pyomo.environ as pyo

from pyomo.common.dependencies import (
    networkx, networkx_available as has_networkx
)

thisfile = os.path.abspath(__file__)
thisdir = os.path.dirname(thisfile)

_run_verbose = True

_default_test_options = ScenarioTreeManagerSolverFactory.register_options()
_default_test_options.solver = 'glpk'
_default_test_options.solver_io = 'lp'
#_default_test_options.symbolic_solver_labels = True
#_default_test_options.keep_solver_files = True
#_default_test_options.disable_advanced_preprocessing = True

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
    for i, proc in enumerate(_taskworker_processes):
        _kill(proc)
        outname = os.path.join(thisdir,
                               "TestCapture_scenariotreeserver_" + \
                               str(i+1) + ".out")
        if os.path.exists(outname):
            try:
                os.remove(outname)
            except OSError:
                pass
    _taskworker_processes = []
    if os.path.exists(os.path.join(thisdir, "Pyro_NS_URI")):
        try:
            os.remove(os.path.join(thisdir, "Pyro_NS_URI"))
        except OSError:
            pass

solver = {}
solver['glpk','lp'] = False
def setUpModule():
    global solver
    import pyomo.environ
    from pyomo.solvers.tests.solvers import test_solver_cases
    for _solver, _io in test_solver_cases():
        if (_solver, _io) in solver and \
            test_solver_cases(_solver, _io).available:
            solver[_solver, _io] = True

def _setUpPyro():
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
    if len(_taskworker_processes) == 0:
        for i in range(3):
            outname = os.path.join(thisdir,
                                   "TestCapture_scenariotreeserver_" + \
                                   str(i+1) + ".out")
            with open(outname, "w") as f:
                _taskworker_processes.append(
                    subprocess.Popen(["scenariotreeserver", "--traceback"] + \
                                     ["--import-module="+thisfile] + \
                                     (["--verbose"] if _run_verbose else []) + \
                                     ["--pyro-host="+str(_pyomo_ns_host)] + \
                                     ["--pyro-port="+str(_pyomo_ns_port)],
                                     stdout=f,
                                     stderr=subprocess.STDOUT))
        time.sleep(2)
        [_poll(proc) for proc in _taskworker_processes]

class _SP_Feasible(object):

    @staticmethod
    def get_factory():
        tree = networkx.DiGraph()
        tree.add_node("r",
                      variables=["x"],
                      cost="t0_cost")
        for i in range(3):
            tree.add_node("s"+str(i),
                          variables=["Y","stale","fixed"],
                          cost="t1_cost")
            tree.add_edge("r", "s"+str(i), weight=1.0/3)

        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.Y = pyo.Var([1])
        model.stale = pyo.Var(initialize=0.0)
        model.fixed = pyo.Var(initialize=0.0)
        model.fixed.fix()
        model.p = pyo.Param(mutable=True)
        model.t0_cost = pyo.Expression(expr=model.x)
        model.t1_cost = pyo.Expression(expr=model.Y[1])
        model.o = pyo.Objective(expr=model.t0_cost + model.t1_cost)
        model.c = pyo.ConstraintList()
        model.c.add(model.x >= 1)
        model.c.add(model.Y[1] >= model.p)

        def _create_model(scenario_name, node_names):
            m = model.clone()
            if scenario_name == "s0":
                m.p.value = 0.0
            elif scenario_name == "s1":
                m.p.value = 1.0
            else:
                assert(scenario_name == "s2")
                m.p.value = 2.0
            return m

        return ScenarioTreeInstanceFactory(
            model=_create_model,
            scenario_tree=tree)

    @staticmethod
    def validate_solve(tester, sp, results, names=None):
        if names is None:
            names = [_s.name for _s in sp.scenario_tree.scenarios]

        tester.assertEqual(["s0","s1","s2"],
                           sorted(s.name for s in \
                                  sp.scenario_tree.scenarios))
        for scenario_name in names:
            scenario = sp.scenario_tree.get_scenario(scenario_name)
            tester.assertEqual(str(results.solver_status[scenario.name]),
                               "ok")
            tester.assertEqual(str(results.termination_condition[scenario.name]),
                               "optimal")
            tester.assertEqual(len(scenario._x['r']), 1)
            tester.assertAlmostEqual(scenario._x['r']['x'], 1.0)
            tester.assertAlmostEqual(scenario._x[scenario.name]['Y:#1'],
                                     float(scenario.name[1:]))
            tester.assertAlmostEqual(scenario._x[scenario.name]['fixed'],
                                     0.0)
            tester.assertAlmostEqual(scenario._x[scenario.name]['stale'],
                                     0.0)
            tester.assertAlmostEqual(scenario._objective,
                                     1.0 + float(scenario.name[1:]))
            tester.assertAlmostEqual(scenario._cost,
                                     1.0 + float(scenario.name[1:]))
            tester.assertAlmostEqual(scenario._objective,
                                     results.objective[scenario.name])
            tester.assertAlmostEqual(scenario._cost,
                                     results.cost[scenario.name])
            tester.assertAlmostEqual(scenario._stage_costs['Stage1'],
                                     1.0)
            tester.assertAlmostEqual(scenario._stage_costs['Stage2'],
                                     float(scenario.name[1:]))
            tester.assertEqual(scenario._stale['r'], set([]))
            tester.assertEqual(scenario._fixed['r'], set([]))
            tester.assertEqual(scenario._stale[scenario.name],
                               set(['stale']))
            tester.assertEqual(scenario._fixed[scenario.name],
                               set(['fixed']))

class _SP_Infeasible(object):

    @staticmethod
    def get_factory():
        tree = networkx.DiGraph()
        tree.add_node("r",
                      variables=["x"],
                      cost="t0_cost")
        for i in range(3):
            tree.add_node("s"+str(i),
                          variables=["Y","stale","fixed"],
                          cost="t1_cost")
            tree.add_edge("r", "s"+str(i), weight=1.0/3)

        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.Y = pyo.Var([1], bounds=(None, 1))
        model.stale = pyo.Var(initialize=0.0)
        model.fixed = pyo.Var(initialize=0.0)
        model.fixed.fix()
        model.p = pyo.Param(mutable=True)
        model.t0_cost = pyo.Expression(expr=model.x)
        model.t1_cost = pyo.Expression(expr=model.Y[1])
        model.o = pyo.Objective(expr=model.t0_cost + model.t1_cost)
        model.c = pyo.ConstraintList()
        model.c.add(model.x >= 1)
        model.c.add(model.Y[1] >= model.p)

        def _create_model(scenario_name, node_names):
            m = model.clone()
            if scenario_name == "s0":
                m.p.value = 0.0
            elif scenario_name == "s1":
                m.p.value = 1.0
            else:
                assert(scenario_name == "s2")
                m.p.value = 2.0
            return m

        return ScenarioTreeInstanceFactory(
            model=_create_model,
            scenario_tree=tree)

    @staticmethod
    def validate_solve(tester, sp, results, names=None):
        if names is None:
            names = [_s.name for _s in sp.scenario_tree.scenarios]

        tester.assertEqual(["s0","s1","s2"],
                           sorted(s.name for s in \
                                  sp.scenario_tree.scenarios))
        for scenario_name in names:
            scenario = sp.scenario_tree.get_scenario(scenario_name)
            if scenario.name == 's2':
                tester.assertTrue(
                    str(results.solver_status[scenario.name]) \
                    in ("warning","ok"))
                tester.assertEqual(
                    str(results.termination_condition[scenario.name]),
                    "infeasible")
                tester.assertEqual(scenario._objective,
                                   None)
                tester.assertEqual(scenario._cost,
                                   None)
                tester.assertEqual(scenario._stage_costs['Stage1'],
                                   None)
                tester.assertEqual(scenario._stage_costs['Stage2'],
                                   None)
                tester.assertEqual(scenario._stale['r'],
                                   set([]))
                tester.assertEqual(scenario._fixed['r'],
                                   set([]))
                tester.assertEqual(scenario._stale[scenario.name],
                                   set([]))
                tester.assertEqual(scenario._fixed[scenario.name],
                                   set([]))
            else:
                assert scenario.name in ('s0', 's1')
                tester.assertEqual(
                    str(results.solver_status[scenario.name]),
                    "ok")
                tester.assertEqual(
                    str(results.termination_condition[scenario.name]),
                    "optimal")
                tester.assertEqual(len(scenario._x['r']), 1)
                tester.assertAlmostEqual(scenario._x['r']['x'], 1.0)
                tester.assertAlmostEqual(scenario._x[scenario.name]['Y:#1'],
                                         float(scenario.name[1:]))
                tester.assertAlmostEqual(scenario._x[scenario.name]['fixed'],
                                         0.0)
                tester.assertAlmostEqual(scenario._x[scenario.name]['stale'],
                                         0.0)
                tester.assertAlmostEqual(scenario._objective,
                                         1.0 + float(scenario.name[1:]))
                tester.assertAlmostEqual(scenario._cost,
                                         1.0 + float(scenario.name[1:]))
                tester.assertAlmostEqual(scenario._objective,
                                         results.objective[scenario.name])
                tester.assertAlmostEqual(scenario._cost,
                                         results.cost[scenario.name])
                tester.assertAlmostEqual(scenario._stage_costs['Stage1'],
                                         1.0)
                tester.assertAlmostEqual(scenario._stage_costs['Stage2'],
                                         float(scenario.name[1:]))
                tester.assertEqual(scenario._stale['r'],
                                   set([]))
                tester.assertEqual(scenario._fixed['r'],
                                   set([]))
                tester.assertEqual(scenario._stale[scenario.name],
                                   set(['stale']))
                tester.assertEqual(scenario._fixed[scenario.name],
                                   set(['fixed']))

class _SP_Bundles_Feasible(object):

    @staticmethod
    def get_factory():
        tree = networkx.DiGraph()
        tree.add_node("r",
                      variables=["x"],
                      cost="t0_cost")
        for i in range(3):
            tree.add_node("s"+str(i),
                          variables=["Y","stale","fixed"],
                          cost="t1_cost",
                          bundle="b"+str(i))
            tree.add_edge("r", "s"+str(i), weight=1.0/3)

        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.Y = pyo.Var([1])
        model.stale = pyo.Var(initialize=0.0)
        model.fixed = pyo.Var(initialize=0.0)
        model.fixed.fix()
        model.p = pyo.Param(mutable=True)
        model.t0_cost = pyo.Expression(expr=model.x)
        model.t1_cost = pyo.Expression(expr=model.Y[1])
        model.o = pyo.Objective(expr=model.t0_cost + model.t1_cost)
        model.c = pyo.ConstraintList()
        model.c.add(model.x >= 1)
        model.c.add(model.Y[1] >= model.p)

        def _create_model(scenario_name, node_names):
            m = model.clone()
            if scenario_name == "s0":
                m.p.value = 0.0
            elif scenario_name == "s1":
                m.p.value = 1.0
            else:
                assert(scenario_name == "s2")
                m.p.value = 2.0
            return m

        return ScenarioTreeInstanceFactory(
            model=_create_model,
            scenario_tree=tree)

    @staticmethod
    def validate_solve(tester, sp, results, names=None):
        if names is None:
            names = [_s.name for _s in sp.scenario_tree.bundles]

        tester.assertEqual(["s0","s1","s2"],
                           sorted(s.name for s in \
                                  sp.scenario_tree.scenarios))
        tester.assertEqual(["b0","b1","b2"],
                           sorted(b.name for b in \
                                  sp.scenario_tree.bundles))
        for bundle_name in names:
            bundle = sp.scenario_tree.get_bundle(bundle_name)
            assert len(bundle.scenario_names) == 1
            scenario_name = bundle.scenario_names[0]
            assert len(scenario_name) == len(bundle_name)
            assert scenario_name[0] == 's'
            assert bundle_name[0] == 'b'
            scenario = sp.scenario_tree.get_scenario(scenario_name)
            tester.assertEqual(str(results.solver_status[bundle.name]),
                               "ok")
            tester.assertEqual(str(results.termination_condition[bundle.name]),
                               "optimal")
            tester.assertEqual(len(scenario._x['r']), 1)
            tester.assertAlmostEqual(scenario._x['r']['x'], 1.0)
            tester.assertAlmostEqual(scenario._x[scenario.name]['Y:#1'],
                                     float(scenario.name[1:]))
            tester.assertAlmostEqual(scenario._x[scenario.name]['fixed'],
                                     0.0)
            tester.assertAlmostEqual(scenario._x[scenario.name]['stale'],
                                     0.0)
            tester.assertAlmostEqual(scenario._objective,
                                     1.0 + float(scenario.name[1:]))
            tester.assertAlmostEqual(scenario._cost,
                                     1.0 + float(scenario.name[1:]))
            tester.assertAlmostEqual(scenario._objective,
                                     results.objective[bundle.name])
            tester.assertAlmostEqual(scenario._cost,
                                     results.cost[bundle.name])
            tester.assertAlmostEqual(scenario._stage_costs['Stage1'],
                                     1.0)
            tester.assertAlmostEqual(scenario._stage_costs['Stage2'],
                                     float(scenario.name[1:]))
            tester.assertEqual(scenario._stale['r'], set([]))
            tester.assertEqual(scenario._fixed['r'], set([]))
            tester.assertEqual(scenario._stale[scenario.name],
                               set(['stale']))
            tester.assertEqual(scenario._fixed[scenario.name],
                               set(['fixed']))

class _SP_Bundles_Infeasible(object):

    @staticmethod
    def get_factory():
        tree = networkx.DiGraph()
        tree.add_node("r",
                      variables=["x"],
                      cost="t0_cost")
        for i in range(3):
            tree.add_node("s"+str(i),
                          variables=["Y","stale","fixed"],
                          cost="t1_cost",
                          bundle="b"+str(i))
            tree.add_edge("r", "s"+str(i), weight=1.0/3)

        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.Y = pyo.Var([1], bounds=(None, 1))
        model.stale = pyo.Var(initialize=0.0)
        model.fixed = pyo.Var(initialize=0.0)
        model.fixed.fix()
        model.p = pyo.Param(mutable=True)
        model.t0_cost = pyo.Expression(expr=model.x)
        model.t1_cost = pyo.Expression(expr=model.Y[1])
        model.o = pyo.Objective(expr=model.t0_cost + model.t1_cost)
        model.c = pyo.ConstraintList()
        model.c.add(model.x >= 1)
        model.c.add(model.Y[1] >= model.p)

        def _create_model(scenario_name, node_names):
            m = model.clone()
            if scenario_name == "s0":
                m.p.value = 0.0
            elif scenario_name == "s1":
                m.p.value = 1.0
            else:
                assert(scenario_name == "s2")
                m.p.value = 2.0
            return m

        return ScenarioTreeInstanceFactory(
            model=_create_model,
            scenario_tree=tree)

    @staticmethod
    def validate_solve(tester, sp, results, names=None):
        if names is None:
            names = [_s.name for _s in sp.scenario_tree.bundles]

        tester.assertEqual(["s0","s1","s2"],
                           sorted(s.name for s in \
                                  sp.scenario_tree.scenarios))
        tester.assertEqual(["b0","b1","b2"],
                           sorted(b.name for b in \
                                  sp.scenario_tree.bundles))
        for bundle_name in names:
            bundle = sp.scenario_tree.get_bundle(bundle_name)
            assert len(bundle.scenario_names) == 1
            scenario_name = bundle.scenario_names[0]
            assert len(scenario_name) == len(bundle_name)
            assert scenario_name[0] == 's'
            assert bundle_name[0] == 'b'
            scenario = sp.scenario_tree.get_scenario(scenario_name)
            if bundle.name == 'b2':
                tester.assertTrue(
                    str(results.solver_status[bundle.name]) \
                    in ("warning","ok"))
                tester.assertEqual(
                    str(results.termination_condition[bundle.name]),
                    "infeasible")
                tester.assertEqual(scenario._objective,
                                   None)
                tester.assertEqual(scenario._cost,
                                   None)
                tester.assertEqual(scenario._stage_costs['Stage1'],
                                   None)
                tester.assertEqual(scenario._stage_costs['Stage2'],
                                   None)
                tester.assertEqual(scenario._stale['r'],
                                   set([]))
                tester.assertEqual(scenario._fixed['r'],
                                   set([]))
                tester.assertEqual(scenario._stale[scenario.name],
                                   set([]))
                tester.assertEqual(scenario._fixed[scenario.name],
                                   set([]))
            else:
                assert bundle.name in ('b0', 'b1')
                tester.assertEqual(
                    str(results.solver_status[bundle.name]),
                    "ok")
                tester.assertEqual(
                    str(results.termination_condition[bundle.name]),
                    "optimal")
                tester.assertEqual(len(scenario._x['r']), 1)
                tester.assertAlmostEqual(scenario._x['r']['x'], 1.0)
                tester.assertAlmostEqual(scenario._x[scenario.name]['Y:#1'],
                                         float(scenario.name[1:]))
                tester.assertAlmostEqual(scenario._x[scenario.name]['fixed'],
                                         0.0)
                tester.assertAlmostEqual(scenario._x[scenario.name]['stale'],
                                         0.0)
                tester.assertAlmostEqual(scenario._objective,
                                         1.0 + float(scenario.name[1:]))
                tester.assertAlmostEqual(scenario._cost,
                                         1.0 + float(scenario.name[1:]))
                tester.assertAlmostEqual(scenario._objective,
                                         results.objective[bundle.name])
                tester.assertAlmostEqual(scenario._cost,
                                         results.cost[bundle.name])
                tester.assertAlmostEqual(scenario._stage_costs['Stage1'],
                                         1.0)
                tester.assertAlmostEqual(scenario._stage_costs['Stage2'],
                                         float(scenario.name[1:]))
                tester.assertEqual(scenario._stale['r'],
                                   set([]))
                tester.assertEqual(scenario._fixed['r'],
                                   set([]))
                tester.assertEqual(scenario._stale[scenario.name],
                                   set(['stale']))
                tester.assertEqual(scenario._fixed[scenario.name],
                                   set(['fixed']))

class _ScenarioTreeManagerSolverTesterBase(object):

    def test_solve_scenarios_optimal(self):
        problem = _SP_Feasible
        for names in [None,
                      ['s0'],
                      ['s1'],
                      ['s2']]:
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    results = manager.solve_scenarios(check_status=False,
                                                      scenarios=names)
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    results = manager.solve_subproblems(check_status=False,
                                                        subproblems=names)
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    results = manager.solve_scenarios(check_status=True,
                                                      scenarios=names)
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    job = manager.solve_scenarios(async_call=True,
                                                  check_status=True,
                                                  scenarios=names)
                    results = job.complete()
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    job = manager.solve_scenarios(async_call=True,
                                                  check_status=False,
                                                  scenarios=names)
                    results = job.complete()
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    with self.assertRaises(RuntimeError):
                        manager.solve_bundles()

    def test_solve_scenarios_infeasible(self):
        problem = _SP_Infeasible
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                results = manager.solve_scenarios(check_status=False)
            problem.validate_solve(self, sp, results)
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                with self.assertRaises(PySPFailedSolveStatus):
                    manager.solve_scenarios(check_status=True)
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                job = manager.solve_scenarios(async_call=True,
                                              check_status=True)
                with self.assertRaises(PySPFailedSolveStatus):
                    job.complete()
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                job = manager.solve_scenarios(async_call=True,
                                              check_status=False)
                results = job.complete()
            problem.validate_solve(self, sp, results)
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                with self.assertRaises(RuntimeError):
                    manager.solve_bundles()

        for names in [None,
                      ['s0'],
                      ['s1'],
                      ['s2']]:
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    results = manager.solve_scenarios(check_status=False,
                                                      scenarios=names)
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    job = manager.solve_scenarios(async_call=True,
                                                  check_status=False,
                                                  scenarios=names)
                    results = job.complete()
                problem.validate_solve(self, sp, results, names=names)

    def test_solve_bundles_optimal(self):
        problem = _SP_Bundles_Feasible
        for names in [None,
                      ['b0'],
                      ['b1'],
                      ['b2']]:
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    results = manager.solve_bundles(check_status=False,
                                                    bundles=names)
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    results = manager.solve_subproblems(check_status=False,
                                                        subproblems=names)
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    results = manager.solve_bundles(check_status=True,
                                                    bundles=names)
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    job = manager.solve_bundles(async_call=True,
                                                check_status=True,
                                                bundles=names)
                    results = job.complete()
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    job = manager.solve_bundles(async_call=True,
                                                check_status=False,
                                                bundles=names)
                    results = job.complete()
                problem.validate_solve(self, sp, results, names=names)

    def test_solve_bundles_infeasible(self):
        problem = _SP_Bundles_Infeasible
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                results = manager.solve_bundles(check_status=False)
            problem.validate_solve(self, sp, results)
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                with self.assertRaises(PySPFailedSolveStatus):
                    manager.solve_bundles(check_status=True)
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                job = manager.solve_bundles(async_call=True,
                                            check_status=True)
                with self.assertRaises(PySPFailedSolveStatus):
                    job.complete()
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                job = manager.solve_bundles(async_call=True,
                                            check_status=False)
                results = job.complete()
            problem.validate_solve(self, sp, results)
        with self._init(problem.get_factory()) as sp:
            with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                with self.assertRaises(RuntimeError):
                    manager.solve_bundles()

        for names in [None,
                      ['b0'],
                      ['b1'],
                      ['b2']]:
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    results = manager.solve_bundles(check_status=False,
                                                    bundles=names)
                problem.validate_solve(self, sp, results, names=names)
            with self._init(problem.get_factory()) as sp:
                with ScenarioTreeManagerSolverFactory(sp, _default_test_options) as manager:
                    job = manager.solve_bundles(async_call=True,
                                                check_status=False,
                                                bundles=names)
                    results = job.complete()
                problem.validate_solve(self, sp, results, names=names)
#
# create the actual testing classes
#

@unittest.skipIf(not has_networkx, "Networkx is not available")
@unittest.skipIf(not has_dill, "Dill is not available")
class TestScenarioTreeManagerSolverSerial(
        unittest.TestCase,
        _ScenarioTreeManagerSolverTesterBase):

    @classmethod
    def setUpClass(cls):
        if not solver['glpk','lp']:
            raise unittest.SkipTest(
                "The glpk solver is not available")

    @unittest.nottest
    def _init(self, factory):
        options = ScenarioTreeManagerClientSerial.register_options()
        sp = ScenarioTreeManagerClientSerial(
            options,
            factory=factory)
        sp.initialize()
        return sp

@unittest.skipIf(not has_networkx, "Networkx is not available")
@unittest.skipIf(not has_dill, "Dill is not available")
@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('parallel')
class TestScenarioTreeManagerSolverPyro(
        unittest.TestCase,
        _ScenarioTreeManagerSolverTesterBase):

    @classmethod
    def setUpClass(cls):
        if not solver['glpk','lp']:
            raise unittest.SkipTest(
                "The glpk solver is not available")

    @unittest.nottest
    def _init(self, factory):
        _setUpPyro()
        [_poll(proc) for proc in _taskworker_processes]
        options = ScenarioTreeManagerClientPyro.register_options()
        options.pyro_port = _pyomo_ns_port
        options.pyro_required_scenariotreeservers = 3
        options.pyro_handshake_at_startup = True
        sp = ScenarioTreeManagerClientPyro(
            options,
            factory=factory)
        sp.initialize()
        return sp

if __name__ == "__main__":
    unittest.main()
