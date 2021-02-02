#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import os
import tempfile
from os.path import join, dirname, abspath
import difflib
import filecmp
import shutil

from pyutilib.misc import import_file
import pyutilib.th as unittest

from six import StringIO

thisdir = dirname(abspath(__file__))
baselinedir = os.path.join(thisdir, "smps_embedded_baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisdir)))), "examples", "pysp")

import pyomo.environ as pyo
from pyomo.pysp.embeddedsp import (EmbeddedSP,
                                   StochasticDataAnnotation,
                                   TableDistribution,
                                   UniformDistribution,
                                   StageCostAnnotation,
                                   VariableStageAnnotation)
from pyomo.pysp.convert.smps import convert_embedded

baa99_basemodel = None
piecewise_model_embedded = None
def setUpModule():
    global baa99_basemodel
    global piecewise_model_embedded
    if "baa99_basemodel" in sys.modules:
        del sys.modules["baa99_basemodel"]
    fname = os.path.join(pysp_examples_dir, "baa99", "baa99_basemodel.py")
    if os.path.exists(fname+"c"):
        os.remove(fname+"c")
    baa99_basemodel = import_file(fname)
    if "piecewise_model_embedded" in sys.modules:
        del sys.modules["piecewise_model_embedded"]
    fname = os.path.join(thisdir, "piecewise_model_embedded.py")
    if os.path.exists(fname+"c"):
        os.remove(fname+"c")
    piecewise_model_embedded = import_file(fname)

def tearDownModule():
    global baa99_basemodel
    global piecewise_model_embedded
    if "baa99_basemodel" in sys.modules:
        del sys.modules["baa99_basemodel"]
    baa99_basemodel = None
    if "piecewise_model_embedded" in sys.modules:
        del sys.modules["piecewise_model_embedded"]
    piecewise_model_embedded = None

@unittest.category('nightly')
class TestSMPSEmbeddedBad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(dir=thisdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir,
                      ignore_errors=True)
        cls.tmpdir = None

    def _get_base_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.y = pyo.Var()
        model.d1 = pyo.Param(mutable=True, initialize=0.0)
        model.d2 = pyo.Param(mutable=True, initialize=0.0)
        model.d3 = pyo.Param(mutable=True, initialize=0.0)
        model.cost = pyo.Expression([1,2])
        model.cost[1].expr = model.x
        model.cost[2].expr = model.d1*model.y
        model.o = pyo.Objective(expr= model.cost[1]+model.cost[2])
        model.c1 = pyo.Constraint(expr= model.x >= 0)
        model.c2 = pyo.Constraint(expr= model.y*model.d2 >= model.d3)
        model.varstage = VariableStageAnnotation()
        model.varstage.declare(model.x, 1)
        model.varstage.declare(model.y, 2)
        model.stagecost = StageCostAnnotation()
        model.stagecost.declare(model.cost[1], 1)
        model.stagecost.declare(model.cost[2], 2)
        model.stochdata = StochasticDataAnnotation()
        model.stochdata.declare(
            model.d1,
            distribution=TableDistribution([0.0,1.0]))
        model.stochdata.declare(
            model.d2,
            distribution=TableDistribution([0.0,1.0]))
        model.stochdata.declare(
            model.d3,
            distribution=TableDistribution([0.0,1.0]))

        return model

    def test_makes_directory(self):
        tmpdir = tempfile.mkdtemp(dir=thisdir)
        self.assertTrue(os.path.exists(tmpdir))
        shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertFalse(os.path.exists(tmpdir))
        sp = EmbeddedSP(self._get_base_model())
        convert_embedded(tmpdir, 'test', sp)
        self.assertTrue(os.path.exists(tmpdir))
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_too_many_stages(self):
        sp = EmbeddedSP(self._get_base_model())
        sp.time_stages = [1,2,3]
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(str(cm.exception),
                         ("SMPS conversion does not yet handle more "
                          "than 2 time-stages"))

    def test_stoch_variable_bounds(self):
        # Note there is no way to test this starting from a real model
        # because the Pyomo AML does not return what is assigned
        # to the bounds without converting it to a raw value (e.g.,
        # we can't detect if a mutable Param was used in a bound)
        class _Junk(object):
            pass
        sp = _Junk()
        sp.time_stages = [1,2]
        sp.has_stochastic_variable_bounds = True
        self.assertEqual(sp.has_stochastic_variable_bounds, True)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(str(cm.exception),
                         ("Problems with stochastic variables bounds "
                         "can not be converted into an embedded "
                         "SMPS representation"))

    def test_nonlinear_stoch_objective(self):
        model = self._get_base_model()
        model.cost[2].expr = model.y**2 + model.d1
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertTrue(str(cm.exception).startswith(
            "Cannot output embedded SP representation for component "
            "'o'. The embedded SMPS writer does not yet handle "
            "stochastic nonlinear expressions. Invalid expression: "))

    def test_stoch_data_too_many_uses_objective(self):
        model = self._get_base_model()
        model.cost[2].expr = model.d1*model.y + model.d1
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(
            str(cm.exception),
            ("Cannot output embedded SP representation for component "
             "'o'. The embedded SMPS writer does not yet handle the "
             "case where a stochastic data component appears in "
             "multiple expressions or locations within a single "
             "expression (e.g., multiple constraints, or multiple "
             "variable coefficients within a constraint). The "
             "parameter 'd1' appearing in component 'o' was "
             "previously encountered in another location in "
             "component 'o'."))

    def test_stoch_data_nontrivial_expression_objective1(self):
        model = self._get_base_model()
        model.cost[2].expr = -model.d1*model.y
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertTrue(str(cm.exception).startswith(
            "Cannot output embedded SP representation for component "
            "'o'. The embedded SMPS writer does not yet handle the "
            "case where a stochastic data component appears "
            "in an expression that defines a single variable's "
            "coefficient. The coefficient for variable 'y' must be "
            "exactly set to parameter 'd1' in the expression. Invalid "
            "expression: "))

    def test_stoch_data_nontrivial_expression_objective2(self):
        model = self._get_base_model()
        model.q = pyo.Param(mutable=True, initialize=0.0)
        model.stochdata.declare(
            model.q,
            distribution=TableDistribution([0.0,1.0]))
        model.cost[2].expr = (model.d1+model.q)*model.y
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(
            str(cm.exception),
            ("Cannot output embedded SP representation for component "
             "'o'. The embedded SMPS writer does not yet handle the "
             "case where multiple stochastic data components appear "
             "in an expression that defines a single variable's "
             "coefficient. The coefficient for variable 'y' involves "
             "stochastic parameters: ['d1', 'q']"))

    def test_bad_distribution_objective(self):
        model = self._get_base_model()
        del model.stochdata
        model.stochdata = StochasticDataAnnotation()
        model.stochdata.declare(
            model.d1,
            distribution=UniformDistribution(0.0,1.0))
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(
            str(cm.exception),
            ("Invalid distribution type 'UniformDistribution' for stochastic "
             "parameter 'd1'. The embedded SMPS writer currently "
             "only supports discrete table distributions of type "
             "pyomo.pysp.embeddedsp.TableDistribution."))

    def test_nonlinear_stoch_constraint(self):
        model = self._get_base_model()
        model.c2._body = model.d2*model.y**2
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertTrue(str(cm.exception).startswith(
            "Cannot output embedded SP representation for component "
            "'c2'. The embedded SMPS writer does not yet handle "
            "stochastic nonlinear expressions. Invalid expression: "))

    def test_stoch_constraint_body_constant(self):
        model = self._get_base_model()
        model.q = pyo.Param(mutable=True, initialize=0.0)
        model.stochdata.declare(
            model.q,
            distribution=TableDistribution([0.0,1.0]))
        model.c2._body = model.d2*model.y + model.q
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(
            str(cm.exception),
            ("Cannot output embedded SP representation for component "
             "'c2'. The embedded SMPS writer does not yet handle the "
             "case where a stochastic data appears in the body of a "
             "constraint expression that must be moved to the bounds. "
             "The constraint must be written so that the stochastic "
             "element 'q' is a simple bound or a simple variable "
             "coefficient."))

    def test_stoch_range_constraint(self):
        model = self._get_base_model()
        model.q = pyo.Param(mutable=True, initialize=0.0)
        model.stochdata.declare(
            model.q,
            distribution=TableDistribution([0.0,1.0]))
        model.c3 = pyo.Constraint(expr=pyo.inequality(model.q, model.y, 0))
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(
            str(cm.exception),
            ("Cannot output embedded SP representation for component "
             "'c3'. The embedded SMPS writer does not yet handle range "
             "constraints that have stochastic data."))

    def test_stoch_data_too_many_uses_constraint(self):
        model = self._get_base_model()
        model.c2._lower = model.d2
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(
            str(cm.exception),
            ("Cannot output embedded SP representation for component "
             "'c2'. The embedded SMPS writer does not yet handle the "
             "case where a stochastic data component appears in "
             "multiple expressions or locations within a single "
             "expression (e.g., multiple constraints, or multiple "
             "variable coefficients within a constraint). The "
             "parameter 'd2' appearing in component 'c2' was "
             "previously encountered in another location in "
             "component 'c2'."))

    def test_stoch_data_nontrivial_expression_constraint1(self):
        model = self._get_base_model()
        model.c2._body = -model.d2*model.y
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertTrue(str(cm.exception).startswith(
            "Cannot output embedded SP representation for component "
            "'c2'. The embedded SMPS writer does not yet handle the "
            "case where a stochastic data component appears "
            "in an expression that defines a single variable's "
            "coefficient. The coefficient for variable 'y' must be "
            "exactly set to parameter 'd2' in the expression. Invalid "
            "expression: "))

    def test_stoch_data_nontrivial_expression_constraint2(self):
        model = self._get_base_model()
        model.q = pyo.Param(mutable=True, initialize=0.0)
        model.stochdata.declare(
            model.q,
            distribution=TableDistribution([0.0,1.0]))
        model.c2._body = (model.d2+model.q)*model.y
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(
            str(cm.exception),
            ("Cannot output embedded SP representation for component "
             "'c2'. The embedded SMPS writer does not yet handle the "
             "case where multiple stochastic data components appear "
             "in an expression that defines a single variable's "
             "coefficient. The coefficient for variable 'y' involves "
             "stochastic parameters: ['d2', 'q']"))

    def test_bad_distribution_constraint(self):
        model = self._get_base_model()
        del model.stochdata
        model.stochdata = StochasticDataAnnotation()
        model.stochdata.declare(
            model.d2,
            distribution=UniformDistribution(0.0,1.0))
        sp = EmbeddedSP(model)
        with self.assertRaises(ValueError) as cm:
            convert_embedded(self.tmpdir, 'test', sp)
        self.assertEqual(
            str(cm.exception),
            ("Invalid distribution type 'UniformDistribution' for stochastic "
             "parameter 'd2'. The embedded SMPS writer currently "
             "only supports discrete table distributions of type "
             "pyomo.pysp.embeddedsp.TableDistribution."))

@unittest.category('nightly')
class TestSMPSEmbedded(unittest.TestCase):

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

    def _run(self, sp, basename, **kwds):
        class_name, test_name = self.id().split('.')[-2:]
        output_directory = join(thisdir, class_name+"."+test_name)
        shutil.rmtree(output_directory,
                      ignore_errors=True)
        os.makedirs(output_directory)
        convert_embedded(output_directory, basename, sp, **kwds)
        return output_directory

    def _get_baa99_sp(self):
        model = baa99_basemodel.model.clone()
        model.varstage = VariableStageAnnotation()
        model.varstage.declare(model.x1, 1)
        model.varstage.declare(model.x2, 1)

        model.stagecost = StageCostAnnotation()
        model.stagecost.declare(model.FirstStageCost, 1)
        model.stagecost.declare(model.SecondStageCost, 2)

        model.stochdata = StochasticDataAnnotation()
        model.stochdata.declare(
            model.d1_rhs,
            distribution=TableDistribution(model.d1_rhs_table))
        model.stochdata.declare(
            model.d2_rhs,
            distribution=TableDistribution(model.d2_rhs_table))

        return EmbeddedSP(model)

    def test_baa99_embedded_LP_symbolic_labels(self):
        baseline_directory = os.path.join(
            baselinedir,
            'baa99_embedded_LP_symbolic_names_baseline')
        output_directory = self._run(
            self._get_baa99_sp(),
            'baa99',
            core_format='lp',
            io_options={'symbolic_solver_labels':True})
        self._diff(baseline_directory,
                   output_directory)

    def test_baa99_embedded_MPS_symbolic_labels(self):
        baseline_directory = os.path.join(
            baselinedir,
            'baa99_embedded_MPS_symbolic_names_baseline')
        output_directory = self._run(
            self._get_baa99_sp(),
            'baa99',
            core_format='mps',
            io_options={'symbolic_solver_labels':True})
        self._diff(baseline_directory,
                   output_directory)

    def test_piecewise_embedded_LP_symbolic_labels(self):
        baseline_directory = os.path.join(
            baselinedir,
            'piecewise_embedded_LP_symbolic_names_baseline')
        output_directory = self._run(
            piecewise_model_embedded.create_embedded(),
            'piecewise',
            core_format='lp',
            io_options={'symbolic_solver_labels':True})
        self._diff(baseline_directory,
                   output_directory)

    def test_piecewise_embedded_MPS_symbolic_labels(self):
        baseline_directory = os.path.join(
            baselinedir,
            'piecewise_embedded_MPS_symbolic_names_baseline')
        output_directory = self._run(
            piecewise_model_embedded.create_embedded(),
            'piecewise',
            core_format='mps',
            io_options={'symbolic_solver_labels':True})
        self._diff(baseline_directory,
                   output_directory)

if __name__ == "__main__":
    unittest.main()
