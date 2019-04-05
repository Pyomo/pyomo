#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyutilib.th as unittest
import pyomo.environ as aml
import os

try:
    import scipy.sparse as spa
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run NLP tests")

try:
    from pyomo.contrib.pynumero.algorithms.solvers.ip_solver import InteriorPointSolver
except ImportError:
    raise unittest.SkipTest("Pynumero needs ma27 or mumps linear solver to run InteriorPointSolver tests")

from pyomo.contrib.pynumero.interfaces.nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.tests import cute_models

from inspect import getmembers, isfunction


class TestInteriorPointSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ipopt = aml.SolverFactory('ipopt')
        cls.ipopt.options['nlp_scaling_method'] = 'none'
        cls.ipopt.options['linear_system_scaling'] = 'none'
        cls.functions_list = {o[0]:o[1] for o in getmembers(cute_models) if isfunction(o[1])}

    def test_model1(self):

        model = self.functions_list['create_model1']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model2(self):
        model = self.functions_list['create_model2']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model3(self):
        model = self.functions_list['create_model3']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model4(self):
        model = self.functions_list['create_model4']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model5(self):
        model = self.functions_list['create_model5']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model6(self):
        model = self.functions_list['create_model6']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model7(self):
        model = self.functions_list['create_model7']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model8(self):
        model = self.functions_list['create_model8']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model9(self):
        model = self.functions_list['create_model9']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model10(self):
        model = self.functions_list['create_model11']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()
        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model11(self):
        model = self.functions_list['create_model13']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model12(self):
        model = self.functions_list['create_model14']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model13(self):
        model = self.functions_list['create_model16']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model14(self):
        model = self.functions_list['create_model17']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model15(self):
        model = self.functions_list['create_model18']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model16(self):
        model = self.functions_list['create_model19']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model17(self):
        model = self.functions_list['create_model20']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False, refine_max_iter=0)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model18(self):
        model = self.functions_list['create_model22']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model19(self):
        model = self.functions_list['create_model23']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))

    def test_model20(self):
        model = self.functions_list['create_model24']()

        nlp1 = PyomoNLP(model)
        opt = InteriorPointSolver(nlp1)
        x, info = opt.solve(linear_solver='mumps', tee=False)

        self.ipopt.solve(model)
        nlp_ipopt = PyomoNLP(model)

        xx = nlp_ipopt.x_init()

        self.assertTrue(np.allclose(x, xx, atol=1e-4))
