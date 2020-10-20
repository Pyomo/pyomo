import os.path
from pyomo.common.fileutils import this_file_dir
import pyutilib.th as unittest
from pyutilib.misc import import_file
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_sparse as spa, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run CyIpopt tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpopt tests")

import pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver as cyipopt_solver
if not cyipopt_solver.ipopt_available:
    raise unittest.SkipTest("PyNumero needs CyIpopt installed to run CyIpopt tests")
import cyipopt as cyipopt_core

example_dir = os.path.join(this_file_dir(), '..', 'examples')

class TestPyomoCyIpoptSolver(unittest.TestCase):
    def test_status_maps(self):
        self.assertEqual(len(cyipopt_core.STATUS_MESSAGES),
                         len(cyipopt_solver._cyipopt_status_enum))
        self.assertEqual(len(cyipopt_core.STATUS_MESSAGES),
                         len(cyipopt_solver._ipopt_term_cond))
        for msg in cyipopt_core.STATUS_MESSAGES.values():
            self.assertIn(msg, cyipopt_solver._cyipopt_status_enum)
        for status in cyipopt_solver._cyipopt_status_enum.values():
            self.assertIn(status, cyipopt_solver._ipopt_term_cond)


class TestExamples(unittest.TestCase):
    def test_external_grey_box_react_example_maximize_cb_outputs(self):
        ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react-example', 'maximize_cb_outputs.py'))
        m = ex.maximize_cb_outputs()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.34381, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.outputs['cb']), 1072.4372, places=2)

    def test_external_grey_box_react_example_maximize_cb_outputs_scaling(self):
        ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react-example', 'maximize_cb_ratio_residuals.py'))
        aoptions={'nlp_scaling_method': 'user-scaling',
                 'output_file': '_cyipopt-external-greybox-react-scaling.log',
                 'file_print_level':10}
        m = ex.maximize_cb_ratio_residuals_with_output_scaling(additional_options=aoptions)
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2)
        self.assertAlmostEqual(pyo.value(m.reactor.outputs['cb_ratio']), 0.15190409266, places=3)

        with open('_cyipopt-external-greybox-react-scaling.log', 'r') as fd:
            solver_trace = fd.read()
        os.remove('_cyipopt-external-greybox-react-scaling.log')

        self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
        self.assertIn('output_file = _cyipopt-external-greybox-react-scaling.log', solver_trace)
        self.assertIn('objective scaling factor = 1', solver_trace)
        self.assertIn('x scaling provided', solver_trace)
        self.assertIn('c scaling provided', solver_trace)
        self.assertIn('d scaling provided', solver_trace)
        self.assertIn('DenseVector "x scaling vector" with 7 elements:', solver_trace)
        self.assertIn('x scaling vector[    1]= 1.2000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    2]= 1.7000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    3]= 1.1000000000000001e+00', solver_trace)
        self.assertIn('x scaling vector[    4]= 1.3000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    5]= 1.3999999999999999e+00', solver_trace)
        self.assertIn('x scaling vector[    6]= 1.5000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    7]= 1.6000000000000001e+00', solver_trace)
        self.assertIn('DenseVector "c scaling vector" with 6 elements:', solver_trace)
        self.assertIn('c scaling vector[    1]= 4.2000000000000000e+01', solver_trace)
        self.assertIn('c scaling vector[    2]= 1.0000000000000001e-01', solver_trace)
        self.assertIn('c scaling vector[    3]= 2.0000000000000001e-01', solver_trace)
        self.assertIn('c scaling vector[    4]= 2.9999999999999999e-01', solver_trace)
        self.assertIn('c scaling vector[    5]= 4.0000000000000002e-01', solver_trace)
        self.assertIn('c scaling vector[    6]= 1.0000000000000000e+01', solver_trace)

    def test_external_grey_box_react_example_maximize_with_output(self):
        ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react-example', 'maximize_cb_ratio_residuals.py'))
        m = ex.maximize_cb_ratio_residuals_with_output()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2)
        self.assertAlmostEqual(pyo.value(m.reactor.outputs['cb_ratio']), 0.15190409266, places=3)

    def test_external_grey_box_react_example_maximize_with_obj(self):
        ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react-example', 'maximize_cb_ratio_residuals.py'))
        m = ex.maximize_cb_ratio_residuals_with_obj()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2)
        self.assertAlmostEqual(pyo.value(m.obj), 0.15190409266, places=3)

    def test_external_grey_box_react_example_maximize_with_additional_pyomo_variables(self):
        ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react-example', 'maximize_cb_ratio_residuals.py'))
        m = ex.maximize_cb_ratio_residuals_with_pyomo_variables()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2)
        self.assertAlmostEqual(pyo.value(m.cb_ratio), 0.15190409266, places=3)
