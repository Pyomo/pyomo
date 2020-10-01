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

cyipopt_available = True
try:
    import ipopt
    from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
        CyIpoptSolver, CyIpoptNLP
    )
except ImportError:
    raise unittest.SkipTest("PyNumero needs CyIpopt installed to run CyIpopt tests")

example_dir = os.path.join(this_file_dir(), '..', 'examples')

class TestExamples(unittest.TestCase):
    def test_external_grey_box_react_example_maximize_cb_outputs(self):
        ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react-example', 'maximize_cb_outputs.py'))
        m = ex.maximize_cb_outputs()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.34381, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.outputs['cb']), 1072.4372, places=2)

    def test_external_grey_box_react_example_maximize_with_output(self):
        ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react-example', 'maximize_cb_ratio_residuals.py'))
        m = ex.maximize_cb_ratio_residuals_with_output()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2)
        self.assertAlmostEqual(pyo.value(m.reactor.outputs['cb_ratio']), 0.15190409266, places=3)

    def test_external_grey_box_react_example_maximize_with_additional_pyomo_variables(self):
        ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react-example', 'maximize_cb_ratio_residuals.py'))
        m = ex.maximize_cb_ratio_residuals_with_pyomo_variables()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2)
        self.assertAlmostEqual(pyo.value(m.cb_ratio), 0.15190409266, places=3)
