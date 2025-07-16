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

import os.path
from io import StringIO
import logging

from pyomo.common.fileutils import this_file_dir, import_file
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
from pyomo.opt import TerminationCondition

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)
from pyomo.common.dependencies.scipy import sparse as spa

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run CyIpopt tests")

pandas, pandas_available = attempt_import(
    'pandas',
    'One of the tests below requires a recent version of pandas for'
    ' comparing with a tolerance.',
    minimum_version='1.1.0',
    defer_import=False,
)

from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero needs the ASL extension to run CyIpopt tests")

import pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver as cyipopt_solver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import cyipopt_available

if not cyipopt_available:
    raise unittest.SkipTest("PyNumero needs CyIpopt installed to run CyIpopt tests")
import cyipopt as cyipopt_core


example_dir = os.path.join(this_file_dir(), '..')


class TestPyomoCyIpoptSolver(unittest.TestCase):
    def test_status_maps(self):
        # verify that all status messages from cyipopy can be cleanly
        # mapped back to a Pyomo TerminationCondition
        for msg in cyipopt_core.STATUS_MESSAGES.values():
            self.assertIn(msg, cyipopt_solver._cyipopt_status_enum)
        for status in cyipopt_solver._cyipopt_status_enum.values():
            self.assertIn(status, cyipopt_solver._ipopt_term_cond)


class TestExamples(unittest.TestCase):
    def test_external_grey_box_react_example_maximize_cb_outputs(self):
        ex = import_file(
            os.path.join(
                example_dir,
                'external_grey_box',
                'react_example',
                'maximize_cb_outputs.py',
            )
        )
        m = ex.maximize_cb_outputs()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.34381, places=3)
        self.assertAlmostEqual(pyo.value(m.reactor.outputs['cb']), 1072.4372, places=2)

    def test_external_grey_box_react_example_maximize_cb_outputs_scaling(self):
        ex = import_file(
            os.path.join(
                example_dir,
                'external_grey_box',
                'react_example',
                'maximize_cb_ratio_residuals.py',
            )
        )

        with TempfileManager.new_context() as temp:
            logfile = temp.create_tempfile(
                '_cyipopt-external-greybox-react-scaling.log'
            )
            aoptions = {
                'nlp_scaling_method': 'user-scaling',
                'output_file': logfile,
                'file_print_level': 10,
            }
            m = ex.maximize_cb_ratio_residuals_with_output_scaling(
                additional_options=aoptions
            )
            self.assertAlmostEqual(
                pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3
            )
            self.assertAlmostEqual(
                pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2
            )
            self.assertAlmostEqual(
                pyo.value(m.reactor.outputs['cb_ratio']), 0.15190409266, places=3
            )

            with open(logfile, 'r') as fd:
                solver_trace = fd.read()

        self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
        self.assertIn(f'output_file = {logfile}', solver_trace)
        self.assertIn('objective scaling factor = 1', solver_trace)
        self.assertIn('x scaling provided', solver_trace)
        self.assertIn('c scaling provided', solver_trace)
        self.assertIn('d scaling provided', solver_trace)
        self.assertIn('DenseVector "x scaling vector" with 7 elements:', solver_trace)
        self.assertIn('x scaling vector[    2]= 1.2000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    7]= 1.7000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    6]= 1.1000000000000001e+00', solver_trace)
        self.assertIn('x scaling vector[    1]= 1.3000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    3]= 1.3999999999999999e+00', solver_trace)
        self.assertIn('x scaling vector[    4]= 1.5000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    5]= 1.6000000000000001e+00', solver_trace)
        self.assertIn('DenseVector "c scaling vector" with 6 elements:', solver_trace)
        self.assertIn('c scaling vector[    1]= 4.2000000000000000e+01', solver_trace)
        self.assertIn('c scaling vector[    2]= 1.0000000000000001e-01', solver_trace)
        self.assertIn('c scaling vector[    3]= 2.0000000000000001e-01', solver_trace)
        self.assertIn('c scaling vector[    4]= 2.9999999999999999e-01', solver_trace)
        self.assertIn('c scaling vector[    5]= 4.0000000000000002e-01', solver_trace)
        self.assertIn('c scaling vector[    6]= 1.0000000000000000e+01', solver_trace)

    def test_external_grey_box_react_example_maximize_with_output(self):
        ex = import_file(
            os.path.join(
                example_dir,
                'external_grey_box',
                'react_example',
                'maximize_cb_ratio_residuals.py',
            )
        )
        m = ex.maximize_cb_ratio_residuals_with_output()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(
            pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2
        )
        self.assertAlmostEqual(
            pyo.value(m.reactor.outputs['cb_ratio']), 0.15190409266, places=3
        )

    def test_external_grey_box_react_example_maximize_with_hessian_with_output(self):
        ex = import_file(
            os.path.join(
                example_dir,
                'external_grey_box',
                'react_example',
                'maximize_cb_ratio_residuals.py',
            )
        )
        m = ex.maximize_cb_ratio_residuals_with_hessian_with_output()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(
            pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2
        )
        self.assertAlmostEqual(
            pyo.value(m.reactor.outputs['cb_ratio']), 0.15190409266, places=3
        )

    def test_external_grey_box_react_example_maximize_with_hessian_with_output_pyomo(
        self,
    ):
        ex = import_file(
            os.path.join(
                example_dir,
                'external_grey_box',
                'react_example',
                'maximize_cb_ratio_residuals.py',
            )
        )
        m = ex.maximize_cb_ratio_residuals_with_hessian_with_output_pyomo()
        self.assertAlmostEqual(pyo.value(m.sv), 1.26541996, places=3)
        self.assertAlmostEqual(pyo.value(m.cb), 1071.7410089, places=2)
        self.assertAlmostEqual(pyo.value(m.cb_ratio), 0.15190409266, places=3)

    def test_pyomo_react_example_maximize_with_obj(self):
        ex = import_file(
            os.path.join(
                example_dir,
                'external_grey_box',
                'react_example',
                'maximize_cb_ratio_residuals.py',
            )
        )
        m = ex.maximize_cb_ratio_residuals_with_obj()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(
            pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2
        )
        self.assertAlmostEqual(pyo.value(m.obj), 0.15190409266, places=3)

    def test_external_grey_box_react_example_maximize_with_additional_pyomo_variables(
        self,
    ):
        ex = import_file(
            os.path.join(
                example_dir,
                'external_grey_box',
                'react_example',
                'maximize_cb_ratio_residuals.py',
            )
        )
        m = ex.maximize_cb_ratio_residuals_with_pyomo_variables()
        self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
        self.assertAlmostEqual(
            pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2
        )
        self.assertAlmostEqual(pyo.value(m.cb_ratio), 0.15190409266, places=3)

    @unittest.skipIf(not pandas_available, "Test uses pandas for data")
    def test_parameter_estimation(self):
        data_fname = os.path.join(
            example_dir, 'external_grey_box', 'param_est', 'smalldata.csv'
        )
        baseline = pandas.read_csv(data_fname)

        # test the data generator
        ex = import_file(
            os.path.join(
                example_dir, 'external_grey_box', 'param_est', 'generate_data.py'
            )
        )
        df1 = ex.generate_data(5, 200, 5, 42)
        df2 = ex.generate_data_external(5, 200, 5, 42)
        pandas.testing.assert_frame_equal(df1, baseline, atol=1e-3)
        pandas.testing.assert_frame_equal(df2, baseline, atol=1e-3)

        # test the estimation
        ex = import_file(
            os.path.join(
                example_dir, 'external_grey_box', 'param_est', 'perform_estimation.py'
            )
        )

        m = ex.perform_estimation_external(data_fname, solver_trace=False)
        self.assertAlmostEqual(pyo.value(m.UA), 204.43761, places=3)

        m = ex.perform_estimation_pyomo_only(data_fname, solver_trace=False)
        self.assertAlmostEqual(pyo.value(m.UA), 204.43761, places=3)

    def test_cyipopt_callbacks(self):
        ex = import_file(os.path.join(example_dir, 'callback', 'cyipopt_callback.py'))

        output = StringIO()
        with LoggingIntercept(output, 'pyomo', logging.INFO):
            ex.main()

        self.assertIn("Residuals for iteration 2", output.getvalue().strip())

    @unittest.skipIf(not pandas_available, "pandas needed to run this example")
    def test_cyipopt_functor(self):
        ex = import_file(
            os.path.join(example_dir, 'callback', 'cyipopt_functor_callback.py')
        )
        df = ex.main()
        self.assertEqual(df.shape, (7, 5))
        # check one of the residuals
        s = df['ca_bal']
        self.assertAlmostEqual(s.iloc[6], 0, places=3)

    @unittest.skipIf(
        cyipopt_solver.PyomoCyIpoptSolver().version() == (1, 4, 0),
        "Terminating Ipopt through a user callback is broken in CyIpopt 1.4.0 "
        "(see mechmotum/cyipopt#249)",
    )
    def test_cyipopt_callback_halt(self):
        ex = import_file(
            os.path.join(example_dir, 'callback', 'cyipopt_callback_halt.py')
        )
        status = ex.main()
        self.assertEqual(
            status.solver.termination_condition, TerminationCondition.userInterrupt
        )
