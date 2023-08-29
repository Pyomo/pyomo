from pyomo.contrib.appsi.examples import getting_started
from pyomo.common import unittest
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib import appsi

numpy, numpy_available = attempt_import('numpy')


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
@unittest.skipUnless(numpy_available, 'numpy is not available')
class TestExamples(unittest.TestCase):
    def test_getting_started(self):
        opt = appsi.solvers.Ipopt()
        if not opt.available():
            raise unittest.SkipTest('ipopt is not available')
        getting_started.main(plot=False, n_points=10)
