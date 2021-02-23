from pyomo.contrib.appsi.examples import getting_started
import pyutilib.th as unittest
import pyomo.environ as pe


class TestExamples(unittest.TestCase):
    def test_getting_started(self):
        opt = pe.SolverFactory('appsi_cplex')
        if not opt.available():
            raise unittest.SkipTest('cplex is not available')
        getting_started.main(plot=False, n_points=10)
