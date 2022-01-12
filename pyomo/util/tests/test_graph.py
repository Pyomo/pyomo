import pyomo.environ as pe
from pyomo.common import unittest
from pyomo.common.dependencies import attempt_import
nx, nx_available = attempt_import('networkx')
plotly, plotly_available = attempt_import('plotly')


@unittest.skipUnless(nx_available, 'plot_pyomo_model requires networkx')
@unittest.skipUnless(plotly_available, 'plot_pyomo_model requires plotly')
class TestPlotPyomoModel(unittest.TestCase):
    def test_plot_pyomo_model(self):
        """
        Unfortunately, this test only ensures the code runs without errors.
        It does not test for correctness.
        """
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.y**2 + m.z**2)
        m.c1 = pe.Constraint(expr=m.y == 2*m.x + 1)
        m.c2 = pe.Constraint(expr=m.z >= m.x)
        from pyomo.util.graph import plot_pyomo_model
        plot_pyomo_model(m, plot_title='test plot', bipartite_plot=False, show_plot=False)
        plot_pyomo_model(m, plot_title='test plot', bipartite_plot=True, show_plot=False)
        
