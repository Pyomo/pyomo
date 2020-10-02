from __future__ import division
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoGreyBoxNLP
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptNLP, CyIpoptSolver
from reactor_model_outputs import ReactorConcentrationsOutputModel

def maximize_cb_outputs(show_solver_log=False):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the concentration of component B coming out of the reactor
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock(
        external_model=ReactorConcentrationsOutputModel()
    )

    # The reaction rate constants and the feed concentration will
    # be fixed for this example
    m.k1con = pyo.Constraint(expr=m.reactor.inputs['k1'] == 5/6)
    m.k2con = pyo.Constraint(expr=m.reactor.inputs['k2'] == 5/3)
    m.k3con = pyo.Constraint(expr=m.reactor.inputs['k3'] == 1/6000)
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb'], sense=pyo.maximize)

    # pyomo_nlp = PyomoGreyBoxNLP(m)

    # options = {'hessian_approximation':'limited-memory',
    #            'print_level': 10}
    # cyipopt_problem = CyIpoptNLP(pyomo_nlp)
    # solver = CyIpoptSolver(cyipopt_problem, options)
    # x, info = solver.solve(tee=show_solver_log)
    # pyomo_nlp.load_x_into_pyomo(x)

    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    solver.solve(m, tee=show_solver_log)
    return m

if __name__ == '__main__':
    m = maximize_cb_outputs(show_solver_log=True)
    m.pprint()
    

