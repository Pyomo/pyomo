import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoGreyBoxNLP
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptNLP, CyIpoptSolver
from reactor_model_residuals import ReactorModel, ReactorModelNoOutputs

def maximize_cb_residuals_with_output(show_solver_log=False):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the ratio of B to the other components coming out of the reactor
    # This example illustrates the use of "equality constraints" or residuals
    # in the external grey box example as well as outputs
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock()
    m.reactor.set_external_model(ReactorModel())

    # reactor block will automatically get the input and output variables
    # inputs: sv, caf, k1, k2, k3
    # outputs: ca, cb, cc, cd
    # let's set reasonable bounds on some of the variables
    m.reactor.inputs['sv'].setlb(0)
    m.reactor.inputs['ca'].setlb(0)
    m.reactor.inputs['cb'].setlb(0)
    m.reactor.inputs['cc'].setlb(0)
    m.reactor.inputs['cd'].setlb(0)

    # The feed concentration will be fixed for this example
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    #m.obj = pyo.Objective(expr=m.reactor.inputs['cb'], sense=pyo.maximize)
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb_ratio'], sense=pyo.maximize)

    # initialize the variables
    m.reactor.inputs['sv'].value = 1
    m.reactor.inputs['caf'].value = 1
    m.reactor.inputs['ca'].value = 1
    m.reactor.inputs['cb'].value = 1
    m.reactor.inputs['cc'].value = 1
    m.reactor.inputs['cd'].value = 1
    m.reactor.outputs['cb_ratio'].value = 1
    pyomo_nlp = PyomoGreyBoxNLP(m)

    options = {'hessian_approximation':'limited-memory'}
    cyipopt_problem = CyIpoptNLP(pyomo_nlp)
    solver = CyIpoptSolver(cyipopt_problem, options)
    x, info = solver.solve(tee=show_solver_log)
    pyomo_nlp.load_x_into_pyomo(x)
    return m

def maximize_cb_residuals_with_pyomo_variables(show_solver_log=False):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the ratio of B to the other components coming out of the reactor
    # This example illustrates the use of "equality constraints" or residuals
    # in the external grey box example as well as additional pyomo variables
    # and constraints
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock()
    m.reactor.set_external_model(ReactorModelNoOutputs())

    # add a variable and constraint for the cb ratio
    m.cb_ratio = pyo.Var(bounds=(0,None), initialize=1)
    u = m.reactor.inputs
    m.cb_ratio_con = pyo.Constraint(expr = \
        m.cb_ratio == u['cb']/(u['ca']+u['cc']+u['cd']))

    # reactor block will automatically get the input and output variables
    # inputs: sv, caf, k1, k2, k3
    # outputs: ca, cb, cc, cd
    # let's set reasonable bounds on some of the variables
    m.reactor.inputs['sv'].setlb(0)
    m.reactor.inputs['ca'].setlb(0)
    m.reactor.inputs['cb'].setlb(0)
    m.reactor.inputs['cc'].setlb(0)
    m.reactor.inputs['cd'].setlb(0)

    # The feed concentration will be fixed for this example
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)
    m.con = pyo.Constraint(expr=m.reactor.inputs['sv'] ==  1.04312815332151)

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    #m.obj = pyo.Objective(expr=m.reactor.inputs['cb'], sense=pyo.maximize)
    m.obj = pyo.Objective(expr=m.cb_ratio, sense=pyo.maximize)

    # initialize the variables
    m.reactor.inputs['sv'].value = 1
    m.reactor.inputs['caf'].value = 1
    m.reactor.inputs['ca'].value = 1
    m.reactor.inputs['cb'].value = 1
    m.reactor.inputs['cc'].value = 1
    m.reactor.inputs['cd'].value = 1
    pyomo_nlp = PyomoGreyBoxNLP(m)

    options = {'hessian_approximation':'limited-memory'}
    cyipopt_problem = CyIpoptNLP(pyomo_nlp)
    solver = CyIpoptSolver(cyipopt_problem, options)
    x, info = solver.solve(tee=show_solver_log)
    pyomo_nlp.load_x_into_pyomo(x)
    return m

if __name__ == '__main__':
    m = maximize_cb_residuals_with_output(show_solver_log=True)
    m.pprint()
    m = maximize_cb_residuals_with_pyomo_variables(show_solver_log=True)
    m.pprint()
    

