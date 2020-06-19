#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.sparse import BlockMatrix
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import matplotlib.pylab as plt

from pyomo.pysp.scenariotree.manager import \
    ScenarioTreeManagerFactory
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory


from gas_network_model import (pysp_instance_creation_callback,
                               nx_scenario_tree)

from pyomo.pysp.ef import create_ef_instance

# define and initialize the SP
instance_factory = ScenarioTreeInstanceFactory(
    pysp_instance_creation_callback,
    nx_scenario_tree)
options = ScenarioTreeManagerFactory.register_options()
options.scenario_tree_manager = 'serial'
sp = ScenarioTreeManagerFactory(options,
                                factory=instance_factory)
sp.initialize()

instance = create_ef_instance(sp.scenario_tree)

#instance = create_model(1.0)
nlp = PyomoNLP(instance)
print("\n----------------------")
print("Problem statistics:")
print("----------------------")
print("Number of variables: {:>25d}".format(nlp.n_primals()))
print("Number of equality constraints: {:>14d}".format(nlp.n_eq_constraints()))
print("Number of inequality constraints: {:>11d}".format(nlp.n_ineq_constraints()))
print("Total number of constraints: {:>17d}".format(nlp.n_constraints()))
print("Number of nnz in Jacobian: {:>20d}".format(nlp.nnz_jacobian()))
print("Number of nnz in hessian of Lagrange: {:>8d}".format(nlp.nnz_hessian_lag()))

x = nlp.init_primals().copy()
y = nlp.create_new_vector('duals')
y.fill(1.0)
nlp.set_primals(x)
nlp.set_duals(y)

# Evaluate jacobian of all constraints
jac_full = nlp.evaluate_jacobian()
plt.spy(jac_full)
plt.title('Jacobian of the all constraints\n')
plt.show()

# Evaluate jacobian of the equality constraints
jac = nlp.evaluate_jacobian_eq()
plt.title('Jacobian of the equality constraints\n')
plt.spy(jac)
plt.show()

# Evaluate jacobian of the inequality constraints
jac = nlp.evaluate_jacobian_ineq()
plt.title('Jacobian of the inequality constraints\n')
plt.spy(jac)
plt.show()

# Evaluate hessian of the lagrangian
hess_lag = nlp.evaluate_hessian_lag()
plt.spy(hess_lag.tocoo())
plt.title('Hessian of the Lagrangian function\n')
plt.show()

# Build KKT matrix
kkt = BlockMatrix(2,2)
kkt.set_block(0, 0, hess_lag)
kkt.set_block(1, 0, jac_full)
kkt.set_block(0, 1, jac_full.transpose())
full_kkt = kkt.tocoo()
plt.spy(full_kkt)
plt.title('Karush-Kuhn-Tucker Matrix\n')
plt.show()

