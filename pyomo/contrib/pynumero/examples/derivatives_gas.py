#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.sparse import BlockSymMatrix
from pyomo.contrib.pynumero.interfaces.nlp import PyomoNLP
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
print("\nHi this is PyNumero")
nlp = PyomoNLP(instance)
print("\n----------------------")
print("Problem statistics:")
print("----------------------")
print("Number of variables: {:>25d}".format(nlp.nx))
print("Number of equality constraints: {:>14d}".format(nlp.nc))
print("Number of inequality constraints: {:>11d}".format(nlp.nd))
print("Total number of constraints: {:>17d}".format(nlp.ng))
print("Number of nnz in Jacobian: {:>20d}".format(nlp.nnz_jacobian_g))
print("Number of nnz in hessian of Lagrange: {:>8d}".format(nlp.nnz_hessian_lag))

x = nlp.x_init()
y = nlp.create_vector_y()
y.fill(1.0)

# Evaluate jacobian of all constraints
jac_g = nlp.jacobian_g(x)
plt.spy(jac_g)
plt.title('Jacobian of the all constraints\n')
plt.show()

# Evaluate jacobian of equality constraints
jac_c = nlp.jacobian_c(x)
plt.title('Jacobian of the equality constraints\n')
plt.spy(jac_c)
plt.show()

# Evaluate jacobian of equality constraints
jac_d = nlp.jacobian_d(x)
plt.title('Jacobian of the inequality constraints\n')
plt.spy(jac_d)
plt.show()

# Evaluate hessian of the lagrangian
hess_lag = nlp.hessian_lag(x, y)
plt.spy(hess_lag.tocoo())
plt.title('Hessian of the Lagrangian function\n')
plt.show()

# Build KKT matrix
kkt = BlockSymMatrix(2)
kkt[0, 0] = hess_lag
kkt[1, 0] = jac_g
full_kkt = kkt.tocoo()
plt.spy(full_kkt)
plt.title('Karush-Kuhn-Tucker Matrix\n')
plt.show()

