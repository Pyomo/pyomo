#
# a model that is missing a declaration of
# a stochastic constraint rhs
#
from pyomo.pysp.tests.smps.utils import *

pysp_scenario_tree_model_callback = \
    simple_twostage_scenario_tree

def pysp_instance_creation_callback(scenario_name, node_names):
    model = simple_twostage_model()
    if scenario_name == "Scenario1":
        model.cc = Constraint(expr=0 <= model.x + model.y <= 1)
    elif scenario_name == "Scenario2":
        model.cc = Constraint(expr=0 <= model.x + model.y <= 2)
    else:
        assert False
    model.smat = PySP_StochasticRHSAnnotation()
    model.smat.declare(model.cc, lb=True, ub=False)
    return model
