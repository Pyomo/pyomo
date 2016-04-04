#
# a model that is missing a declaration of
# a stochastic objective constant
#
from pyomo.pysp.tests.smps.utils import *

pysp_scenario_tree_model_callback = \
    simple_twostage_scenario_tree

def pysp_instance_creation_callback(scenario_name, node_names):
    model = simple_twostage_model()
    model.sobj = PySP_StochasticObjectiveAnnotation()
    model.sobj.declare(model.o, include_constant=False)
    if scenario_name == "Scenario1":
        model.o.expr = model.o.expr + 0
    elif scenario_name == "Scenario2":
        model.o.expr = model.o.expr + 1
    else:
        assert False
    return model
