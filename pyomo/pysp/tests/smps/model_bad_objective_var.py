#
# a model that is missing a declaration of
# a stochastic objective variable
#
from pyomo.pysp.tests.smps.utils import *

pysp_scenario_tree_model_callback = \
    simple_twostage_scenario_tree

def pysp_instance_creation_callback(scenario_name, node_names):
    model = simple_twostage_model()
    model.sobj = PySP_StochasticObjectiveAnnotation()
    model.sobj.declare(model.o,
                       variables=(model.y,))
    if scenario_name == "Scenario1":
        model.StageCost[1].set_value(model.x * 1)
    elif scenario_name == "Scenario2":
        model.StageCost[1].set_value(model.x * 2)
    else:
        assert False
    return model
