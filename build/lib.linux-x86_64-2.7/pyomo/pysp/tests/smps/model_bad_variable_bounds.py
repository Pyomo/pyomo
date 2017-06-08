#
# a model with stochastic variable bounds
#
# these can not be annotated so they must be
# moved to a constraint
#
from pyomo.pysp.tests.smps.utils import *

pysp_scenario_tree_model_callback = \
    simple_twostage_scenario_tree

def pysp_instance_creation_callback(scenario_name, node_names):
    model = simple_twostage_model()
    model.srhs = PySP_StochasticRHSAnnotation()
    model.smat = PySP_StochasticMatrixAnnotation()
    model.sobj = PySP_StochasticObjectiveAnnotation()
    if scenario_name == "Scenario1":
        model.x.setlb(0)
    elif scenario_name == "Scenario2":
        model.x.setlb(1)
    else:
        assert False
    return model
