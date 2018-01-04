#
# a model that declares the wrong component type
# within an annotation
#
from pyomo.pysp.tests.convert.utils import *

pysp_scenario_tree_model_callback = \
    simple_twostage_scenario_tree

def pysp_instance_creation_callback(scenario_name, node_names):
    model = simple_twostage_model()
    model.svb = StochasticVariableBoundsAnnotation()
    return model
