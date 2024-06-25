from experiment_class_example import *
from pyomo.contrib.doe import *

experiment = FullReactorExperiment(data_ex, 32, 3)
doe_obj = DesignOfExperiments_(experiment)
doe_obj.fd_formula = 'central'
doe_obj.step = 0.001
doe_obj._generate_scenario_blocks()