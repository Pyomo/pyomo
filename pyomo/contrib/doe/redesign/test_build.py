from experiment_class_example import *
from pyomo.contrib.doe import *


doe_obj = [0, 0, 0,]

for ind, fd in enumerate(['central', 'backward', 'forward']):
    experiment = FullReactorExperiment(data_ex, 32, 3)
    doe_obj[ind] = DesignOfExperiments_(experiment)
    doe_obj[ind].fd_formula = fd
    doe_obj[ind].step = 0.001
    doe_obj[ind]._generate_scenario_blocks()