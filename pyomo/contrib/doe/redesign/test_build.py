from experiment_class_example import *
from pyomo.contrib.doe import *

from simple_reaction_example import *

import numpy as np

doe_obj = [0, 0, 0, 0,]
obj = ['trace', 'det', 'det']

for ind, fd in enumerate(['central', 'backward', 'forward']):
    print(fd)
    experiment = FullReactorExperiment(data_ex, 10, 3)
    doe_obj[ind] = DesignOfExperiments_(
        experiment, 
        fd_formula='central',
        step=1e-3,
        objective_option=ObjectiveLib(obj[ind]),
        scale_constant_value=1,
        scale_nominal_param_value=(True and (ind != 2)),
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_initial=None,
        L_LB=1e-7,
        solver=None,
        tee=True,
        args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )
    doe_obj[ind].run_doe()


ind = 3

doe_obj[ind] = DesignOfExperiments_(
        experiment, 
        fd_formula='central',
        step=1e-3,
        objective_option=ObjectiveLib.det,
        scale_constant_value=1,
        scale_nominal_param_value=(True and (ind != 2)),
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_initial=None,
        L_LB=1e-7,
        solver=None,
        tee=False,
        args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )
doe_obj[ind].model.set_blocks = pyo.Set(initialize=[0, 1, 2])
doe_obj[ind].model.block_instances = pyo.Block(doe_obj[ind].model.set_blocks)
doe_obj[ind].create_doe_model(mod=doe_obj[ind].model.block_instances[0])
doe_obj[ind].create_doe_model(mod=doe_obj[ind].model.block_instances[1])
doe_obj[ind].create_doe_model(mod=doe_obj[ind].model.block_instances[2])
# doe_obj[ind].run_doe()

print('Multi-block build complete')

# Old interface comparison
def create_model(m=None, ):
    experiment = FullReactorExperiment(data_ex, 10, 3)
    m = experiment.get_labeled_model().clone()
    return m

### Define inputs
# Control time set [h]
t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
# Define parameter nominal value
parameter_dict = {"A1": 85, "A2": 372, "E1": 8, "E2": 15}

# measurement object
measurements = MeasurementVariables()
measurements.add_variables(
    "CA",  # name of measurement
    indices={0: t_control},  # indices of measurement
    time_index_position=0,
    variance=1e-2,
)  # position of time index

measurements.add_variables(
    "CB",  # name of measurement
    indices={0: t_control},  # indices of measurement
    time_index_position=0,
    variance=1e-2,
)  # position of time index

measurements.add_variables(
    "CC",  # name of measurement
    indices={0: t_control},  # indices of measurement
    time_index_position=0,
    variance=1e-2,
)  # position of time index

# design object
exp_design = DesignVariables()

# add CAO as design variable
exp_design.add_variables(
    "CA",  # name of design variable
    indices={0: [0]},  # indices of design variable
    time_index_position=0,  # position of time index
    values=[5],  # nominal value of design variable
    lower_bounds=1,  # lower bound of design variable
    upper_bounds=5,  # upper bound of design variable
)

# add T as design variable
exp_design.add_variables(
    "T",  # name of design variable
    indices={0: t_control},  # indices of design variable
    time_index_position=0,  # position of time index
    values=[
        300,
        300,
        300,
        300,
        300,
        300,
        300,
        300,
        300,
    ],  # nominal value of design variable
    lower_bounds=300,  # lower bound of design variable
    upper_bounds=700,  # upper bound of design variable
)

design_names = exp_design.variable_names
exp1 = [5, 300, 300, 300, 300, 300, 300, 300, 300, 300]
exp1_design_dict = dict(zip(design_names, exp1))
exp_design.update_values(exp1_design_dict)

# add a prior information (scaled FIM with T=500 and T=300 experiments)
# prior = np.asarray(
    # [
        # [28.67892806, 5.41249739, -81.73674601, -24.02377324],
        # [5.41249739, 26.40935036, -12.41816477, -139.23992532],
        # [-81.73674601, -12.41816477, 240.46276004, 58.76422806],
        # [-24.02377324, -139.23992532, 58.76422806, 767.25584508],
    # ]
# )

prior = None

doe_object = DesignOfExperiments(
    parameter_dict,  # dictionary of parameters
    exp_design,  # design variables
    measurements,  # measurement variables
    create_model,  # function to create model
    only_compute_fim_lower=True,
)

square_result, optimize_result = doe_object.stochastic_program(
    if_optimize=True,  # if optimize
    if_Cholesky=True,  # if use Cholesky decomposition
    scale_nominal_param_value=True,  # if scale nominal parameter value
    objective_option="det",  # objective option
)

doe_object2 = DesignOfExperiments(
    parameter_dict,  # dictionary of parameters
    exp_design,  # design variables
    measurements,  # measurement variables
    create_model,  # function to create model
    only_compute_fim_lower=True,
)

square_result2, optimize_result2 = doe_object2.stochastic_program(
    if_optimize=True,  # if optimize
    if_Cholesky=True,  # if use Cholesky decomposition
    scale_nominal_param_value=False,  # if scale nominal parameter value
    objective_option="det",  # objective option
)

# Optimal values
print("Optimal values for determinant optimized experimental design:")
print("New formulation, scaled: {}".format(pyo.value(doe_obj[1].model.Obj)))
print("New formulation, unscaled: {}".format(pyo.value(doe_obj[2].model.Obj)))
print("Old formulation, scaled: {}".format(pyo.value(optimize_result.model.Obj)))
print("Old formulation, unscaled: {}".format(pyo.value(optimize_result2.model.Obj)))