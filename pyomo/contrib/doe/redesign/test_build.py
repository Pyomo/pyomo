from experiment_class_example import *
from pyomo.contrib.doe import *

import numpy as np
import logging

f = open('result.json')
data_ex = json.load(f)
data_ex['control_points'] = {float(k): v for k, v in data_ex['control_points'].items()}

doe_obj = [0, 0, 0, 0,]
obj = ['trace', 'det', 'det']

for ind, fd in enumerate(['central', 'backward', 'forward']):
    experiment = FullReactorExperiment(data_ex, 10, 3)
    doe_obj[ind] = DesignOfExperiments(
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
        tee=False,
        args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
        logger_level=logging.INFO,
    )
    doe_obj[ind].run_doe()

# add a prior information (scaled FIM with T=500 and T=300 experiments)
# prior = np.asarray(
    # [
        # [  1745.81343391   1599.21859987  -3512.47892155  -7589.26220445]
        # [  1599.21859987   3525.63856364  -2900.94638673 -16465.46338508]
        # [ -3512.47892155  -2900.94638673   7190.13048958  13849.96839993]
        # [ -7589.26220445 -16465.46338508  13849.96839993  77674.03976715]]
    # ]
# )

prior = None
design_ranges = {
    'CA[0]': [1, 5, 3], 
    'T[0]': [300, 700, 3],
}
doe_obj[0].compute_FIM_full_factorial(design_ranges=design_ranges, method='kaug')

doe_obj[0].compute_FIM(method='kaug')

doe_obj[0].compute_FIM(method='sequential')

print(doe_obj[0].kaug_FIM)

# Optimal values
print("Optimal values for determinant optimized experimental design:")
print("New formulation, scaled: {}".format(pyo.value(doe_obj[1].model.objective)))
print("New formulation, unscaled: {}".format(pyo.value(doe_obj[2].model.objective)))

# New values
FIM_vals_new = [pyo.value(doe_obj[1].model.fim[i, j]) for i in doe_obj[1].model.parameter_names for j in doe_obj[1].model.parameter_names]
L_vals_new = [pyo.value(doe_obj[1].model.L[i, j]) for i in doe_obj[1].model.parameter_names for j in doe_obj[1].model.parameter_names]
Q_vals_new = [pyo.value(doe_obj[1].model.sensitivity_jacobian[i, j]) for i in doe_obj[1].model.output_names for j in doe_obj[1].model.parameter_names]
sigma_inv_new = [1 / v for k,v in doe_obj[1].model.scenario_blocks[0].measurement_error.items()]
param_vals = np.array([[v for k, v in doe_obj[1].model.scenario_blocks[0].unknown_parameters.items()], ])

FIM_vals_new_np = np.array(FIM_vals_new).reshape((4, 4))

for i in range(4):
    for j in range(4):
        if j < i:
            FIM_vals_new_np[j, i] = FIM_vals_new_np[i, j]

L_vals_new_np = np.array(L_vals_new).reshape((4, 4))
Q_vals_new_np = np.array(Q_vals_new).reshape((27, 4))

sigma_inv_new_np = np.zeros((27, 27))
for i in range(27):
    sigma_inv_new_np[i, i] = sigma_inv_new[i]

rescaled_FIM = rescale_FIM(FIM=FIM_vals_new_np, param_vals=param_vals)

# Comparing values from compute FIM
print("Results from using compute FIM (first old, then new)")
print(doe_obj[0].kaug_FIM)
print(doe_obj[0].seq_FIM)
print(np.isclose(doe_obj[0].kaug_FIM, doe_obj[0].seq_FIM, 1e-2))
print(np.log10(np.linalg.det(doe_obj[0].kaug_FIM)))
print(np.log10(np.linalg.det(doe_obj[0].seq_FIM)))
A = doe_obj[0].kaug_jac
B = doe_obj[0].seq_jac
print(np.sum((A - B) ** 2))

measurement_vals_model = []
meas_from_model = []
mod = doe_obj[0].model
for p in mod.parameter_names:
    fd_step_mult = 1
    param_ind = mod.parameter_names.data().index(p)
    
    # Different FD schemes lead to different scenarios for the computation
    if doe_obj[0].fd_formula == FiniteDifferenceStep.central:
        s1 = param_ind * 2
        s2 = param_ind * 2 + 1
        fd_step_mult = 2
    elif doe_obj[0].fd_formula == FiniteDifferenceStep.forward:
        s1 = param_ind + 1
        s2 = 0
    elif doe_obj[0].fd_formula == FiniteDifferenceStep.backward:
        s1 = 0
        s2 = param_ind + 1

    var_up = [pyo.value(k) for k, v in mod.scenario_blocks[s1].experiment_outputs.items()]
    var_lo = [pyo.value(k) for k, v in mod.scenario_blocks[s2].experiment_outputs.items()]
    
    meas_from_model.append(var_up)
    meas_from_model.append(var_lo)


# Optimal values
print("Optimal values for determinant optimized experimental design:")
print("New formulation, scaled: {}".format(pyo.value(doe_obj[1].model.objective)))
print("New formulation, unscaled: {}".format(pyo.value(doe_obj[2].model.objective)))
print("New formulation, rescaled: {}".format(np.log10(np.linalg.det(rescaled_FIM))))

# Draw figures
sens_vars = ['CA[0]', 'T[0]']
des_vars_fixed = {'T[' + str((i + 1) / 8) + ']': 300 for i in range(7)}
des_vars_fixed['T[1]'] = 300
doe_obj[0].draw_factorial_figure(title_text='', xlabel_text='', ylabel_text='', sensitivity_design_variables=sens_vars, fixed_design_variables=des_vars_fixed,)