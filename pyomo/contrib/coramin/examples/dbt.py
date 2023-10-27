"""
This example demonstrates how to used decomposed bounds
tightening. The example problem is an ACOPF problem.
"""
import pyomo.environ as pe
import coramin
from egret.data.model_data import ModelData
from egret.thirdparty.get_pglib_opf import get_pglib_opf
from egret.models.ac_relaxations import create_polar_acopf_relaxation
from egret.models.acopf import create_psv_acopf_model
import itertools
import os
import time


# Create the NLP and the relaxation
print('Downloading Power Grid Lib')
if not os.path.exists('pglib-opf-master'):
    get_pglib_opf()

print('Creating NLP and relaxation')
md = ModelData.read('pglib-opf-master/api/pglib_opf_case73_ieee_rts__api.m')
nlp, scaled_md = create_psv_acopf_model(md)
relaxation, scaled_md2 = create_polar_acopf_relaxation(md)

# perform decomposition
print('Decomposing relaxation')
relaxation, component_map, termination_reason = coramin.domain_reduction.decompose_model(relaxation, max_leaf_nnz=1000)

# Add more outer approximation points for the second order cone constraints
print('Adding extra outer-approximation points for SOC constraints')
for b in coramin.relaxations.relaxation_data_objects(relaxation, descend_into=True, active=True, sort=True):
    if isinstance(b, coramin.relaxations.MultivariateRelaxationData):
        b.clear_oa_points()
        for bnd_combination in itertools.product(*[itertools.product(['L', 'U'], [v]) for v in b.get_rhs_vars()]):
            bnd_dict = pe.ComponentMap()
            for lower_or_upper, v in bnd_combination:
                if lower_or_upper == 'L':
                    if v.has_lb():
                        bnd_dict[v] = v.lb
                    else:
                        bnd_dict[v] = -1
                else:
                    assert lower_or_upper == 'U'
                    if v.has_ub():
                        bnd_dict[v] = v.ub
                    else:
                        bnd_dict[v] = 1
            b.add_oa_point(var_values=bnd_dict)

# rebuild the relaxations
for b in coramin.relaxations.relaxation_data_objects(relaxation, descend_into=True, active=True, sort=True):
    b.rebuild()

# create solvers
nlp_opt = pe.SolverFactory('ipopt')
rel_opt = pe.SolverFactory('gurobi_persistent')

# solve the nlp to get the upper bound
print('Solving NLP')
res = nlp_opt.solve(nlp)
assert res.solver.termination_condition == pe.TerminationCondition.optimal
ub = pe.value(coramin.utils.get_objective(nlp))

# solve the relaxation to get the lower bound
print('Solving relaxation')
rel_opt.set_instance(relaxation)
res = rel_opt.solve(save_results=False)
assert res.solver.termination_condition == pe.TerminationCondition.optimal
lb = pe.value(coramin.utils.get_objective(relaxation))
gap = (ub - lb) / ub * 100
print('{ub:<20}{lb:<20}{gap:<20}{time:<20}'.format(ub='UB', lb='LB', gap='% gap', time='Time'))
t0 = time.time()
print('{ub:<20.2f}{lb:<20.2f}{gap:<20.2f}{time:<20.2f}'.format(ub=ub, lb=lb, gap=gap, time=time.time() - t0))

for _iter in range(3):
    coramin.domain_reduction.perform_dbt(relaxation=relaxation,
                                         solver=rel_opt,
                                         obbt_method=coramin.domain_reduction.OBBTMethod.DECOMPOSED,
                                         filter_method=coramin.domain_reduction.FilterMethod.AGGRESSIVE,
                                         objective_bound=ub,
                                         with_progress_bar=True)
    for r in coramin.relaxations.relaxation_data_objects(relaxation, descend_into=True, active=True, sort=True):
        r.rebuild()
    rel_opt.set_instance(relaxation)
    res = rel_opt.solve(save_results=False)
    assert res.solver.termination_condition == pe.TerminationCondition.optimal
    lb = pe.value(coramin.utils.get_objective(relaxation))
    gap = (ub - lb) / ub * 100
    print('{ub:<20.2f}{lb:<20.2f}{gap:<20.2f}{time:<20.2f}'.format(ub=ub, lb=lb, gap=gap, time=time.time() - t0))
