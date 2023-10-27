"""
This example demonstrates how to used decomposed bounds
tightening. The example problem is from minlplib.  In order to run
this example, you have to download the problem file corresponding to
the filename in the "read_osil" function bedlow. The file can be
downloaded from minlplib.org. Suspect is also needed.
"""
import pyomo.environ as pe
import coramin
import itertools
import os
import time
from suspect.pyomo import read_osil
from coramin.third_party.minlplib_tools import get_minlplib


print('Downloading camshape800 from MINLPLib')
if not os.path.exists(os.path.join('minlplib', 'osil', 'camshape800.osil')):
    get_minlplib(problem_name='camshape800')

print('Creating NLP and relaxation')
nlp = read_osil('minlplib/osil/camshape800.osil', objective_prefix='obj_', constraint_prefix='con_')
relaxation = coramin.relaxations.relax(nlp, in_place=False, use_fbbt=True, fbbt_options={'deactivate_satisfied_constraints': True,
                                                                                         'max_iter': 2})

# perform decomposition
print('Decomposing relaxation')
relaxation, component_map, termination_reason = coramin.domain_reduction.decompose_model(relaxation, max_leaf_nnz=1000)

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
gap = (ub - lb) / abs(ub) * 100
var_bounds = pe.ComponentMap()
for r in coramin.relaxations.relaxation_data_objects(relaxation, descend_into=True, active=True, sort=True):
    for v in r.get_rhs_vars():
        var_bounds[v] = v.ub - v.lb
avg_bound_range = sum(var_bounds.values()) / len(var_bounds)
print('{ub:<20}{lb:<20}{gap:<20}{avg_rng:<20}{time:<20}'.format(ub='UB', lb='LB', gap='% gap', avg_rng='Avg Var Range', time='Time'))
t0 = time.time()
print('{ub:<20.3f}{lb:<20.3f}{gap:<20.3f}{avg_rng:<20.3e}{time:<20.3f}'.format(ub=ub, lb=lb, gap=gap, avg_rng=avg_bound_range, time=time.time() - t0))

# Perform bounds tightening
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
    gap = (ub - lb) / abs(ub) * 100
    var_bounds = pe.ComponentMap()
    for r in coramin.relaxations.relaxation_data_objects(relaxation, descend_into=True, active=True, sort=True):
        for v in r.get_rhs_vars():
            var_bounds[v] = v.ub - v.lb
    avg_bound_range = sum(var_bounds.values()) / len(var_bounds)
    print('{ub:<20.3f}{lb:<20.3f}{gap:<20.3f}{avg_rng:<20.3e}{time:<20.3f}'.format(ub=ub, lb=lb, gap=gap, avg_rng=avg_bound_range, time=time.time() - t0))
