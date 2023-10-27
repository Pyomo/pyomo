import pyomo.environ as pe
import coramin
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr


"""
This example demonstrates a couple features of Coramin:
- Using the "add_cut" methods to refine relaxations with linear constraints
- Optimization-based bounds tightening

The example problem is 

min x**4 - 3*x**2 + x
"""


# Build and solve the NLP
nlp = pe.ConcreteModel()
nlp.x = pe.Var(bounds=(-2, 2))
nlp.obj = pe.Objective(expr=nlp.x**4 - 3*nlp.x**2 + nlp.x)
opt = pe.SolverFactory('ipopt')
res = opt.solve(nlp)
ub = pe.value(nlp.obj)

# Build the relaxation
"""
Reformulate the NLP as

min x4 - 3*x2 + x
s.t.
    x2 = x**2
    x4 = x2**2

Then relax the two constraints with PWXSquaredRelaxation objects.
"""
rel = pe.ConcreteModel()
rel.x = pe.Var(bounds=(-2, 2))
rel.x2 = pe.Var(bounds=compute_bounds_on_expr(rel.x**2))
rel.x4 = pe.Var(bounds=compute_bounds_on_expr(rel.x2**2))
rel.x2_con = coramin.relaxations.PWXSquaredRelaxation()
rel.x2_con.build(x=rel.x, aux_var=rel.x2, use_linear_relaxation=True)
rel.x4_con = coramin.relaxations.PWXSquaredRelaxation()
rel.x4_con.build(x=rel.x2, aux_var=rel.x4, use_linear_relaxation=True)
rel.obj = pe.Objective(expr=rel.x4 - 3*rel.x2 + rel.x)


# Now solve the relaxation and refine the convex sides of the constraints with add_cut
print('*********************************')
print('OA Cut Generation')
print('*********************************')
opt = pe.SolverFactory('gurobi_direct')
res = opt.solve(rel)
lb = pe.value(rel.obj)
print('gap: ' + str(100 * abs(ub - lb) / abs(ub)) + ' %')

for _iter in range(10):
    for b in rel.component_data_objects(pe.Block, active=True, sort=True, descend_into=True):
        if isinstance(b, coramin.relaxations.BaseRelaxationData):
            b.add_cut()
    res = opt.solve(rel)
    lb = pe.value(rel.obj)
    print('gap: ' + str(100 * abs(ub - lb) / abs(ub)) + ' %')

# we want to discard the cuts generated above just to demonstrate OBBT
for b in rel.component_data_objects(pe.Block, active=True, sort=True, descend_into=True):
    if isinstance(b, coramin.relaxations.BasePWRelaxationData):
        b.clear_oa_points()
        b.rebuild()

# Now refine the relaxation with OBBT
print('\n*********************************')
print('OBBT')
print('*********************************')
res = opt.solve(rel)
lb = pe.value(rel.obj)
print('gap: ' + str(100 * abs(ub - lb) / abs(ub)) + ' %')
for _iter in range(10):
    coramin.domain_reduction.perform_obbt(rel, opt, [rel.x, rel.x2], objective_bound=ub)
    for b in rel.component_data_objects(pe.Block, active=True, sort=True, descend_into=True):
        if isinstance(b, coramin.relaxations.BasePWRelaxationData):
            b.rebuild()
    res = opt.solve(rel)
    lb = pe.value(rel.obj)
    print('gap: ' + str(100 * abs(ub - lb) / abs(ub)) + ' %')


