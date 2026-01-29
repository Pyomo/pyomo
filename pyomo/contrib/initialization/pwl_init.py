from pyomo.core.base.block import BlockData
import pyomo.environ as pe
from pyomo.contrib.initialization.bounds.bound_variables import bound_all_nonlinear_variables
from pyomo.contrib.initialization.utils import fix_vars_with_equal_bounds


def _minimize_infeasibility(m):
    m.slacks = pe.VarList()
    m.extra_cons = pe.ConstraintList()

    obj_expr = 0

    found_obj = False
    for obj in m.component_data_objects(pe.Objective, active=True, descend_into=True):
        assert not found_obj
        if obj.sense == pe.minimize:
            obj_expr += 0.1*obj.expr
        else:
            obj_expr -= 0.1*obj_expr
        obj.deactivate()
        found_obj = True

    for con in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        lb, body, ub = con.to_bounded_expression(evaluate_bounds=True)
        if lb == ub:
            ps = m.slacks.add()
            ns = m.slacks.add()
            ps.setlb(0)
            ns.setlb(0)
            con.set_value(body - lb - ps + ns == 0)
        elif lb is None:
            ps = m.slacks.add()
            ps.setlb(0)
            con.set_value(body - ub - ps <= 0)
        elif ub is None:
            ns = m.slacks.add()
            ns.setlb(0)
            con.set_value(body - lb + ns >= 0)
        else:
            con.deactivate()
            ps = m.slacks.add()
            ns = m.slacks.add()
            ps.setlb(0)
            ns.setlb(0)
            m.extra_cons.add(body - ub - ps <= 0)
            m.extra_cons.add(body - lb + ns >= 0)

    m.slack_obj = pe.Objective(expr=10*sum(m.slacks.values()) + obj_expr)


def _initialize_with_piecewise_linear_approximation(nlp: BlockData):
    # first introduce auxiliary variables so that we don't try to 
    # approximate any functions of more than two variables
    trans = pe.TransformationFactory('contrib.piecewise.univariate_nonlinear_decomposition')
    trans.apply_to(nlp, aggressive_substitution=True)

    # now we need to try to get bounds on all of the nonlinear variables
    bound_all_nonlinear_variables(nlp)

    # Now, we need to fix variables with equal (or nearly equal) bounds.
    # Otherwise, the PWL transformation complains
    fix_vars_with_equal_bounds(nlp)

    # now we modify the model by introducing slacks to make sure the PWL
    # approximatin is feasible
    _minimize_infeasibility(nlp)

    raise NotImplementedError('not done yet')
