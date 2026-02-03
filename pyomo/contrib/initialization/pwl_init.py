from pyomo.core.base.block import BlockData
import pyomo.environ as pe
from pyomo.contrib.initialization.bounds.bound_variables import bound_all_nonlinear_variables
from pyomo.contrib.initialization.utils import fix_vars_with_equal_bounds, shallow_clone


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


def _refine_pwl_approx(m):


def _initialize_with_piecewise_linear_approximation(nlp: BlockData, default_bound=1.0e8):
    pwl = shallow_clone(nlp)

    # first introduce auxiliary variables so that we don't try to 
    # approximate any functions of more than two variables
    trans = pe.TransformationFactory('contrib.piecewise.univariate_nonlinear_decomposition')
    trans.apply_to(pwl, aggressive_substitution=True)

    # now we need to try to get bounds on all of the nonlinear variables
    bound_all_nonlinear_variables(pwl, default_bound=default_bound)

    # Now, we need to fix variables with equal (or nearly equal) bounds.
    # Otherwise, the PWL transformation complains
    fix_vars_with_equal_bounds(pwl)

    # now we modify the model by introducing slacks to make sure the PWL
    # approximatin is feasible
    # all of the slacks appear linearly, so we don't need to worry about 
    # upper bounds for them
    _minimize_infeasibility(pwl)

    # build the PWL approximation
    trans = pe.TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
    trans.apply_to(pwl, num_points=num_points, additively_decompose=False)

    """
    Now we want to 
    1. solve the PWL approximation
    2. Initialize the NLP to the solution
    3. Try solving the NLP
    4. If the NLP converges => done
    5. If the NLP does not converge, refine the PWL approximation and repeat
    """

    raise NotImplementedError('not done yet')
