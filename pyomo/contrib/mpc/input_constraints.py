from pyomo.core.base.constraint import Constraint
from pyomo.core.base.set import Set


def get_piecewise_constant_constraints(
        inputs,
        time,
        sample_points,
        use_next=True,
        ):
    """
    """
    input_set = Set(initialize=range(len(inputs)))
    sample_point_set = set(sample_points)
    def piecewise_constant_rule(m, i, t):
        if t in sample_point_set:
            return Constraint.Skip
        else:
            # I think whether we want prev or next here depends on whether
            # we use an explicit or implicit time discretization. I.e. whether
            # an input is applied to the finite element in front of or behind
            # its time point. If the wrong direction for a discretization
            # is used, we could have different inputs applied within the same
            # finite element, which I think we never want.
            #t_prev = time.prev(t)
            var = inputs[i]
            if use_next:
                t_next = time.next(t)
                return var[t] == var[t_next]
            else:
                t_prev = time.prev(t)
                return var[t_prev] == var[t]
    pwc_con = Constraint(input_set, time, rule=piecewise_constant_rule)
    return input_set, pwc_con
