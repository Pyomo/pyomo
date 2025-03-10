#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.constraint import Constraint
from pyomo.core.base.set import Set


def get_piecewise_constant_constraints(inputs, time, sample_points, use_next=True):
    """Returns an IndexedConstraint that constrains the provided variables
    to be constant between the provided sample points

    Arguments
    ---------
    inputs: list of variables
        Time-indexed variables that will be constrained piecewise constant
    time: Set
        Set of points at which provided variables will be constrained
    sample_points: List of floats
        Points at which "constant constraints" will be omitted; these are
        points at which the provided variables may vary.
    use_next: Bool (default True)
        Whether the next time point will be used in the constant constraint
        at each point in time. Otherwise, the previous time point is used.

    Returns
    -------
    Set, IndexedConstraint
        A RangeSet indexing the list of variables provided and a Constraint
        indexed by the product of this RangeSet and time.

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
            var = inputs[i]
            if use_next:
                t_next = time.next(t)
                return var[t] - var[t_next] == 0
            else:
                t_prev = time.prev(t)
                return var[t_prev] - var[t] == 0

    pwc_con = Constraint(input_set, time, rule=piecewise_constant_rule)
    return input_set, pwc_con
