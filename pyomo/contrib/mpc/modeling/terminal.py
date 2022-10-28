#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression

from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData


def get_quadratic_penalty_at_time(var, t, target, weight=None):
    if weight is None:
        weight = 1.0
    return weight * (var[t] - target)**2


#
# TODO: These penalty expressions should be indexed by a set of variable
# indices
#
def get_penalty_expressions_at_time(
    variables,
    time,
    t,
    target_data,
    weight_data=None,
):
    """
    """
    if not isinstance(target_data, ScalarData):
        target_data = ScalarData(target_data)
    target_data = target_data.get_data()

    cuids = [
        get_indexed_cuid(var, sets=(time,))
        for var in variables
    ]
    # TODO: Weight data (and setpoint data) are user-provided and don't
    # necessarily have CUIDs as keys. Should I process the keys here
    # with get_indexed_cuid?
    if weight_data is None:
        weight_data = {cuid: 1.0 for cuid in cuids}
    for i, cuid in enumerate(cuids):
        if cuid not in target_data:
            raise KeyError(
                "Target data dictionary does not contain a key for variable"
                " %s with ComponentUID %s" % (variables[i].name, cuid)
            )
        if cuid not in weight_data:
            raise KeyError(
                "Terminal penalty weight dictionary does not contain a key for"
                " variable %s with ComponentUID %s" % (variables[i].name, cuid)
            )

    penalties = [
        get_quadratic_penalty_at_time(
            var, t, target_data[cuid], weight_data[cuid]
        ) for var, cuid in zip(variables, cuids)
    ]
    return penalties


def get_penalty_at_time(
    variables,
    time,
    t,
    target_data,
    weight_data=None,
):
    terminal_penalty = Expression(
        expr=sum(get_penalty_expressions_at_time(
            variables, time, t, target_data, weight_data
        ))
    )
    return terminal_penalty


def get_terminal_penalty(
    variables,
    time,
    target_data,
    weight_data=None,
):
    t = time.last()
    #terminal_penalty = Expression(
    #    expr=sum(get_penalty_expressions_at_time(
    #        variables, time, t, target_data, weight_data
    #    ))
    #)
    return get_penalty_at_time(
        variables, time, t, target_data, weight_data=weight_data
    )
