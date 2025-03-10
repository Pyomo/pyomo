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

from pyomo.core.expr.numvalue import value as pyo_value


iterable_scalars = (str, bytes)


def _to_iterable(item):
    if hasattr(item, "__iter__"):
        if isinstance(item, iterable_scalars):
            yield item
        else:
            for obj in item:
                yield obj
    else:
        yield item


def copy_values_at_time(
    source_vars, target_vars, source_time_points, target_time_points
):
    # Process input arguments to wrap scalars in a list
    source_time_points = list(_to_iterable(source_time_points))
    target_time_points = list(_to_iterable(target_time_points))
    if (
        len(source_time_points) != len(target_time_points)
        and len(source_time_points) != 1
    ):
        raise ValueError(
            "copy_values_at_time can only copy values when lists of time\n"
            "points have the same length or the source list has length one."
        )
    n_points = len(target_time_points)
    if len(source_time_points) == 1:
        source_time_points = source_time_points * n_points
    for s_var, t_var in zip(source_vars, target_vars):
        for s_t, t_t in zip(source_time_points, target_time_points):
            # Using the value function allows expressions to substitute
            # for variables. However, it raises an error if the expression
            # cannot be evaluated (e.g. has value None).
            # t_var[t_t].set_value(pyo_value(s_var[s_t]))
            t_var[t_t].set_value(s_var[s_t].value)
