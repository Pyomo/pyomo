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

import logging

logger = logging.getLogger(__name__)

from contextlib import contextmanager

from pyomo.common.dependencies import numpy as numpy, numpy_available

if numpy_available:
    import numpy.random
    from numpy.linalg import norm

import pyomo.environ as pyo
from pyomo.common.modeling import unique_component_name
from pyomo.common.collections import ComponentSet
import pyomo.util.vars_from_expressions as vfe


@contextmanager
def logcontext(level):
    """
    This context manager is used to dynamically set the specified logging level
    and then execute a block of code using that logging level.  When the context is
    deleted, the logging level is reset to the original value.

    Examples
    --------
    >>> with logcontext(logging.INFO):
    ...    logging.debug("This will not be printed")
    ...    logging.info("This will be printed")

    """
    logger = logging.getLogger()
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)


def get_active_objective(model):
    """
    Finds and returns the active objective function for a model. Currently
    assume that there is exactly one active objective.
    """

    active_objs = list(model.component_data_objects(pyo.Objective, active=True))
    assert (
        len(active_objs) == 1
    ), "Model has {} active objective functions, exactly one is required.".format(
        len(active_objs)
    )

    return active_objs[0]


def _add_aos_block(model, name="_aos_block"):
    """Adds an alternative optimal solution block with a unique name."""
    aos_block = pyo.Block()
    model.add_component(unique_component_name(model, name), aos_block)
    return aos_block


def _add_objective_constraint(
    aos_block, objective, objective_value, rel_opt_gap, abs_opt_gap
):
    """
    Adds a relative and/or absolute objective function constraint to the
    specified block.
    """

    assert (
        rel_opt_gap is None or rel_opt_gap >= 0.0
    ), "rel_opt_gap must be None or >= 0.0"
    assert (
        abs_opt_gap is None or abs_opt_gap >= 0.0
    ), "abs_opt_gap must be None or >= 0.0"

    objective_constraints = []

    objective_is_min = objective.is_minimizing()
    objective_expr = objective.expr

    objective_sense = -1
    if objective_is_min:
        objective_sense = 1

    if rel_opt_gap is not None:
        objective_cutoff = objective_value + objective_sense * rel_opt_gap * abs(
            objective_value
        )

        if objective_is_min:
            aos_block.optimality_tol_rel = pyo.Constraint(
                expr=objective_expr <= objective_cutoff
            )
        else:
            aos_block.optimality_tol_rel = pyo.Constraint(
                expr=objective_expr >= objective_cutoff
            )
        objective_constraints.append(aos_block.optimality_tol_rel)

    if abs_opt_gap is not None:
        objective_cutoff = objective_value + objective_sense * abs_opt_gap

        if objective_is_min:
            aos_block.optimality_tol_abs = pyo.Constraint(
                expr=objective_expr <= objective_cutoff
            )
        else:
            aos_block.optimality_tol_abs = pyo.Constraint(
                expr=objective_expr >= objective_cutoff
            )
        objective_constraints.append(aos_block.optimality_tol_abs)

    return objective_constraints


if numpy_available:
    rng = numpy.random.default_rng(9283749387)
else:
    rng = None


def _set_numpy_rng(seed):
    global rng
    rng = numpy.random.default_rng(seed)


def _get_random_direction(num_dimensions, iterations=1000, min_norm=1e-4):
    """
    Get a unit vector of dimension num_dimensions by sampling from and
    normalizing a standard multivariate Gaussian distribution.
    """
    for idx in range(iterations):
        samples = rng.normal(size=num_dimensions)
        samples_norm = norm(samples)
        if samples_norm > min_norm:
            return samples / samples_norm
    raise Exception(  # pragma: no cover
        (
            "Generated {} sequential Gaussian draws with a norm of "
            "less than {}.".format(iterations, min_norm)
        )
    )


def _filter_model_variables(
    variable_set,
    var_generator,
    include_continuous=True,
    include_binary=True,
    include_integer=True,
    include_fixed=False,
):
    """
    Filters variables from a variable generator and adds them to a set.
    """
    for var in var_generator:
        if var in variable_set or (var.is_fixed() and not include_fixed):
            continue
        if (
            (var.is_continuous() and include_continuous)
            or (var.is_binary() and include_binary)
            or (var.is_integer() and include_integer)
        ):
            variable_set.add(var)


def get_model_variables(
    model,
    components=None,
    include_continuous=True,
    include_binary=True,
    include_integer=True,
    include_fixed=False,
):
    """Gathers and returns all variables or a subset of variables from a
    Pyomo model.

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model.
    components: None or a collection of Pyomo components
        The components from which variables should be collected. None
        indicates that all variables will be included. Alternatively, a
        collection of Pyomo Blocks, Constraints, or Variables (indexed or
        non-indexed) from which variables will be gathered can be provided.
        If a Block is provided, all variables associated with constraints
        in that that block and its sub-blocks will be returned. To exclude
        sub-blocks, a tuple element with the format (Block, False) can be
        used.
    include_continuous : boolean
        Boolean indicating that continuous variables should be included.
    include_binary : boolean
        Boolean indicating that binary variables should be included.
    include_integer : boolean
        Boolean indicating that integer variables should be included.
    include_fixed : boolean
        Boolean indicating that fixed variables should be included.

    Returns
    -------
    variable_set
        A Pyomo ComponentSet containing _GeneralVarData variables.

    """

    component_list = (pyo.Objective, pyo.Constraint)
    variable_set = ComponentSet()
    if components == None:
        var_generator = vfe.get_vars_from_components(
            model, component_list, include_fixed=include_fixed
        )
        _filter_model_variables(
            variable_set,
            var_generator,
            include_continuous,
            include_binary,
            include_integer,
            include_fixed,
        )
    else:
        for comp in components:
            if hasattr(comp, "ctype") and comp.ctype == pyo.Block:
                blocks = comp.values() if comp.is_indexed() else (comp,)
                for item in blocks:
                    variables = vfe.get_vars_from_components(
                        item, component_list, include_fixed=include_fixed
                    )
                    _filter_model_variables(
                        variable_set,
                        variables,
                        include_continuous,
                        include_binary,
                        include_integer,
                        include_fixed,
                    )
            elif (
                isinstance(comp, tuple)
                and hasattr(comp[0], "ctype")
                and comp[0].ctype == pyo.Block
            ):
                block = comp[0]
                descend_into = pyo.Block if comp[1] else False
                blocks = block.values() if block.is_indexed() else (block,)
                for item in blocks:
                    variables = vfe.get_vars_from_components(
                        item,
                        component_list,
                        include_fixed=include_fixed,
                        descend_into=descend_into,
                    )
                    _filter_model_variables(
                        variable_set,
                        variables,
                        include_continuous,
                        include_binary,
                        include_integer,
                        include_fixed,
                    )
            elif hasattr(comp, "ctype") and comp.ctype in component_list:
                constraints = comp.values() if comp.is_indexed() else (comp,)
                for item in constraints:
                    variables = pyo.expr.identify_variables(
                        item.expr, include_fixed=include_fixed
                    )
                    _filter_model_variables(
                        variable_set,
                        variables,
                        include_continuous,
                        include_binary,
                        include_integer,
                        include_fixed,
                    )
            elif hasattr(comp, "ctype") and comp.ctype == pyo.Var:
                variables = comp.values() if comp.is_indexed() else (comp,)
                _filter_model_variables(
                    variable_set,
                    variables,
                    include_continuous,
                    include_binary,
                    include_integer,
                    include_fixed,
                )
            else:  # pragma: no cover
                logger.info(
                    ("No variables added for unrecognized component {}.").format(comp)
                )

    return variable_set
