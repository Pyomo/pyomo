# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging

import pyomo.environ as pyo
from pyomo.core.expr import replace_expressions, identify_mutable_parameters
from pyomo.core.base.var import IndexedVar, VarData
from pyomo.core.base.param import IndexedParam, ParamData
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import numpy as np

from pyomo.environ import ComponentUID

logger = logging.getLogger(__name__)


def convert_params_to_vars(model, param_names=None, fix_vars=False):
    """
    Convert select Params to Vars

    Parameters
    ----------
    model : Pyomo concrete model
        Original model
    param_names : list of strings
        List of parameter names to convert, if None then all Params are converted
    fix_vars : bool
        Fix the new variables, default is False

    Returns
    -------
    model : Pyomo concrete model
        Model with select Params converted to Vars
    """

    model = model.clone()

    if param_names is None:
        param_names = [param.name for param in model.component_data_objects(pyo.Param)]

    indexed_param_names = []

    # Convert Params to Vars, unfix Vars, and create a substitution map
    substitution_map = {}
    comp_map = ComponentMap()
    for i, param_name in enumerate(param_names):
        # Leverage the parser in ComponentUID to locate the component.
        theta_cuid = ComponentUID(param_name)
        theta_object = theta_cuid.find_component_on(model)

        # Param
        if theta_object.is_parameter_type():
            # Delete Param, add Var
            vals = theta_object.extract_values()
            model.del_component(theta_object)
            model.add_component(theta_object.name, pyo.Var(initialize=vals[None]))

            # Update substitution map
            theta_var_cuid = ComponentUID(theta_object.name)
            theta_var_object = theta_var_cuid.find_component_on(model)
            substitution_map[id(theta_object)] = theta_var_object
            comp_map[theta_object] = theta_var_object

        # Indexed Param
        elif isinstance(theta_object, IndexedParam):
            # Delete Param, add Var
            # Before deleting the Param, create a list of the indexed param names
            vals = theta_object.extract_values()
            param_theta_objects = []
            for theta_obj in theta_object:
                indexed_param_name = theta_object.name + '[' + str(theta_obj) + ']'
                theta_cuid = ComponentUID(indexed_param_name)
                param_theta_objects.append(theta_cuid.find_component_on(model))
                indexed_param_names.append(indexed_param_name)

            model.del_component(theta_object)

            index_name = theta_object.index_set().name
            index_cuid = ComponentUID(index_name)
            index_object = index_cuid.find_component_on(model)
            model.add_component(
                theta_object.name, pyo.Var(index_object, initialize=vals)
            )

            # Update substitution map (map each indexed param to indexed var)
            theta_var_cuid = ComponentUID(theta_object.name)
            theta_var_object = theta_var_cuid.find_component_on(model)
            comp_map[theta_object] = theta_var_object
            var_theta_objects = []
            for theta_obj in theta_var_object:
                theta_cuid = ComponentUID(
                    theta_var_object.name + '[' + str(theta_obj) + ']'
                )
                var_theta_objects.append(theta_cuid.find_component_on(model))

            for param_theta_obj, var_theta_obj in zip(
                param_theta_objects, var_theta_objects
            ):
                substitution_map[id(param_theta_obj)] = var_theta_obj
                comp_map[param_theta_obj] = var_theta_obj

        # Var or Indexed Var
        elif isinstance(theta_object, IndexedVar) or theta_object.is_variable_type():
            theta_var_object = theta_object

        else:
            logger.warning("%s is not a Param or Var on the model", (param_name))
            return model

        if fix_vars:
            theta_var_object.fix()
        else:
            theta_var_object.unfix()

    # If no substitutions are needed, return the model
    if len(substitution_map) == 0:
        return model

    # Update the list of param_names if the parameters were indexed
    if len(indexed_param_names) > 0:
        param_names = indexed_param_names

    # Convert Params to Vars in Expressions
    for expr in model.component_data_objects(pyo.Expression):
        if expr.active and any(
            v.name in param_names for v in identify_mutable_parameters(expr)
        ):
            new_expr = replace_expressions(expr=expr, substitution_map=substitution_map)
            expr.expr = new_expr

    # Convert Params to Vars in Constraint expressions
    num_constraints = len(list(model.component_objects(pyo.Constraint, active=True)))
    if num_constraints > 0:
        model.constraints = pyo.ConstraintList()
        for c in model.component_data_objects(pyo.Constraint):
            if c.active and any(
                v.name in param_names for v in identify_mutable_parameters(c.expr)
            ):
                if c.equality:
                    model.constraints.add(
                        replace_expressions(
                            expr=c.lower, substitution_map=substitution_map
                        )
                        == replace_expressions(
                            expr=c.body, substitution_map=substitution_map
                        )
                    )
                elif c.lower is not None:
                    model.constraints.add(
                        replace_expressions(
                            expr=c.lower, substitution_map=substitution_map
                        )
                        <= replace_expressions(
                            expr=c.body, substitution_map=substitution_map
                        )
                    )
                elif c.upper is not None:
                    model.constraints.add(
                        replace_expressions(
                            expr=c.upper, substitution_map=substitution_map
                        )
                        >= replace_expressions(
                            expr=c.body, substitution_map=substitution_map
                        )
                    )
                else:
                    raise ValueError(
                        "Unable to parse constraint to convert params to vars."
                    )
                c.deactivate()

    # Convert Params to Vars in Objective expressions
    for obj in model.component_data_objects(pyo.Objective):
        if obj.active and any(
            v.name in param_names for v in identify_mutable_parameters(obj)
        ):
            expr = replace_expressions(expr=obj.expr, substitution_map=substitution_map)
            model.del_component(obj)
            model.add_component(obj.name, pyo.Objective(rule=expr, sense=obj.sense))

    # Convert Params to Vars in Suffixes
    for s in model.component_objects(pyo.Suffix):
        current_keys = list(s.keys())
        for c in current_keys:
            if c in comp_map:
                s[comp_map[c]] = s.pop(c)

        assert len(current_keys) == len(s.keys())

    # print('--- Updated Model ---')
    # model.pprint()
    # solver = pyo.SolverFactory('ipopt')
    # solver.solve(model)

    return model


def update_model_from_suffix(suffix_obj: pyo.Suffix, values):
    """
    Overwrite each variable/parameter referenced by ``suffix_obj`` with the
    corresponding value in ``values``. The provided values are expected to
    be in the same order as the components in the suffix from when it was
    created.

    Parameters
    ----------
    suffix_obj : pyomo.core.base.suffix.Suffix
        The suffix whose *keys* are the components you want to update.
        Call like ``update_from_suffix(model.unknown_parameters, vals)``.
    values : iterable of numbers
        New numerical values for the components referenced by the suffix.
        Must be the same length as ``suffix_obj``.

    Notes
    -----
    The measurement_error suffix is a special case: instead of updating the value of the
    keys (variables/parameters), it updates the value stored in the suffix itself.
    """
    # Check that the length of values matches the suffix length
    comps = list(suffix_obj.keys())
    if len(comps) != len(values):
        raise ValueError("values length does not match suffix length")

    # Add a check for measurement error suffix
    is_me_err = "measurement_error" in suffix_obj.name
    # Iterate through the keys in the suffix and update their values
    # First loop: check all values are the right type
    for comp in comps:
        if not isinstance(comp, (VarData, ParamData)):
            raise TypeError(
                f"Unsupported component type {type(comp)}; expected VarData or ParamData."
            )
    # Second loop: adjust the values
    for comp, new_val in zip(comps, values):
        if is_me_err:
            suffix_obj[comp] = float(new_val)
        else:
            comp.set_value(float(new_val))


def _build_meas_error_covariance_matrix(model, estimated_var=None):
    """
    Builds a full measurement-error covariance matrix

    Note: The code does not automatically build the full covariance matrix
    It only places whatever covariances the user explicitly supplies
    Therefore, for correlation in time, shared timepoints, or other types of
    correlation, the user must provide all the desired covariance terms
    The diagonal elements can be constructed automatically or the standard
    deviations can be specified by the user. The standard deviations may be
    constant or depend on the value (i.e., data) of the measured or input
    variables (e.g., be proportional to them)

    Parameters
    ----------
    model : ConcreteModel
        Annotated Pyomo model
    estimated_var: float or int, optional
        Value of the estimated variance of the measurement error
        in cases where the user does not supply the
        measurement-error standard deviation

    Returns
    -------
    Sigma_y: numpy.ndarray
        Full measurement-error covariance matrix
    """
    # get the output variables
    outputs = list(model.experiment_outputs.keys())
    outputs_name = [y_hat.name for y_hat in outputs]
    outputs_index = {y_hat_name: i for i, y_hat_name in enumerate(outputs_name)}

    number_outputs = len(outputs)

    # define the measurement-error covariance matrix
    Sigma_y = np.zeros((number_outputs, number_outputs))

    if hasattr(model, "measurement_error"):
        all_known_errors = all(
            model.measurement_error[y_hat] is not None
            for y_hat in model.experiment_outputs
        )

        # fill the leading-diagonal elements from the standard deviation of
        # the measurement errors
        for y_hat in outputs:
            i = outputs_index[y_hat.name]

            if all_known_errors:
                standard_dev = model.measurement_error[y_hat]
                Sigma_y[i, i] = standard_dev**2
            else:
                Sigma_y[i, i] = estimated_var

        # fill the off-diagonal elements from covariance entries
        # supplied by the user
        for key, err_cov in model.measurement_error.items():
            if isinstance(key, tuple) and len(key) == 2:
                yi, yj = key
                if yi.name not in outputs_name or yj.name not in outputs_name:
                    raise ValueError(
                        "Measurement-error covariance must be defined only between "
                        "experiment output variables."
                    )

                i = outputs_index[yi.name]
                j = outputs_index[yj.name]

                # update the measurement-error covariance matrix which
                # is a symmetric matrix
                Sigma_y[i, j] = err_cov
                Sigma_y[j, i] = err_cov
            elif not isinstance(key, tuple) and not hasattr(key, "name"):
                raise TypeError(
                    "Expected a tuple of two measured variables when specifying a "
                    "measurement-error covariance, e.g., "
                    "measurement_error[(y1, y2)] = covariance."
                )

    return Sigma_y


def get_meas_error_covariance_matrix_inv(model, estimated_var=None):
    """
    Computes the inverse of the measurement-error covariance matrix

    Parameters
    ----------
    model : ConcreteModel
        Annotated Pyomo model
    estimated_var: float or int, optional
        Value of the estimated variance of the measurement error
        in cases where the user does not supply the
        measurement-error standard deviation

    Returns
    -------
    Sigma_y_inv: numpy.ndarray
        Inverse of the measurement-error covariance matrix
    """
    # get the measurement-error covariance matrix
    Sigma_y = _build_meas_error_covariance_matrix(model, estimated_var)

    # compute the inverse of the measurement-error covariance matrix
    try:
        Sigma_y_inv = np.linalg.inv(Sigma_y)
    except np.linalg.LinAlgError:
        Sigma_y_inv = np.linalg.pinv(Sigma_y)
        logger.warning(
            "The measurement-error covariance matrix is singular. "
            "Using pseudo-inverse instead."
        )

    return Sigma_y_inv
