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
from pyomo.common.collections import ComponentMap
from pyomo.core.base import Block, Var, Constraint, Objective, Suffix, value
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.base import TransformationFactory
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.expr import replace_expressions
from pyomo.util.components import rename_components

logger = logging.getLogger("pyomo.core.plugins.transform.scaling")


@TransformationFactory.register(
    'core.scale_model', doc="Scale model variables, constraints, and objectives."
)
class ScaleModel(Transformation):
    """
    Transformation to scale a model.

    This plugin performs variable, constraint, and objective scaling on
    a model based on the scaling factors in the suffix 'scaling_factor'
    set for the variables, constraints, and/or objective. This is typically
    done to scale the problem for improved numerical properties.

    Supported transformation methods:
        * :py:meth:`apply_to <pyomo.core.plugins.transform.scaling.ScaleModel.apply_to>`
        * :py:meth:`create_using <pyomo.core.plugins.transform.scaling.ScaleModel.create_using>`
        * :py:meth:`propagate_solution <pyomo.core.plugins.transform.scaling.ScaleModel.propagate_solution>`

    By default, scaling components are renamed with the prefix ``scaled_``. To disable
    this behavior and scale variables in-place (or keep the same names in a new model),
    use the ``rename=False`` argument to ``apply_to`` or ``create_using``.


    Examples
    --------

    .. doctest::

        >>> from pyomo.environ import *
        >>> # create the model
        >>> model = ConcreteModel()
        >>> model.x = Var(bounds=(-5, 5), initialize=1.0)
        >>> model.y = Var(bounds=(0, 1), initialize=1.0)
        >>> model.obj = Objective(expr=1e8*model.x + 1e6*model.y)
        >>> model.con = Constraint(expr=model.x + model.y == 1.0)
        >>> # create the scaling factors
        >>> model.scaling_factor = Suffix(direction=Suffix.EXPORT)
        >>> model.scaling_factor[model.obj] = 1e-6 # scale the objective
        >>> model.scaling_factor[model.con] = 2.0  # scale the constraint
        >>> model.scaling_factor[model.x] = 0.2    # scale the x variable
        >>> # transform the model
        >>> scaled_model = TransformationFactory('core.scale_model').create_using(model)
        >>> # print the value of the objective function to show scaling has occurred
        >>> print(value(model.x))
        1.0
        >>> print(value(scaled_model.scaled_x))
        0.2
        >>> print(value(scaled_model.scaled_x.lb))
        -1.0
        >>> print(value(model.obj))
        101000000.0
        >>> print(value(scaled_model.scaled_obj))
        101.0

    """

    def __init__(self, **kwds):
        kwds['name'] = "scale_model"
        self._scaling_method = kwds.pop('scaling_method', 'user')
        self._suffix_finder = None
        super(ScaleModel, self).__init__(**kwds)

    def _create_using(self, original_model, **kwds):
        scaled_model = original_model.clone()
        self._apply_to(scaled_model, **kwds)
        return scaled_model

    def _apply_to(self, model, rename=True):
        # create a map of component to scaling factor
        component_scaling_factor_map = ComponentMap()
        self._suffix_finder = SuffixFinder('scaling_factor', 1.0, model)

        # if the scaling_method is 'user', get the scaling parameters from the suffixes
        if self._scaling_method == 'user':
            # get the scaling factors
            for c in model.component_data_objects(
                ctype=(Var, Constraint, Objective), descend_into=True
            ):
                component_scaling_factor_map[c] = self._suffix_finder.find(c)
        else:
            raise ValueError(
                "ScaleModel transformation: unknown scaling_method found"
                "-- supported values: 'user' "
            )

        if rename:
            # rename all the Vars, Constraints, and Objectives
            # from foo to scaled_foo
            component_list = list(
                model.component_objects(ctype=[Var, Constraint, Objective])
            )
            scaled_component_to_original_name_map = rename_components(
                model=model, component_list=component_list, prefix='scaled_'
            )
        else:
            scaled_component_to_original_name_map = ComponentMap(
                [
                    (comp, comp.name)
                    for comp in model.component_objects(
                        ctype=[Var, Constraint, Objective]
                    )
                ]
            )

        # scale the variable bounds and values and build the variable
        # substitution map for scaling vars in constraints
        variable_substitution_map = ComponentMap()
        already_scaled = set()
        for variable in [
            var for var in model.component_objects(ctype=Var, descend_into=True)
        ]:
            if variable.is_reference():
                # Skip any references - these should get picked up when
                # handling the actual variable
                continue

            # set the bounds/value for the scaled variable
            for k in variable:
                v = variable[k]
                if id(v) in already_scaled:
                    continue
                already_scaled.add(id(v))
                scaling_factor = component_scaling_factor_map[v]
                variable_substitution_map[v] = v / scaling_factor

                if v.lb is not None:
                    v.setlb(v.lb * scaling_factor)
                if v.ub is not None:
                    v.setub(v.ub * scaling_factor)
                if scaling_factor < 0:
                    temp = v.lb
                    v.setlb(v.ub)
                    v.setub(temp)

                if v.value is not None:
                    # Since the value was OK in the unscaled space, it
                    # should be safe to assume it is still valid in the
                    # scaled space)
                    v.set_value(value(v) * scaling_factor, skip_validation=True)

        # scale the objectives/constraints and perform the scaled variable substitution
        scale_constraint_dual = False
        if type(model.component('dual')) is Suffix:
            scale_constraint_dual = True

        # translate the variable_substitution_map (ComponentMap)
        # to variable_substitution_dict (key: id() of component)
        # ToDo: We should change replace_expressions to accept a ComponentMap as well
        variable_substitution_dict = {
            id(k): v for k, v in variable_substitution_map.items()
        }

        already_scaled = set()
        for component in model.component_objects(
            ctype=(Constraint, Objective), descend_into=True
        ):
            if component.is_reference():
                # Skip any references - these should get picked up when
                # handling the actual component
                continue

            for k in component:
                c = component[k]
                if id(c) in already_scaled:
                    continue
                already_scaled.add(id(c))
                # perform the constraint/objective scaling and variable sub
                scaling_factor = component_scaling_factor_map[c]
                if c.ctype is Constraint:
                    body = scaling_factor * replace_expressions(
                        expr=c.body,
                        substitution_map=variable_substitution_dict,
                        descend_into_named_expressions=True,
                        remove_named_expressions=True,
                    )

                    # scale the rhs
                    lower = c.lower
                    upper = c.upper
                    if lower is not None:
                        lower = lower * scaling_factor
                    if upper is not None:
                        upper = upper * scaling_factor

                    if scaling_factor < 0:
                        lower, upper = upper, lower

                    if scale_constraint_dual and c in model.dual:
                        dual_value = model.dual[c]
                        if dual_value is not None:
                            model.dual[c] = dual_value / scaling_factor

                    if c.equality:
                        c.set_value((lower, body))
                    else:
                        c.set_value((lower, body, upper))

                elif c.ctype is Objective:
                    c.expr = scaling_factor * replace_expressions(
                        expr=c.expr,
                        substitution_map=variable_substitution_dict,
                        descend_into_named_expressions=True,
                        remove_named_expressions=True,
                    )
                else:
                    raise NotImplementedError(
                        'Unknown object type found when applying scaling factors '
                        'in ScaleModel transformation - Internal Error'
                    )

        model.component_scaling_factor_map = component_scaling_factor_map
        model.scaled_component_to_original_name_map = (
            scaled_component_to_original_name_map
        )

        # Now that we have scaled the model, deactivate the relevant
        # scaling suffixes so that we don't accidentally (later)
        # double-scale.
        for s in self._suffix_finder.all_suffixes:
            s.deactivate()

        return model

    def propagate_solution(self, scaled_model, original_model):
        """This method takes the solution in scaled_model and maps it back to
        the original model.

        It will also transform duals and reduced costs if the suffixes
        'dual' and/or 'rc' are present.  The :code:`scaled_model`
        argument must be a model that was already scaled using this
        transformation as it expects data from the transformation to
        perform the back mapping.

        Parameters
        ----------
        scaled_model : Pyomo Model
           The model that was previously scaled with this transformation
        original_model : Pyomo Model
           The original unscaled source model

        """
        if not hasattr(scaled_model, 'component_scaling_factor_map'):
            raise AttributeError(
                'ScaleModel:propagate_solution called with scaled_model that does '
                'not have a component_scaling_factor_map. It is possible this '
                'method was called using a model that was not scaled with the '
                'ScaleModel transformation'
            )
        if not hasattr(scaled_model, 'scaled_component_to_original_name_map'):
            raise AttributeError(
                'ScaleModel:propagate_solution called with scaled_model that does '
                'not have a scaled_component_to_original_name_map. It is possible '
                'this method was called using a model that was not scaled with '
                'the ScaleModel transformation'
            )

        component_scaling_factor_map = scaled_model.component_scaling_factor_map
        scaled_component_to_original_name_map = (
            scaled_model.scaled_component_to_original_name_map
        )

        # transfer the variable values and reduced costs
        check_reduced_costs = type(scaled_model.component('rc')) is Suffix
        check_dual = (
            type(scaled_model.component('dual')) is Suffix
            and type(original_model.component('dual')) is Suffix
        )

        if check_reduced_costs or check_dual:
            # get the objective scaling factor
            scaled_objectives = list(
                scaled_model.component_data_objects(
                    ctype=Objective, active=True, descend_into=True
                )
            )
            if len(scaled_objectives) != 1:
                raise NotImplementedError(
                    'ScaleModel.propagate_solution requires a single active '
                    'objective function, but %d objectives found.'
                    % (len(scaled_objectives))
                )
            else:
                objective_scaling_factor = component_scaling_factor_map[
                    scaled_objectives[0]
                ]

        for scaled_v in scaled_model.component_objects(ctype=Var, descend_into=True):
            # get the unscaled_v from the original model
            original_v_path = scaled_component_to_original_name_map[scaled_v]
            # This will not work if decimal indices are present:
            original_v = original_model.find_component(original_v_path)

            for k in scaled_v:
                if scaled_v[k].value is None and original_v[k].value is not None:
                    logger.warning(
                        "Variable with value None in the scaled model is replacing"
                        f" value of variable {original_v[k].name} in the original"
                        f" model with None (was {original_v[k].value})."
                    )
                    original_v[k].set_value(None, skip_validation=True)
                elif scaled_v[k].value is not None:
                    original_v[k].set_value(
                        value(scaled_v[k]) / component_scaling_factor_map[scaled_v[k]],
                        skip_validation=True,
                    )
                if check_reduced_costs and scaled_v[k] in scaled_model.rc:
                    original_model.rc[original_v[k]] = (
                        scaled_model.rc[scaled_v[k]]
                        * component_scaling_factor_map[scaled_v[k]]
                        / objective_scaling_factor
                    )

        # transfer the duals
        if check_dual:
            for scaled_c in scaled_model.component_objects(
                ctype=Constraint, descend_into=True
            ):
                original_c = original_model.find_component(
                    scaled_component_to_original_name_map[scaled_c]
                )

                for k in scaled_c:
                    original_model.dual[original_c[k]] = (
                        scaled_model.dual[scaled_c[k]]
                        * component_scaling_factor_map[scaled_c[k]]
                        / objective_scaling_factor
                    )
