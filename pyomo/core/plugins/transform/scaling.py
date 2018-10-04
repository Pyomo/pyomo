#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Var, Constraint, Objective, _ConstraintData, _ObjectiveData, Suffix, value
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.util.components import rename_components
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.current import replace_expressions


@TransformationFactory.register('core.scale_model',
                                doc="Scale model variables, constraints, and objectives.")
class ScaleModel(Transformation):
    """
    Transformation to scale a model.

    This plugin performs variable, constraint, and objective scaling on
    a model based on the scaling factors in the suffix 'scaling_parameter'
    set for the variables, constraints, and/or objective. This is typically 
    done to scale the problem for improved numerical properties.

    Supported transformation methods:
        * :py:meth:`apply_to <pyomo.core.plugins.transform.scaling.ScaleModel.apply_to>`
        * :py:meth:`create_using <pyomo.core.plugins.transform.scaling.ScaleModel.create_using>`
        * :py:meth:`propagate_solution <pyomo.core.plugins.transform.scaling.ScaleModel.propagate_solution>`


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

    ToDo
    ====
    - implement an option to change the variables names or not

    """

    def __init__(self, **kwds):
        kwds['name'] = "scale_model"
        self._scaling_method = kwds.pop('scaling_method', 'user')
        super(ScaleModel, self).__init__(**kwds)

    def _create_using(self, original_model, **kwds):
        scaled_model = original_model.clone()
        self._apply_to(scaled_model, **kwds)
        return scaled_model

    def _get_float_scaling_factor(self, instance, component_data):
        scaling_factor = None
        if component_data in instance.scaling_factor:
            scaling_factor = instance.scaling_factor[component_data]
        elif component_data.parent_component() in instance.scaling_factor:
            scaling_factor = instance.scaling_factor[component_data.parent_component()]

        if scaling_factor is None:
            return 1.0

        try:
            scaling_factor = float(scaling_factor)
        except ValueError:
            raise ValueError(
                "Suffix 'scaling_factor' has a value %s for component %s that cannot be converted to a float. "
                "Floating point values are required for this suffix in the ScaleModel transformation."
                % (scaling_factor, component_data))
        return scaling_factor

    def _apply_to(self, model, **kwds):
        # create a map of component to scaling factor
        component_scaling_factor_map = ComponentMap()

        # if the scaling_method is 'user', get the scaling parameters from the suffixes
        if self._scaling_method == 'user':
            # perform some checks to make sure we have the necessary suffixes
            if type(model.component('scaling_factor')) is not Suffix:
                raise ValueError("ScaleModel transformation called with scaling_method='user'"
                                 ", but cannot find the suffix 'scaling_factor' on the model")

            # get the scaling factors
            for c in model.component_data_objects(ctype=(Var, Constraint, Objective), descend_into=True):
                component_scaling_factor_map[c] = self._get_float_scaling_factor(model, c)
        else:
            raise ValueError("ScaleModel transformation: unknown scaling_method found"
                             "-- supported values: 'user' ")

        # rename all the Vars, Constraints, and Objectives from foo to scaled_foo
        scaled_component_to_original_name_map = \
            rename_components(model=model,
                              component_list=list(model.component_objects(ctype=[Var, Constraint, Objective])),
                              prefix='scaled_')

        # scale the variable bounds and values and build the variable substitution map
        # for scaling vars in constraints
        variable_substitution_map = ComponentMap()
        for variable in [var for var in model.component_objects(ctype=Var, descend_into=True)]:
            # set the bounds/value for the scaled variable
            for k in variable:
                v = variable[k]
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
                    v.value = value(v) * scaling_factor

        # scale the objectives/constraints and perform the scaled variable substitution
        scale_constraint_dual = False
        if type(model.component('dual')) is Suffix:
            scale_constraint_dual = True

        # translate the variable_substitution_map (ComponentMap)
        # to variable_substition_dict (key: id() of component)
        # ToDo: We should change replace_expressions to accept a ComponentMap as well
        variable_substitution_dict = dict()
        for k in variable_substitution_map:
            variable_substitution_dict[id(k)] = variable_substitution_map[k]

        for component in model.component_objects(ctype=(Constraint, Objective), descend_into=True):
            for k in component:
                c = component[k]
                # perform the constraint/objective scaling and variable sub
                scaling_factor = component_scaling_factor_map[c]
                if isinstance(c, _ConstraintData):
                    body = scaling_factor * \
                           replace_expressions(expr=c.body,
                                               substitution_map=variable_substitution_dict,
                                               descend_into_named_expressions=True,
                                               remove_named_expressions=True)

                    # scale the rhs
                    if c._lower is not None:
                        c._lower = c._lower * scaling_factor
                    if c._upper is not None:
                        c._upper = c._upper * scaling_factor

                    if scaling_factor < 0:
                        c._lower, c._upper = c._upper, c._lower

                    if scale_constraint_dual and c in model.dual:
                        dual_value = model.dual[c]
                        if dual_value is not None:
                            model.dual[c] = dual_value / scaling_factor

                    c.set_value((c._lower, body, c._upper))

                elif isinstance(c, _ObjectiveData):
                    c.expr = scaling_factor * \
                             replace_expressions(expr=c.expr,
                                                 substitution_map=variable_substitution_dict,
                                                 descend_into_named_expressions=True,
                                                 remove_named_expressions=True)
                else:
                    raise NotImplementedError(
                        'Unknown object type found when applying scaling factors in ScaleModel transformation - Internal Error')

        model.component_scaling_factor_map = component_scaling_factor_map
        model.scaled_component_to_original_name_map = scaled_component_to_original_name_map

        return model

    def propagate_solution(self, scaled_model, original_model):
        """
        This method takes the solution in scaled_model and maps it back to the original model.

        It will also transform duals and reduced costs if the suffixes 'dual' and/or 'rc' are present.
        The :code:`scaled_model` argument must be a model that was already scaled using this transformation
        as it expects data from the transformation to perform the back mapping.

        Parameters
        ----------
        scaled_model : Pyomo Model
           The model that was previously scaled with this transformation
        original_model : Pyomo Model
           The original unscaled source model

        """
        if not hasattr(scaled_model, 'component_scaling_factor_map'):
            raise AttributeError('ScaleModel:propagate_solution called with scaled_model that does not '
                                 'have a component_scaling_factor_map. It is possible this method was called '
                                 'using a model that was not scaled with the ScaleModel transformation')
        if not hasattr(scaled_model, 'scaled_component_to_original_name_map'):
            raise AttributeError('ScaleModel:propagate_solution called with scaled_model that does not '
                                 'have a scaled_component_to_original_name_map. It is possible this method was called '
                                 'using a model that was not scaled with the ScaleModel transformation')

        component_scaling_factor_map = scaled_model.component_scaling_factor_map
        scaled_component_to_original_name_map = scaled_model.scaled_component_to_original_name_map

        # get the objective scaling factor
        scaled_objectives = list(scaled_model.component_data_objects(ctype=Objective, active=True, descend_into=True))
        if len(scaled_objectives) != 1:
            raise NotImplementedError(
                'ScaleModel.propagate_solution requires a single active objective function, but %d objectives found.' % (
                    len(objectives)))
        objective_scaling_factor = component_scaling_factor_map[scaled_objectives[0]]

        # transfer the variable values and reduced costs
        check_reduced_costs = type(scaled_model.component('rc')) is Suffix
        for scaled_v in scaled_model.component_objects(ctype=Var, descend_into=True):
            # get the unscaled_v from the original model
            original_v_path = scaled_component_to_original_name_map[scaled_v]
            original_v = original_model.find_component(original_v_path)

            for k in scaled_v:
                original_v[k].value = value(scaled_v[k]) / component_scaling_factor_map[scaled_v[k]]
                if check_reduced_costs and scaled_v[k] in scaled_model.rc:
                    original_model.rc[original_v[k]] = scaled_model.rc[scaled_v[k]] * component_scaling_factor_map[
                        scaled_v[k]] / objective_scaling_factor

        # transfer the duals
        if type(scaled_model.component('dual')) is Suffix and type(original_model.component('dual')) is Suffix:
            for scaled_c in scaled_model.component_objects(ctype=Constraint, descend_into=True):
                original_c = original_model.find_component(scaled_component_to_original_name_map[scaled_c])

                for k in scaled_c:
                    original_model.dual[original_c[k]] = scaled_model.dual[scaled_c[k]] * component_scaling_factor_map[
                        scaled_c[k]] / objective_scaling_factor
