#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.util.plugin import alias
from pyomo.core.base import Var, Constraint, Objective, Param, _VarData, _ConstraintData, _ObjectiveData, Suffix
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.kernel import ComponentMap

from pyomo.core.base import expr_coopr3, clone_expression
#from pyomo.core.base import expr_common as common


def substitute_var(expression, sub_component_map):
    stack = list()
    # this is a stack of the expression nodes to process
    stack.append(expression)

    while stack:
        e = stack.pop()
        e_type = type(e)
        print(e_type, e)
        if e_type is expr_coopr3._ProductExpression:
            # _ProductExpression is fundamentally different in _Coopr3
            for i, child in enumerate(e._numerator):
                ## TODO: don't need the isinstance... just check in sub_component_map
                if isinstance(child, _VarData):
                    if child in sub_component_map:
                        e._numerator[i] = clone_expression(sub_component_map[i])
                else:
                    stack.append(child)
            for i, child in enumerate(e._denominator):
                if isinstance(child, _VarData):
                    if child in sub_component_map:
                        e._denominator[i] = sub_component_map[i]
                else:
                    stack.append(child)
        elif hasattr(e, '_args'):
            for i, child in enumerate(e._args):
                if isinstance(child, _VarData):
                    if child in sub_component_map:
                        e._args[i] = clone_expression(sub_component_map[child])
                else:
                    stack.append(child)


class ScaleModel(Transformation):
    """
    This plugin performs variable, constraint, and objective scaling. Scaling parameters
    must be in the suffix 'scaling_parameter'.

    Given an example:
       min obj: x^2 + y
        s.t.  c1: x + y = 4
              0.5 <= x
              0 <= y <= 8
    
    and a set of scaling parameters: 
       k_obj = 5.0
       k_x = 10.0
       k_y = 3.0
       k_c1 = 2.0

    the new problem becomes:
       min obj: k_obj * (scaled_x/k_x)^2 + scaled_y/k_y
        s.t.  c1: k_c1 * (scaled_x/k_x + scaled_y/k_y) = k_c1 * 4
              0.5 * k_x <= scaled_x
              0.0 * k_y <= scaled_y <= k_y * 8

    where:
       scaled_x is defined as k_x * x
       scaled_y is defined as k_y * y

    """

    alias('core.scale_model',\
          doc="Create a model where the variables and constraints are scaled by user-provided values.")

    def __init__(self, **kwds):
        kwds['name'] = "scale_model"
        self._scaling_method = kwds.pop('scaling_method', 'user')
        super(ScaleModel, self).__init__(**kwds)

    def _apply_to(self, instance, **kwds):
        raise NotImplementedError('in-place transormations not supported for ScaleModel transformation')
    
    def _get_float_scaling_factor(self, instance, component_data):
        if component_data not in instance.scaling_factor or instance.scaling_factor[component_data] is None:
            return 1.0
        
        try:
            scaling_factor = float(instance.scaling_factor[component_data])
        except ValueError:
            raise ValueError("Suffix 'scaling_factor' has a value %s for component %s that cannot be converted to a float. Floating point values are required for this suffix in the ScaleModel transformation." % (instance.scaling_factor[component_data], component_data))
        return scaling_factor

    def _push_transformation_data_to_model(self, instance, transformation_name, component_data_scaling):
        if not hasattr(instance, 'transformation_data') or transformation_name not in instance.transformation_data:
            instance.transformation_data = {transformation_name: [component_data_scaling]}
        else:
            instance.transformation_data[transformation_name].append(component_data_scaling)

    def _pop_transformation_data_from_model(self, instance):
        if not hasattr(instance, 'transformation_data') or transformation_name not in instance.transformation_data:
            raise AttributeError("Error: no transformation_data exists for %s."
                                 " It is possible that you are reversing a transformation without having"
                                 " first done a forward transformation." % (transformation_name))
        return instance.transformation_data[transformation_name].pop()

    def _apply_to(self, model, **kwds):
        # need to have a map of unscaled components to scaling factors
        # and a map of scaled components to unscaled components
        component_scaling_factor_map = ComponentMap()
        transformed_to_original_component_map = ComponentMap()
        
        # if the scaling_method is 'user', get the scaling parameters from the suffixes
        if self._scaling_method == 'user':
            # perform some checks to make sure we have the necessary suffixes
            if type(model.component('scaling_factor')) is not Suffix:
                raise ValueError("ScaleModel transformation called with scaling_method='user'"
                                 ", but cannot find the suffix 'scaling_factor' on the model")

            # get the scaling factors
            for c in model.component_data_objects(ctype=(Var,Constraint,Objective), descend_into=True):
                component_scaling_factor_map[c] = self._get_float_scaling_factor(model, c)

        else:
            raise ValueError("ScaleModel transformation: unknown scaling_method found"
                             "-- supported values: 'user' ")

        # build the scaled variables and the variable substitution map
        # for replacing unscaled with scaled vars
        variable_substitution_map = ComponentMap()
        unscaled_variables = [v for v in model.component_objects(ctype=Var, descend_into=True)]
        for v in unscaled_variables:
            # create the scaled variable
            scaled_v = None
            if v.is_indexed():
                scaled_v = Var(v.index())
            else:
                scaled_v = Var()

            # add the scaled variable to the model
            setattr(v.parent_block(), '_scaled_variable_' + v.local_name, new_v)

            # set the bounds/value for the scaled variable
            # and extend the component maps
            for k in scaled_v.index():
                scaled_v_data = scale_v[k]
                scaling_factor = component_scaling_factor_map[v[k]]
                transformed_to_original_component_map[scaled_v_data] = v[k]
                variable_substitution_map[v[k]] = scaled_v_data/scaling_factor
                
                scaled_v_data.setlb(v.lb * scaling_factor)
                scaled_v_data.setub(v.ub * scaling_factor)
                if pe.value(scaling_factor) < 0:
                    temp = scaled_v_data.lb
                    scaled_v_data.setlb(new_v_data.ub)
                    scaled_v_data.setub(temp)
                scaled_v_data.value = pe.value(v) * pe.value(scaling_factor)

        # create the scaled constraints
        CRASH HERE!
        Write a loop to create a copy of the scaled constraints....

        
        # loop through all the constraint and objectives, perform substitution
        # and apply scaling factors
        for component in model.component_objects(ctype=(Constraint,Objective), descend_into=True):
            for k in component:
                c = component[k]
                # perform the constraint/objective scaling and variable sub
                scaling_factor = component_scaling_factor_map[c]
                if isinstance(c, _ConstraintData):
                    substitute_var(c._body, variable_substitution_map)

                    if c._lower is not None:
                        c._lower = c._lower * scaling_factor
                    if c._upper is not None:
                        c._upper = c._upper * scaling_factor

                    if scaling_factor < 0:
                        # swap inequalities
                        temp = c._lower
                        c._lower = c._upper
                        c._upper = temp

                    c._body = c.body * scaling_factor
                elif isinstance(c, _ObjectiveData):
                    substitute_var(c.expr, variable_substitution_map)
                    c.expr = c.expr * scaling_factor
                else:
                    raise NotImplementedError('Unknown object type found when applying scaling factors in ScaleModel transformation - Internal Error')

        # store the scaling factors for solution propagation
        self._push_transformation_data_to_model(instance, 'ScaleModel', component_scaling_factor_map)

        return instance


    def propagate_solution(scaled_model, original_model, primal_variables=True, dual_variables=True):
        '''
        This method takes the solution in the scaled model and reverses the transformation to put
        the solution in the original model. It will do primal, dual, or both depending on keyword arguments.
        '''

        # 
        
