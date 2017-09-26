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
from pyomo.core.base import Var, Constraint, Objective, Param, _VarData, _ConstraintData, _ObjectiveData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.kernel import ComponentMap

from pyomo.core.base import expr as EXPR
from pyomo.core.base import expr_common as common

def substitute_var(expression, sub_component_map):
    stack = list()
    # this is a stack of the expression nodes to process
    stack.append(expression)

    while stack:
        e = stack.pop()
        e_type = type(e)
        print(e_type, e)
        if e_type is EXPR._ProductExpression and common.mode is common.Mode.coopr3_trees:
            # _ProductExpression is fundamentally different in _Coopr3
            for i, child in enumerate(e._numerator):
                if isinstance(child, _VarData):
                    if child in sub_component_map:
                        e._numerator[i] = sub_component_map[i]
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
                        e._args[i] = sub_component_map[child]
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
        
        scaling_factor = instance.scaling_factor[component_data]
        try:
            scaling_factor = float(scaling_factor)
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

    def _default_dict(dictionary, index, default_value):
        if index in dictionary:
            return dictionary[index]
        return default_value
    
    def _create_using(self, model, **kwds):
        assert not kwds
        instance = model.clone()

        # if the scaling_method is 'user', get the scaling parameters from the suffixes
        component_scaling_factor_map = ComponentMap()
        if self._scaling_method == 'user':
            # perform some checks to make sure we have the necessary suffixes
            if not hasattr(instance, 'scaling_factor'):
                raise ValueError("ScaleModel transformation called with scaling_method='user'"
                                 ", but cannot find the suffix 'scaling_factor' on the model")

            # get the scaling factors
            for c in instance.component_data_objects(ctype=(Var,Constraint,Objective), descend_into=True):
                scaling_factor = self._get_float_scaling_factor(instance, c)
                component_scaling_factor_map[c] = scaling_factor

        else:
            raise ValueError("ScaleModel transformation: unknown scaling_method found"
                             "-- supported values: 'user' ")

        """
        # loop through constraints, vars and objectives and create the necessary param objects
        # to store the scaling factors
        def _scaling_factor_param_rule(parent, component, index, scaling_factor_component_map):
            return scaling_factor_component_map[component[index]]
        component_scaling_parameters_map = ComponentMap()
        for c in instance.component_objects(ctype=(Var, Constraint, Objective), descend_into=True):
            # create the parameters to store the scaling factors
            param_name = '_scaling_factor_' + c.local_name
            if c.is_indexed():
                _lambda_scaling_factor_param_rule = lambda model, index: _scaling_factor_param_rule(model, c, index, component_scaling_factor_map)
                c.parent_block().add_component(param_name, Param(c._index,
                                                                 rule=_lambda_scaling_factor_param_rule,
                                                                 default=1,
                                                                 mutable=False))
            else:
                c.parent_block().add_component(param_name, Param(initialize=component_scaling_factor_map[c], mutable=False))
            component_scaling_parameters_map[c] = getattr(c.parent_block(), param_name)
        """
        
        # loop through all the variables and build the scaled variables and
        # the substitution map
        variable_substitution_map = ComponentMap()
        for v in list(instance.component_objects(ctype=Var, descend_into=True)):
            new_v = None
            if v.is_indexed():
                new_v = Var(v._index)
            else:
                new_v = Var()
                
            for k in new_v:
                new_v_data = new_v[k]
                scaling_factor = component_scaling_factor_map[v]
                variable_substitution_map[v] = new_v_data/scaling_factor
                new_v_data.setlb(v.lb * scaling_factor)
                new_v_data.setub(v.ub * scaling_factor)
                if pe.value(scaling_factor) < 0:
                    temp = new_v_data.lb
                    new_v_data.setlb(new_v_data.ub)
                    new_v_data.setub(temp)
                new_v_data.value = pe.value(v) * pe.value(scaling_factor)
            setattr(v.parent_block(), '_scaled_variable_' + v.local_name, new_v)

        # loop through all the constraint and objectives, perform substitution
        # and apply scaling factors
        for component in instance.component_objects(ctype=(Constraint,Objective), descend_into=True):
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
