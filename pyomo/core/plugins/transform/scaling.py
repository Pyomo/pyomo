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
from pyomo.core.base import Var, Constraint, Objective, Param, _VarData, _ConstraintData, _ObjectiveData, Suffix, value, ComponentUID
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.kernel import ComponentMap

from pyomo.core.base import expr_coopr3
from pyomo.core.base.expr import clone_expression
#from pyomo.core.base import expr_common as common

def _substitute_var_in_constraint(constraint, substitution_map):
    _substitute_var(constraint._body, substitution_map)
    
def _substitute_var(expression, sub_component_map):
    stack = list()
    # this is a stack of the expression nodes to process
    stack.append(expression)

    while stack:
        e = stack.pop()
        e_type = type(e)
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
                        new_expr = clone_expression(sub_component_map[child])
                        e._args = _replace_entry_in_list_or_tuple(e._args, new_expr, i)
                else:
                    stack.append(child)

def _replace_entry_in_list_or_tuple(list_or_tuple, new_entry, i):
    if type(list_or_tuple) is tuple:
        return list_or_tuple[:i] + (new_entry,) + list_or_tuple[i+1:]
    else:
        list_or_tuple[i] = new_entry
        return list_or_tuple
    
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


    Examples:
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
        >>> print(value(scaled_model.x))
        0.2
        >>> print(value(scaled_model.x.lb))
        -1.0
        >>> print(value(model.obj))
        101000000.0
        >>> print(value(scaled_model.obj))
        101.0

    """

    alias('core.scale_model',\
          doc="Create a model where the variables and constraints are scaled by user-provided values.")

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
            raise ValueError("Suffix 'scaling_factor' has a value %s for component %s that cannot be converted to a float. "
                             "Floating point values are required for this suffix in the ScaleModel transformation."
                             % (scaling_factor, component_data))
        return scaling_factor

    def _apply_to(self, model, **kwds):
        # need to have a map of unscaled components to scaling factors
        # and a map of scaled components to unscaled components
        component_scaling_factor_map = ComponentMap()
        scaled_to_original_map = None
        
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

        # scale the variable bounds and values and build the
        # variable substitution map for scaling vars in constraints
        variable_substitution_map = ComponentMap()
        for variable in [var for var in model.component_objects(ctype=Var, descend_into=True)]:
            # set the bounds/value for the scaled variable
            for k in variable:
                v = variable[k]
                scaling_factor = component_scaling_factor_map[v]
                variable_substitution_map[v] = v/scaling_factor

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
            
        for component in model.component_objects(ctype=(Constraint,Objective), descend_into=True):
            for k in component:
                c = component[k]
                # perform the constraint/objective scaling and variable sub
                scaling_factor = component_scaling_factor_map[c]
                if isinstance(c, _ConstraintData):
                    _substitute_var_in_constraint(c, variable_substitution_map)

                    if c._lower is not None:
                        c._lower = c._lower * scaling_factor
                    if c._upper is not None:
                        c._upper = c._upper * scaling_factor

                    if scaling_factor < 0:
                        # swap inequalities
                        temp = c._lower
                        c._lower = c._upper
                        c._upper = temp

                    c._body = c._body * scaling_factor
                    
                    if scale_constraint_dual and c in model.dual:
                        dual_value = model.dual[c]
                        if dual_value is not None:
                            model.dual[c] = dual_value/scaling_factor
                    
                elif isinstance(c, _ObjectiveData):
                    _substitute_var(c.expr, variable_substitution_map)
                    c.expr = c.expr * scaling_factor
                else:
                    raise NotImplementedError('Unknown object type found when applying scaling factors in ScaleModel transformation - Internal Error')
                
        model.component_scaling_factor_map = component_scaling_factor_map

        return model

    def propagate_solution(self, scaled_model, original_model):
        '''
        This method takes the solution in scaled_model and maps it back to the original model.

        It will also transform duals and reduced costs if the suffixes 'dual' and/or 'rc' are present.
        The :code:`scaled_model` argument must be a model that was already scaled using this transformation
        as it expects data from the transformation to perform the back mapping.
        '''
        if not hasattr(scaled_model, 'component_scaling_factor_map'):
            raise AttributeError('ScaleModel:propagate_solution called with scaled_model that does not '
                                 'have a component_scaling_factor_map. It is possible this method was called '
                                 'using a model that was not scaled with the ScaleModel transformation')
        component_scaling_factor_map = scaled_model.component_scaling_factor_map

        # get the objective scaling factor
        objectives = list()
        for obj in scaled_model.component_data_objects(ctype=Objective, active=True, descend_into=True):
            objectives.append(obj)
        if len(objectives) != 1:
            raise NotImplementedError('ScaleModel.propagate_solution requires a single active objective function, but %d objectives found.' % (len(objectives)))
        scaled_objective = objectives[0]
        cuid = ComponentUID(scaled_objective)
        original_objective = cuid.find_component_on(original_model)
        objective_scaling_factor = component_scaling_factor_map[scaled_objective]
        
        # transfer the variable values and reduced costs
        check_reduced_costs = type(scaled_model.component('rc')) is Suffix
        for scaled_v in scaled_model.component_objects(ctype=Var, descend_into=True):
            # get the component UID of scaled_v and use it to get the unscaled_v from the original model
            cuid = ComponentUID(scaled_v)  # you probably want to use a cache / memo -- which is an arg to teh constructor - to mitigate the cost of looking up the index
            original_v = cuid.find_component_on(original_model)

            for k in scaled_v:
                original_v[k].value = value(scaled_v[k])/component_scaling_factor_map[scaled_v[k]]
                if check_reduced_costs and scaled_v[k] in scaled_model.rc:
                    original_model.rc[original_v[k]] = scaled_model.rc[scaled_v[k]]*component_scaling_factor_map[scaled_v[k]]/objective_scaling_factor

        # transfer the duals
        if type(scaled_model.component('dual')) is Suffix and type(original_model.component('dual')) is Suffix:
            for scaled_c in scaled_model.component_objects(ctype=Constraint, descend_into=True):
                cuid = ComponentUID(scaled_c)
                original_c = cuid.find_component_on(original_model)

                for k in scaled_c:
                    original_model.dual[original_c[k]] = scaled_model.dual[scaled_c[k]]*component_scaling_factor_map[scaled_c[k]]/objective_scaling_factor




