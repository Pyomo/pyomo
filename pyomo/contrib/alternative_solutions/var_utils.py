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

from pyomo.common.collections import ComponentSet
import pyomo.environ as pe
import pyomo.util.vars_from_expressions as vfe
import pyomo.contrib.alternative_solutions.aos_utils as aos_utils

"""
This file provides a collection of utilites for gathering and filtering 
variables from a model to support analysis of alternative solutions, and other 
related tasks.
"""

def _filter_model_variables(variable_set, var_generator, 
                            include_continuous=True, include_binary=True, 
                            include_integer=True, include_fixed=False):
    """Filters variables from a variable generator and adds them to a set."""
    for var in var_generator:
        if var in variable_set or var.is_fixed() and not include_fixed:
            continue
        if (var.is_continuous() and include_continuous or
                var.is_binary() and include_binary or
                var.is_integer() and include_integer):
            variable_set.add(var)

def get_model_variables(model, components='all', include_continuous=True, 
                        include_binary=True, include_integer=True, 
                        include_fixed=False):
    '''
    Gathers and returns all or a subset of varaibles from a Pyomo model.

        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model.
        components: 'all' or a collection Pyomo components
            The components from which variables should be collected. 'all' 
            indicates that all variables will be included. Alternatively, a 
            collection of Pyomo Blocks, Constraints, or Variables (indexed or
            non-indexed) from which variables will be gathered can be provided. 
            By default all variables in sub-Blocks will be added if a Block 
            element is provided. A tuple element with the format (Block, False) 
            indicates that only variables from the Block should be added but 
            not any of its sub-Blocks.
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
    '''

    # Validate inputs
    aos_utils._is_concrete_model(model)
    assert isinstance(include_continuous, bool), \
        'include_continuous must be a Boolean'
    assert isinstance(include_binary, bool), 'include_binary must be a Boolean'
    assert isinstance(include_integer, bool), \
        'include_integer must be a Boolean'
    assert isinstance(include_fixed, bool), 'include_fixed must be a Boolean'
    
    # Gather variables
    variable_set = ComponentSet()
    if components == 'all':
        var_generator = vfe.get_vars_from_components(model, pe.Constraint, 
                                                     include_fixed=\
                                                         include_fixed)
        _filter_model_variables(variable_set, var_generator, 
                                include_continuous, include_binary, 
                                include_integer, include_fixed)
    else:
        assert hasattr(components, '__iter__'), \
            ('components parameters must be an iterable collection of Pyomo'
             'objects'
            )
             
        for comp in components:
            if (hasattr(comp, 'ctype') and comp.ctype == pe.Block):
                blocks = comp.values() if comp.is_indexed() else (comp,)
                for item in blocks:
                    variables = vfe.get_vars_from_components(item, 
                         pe.Constraint, include_fixed=include_fixed)
                    _filter_model_variables(variable_set, variables, 
                        include_continuous, include_binary, include_integer, 
                        include_fixed)
            elif (isinstance(comp, tuple) and isinstance(comp[1], bool) and 
                  hasattr(comp[0], 'ctype') and comp[0].ctype == pe.Block):
                block = comp[0]
                descend_into = pe.Block if comp[1] else False
                blocks = block.values() if block.is_indexed() else (block,)
                for item in blocks:
                    variables = vfe.get_vars_from_components(item, 
                         pe.Constraint, include_fixed=include_fixed, 
                         descend_into=descend_into)
                    _filter_model_variables(variable_set, variables, 
                        include_continuous, include_binary, include_integer, 
                        include_fixed)   
            elif hasattr(comp, 'ctype') and comp.ctype == pe.Constraint:
                constraints = comp.values() if comp.is_indexed() else (comp,)
                for item in constraints:
                    variables = pe.expr.identify_variables(item.expr,
                                               include_fixed=include_fixed)
                    _filter_model_variables(variable_set, variables, 
                        include_continuous, include_binary, include_integer, 
                        include_fixed)   
            elif (hasattr(comp, 'ctype') and comp.ctype == pe.Var):
                variables = comp.values() if comp.is_indexed() else (comp,)
                _filter_model_variables(variable_set, variables, 
                    include_continuous, include_binary, include_integer, 
                    include_fixed)
            else:
                print(('No variables added for unrecognized component {}.').
                      format(comp))
                
    return variable_set

def check_variables(model, variables):
    pass