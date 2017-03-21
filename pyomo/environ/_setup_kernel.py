#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# This is a short term hack that to set up the ctype
# property objects in pyomo/core/kernel while preventing a
# circular dependency between the core/base and core/kernel
# directories.

def _setup():
    from pyomo.core.base import (Var,
                                 Constraint,
                                 Objective,
                                 Block,
                                 Param,
                                 Expression,
                                 SOSConstraint,
                                 Suffix)

    from pyomo.core.kernel.component_block import (block,
                                                   tiny_block,
                                                   block_tuple,
                                                   block_list,
                                                   block_dict)
    from pyomo.core.kernel.component_variable import (variable,
                                                      variable_tuple,
                                                      variable_list,
                                                      variable_dict)
    from pyomo.core.kernel.component_constraint import (constraint,
                                                        linear_constraint,
                                                        constraint_tuple,
                                                        constraint_list,
                                                        constraint_dict)
    from pyomo.core.kernel.component_parameter import (parameter,
                                                       parameter_tuple,
                                                       parameter_list,
                                                       parameter_dict)
    from pyomo.core.kernel.component_expression import (expression,
                                                        data_expression,
                                                        expression_tuple,
                                                        expression_list,
                                                        expression_dict)
    from pyomo.core.kernel.component_objective import (objective,
                                                       objective_tuple,
                                                       objective_list,
                                                       objective_dict)
    from pyomo.core.kernel.component_sos import (sos,
                                                 sos1,
                                                 sos2,
                                                 sos_tuple,
                                                 sos_list,
                                                 sos_dict)
    from pyomo.core.kernel.component_suffix import (suffix,
                                                    export_suffix_generator,
                                                    import_suffix_generator,
                                                    local_suffix_generator,
                                                    suffix_generator)
    


    #
    # setup ctypes
    #

    variable._ctype = Var
    variable_tuple._ctype = Var
    variable_list._ctype = Var
    variable_dict._ctype = Var

    block._ctype = Block
    block_tuple._ctype = Block
    block_list._ctype = Block
    block_dict._ctype = Block
    tiny_block._ctype = Block

    constraint._ctype = Constraint
    linear_constraint._ctype = Constraint
    constraint_tuple._ctype = Constraint
    constraint_list._ctype = Constraint
    constraint_dict._ctype = Constraint

    parameter._ctype = Param
    parameter_tuple._ctype = Param
    parameter_list._ctype = Param
    parameter_dict._ctype = Param

    objective._ctype = Objective
    objective_tuple._ctype = Objective
    objective_list._ctype = Objective
    objective_dict._ctype = Objective

    expression._ctype = Expression
    expression_tuple._ctype = Expression
    expression_list._ctype = Expression
    expression_dict._ctype = Expression

    sos._ctype = SOSConstraint
    sos_tuple._ctype = SOSConstraint
    sos_list._ctype = SOSConstraint
    sos_dict._ctype = SOSConstraint

    suffix._ctype = Suffix
