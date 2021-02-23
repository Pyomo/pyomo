#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.current import ExpressionBase
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core import Constraint, Objective, TransformationFactory
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.util import sequence
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation


@TransformationFactory.register('core.remove_fixed_vars', doc="Create an equivalent model that omits all fixed variables.")
class EliminateFixedVars(IsomorphicTransformation):
    """
    Create an equivalent model that omits all fixed variables.
    """

    def __init__(self, **kwds):
        kwds['name'] = "eliminate_fixed_vars"
        super(EliminateFixedVars, self).__init__(**kwds)

    def _create_using(self, model, **kwds):
        #
        # Clone the model
        #
        M = model.clone()
        #
        # Iterate over the expressions in all objectives and constraints, replacing fixed
        # variables with their associated constants.
        #
        for ctype in [Objective, Constraint]:
            for obj in M.component_map(Objective).values():
                for name in obj:
                    if not obj[name].expr is None:
                        obj[name].expr = self._fix_vars(obj[name].expr, model)
        #
        # Iterate over variables, omitting those that have fixed values
        #
        ctr = 0
        for i in sequence(M.nvariables()):
            var = M.variable(i)
            del M._var[ i-1 ]
            if var.fixed:
                if var.is_binary():
                    M.statistics.number_of_binary_variables -= 1
                elif var.is_integer():
                    M.statistics.number_of_integer_variables -= 1
                elif var.is_continuous():
                    M.statistics.number_of_continuous_variables -= 1
                M.statistics.number_of_variables -= 1
                del M._label_var_map[ var.label ]
                del var.component()._data[ var.index ]
            else:
                M._var[ ctr ] = var
                var._old_id = var.id
                var.id = ctr
                ctr += 1
        return M

    def _fix_vars(self, expr, model):
        """ Walk through the S-expression, fixing variables. """
        # TODO - Change this to use a visitor pattern!
        if expr._args is None:
            return expr
        _args = []
        for i in range(len(expr._args)):
            if isinstance(expr._args[i],ExpressionBase):
                _args.append( self._fix_vars(expr._args[i], model) )
            elif (isinstance(expr._args[i],Var) or isinstance(expr._args[i],_VarData)) and expr._args[i].fixed:
                if expr._args[i].value != 0.0:
                    _args.append( as_numeric(expr._args[i].value) )
            else:
                _args.append( expr._args[i] )
        expr._args = _args
        return expr
