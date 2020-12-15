#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.version import version_info, __version__

#
# Load solver functionality
#
import pyomo.environ
import pyomo.opt
from pyomo.opt import (SolverFactory,
                       SolverStatus,
                       TerminationCondition)

#
# Define the modeling namespace
#
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.expr import (
    numvalue, numeric_expr, boolean_value, logical_expr, current,
    calculus, symbol_map, expr_errors, visitor, sympy_tools, taylor_series,
    expr_common, cnf_walker, template_expr
)


from pyomo.core.expr.numvalue import (
    value, is_constant, is_fixed, is_variable_type,
    is_potentially_variable, NumericValue, ZeroConstant,
    native_numeric_types, native_types, polynomial_degree,
)

from pyomo.core.expr.boolean_value import BooleanValue

from pyomo.core.expr.numeric_expr import linear_expression, nonlinear_expression

from pyomo.core.expr.logical_expr import (land, lor, equivalent, exactly,
                                          atleast, atmost, implies, lnot,
                                          xor, inequality)

from pyomo.core.expr.current import (
    log, log10, sin, cos, tan, cosh, sinh, tanh,
    asin, acos, atan, exp, sqrt, asinh, acosh,
    atanh, ceil, floor,
    Expr_if,
)

from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.taylor_series import taylor_series_expansion
import pyomo.core.kernel
from pyomo.kernel.util import (generate_names,
                               preorder_traversal,
                               pprint)
from pyomo.core.kernel.variable import \
    (variable,
     variable_tuple,
     variable_list,
     variable_dict)
from pyomo.core.kernel.constraint import \
    (constraint,
     linear_constraint,
     constraint_tuple,
     constraint_list,
     constraint_dict)
from pyomo.core.kernel.matrix_constraint import \
    matrix_constraint
import pyomo.core.kernel.conic as conic
from pyomo.core.kernel.parameter import \
    (parameter,
     functional_value,
     parameter_tuple,
     parameter_list,
     parameter_dict)
from pyomo.core.kernel.expression import \
    (noclone,
     expression,
     data_expression,
     expression_tuple,
     expression_list,
     expression_dict)
from pyomo.core.kernel.objective import \
    (maximize,
     minimize,
     objective,
     objective_tuple,
     objective_list,
     objective_dict)
from pyomo.core.kernel.sos import \
    (sos,
     sos1,
     sos2,
     sos_tuple,
     sos_list,
     sos_dict)
from pyomo.core.kernel.suffix import \
    (suffix,
     suffix_dict,
     export_suffix_generator,
     import_suffix_generator,
     local_suffix_generator,
     suffix_generator)
from pyomo.core.kernel.block import \
    (block,
     block_tuple,
     block_list,
     block_dict)
from pyomo.core.kernel.piecewise_library.transforms import \
    piecewise
from pyomo.core.kernel.piecewise_library.transforms_nd import \
    piecewise_nd
from pyomo.core.kernel.set_types import \
    (RealSet,
     IntegerSet,
     BooleanSet)
from pyomo.environ import (
     Reals,
     PositiveReals,
     NonPositiveReals,
     NegativeReals,
     NonNegativeReals,
     PercentFraction,
     UnitInterval,
     Integers,
     PositiveIntegers,
     NonPositiveIntegers,
     NegativeIntegers,
     NonNegativeIntegers,
     Boolean,
     Binary,
     RealInterval,
     IntegerInterval,
)
#
# allow the use of standard kernel modeling components
# as the ctype argument for the general iterator method
#

from pyomo.core.kernel.base import _convert_ctype
_convert_ctype[block] = \
    pyomo.core.kernel.block.IBlock
_convert_ctype[variable] = \
    pyomo.core.kernel.variable.IVariable
_convert_ctype[constraint] = \
    pyomo.core.kernel.constraint.IConstraint
_convert_ctype[parameter] = \
    pyomo.core.kernel.parameter.IParameter
_convert_ctype[expression] = \
    pyomo.core.kernel.expression.IExpression
_convert_ctype[objective] = \
    pyomo.core.kernel.objective.IObjective
_convert_ctype[sos] = \
    pyomo.core.kernel.sos.ISOS
_convert_ctype[suffix] = \
    pyomo.core.kernel.suffix.ISuffix
del _convert_ctype

#
#
# Hacks needed for this interface to work with Pyomo solvers
#
#

#
# Set up mappings between AML and Kernel ctypes
#

from pyomo.core.kernel.base import _convert_ctype
_convert_ctype[pyomo.environ.Block] = \
    pyomo.core.kernel.block.IBlock
_convert_ctype[pyomo.environ.Var] = \
    pyomo.core.kernel.variable.IVariable
_convert_ctype[pyomo.environ.Constraint] = \
    pyomo.core.kernel.constraint.IConstraint
_convert_ctype[pyomo.environ.Param] = \
    pyomo.core.kernel.parameter.IParameter
_convert_ctype[pyomo.environ.Expression] = \
    pyomo.core.kernel.expression.IExpression
_convert_ctype[pyomo.environ.Objective] = \
    pyomo.core.kernel.objective.IObjective
_convert_ctype[pyomo.environ.SOSConstraint] = \
    pyomo.core.kernel.sos.ISOS
_convert_ctype[pyomo.environ.Suffix] = \
    pyomo.core.kernel.suffix.ISuffix
del _convert_ctype

#
# Now cleanup the namespace a bit
#

import pyomo.core.kernel.piecewise_library.util as \
    piecewise_util
del util
del pyomo

#
# Ducktyping to work with a solver interfaces. Ideally,
# everything below here could be deleted one day.
#
from pyomo.core.kernel.heterogeneous_container import (heterogeneous_containers,
                                                       IHeterogeneousContainer)
def _component_data_objects(self, *args, **kwds):
    # this is not yet handled
    kwds.pop('sort', None)
    if 'active' not in kwds:
        kwds['active'] = None
    for component in self.components(*args, **kwds):
        yield component
IHeterogeneousContainer.component_data_objects = \
    _component_data_objects
del _component_data_objects

def _component_objects(self, *args, **kwds):
    # this is not yet handled
    kwds.pop('sort', None)
    # not handled
    assert kwds.pop('descent_order', None) is None
    active = kwds.pop('active', None)
    descend_into = kwds.pop('descend_into', True)
    for item in heterogeneous_containers(self,
                                         active=active,
                                         descend_into=descend_into):
        for child in item.children(*args, **kwds):
            yield child
IHeterogeneousContainer.component_objects = \
    _component_objects
del _component_objects
del IHeterogeneousContainer

def _block_data_objects(self, **kwds):
    # this is not yet handled
    kwds.pop('sort', None)
    active = kwds.get("active", None)
    assert active in (None, True)
    # if not active, then nothing below is active
    if (active is not None) and \
       (not self.active):
        return
    yield self
    for component in self.components(
            ctype=self.ctype,
            **kwds):
        yield component
block.block_data_objects = _block_data_objects
del _block_data_objects

# Note sure where this gets used or why we need it
def _valid_problem_types(self):
    import pyomo.opt
    return [pyomo.opt.base.ProblemFormat.pyomo]
block.valid_problem_types = _valid_problem_types
del _valid_problem_types

# update the reserved block attributes now that
# new hacked methods have been placed on blocks
block._refresh_block_reserved_words()
