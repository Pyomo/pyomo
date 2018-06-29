#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ
from pyomo.version import version_info, __version__
import pyomo.opt
from pyomo.opt import (SolverFactory,
                       SolverStatus,
                       TerminationCondition)
from pyomo.kernel.util import (generate_names,
                               pprint)
from pyomo.core.kernel import *

# allow the use of standard kernel modeling components
# as the ctype argument for the general iterator method
from pyomo.core.kernel.component_interface import _convert_ctype
_convert_ctype[block] = \
    pyomo.core.kernel.component_block.IBlock
_convert_ctype[variable] = \
    pyomo.core.kernel.component_variable.IVariable
_convert_ctype[constraint] = \
    pyomo.core.kernel.component_constraint.IConstraint
_convert_ctype[parameter] = \
    pyomo.core.kernel.component_parameter.IParameter
_convert_ctype[expression] = \
    pyomo.core.kernel.component_expression.IExpression
_convert_ctype[objective] = \
    pyomo.core.kernel.component_objective.IObjective
_convert_ctype[sos] = \
    pyomo.core.kernel.component_sos.ISOS
_convert_ctype[suffix] = \
    pyomo.core.kernel.component_suffix.ISuffix
del _convert_ctype

#
#
# Hacks needed for this interface to work with Pyomo solvers
#
#

#
# Set up mappings between AML and Kernel ctypes
#

from pyomo.core.kernel.component_interface import _convert_ctype
_convert_ctype[pyomo.environ.Block] = \
    pyomo.core.kernel.component_block.IBlock
_convert_ctype[pyomo.environ.Var] = \
    pyomo.core.kernel.component_variable.IVariable
_convert_ctype[pyomo.environ.Constraint] = \
    pyomo.core.kernel.component_constraint.IConstraint
_convert_ctype[pyomo.environ.Param] = \
    pyomo.core.kernel.component_parameter.IParameter
_convert_ctype[pyomo.environ.Expression] = \
    pyomo.core.kernel.component_expression.IExpression
_convert_ctype[pyomo.environ.Objective] = \
    pyomo.core.kernel.component_objective.IObjective
_convert_ctype[pyomo.environ.SOSConstraint] = \
    pyomo.core.kernel.component_sos.ISOS
_convert_ctype[pyomo.environ.Suffix] = \
    pyomo.core.kernel.component_suffix.ISuffix
del _convert_ctype

#
# Ducktyping to work with a few solver interfaces
# Ideally, everything below here could be deleted one day
#

def _component_data_objects(self, *args, **kwds):
    # this is not yet handled
    kwds.pop('sort', None)
    for component in self.components(*args, **kwds):
        yield component
block.component_data_objects = _component_data_objects
del _component_data_objects

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

# This method no longer makes sense
def _component_objects(self, *args, **kwds):
    # this is not yet handled
    kwds.pop('sort', None)
    for component in self.components(*args, **kwds):
        yield component
block.component_objects = _component_objects
del _component_objects

# This method no longer makes sense
def _component(self, name):
    return getattr(self, name, None)
block.component = _component
del _component

# Note sure where this gets used or why we need it
def _valid_problem_types(self):
    import pyomo.opt
    return [pyomo.opt.base.ProblemFormat.pyomo]
block.valid_problem_types = _valid_problem_types
del _valid_problem_types

# canonical repn checks type instead of ctype
from pyomo.core.kernel.component_interface import _ICategorizedObjectMeta
_ICategorizedObjectMeta.type = _ICategorizedObjectMeta.ctype
del _ICategorizedObjectMeta

#
# Now cleanup the namespace a bit
#

import pyomo.core.kernel.component_piecewise.util as \
    piecewise_util
del component_interface
del component_map
del component_set
del component_dict
del component_tuple
del component_list
del component_block
del component_variable
del component_constraint
del component_objective
del component_expression
del component_parameter
del component_piecewise
del component_sos
del component_suffix
del component_matrix_constraint
del util
del pyomo
