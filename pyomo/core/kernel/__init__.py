#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyomo.core.kernel.log_config
from  pyomo.core.kernel.expr import *
import pyomo.core.kernel.register_numpy_types
import pyomo.core.kernel.component_interface

import pyomo.opt
from pyomo.opt import (SolverFactory,
                       SolverStatus,
                       TerminationCondition)
import pyomo.opt.base
from pyomo.core.kernel.component_map import ComponentMap
import pyomo.core.kernel.component_block
from pyomo.core.kernel.component_block import (block,
                                               tiny_block,
                                               block_tuple,
                                               block_list,
                                               block_dict)
import pyomo.core.kernel.component_variable
from pyomo.core.kernel.component_variable import (variable,
                                                  variable_tuple,
                                                  variable_list,
                                                  variable_dict)
import pyomo.core.kernel.component_constraint
from pyomo.core.kernel.component_constraint import (constraint,
                                                    linear_constraint,
                                                    constraint_tuple,
                                                    constraint_list,
                                                    constraint_dict)
import pyomo.core.kernel.component_parameter
from pyomo.core.kernel.component_parameter import (parameter,
                                                   parameter_tuple,
                                                   parameter_list,
                                                   parameter_dict)
import pyomo.core.kernel.component_expression
from pyomo.core.kernel.component_expression import (expression,
                                                    data_expression,
                                                    expression_tuple,
                                                    expression_list,
                                                    expression_dict)
import pyomo.core.kernel.component_objective
from pyomo.core.kernel.component_objective import (maximize,
                                                   minimize,
                                                   objective,
                                                   objective_tuple,
                                                   objective_list,
                                                   objective_dict)
import pyomo.core.kernel.component_sos
from pyomo.core.kernel.component_sos import (sos,
                                             sos1,
                                             sos2,
                                             sos_tuple,
                                             sos_list,
                                             sos_dict)
import pyomo.core.kernel.component_suffix
from pyomo.core.kernel.component_suffix import (suffix,
                                                export_suffix_generator,
                                                import_suffix_generator,
                                                local_suffix_generator,
                                                suffix_generator)
import pyomo.core.kernel.component_piecewise
import pyomo.core.kernel.component_piecewise.util
import pyomo.core.kernel.component_piecewise.transforms
import pyomo.core.kernel.component_piecewise.transforms_nd
from pyomo.core.kernel.component_piecewise.transforms import piecewise
from pyomo.core.kernel.component_piecewise.transforms_nd import piecewise_nd

import pyomo.core.kernel.set_types
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet,
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
                                         IntegerInterval)
from pyomo.core.kernel.numvalue import value

# Short term helper method for debugging models
def _pprint(obj, indent=0):
    import pyomo.core.base
    if not isinstance(obj, ICategorizedObject):
        import pprint
        assert indent == 0
        pprint.pprint(obj, indent=indent+1)
        return
    if not obj._is_component:
        # a container but not a block
        assert obj._is_container
        print((" "*indent)+" - %s: container(size=%s, ctype=%s)"
              % (str(obj), len(obj), obj.ctype.__name__))
        for c in obj.children():
            _pprint(c, indent=indent+1)
    elif not obj._is_container:
        # not a block
        if obj.ctype is pyomo.core.base.Var:
            print((" "*indent)+" - %s: variable(value=%s, lb=%s, ub=%s, domain_type=%s, fixed=%s)"
                  % (str(obj),
                     obj.value,
                     obj.lb,
                     obj.ub,
                     obj.domain_type.__name__,
                     obj.fixed))
        elif obj.ctype is pyomo.core.base.Constraint:
              print((" "*indent)+" - %s: constraint(expr=%s)"
                  % (str(obj),
                     str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Objective:
            print((" "*indent)+" - %s: objective(expr=%s)"
                  % (str(obj), str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Expression:
            print((" "*indent)+" - %s: expression(expr=%s)"
                  % (str(obj), str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Param:
            print((" "*indent)+" - %s: parameter(value=%s)"
                  % (str(obj), str(obj.value)))
        elif obj.ctype is pyomo.core.base.SOSConstraint:
            print((" "*indent)+" - %s: sos(level=%s, entries=%s)"
                  % (str(obj),
                     obj.level,
                     str(["(%s,%s)" % (str(v), w)
                          for v,w in zip(obj.variables,
                                         obj.weights)])))
        else:
            assert obj.ctype is pyomo.core.base.Suffix
            print((" "*indent)+" - %s: suffix(size=%s)"
                  % (str(obj), str(len(obj))))
    else:
        # a block
        for block in obj.blocks():
            print("")
            print((" "*indent)+"block: %s" % (str(block)))
            ctypes = block.collect_ctypes(descend_into=False)
            for ctype in sorted(ctypes,
                                key=lambda x: str(x)):
                print((" "*indent)+'ctype='+ctype.__name__+" declarations:")
                for c in block.children(ctype=ctype):
                    if ctype is pyomo.core.base.Block:
                        if c._is_component:
                            print((" "*indent)+"  - %s: block(children=%s)"
                                  % (str(c), len(list(c.children()))))
                        else:
                            print((" "*indent)+"  - %s: block_container(size=%s)"
                                  % (str(c), len(list(c))))

                    else:
                        _pprint(c, indent=indent+1)

#
# Collecting all of the hacks that needed to be added into
# this interface in order to work with the Pyomo solver
# interface into the code below:
#

#
#
# Ducktyping to work with a few solver interfaces
#
from pyomo.core.kernel.component_block import _block_base

# This is ugly and bad (keys are local names
# so they can overwrite each other). Not sure
# how the actual method in block.py is supposed to
# behave, but I think we can move away from using
# this method in most places.
def _component_map(self, *args, **kwds):
    import six
    kwds['return_key'] = True
    kwds['include_parent_blocks'] = False
    traversal = self.preorder_traversal(*args, **kwds)
    d = {}
    for key, obj in traversal:
        if obj._is_component:
            d[key] = obj
    return d
_block_base.component_map = _component_map
del _component_map

def _component_data_objects(self, *args, **kwds):
    # this is not yet handled
    kwds.pop('sort', None)
    for component in self.components(*args, **kwds):
        yield component
_block_base.component_data_objects = _component_data_objects
del _component_data_objects

def _block_data_objects(self, *args, **kwds):
    # this is not yet handled
    kwds.pop('sort', None)
    for component in self.blocks(*args, **kwds):
        yield component
_block_base.block_data_objects = _block_data_objects
del _block_data_objects

# This method no longer makes sense
def _component_objects(self, *args, **kwds):
    # this is not yet handled
    kwds.pop('sort', None)
    for component in self.components(*args, **kwds):
        yield component
_block_base.component_objects = _component_objects
del _component_objects

# This method no longer makes sense
def _component(self, name):
    return getattr(self, name, None)
_block_base.component = _component
del _component

# Note sure where this gets used or why we need it
def _valid_problem_types(self):
    return [pyomo.opt.base.ProblemFormat.pyomo]
_block_base.valid_problem_types = _valid_problem_types
del _valid_problem_types

# canonical repn checks type instead of ctype
from pyomo.core.kernel.component_interface import _ICategorizedObjectMeta
_ICategorizedObjectMeta.type = _ICategorizedObjectMeta.ctype
