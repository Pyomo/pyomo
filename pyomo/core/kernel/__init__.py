#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
from pyomo.core.kernel.component_set import ComponentSet
import pyomo.core.kernel.component_block
from pyomo.core.kernel.component_block import (block,
                                               tiny_block,
                                               block_tuple,
                                               block_list,
                                               block_dict)
import pyomo.core.kernel.component_variable
from pyomo.core.kernel.component_variable import (variable,
                                                  variable_tuple,
                                                  create_variable_tuple,
                                                  variable_list,
                                                  create_variable_list,
                                                  variable_dict,
                                                  create_variable_dict)
import pyomo.core.kernel.component_constraint
from pyomo.core.kernel.component_constraint import (constraint,
                                                    linear_constraint,
                                                    constraint_tuple,
                                                    constraint_list,
                                                    constraint_dict)
import pyomo.core.kernel.component_matrix_constraint
from pyomo.core.kernel.component_matrix_constraint import \
    matrix_constraint
import pyomo.core.kernel.component_parameter
from pyomo.core.kernel.component_parameter import (parameter,
                                                   parameter_tuple,
                                                   parameter_list,
                                                   parameter_dict)
import pyomo.core.kernel.component_expression
from pyomo.core.kernel.component_expression import (noclone,
                                                    expression,
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
import pprint as _pprint_
import six
def pprint(obj, indent=0):
    """pprint a modeling object"""
    import pyomo.core.base
    if not isinstance(obj, pyomo.core.kernel.component_interface.ICategorizedObject):
        if isinstance(obj, pyomo.core.kernel.numvalue.NumericValue):
            prefix = ""
            if indent > 0:
                prefix = (" "*indent)+" - "
            print(prefix+str(obj))
        else:
            assert indent == 0
            _pprint_.pprint(obj, indent=indent+1)
        return
    if not obj._is_component:
        # a container but not a block
        assert obj._is_container
        prefix = ""
        if indent > 0:
            prefix = (" "*indent)+" - "
        print(prefix+"%s: container(size=%s, ctype=%s)"
              % (str(obj), len(obj), obj.ctype.__name__))
        for c in obj.children():
            pprint(c, indent=indent+1)
    elif not obj._is_container:
        prefix = ""
        if indent > 0:
            prefix = (" "*indent)+" - "
        # not a block
        clsname = obj.__class__.__name__
        if obj.ctype is pyomo.core.base.Var:
            print(prefix+"%s: %s(value=%s, bounds=(%s,%s), domain_type=%s, fixed=%s, stale=%s)"
                  % (str(obj),
                     clsname,
                     obj.value,
                     obj.lb,
                     obj.ub,
                     obj.domain_type.__name__,
                     obj.fixed,
                     obj.stale))
        elif obj.ctype is pyomo.core.base.Constraint:
              print(prefix+"%s: %s(active=%s, expr=%s)"
                  % (str(obj),
                     clsname,
                     obj.active,
                     str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Objective:
            print(prefix+"%s: %s(active=%s, expr=%s)"
                  % (str(obj), clsname, obj.active, str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Expression:
            print(prefix+"%s: %s(expr=%s)"
                  % (str(obj), clsname, str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Param:
            print(prefix+"%s: %s(value=%s)"
                  % (str(obj), clsname, str(obj.value)))
        elif obj.ctype is pyomo.core.base.SOSConstraint:
            print(prefix+"%s: %s(active=%s, level=%s, entries=%s)"
                  % (str(obj),
                     clsname,
                     obj.active,
                     obj.level,
                     str(["(%s,%s)" % (str(v), w)
                          for v,w in zip(obj.variables,
                                         obj.weights)])))
        else:
            assert obj.ctype is pyomo.core.base.Suffix
            print(prefix+"%s: %s(size=%s)"
                  % (str(obj.name), clsname,str(len(obj))))
    else:
        # a block
        for i, block in enumerate(obj.blocks()):
            if i > 0:
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
                        pprint(c, indent=indent+1)

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
    kwds['include_all_parents'] = False
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
