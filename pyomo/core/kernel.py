#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyomo.opt
from pyomo.opt import (SolverFactory,
                       SolverStatus,
                       TerminationCondition)
import pyomo.opt.base
from pyomo.core.base.component_map import ComponentMap
from pyomo.core.base.component_block import (block,
                                             tiny_block,
                                             block_list,
                                             block_dict)
from pyomo.core.base.component_variable import (variable,
                                                variable_list,
                                                variable_dict)
from pyomo.core.base.component_constraint import (constraint,
                                                  linear_constraint,
                                                  constraint_list,
                                                  constraint_dict)
from pyomo.core.base.component_parameter import (parameter,
                                                 parameter_list,
                                                 parameter_dict)
from pyomo.core.base.component_expression import (expression,
                                                  data_expression,
                                                  expression_list,
                                                  expression_dict)
from pyomo.core.base.component_objective import (objective,
                                                 objective_list,
                                                 objective_dict)
from pyomo.core.base.component_sos import (sos,
                                           sos1,
                                           sos2,
                                           sos_list,
                                           sos_dict)
from pyomo.core.base.component_suffix import (suffix,
                                              export_suffix_generator,
                                              import_suffix_generator,
                                              local_suffix_generator,
                                              suffix_generator)
from pyomo.core.base.component_piecewise.transforms import (piecewise,
                                                            piecewise_sos2,
                                                            piecewise_dcc,
                                                            piecewise_cc,
                                                            piecewise_mc,
                                                            piecewise_inc,
                                                            piecewise_dlog,
                                                            piecewise_log)
from pyomo.core.base.component_piecewise.transforms_nd import piecewise_nd

from pyomo.core.base.expr import *
from pyomo.core.base.set_types import (RealSet,
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
from pyomo.core.base.objective import (minimize,
                                       maximize)
from pyomo.core.base import value

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
from pyomo.core.base.component_block import _block_base

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

# I would really like to see this method changed to
# REQUIRE a filename as an argument and simply return
# the symbol map.
def _write(self,
          filename=None,
          format=None,
          solver_capability=None,
          io_options={}):
    """
    Write the model to a file, with a given format.
    """
    #
    # Guess the format if none is specified
    #
    if (filename is None) and (format is None):
        # Preserving backwards compatibility here.
        # The function used to be defined with format='lp' by
        # default, but this led to confusing behavior when a
        # user did something like 'model.write("f.nl")' and
        # expected guess_format to create an NL file.
        format = pyomo.opt.base.ProblemFormat.cpxlp
    if (filename is not None) and (format is None):
        format = pyomo.opt.base.guess_format(filename)
    problem_writer = pyomo.opt.WriterFactory(format)
    if problem_writer is None:
        raise ValueError(
            "Cannot write model in format '%s': no model "
            "writer registered for that format"
            % str(format))

    if solver_capability is None:
        solver_capability = lambda x: True
    (filename, smap) = problem_writer(self,
                                      filename,
                                      solver_capability,
                                      io_options)
    smap_id = id(smap)

    # BIG HACK
    if not hasattr(self, "._symbol_maps"):
        setattr(self, "._symbol_maps", {})
    getattr(self, "._symbol_maps")[smap_id] = smap

    return filename, smap_id
_block_base.write = _write
del _write

# canonical repn checks type instead of ctype
from pyomo.core.base.component_interface import ICategorizedObject
ICategorizedObject.type = ICategorizedObject.ctype
