#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging
import weakref
from six import iterkeys

from pyomo.util.plugin import alias
from pyomo.core.base import (Transformation,
                             Constraint,
                             ConstraintList,
                             Var,
                             VarList,
                             Block,
                             ComponentUID,
                             SortComponents)
from pyomo.mpec.complementarity import Complementarity, ComplementarityList, complements

logger = logging.getLogger('pyomo.core')


#
# This transformation reworks each Complementarity block to 
# create a square mixed-complementarity problem
#
class MPEC4_Transformation(Transformation):

    alias('mpec.square_mcp', doc="Generate a square mixed-complementarity problem")

    def __init__(self):
        super(MPEC4_Transformation, self).__init__()

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})
        tdata = instance._transformation_data['mpec.square_mcp']
        tdata.compl_cuids = []
        #
        # Add a list of complementary conditions that are added for equality constraints
        #
        instance._square_block = Block()
        instance._square_block.clist = ConstraintList()
        instance._square_block.cclist = ComplementarityList()
        instance._square_block.vlist = VarList()
        #
        # Find the free variables
        #
        free_vars = {}
        id_list = []
        for vdata in instance.component_data_objects(Var, active=True, sort=SortComponents.deterministic):
            id_list.append( id(vdata) )
            free_vars[id(vdata)] = vdata
        #
        # Iterate over the Complementarity components
        #
        cobjs = []
        for bdata in instance.block_data_objects(active=True, sort=SortComponents.deterministic):
            for cobj in bdata.component_objects(Complementarity, active=True, descend_into=False):
                cobjs.append(cobj)
                for index in sorted(iterkeys(cobj)):
                    _cdata = cobj[index]
                    if not _cdata.active:
                        continue
                    #
                    # Apply a variant of the standard form logic
                    #
                    self.to_square_form(_cdata, free_vars)
        #
        # Now we need to add constraints for free variables with finite bounds
        #
        tmp = []
        for id_ in id_list:
            if id_ in free_vars:
                vdata = free_vars[id_]
                if not (vdata.bounds[0] is None and vdata.bounds[1] is None):
                    instance._square_block.clist.add( (vdata.bounds[0], vdata, vdata.bounds[1]) )
                    vdata.setlb(None)
                    vdata.setub(None)
                tmp.append( id_ )
        id_list = tmp
        i=0
        #
        # Now we iterate over constraints
        #
        for block in instance.block_data_objects(active=True, sort=SortComponents.deterministic):
            #
            # For each equality constraint, add a complementarity condition.
            #
            for cdata in block.component_data_objects(Constraint, active=True, descend_into=False):
                if cdata.equality:
                    tmp = instance._square_block.cclist.add( complements( cdata.lower == cdata.body, free_vars[id_list[i]] ) )
                    self.to_square_form(tmp, free_vars)
                    i += 1
                else:
                    if not cdata.lower is None:
                        vdata = instance._square_block.vlist.add() 
                        free_vars[id(vdata)] = vdata
                        tmp = instance._square_block.cclist.add( complements( cdata.lower <= cdata.body, 0 <= vdata) )
                        self.to_square_form(tmp, free_vars)
                    if not cdata.upper is None:
                        vdata = instance._square_block.vlist.add() 
                        free_vars[id(vdata)] = vdata
                        tmp = instance._square_block.cclist.add( complements( - cdata.upper <= - cdata.body, 0 <= vdata) )
                        self.to_square_form(tmp, free_vars)
        #
        for cobj in cobjs:
            tdata.compl_cuids.append( ComponentUID(cobj) )
            cobj.parent_block().reclassify_component_type(cobj, Block)
        #
        instance.preprocess()
        #
        #self.print_square_form(instance)
        return instance

    def print_square_form(self, instance):
        """
        Summarize the square form of this complementarity problem.
        """
        vmap = {}
        for vdata in instance.component_data_objects(Var, active=True):
            vmap[id(vdata)] = vdata
        print("-------------------- SQUARE MCP ----------------------")
        for bdata in instance.block_data_objects(active=True, sort=SortComponents.deterministic):
            for cobj in bdata.component_data_objects(Constraint, active=True, descend_into=False):
                print("%s %s\t\t\t%s" % (getattr(cobj, '_complementarity', None), str(cobj.lower)+" < "+str(cobj.body)+" < "+str(cobj.upper) , vmap.get(getattr(cobj, '_vid', None),None)))
        print("-------------------- SQUARE MCP ----------------------")
        
    def to_square_form(self, cdata, free_vars):
        """
        Convert a single complementarity condition.
        """
        _e1 = cdata._canonical_expression(cdata._args[0], triple=True)
        _e2 = cdata._canonical_expression(cdata._args[1], triple=True)
        if False:
            if _e1[0] is None:
                print(None)
            else:
                print(str(_e1[0]))
            if _e1[1] is None:
                print(None)
            else:
                print(str(_e1[1]))
            if _e1[2] is None:
                print(None)
            else:
                print(str(_e1[2]))
            if _e2[0] is None:
                print(None)
            else:
                print(str(_e2[0]))
            if _e2[1] is None:
                print(None)
            else:
                print(str(_e2[1]))
            if _e2[2] is None:
                print(None)
            else:
                print(str(_e2[2]))
        #
        # Swap if the body of the second constraint is not a free variable
        #
        if not id(_e2[1]) in free_vars:
            _e1, _e2 = _e2, _e1
        #
        # Add the complementarity constraint expression
        #
        cdata.c = Constraint(expr=_e1)
        if _e1[0] is None:
            if _e1[2] is None:
                cdata.c._complementarity = 4             # unbounded
            else:
                cdata.c._complementarity = 2             # upper-bound
        else:
            if _e1[2] is None:
                cdata.c._complementarity = 1             # lower-bound
            else:
                cdata.c._complementarity = 3             # range constraint
        #
        # If the body of the second constraint is a free variable, then keep it.
        # Otherwise, create a new variable and a new constraint.
        #
        if id(_e2[1]) in free_vars:
            cdata.c._vid = id(_e2[1])
            del free_vars[cdata.c._vid]
        else:
            cdata.v = Var(bounds=(_e2[0], _e2[2]))
            cdata.c._vid = id(cdata.v)
            cdata.e = Constraint(expr=cdata.v == _e2[1])
            
      
