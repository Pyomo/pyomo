#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
from six import iterkeys

from pyomo.core.base import (Transformation,
                             TransformationFactory,
                             Constraint,
                             Var,
                             Block,
                             ComponentUID,
                             SortComponents,
                             value)
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp import Disjunct

logger = logging.getLogger('pyomo.core')

#
# This transformation reworks each Complementarity block to 
# create a mixed-complementarity problem that can be written to an NL file.
#
@TransformationFactory.register('mpec.nl', doc="Transform a MPEC into a form suitable for the NL writer")
class MPEC4_Transformation(Transformation):


    def __init__(self):
        super(MPEC4_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        #
        # Find the free variables
        #
        free_vars = {}
        id_list = []
        # [ESJ 07/12/2019] Look on the whole model in case instance is a Block or a Disjunct
        for vdata in instance.model().component_data_objects(Var, active=True,
                                                             sort=SortComponents.deterministic,
                                                             descend_into=(Block, Disjunct)):
            id_list.append( id(vdata) )
            free_vars[id(vdata)] = vdata
        #
        # Iterate over the Complementarity components
        #
        cobjs = []
        for cobj in instance.component_objects(Complementarity, active=True,
                                               descend_into=(Block, Disjunct),
                                               sort=SortComponents.deterministic):
            cobjs.append(cobj)
            for index in sorted(iterkeys(cobj)):
                _cdata = cobj[index]
                if not _cdata.active:
                    continue
                #
                # Apply a variant of the standard form logic
                #
                self.to_common_form(_cdata, free_vars)
                #
        tdata = instance._transformation_data['mpec.nl']
        tdata.compl_cuids = []
        for cobj in cobjs:
            tdata.compl_cuids.append( ComponentUID(cobj) )
            cobj.parent_block().reclassify_component_type(cobj, Block)

    #instance.pprint()
    #self.print_nl_form(instance)

    def print_nl_form(self, instance):          #pragma:nocover
        """
        Summarize the complementarity relations in this problem.
        """
        vmap = {}
        for vdata in instance.component_data_objects(Var, active=True):
            vmap[id(vdata)] = vdata
        print("-------------------- Complementary Relations ----------------------")
        for bdata in instance.block_data_objects(active=True, sort=SortComponents.            deterministic):
            for cobj in bdata.component_data_objects(Constraint, active=True,                 descend_into=False):
                print("%s %s\t\t\t%s" % (getattr(cobj, '_complementarity', None), str(cobj.   lower)+" < "+str(cobj.body)+" < "+str(cobj.upper) , vmap.get(getattr(cobj, '_vid', None),     None)))
        print("-------------------- Complementary Relations ----------------------")

    def to_common_form(self, cdata, free_vars):
        """
        Convert a common form that can processed by AMPL
        """
        _e1 = cdata._canonical_expression(cdata._args[0])
        _e2 = cdata._canonical_expression(cdata._args[1])
        if False:                       #pragma:nocover
            if _e1[0] is None:
                print(None)
            else:
                print(str(_e1[0]))
            if _e1[1] is None:
                print(None)
            else:
                print(str(_e1[1]))
            if len(_e1) > 2:
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
            if len(_e2) > 2:
                if _e2[2] is None:
                    print(None)
                else:
                    print(str(_e2[2]))
        if len(_e1) == 2:
            cdata.c = Constraint(expr=_e1)
            return
        if len(_e2) == 2:
            cdata.c = Constraint(expr=_e2)
            return
        if (_e1[0] is None) + (_e1[2] is None) + (_e2[0] is None) + (_e2[2] is None) != 2:
            raise RuntimeError("Complementarity condition %s must have exactly two finite bounds" % cdata.name)
        #
        # Swap if the body of the second constraint is not a free variable
        #
        if not id(_e2[1]) in free_vars and id(_e1[1]) in free_vars:
            _e1, _e2 = _e2, _e1
        #
        # Rework the first constraint to have a zero bound.
        # The bound is a lower or upper bound depending on the
        # variable bound.
        #
        if not _e1[0] is None:
            cdata.bv = Var()
            cdata.c = Constraint(expr=0 <= cdata.bv)
            if not _e2[0] is None:
                cdata.bc = Constraint(expr=cdata.bv == _e1[1] - _e1[0])
            else:
                cdata.bc = Constraint(expr=cdata.bv == _e1[0] - _e1[1])
        elif not _e1[2] is None:
            cdata.bv = Var()
            cdata.c = Constraint(expr=0 <= cdata.bv)
            if not _e2[2] is None:
                cdata.bc = Constraint(expr=cdata.bv == _e1[1] - _e1[2])
            else:
                cdata.bc = Constraint(expr=cdata.bv == _e1[2] - _e1[1])
        else:
            cdata.bv = Var()
            cdata.bc = Constraint(expr=cdata.bv == _e1[1])
            cdata.c = Constraint(expr=(None, cdata.bv, None))
        #
        # If the body of the second constraint is a free variable, then keep it.
        # Otherwise, create a new variable and a new constraint.
        #
        if id(_e2[1]) in free_vars:
            var = _e2[1]
            cdata.c._vid = id(_e2[1])
            del free_vars[cdata.c._vid]
        else:
            var = cdata.v = Var()
            cdata.c._vid = id(cdata.v)
            cdata.e = Constraint(expr=cdata.v == _e2[1])
        #
        # Set the variable bound values, and corresponding _complementarity value
        #
        cdata.c._complementarity = 0
        if not _e2[0] is None:
            if var.lb is None or value(_e2[0]) > value(var.lb):
                var.setlb(_e2[0])
            cdata.c._complementarity += 1
        if not _e2[2] is None:
            if var.ub is None or value(_e2[2]) > value(var.ub):
                var.setub(_e2[2])
            cdata.c._complementarity += 2
      
