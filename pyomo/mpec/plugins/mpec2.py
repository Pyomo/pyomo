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

from pyomo.core.expr import inequality
from pyomo.core.base import (Transformation,
                             TransformationFactory,
                             Constraint,
                             Block,
                             SortComponents,
                             ComponentUID)
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp.disjunct import Disjunct, Disjunction


logger = logging.getLogger('pyomo.core')


@TransformationFactory.register('mpec.simple_disjunction', doc="Disjunctive transformations of complementarity conditions when all variables are non-negative")
class MPEC2_Transformation(Transformation):


    def __init__(self):
        super(MPEC2_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})
        #
        # Setup transformation data
        #
        tdata = instance._transformation_data['mpec.simple_disjunction']
        tdata.compl_cuids = []
        #
        # Iterate over the model finding Complementarity components
        #
        for block in instance.block_data_objects(active=True, sort=SortComponents.deterministic):
            for complementarity in block.component_objects(Complementarity, active=True, descend_into=False):
                for index in sorted(iterkeys(complementarity)):
                    _data = complementarity[index]
                    if not _data.active:
                        continue
                    #
                    _e1 = _data._canonical_expression(_data._args[0])
                    _e2 = _data._canonical_expression(_data._args[1])
                    if len(_e1)==3 and len(_e2) == 3 and (_e1[0] is None) + (_e1[2] is None) + (_e2[0] is None) + (_e2[2] is None) != 2:
                        raise RuntimeError("Complementarity condition %s must have exactly two finite bounds" % _data.name)
                    if len(_e1) == 3 and _e1[0] is None and _e1[2] is None:
                        #
                        # Swap _e1 and _e2.  The ensures that 
                        # only e2 will be an unconstrained expression
                        #
                        _e1, _e2 = _e2, _e1
                    if _e2[0] is None and _e2[2] is None:
                        if len(_e1) == 2:
                            _data.c = Constraint(expr=_e1)
                        else:
                            _data.expr1 = Disjunct()
                            _data.expr1.c0 = Constraint(expr= _e1[0] == _e1[1])
                            _data.expr1.c1 = Constraint(expr= _e2[1] >= 0)
                            #
                            _data.expr2 = Disjunct()
                            _data.expr2.c0 = Constraint(expr= _e1[1] == _e1[2])
                            _data.expr2.c1 = Constraint(expr= _e2[1] <= 0)
                            #
                            _data.expr3 = Disjunct()
                            _data.expr3.c0 = Constraint(expr= inequality(_e1[0], _e1[1], _e1[2]))
                            _data.expr3.c1 = Constraint(expr= _e2[1] == 0)
                            _data.complements = Disjunction(expr=(_data.expr1, _data.expr2, _data.expr3))
                    else:
                        if _e1[0] is None:
                            tmp1 = _e1[2] - _e1[1]
                        else:
                            tmp1 = _e1[1] - _e1[0]
                        if _e2[0] is None:
                            tmp2 = _e2[2] - _e2[1]
                        else:
                            tmp2 = _e2[1] - _e2[0]
                        _data.expr1 = Disjunct()
                        _data.expr1.c0 = Constraint(expr= tmp1 >= 0)
                        _data.expr1.c1 = Constraint(expr= tmp2 == 0)
                        #
                        _data.expr2 = Disjunct()
                        _data.expr2.c0 = Constraint(expr= tmp1 == 0)
                        _data.expr2.c1 = Constraint(expr= tmp2 >= 0)
                        #
                        _data.complements = Disjunction(expr=(_data.expr1, _data.expr2))
                tdata.compl_cuids.append( ComponentUID(complementarity) )
                block.reclassify_component_type(complementarity, Block)
