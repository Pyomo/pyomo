#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

from six import iterkeys
from pyomo.util.plugin import alias
from pyomo.core.base import Transformation, Var, Constraint, Objective, active_components, Block
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp.disjunct import Disjunct, Disjunction

import logging
logger = logging.getLogger('pyomo.core')


class MPEC2_Transformation(Transformation):

    alias('mpec.simple_disjunction', doc="Disjunctive transformations of complementarity conditions when all variables are non-negative")

    def __init__(self):
        super(MPEC2_Transformation, self).__init__()

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})
        #
        # Iterate over the model finding Complementarity components
        #
        for block in instance.all_blocks(sort_by_keys=True):
            for complementarity in active_components(block,Complementarity):
                for index in sorted(iterkeys(complementarity)):
                    _data = complementarity[index]
                    if not _data.active:
                        continue
                    #
                    _e1 = _data._canonical_expression(_data._args[0])
                    _e2 = _data._canonical_expression(_data._args[1])
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
                            _data.expr2 = Disjunct()
                            _data.expr2.c0 = Constraint(expr= _e1[1] == _e1[2])
                            _data.expr2.c1 = Constraint(expr= _e2[1] <= 0)
                            _data.expr3 = Disjunct()
                            # This should be strict inequalities
                            _data.expr3.c0 = Constraint(expr= _e1[0] <= _e1[1] <= _e1[2])
                            _data.expr3.c1 = Constraint(expr= _e2[1] == 0)
                            _data.complements = Disjunction(expr=(_data.expr1, _data.expr2, _data.expr3))
                    else:
                        _data.expr1 = Disjunct()
                        _data.expr1.c = Constraint(expr= _e1)
                        _data.expr2 = Disjunct()
                        _data.expr2.c = Constraint(expr= _e2)
                        _data.complements = Disjunction(expr=(_data.expr1, _data.expr2))
                block.reclassify_component_type(complementarity, Block)

        #
        instance.preprocess()
        return instance

