#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

from six import itervalues, iteritems

from pyomo.util.plugin import alias
from pyomo.core import *
from pyomo.core.base import expr, Transformation
from pyomo.core.base.expr import _ProductExpression
from pyomo.core.base.set_types import BooleanSet
#from pyomo.core.base import _ExpressionData
from pyomo.core.base.var import _VarData
from pyomo.gdp import *

import logging
logger = logging.getLogger('pyomo.core')

class Bilinear_Transformation(Transformation):

    alias('gdp.bilinear', doc="Creates a disjunctive model where bilinear terms are replaced with disjunctive expressions.")

    def __init__(self):
        super(Bilinear_Transformation, self).__init__()

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})
        if getattr(instance, 'bilinear_vars_', None) is None:
            instance.bilinear_vars_ = VarList()
            instance.disjunction_index_ = Set()
            instance.disjuncts_   = Disjunct(instance.disjunction_index_*[0,1])
            instance.disjunction_ = Disjunction(instance.disjunction_index_)
        #
        # Iterate over all blocks
        #
        for block in instance.all_blocks(sort_by_keys=True):
            self._transformBlock(block, instance)
        #
        # Preprocess the instance
        #
        instance.preprocess()
        return instance

    def _transformBlock(self, block, instance):
        for component in active_components(block,Objective):
            component.expr = self._transformExpression(component.expr, instance)
        for component in active_components(block,Constraint):
            component.body = self._transformExpression(component.body, instance)

    def _transformExpression(self, expr, instance):
        if expr.polynomial_degree() > 2:
            raise ValueError("Cannot transform polynomial terms with degree > 2")
        if expr.polynomial_degree() < 2:
            return expr
        #
        expr = self._replace_bilinear(expr, instance)
        return expr

    def _replace_bilinear(self, expr, instance):
        idMap = {}
        terms = generate_canonical_repn(expr, idMap=idMap)
        # Constant
        if 0 in terms:
            e = terms[0][None]
        else:
            e = 0
        # Linear terms
        if 1 in terms:
            #print("HERE %s" % str(terms[1]))
            #print("HERE %s %s" % (str(type(idMap)), str(idMap)))
            for key in terms[1]:
                e += terms[1][key] * idMap[key]
        # Quadratic terms
        if 2 in terms:
            coef = terms[2].values()[0]
            vars = []
            for key in terms[2]:
                for v in key:
                    vars.append(idMap[v])
            #
            if isinstance(vars[0].domain, BooleanSet):
                v = instance.bilinear_vars_.add()
                id = len(instance.bilinear_vars_)
                instance.disjunction_index_.add(id)
                # First disjunct
                d0 = instance.bilinear_disjunct_[id,0]
                d0.c1 = Constraint(expr=v == 1)
                d0.c2 = Constraint(expr=v == e+coef*vars[1])
                # Second disjunct
                d1 = instance.bilinear_disjunct_[id,1]
                d0.c1 = Constraint(expr=v == 0)
                # Disjunction
                instance.bilinear_disjunction_[id]._disjuncts[id] = [instance.bilinear_disjunct_[id,0], instance.bilinear_disjunct_[id,1]]
                # The disjunctive variable is the expression
                return v
            #
            elif isinstance(vars[1].domain, BooleanSet):
                v = instance.bilinear_vars_.add()
                id = len(instance.bilinear_vars_)
                instance.disjunction_index_.add(id)
                # First disjunct
                d0 = instance.bilinear_disjunct_[id,0]
                d0.c1 = Constraint(expr=v == 1)
                d0.c2 = Constraint(expr=v == e+coef*vars[0])
                # Second disjunct
                d1 = instance.bilinear_disjunct_[id,1]
                d0.c1 = Constraint(expr=v == 0)
                # Disjunction
                instance.bilinear_disjunction_[id]._disjuncts[id] = [instance.bilinear_disjunct_[id,0], instance.bilinear_disjunct_[id,1]]
                # The disjunctive variable is the expression
                return v
            else:
                e += coef*vars[0]*vars[1]
                return e
            
                

    def X_replace_bilinear(self, expr, instance):
        if not expr.is_expression():
            return expr
        if type(expr) is _ProductExpression:
            if len(expr._numerator) != 2:
                expr._numerator = [self._replace_bilinear(e, instance) for e in expr._numerator]
                return expr
            if not isinstance(expr._numerator[0], _VarData) or \
                    not isinstance(expr._numerator[1], _VarData):
                raise RuntimeError("Cannot yet handle complex subexpressions")
            return expr
        # Else ...
        expr._args = [self._replace_bilinear(e, instance) for e in expr._args]
        return expr
             

