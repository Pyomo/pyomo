#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging
from six import iteritems

from pyomo.util.plugin import alias
from pyomo.core import *
from pyomo.core.base import expr, Transformation
from pyomo.core.base.expr import _ProductExpression
from pyomo.core.base.set_types import BooleanSet
from pyomo.core.base.var import _VarData
from pyomo.gdp import *
from pyomo.repn import generate_canonical_repn

logger = logging.getLogger('pyomo.core')

class Bilinear_Transformation(Transformation):

    alias('gdp.bilinear', doc="Creates a disjunctive model where bilinear terms are replaced with disjunctive expressions.")

    def __init__(self):
        super(Bilinear_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})
        # TODO: This data should be stored differently.  We cannot nest this transformation with itself
        if getattr(instance, 'bilinear_data_', None) is None:
            instance.bilinear_data_ = Block()
            instance.bilinear_data_.vlist = VarList()
            instance.bilinear_data_.vlist_boolean = []
            instance.bilinear_data_.index = Set()
            instance.bilinear_data_.disjuncts_   = Disjunct(instance.bilinear_data_.index*[0,1])
            instance.bilinear_data_.disjunction_data = {}
            instance.bilinear_data_.o_expr = {}
            instance.bilinear_data_.c_body = {}
        #
        # Iterate over all blocks
        #
        for block in instance.block_data_objects(
                active=True, sort=SortComponents.deterministic ):
            self._transformBlock(block, instance)
        #
        # WEH: I wish I had a DisjunctList and DisjunctionList object...
        #
        def rule(block, i):
            return instance.bilinear_data_.disjunction_data[i]
        instance.bilinear_data_.disjunction_ = Disjunction(instance.bilinear_data_.index, rule=rule)

    def _transformBlock(self, block, instance):
        for component in block.component_objects(Objective, active=True, descend_into=False):
            expr = self._transformExpression(component.expr, instance)
            instance.bilinear_data_.o_expr[ id(component) ] = component.expr
            component.expr = expr
        for component in block.component_data_objects(Constraint, active=True, descend_into=False):
            expr = self._transformExpression(component.body, instance)
            instance.bilinear_data_.c_body[ id(component) ] = component.body
            component._body = expr

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
            for key in terms[1]:
                e += terms[1][key] * idMap[key]
        # Quadratic terms
        if 2 in terms:
            for key in terms[2]:
                vars = []
                for v in key:
                    vars.append(idMap[v])
                coef = terms[2][key]
                #
                if isinstance(vars[0].domain, BooleanSet):
                    instance.bilinear_data_.vlist_boolean.append(vars[0])
                    v = instance.bilinear_data_.vlist.add()
                    bounds = vars[1].bounds
                    v.setlb(bounds[0])
                    v.setub(bounds[1])
                    id = len(instance.bilinear_data_.vlist)
                    instance.bilinear_data_.index.add(id)
                    # First disjunct
                    d0 = instance.bilinear_data_.disjuncts_[id,0]
                    d0.c1 = Constraint(expr=vars[0] == 1)
                    d0.c2 = Constraint(expr=v == coef*vars[1])
                    # Second disjunct
                    d1 = instance.bilinear_data_.disjuncts_[id,1]
                    d1.c1 = Constraint(expr=vars[0] == 0)
                    d1.c2 = Constraint(expr=v == 0)
                    # Disjunction
                    instance.bilinear_data_.disjunction_data[id] = [instance.bilinear_data_.disjuncts_[id,0], instance.bilinear_data_.disjuncts_[id,1]]
                    instance.bilinear_data_.disjunction_data[id] = [instance.bilinear_data_.disjuncts_[id,0], instance.bilinear_data_.disjuncts_[id,1]]
                    # The disjunctive variable is the expression
                    e += v
                #
                elif isinstance(vars[1].domain, BooleanSet):
                    instance.bilinear_data_.vlist_boolean.append(vars[1])
                    v = instance.bilinear_data_.vlist.add()
                    bounds = vars[0].bounds
                    v.setlb(bounds[0])
                    v.setub(bounds[1])
                    id = len(instance.bilinear_data_.vlist)
                    instance.bilinear_data_.index.add(id)
                    # First disjunct
                    d0 = instance.bilinear_data_.disjuncts_[id,0]
                    d0.c1 = Constraint(expr=vars[1] == 1)
                    d0.c2 = Constraint(expr=v == coef*vars[0])
                    # Second disjunct
                    d1 = instance.bilinear_data_.disjuncts_[id,1]
                    d1.c1 = Constraint(expr=vars[1] == 0)
                    d1.c2 = Constraint(expr=v == 0)
                    # Disjunction
                    instance.bilinear_data_.disjunction_data[id] = [instance.bilinear_data_.disjuncts_[id,0], instance.bilinear_data_.disjuncts_[id,1]]
                    # The disjunctive variable is the expression
                    e += v
                else:
                    # If neither variable is boolean, just reinsert the original bilinear term
                    e += coef*vars[0]*vars[1]
        #
        return e
            
