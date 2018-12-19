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
from six import iteritems

from pyomo.core.expr.current import ProductExpression
from pyomo.core import *
from pyomo.core.base.set_types import BooleanSet
from pyomo.core.base.var import _VarData
from pyomo.gdp import *
from pyomo.repn import generate_standard_repn

logger = logging.getLogger('pyomo.gdp')


@TransformationFactory.register('gdp.bilinear', doc="Creates a disjunctive model where bilinear terms are replaced with disjunctive expressions.")
class Bilinear_Transformation(Transformation):

    def __init__(self):
        super(Bilinear_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})
        # TODO: This data should be stored differently.  We cannot nest this transformation with itself
        if getattr(instance, 'bilinear_data_', None) is None:
            instance.bilinear_data_ = Block()
            instance.bilinear_data_.cache = {}
            instance.bilinear_data_.vlist = VarList()
            instance.bilinear_data_.vlist_boolean = []
            instance.bilinear_data_.IDX = Set()
            instance.bilinear_data_.disjuncts_   = Disjunct(instance.bilinear_data_.IDX*[0,1])
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
        instance.bilinear_data_.disjunction_ = Disjunction(instance.bilinear_data_.IDX, rule=rule)

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
        terms = generate_standard_repn(expr, idMap=idMap)
        # Constant
        e = terms.constant
        # Linear terms
        for var, coef in zip(terms.linear_vars, terms.linear_coefs):
            e += coef * var
        # Quadratic terms
        if len(terms.quadratic_coefs) > 0:
            for vars_, coef_ in zip(terms.quadratic_vars, terms.quadratic_coefs):
                #
                if isinstance(vars_[0].domain, BooleanSet):
                    v = instance.bilinear_data_.cache.get( (id(vars_[0]),id(vars_[1])), None )
                    if v is None:
                        instance.bilinear_data_.vlist_boolean.append(vars_[0])
                        v = instance.bilinear_data_.vlist.add()
                        instance.bilinear_data_.cache[id(vars_[0]), id(vars_[1])] = v
                        bounds = vars_[1].bounds
                        v.setlb(bounds[0])
                        v.setub(bounds[1])
                        id_ = len(instance.bilinear_data_.vlist)
                        instance.bilinear_data_.IDX.add(id_)
                        # First disjunct
                        d0 = instance.bilinear_data_.disjuncts_[id_,0]
                        d0.c1 = Constraint(expr=vars_[0] == 1)
                        d0.c2 = Constraint(expr=v == vars_[1])
                        # Second disjunct
                        d1 = instance.bilinear_data_.disjuncts_[id_,1]
                        d1.c1 = Constraint(expr=vars_[0] == 0)
                        d1.c2 = Constraint(expr=v == 0)
                        # Disjunction
                        instance.bilinear_data_.disjunction_data[id_] = [instance.bilinear_data_.disjuncts_[id_,0], instance.bilinear_data_.disjuncts_[id_,1]]
                        instance.bilinear_data_.disjunction_data[id_] = [instance.bilinear_data_.disjuncts_[id_,0], instance.bilinear_data_.disjuncts_[id_,1]]
                    # The disjunctive variable is the expression
                    e += coef_*v
                #
                elif isinstance(vars_[1].domain, BooleanSet):
                    v = instance.bilinear_data_.cache.get( (id(vars_[1]),id(vars_[0])), None )
                    if v is None:
                        instance.bilinear_data_.vlist_boolean.append(vars_[1])
                        v = instance.bilinear_data_.vlist.add()
                        instance.bilinear_data_.cache[id(vars_[1]), id(vars_[0])] = v
                        bounds = vars_[0].bounds
                        v.setlb(bounds[0])
                        v.setub(bounds[1])
                        id_ = len(instance.bilinear_data_.vlist)
                        instance.bilinear_data_.IDX.add(id_)
                        # First disjunct
                        d0 = instance.bilinear_data_.disjuncts_[id_,0]
                        d0.c1 = Constraint(expr=vars_[1] == 1)
                        d0.c2 = Constraint(expr=v == vars_[0])
                        # Second disjunct
                        d1 = instance.bilinear_data_.disjuncts_[id_,1]
                        d1.c1 = Constraint(expr=vars_[1] == 0)
                        d1.c2 = Constraint(expr=v == 0)
                        # Disjunction
                        instance.bilinear_data_.disjunction_data[id_] = [instance.bilinear_data_.disjuncts_[id_,0], instance.bilinear_data_.disjuncts_[id_,1]]
                    # The disjunctive variable is the expression
                    e += coef_*v
                else:
                    # If neither variable is boolean, just reinsert the original bilinear term
                    e += coef_*vars_[0]*vars_[1]
        #
        return e
            
