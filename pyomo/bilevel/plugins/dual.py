#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import itertools
from six import iteritems

from pyutilib.misc import Bunch
from pyomo.util.plugin import alias
from pyomo.core.base import Transformation, Var, Constraint, VarList, ConstraintList, Objective, Set, maximize, minimize, NonNegativeReals, NonPositiveReals, Reals, Block, ComponentUID
from pyomo.repn.canonical_repn import generate_canonical_repn
from pyomo.repn.canonical_repn import LinearCanonicalRepn
from pyomo.core.plugins.transform.util import process_canonical_repn
from pyomo.bilevel import SubModel
from pyomo.core.base.plugin import TransformationFactory
import logging
logger = logging.getLogger('pyomo.core')


#
# This transformation creates a new SubModel block that
# is the dual of the specified block.  If no block is 
# specified, then the first SubModel block is dualized.
# After creating the dual SubModel, the original SubModel is
# disabled.
#
class TransformationData(object): pass

class LinearDual_BilevelTransformation(Transformation):

    alias('bilevel.linear_dual', doc="Dualize a SubModel block")

    def __init__(self):
        super(LinearDual_BilevelTransformation, self).__init__()

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})
        sub = options.get('submodel',None)
        #
        # Iterate over the model collecting variable data,
        # until the submodel is found.
        #
        var = {}
        submodel = None
        for (name, data) in instance.active_components().items():
            if isinstance(data,Var):
                var[name] = data
            elif isinstance(data,SubModel):
                if sub is None or sub == name:
                    sub = name
                    submodel = data
                    break
        if submodel is None:
            raise RuntimeError("Missing submodel: "+sub)
        #
        instance._transformation_data = TransformationData()
        instance._transformation_data.submodel = [name]
        #
        # Fix variables
        #
        if submodel._fixed:
            fixed = [i.name for i in submodel._fixed]
            unfixed = []
            for v in var:
                if not v in fixed:
                    unfixed.append((v,getattr(submodel._parent(),v).is_indexed()))
        elif submodel._var:
            _var = set(submodel._var)
            unfixed = [(v,getattr(submodel._parent(),v).is_indexed()) for v in _var]
            fixed = []
            for v in var:
                if not v in _var:
                    fixed.append(v)
        else:
            raise RuntimeError("Must specify either 'fixed' or 'var' option for SubModel")
        instance._transformation_data.fixed = [ComponentUID(var[v]) for v in fixed]
        fixed_cache = {}
        #print("VAR KEYS "+str(var.keys()))
        #print("VAR FIXED "+str(fixed))
        for v in fixed:
            fixed_cache[v] = self._fix(var[v])
        #
        # Generate the dual
        #
        setattr(instance, sub+'_dual', self._dualize(submodel, unfixed))
        instance.reclassify_component_type(sub+'_dual', Block)
        #
        # Deactivate the original subproblem and upper-level objective
        #
        for (oname, odata) in submodel._parent().active_components(Objective).items():
            odata.deactivate()
        submodel.deactivate()
        #
        # Unfix the upper variables
        #
        for v in fixed:
            self._unfix(var[v], fixed_cache[v])
        #
        # Go through the subproblem and look for bilinear terms
        #
        ##self._xfrm_bilinearities(getattr(instance, sub+'_dual')) 
        #
        # Disable the original submodel and
        # execute the preprocessor
        #
        getattr(instance,sub).deactivate()
        instance.preprocess()
        return instance

    def _fix(self, var):
        """
        Fix the upper level variables, tracking the variables that were
        modified.
        """
        cache = []
        for i,vardata in var.items():
            if not vardata.fixed:
                vardata.fix()
                cache.append(i)
        return cache

    def _unfix(self, var, cache):
        """
        Unfix the upper level variables.
        """
        for i in cache:
            var[i].unfix()

    def _dualize(self, submodel, unfixed):
        """
        Generate the dual of a submodel
        """ 
        transform = TransformationFactory('base.linear_dual')
        return transform._dualize(submodel, unfixed)

    def _xfrm_bilinearities(self, dual):
        """
        Replace bilinear terms in constraints with disjunctions
        """ 
        for (name, data) in dual.active_components(Constraint).items():
            for ndx in data:
                con = data[ndx]
                degree = con.body.polynomial_degree()
                if degree > 2:
                    raise "RuntimeError: Cannot transform a model with polynomial degree %d" % degree
                if degree == 2:
                    terms = generate_canonical_repn(con.body)
                    for term in terms:
                        print("%s %s %s" % (name, ndx, term))


