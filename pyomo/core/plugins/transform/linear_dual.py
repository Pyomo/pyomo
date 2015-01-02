#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from six import iteritems

from pyutilib.misc import Bunch
import pyomo.util
from pyomo.util.plugin import alias
from pyomo.core.base import Transformation, Var, Constraint, VarList, ConstraintList, Objective, Set, maximize, minimize, NonNegativeReals, NonPositiveReals, Reals, Block, Model, ConcreteModel
from pyomo.repn.canonical_repn import generate_canonical_repn
from pyomo.repn.canonical_repn import LinearCanonicalRepn
from pyomo.core.plugins.transform.util import process_canonical_repn
from pyomo.repn.collect import collect_linear_terms

import logging
logger = logging.getLogger('pyomo.core')


#
# This transformation creates a new block that
# is the dual of the specified block.  If no block is 
# specified, then the entire model is dualized.
# This returns a new Block object.
#
class LinearDual_PyomoTransformation(Transformation):

    alias('core.linear_dual', doc="Dualize a linear model")

    def __init__(self):
        super(LinearDual_PyomoTransformation, self).__init__()

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})
        bname = options.get('block',None)
        #
        # Iterate over the model collecting variable data,
        # until the block is found.
        #
        block = None
        if block is None:
            block = instance
        else:
            for (name, data) in instance.active_components(Block).items():
                if name == bname:
                    block = instance
        if block is None:
            raise RuntimeError("Missing block: "+bname)
        #
        # Collect variables
        #
        var = {}
        for (name, data) in block.active_components().items():
            if isinstance(data,Var):
                var[name] = data
        #
        # Generate the dual
        #
        instance_ = self._dualize(block)
        #
        # Execute the preprocessor
        #
        preprocessor = instance.model().config.preprocessor
        pyomo.util.PyomoAPIFactory(preprocessor)(instance.model().config, model=instance_)
        #instance_.preprocess()
        #
        return instance_

    def _dualize(self, block, unfixed=[]):
        """
        Generate the dual of a block
        """ 
        #
        # Collect linear terms from the block
        #
        A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain = collect_linear_terms(block, unfixed)
        #
        # Construct the block
        #
        if isinstance(block, Model):
            dual = ConcreteModel()
        else:
            dual = Block()
        for v, is_indexed in vnames:
            if is_indexed:
                setattr(dual, v+'_Index', Set(dimen=None))
                setattr(dual, v, Var(getattr(dual, v+'_Index')))
            else:
                setattr(dual, v, Var())
        for cname, is_indexed in cnames:
            if is_indexed:
                setattr(dual, cname+'_Index', Set(dimen=None))
                setattr(dual, cname, Constraint(getattr(dual, cname+'_Index'), noruleinit=True))
                setattr(dual, cname+'_lower_', Var(getattr(dual, cname+'_Index')))
                setattr(dual, cname+'_upper_', Var(getattr(dual, cname+'_Index')))
            else:
                setattr(dual, cname, Constraint(noruleinit=True))
                setattr(dual, cname+'_lower_', Var())
                setattr(dual, cname+'_upper_', Var())
        dual.construct()
        #
        # Construct the objective
        #
        if d_sense == minimize:
            dual.o = Objective(expr=sum(- b_coef[name,ndx]*getattr(dual,name)[ndx] for name,ndx in b_coef), sense=d_sense)
        else:
            dual.o = Objective(expr=sum(b_coef[name,ndx]*getattr(dual,name)[ndx] for name,ndx in b_coef), sense=d_sense)
        #
        # Construct the constraints
        #
        for cname in A:
            c = getattr(dual, cname)
            c_index = getattr(dual, cname+"_Index") if c.is_indexed() else None
            for ndx,terms in iteritems(A[cname]):
                if not c_index is None and not ndx in c_index:
                    c_index.add(ndx)
                expr = 0
                for term in terms:
                    v = getattr(dual,term.var)
                    if not term.ndx in v:
                        v.add(term.ndx)
                    expr += term.coef * v[term.ndx]
                if not (cname, ndx) in c_rhs:
                    c_rhs[cname, ndx] = 0.0
                if c_sense[cname,ndx] == 'e':
                    c.add(ndx, expr - c_rhs[cname,ndx] == 0)
                elif c_sense[cname,ndx] == 'l':
                    c.add(ndx, expr - c_rhs[cname,ndx] <= 0)
                else:
                    c.add(ndx, expr - c_rhs[cname,ndx] >= 0)
            for (name, ndx), domain in iteritems(v_domain):
                v = getattr(dual, name)
                flag = type(ndx) is tuple and (ndx[-1] == 'lb' or ndx[-1] == 'ub')
                if domain == 1:
                    if flag:
                        v[ndx].domain = NonNegativeReals
                    else:
                        v.domain = NonNegativeReals
                elif domain == -1:
                    if flag:
                        v[ndx].domain = NonPositiveReals
                    else:
                        v.domain = NonPositiveReals
                else:
                    if flag:
                        # TODO: verify that this case is possible
                        v[ndx].domain = Reals
                    else:
                        v.domain = Reals
        return dual

