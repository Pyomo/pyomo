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

from pyomo.common.deprecation import deprecated
from pyomo.core.base import (Transformation,
                             TransformationFactory,
                             Var,
                             Constraint,
                             Objective,
                             minimize,
                             NonNegativeReals,
                             NonPositiveReals,
                             Reals,
                             Block,
                             Model,
                             ConcreteModel)
from pyomo.duality.collect import collect_linear_terms

def load():
    pass

logger = logging.getLogger('pyomo.core')


#
# This transformation creates a new block that
# is the dual of the specified block.  If no block is
# specified, then the entire model is dualized.
# This returns a new Block object.
#
@TransformationFactory.register('duality.linear_dual', doc="Dualize a linear model")
class LinearDual_PyomoTransformation(Transformation):

    @deprecated(
        "Use of the pyomo.duality package is deprecated. There are known bugs "
        "in pyomo.duality, and we do not recommend the use of this code. "
        "Development of dualization capabilities has been shifted to "
        "the Pyomo Adversarial Optimization (PAO) library. Please contact "
        "William Hart for further details (wehart@sandia.gov).",
        version='5.6.2')
    def __init__(self):
        super(LinearDual_PyomoTransformation, self).__init__()

    def _create_using(self, instance, **kwds):
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
            for (name, data) in instance.component_map(Block, active=True).items():
                if name == bname:
                    block = instance
        if block is None:
            raise RuntimeError("Missing block: "+bname)
        #
        # Generate the dual
        #
        instance_ = self._dualize(block)

        return instance_

    def _dualize(self, block, unfixed=[]):
        """
        Generate the dual of a block
        """
        #
        # Collect linear terms from the block
        #
        A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain = collect_linear_terms(block, unfixed)
        ##print(A)
        ##print(vnames)
        ##print(cnames)
        ##print(list(A.keys()))
        ##print("---")
        ##print(A.keys())
        ##print(c_sense)
        ##print(c_rhs)
        #
        # Construct the block
        #
        if isinstance(block, Model):
            dual = ConcreteModel()
        else:
            dual = Block()
        dual.construct()
        _vars = {}
        def getvar(name, ndx=None):
            v = _vars.get((name,ndx), None)
            if v is None:
                v = Var()
                if ndx is None:
                    v_name = name
                elif type(ndx) is tuple:
                    v_name = "%s[%s]" % (name, ','.join(map(str,ndx)))
                else:
                    v_name = "%s[%s]" % (name, str(ndx))
                setattr(dual, v_name, v)
                _vars[name,ndx] = v
            return v
        #
        # Construct the objective
        #
        if d_sense == minimize:
            dual.o = Objective(expr=sum(- b_coef[name,ndx]*getvar(name,ndx) for name,ndx in b_coef), sense=d_sense)
        else:
            dual.o = Objective(expr=sum(b_coef[name,ndx]*getvar(name,ndx) for name,ndx in b_coef), sense=d_sense)
        #
        # Construct the constraints
        #
        for cname in A:
            for ndx, terms in iteritems(A[cname]):
                expr = 0
                for term in terms:
                    expr += term.coef * getvar(term.var, term.ndx)
                if not (cname, ndx) in c_rhs:
                    c_rhs[cname, ndx] = 0.0
                if c_sense[cname, ndx] == 'e':
                    e = expr - c_rhs[cname,ndx] == 0
                elif c_sense[cname, ndx] == 'l':
                    e = expr - c_rhs[cname,ndx] <= 0
                else:
                    e = expr - c_rhs[cname,ndx] >= 0
                c = Constraint(expr=e)
                if ndx is None:
                    c_name = cname
                elif type(ndx) is tuple:
                    c_name = "%s[%s]" % (cname, ','.join(map(str,ndx)))
                else:
                    c_name = "%s[%s]" % (cname, str(ndx))
                setattr(dual, c_name, c)
            #
            for (name, ndx), domain in iteritems(v_domain):
                v = getvar(name, ndx)
                flag = type(ndx) is tuple and (ndx[-1] == 'lb' or ndx[-1] == 'ub')
                if domain == 1:
                    v.domain = NonNegativeReals
                elif domain == -1:
                    v.domain = NonPositiveReals
                else:
                    # TODO: verify that this case is possible
                    v.domain = Reals

        return dual
