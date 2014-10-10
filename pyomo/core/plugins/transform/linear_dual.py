#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import itertools

from pyutilib.misc import Bunch
from coopr.core.plugin import alias
from coopr.pyomo.base import Transformation, Var, Constraint, VarList, ConstraintList, Objective, Set, maximize, minimize, NonNegativeReals, NonPositiveReals, Reals
from coopr.pyomo.expr.canonical_repn import generate_canonical_repn
from coopr.pyomo.expr.canonical_repn import LinearCanonicalRepn
from coopr.pyomo.plugins.transform.util import process_canonical_repn
from coopr.bilevel import SubModel

import logging
logger = logging.getLogger('coopr.pyomo')


#
# This transformation creates a new block that
# is the dual of the specified block.  If no block is 
# specified, then the entire model is dualized.
# This returns a new Block object.
#
class LinearDual_PyomoTransformation(Transformation):

    alias('pyomo.linear_dual', doc="Dualize a linear model")

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
        instance_.preprocess()
        return instance_

    def _dualize(self, block, unfixed=[], model=True):
        """
        Generate the dual of a block
        """ 
        #
        # Start constructing the block
        #
        # Variables are constraints of block
        # Constraints are unfixed variables of block and the parent model.
        #
        vnames = set()
        for (name, data) in block.active_components(Constraint).items():
            vnames.add((name, data.is_indexed()))
        cnames = set(unfixed)
        for (name, data) in block.active_components(Var).items():
            cnames.add((name, data.is_indexed()))
        #
        dual = SubModel()
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
        A = {}
        b_coef = {}
        c_rhs = {}
        c_sense = {}
        d_sense = None
        #
        # Collect objective
        #
        for (oname, odata) in block.active_components(Objective).items():
            for ndx in odata:
                if odata[ndx].sense == maximize:
                    o_terms = generate_canonical_repn(-1*odata[ndx].expr, compute_values=False)
                    d_sense = minimize
                else:
                    o_terms = generate_canonical_repn(odata[ndx].expr, compute_values=False)
                    d_sense = maximize
                for i in range(len(o_terms.variables)):
                    c_rhs[ o_terms.variables[i].component().name, o_terms.variables[i].index() ] = o_terms.linear[i]
            # Stop after the first objective
            break
        #
        # Collect constraints
        #
        for (name, data) in block.active_components(Constraint).items():
            for ndx in data:
                con = data[ndx]
                body_terms = generate_canonical_repn(con.body, compute_values=False)
                lower_terms = generate_canonical_repn(con.lower, compute_values=False) if not con.lower is None else None
                upper_terms = generate_canonical_repn(con.upper, compute_values=False) if not con.upper is None else None
                #
                if body_terms.constant is None:
                    body_terms.constant = 0
                if not lower_terms is None and not lower_terms.variables is None:
                    raise(RuntimeError, "Error during dualization:  Constraint '%s' has a lower bound that is non-constant")
                if not upper_terms is None and not upper_terms.variables is None:
                    raise(RuntimeError, "Error during dualization:  Constraint '%s' has an upper bound that is non-constant")
                #
                for i in range(len(body_terms.variables)):
                    varname = body_terms.variables[i].component().name
                    varndx = body_terms.variables[i].index()
                    A.setdefault(body_terms.variables[i].component().name, {}).setdefault(varndx,[]).append( Bunch(coef=body_terms.linear[i], var=name, ndx=ndx) )
                    
                #
                if not con.equality:
                    #
                    # Inequality constraint
                    #
                    #if not (upper_terms is None or upper_terms.constant is None):
                    if lower_terms is None or lower_terms.constant is None:
                        #
                        # body <= upper
                        #
                        v = getattr(dual,name)
                        vardata = v.add(ndx)
                        v.domain = NonPositiveReals
                        b_coef[name,ndx] = upper_terms.constant - body_terms.constant
                    #elif not (lower_terms is None or lower_terms.constant is None):
                    elif upper_terms is None or upper_terms.constant is None:
                        #
                        # lower <= body
                        #
                        v = getattr(dual,name)
                        vardata = v.add(ndx)
                        v.domain = NonNegativeReals
                        b_coef[name,ndx] = lower_terms.constant - body_terms.constant
                    else:
                        #
                        # lower <= body <= upper
                        #
                        v = getattr(dual,name)
                        #
                        # Dual for lower bound
                        #
                        ndx_ = tuple(list(ndx).append('lb'))
                        vardata = v.add(ndx_)
                        vardata.domain = NonNegativeReals
                        b_coef[name,ndx] = lower_terms.constant - body_terms.constant
                        #
                        # Dual for upper bound
                        #
                        ndx_ = tuple(list(ndx).append('ub'))
                        vardata = v.add(ndx_)
                        vardata.domain = NonPositiveReals
                        b_coef[name,ndx] = upper_terms.constant - body_terms.constant
                else:
                    #
                    # Equality constraint
                    #
                    v = getattr(dual,name)
                    vardata = v.add(ndx)
                    v.domain = Reals
                    b_coef[name,ndx] = lower_terms.constant - body_terms.constant
        #
        # Collect bound constraints
        #
        for (name, data) in itertools.chain(block.active_components(Var).items(), block._parent().active_components(Var).items()):
            #
            # Skip fixed variables (in the parent)
            #
            if not (name, data.is_indexed()) in cnames:
                continue
            #
            # Iterate over all variable indices
            #
            for ndx in data:
                var = data[ndx]
                bounds = var.bounds
                if bounds[0] is None and bounds[1] is None:
                    c_sense[name,ndx] = 'e'
                elif bounds[0] is None:
                    if bounds[1] == 0.0:
                        c_sense[name,ndx] = 'g'
                    else:
                        c_sense[name,ndx] = 'e'
                        #
                        # Add constraint that defines the upper bound
                        #
                        name_ = name + "_upper_"
                        varname = data.component().name
                        varndx = data[ndx].index()
                        A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                        #
                        v = getattr(dual,name_)
                        vardata = v.add(ndx)
                        v.domain = NonPositiveReals
                        b_coef[name_,ndx] = bounds[1]
                elif bounds[1] is None:
                    if bounds[0] == 0.0:
                        c_sense[name,ndx] = 'l'
                    else:
                        c_sense[name,ndx] = 'e'
                        #
                        # Add constraint that defines the lower bound
                        #
                        name_ = name + "_lower_"
                        varname = data.component().name
                        #from coopr.pyomo.base.component import Component
                        varndx = data[ndx].index()
                        A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                        #
                        v = getattr(dual,name_)
                        vardata = v.add(ndx)
                        v.domain = NonNegativeReals
                        b_coef[name_,ndx] = bounds[0]
                else:
                    # Bounded above and below
                    c_sense[name,ndx] = 'e'
                    #
                    # Add constraint that defines the upper bound
                    #
                    name_ = name + "_upper_"
                    varname = data.component().name
                    varndx = data[ndx].index()
                    A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                    #
                    v = getattr(dual,name_)
                    vardata = v.add(ndx)
                    v.domain = NonPositiveReals
                    b_coef[name_,ndx] = bounds[1]
                    #
                    # Add constraint that defines the lower bound
                    #
                    name_ = name + "_lower_"
                    varname = data.component().name
                    varndx = data[ndx].index()
                    A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                    #
                    v = getattr(dual,name_)
                    vardata = v.add(ndx)
                    v.domain = NonNegativeReals
                    b_coef[name_,ndx] = bounds[0]
                    #raise IOError, "Variable bounded by (%s,%s)" % (str(bounds[0]), str(bounds[1]))
        #
        if d_sense == minimize:
            dual.o = Objective(expr=sum(- b_coef[name,ndx]*getattr(dual,name)[ndx] for name,ndx in b_coef), sense=d_sense)
        else:
            dual.o = Objective(expr=sum(b_coef[name,ndx]*getattr(dual,name)[ndx] for name,ndx in b_coef), sense=d_sense)
        #
        for cname in A:
            c = getattr(dual, cname)
            c_index = getattr(dual, cname+"_Index") if c.is_indexed() else None
            for ndx,terms in A[cname].iteritems():
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
        return dual

