#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import itertools
from six import iteritems

from pyutilib.misc import Bunch
from pyomo.misc.plugin import alias
from pyomo.core.base import Transformation, Var, Constraint, VarList, ConstraintList, Objective, Set, maximize, minimize, NonNegativeReals, NonPositiveReals, Reals, Block, ComponentUID
from pyomo.core.expr.canonical_repn import generate_canonical_repn
from pyomo.core.expr.canonical_repn import LinearCanonicalRepn
from pyomo.core.plugins.transform.util import process_canonical_repn
from pyomo.bilevel import SubModel

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
        #
        # Start constructing the submodel
        #
        # Variables are constraints of submodel
        # Constraints are unfixed variables of submodel and the parent model.
        #
        vnames = set()
        for (name, data) in submodel.active_components(Constraint).items():
            vnames.add((name, data.is_indexed()))
        cnames = set(unfixed)
        for (name, data) in submodel.active_components(Var).items():
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
        for (oname, odata) in submodel.active_components(Objective).items():
            for ndx in odata:
                if odata[ndx].sense == maximize:
                    o_terms = generate_canonical_repn(-1*odata[ndx].expr, compute_values=False)
                    d_sense = minimize
                else:
                    o_terms = generate_canonical_repn(odata[ndx].expr, compute_values=False)
                    d_sense = maximize
                for i in range(len(o_terms.variables)):
                    c_rhs[ o_terms.variables[i].parent_component().name, o_terms.variables[i].index() ] = o_terms.linear[i]
            # Stop after the first objective
            break
        #
        # Collect constraints
        #
        for (name, data) in submodel.active_components(Constraint).items():
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
                    varname = body_terms.variables[i].parent_component().name
                    varndx = body_terms.variables[i].index()
                    A.setdefault(body_terms.variables[i].parent_component().name, {}).setdefault(varndx,[]).append( Bunch(coef=body_terms.linear[i], var=name, ndx=ndx) )
                    
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
        for (name, data) in itertools.chain(submodel.active_components(Var).items(), submodel._parent().active_components(Var).items()):
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
                        varname = data.parent_component().name
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
                        varname = data.parent_component().name
                        #from pyomo.core.base.component import Component
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
                    varname = data.parent_component().name
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
                    varname = data.parent_component().name
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
        return dual

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


