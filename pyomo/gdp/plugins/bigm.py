#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six.moves import xrange as range
from six import iteritems, iterkeys

from pyomo.util.plugin import alias
from pyomo.core import *
from pyomo.repn import *
from pyomo.core.base import Transformation
from pyomo.core.base.block import SortComponents
from pyomo.repn import LinearCanonicalRepn
from pyomo.gdp import *

from random import randint

import weakref
import logging
logger = logging.getLogger('pyomo.core')

# DEBUG
from nose.tools import set_trace

class BigM_Transformation(Transformation):

    alias('gdp.bigm', doc="Relaxes a disjunctive model into an algebraic model by adding Big-M terms to all disjunctive constraints.")

    def __init__(self):
        super(BigM_Transformation, self).__init__()
        self.handlers = {
            Constraint: self._xform_constraint,
            Var:       False,
            Connector: False,
            Suffix:    False,
            Param:     False,
            Set:       False,
            }


    # QUESTION: I copied and pasted this from add slacks for now, but is there somehwere it can live
    # so that code isn't duplicated?
    def _get_unique_name(self, instance, name):
        # test if this name already exists in model. If not, we're good. 
        # Else, we add random numbers until it doesn't
        while True:
            if not instance.component(name):
                return name
            else:
                name += str(randint(0,9))


    def get_bigm_suffix_list(self, block):
        suffix_list = []
        while block is not None:
            bigm = block.component('BigM')
            if bigm is not None and type(bigm) is Suffix:
                suffix_list.append(bigm)
            block = block.parent_block()
        return suffix_list


    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})

        # TODO: I'm going to pass the dictionary of args through until I need it
        # so all the way to transform constraint, I guess.
        bigM = options.pop('bigM', None)
        bigM = kwds.pop('bigM', bigM)
        # TODO: this is all changing so that we first use M's from args, then
        # suffixes, then estimate if we still don't have anything.
        # QUESTION: So is this making the args/options a suffix?
        # How does this work if the args are more specific?
        if bigM is not None:
            #
            # Test for the suffix - this test will (correctly) generate
            # a warning if the component is already declared, but is a
            # different ctype (e.g., a constraint or block)
            #
            if 'BigM' not in instance.component_map(Suffix):
                instance.BigM = Suffix(direction=Suffix.LOCAL)
            #
            # Note: this will implicitly change the model default BigM
            # value so that the argument overrides the option, which
            # overrides any default specified on the model.
            #
            instance.BigM[None] = bigM

        targets = kwds.pop('targets', None)

        if kwds:
            logger.warning("GDP(BigM): unrecognized keyword arguments:\n%s"
                           % ( '\n'.join(iterkeys(kwds)), ))
        if options:
            logger.warning("GDP(BigM): unrecognized options:\n%s"
                        % ( '\n'.join(iterkeys(options)), ))

        # make a transformation block to put transformed disjuncts on
        transBlockName = self._get_unique_name(instance, '_pyomo_gdp_relaxation')
        instance.add_component(transBlockName, Block(Any))
        transBlock = instance.component(transBlockName)
        transBlock.lbub = Set(initialize = ['lb','ub'])

        if targets is None:
            for block in instance.block_data_objects(
                    active=True, 
                    sort=SortComponents.deterministic ):
                self._transformBlock(block, transBlock)
        # ESJ: I've yet to touch this, and I don't get it yet...
        else:
            if isinstance(targets, Component):
                targets = (targets, )
            for _t in target:
                if not _t.active:
                    continue
                if _t.parent_component() is _t:
                    _name = _t.local_name
                    for _idx, _obj in _t.iteritems():
                        if _obj.active:
                            self._transformDisjunction(_name, _idx, _obj)
                else:
                    self._transformDisjunction(
                        _t.parent_component().local_name, _t.index(), _t)


    def _transformBlock(self, block, transBlock):
        # Transform every (active) disjunction in the block
        for disjunction in block.component_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic):
            self._transformDisjunction(disjunction, transBlock)

    
    def _transformDisjunction(self, obj, transBlock): 
        # Put the disjunction constraint on its parent block, then relax
        # each of the disjuncts
        
        parent = obj.parent_block()

        # add the XOR (or OR) constraints to parent block (with unique name)
        # It's indexed if this is an IndexedDisjunction.
        orC = Constraint(obj.index_set())
        # TODO: can you just add it here instead of constructing it?
        orC.construct()
        for i in obj.index_set():
            or_expr = 0
            for disjunct in obj[i].disjuncts:
                or_expr += disjunct.indicator_var
            c_expr = or_expr==1 if obj.xor else or_expr >= 1
            orC.add(i, c_expr)
        
        nm = '_xor' if obj.xor else '_or'
        orCname = self._get_unique_name(parent, '_pyomo_gdp_relaxation_' + \
                                            obj.name + nm)
        parent.add_component(orCname, orC)

        # relax each of the disjunctions (or the SimpleDisjunction if it wasn't indexed)
        for i in obj:
            #self._transformDisjunctionData(obj[i])
            for disjunct in obj[i].disjuncts:
                self._bigM_relax_disjunct(disjunct, transBlock)
            obj[i].deactivate()

        # deactivate so we know we relaxed
        obj.deactivate()


    def _bigM_relax_disjunct(self, disjunct, transBlock):
        infodict = disjunct.component("_gdp_trans_info")
        # deactivated means fix indicator var at 0 and be done
        if not disjunct.active:
            if infodict is None or type(infodict) is not dict or 'bigm' not in infodict:
                # if we haven't transformed it, assume user deactivated and fix ind var
                disjunct.indicator_var.fix(0)
            return
        
        m = disjunct.model()

        # add reference to original disjunct to info dict on transformation block
        disjBlock = transBlock[len(transBlock)]
        if not hasattr(disjBlock, "_gdp_trans_info"):
            disjBlock._gdp_trans_info = {}
        transdict = getattr(disjBlock, "_gdp_trans_info")
        transdict['src'] = weakref.ref(disjunct)

        # add reference to transformation block on original disjunct
        if not hasattr(disjunct, "_gdp_trans_info"):
            disjunct._gdp_trans_info = {}
        disjdict = getattr(disjunct, "_gdp_trans_info")
        disjdict['bigm'] = weakref.ref(disjBlock)
        
        # Transform each component within this disjunct
        for name, obj in list(disjunct.component_map().iteritems()):
            handler = self.handlers.get(obj.type(), None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No BigM transformation handler registered "
                        "for modeling components of type %s" % obj.type() )
                continue
            handler(name, obj, disjunct, disjBlock)
        
        # deactivate disjunct so we know we've relaxed it
        disjunct.deactivate()


    # TODO: don't need to be passing name through here...
    def _xform_constraint(self, _name, constraint, disjunct, disjBlock):
        # add constraint to the transformation block, we'll transform it there.

        # TODO: I don't think this is where we are actually getting bigm...?
        if 'BigM' in disjunct.component_map(Suffix):
            M = disjunct.component('BigM').get(constraint)
        else:
            M = None

        transBlock = disjBlock.parent_component()
        name = constraint.local_name
        
        if constraint.is_indexed():
            newC = Constraint(constraint.index_set(), transBlock.lbub)
        else:
            newC = Constraint(transBlock.lbub)
        disjBlock.add_component(name, newC)
        
        #for cname, c in iteritems(constraint._data):
        for i in constraint:
            c = constraint[i]
            if not c.active:
                continue
            c.deactivate()

            #name = _name + ('.'+str(cname) if cname is not None else '')

            if isinstance(M, list):
                if len(M):
                    m = M.pop(0)
                else:
                    m = (None,None)
            else:
                m = M
            if not isinstance(m, tuple):
                if m is None:
                    m = (None, None)
                else:
                    m = (-1*m,m)
            
            # If we need an M (either for upper and/or lower bounding of
            # the expression, then try and estimate it
            if ( c.lower is not None and m[0] is None ) or \
                   ( c.upper is not None and m[1] is None ):
                m = self._estimate_M(c.body, name, m, disjunct)

            # TODO: I can't get this to work because ('lb',) isn't the same as 'lb'...
            # I get the DeveloperError about IndexedConstraint failing to define
            # _default(). So for now I'll just check if the constraint's indexed below.
            # if i.__class__ is tuple:
            #     pass
            # elif constraint.is_indexed():
            #     i = (i,)
            # else:
            #     i = ()
            if c.lower is not None:
                if m[0] is None:
                    raise GDP_Error("Cannot relax disjunctive " + \
                          "constraint %s because M is not defined." % name)
                M_expr = (m[0]-c.lower)*(1-disjunct.indicator_var)
                #newC.add(i+('lb',), c.lower <= c. body - M_expr)
                if constraint.is_indexed():
                    newC.add((i, 'lb'), c.lower <= c.body - M_expr)
                else:
                    newC.add('lb', c.lower <= c.body - M_expr)
            if c.upper is not None:
                if m[1] is None:
                    raise GDP_Error("Cannot relax disjunctive " + \
                          "constraint %s because M is not defined." % name)
                M_expr = (m[1]-c.upper)*(1-disjunct.indicator_var)
                #newC.add(i+('ub',), c.body - M_expr <= c.upper)
                if constraint.is_indexed():
                    newC.add((i, 'ub'), c.body - M_expr <= c.upper)
                else:
                    newC.add('ub', c.body - M_expr <= c.upper)


    def _estimate_M(self, expr, name, m, disjunct):
        # Calculate a best guess at M
        repn = generate_canonical_repn(expr)
        M = [0,0]

        if isinstance(repn, LinearCanonicalRepn):
            if repn.constant != None:
                for i in (0,1):
                    if M[i] is not None:
                        M[i] += repn.constant

            for i, coef in enumerate(repn.linear or []):
                var = repn.variables[i]
                coef = repn.linear[i]
                bounds = (value(var.lb), value(var.ub))
                for i in (0,1):
                    # reverse the bounds if the coefficient is negative
                    if coef > 0:
                        j = i
                    else:
                        j = 1-i

                    try:
                        M[j] += value(bounds[i]) * coef
                    except:
                        M[j] = None
        else:
            logger.info("GDP(BigM): cannot estimate M for nonlinear "
                        "expressions.\n\t(found while processing %s)",
                        name)
            M = [None,None]


        # Allow user-defined M values to override the estimates
        for i in (0,1):
            if m[i] is not None:
                M[i] = m[i]

        # Search for global BigM values: if there are still undefined
        # M's, then search up the block hierarchy for the first block
        # that contains a BigM Suffix with a non-None value for the
        # "None" component.
        if None in M:
            m = None
            while m is None and disjunct is not None:
                if 'BigM' in disjunct.component_map(Suffix):
                    m = disjunct.component('BigM').get(None)
                disjunct = disjunct.parent_block()
            if m is not None:
                try:
                    # We always allow M values to be specified as pairs
                    # (for lower / upper bounding)
                    M = [m[i] if x is None else x for i,x in enumerate(M)]
                except:
                    # We assume the default M is positive (so we need to
                    # invert it for the lower-bound M)
                    M = [(2*i-1)*m if x is None else x for i,x in enumerate(M)]

        return tuple(M)

