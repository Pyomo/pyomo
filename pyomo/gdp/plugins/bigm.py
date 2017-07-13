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
from pyomo.core.base.block import SortComponents, _BlockData
from pyomo.repn import LinearCanonicalRepn
from pyomo.gdp import *

from random import randint

import weakref
import logging
logger = logging.getLogger('pyomo.core')

# DEBUG
from nose.tools import set_trace

class BigM_Transformation(Transformation):

    alias('gdp.bigm', doc="Relaxes a disjunctive model into an algebraic model "
          "by adding Big-M terms to all disjunctive constraints.")

    def __init__(self):
        super(BigM_Transformation, self).__init__()
        self.handlers = {
            Constraint: self._xform_constraint,
            Var:        False,
            Connector:  False,
            Suffix:     False,
            Param:      False,
            Set:        False,
            Disjunction: False,
            Disjunct:   False, # These are OK because we are always
                              # traversing with postfix DFS, so if we
                              # encounter a disjunct inside something,
                              # it has already been transformed if it
                              # was going to be.
            }


    # QUESTION: I copied and pasted this from add slacks for now, but is there 
    # somehwere it can live so that code isn't duplicated?
    def _get_unique_name(self, instance, name):
        # test if this name already exists in model. If not, we're good. 
        # Else, we add random numbers until it doesn't
        while True:
            if instance.component(name) is None:
                return name
            else:
                name += str(randint(0,9))


    def get_bigm_suffix_list(self, block):
        # Note that you can only specify suffixes on BlockData objects or
        # SimpleBlocks. Though it is possible at this point to stick them
        # on whatever components you want, we won't pick them up.
        suffix_list = []
        while block is not None:
            bigm = block.component('BigM')
            if type(bigm) is Suffix:
                suffix_list.append(bigm)
            block = block.parent_block()
        return suffix_list


    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})

        # For now, we're not accepting options. We will let args override 
        # suffixes and estimate as a last resort. More specific args/suffixes 
        # override ones higher up in the tree.
        bigM = kwds.pop('bigM', None)
        if bigM is not None and type(bigM) is not dict:
            if type(bigM) in (float, int, tuple, list):
                bigM = {None: bigM}
            else:
                raise GDP_Error(
                    "'bigM' argument was not a dictionary! Expected cuids as "
                    "keys and big-m values (or tuples) as values.")

        targets = kwds.pop('targets', None)

        if kwds:
            logger.warning("GDP(BigM): unrecognized keyword arguments:\n%s"
                           % ( '\n'.join(iterkeys(kwds)), ))
        if options:
            logger.warning("GDP(BigM): unrecognized options:\n%s"
                        % ( '\n'.join(iterkeys(options)), ))

        # make a transformation block to put transformed disjuncts on
        transBlockName = self._get_unique_name(
            instance, 
            '_pyomo_gdp_bigm_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = Block(Any)
        transBlock.lbub = Set(initialize = ['lb','ub'])

        if targets is None:
            targets = ( instance, )
        for _t in targets:
            t = _t.find_component(instance)
            if t is None:
                raise GDP_Error(
                    "Target %s is not a component on the instance!" % _t)
            if not t.active:
                continue
            # TODO: This is compensating for Issue #185. I do need to
            # check if something is a DisjunctData, but the other
            # places where I am checking type I would like to only
            # check ctype.
            if type(t) is disjunct._DisjunctionData:
                self._transformDisjunctionData(t, transBlock, bigM, t.index())
            elif type(t) is disjunct._DisjunctData:
                self._transformBlock(t, transBlock, bigM)
            elif type(t) is _BlockData or t.type() in (Block, Disjunct):
                self._transformBlock(t, transBlock, bigM)
            elif t.type() is Disjunction:
                self._transformDisjunction(t, transBlock, bigM)
            else:
                raise GDP_Error(
                    "Target %s was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed" 
                    % (t.name, type(t)) )


    def _transformBlock(self, block, transBlock, bigM):
        if block.is_indexed():
            for i in block:
                self._transformBlockData(block[i], transBlock, bigM)
        else:
            self._transformBlockData(block, transBlock, bigM)


    def _transformBlockData(self, block, transBlock, bigM):
         # Transform every (active) disjunction in the block
            for disjunction in block.component_objects(
                    Disjunction,
                    active=True,
                    sort=SortComponents.deterministic,
                    descend_into=(Block,Disjunct),
                    descent_order=TraversalStrategy.PostfixDFS):
                self._transformDisjunction(disjunction, transBlock, bigM)

    
    def _declareXorConstraint(self, obj):
        # Put the disjunction constraint on its parent block and
        # determine whether it is an OR or XOR constraint.
        parent = obj.parent_block()
        if hasattr(parent, "_gdp_transformation_info"):
            infodict = parent._gdp_transformation_info
            if type(infodict) is not dict:
                raise GDP_Error(
                    "Component %s contains an attribute named "
                    "_gdp_transformation_info. The transformation requires that "
                    "it can create this attribute!" % parent.name)
        else:
            infodict = parent._gdp_transformation_info = {}

        # add the XOR (or OR) constraints to parent block (with unique name)
        # It's indexed if this is an IndexedDisjunction, not otherwise
        orC = Constraint(obj.index_set()) if obj.is_indexed() else Constraint()
        if hasattr(obj, 'xor'):
            xor = obj.xor
        else:
            assert type(obj) is disjunct._DisjunctionData
            xor = obj.parent_component().xor
        nm = '_xor' if xor else '_or'
        orCname = self._get_unique_name(parent, '_gdp_bigm_relaxation_' + \
                                        obj.local_name + nm)
        parent.add_component(orCname, orC)
        infodict[obj.name] = weakref.ref(orC)
        return orC, xor

        
    def _transformDisjunction(self, obj, transBlock, bigM): 
        # create the disjunction constraint and then relax each of the
        # disjunctionDatas
        orC, xor = self._declareXorConstraint(obj)
        if obj.is_indexed():
            for i in obj:
                self._transformDisjunctionData(obj[i], transBlock,
                                               bigM, i, orC, xor)
        else:
            self._transformDisjunctionData(obj, transBlock, bigM, None, orC, xor)

        # TODO: update the map dictionaries to include the 
        # disjunction <-> constraint maps
        # and the constraint <-> transformed constraint maps
       
        # deactivate so we know we relaxed
        obj.deactivate()


    def _transformDisjunctionData(self, obj, transBlock, bigM,
                                  index, orConstraint=None, xor=None):
        parent_component = obj.parent_component()
        if xor is None:
            # 1) if the orConstraint is aldready on the block (using
            # the dictionary that you haven't finished yet) fetch it.
            # Otherwise call _declareXorConstraint.
            # 2) fetch xor
            parent_block = obj.parent_block()
            if (hasattr(parent_block, "_gdp_transformation_info") and 
            type(parent_block._gdp_transformation_info) is dict):
                infodict = parent_block._gdp_transformation_info
                if parent_component.name in infodict:
                    orConstraint = infodict[parent_component.name]()
                    xor = parent_component.xor
            #if (( orConstraint has already been declared)):
                #fetch it.
                #set xor from parent.
            if xor is None:
                orConstraint, xor = self._declareXorConstraint(
                    obj.parent_component() )
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var
            # relax the disjunct
            self._bigM_relax_disjunct(disjunct, transBlock, bigM)
        # add or (or xor) constraint
        orConstraint.add(index, (1, or_expr, 1 if xor else None))
        obj.deactivate()


    def _bigM_relax_disjunct(self, disjunct, transBlock, bigM):
        if hasattr(disjunct, "_gdp_transformation_info"):
            infodict = disjunct._gdp_transformation_info
            # If the user has something with our name that is not a dict, we 
            # scream. If they have a dict with this name then we are just going 
            # to use it...
            if type(infodict) is not dict:
                raise GDP_Error(
                    "Disjunct %s contains an attribute named "
                    "_gdp_transformation_info. The transformation requires that "
                    "it can create this attribute!" % disjunct.name)
        else:
            infodict = {}
        # deactivated means either we've already transformed or user deactivated
        if not disjunct.active:
            if not infodict.get('relaxed', False):
                # If it hasn't been relaxed, user deactivated it and so we 
                # fix ind var to 0 and be done. If it has been relaxed, we will
                # check if it was bigm that did it, and if not, we will apply
                # bigm.
                disjunct.indicator_var.fix(0)
                return
        if 'bigm' in infodict:
            # we've transformed it, so don't do it again.
            return
        
        m = disjunct.model()

        # add reference to original disjunct to info dict on transformation block
        disjuncts = transBlock.relaxedDisjuncts
        relaxedBlock = disjuncts[len(disjuncts)]
        # TODO: trans info should be a class that implements __getstate and 
        # __setstate (for pickling)
        relaxedBlock._gdp_transformation_info = {'src': weakref.ref(disjunct)}

        # add reference to transformation block on original disjunct
        assert 'bigm' not in infodict
        infodict['bigm'] = weakref.ref(relaxedBlock)
        
        # Transform each component within this disjunct
        for name, obj in list(disjunct.component_map().iteritems()):
            handler = self.handlers.get(obj.type(), None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No BigM transformation handler registered "
                        "for modeling components of type %s" % obj.type() )
                continue
            handler(obj, disjunct, relaxedBlock, bigM, infodict)
        
        # deactivate disjunct so we know we've relaxed it
        disjunct.deactivate()
        infodict['relaxed'] = True
        disjunct._gdp_transformation_info = infodict


    def _xform_constraint(self, constraint, disjunct, relaxedBlock,
                          bigMargs, infodict):
        # add constraint to the transformation block, we'll transform it there.

        transBlock = relaxedBlock.parent_block()
        name = constraint.name
        infodict['relaxedConstraints'] = {}
        
        if constraint.is_indexed():
            newC = Constraint(constraint.index_set(), transBlock.lbub)
        else:
            newC = Constraint(transBlock.lbub)
        relaxedBlock.add_component(name, newC)
        # add mapping of original constraint to transformed constraint
        # in transformation info dictionary
        infodict['relaxedConstraints'][
            ComponentUID(constraint)] = weakref.ref(newC)
        
        for i in constraint:
            c = constraint[i]
            if not c.active:
                continue
            c.deactivate()

            # first, we see if an M value was specified in the arguments.
            # (This returns None if not)
            M = self._get_M_from_args(c, bigMargs)
            
            # DEBUG
            # print("after args, M is: ")
            # print(M)
            
            # if we didn't get something from args, try suffixes:
            # TODO: wouldn't hurt to generate the suffix list at the disjunct 
            # level and pass it through to use here.
            if M is None:
                M = self._get_M_from_suffixes(c)
                
            # DEBUG
            # print("after suffixes, M is: ")
            # print(M)
           
            if not isinstance(M, (tuple, list)):
                if M is None:
                    M = (None, None)
                else:
                    M = (-M, M)
            if len(M) != 2:
                raise GDP_Error("Big-M %s for constraint %s is not of "
                                "length two. Expected either a single value or "
                                "tuple or list of length two for M." 
                                % (str(M), name))

            M = list(M)
            if c.lower is not None and M[0] is None:
                M[0] = self._estimate_M(c.body, name)[0] - c.lower
            if c.upper is not None and M[1] is None:
                M[1] = self._estimate_M(c.body, name)[1] - c.upper

            # DEBUG
            # print("after estimating, m is: ")
            # print(M)

            # TODO: I can't get this to work because ('lb',) isn't the same as 
            # 'lb'... I get the DeveloperError about IndexedConstraint failing 
            # to define _default(). So for now I'll just check if the constraint
            # is indexed below.
            # if i.__class__ is tuple:
            #     pass
            # elif constraint.is_indexed():
            #     i = (i,)
            # else:
            #     i = ()
            if c.lower is not None:
                if M[0] is None:
                    raise GDP_Error("Cannot relax disjunctive " + \
                          "constraint %s because M is not defined." % name)
                M_expr = M[0]*(1 - disjunct.indicator_var)
                #newC.add(i+('lb',), c.lower <= c. body - M_expr)
                if constraint.is_indexed():
                    newC.add((i, 'lb'), c.lower <= c.body - M_expr)
                else:
                    newC.add('lb', c.lower <= c.body - M_expr)
            if c.upper is not None:
                if M[1] is None:
                    raise GDP_Error("Cannot relax disjunctive " + \
                          "constraint %s because M is not defined." % name)
                M_expr = M[1]*(1-disjunct.indicator_var)
                #newC.add(i+('ub',), c.body - M_expr <= c.upper)
                if constraint.is_indexed():
                    newC.add((i, 'ub'), c.body - M_expr <= c.upper)
                else:
                    newC.add('ub', c.body - M_expr <= c.upper)


    def _get_M_from_args(self, cons, bigMargs):
        M = None
        # check args: we only have to look for constraint, constraintdata, and 
        # None
        if bigMargs is not None:
            cuid = ComponentUID(cons)
            parentcuid = ComponentUID(cons.parent_component())
            if cuid in bigMargs:
                M = bigMargs[cuid]
            elif parentcuid in bigMargs:
                M = bigMargs[parentcuid]
            elif None in bigMargs:
                M = bigMargs[None]
        return M


    def _get_M_from_suffixes(self, cons):
        M = None
        # make suffix list
        suffix_list = self.get_bigm_suffix_list(cons.parent_block())
        # first we check if the constraint or its parent is a key in any of the
        # suffix lists
        for bigm in suffix_list:
            if cons in bigm:
                M = bigm[cons]
                break
        
            # if c is indexed, check for the parent component
            if cons.parent_component() in bigm:
                M = bigm[cons.parent_component()]
                break

        # if we didn't get an M that way, traverse upwards through the blocks 
        # and see if None has a value on any of them.
        if M is None:
            for bigm in suffix_list:
                if None in bigm:
                    M = bigm[None]
                    break
        return M


    def _estimate_M(self, expr, name):
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

                    # try:
                    #     M[j] += value(bounds[i]) * coef
                    # except:
                    #     M[j] = None
                    if bounds[i] is not None:
                        M[j] += value(bounds[i]) * coef
                    else:
                        raise GDP_Error("Cannot estimate M for "
                                        "expressions with unbounded variables."
                                        "\n\t(found while processing constraint "
                                        "%s)" % name)
        else:
            raise GDP_Error("Cannot estimate M for nonlinear "
                            "expressions.\n\t(found while processing constraint "
                            "%s)" % name)

        return tuple(M)

