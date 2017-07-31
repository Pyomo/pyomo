#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import weakref
import logging

from pyomo.util.plugin import alias
from pyomo.core import *
from pyomo.core.base import expr as EXPR, Transformation
from pyomo.core.base.block import SortComponents, _BlockData
from pyomo.core.base import _ExpressionData
from pyomo.core.base.var import _VarData
from pyomo.repn import generate_canonical_repn, LinearCanonicalRepn
from pyomo.core.kernel import ComponentMap, ComponentSet
from pyomo.core.kernel.expr_common import clone_expression
from pyomo.core.base.expr import identify_variables
from pyomo.gdp import *

from six import iteritems, iterkeys

# DEBUG
from nose.tools import set_trace

logger = logging.getLogger('pyomo.core')

# NL_Mode_LeeGrossmann is the original NL convex hull from Lee &
# Grossmann (2000), which substitutes nonlinear constraints
#
#     h_ik(x) <= 0
#
# with
#
#     x_k = sum( nu_ik )
#     y_ik * h_ik( nu_ik/y_ik ) <= 0
#
# [1] Lee, S., & Grossmann, I. E. (2000). New algorithms for nonlinear
# generalized disjunctive programming.  Computers and Chemical
# Engineering, 24, 2125-2141
#
NL_Mode_LeeGrossmann = 1
#
# NL_Mode_GrossmannLee is an updated formulation from Grossmann & Lee
# (2003), which substitutes nonlinear constraints
#
#     h_ik(x) <= 0
#
# with
#
#     x_k = sum( nu_ik )
#     (y_ik + eps) * h_ik( nu_ik/(y_ik + eps) ) <= 0
#
# [2] Grossmann, I. E., & Lee, S. (2003). Generalized disjunctive
# programming: Nonlinear convex hull relaxation and algorithms.
# Computational Optimization and Applications, 26, 83-100.
#
NL_Mode_GrossmannLee = 2
#
# NL_Mode_FurmanSawayaGrossmann is an improved relaxation that avoids
# numerical issues from the Lee & Grossmann formulation by using:
#
#     x_k = sum( nu_ik )
#     ((1-eps)*y_ik + eps) * h_ik( nu_ik/((1-eps)*y_ik + eps) ) \
#        - eps * r_ki(0) * ( 1-y_ik ) <= 0
#
# [3] Furman, K., Sawaya, N., and Grossmann, I.  A computationally
# useful algebraic representation of nonlinear disjunctive convex sets
# using the perspective function.  Optimization Online (2016).
#
# http://www.optimization-online.org/DB_HTML/2016/07/5544.html.
#
NL_Mode_FurmanSawayaGrossmann = 3

EPS = 1e-2

class ConvexHull_Transformation(Transformation):

    alias('gdp.chull', doc="Relaxes a disjunctive model into an algebraic "
          "model by forming the convex hull relaxation of each disjunction.")

    def __init__(self):
        super(ConvexHull_Transformation, self).__init__()
        self.handlers = {
            Constraint : self._xform_constraint,
            Var :        False,
            Connector :  False,
            Param :      False,
            Set :        False,
            Suffix :     False,
            Disjunction: self._warn_for_active_disjunction,
            Disjunct:    self._warn_for_active_disjunct,
            Block:       self._transform_block_on_disjunct,
            }

        # TODO: I don't know what this is. I doubt I need it.
        self._promote_vars = []
        #self._mode = NL_Mode_LeeGrossmann
        #self._mode = NL_Mode_GrossmannLee
        self._mode = NL_Mode_FurmanSawayaGrossmann


    # ESJ: TODO: this is copied and pasted for now but going somewhere else
    # later.
    def _get_unique_name(self, instance, name):
        # test if this name already exists in model. If not, we're good. 
        # Else, we add random numbers until it doesn't
        while True:
            if instance.component(name) is None:
                return name
            else:
                name += str(randint(0,9))


    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})
        targets = kwds.pop('targets', None)

        if kwds:
            logger.warning("GDP(CHull): unrecognized keyword arguments:\n\t%s"
                           % ( '\n\t'.join(iterkeys(kwds)), ))

        # We don't accept any options at the moment.
        if options:
            logger.warning("GDP(CHull): unrecognized options:\n%s"
                        % ( '\n'.join(iterkeys(options)), ))

        # make a transformation block
        transBlockName = self._get_unique_name(
            instance,
            '_pyomo_gdp_chull_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = Block(Any)
        transBlock.lbub = Set(initialize = ['lb','ub'])
        transBlock.disjContainers = ComponentSet()

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
                self._transformDisjunctionData(t, transBlock, t.index())
            elif type(t) is disjunct._DisjunctData:
                self._transformBlock(t, transBlock)
            elif type(t) is _BlockData or t.type() in (Block, Disjunct):
                self._transformBlock(t, transBlock)
            elif t.type() is Disjunction:
                self._transformDisjunction(t, transBlock)
            else:
                raise GDP_Error(
                    "Target %s was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed" 
                    % (t.name, type(t)) )

        # Go through our dictionary of indexed things and deactivate
        # the containers that don't have any active guys inside of
        # them. So the invalid component logic will tell us if we
        # missed something getting transformed.
        for obj in transBlock.disjContainers:
            if obj.active:
                for i in obj:
                    if obj[i].active:
                        break
                else:
                    obj.deactivate()


    def _transformBlock(self, obj, transBlock):
        if obj.is_indexed():
            for i in obj:
                self._transformBlockData(obj[i], transBlock)
        else:
            self._transformBlockData(obj, transBlock)


    def _transformBlockData(self, obj, transBlock):
        # Transform every (active) disjunction in the block
        for disjunction in obj.component_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=(Block,Disjunct),
                descent_order=TraversalStrategy.PostfixDFS):
            self._transformDisjunction(disjunction, transBlock)


    def _declareDisjunctionConstraints(self, disjunction):
        # Put the disjunction constraint on its parent block and
        # determine whether it is an OR or XOR constraint.

        # We never do this for just a DisjunctionData because we need
        # to know about the index set of its parent component. So if
        # we called this on a DisjunctionData, we did something wrong.
        assert type(disjunction) in (disjunct.SimpleDisjunction,
                             disjunct.IndexedDisjunction)
        parent = disjunction.parent_block()
        if hasattr(parent, "_gdp_transformation_info"):
            infodict = parent._gdp_transformation_info
            if type(infodict) is not dict:
                raise GDP_Error(
                    "Component %s contains an attribute named "
                    "_gdp_transformation_info. The transformation requires that "
                    "it can create this attribute!" % parent.name)
        else:
            infodict = parent._gdp_transformation_info = {}

        # add the disaggregation constraint
        disaggregationConstraint = Constraint(Any)
        parent.add_component(
            self._get_unique_name(parent, '_gdp_chull_relaxation_' + \
                                  disjunction.name + '_disaggregation'), 
            disaggregationConstraint)

        # add the XOR (or OR) constraints to parent block (with unique name)
        # It's indexed if this is an IndexedDisjunction, not otherwise
        orC = Constraint(disjunction.index_set()) if \
              disjunction.is_indexed() else Constraint()
        xor = disjunction.xor
        # Convex hull doesn't work if this is an or constriant. So if
        # xor is false, give up
        if not xor:
            raise GDP_Error("Cannot do convex hull transformation for "
                            "disjunction %s with or constraint. Must be an xor!"
                            % disjunction.name)
        nm = '_xor'
        orCname = self._get_unique_name(parent, '_gdp_chull_relaxation_' + \
                                        disjunction.name + nm)
        parent.add_component(orCname, orC)
        infodict.setdefault('disjunction_or_constraint', {})[
            disjunction.local_name] = orC
        infodict.setdefault('disjunction_disaggregation_constraints', {})[
            disjunction.local_name] = disaggregationConstraint
        return orC, disaggregationConstraint


    def _transformDisjunction(self, obj, transBlock): 
        # create the disjunction constraint and disaggregation
        # constraints and then relax each of the disjunctionDatas
        orConstraint, disaggregationConstraint = \
        self._declareDisjunctionConstraints(obj)
        if obj.is_indexed():
            transBlock.disjContainers.add(obj)
            for i in obj:
                self._transformDisjunctionData(obj[i], transBlock, i,
                                               orConstraint,
                                               disaggregationConstraint)
        else:
            self._transformDisjunctionData(obj, transBlock, None,
                                           orConstraint,
                                           disaggregationConstraint)
       
        # deactivate so we know we relaxed
        obj.deactivate()


    def _transformDisjunctionData(self, obj, transBlock, index,
                                  orConstraint=None,
                                  disaggregationConstraint=None):
        parent_component = obj.parent_component()
        if orConstraint is None:
            parent_bock = obj.parent_block()
            if (hasattr(parent_block, "_gdp_transformation_info") and 
            type(parent_block._gdp_transformation_info) is dict):
                infodict = parent_block._gdp_transformation_info
                if parent_component.name in infodict:
                    # if the orConstraint and disaggregation
                    # constraints have already been declared, we fetch
                    # them.
                    orConstraint = infodict['disjunction_or_constraint'][
                        parent_component.local_name]
                    disaggregationConstraint = infodict[
                        'disjunction_disaggregation_constraints'][
                            parent_component.local_name]
            if orConstraint is None:
                # orConstraint wasn't already declared, so we declare
                # it and the disaggregationConstraint
                orConstraint, disaggregationConstraint = \
                self._declareDisjunctionConstraints(obj.parent_component() )

        # We first go through and collect all the variables that we
        # are going to disaggregate.
        varSet = ComponentSet()
        for disjunct in obj.disjuncts:
            for cons in disjunct.component_objects(
                    Constraint,
                    active = True,
                    sort=SortComponents.deterministic,
                    descend_into=Block):
                # we aren't going to disaggregate fixed
                # variables. This means there is trouble if they are
                # unfixed later...  
                # TODO: we need a way to identify
                # Expressions and throw an error if they are there.
                vars = identify_variables(cons.body, include_fixed=False)
                varSet.update(vars)

        # Now that we know who we need to disaggregate, we will do it
        # while we also transform the disjuncts.
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var
            self._transform_disjunct(disjunct, transBlock, varSet)
         
        orConstraint.add(index, (1, or_expr, 1))
        for var in varSet:
            disaggregatedExpr = 0
            for disjunct in obj.disjuncts:
                disaggregatedVar = disjunct._gdp_transformation_info[
                    'disaggregatedVars'][var]
                disaggregatedExpr += disaggregatedVar
            consName = self._get_unique_name(disjunct, var.name)
            consIdx = index + (consName,) if index is not None else consName
            disaggregationConstraint.add(
                consIdx,
                var == disaggregatedExpr)


    def _transform_disjunct(self, obj, transBlock, varSet):
        if hasattr(obj, "_gdp_transformation_info"):
            infodict = obj._gdp_transformation_info
            # If the user has something with our name that is not a dict, we 
            # scream. If they have a dict with this name then we are just going 
            # to use it...
            if type(infodict) is not dict:
                raise GDP_Error(
                    "Disjunct %s contains an attribute named "
                    "_gdp_transformation_info. The transformation requires that "
                    "it can create this attribute!" % obj.name)
        else:
            infodict = {}
        # deactivated means either we've already transformed or user deactivated
        if not obj.active:
            if not infodict.get('relaxed', False):
                # If it hasn't been relaxed, user deactivated it and so we 
                # fix ind var to 0 and be done. If it has been relaxed, we will
                # check if it was chull that did it, and if not, we will apply
                # chull.
                obj.indicator_var.fix(0)
                return
        if 'chull' in infodict:
            # we've transformed it (with BigM), so don't do it again.
            return

        # add reference to original disjunct to info dict on transformation block
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]
        relaxationBlock._gdp_transformation_info = {'src': obj}
        infodict['chull'] = relaxationBlock

        # if this is a disjunctData from an indexed disjunct, we are
        # going to want to check at the end that the container is
        # deactivated if everything in it is. So we save it in our
        # dictionary of things to check if it isn't there already.
        disjParent = obj.parent_component()
        if disjParent.is_indexed() and \
           disjParent not in transBlock.disjContainers:
            transBlock.disjContainers.add(disjParent)

        # add the disaggregated variables and their bigm constraints
        # to the relaxationBlock
        infodict['disaggregatedVars'] = ComponentMap()
        infodict['bigmConstraints'] = ComponentMap()
        for var in varSet:
            disaggregatedVar = Var(within=Reals)
            # naming conflicts are possible here since this is a bunch
            # of variables from different blockscoming together, so we
            # get a unique name
            disaggregatedVarName = self._get_unique_name(obj, var.local_name)
            relaxationBlock.add_component(disaggregatedVarName, disaggregatedVar)
            infodict['disaggregatedVars'][var] = disaggregatedVar
            lb = var.lb
            ub = var.ub
            if lb is None or ub is None:
                raise GDP_Error("Variables that appear in disjuncts must be "
                                "bounded in order to use the chull "
                                "transfromation! Missing bound for %s on "
                                "disjunct %s." % (var.name, disjunct.name))
            bigmConstraint = Constraint(transBlock.lbub)
            bigmConstraint.add('lb', obj.indicator_var*lb <= disaggregatedVar)
            bigmConstraint.add('ub', disaggregatedVar <= obj.indicator_var*ub)
            relaxationBlock.add_component(
                '_bounds_' + disaggregatedVarName, bigmConstraint)
            infodict['bigmConstraints'][var] = bigmConstraint

        var_substitute_map = dict((id(v), newV) for v, newV in 
                                  iteritems(infodict['disaggregatedVars']))
        zero_substitute_map = dict((id(v), NumericConstant(0)) for v, newV in 
                                   iteritems(infodict['disaggregatedVars']))
        
        # Transform each component within this disjunct
        self._transform_block_components(obj, obj, relaxationBlock,
                                         infodict, var_substitute_map,
                                         zero_substitute_map)
        
        # deactivate disjunct so we know we've relaxed it
        obj.deactivate()
        infodict['relaxed'] = True
        obj._gdp_transformation_info = infodict
        

    def _transform_block_components(self, block, disjunct,
                                    relaxedBlock, infodict,
                                    var_substitute_map,
                                    zero_substitute_map):
        # Look through the component map of block and transform
        # everything we have a handler for. Yell if we don't know how
        # to handle it.
        for name, obj in list(block.component_map().iteritems()):
            if hasattr(obj, 'active') and not obj.active:
                continue
            handler = self.handlers.get(obj.type(), None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No chull transformation handler registered "
                        "for modeling components of type %s" % obj.type() )
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, relaxedBlock, infodict,
                    var_substitute_map, zero_substitute_map)


    def _warn_for_active_disjunction(self, disjunction, disjunct,
                                     relaxedBlock, infodict,
                                     var_substitute_map,
                                     zero_substitute_map):
        # this should only have gotten called if the disjunction is active
        assert disjunction.active
        problemdisj = disjunction
        if disjunction.is_indexed():
            for i in disjunction:
                if disjunction[i].active:
                    # a _DisjunctionData is active, we will yell about
                    # it specifically.
                    problemdisj = disjunction[i]
                    break
            # None of the _DisjunctionDatas were actually active. We
            # are OK and we can deactivate the container.
            else:
                disjunction.deactivate()
                return
        parentblock = problemdisj.parent_block()
        # the disjunction should only have been active if it wasn't transformed
        assert (not hasattr(parentblock, "_gdp_transformation_info")) or \
            problemdisj.name not in parentblock._gdp_transformation_info
        raise GDP_Error("Found untransformed disjunction %s in disjunct %s! "
                        "The disjunction must be transformed before the "
                        "disjunct. If you are using targets, put the "
                        "disjunction before the disjunct in the list." \
                        % (problemdisj.name, disjunct.name))


    def _warn_for_active_disjunct(self, innerdisjunct, outerdisjunct,
                                  relaxedBlock, infodict,
                                  var_substitute_map,
                                  zero_substitute_map):
        assert innerdisjunct.active
        problemdisj = innerdisjunct
        if innerdisjunct.is_indexed():
            for i in innerdisjunct:
                if innerdisjunct[i].active:
                    # This is shouldn't be true, we will complain about it.
                    problemdisj = innerdisjunct[i]
                    break
            # None of the _DisjunctDatas were actually active, so we
            # are fine and we can deactivate the container.
            else:
                innerdisjunct.deactivate()
                return
        raise GDP_Error("Found active disjunct {0} in disjunct {1}! Either {0} "
                        "is not in a disjunction or the disjunction it is in "
                        "has not been transformed. {0} needs to be deactivated "
                        "or its disjunction transformed before {1} can be "
                        "transformed.".format(problemdisj.name, 
                                              outerdisjunct.name))


    def _transform_block_on_disjunct(self, block, disjunct,
                                     relaxationBlock, infodict,
                                     var_substitute_map,
                                     zero_substitute_map):
        # We look through everything on the component map of the block
        # and transform it just as we would if it was on the disjunct
        # directly.  (We are passing the disjunct through so that when
        # we find constraints, _xform_constraint will have access to
        # the correct indicator variable.
        self._transform_block_components(block, disjunct, relaxationBlock,
                                            varSet, infodict)


    def _xform_constraint(self, obj, disjunct, relaxationBlock,
                          infodict, var_substitute_map, zero_substitute_map):
        # we will put a new transformed constraint on the relaxation block.

        transBlock = relaxationBlock.parent_block()
        varMap = infodict['disaggregatedVars']

        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name and we record the
        # mapping.
        name = self._get_unique_name(relaxationBlock, obj.name)
        infodict['relaxedConstraints'] = {}
        
        if obj.is_indexed():
            newConstraint = Constraint(obj.index_set(), transBlock.lbub)
        else:
            newConstraint = Constraint(transBlock.lbub)
        relaxationBlock.add_component(name, newConstraint)
        # add mapping of original constraint to transformed constraint
        # in transformation info dictionary
        infodict['relaxedConstraints'][obj] = newConstraint

        for i in obj:
            c = obj[i]
            if not c.active:
                continue
            c.deactivate()
        
            NL = c.body.polynomial_degree() not in (0,1)

            # We need to evaluate the expression at the origin *before*
            # we substitute the expression variables with the
            # disaggregated variables
            if not NL or self._mode == NL_Mode_FurmanSawayaGrossmann:
                h_0 = clone_expression(c.body, substitute=zero_substitute_map)

            expr = clone_expression(c.body, substitute=var_substitute_map)
            y = disjunct.indicator_var
            if NL:
                if self._mode == NL_Mode_LeeGrossmann:
                    expr = expr * y
                elif self._mode == NL_Mode_GrossmannLee:
                    expr = (y + EPS) * expr
                elif self._mode == NL_Mode_FurmanSawayaGrossmann:
                    expr = ((1-EPS)*y + EPS)*expr - EPS*h_0*(1-y)
                else:
                    raise RuntimeError("Unknown NL CHull mode")

            if c.lower is not None:
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(cHull): Transforming constraint " +
                                 "'%s'", c.name)
                if NL:
                    newConsExpr = expr >= c.lower*y
                else:
                    newConsExpr = expr - (1-y)*h_0 >= c.lower*y

                if obj.is_indexed():
                    newConstraint.add((i, 'lb'), newConsExpr)
                else:
                    newConstraint.add('lb', newConsExpr)
                
            if c.upper is not None:
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(cHull): Transforming constraint " +
                                 "'%s'", c.name)
                if NL:
                    newConsExpr = expr <= c.upper*y
                else:
                    newConsExpr = expr - (1-y)*h_0 <= c.upper*y

                if obj.is_indexed():
                    newConstraint.add((i, 'ub'), newConsExpr)
                else:
                    newConstraint.add('ub', newConsExpr)
