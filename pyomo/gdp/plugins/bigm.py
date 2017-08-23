# -*- coding: UTF-8 -*-
"""Big-M Generalized Disjunctive Programming transformation module."""
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
import textwrap

from pyomo.core import (Block, Connector, Constraint, Param, Set, Suffix, Var,
                        value)
from pyomo.core.base import Transformation
from pyomo.core.base.block import SortComponents, TraversalStrategy, _BlockData
from pyomo.core.base.component import ComponentUID
from pyomo.core.base.set_types import Any
from pyomo.core.kernel import ComponentMap, ComponentSet
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import (IndexedDisjunction, SimpleDisjunction,
                                _DisjunctData, _DisjunctionData)
from pyomo.repn import LinearCanonicalRepn, generate_canonical_repn
from pyomo.util.modeling import unique_component_name
from pyomo.util.plugin import alias
from six import iterkeys

logger = logging.getLogger('pyomo.core')


class BigM_Transformation(Transformation):
    """Relax disjunctive model using big-M terms.

    Relaxes a disjunctive model into an algebraic model by adding Big-M terms
    to all disjunctive constraints.
    """

    alias('gdp.bigm', doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize transformation object."""
        super(BigM_Transformation, self).__init__()
        self.handlers = {
            Constraint:  self._xform_constraint,
            Var:         False,
            Connector:   False,
            Suffix:      False,
            Param:       False,
            Set:         False,
            Disjunction: self._warn_for_active_disjunction,
            Disjunct:    self._warn_for_active_disjunct,
            Block:       self._transform_block_on_disjunct,
        }

    def _get_bigm_suffix_list(self, block):
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
        # override ones anywhere in the tree. Suffixes lower down in the tree
        # override ones higher up.
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
                           % ('\n'.join(iterkeys(kwds)), ))
        if options:
            logger.warning("GDP(BigM): unrecognized options:\n%s"
                           % ('\n'.join(iterkeys(options)), ))

        # make a transformation block to put transformed disjuncts on
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_bigm_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = Block(Any)
        transBlock.lbub = Set(initialize=['lb', 'ub'])
        # this is a dictionary for keeping track of IndexedDisjuncts
        # and IndexedDisjunctions so that, at the end of the
        # transformation, we can check that the ones with no active
        # DisjstuffDatas are deactivated.
        transBlock.disjContainers = ComponentSet()

        if targets is None:
            targets = (instance, )
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
            if type(t) is _DisjunctionData:
                self._transformDisjunctionData(t, transBlock, bigM, t.index())
            elif type(t) is _DisjunctData:
                self._transformBlock(t, transBlock, bigM)
            elif type(t) is _BlockData or t.type() in (Block, Disjunct):
                self._transformBlock(t, transBlock, bigM)
            elif t.type() is Disjunction:
                self._transformDisjunction(t, transBlock, bigM)
            else:
                raise GDP_Error(
                    "Target %s was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed"
                    % (t.name, type(t)))

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

    def _transformBlock(self, obj, transBlock, bigM):
        if obj.is_indexed():
            for i in obj:
                self._transformBlockData(obj[i], transBlock, bigM)
        else:
            self._transformBlockData(obj, transBlock, bigM)

    def _transformBlockData(self, obj, transBlock, bigM):
        # Transform every (active) disjunction in the block
        for disjunction in obj.component_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=(Block, Disjunct),
                descent_order=TraversalStrategy.PostfixDFS):
            self._transformDisjunction(disjunction, transBlock, bigM)

    def _declareXorConstraint(self, disjunction):
        # Put the disjunction constraint on its parent block and
        # determine whether it is an OR or XOR constraint.

        # We never do this for just a DisjunctionData because we need
        # to know about the index set of its parent component. So if
        # we called this on a DisjunctionData, we did something wrong.
        assert type(disjunction) in (SimpleDisjunction,
                                     IndexedDisjunction)
        parent = disjunction.parent_block()
        if hasattr(parent, "_gdp_transformation_info"):
            infodict = parent._gdp_transformation_info
            if type(infodict) is not dict:
                raise GDP_Error(
                    "Component %s contains an attribute named "
                    "_gdp_transformation_info. "
                    "The transformation requires that "
                    "it can create this attribute!" % parent.name)
        else:
            infodict = parent._gdp_transformation_info = {}

        # add the XOR (or OR) constraints to parent block (with unique name)
        # It's indexed if this is an IndexedDisjunction, not otherwise
        orC = Constraint(disjunction.index_set()) if \
            disjunction.is_indexed() else Constraint()
        xor = disjunction.xor
        nm = '_xor' if xor else '_or'
        orCname = unique_component_name(parent, '_gdp_bigm_relaxation_' +
                                        disjunction.name + nm)
        parent.add_component(orCname, orC)
        infodict.setdefault('disjunction_or_constraint', {})[
            disjunction.local_name] = orC
        return orC, xor

    def _transformDisjunction(self, obj, transBlock, bigM):
        # create the disjunction constraint and then relax each of the
        # disjunctionDatas
        orConstraint, xor = self._declareXorConstraint(obj)
        if obj.is_indexed():
            transBlock.disjContainers.add(obj)
            for i in obj:
                self._transformDisjunctionData(obj[i], transBlock,
                                               bigM, i, orConstraint, xor)
        else:
            self._transformDisjunctionData(obj, transBlock, bigM,
                                           None, orConstraint, xor)

        # deactivate so we know we relaxed
        obj.deactivate()

    def _transformDisjunctionData(self, obj, transBlock, bigM,
                                  index, orConstraint=None, xor=None):
        parent_component = obj.parent_component()
        if xor is None:
            # If the orConstraint is already on the block fetch it.
            # Otherwise call _declareXorConstraint.
            parent_block = obj.parent_block()
            if hasattr(parent_block, "_gdp_transformation_info"):
                infodict = parent_block._gdp_transformation_info
                if type(infodict) is dict and \
                        'disjunction_or_constraint' in infodict:
                    orConsDict = parent_block._gdp_transformation_info[
                        'disjunction_or_constraint']
                    if parent_component.local_name in orConsDict:
                        # if the orConstraint has already been declared,
                        # we fetch it and get the value of xor from the
                        # parent.
                        orConstraint = orConsDict[parent_component.local_name]
                        xor = parent_component.xor
            if xor is None:
                # orConstraint wasn't already declared, so we declare it
                orConstraint, xor = self._declareXorConstraint(
                    obj.parent_component())
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var
            # make suffix list. (We don't need it until we are
            # transforming constraints, but it gets created at the
            # disjunct level, so more efficient to make it here and
            # pass it down.
            suffix_list = self._get_bigm_suffix_list(disjunct)
            # relax the disjunct
            self._bigM_relax_disjunct(disjunct, transBlock, bigM, suffix_list)
        # add or (or xor) constraint
        orConstraint.add(index, (1, or_expr, 1 if xor else None))
        obj.deactivate()

    def _bigM_relax_disjunct(self, obj, transBlock, bigM, suffix_list):
        if hasattr(obj, "_gdp_transformation_info"):
            infodict = obj._gdp_transformation_info
            # If the user has something with our name that is not a dict, we
            # scream. If they have a dict with this name then we are just going
            # to use it...
            if type(infodict) is not dict:
                raise GDP_Error(
                    "Disjunct %s contains an attribute named "
                    "_gdp_transformation_info. "
                    "The transformation requires that "
                    "it can create this attribute!" % obj.name)
        else:
            infodict = {}
        # deactivated -> either we've already transformed or user deactivated
        if not obj.active:
            if not infodict.get('relaxed', False):
                # If it hasn't been relaxed, user deactivated it and so we
                # fix ind var to 0 and be done. If it has been relaxed, we will
                # check if it was bigm that did it, and if not, we will apply
                # bigm.
                obj.indicator_var.fix(0)
                return
        if 'bigm' in infodict:
            # we've transformed it (with BigM), so don't do it again.
            return

        # add reference to original disjunct to info dict on transformation
        # block
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]
        # TODO: trans info should be a class that implements __getstate and
        # __setstate (for pickling)
        relaxationBlock._gdp_transformation_info = {'src': obj}

        # add reference to transformation block on original disjunct
        assert 'bigm' not in infodict
        infodict['bigm'] = relaxationBlock

        # if this is a disjunctData from an indexed disjunct, we are
        # going to want to check at the end that the container is
        # deactivated if everything in it is. So we save it in our
        # dictionary of things to check if it isn't there already.
        disjParent = obj.parent_component()
        if disjParent.is_indexed() and \
           disjParent not in transBlock.disjContainers:
            transBlock.disjContainers.add(disjParent)

        # Transform each component within this disjunct
        self._transform_block_components(obj, obj, relaxationBlock,
                                         bigM, infodict, suffix_list)

        # deactivate disjunct so we know we've relaxed it
        obj.deactivate()
        infodict['relaxed'] = True
        obj._gdp_transformation_info = infodict

    def _transform_block_components(self, block, disjunct, relaxedBlock,
                                    bigM, infodict, suffix_list):
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
                        "No BigM transformation handler registered "
                        "for modeling components of type %s" % obj.type())
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, relaxedBlock, bigM, infodict, suffix_list)

    def _warn_for_active_disjunction(self, disjunction, disjunct, relaxedBlock,
                                     bigMargs, infodict, suffix_list):
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
                        "disjunction before the disjunct in the list."
                        % (problemdisj.name, disjunct.name))

    def _warn_for_active_disjunct(self, innerdisjunct, outerdisjunct,
                                  relaxedBlock, bigMargs,
                                  infodict, suffix_list):
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
        raise GDP_Error("Found active disjunct {0} in disjunct {1}! "
                        "Either {0} "
                        "is not in a disjunction or the disjunction it is in "
                        "has not been transformed. "
                        "{0} needs to be deactivated "
                        "or its disjunction transformed before {1} can be "
                        "transformed.".format(problemdisj.name,
                                              outerdisjunct.name))

    def _transform_block_on_disjunct(self, block, disjunct, relaxationBlock,
                                     bigMargs, infodict, suffix_list):
        # We look through everything on the component map of the block
        # and transform it just as we would if it was on the disjunct
        # directly.  (We are passing the disjunct through so that when
        # we find constraints, _xform_constraint will have access to
        # the correct indicator variable.
        self._transform_block_components(block, disjunct, relaxationBlock,
                                         bigMargs, infodict, suffix_list)

    def _xform_constraint(self, obj, disjunct, relaxationBlock,
                          bigMargs, infodict, suffix_list):
        # add constraint to the transformation block, we'll transform it there.

        transBlock = relaxationBlock.parent_block()
        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name
        name = unique_component_name(relaxationBlock, obj.name)

        if obj.is_indexed():
            newConstraint = Constraint(obj.index_set(), transBlock.lbub)
        else:
            newConstraint = Constraint(transBlock.lbub)
        relaxationBlock.add_component(name, newConstraint)
        # add mapping of original constraint to transformed constraint
        # in transformation info dictionary
        infodict.setdefault('relaxedConstraints', ComponentMap())[
            obj] = newConstraint
        # add mapping of transformed constraint back to original constraint (we
        # know that the info dict is already created because this only got
        # called if we were transforming a disjunct...)
        relaxationBlock._gdp_transformation_info.setdefault(
            'srcConstraints', ComponentMap())[newConstraint] = obj

        for i in obj:
            c = obj[i]
            if not c.active:
                continue
            c.deactivate()

            # first, we see if an M value was specified in the arguments.
            # (This returns None if not)
            M = self._get_M_from_args(c, bigMargs)

            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "from the BigM argument is %s." % (obj.name,
                                                                str(M)))

            # if we didn't get something from args, try suffixes:
            if M is None:
                M = self._get_M_from_suffixes(c, suffix_list)

            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "after checking suffixes is %s." % (obj.name,
                                                                 str(M)))

            if not isinstance(M, (tuple, list)):
                if M is None:
                    M = (None, None)
                else:
                    M = (-M, M)
            if len(M) != 2:
                raise GDP_Error("Big-M %s for constraint %s is not of "
                                "length two. "
                                "Expected either a single value or "
                                "tuple or list of length two for M."
                                % (str(M), name))

            M = list(M)
            if c.lower is not None and M[0] is None:
                M[0] = self._estimate_M(c.body, name)[0] - c.lower
            if c.upper is not None and M[1] is None:
                M[1] = self._estimate_M(c.body, name)[1] - c.upper

            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "after estimating (if needed) is %s." %
                             (obj.name, str(M)))

            # TODO: The commented out code should work here after
            # issue #116 is resolved. As it is, I can't get this to
            # work because ('lb',) isn't the same as 'lb'... I get the
            # DeveloperError about IndexedConstraint failing to define
            # _default(). So for now I'll just check if the constraint
            # is indexed below.

            # if i.__class__ is tuple:
            #     pass
            # elif obj.is_indexed():
            #     i = (i,)
            # else:
            #     i = ()
            if c.lower is not None:
                if M[0] is None:
                    raise GDP_Error("Cannot relax disjunctive constraint %s "
                                    "because M is not defined." % name)
                M_expr = M[0] * (1 - disjunct.indicator_var)
                # newConstraint.add(i+('lb',), c.lower <= c. body - M_expr)
                if obj.is_indexed():
                    newConstraint.add((i, 'lb'), c.lower <= c.body - M_expr)
                else:
                    newConstraint.add('lb', c.lower <= c.body - M_expr)
            if c.upper is not None:
                if M[1] is None:
                    raise GDP_Error("Cannot relax disjunctive constraint %s "
                                    "because M is not defined." % name)
                M_expr = M[1] * (1 - disjunct.indicator_var)
                # newConstraint.add(i+('ub',), c.body - M_expr <= c.upper)
                if obj.is_indexed():
                    newConstraint.add((i, 'ub'), c.body - M_expr <= c.upper)
                else:
                    newConstraint.add('ub', c.body - M_expr <= c.upper)

    def _get_M_from_args(self, constraint, bigMargs):
        M = None
        # check args: we only have to look for constraint, constraintdata, and
        # None
        if bigMargs is not None:
            cuid = ComponentUID(constraint)
            parentcuid = ComponentUID(constraint.parent_component())
            if cuid in bigMargs:
                M = bigMargs[cuid]
            elif parentcuid in bigMargs:
                M = bigMargs[parentcuid]
            elif None in bigMargs:
                M = bigMargs[None]
        return M

    def _get_M_from_suffixes(self, constraint, suffix_list):
        M = None
        # first we check if the constraint or its parent is a key in any of the
        # suffix lists
        for bigm in suffix_list:
            if constraint in bigm:
                M = bigm[constraint]
                break

            # if c is indexed, check for the parent component
            if constraint.parent_component() in bigm:
                M = bigm[constraint.parent_component()]
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
        M = [0, 0]

        if isinstance(repn, LinearCanonicalRepn):
            if repn.constant is not None:
                for i in (0, 1):
                    if M[i] is not None:
                        M[i] += repn.constant

            for i, coef in enumerate(repn.linear or []):
                var = repn.variables[i]
                coef = repn.linear[i]
                bounds = (value(var.lb), value(var.ub))
                for i in (0, 1):
                    # reverse the bounds if the coefficient is negative
                    if coef > 0:
                        j = i
                    else:
                        j = 1 - i

                    if bounds[i] is not None:
                        M[j] += value(bounds[i]) * coef
                    else:
                        raise GDP_Error("Cannot estimate M for "
                                        "expressions with unbounded variables."
                                        "\n\t(found while processing "
                                        "constraint %s)" % name)
        else:
            raise GDP_Error("Cannot estimate M for nonlinear "
                            "expressions.\n\t(found while processing "
                            "constraint %s)" % name)

        return tuple(M)
