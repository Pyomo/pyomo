#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools

from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.modeling import unique_component_name

from pyomo.core import (
    Block, BooleanVar, Connector, Constraint, Expression, ExternalFunction,
    maximize, NonNegativeIntegers, Objective, Param, RangeSet, Set, SetOf,
    SortComponents, Suffix, value, Var
)
from pyomo.core.base import Reference, Transformation, TransformationFactory
from pyomo.core.base.boolean_var import (
    _DeprecatedImplicitAssociatedBinaryVariable)
import pyomo.core.expr.current as EXPR
from pyomo.core.util import target_list

from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import (
    get_gdp_tree
)
from pyomo.network import Port
from pyomo.opt import SolverFactory, TerminationCondition

from weakref import ref as weakref_ref

## DEBUG
from pytest import set_trace

@TransformationFactory.register(
    'gdp.mbigm', 
    doc="Relax disjunctive model using big-M terms specific to each disjunct")
class MultipleBigMTransformation(Transformation):
    """
    Implements the multiple big-M transformation from [1]. Note that this 
    transformation is no different than the big-M transformation for two-
    term disjunctions, but that it may provide a tighter relaxation for 
    models containing some disjunctions with three or more terms.


    [1] Francisco Trespalaios and Ignacio E. Grossmann, "Improved Big-M
        reformulation for generalized disjunctive programs," Computers and
        Chemical Engineering, vol. 76, 2015, pp. 98-103
    """

    CONFIG = ConfigBlock('gdp.mbigm')
    CONFIG.declare('targets', ConfigValue(
        default=None,
        domain=target_list,
        description="target or list of targets that will be relaxed",
        doc="""

        This specifies the list of components to relax. If None (default), the
        entire model is transformed. Note that if the transformation is done out
        of place, the list of targets should be attached to the model before it
        is cloned, and the list will specify the targets on the cloned
        instance."""
    ))
    CONFIG.declare('assume_fixed_vars_permanent', ConfigValue(
        default=False,
        domain=bool,
        description="Boolean indicating whether or not to transform so that "
        "the transformed model will still be valid when fixed Vars are "
        "unfixed.",
        doc="""
        This is only relevant when the transformation will be calculating M
        values. If True, the transformation will calculate M values assuming 
        that fixed variables will always be fixed to their current values. This
        means that if a fixed variable is unfixed after transformation, the
        transformed model is potentially no longer valid. By default, the
        transformation will assume fixed variables could be unfixed in the
        future and will use their bounds to calculate the M value rather than
        their value. Note that this could make for a weaker LP relaxation
        while the variables remain fixed.
        """
    ))
    CONFIG.declare('solver', ConfigValue(
        default=SolverFactory('gurobi'),
        description="A solver to use to solve the continuous subproblems for "
        "calculating the M values",
    ))

    def __init__(self):
        super(MultipleBigMTransformation, self).__init__()
        self.handlers = {
            Constraint:  self._transform_constraint,
            Var:         False, # Note that if a Var appears on a Disjunct, we
                                # still treat its bounds as global. If the
                                # intent is for its bounds to be on the
                                # disjunct, it should be declared with no bounds
                                # and the bounds should be set in constraints on
                                # the Disjunct.
            BooleanVar:  False,
            Connector:   False,
            Expression:  False,
            Suffix:      False,
            Param:       False,
            Set:         False,
            SetOf:       False,
            RangeSet:    False,
            Disjunction: False,# It's impossible to encounter an active
                               # Disjunction because preprocessing would have
                               # put it before its parent Disjunct in the order
                               # of transformation.
            Disjunct:    self._warn_for_active_disjunct,
            Block:       False,
            ExternalFunction: False,
            Port:        False, # not Arcs, because those are deactivated after
                                # the network.expand_arcs transformation
        }
        self._transformation_blocks = {}

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._transformation_blocks.clear()

    def _apply_to_impl(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error("Transformation called on %s of type %s. 'instance'"
                            " must be a ConcreteModel, Block, or Disjunct (in "
                            "the case of nested disjunctions)." %
                            (instance.name, instance.ctype))

        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        targets = self._config.targets
        knownBlocks = {}
        if targets is None:
            targets = (instance, )
            
        # We will transform from leaf to root, but we have to transform a
        # Disjunction at a time because, more similarly to hull than bigm, we
        # need information from the other Disjuncts in the Disjunction.
        gdp_tree = get_gdp_tree(targets, instance, knownBlocks)
        preprocessed_targets = gdp_tree.reverse_topological_sort()

        # transform any logical constraints that might be anywhere on the stuff
        # we're about to transform.
        TransformationFactory('core.logical_to_linear').apply_to(
            instance,
            targets=[blk for blk in targets if blk.ctype is Block] +
            [disj for disj in preprocessed_targets if disj.ctype is Disjunct])

        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(
                    t, t.index(), parent_disjunct=gdp_tree.parent(t),
                    root_disjunct=gdp_tree.root_disjunct(t))

    def _transform_disjunctionData(self, obj, index, parent_disjunct,
                                   root_disjunct):
        if not obj.xor:
            # This transformation assumes it can relax constraints assuming that
            # another Disjunct is chosen. If it could be possible to choose both
            # then that logic might fail.
            raise GDP_Error("Cannot do multiple big-M reformulation for "
                            "Disjunction '%s' with OR constraint.  "
                            "Must be an XOR!" % obj.name)

        # Create or fetch the transformation block
        if root_disjunct is not None:
            # We want to put all the transformed things on the root
            # Disjunct's parent's block so that they do not get
            # re-transformed
            transBlock = self._add_transformation_block(
                root_disjunct.parent_block())
        else:
            # This isn't nested--just put it on the parent block.
            transBlock = self._add_transformation_block(obj.parent_block())

        # Get the (possibly indexed) algebraic constraint for this disjunction
        algebraic_constraint = self._add_exactly_one_constraint(
            obj.parent_component(), transBlock)

        # Just because it's unlikely this is what someone meant to do...
        if len(obj.disjuncts) == 0:
            raise GDP_Error("Disjunction '%s' is empty. This is "
                            "likely indicative of a modeling error."  %
                            obj.getname(fully_qualified=True))
        
        Ms = transBlock.calculated_m_values = self.\
             _calculate_disjunction_M_values(obj)
                
        ## Here's the actual transformation
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var.get_associated_binary()
            self._transform_disjunct(disjunct, transBlock, obj.disjuncts, Ms)
        rhs = 1 if parent_disjunct is None else \
              parent_disjunct.binary_indicator_var
        algebraic_constraint.add(index, (or_expr, rhs))
        # map the DisjunctionData to its XOR constraint to mark it as
        # transformed
        obj._algebraic_constraint = weakref_ref(algebraic_constraint[index])

        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock, all_disjuncts, Ms):
        # We're not using the preprocessed list here, so this could be
        # inactive. We've already done the error checking in preprocessing, so
        # we just skip it here.
        if not obj.active:
            return

        # create a relaxation block for this disjunct
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]

        relaxationBlock.localVarReferences = Block()

        # add the map that will link back and forth between transformed
        # constraints and their originals.
        relaxationBlock._constraintMap = {
            'srcConstraints': ComponentMap(),
            'transformedConstraints': ComponentMap()
        }

        # add mappings to source disjunct (so we'll know we've relaxed)
        obj._transformation_block = weakref_ref(relaxationBlock)
        relaxationBlock._srcDisjunct = weakref_ref(obj)

        self._transform_block_components(obj, all_disjuncts, Ms)

        # deactivate disjunct so writers can be happy
        obj._deactivate_without_fixing_indicator()

    def _transform_block_components(self, disjunct, all_disjuncts, Ms):
        # We don't know where all the BooleanVars are used, so if there are any
        # that logical_to_linear didn't transform, we need to do it now
        for boolean in disjunct.component_data_objects(BooleanVar,
                                                       descend_into=Block,
                                                       active=None):
            if isinstance(boolean._associated_binary,
                          _DeprecatedImplicitAssociatedBinaryVariable):
                parent_block = boolean.parent_block()
                new_var = Var(domain=Binary)
                parent_block.add_component(
                    unique_component_name(parent_block,
                                          boolean.local_name + "_asbinary"),
                    new_var)
                boolean.associate_binary_var(new_var)

        # add references to all local variables on block (including the
        # indicator_var). We won't have to do this when the writers can find
        # Vars not in the active subtree.
        varRefBlock = disjunct._transformation_block().localVarReferences
        for v in disjunct.component_objects(Var, descend_into=Block,
                                            active=None):
            varRefBlock.add_component(unique_component_name(
                varRefBlock, v.getname(fully_qualified=True)), Reference(v))

        # Look through the component map of block and transform everything we
        # have a handler for. Yell if we don't know how to handle it. (Note that
        # because we only iterate through active components, this means
        # non-ActiveComponent types cannot have handlers.)
        for obj in disjunct.component_objects(active=True, descend_into=Block):
            handler = self.handlers.get(obj.ctype, None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No muliple bigM transformation handler registered "
                        "for modeling components of type %s. If your "
                        "disjuncts contain non-GDP Pyomo components that "
                        "require transformation, please transform them first."
                        % obj.ctype )
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, all_disjuncts, Ms)

    def _warn_for_active_disjunct(self, innerdisjunct, outerdisjunct, Ms):
        _warn_for_active_disjunct(innerdisjunct, outerdisjunct)

    def _transform_constraint(self, obj, disjunct, all_disjuncts, Ms):
        # we will put a new transformed constraint on the relaxation block.
        relaxationBlock = disjunct._transformation_block()
        constraintMap = relaxationBlock._constraintMap
        transBlock = relaxationBlock.parent_block()

        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name
        name = unique_component_name(relaxationBlock, obj.getname(
            fully_qualified=True))

        if obj.is_indexed():
            newConstraint = Constraint(obj.index_set(), transBlock.lbub)
        else:
            newConstraint = Constraint(transBlock.lbub)
        relaxationBlock.add_component(name, newConstraint)
        # add mapping of original constraint to transformed constraint
        if obj.is_indexed():
            constraintMap['transformedConstraints'][obj] = newConstraint
        # add mapping of transformed constraint container back to original
        # constraint container (or ScalarConstraint)
        constraintMap['srcConstraints'][newConstraint] = obj

        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue

            if c.lower is not None:
                rhs = sum(Ms[c,
                             disj][0]*disj.indicator_var.get_associated_binary()
                          for disj in all_disjuncts if disj is not disjunct)
                if obj.is_indexed():
                    newConstraint.add((i, 'lb'), c.body - c.lower >= rhs)
                else:
                    newConstraint.add('lb', c.body - c.lower >= rhs)

            if c.upper is not None:
                rhs = sum(Ms[c,
                             disj][1]*disj.indicator_var.get_associated_binary()
                          for disj in all_disjuncts if disj is not disjunct)
                if obj.is_indexed():
                    newConstraint.add((i, 'ub'), c.body - c.upper <= rhs)
                else:
                    newConstraint.add('ub', c.body - c.upper <= rhs)
        
        # deactivate now that we have transformed
        obj.deactivate()

    def _add_transformation_block(self, block):
        if block in self._transformation_blocks:
            return self._transformation_blocks[block]

        # make a transformation block on instance where we will store
        # transformed components
        transBlockName = unique_component_name(
            block,
            '_pyomo_gdp_hull_reformulation')
        transBlock = Block()
        block.add_component(transBlockName, transBlock)
        self._transformation_blocks[block] = transBlock
        transBlock.relaxedDisjuncts = Block(NonNegativeIntegers)
        transBlock.lbub = Set(initialize = ['lb','ub'])

        return transBlock

    def _add_exactly_one_constraint(self, disjunction, transBlock):
        # Put XOR constraint on the transformation block

        # check if the constraint already exists
        if disjunction._algebraic_constraint is not None:
            return disjunction._algebraic_constraint()

        # add the XOR constraints to parent block (with unique name) It's
        # indexed if this is an IndexedDisjunction, not otherwise
        orC = Constraint(disjunction.index_set())
        transBlock.add_component(
            unique_component_name(transBlock,
                                  disjunction.getname(
                                      fully_qualified=True) + '_xor'), orC)
        disjunction._algebraic_constraint = weakref_ref(orC)

        return orC

    def _get_all_var_objects(self, disjunction):
        # This is actually a general utility for getting all Vars that appear in
        # active Disjuncts in a Disjunction.
        seen = set()
        for disj in disjunction.disjuncts:
            if not disj.active:
                # TODO: This should depend on the fixed var promise I
                # guess... Safest would actually be to include them...
                continue
            for constraint in disj.component_data_objects(
                    Constraint,
                    active=True,
                    sort=SortComponents.deterministic,
                    descend_into=Block):
                for var in EXPR.identify_variables(
                        constraint.expr,
                        include_fixed=True):
                    if id(var) not in seen:
                        seen.add(id(var))
                        yield var

    def _calculate_disjunction_M_values(self, obj):
        scratch_blocks = {}
        Ms = {}
        all_vars = list(self._get_all_var_objects(obj))
        del_later = []
        for disjunct, other_disjunct in itertools.product(obj.disjuncts,
                                                          obj.disjuncts):
            if ((disjunct is other_disjunct) or (not disjunct.active) or 
                (not other_disjunct.active)):
                continue
            if id(other_disjunct) in scratch_blocks:
                scratch = scratch_blocks[id(other_disjunct)]
            else:
                # TODO: there's no point in scratch being a Block since I need
                # to put references on the Disjunct itself. Should just make it
                # the objective.
                scratch = scratch_blocks[id(other_disjunct)] = Block()
                other_disjunct.add_component(
                    unique_component_name(other_disjunct, "scratch"), scratch)
                scratch.obj = Objective(expr=0) # placeholder, but I want to
                                                # take the name before I add a
                                                # bunch of random reference
                                                # objects.

                # If the writers don't assume Vars are declared on the Block
                # being solved, we won't need this!
                for v in all_vars:
                    ref = Reference(v)
                    del_later.append(ref)
                    other_disjunct.add_component(
                        unique_component_name(other_disjunct, v.name), ref)

            for constraint in disjunct.component_data_objects(
                    Constraint,
                    active=True,
                    descend_into=Block,
                    sort=SortComponents.deterministic):
                (lower_M, upper_M) = (None, None)
                if constraint.lower is not None:
                    body = constraint.body - constraint.lower
                    scratch.obj.expr = body
                    results = self._config.solver.solve(other_disjunct)
                    if results.solver.termination_condition is not \
                       TerminationCondition.optimal:
                        raise RuntimeError(
                            "Unsuccessful solve to calculate M value to relax "
                            "constraint '%s' on Disjunct '%s' when Disjunct "
                            "'%s' is selected." % (constraint.name, 
                                                   disjunct.name, 
                                                   other_disjunct.name))
                    lower_M = value(scratch.obj.expr)
                if constraint.upper is not None:
                    body = constraint.body - constraint.upper
                    scratch.obj.expr = body
                    scratch.obj.sense = maximize
                    results = self._config.solver.solve(other_disjunct)
                    if results.solver.termination_condition is not \
                       TerminationCondition.optimal:
                        raise RuntimeError(
                            "Unsuccessful solve to calculate M value to relax "
                            "constraint '%s' on Disjunct '%s' when Disjunct "
                            "'%s' is selected." % (constraint.name, 
                                                   disjunct.name, 
                                                   other_disjunct.name))
                    upper_M = value(scratch.obj.expr)
                Ms[constraint, other_disjunct] = (lower_M, upper_M)

        # clean up the scratch blocks
        for blk in scratch_blocks.values():
            blk.parent_block().del_component(blk)
        for ref in del_later:
            ref.parent_block().del_component(ref)

        return Ms
