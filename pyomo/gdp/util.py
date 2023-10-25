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

from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct

import pyomo.core.expr as EXPR
from pyomo.core.base.component import _ComponentBase
from pyomo.core import (
    Block,
    Suffix,
    TraversalStrategy,
    SortComponents,
    LogicalConstraint,
    value,
)
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
from pyomo.opt import TerminationCondition, SolverStatus

from weakref import ref as weakref_ref
from collections import defaultdict
import logging

logger = logging.getLogger('pyomo.gdp')

_acceptable_termination_conditions = set(
    [
        TerminationCondition.optimal,
        TerminationCondition.globallyOptimal,
        TerminationCondition.locallyOptimal,
    ]
)
_infeasible_termination_conditions = set(
    [TerminationCondition.infeasible, TerminationCondition.invalidProblem]
)


class NORMAL(object):
    pass


class INFEASIBLE(object):
    pass


class NONOPTIMAL(object):
    pass


def verify_successful_solve(results):
    status = results.solver.status
    term = results.solver.termination_condition

    if status == SolverStatus.ok and term in _acceptable_termination_conditions:
        return NORMAL
    elif term in _infeasible_termination_conditions:
        return INFEASIBLE
    else:
        return NONOPTIMAL


def clone_without_expression_components(expr, substitute=None):
    """A function that is used to clone an expression.

    Cloning is roughly equivalent to calling ``copy.deepcopy``.
    However, the :attr:`clone_leaves` argument can be used to
    clone only interior (i.e. non-leaf) nodes in the expression
    tree.   Note that named expression objects are treated as
    leaves when :attr:`clone_leaves` is :const:`True`, and hence
    those subexpressions are not cloned.

    This function uses a non-recursive
    logic, which makes it more scalable than the logic in
    ``copy.deepcopy``.

    Args:
        expr: The expression that will be cloned.
        substitute (dict): A dictionary mapping object ids to
            objects.  This dictionary has the same semantics as
            the memo object used with ``copy.deepcopy``.  Defaults
            to None, which indicates that no user-defined
            dictionary is used.

    Returns:
        The cloned expression.
    """
    if substitute is None:
        substitute = {}
    #
    visitor = EXPR.ExpressionReplacementVisitor(
        substitute=substitute, remove_named_expressions=True
    )
    return visitor.walk_expression(expr)


def _raise_disjunct_in_multiple_disjunctions_error(disjunct, disjunction):
    # we've transformed it, which means this is the second time it's appearing
    # in a Disjunction
    raise GDP_Error(
        "The disjunct '%s' has been transformed, but '%s', a disjunction "
        "it appears in, has not. Putting the same disjunct in "
        "multiple disjunctions is not supported." % (disjunct.name, disjunction.name)
    )


class GDPTree:
    """
    Stores a forest representing the hierarchy between GDP components on a
    model: for single-level GDPs, each tree is rooted at a Disjunction and
    each of the Disjuncts in the Disjunction is a leaf. For nested GDPs, the
    Disjuncts may not be leaves, and could have child Disjunctions of their
    own.
    """

    def __init__(self):
        self._children = {}
        self._in_degrees = {}
        # Every node has exactly one or 0 parents.
        self._parent = {}

        self._root_disjunct = {}
        # This needs to be ordered so that topological sort is deterministic
        self._vertices = OrderedSet()

    @property
    def vertices(self):
        return self._vertices

    def add_node(self, u):
        self._vertices.add(u)

    def parent(self, u):
        """Returns the parent node of u, or None if u is a root.

        Arg:
            u : A node in the tree
        """
        if u not in self._vertices:
            raise ValueError(
                "'%s' is not a vertex in the GDP tree. Cannot "
                "retrieve its parent." % u
            )
        if u in self._parent:
            return self._parent[u]
        else:
            return None

    def children(self, u):
        """Returns the direct descendents of node u.

        Arg:
            u : A node in the tree
        """
        return self._children[u]

    def parent_disjunct(self, u):
        """Returns the parent Disjunct of u, or None if u is the
        closest-to-root Disjunct in the forest.

        Arg:
            u : A node in the forest
        """
        return self.parent(self.parent(u))

    def root_disjunct(self, u):
        """Returns the highest parent Disjunct in the hierarchy, or None if
        the component is not nested.

        Arg:
            u : A node in the tree
        """
        rootmost_disjunct = None
        parent = self.parent(u)
        while True:
            if parent is None:
                return rootmost_disjunct
            if isinstance(parent, _DisjunctData) or parent.ctype is Disjunct:
                rootmost_disjunct = parent
            parent = self.parent(parent)

    def add_node(self, u):
        if u not in self._children:
            self._children[u] = OrderedSet()
        self._vertices.add(u)

    def add_edge(self, u, v):
        self.add_node(u)
        self.add_node(v)
        self._children[u].add(v)
        if v in self._parent and self._parent[v] is not u:
            _raise_disjunct_in_multiple_disjunctions_error(v, u)
        self._parent[v] = u

    def _visit_vertex(self, u, leaf_to_root):
        if u in self._children:
            for v in self._children[u]:
                if v not in leaf_to_root:
                    self._visit_vertex(v, leaf_to_root)
        # we're done--we've been to all its children
        leaf_to_root.add(u)

    def _reverse_topological_iterator(self):
        # this returns nodes of the tree ordered so that no node is before any
        # of its descendents.
        leaf_to_root = OrderedSet()
        for u in self.vertices:
            if u not in leaf_to_root:
                self._visit_vertex(u, leaf_to_root)

        return leaf_to_root

    def topological_sort(self):
        return list(reversed(self._reverse_topological_iterator()))

    def reverse_topological_sort(self):
        return self._reverse_topological_iterator()

    def in_degree(self, u):
        if u not in self._parent:
            return 0
        return 1

    def is_leaf(self, u):
        if len(self._children[u]) == 0:
            return True
        return False

    @property
    def leaves(self):
        for u, children in self._children.items():
            if len(children) == 0:
                yield u

    @property
    def disjunct_nodes(self):
        for v in self._vertices:
            if isinstance(v, _DisjunctData) or v.ctype is Disjunct:
                yield v


def _parent_disjunct(obj):
    parent = obj.parent_block()
    while parent is not None:
        if parent.ctype is Disjunct:
            return parent
        parent = parent.parent_block()

    return None


def _check_properly_deactivated(disjunct):
    if disjunct.indicator_var.is_fixed():
        if not value(disjunct.indicator_var):
            # The user cleanly deactivated the disjunct: there
            # is nothing for us to do here.
            return
        else:
            raise GDP_Error(
                "The disjunct '%s' is deactivated, but the "
                "indicator_var is fixed to %s. This makes no sense."
                % (disjunct.name, value(disjunct.indicator_var))
            )
    if disjunct._transformation_block is None:
        raise GDP_Error(
            "The disjunct '%s' is deactivated, but the "
            "indicator_var is not fixed and the disjunct does not "
            "appear to have been transformed. This makes no sense. "
            "(If the intent is to deactivate the disjunct, fix its "
            "indicator_var to False.)" % (disjunct.name,)
        )


def _gather_disjunctions(block, gdp_tree, include_root=True):
    if not include_root:
        # The argument 'block' may be a root node (it was in the list of
        # targets): We will not add it to the tree in this call. It may be added
        # later if in fact it is a descendent of another target, but as far as
        # we know now, it does not belong in the tree.
        root = block
    to_explore = [block]
    while to_explore:
        block = to_explore.pop()
        for disjunction in block.component_data_objects(
            Disjunction,
            active=True,
            sort=SortComponents.deterministic,
            descend_into=Block,
        ):
            # add the node (because it might be an empty Disjunction and block
            # might be a Block, in case it wouldn't get added below.)
            gdp_tree.add_node(disjunction)
            for disjunct in disjunction.disjuncts:
                if not disjunct.active:
                    if disjunct.transformation_block is not None:
                        _raise_disjunct_in_multiple_disjunctions_error(
                            disjunct, disjunction
                        )
                    _check_properly_deactivated(disjunct)
                    continue
                gdp_tree.add_edge(disjunction, disjunct)
                to_explore.append(disjunct)
            if block.ctype is Disjunct:
                if not include_root and block is root:
                    continue
                gdp_tree.add_edge(block, disjunction)

    return gdp_tree


def get_gdp_tree(targets, instance, knownBlocks=None):
    if knownBlocks is None:
        knownBlocks = {}
    gdp_tree = GDPTree()
    for t in targets:
        # first check it's not insane, that is, it is at least on the instance
        if not is_child_of(parent=instance, child=t, knownBlocks=knownBlocks):
            raise GDP_Error(
                "Target '%s' is not a component on instance "
                "'%s'!" % (t.name, instance.name)
            )
        if t.ctype is Block or isinstance(t, _BlockData):
            _blocks = t.values() if t.is_indexed() else (t,)
            for block in _blocks:
                if not block.active:
                    continue
                gdp_tree = _gather_disjunctions(block, gdp_tree, include_root=False)
        elif t.ctype is Disjunction:
            parent = _parent_disjunct(t)
            if parent is not None and parent in targets:
                gdp_tree.add_edge(parent, t)
            _disjunctions = t.values() if t.is_indexed() else (t,)
            for disjunction in _disjunctions:
                if disjunction.algebraic_constraint is not None:
                    # It's already transformed.
                    continue
                gdp_tree.add_node(disjunction)
                for disjunct in disjunction.disjuncts:
                    if not disjunct.active:
                        if disjunct.transformation_block is not None:
                            _raise_disjunct_in_multiple_disjunctions_error(
                                disjunct, disjunction
                            )
                        _check_properly_deactivated(disjunct)
                        continue
                    gdp_tree.add_edge(disjunction, disjunct)
                    gdp_tree = _gather_disjunctions(disjunct, gdp_tree)
        else:
            # There's nothing else we care about, so we don't know how to
            # deal with this
            raise GDP_Error(
                "Target '%s' was not a Block, Disjunct, or Disjunction. "
                "It was of type %s and can't be transformed." % (t.name, type(t))
            )
    return gdp_tree


def preprocess_targets(targets, instance, knownBlocks, gdp_tree=None):
    if gdp_tree is None:
        gdp_tree = get_gdp_tree(targets, instance, knownBlocks)
    # this is for bigm: We need to transform from the leaves up, so we want a
    # reverse of a topological sort: no parent can come before its child.
    return gdp_tree.reverse_topological_sort()


# [ESJ 07/09/2019 Should this be a more general utility function elsewhere?  I'm
#  putting it here for now so that all the gdp transformations can use it.
#  Returns True if child is a node or leaf in the tree rooted at parent, False
#  otherwise. Accepts list of known components in the tree and updates this list
#  to enhance performance in future calls. Note that both child and parent must
#  be blocks!
def is_child_of(parent, child, knownBlocks=None):
    # Note: we can get away with a dictionary and not ComponentMap because we
    # will only store Blocks (or their ilk), and Blocks are hashable (only
    # derivatives of NumericValue are not hashable)
    if knownBlocks is None:
        knownBlocks = {}
    tmp = set()
    node = child if isinstance(child, (Block, _BlockData)) else child.parent_block()
    while True:
        known = knownBlocks.get(node)
        if known:
            knownBlocks.update({c: True for c in tmp})
            return True
        if known is not None and not known:
            knownBlocks.update({c: False for c in tmp})
            return False
        if node is parent:
            knownBlocks.update({c: True for c in tmp})
            return True
        if node is None:
            knownBlocks.update({c: False for c in tmp})
            return False

        tmp.add(node)
        container = node.parent_component()
        if container is node:
            node = node.parent_block()
        else:
            node = container


def _to_dict(val):
    if isinstance(val, (dict, ComponentMap)):
        return val
    return {None: val}


def get_src_disjunction(xor_constraint):
    """Return the Disjunction corresponding to xor_constraint

    Parameters
    ----------
    xor_constraint: Constraint, which must be the logical constraint
                    (located on the transformation block) of some
                    Disjunction
    """
    # NOTE: This is indeed a linear search through the Disjunctions on the
    # model. I am leaving it this way on the assumption that asking XOR
    # constraints for their Disjunction is not going to be a common
    # question. If we ever need efficiency then we should store a reverse
    # map from the XOR constraint to the Disjunction on the transformation
    # block while we do the transformation. And then this method could query
    # that map.
    m = xor_constraint.model()
    for disjunction in m.component_data_objects(
        Disjunction, descend_into=(Block, Disjunct)
    ):
        if disjunction._algebraic_constraint:
            if disjunction._algebraic_constraint() is xor_constraint:
                return disjunction
    raise GDP_Error(
        "It appears that '%s' is not an XOR or OR constraint "
        "resulting from transforming a Disjunction." % xor_constraint.name
    )


def get_src_disjunct(transBlock):
    """Return the Disjunct object whose transformed components are on
    transBlock.

    Parameters
    ----------
    transBlock: _BlockData which is in the relaxedDisjuncts IndexedBlock
                on a transformation block.
    """
    if (
        not hasattr(transBlock, "_src_disjunct")
        or type(transBlock._src_disjunct) is not weakref_ref
    ):
        raise GDP_Error(
            "Block '%s' doesn't appear to be a transformation "
            "block for a disjunct. No source disjunct found." % transBlock.name
        )
    return transBlock._src_disjunct()


def get_src_constraint(transformedConstraint):
    """Return the original Constraint whose transformed counterpart is
    transformedConstraint

    Parameters
    ----------
    transformedConstraint: Constraint, which must be a component on one of
    the BlockDatas in the relaxedDisjuncts Block of
    a transformation block
    """
    transBlock = transformedConstraint.parent_block()
    # This should be our block, so if it's not, the user messed up and gave
    # us the wrong thing. If they happen to also have a _constraintMap then
    # the world is really against us.
    if not hasattr(transBlock, "_constraintMap"):
        raise GDP_Error(
            "Constraint '%s' is not a transformed constraint"
            % transformedConstraint.name
        )
    # if something goes wrong here, it's a bug in the mappings.
    return transBlock._constraintMap['srcConstraints'][transformedConstraint]


def _find_parent_disjunct(constraint):
    # traverse up until we find the disjunct this constraint lives on
    parent_disjunct = constraint.parent_block()
    while not isinstance(parent_disjunct, _DisjunctData):
        if parent_disjunct is None:
            raise GDP_Error(
                "Constraint '%s' is not on a disjunct and so was not "
                "transformed" % constraint.name
            )
        parent_disjunct = parent_disjunct.parent_block()

    return parent_disjunct


def _get_constraint_transBlock(constraint):
    parent_disjunct = _find_parent_disjunct(constraint)
    # we know from _find_parent_disjunct that parent_disjunct is a Disjunct,
    # so the below is OK
    transBlock = parent_disjunct._transformation_block
    if transBlock is None:
        raise GDP_Error(
            "Constraint '%s' is on a disjunct which has not been "
            "transformed" % constraint.name
        )
    # if it's not None, it's the weakref we wanted.
    transBlock = transBlock()

    return transBlock


def get_transformed_constraints(srcConstraint):
    """Return the transformed version of srcConstraint

    Parameters
    ----------
    srcConstraint: ScalarConstraint or _ConstraintData, which must be in
    the subtree of a transformed Disjunct
    """
    if srcConstraint.is_indexed():
        raise GDP_Error(
            "Argument to get_transformed_constraint should be "
            "a ScalarConstraint or _ConstraintData. (If you "
            "want the container for all transformed constraints "
            "from an IndexedDisjunction, this is the parent "
            "component of a transformed constraint originating "
            "from any of its _ComponentDatas.)"
        )
    transBlock = _get_constraint_transBlock(srcConstraint)
    try:
        return transBlock._constraintMap['transformedConstraints'][srcConstraint]
    except:
        logger.error("Constraint '%s' has not been transformed." % srcConstraint.name)
        raise


def _warn_for_active_disjunct(innerdisjunct, outerdisjunct):
    assert innerdisjunct.active
    problemdisj = innerdisjunct
    if innerdisjunct.is_indexed():
        for i in sorted(innerdisjunct.keys()):
            if innerdisjunct[i].active:
                # This shouldn't be true, we will complain about it.
                problemdisj = innerdisjunct[i]
                break

    raise GDP_Error(
        "Found active disjunct '{0}' in disjunct '{1}'! Either {0} "
        "is not in a disjunction or the disjunction it is in "
        "has not been transformed. {0} needs to be deactivated "
        "or its disjunction transformed before {1} can be "
        "transformed.".format(
            problemdisj.getname(fully_qualified=True),
            outerdisjunct.getname(fully_qualified=True),
        )
    )


def check_model_algebraic(instance):
    """Checks if there are any active Disjuncts or Disjunctions reachable via
    active Blocks. If there are not, it returns True. If there are, it issues
    a warning detailing where in the model there are remaining non-algebraic
    components, and returns False.

    Parameters
    ----------
    instance: a Model or Block
    """
    disjunction_set = {
        i
        for i in instance.component_data_objects(
            Disjunction, descend_into=(Block, Disjunct), active=None
        )
    }
    active_disjunction_set = {
        i
        for i in instance.component_data_objects(
            Disjunction, descend_into=(Block, Disjunct), active=True
        )
    }
    disjuncts_in_disjunctions = set()
    for i in disjunction_set:
        disjuncts_in_disjunctions.update(i.disjuncts)
    disjuncts_in_active_disjunctions = set()
    for i in active_disjunction_set:
        disjuncts_in_active_disjunctions.update(i.disjuncts)

    for disjunct in instance.component_data_objects(
        Disjunct, descend_into=(Block,), descent_order=TraversalStrategy.PostfixDFS
    ):
        # check if it's relaxed
        if disjunct.transformation_block is not None:
            continue
        # It's not transformed, check if we should complain
        elif (
            disjunct.active
            and _disjunct_not_fixed_true(disjunct)
            and _disjunct_on_active_block(disjunct)
        ):
            # If someone thinks they've transformed the whole instance, but
            # there is still an active Disjunct on the model, we will warn
            # them. In the future this should be the writers' job.)
            if disjunct not in disjuncts_in_disjunctions:
                logger.warning(
                    'Disjunct "%s" is currently active, '
                    'but was not found in any Disjunctions. '
                    'This is generally an error as the model '
                    'has not been fully relaxed to a '
                    'pure algebraic form.' % (disjunct.name,)
                )
                return False
            elif disjunct not in disjuncts_in_active_disjunctions:
                logger.warning(
                    'Disjunct "%s" is currently active. While '
                    'it participates in a Disjunction, '
                    'that Disjunction is currently deactivated. '
                    'This is generally an error as the '
                    'model has not been fully relaxed to a pure '
                    'algebraic form. Did you deactivate '
                    'the Disjunction without addressing the '
                    'individual Disjuncts?' % (disjunct.name,)
                )
                return False
            else:
                logger.warning(
                    'Disjunct "%s" is currently active. It must be '
                    'transformed or deactivated before solving the '
                    'model.' % (disjunct.name,)
                )
                return False

    for cons in instance.component_data_objects(
        LogicalConstraint, descend_into=Block, active=True
    ):
        if cons.active:
            logger.warning(
                'LogicalConstraint "%s" is currently active. It '
                'must be transformed or deactivated before solving '
                'the model.' % cons.name
            )
            return False

    # We didn't find anything bad.
    return True


def _disjunct_not_fixed_true(disjunct):
    # Return true if the disjunct indicator variable is not fixed to True
    return not (disjunct.indicator_var.fixed and disjunct.indicator_var.value)


def _disjunct_on_active_block(disjunct):
    # Check first to make sure that the disjunct is not a descendent of an
    # inactive Block or fixed and deactivated Disjunct, before raising a
    # warning.
    parent_block = disjunct.parent_block()
    while parent_block is not None:
        # deactivated Block
        if parent_block.ctype is Block and not parent_block.active:
            return False
        # properly deactivated Disjunct
        elif (
            parent_block.ctype is Disjunct
            and not parent_block.active
            and parent_block.indicator_var.value == False
            and parent_block.indicator_var.fixed
        ):
            return False
        else:
            # Step up one level in the hierarchy
            parent_block = parent_block.parent_block()
            continue
    return True
