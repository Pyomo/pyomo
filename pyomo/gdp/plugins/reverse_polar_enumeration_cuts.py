# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
from collections import defaultdict
from pyomo.core.base import Transformation, TransformationFactory, NonNegativeIntegers
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.block import SubclassOf
from pyomo.core.util import target_list
from pyomo.core.base.enums import SortComponents
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.core import Block, Constraint
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import get_gdp_tree
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import OrderedVarRecorder
from math import log, exp

from pyomo.common.dependencies import networkx as nx, networkx_available

logger = logging.getLogger(__name__)
EPS = 1e-6
# Hashable object representing the dummy x0 variable that is not None,
# since None cannot be used as a networkx node
_x0 = object()


class _ReversePolarEnumerationCutsData(AutoSlots.Mixin):
    __slots__ = "disjunction_constraints_map"

    def __init__(self):
        self.disjunction_constraints_map = ComponentMap()


Block.register_private_data_initializer(_ReversePolarEnumerationCutsData)


@TransformationFactory.register(
    'gdp.reverse_polar_enumeration_cuts',
    doc="Add cuts to a GDP with 'simple disjunctions', according to the "
    "reverse-polar vertex enumeration algorithm of [reference "
    "forthcoming]. A simple disjunction is one in which each disjunct "
    "contains only exactly one linear inequality on "
    "nonnegative-constrained variables, which is outwards facing in the "
    "sense that it can be rewritten in the form d . x >= 1.",
)
class ReversePolarEnumerationCuts(Transformation):
    """Add cuts to a GDP with 'simple disjunctions', according to the
    reverse-polar vertex enumeration algorithm of [reference
    forthcoming]. A simple disjunction is one in which each disjunct
    contains only exactly one linear inequality on
    nonnegative-constrained variables, which is outwards facing in the
    sense that it can be rewritten in the form d . x >= 1.
    """

    transformation_name = 'reverse_polar_enumeration_cuts'
    CONFIG = ConfigDict('gdp.reverse_polar_enumeration_cuts')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets to transform",
            doc="""
            This specifies the list of Disjunctions or Blocks to be (partially)
            transformed. If None (default), the entire model is transformed. 
            Note that if the transformation is done out of place, the list of
            targets should be attached to the model before it is cloned, and
            the list will specify the targets on the cloned instance.
            """,
        ),
    )
    CONFIG.declare(
        'num_cuts',
        ConfigValue(
            default=None,
            description="number of cuts to generate",
            doc="""
            Maximum number of cuts to generate. If None is passed, keep going until there
            are no more cuts available, at which point the cuts that have been added will
            fully determine the closed convex hull of the feasible region. This is
            applied separately on each target, so if you need to make a different
            number of cuts to each of multiple targets you should call this
            transformation multiple times.
            """,
        ),
    )
    CONFIG.declare(
        'num_skip',
        ConfigValue(
            default=0,
            domain=int,
            description="Skip the first N cuts generated.",
            doc="""
            Skip the first N cuts generated. Applied separately to each target like
            num_cuts.
            """,
        ),
    )

    def __init__(self):
        super().__init__()
        self.logger = logger

    def _apply_to(self, instance, **kwds):
        if not networkx_available:
            raise GDP_Error(
                "Networkx is required for this transformation, but it could "
                "not be imported."
            )
        if instance.ctype not in (Block, Disjunct):
            raise GDP_Error(
                "Transformation called on %s of type %s. 'instance'"
                " must be a ConcreteModel, Block, or Disjunct (in "
                "the case of nested disjunctions)." % (instance.name, instance.ctype)
            )

        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        targets = self._config.targets
        if targets is None:
            targets = (instance,)

        xf_block = Block()
        instance.add_component(
            unique_component_name(instance, "_reverse_polar_enumeration_cuts"), xf_block
        )
        tree = get_gdp_tree(targets, instance)
        for t in tree.reverse_topological_sort():
            if t.ctype is Disjunction:
                self._validate_disjunction(t, tree)
                self._generate_cuts(instance, xf_block, t, tree)

    def _validate_disjunction(self, disj, tree):
        # NOTE: some validation is deferred until we actually walk them
        if tree.root_disjunct(disj) is not None:
            raise GDP_Error(
                "Nested disjunctions are not supported for "
                f"{self.transformation_name}"
            )
        for b in tree.children(disj):
            found = False
            for c in b.component_data_objects(SubclassOf(ActiveComponent)):
                # To be safe, no active components except exactly one
                # constraint are permitted.  Non-active things like
                # params and vars are fine.
                if found or c.ctype is not Constraint:
                    raise GDP_Error(
                        "No active components except exactly one constraint "
                        "are permitted on disjuncts of a disjunction "
                        f"transformed by {self.transformation_name}."
                    )
                found = True
            if not found:
                # Likely user error. It's trivial in any case.
                raise GDP_Error(
                    "Empty disjunct on disjunction transformed by "
                    f"{self.transformation_name} - no cuts are possible."
                )

    def _add_cut(self, instance, xf_block, delta, disj, var_map, Jp, Jm):
        expr = 0
        for k, v in delta.items():
            if k != _x0:
                var = var_map[k]
                if k in Jp:
                    alpha = exp(delta[k])
                elif k in Jm:
                    alpha = -exp(delta[k])
                expr += alpha * var
        if disj in instance.private_data().disjunction_constraints_map:
            con = instance.private_data().disjunction_constraints_map[disj]
        else:
            con = Constraint(NonNegativeIntegers)
            xf_block.add_component(unique_component_name(xf_block, disj.name), con)
            instance.private_data().disjunction_constraints_map[disj] = con
        con[len(con)] = expr >= 1

    def _near_match(self, d1, d2):
        for k, v in d1.items():
            if not abs(v - d2[k]) < EPS:
                return False
        return True

    def _generate_cuts(self, instance, xf_block, disj, tree):
        num_cuts = self._config.num_cuts
        if num_cuts == 0:
            return
        num_skip = self._config.num_skip

        coef = defaultdict(lambda: 0)  # coef[(k, t)] = d_k^t

        # An insertion-ordered set type is desired here to enable fast
        # membership checks but maintain stable iteration order for
        # determinism and testing. Python does not provide this type, so
        # I will use the keys of a dictionary for the same effect.
        Jm = {_x0: None}  # {k | \forall t d_k^t < 0} \cup {0}
        Jp = {}  # {k | \exists t d_k^t > 0}
        found_order = {}  # for maintaining Jp and Jm in the same order we iterated
        found_idx = 1
        disjunct_idx = 1

        # Preprocessing
        visitor = LinearRepnVisitor(
            {}, var_recorder=OrderedVarRecorder({}, {}, SortComponents.deterministic)
        )
        for d in tree.children(disj):
            con = next(d.component_data_objects(Constraint))
            repn = visitor.walk_expression(con.body)
            if repn.nonlinear:
                raise GDP_Error(
                    f"Disjunction transformed by {self.transformation_name} "
                    "must not have a nonlinear constraint."
                )
            # Standardize form to dx >= d0, d0 = 1. This transformation
            # errors if we cannot do this.

            # NOTE: repn.multiplier is always 1 when obtained from LinearRepnVisitor
            multiplier = 1
            if con.ub is not None:
                if con.lb is not None:
                    raise GDP_Error(
                        "Equality constraint is not permitted in "
                        f"{self.transformation_name} transformation."
                    )
                else:
                    lb = repn.constant - con.ub
                    multiplier *= -1
            else:
                lb = con.lb - repn.constant
            if lb <= 0:
                raise GDP_Error(
                    "Nonpositive RHS is not valid for reverse polar cut generator."
                )
            multiplier /= lb

            for vid, c in repn.linear.items():
                if vid not in found_order:
                    # First time we found a variable; error-check that
                    # it's nonnegative constrained like we want. If the
                    # bound is positive then we aren't being sharp but
                    # the transformation is still valid.
                    bounds = visitor.var_map[vid].bounds
                    if bounds[0] is None or bounds[0] < 0:
                        raise GDP_Error(
                            f"Variables for {self.transformation_name} "
                            "should have nonnegative lower bounds."
                        )
                    found_order[vid] = found_idx
                    found_idx += 1
                c = c * multiplier
                if c > 0:
                    Jp[vid] = None
                    Jm.pop(vid, None)
                # NOTE: a variable can be neither Jp nor Jm at this
                # stage, but this will put such vars in Jm since we
                # aren't catching zero coefficients. We handle this
                # below.
                elif c < 0:
                    if vid not in Jp:
                        Jm[vid] = None

                coef[(vid, disjunct_idx)] = c

            disjunct_idx += 1

        # Keep these sorted for consistency. Only Jp could fail to be
        # here (since items can be added late if they were initially
        # placed in Jm).
        Jp = dict(sorted(Jp.items(), key=lambda x: found_order[x[0]]))

        for t in range(1, disjunct_idx):
            coef[_x0, t] = 1
            for j in [j for j in Jm if (j, t) not in coef]:
                # In this case, we effectively delete this variable
                # completely from the disjunction. It is never necessary
                # to include it on a generated cut.

                # NOTE: Here we rely on the fact that zero coefficients
                # never show up in the repn.
                Jm.pop(j, None)

        # Preprocessing ("sparse positive intersections" lemma from
        # Connor): Recreate the disjunction to have one disjunct for
        # each Jp variable, performing various alterations to the
        # coefficients. In the end d_k^t has a block form consisting of
        # a square diagonal matrix of size |Jp|x|Jp| with positive
        # diagonal values, and below that a block of all negative values
        # corresponding to variables in Jm.
        coef_preproc = defaultdict(lambda: 0)
        for j in Jp:
            # leave at zero other coef_preproc[k, j] for both indices in Jp
            coef_preproc[j, j] = max([coef[j, t] for t in range(1, disjunct_idx)])
            for k in Jm:
                coef_preproc[k, j] = (
                    -min(
                        [
                            abs(coef[k, t]) / coef[j, t]
                            for t in range(1, disjunct_idx)
                            if coef[j, t] > 0
                        ]
                    )
                    * coef_preproc[j, j]
                )
        coef = coef_preproc

        # First: NEEC cut
        delta = {}
        for j in Jp:
            delta[j] = log(coef[j, j])
        for k in Jm:
            delta[k] = log(min([abs(coef[k, j]) for j in Jp]))

        if not num_skip:
            self._add_cut(instance, xf_block, delta, disj, visitor.var_map, Jp, Jm)
        added_cuts = 1
        if num_cuts == 1:
            return

        cost = {}
        for j in Jp:
            for k in Jm:
                cost[j, k] = log(abs(coef[k, j]) / coef[j, j])

        # State machine: perform breadth-first search on D^# by using
        # the properties of the auxiliary graph G_dstar at each vertex
        # dstar in D^# to find (some) vertices adjacent to dstar,
        # checking each against used_list in case they are not new.
        vertex_queue = [delta]
        used_list = [delta]
        # indexes into Jm. Start at sentinel value
        k0 = len(Jm)
        # tuples of lists: (X, Xbar)
        cuts_stack = []

        while True:
            if not cuts_stack:
                if k0 == len(Jm):
                    # get a new vertex and reset k0
                    if not vertex_queue:
                        return  # all cuts generated
                    dstar = vertex_queue.pop(0)
                    G_dstar = nx.Graph()
                    G_dstar.add_nodes_from(Jp)
                    G_dstar.add_nodes_from(Jm)
                    for j in Jp:
                        for k in Jm:
                            if abs(dstar[k] - dstar[j] - cost[j, k]) < EPS:
                                G_dstar.add_edge(k, j)
                    k0 = 1
                    continue
                else:
                    # start a new [set of] graph cuts
                    Xbar = []
                    it = iter(Jm)
                    for i in range(k0):
                        Xbar.append(next(it))
                    X = [next(it)]
                    cuts_stack.append((X, Xbar))
                    k0 += 1
                    continue
            else:
                # there are candidate graph cuts in the queue; process them
                X, Xbar = cuts_stack.pop(-1)

                # Going forward we often need access to G_dstar[N_0 \ X]
                G_working = G_dstar.copy()
                G_working.remove_nodes_from(X)

                # Forcing rules that necessarily put certain nodes in
                # X. Iterate these until no more work is done. The loop
                # could be removed and this done only once, which may or
                # may not be faster, but the output might require more
                # filtering, in particular for property (2).
                did_forcing = True
                while did_forcing:
                    did_forcing = False
                    # Forcing rule 1
                    for k in Jm:
                        if k in X:
                            for j in G_dstar.neighbors(k):
                                if j not in X:
                                    # These are in Jp only
                                    X.append(j)
                                    did_forcing = True
                                    if j in G_working.nodes:
                                        G_working.remove_node(j)
                    # Forcing rule 2
                    for k in Jm:
                        if k not in X and k not in Xbar:
                            for j in G_dstar.neighbors(k):
                                if j in X and not nx.has_path(G_working, k, _x0):
                                    # Forcing rule 2
                                    X.append(k)
                                    did_forcing = True
                                    if k in G_working.nodes:
                                        G_working.remove_node(k)
                                    break

                for j in Jp:
                    if j in X:
                        done = False
                        for k in G_dstar.neighbors(j):
                            # these are in Jm only
                            if k not in X and k not in Xbar:
                                if nx.has_path(G_working, k, _x0):
                                    cuts_stack.append((X, Xbar + [k]))
                                    cuts_stack.append((X + [k], Xbar))
                                else:
                                    cuts_stack.append((X + [k], Xbar))
                                done = True
                                break
                        if done:
                            # We will see the other neighbors on
                            # subsequent iterations
                            continue  # cuts_queue is still populated
                # From here on any remaining elements of Jp and Jm
                # are treated as part of Xbar
                if self._validate_cut(X, G_dstar, G_working, Jp, Jm):
                    # perform mip cut
                    lstar = min(
                        [
                            cost[j, k] - dstar[k] + dstar[j]
                            for j in Jp
                            for k in Jm
                            if k in X and j not in X
                        ]
                    )
                    d_candidate = {
                        k: (v + lstar if k in X else v) for k, v in dstar.items()
                    }
                    # check if new, up to floating-point rounding
                    # error. If not, do not add cut
                    done = False
                    for d in used_list:
                        if self._near_match(d_candidate, d):
                            done = True
                            break
                    if done:
                        continue
                    vertex_queue.append(d_candidate)
                    used_list.append(d_candidate)
                    added_cuts += 1
                    if added_cuts > num_skip:
                        self._add_cut(
                            instance,
                            xf_block,
                            d_candidate,
                            disj,
                            visitor.var_map,
                            Jp,
                            Jm,
                        )
                    if added_cuts == num_cuts:
                        # early termination when requesting just a few
                        return
                # Depending on whether we used up cuts_queue, either get
                # a new initial cut or continue processing
                continue

    def _validate_cut(self, cut, G_dstar, G_Xbar, Jp, Jm):
        # Verify that the found cut meets the requirements from the paper.
        G_X = G_dstar.copy()
        G_X.remove_nodes_from(G_Xbar.nodes)
        # We know that (1) and (3) can fail with the algorithm as written
        # (3) X intersects Jm and Xbar intersects Jp
        if set(Jm).isdisjoint(set(cut)) or set(Jp).isdisjoint(set(G_Xbar.nodes)):
            return False
        # (1) X and Xbar induce connected subgraphs of G_dstar
        if not nx.is_connected(G_Xbar) or not nx.is_connected(G_X):
            return False

        # (2) No directed (Jm->Jp) edges run from X to Xbar

        # This should be guaranteed by forcing rule 1, given that we
        # iterate the forcing rules until they cannot be performed any
        # more, and we don't add any Jm variables to X later in the
        # iteration. Should this logic fail somehow (eg if we didn't
        # have the loop around the forcing rules), we would need to
        # check this here too.
        return True


def get_constraint(transformed_block, disjunction):
    if disjunction in transformed_block.private_data().disjunction_constraints_map:
        return transformed_block.private_data().disjunction_constraints_map[disjunction]
    else:
        raise ValueError(
            f"Disjunction {disjunction} was not used for cut generation by "
            f"a call to gdp.reverse_polar_enumeration_cuts on model {transformed_block}"
        )
