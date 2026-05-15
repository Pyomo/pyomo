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

from pyomo.common.dependencies import numpy as np, numpy_available

from pyomo.common.autoslots import AutoSlots
import pyomo.common.config as cfg
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet, DefaultComponentMap
from pyomo.common.modeling import unique_component_name
import pyomo.core.expr as EXPR
from pyomo.core.base import TransformationFactory, SortComponents
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import OrderedVarRecorder
from pyomo.core import (
    Block,
    Constraint,
    ConstraintList,
    Set,
    Suffix,
    Var,
    Expression,
    Reals,
    NonNegativeReals,
    value,
    NonNegativeIntegers,
    ConcreteModel,
    Objective,
    Reference,
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.util import clone_without_expression_components
from pyomo.gdp.plugins.multiple_bigm import Solver
from pyomo.core.util import target_list
from pyomo.core.expr.visitor import (
    IdentifyVariableVisitor,
    StreamBasedExpressionVisitor,
)
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import VarRecorder
from pyomo.opt.results.solver import TerminationCondition
from pyomo.opt.solver import SolverStatus
from weakref import ref as weakref_ref
import math
import itertools

logger = logging.getLogger('pyomo.gdp.hull')


class _HullTransformationData(AutoSlots.Mixin):
    __slots__ = (
        'disaggregated_var_map',
        'original_var_map',
        'bigm_constraint_map',
        'disaggregation_constraint_map',
        'well_defined_points_map',
        'exact_quadratic_aux_var_map',
    )

    def __init__(self):
        self.disaggregated_var_map = DefaultComponentMap(ComponentMap)
        self.original_var_map = ComponentMap()
        self.bigm_constraint_map = DefaultComponentMap(ComponentMap)
        self.disaggregation_constraint_map = DefaultComponentMap(ComponentMap)
        self.well_defined_points_map = {}
        self.exact_quadratic_aux_var_map = ComponentMap()


Block.register_private_data_initializer(_HullTransformationData)


@TransformationFactory.register(
    'gdp.hull', doc="Relax disjunctive model by forming the hull reformulation."
)
class Hull_Reformulation(GDP_to_MIP_Transformation):
    """Relax disjunctive model by forming the hull reformulation.

    Relaxes a disjunctive model into an algebraic model by forming the
    hull reformulation of each disjunction.

    This transformation accepts the following keyword arguments:

    The transformation will create a new Block with a unique
    name beginning "_pyomo_gdp_hull_reformulation". It will contain an
    indexed Block named "relaxedDisjuncts" that will hold the relaxed
    disjuncts. This block is indexed by an integer indicating the order
    in which the disjuncts were relaxed. All transformed Disjuncts will
    have a pointer to the block their transformed constraints are on,
    and all transformed Disjunctions will have a pointer to the
    corresponding OR or XOR constraint.

    Parameters
    ----------
    perspective_function : str
        The perspective function used for the disaggregated variables.
        Must be one of 'FurmanSawayaGrossmann' (default),
        'LeeGrossmann', or 'GrossmannLee'
    EPS : float
        The value to use for epsilon [default: 1e-4]
    exact_hull_quadratic : bool
        If ``True``, quadratic constraints (polynomial degree 2) are
        reformulated using the exact hull instead of the standard
        perspective function, following Gusev & Bernal Neira (2025) [4]_.
        Convex quadratics are handled with a conic reformulation
        (rotated second-order cone), while non-convex quadratics and
        equalities use the general exact hull reformulation. Convexity
        is determined via eigenvalue decomposition of the Hessian matrix.
        **Important:** unlike most GDP transformations, coefficients are
        taken from a quadratic representation that **evaluates mutable**
        :class:`Param` **components to numeric values at transformation**
        **time**. The reformulated constraints do **not** stay algebraically
        tied to those Params; changing a mutable Param after transformation
        does not update the relaxation. Set parameter values before
        transforming, or use immutable Params, if you need that linkage.
        [default: False]
    eigenvalue_tolerance : float
        Numerical tolerance for eigenvalue-based positive/negative
        semi-definite checks when using the exact hull reformulation for
        quadratic constraints (``exact_hull_quadratic=True``). An
        eigenvalue :math:`\\lambda` is treated as non-negative if
        :math:`\\lambda >= -\\text{eigenvalue_tolerance}` and as
        non-positive if :math:`\\lambda <= \\text{eigenvalue_tolerance}`
        (i.e., eigenvalues in
        ``[-eigenvalue_tolerance, eigenvalue_tolerance]`` are treated as
        zero). Increasing this value makes the convexity check more
        permissive; decreasing it makes it more conservative.
        [default: 1e-10]
    targets : block, disjunction, or list of those types
        The targets to transform. This can be a block, disjunction, or a
        list of blocks and Disjunctions [default: the instance]
    well_defined_points: dict-like of ComponentMaps
        Base points used for handling of functions with restricted
        domain. See the generated documentation of the config option
        for full details.
    well_defined_points_heuristic_solver: Solver object supporting v1 api
        or corresponding string [default: 'gurobi_direct_minlp']
        Solver used to search for a base point when constraints are not
        well-defined at zero. Must be a MINLP solver in general. Override
        using 'well_defined_points'
    """

    CONFIG = cfg.ConfigDict('gdp.hull')
    CONFIG.declare(
        'targets',
        cfg.ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be relaxed",
            doc="""

        This specifies the target or list of targets to relax as either a
        component or a list of components. If None (default), the entire model
        is transformed. Note that if the transformation is done out of place,
        the list of targets should be attached to the model before it is cloned,
        and the list will specify the targets on the cloned instance.""",
        ),
    )
    CONFIG.declare(
        'perspective function',
        cfg.ConfigValue(
            default='FurmanSawayaGrossmann',
            domain=cfg.In(['FurmanSawayaGrossmann', 'LeeGrossmann', 'GrossmannLee']),
            description='perspective function used for variable disaggregation',
            doc="""
        The perspective function used to transform nonlinear functions

        "LeeGrossmann" is the original NL convex hull from Lee &
        Grossmann (2000) [1]_, which substitutes nonlinear constraints

            h_ik(x) <= 0

        with

            x_k = sum( nu_ik )
            y_ik * h_ik( nu_ik/y_ik ) <= 0

        "GrossmannLee" is an updated formulation from Grossmann &
        Lee (2003) [2]_, which avoids divide-by-0 errors by using:

            x_k = sum( nu_ik )
            (y_ik + eps) * h_ik( nu_ik/(y_ik + eps) ) <= 0

        "FurmanSawayaGrossmann" (default) is an improved relaxation [3]_
        that is exact at 0 and 1 while avoiding numerical issues from
        the Lee & Grossmann formulation by using:

            x_k = sum( nu_ik )
            ((1-eps)*y_ik + eps) * h_ik( nu_ik/((1-eps)*y_ik + eps) ) \
                - eps * h_ki(0) * ( 1-y_ik ) <= 0


         The default, "FurmanSawayaGrossmann", is strongly
         recommended. When "FurmanSawayaGrossmann" is used, any value of
         epsilon in (0, 1) leads to an exact reformulation, and
         decreasing epsilon improves the quality of the continuous
         relaxation (see [3]_). When "GrossmannLee" is used, epsilon
         should be set very small, as the formulation is only correct in
         the epsilon -> 0 limit. In particular, it should be small
         enough to put spurious O(eps) sized constraint violations
         within solver tolerances. Both "GrossmannLee" (when epsilon is
         small enough) and the original "LeeGrossmann" have numerical
         and feasibility issues.


        References
        ----------
        .. [1] Lee, S., & Grossmann, I. E. (2000). New algorithms for
           nonlinear generalized disjunctive programming.  Computers and
           Chemical Engineering, 24, 2125-2141

        .. [2] Grossmann, I. E., & Lee, S. (2003). Generalized disjunctive
           programming: Nonlinear convex hull relaxation and algorithms.
           Computational Optimization and Applications, 26, 83-100.

        .. [3] Furman, K., Sawaya, N., and Grossmann, I.  A computationally
           useful algebraic representation of nonlinear disjunctive convex
           sets using the perspective function.  Optimization Online
           (2016). http://www.optimization-online.org/DB_HTML/2016/07/5544.html.

        .. [4] Gusev, S., & Bernal Neira, D. E. (2025). Exact Hull
           Reformulation for Quadratically Constrained Generalized
           Disjunctive Programs. arXiv preprint arXiv:2508.16093.
           https://arxiv.org/abs/2508.16093
        """,
        ),
    )
    CONFIG.declare(
        'EPS',
        cfg.ConfigValue(
            default=1e-4,
            domain=cfg.PositiveFloat,
            description="Epsilon value to use in perspective function",
            doc="See the doc for 'perspective function' for discussion.",
        ),
    )
    CONFIG.declare(
        'exact_hull_quadratic',
        cfg.ConfigValue(
            default=False,
            domain=bool,
            description="Use exact hull reformulation for quadratic constraints "
            "(mutable Params are evaluated at transformation time; see doc).",
            doc="""
        If True, quadratic constraints (polynomial degree 2) are reformulated
        using the exact hull instead of the standard perspective function,
        following Gusev & Bernal Neira (2025), arXiv:2508.16093.

        For a quadratic constraint of the form

            x'Qx + c'x + d <= 0

        the reformulation depends on convexity:

        Conic exact hull (convex quadratics): An auxiliary variable ``t``
        and a rotated second-order cone constraint ``v'Qv <= t * y`` are
        introduced, and the original bound becomes ``t + c'v + d*y <= 0``.
        Convexity is determined via eigenvalue decomposition of the Hessian
        matrix ``Q``: the quadratic is convex for an upper-bound constraint
        when ``Q`` is positive semi-definite, and for a lower-bound constraint
        when ``Q`` is negative semi-definite.

        General exact hull (non-convex quadratics and equalities): The
        constraint is reformulated as ``v'Qv + c'v*y + d*y**2``, where ``v``
        are the disaggregated variables and ``y`` is the binary indicator.

        Default is False, which uses the standard perspective function for
        all nonlinear constraints.

        **Departure from typical GDP hull behavior (mutable Parameters):**
        Coefficients are obtained via ``QuadraticRepnVisitor``, which freezes
        mutable :class:`Param` values (and other fixed numeric leaves) when the
        expression is walked. The transformed model is not updated if those
        Params change later. Most other hull paths keep Param references in the
        transformed constraints; this option does not. Set mutable Params to
        their intended values before transforming, or avoid mutable Params in
        the quadratic body, if you need algebraic linkage to parameters.

        Alternatively, use fixed :class:`Var` objects instead of mutable
        Params with ``assume_fixed_vars_permanent=False`` (the default): such
        Vars are kept as symbolic references in the transformed constraints
        and can be updated by re-fixing them.
        """,
        ),
    )
    CONFIG.declare(
        'eigenvalue_tolerance',
        cfg.ConfigValue(
            default=1e-10,
            domain=cfg.NonNegativeFloat,
            description="Numerical tolerance for eigenvalue-based PSD/NSD checks "
            "in exact hull quadratic reformulations",
            doc="""
        Numerical tolerance used when determining positive semi-definiteness
        (PSD) or negative semi-definiteness (NSD) of the Hessian matrix Q in
        the exact hull reformulation for quadratic constraints
        (``exact_hull_quadratic=True``).

        An eigenvalue ``lam`` is treated as non-negative if
        ``lam >= -eigenvalue_tolerance``, and non-positive if
        ``lam <= eigenvalue_tolerance``. Increasing this value makes the
        convexity classification more permissive (i.e., a wider band around
        zero is treated as numerically zero, so more eigenvalues are accepted
        as PSD/NSD); decreasing it makes the check more conservative (i.e.,
        eigenvalues must be further from zero). For ill-conditioned Q matrices
        a larger tolerance may be appropriate.
        """,
        ),
    )
    CONFIG.declare(
        'assume_fixed_vars_permanent',
        cfg.ConfigValue(
            default=False,
            domain=bool,
            description="Boolean indicating whether or not to transform so that "
            "the transformed model will still be valid when fixed Vars are "
            "unfixed.",
            doc="""
        If True, the transformation will not disaggregate fixed variables.
        This means that if a fixed variable is unfixed after transformation,
        the transformed model is no longer valid. By default, the transformation
        will disagregate fixed variables so that any later fixing and unfixing
        will be valid in the transformed model.
        """,
        ),
    )
    CONFIG.declare(
        'well_defined_points',
        cfg.ConfigValue(
            default={},
            description="Distinguished points at which constraints with restricted "
            "domain are well-defined. This will be used as a center point for "
            "transformed constraints.",
            doc="""
        Dict-like mapping Disjunctions to ComponentMaps
        mapping Vars appearing on the Disjuncts of that Disjunction to
        float values, such that each constraint function appearing on
        those disjuncts is well-defined (no division by zero, logarithm
        of negative, etc.) when those vars are set to those values. The
        outer map need not contain every Disjunction in the model as a
        key, but the inner ComponentMaps (if any) should have as keys
        every variable appearing on those disjuncts (including fixed
        vars unless assume_fixed_vars_permanent is set to True). When
        this is not provided for a disjunction, as it usually need not
        be, we first try the point with all variables zero, then we
        make a best effort to find a nonzero point through a subsolver
        call, then we raise GDP_Error if neither attempt was successful.
        """,
        ),
    )
    CONFIG.declare(
        'well_defined_points_heuristic_solver',
        cfg.ConfigValue(
            default='gurobi_direct_minlp',
            domain=Solver,
            description="Solver used for the base points heuristic",
            doc="""
        Solver used for the base points heuristic. This must be a
        nonconvex NLP solver in general. Pass as a solver object
        supporting the V1 solver API version, or a corresponding string
        for SolverFactory.
        """,
        ),
    )
    transformation_name = 'hull'

    def __init__(self):
        super().__init__(logger)
        self._targets = set()

    def _collect_local_vars_from_block(self, block, local_var_dict):
        localVars = block.component('LocalVars')
        if localVars is not None and localVars.ctype is Suffix:
            for disj, var_list in localVars.items():
                local_var_dict[disj].update(var_list)

    def _get_user_defined_local_vars(self, targets):
        user_defined_local_vars = defaultdict(ComponentSet)
        seen_blocks = set()
        # we go through the targets looking both up and down the hierarchy, but
        # we cache what Blocks/Disjuncts we've already looked on so that we
        # don't duplicate effort.
        for t in targets:
            if t.ctype is Disjunct:
                # first look beneath where we are (there could be Blocks on this
                # disjunct)
                for b in t.component_data_objects(
                    Block,
                    descend_into=Block,
                    active=True,
                    sort=SortComponents.deterministic,
                ):
                    if b not in seen_blocks:
                        self._collect_local_vars_from_block(b, user_defined_local_vars)
                        seen_blocks.add(b)
                # now look up in the tree
                blk = t
                while blk is not None:
                    if blk in seen_blocks:
                        break
                    self._collect_local_vars_from_block(blk, user_defined_local_vars)
                    seen_blocks.add(blk)
                    blk = blk.parent_block()
        return user_defined_local_vars

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._restore_state()
            self._transformation_blocks.clear()
            self._algebraic_constraints.clear()

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)

        # filter out inactive targets and handle case where targets aren't
        # specified.
        targets = self._filter_targets(instance)
        # transform logical constraints based on targets
        self._transform_logical_constraints(instance, targets)

        # Preprocess in order to find what disjunctive components need
        # transformation
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        # Transform from leaf to root: This is important for hull because for
        # nested GDPs, we will introduce variables that need disaggregating into
        # parent Disjuncts as we transform their child Disjunctions.
        preprocessed_targets = gdp_tree.reverse_topological_sort()
        # Get all LocalVars from Suffixes ahead of time
        local_vars_by_disjunct = self._get_user_defined_local_vars(preprocessed_targets)

        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(
                    t, t.index(), gdp_tree.parent(t), local_vars_by_disjunct
                )
            # We skip disjuncts now, because we need information from the
            # disjunctions to transform them (which variables to disaggregate),
            # so for hull's purposes, they need not be in the tree.

    def _add_transformation_block(self, to_block):
        transBlock, new_block = super()._add_transformation_block(to_block)
        if not new_block:
            return transBlock, new_block

        transBlock.lbub = Set(initialize=['lb', 'ub', 'eq'])

        # We will store all of the disaggregation constraints for any
        # Disjunctions we transform onto this block here.
        transBlock.disaggregationConstraints = Constraint(NonNegativeIntegers)

        # we are going to store some of the disaggregated vars directly here
        # when we have vars that don't appear in every disjunct
        transBlock._disaggregatedVars = Var(NonNegativeIntegers, dense=False)
        transBlock._boundsConstraints = Constraint(NonNegativeIntegers, transBlock.lbub)

        return transBlock, True

    # From a test expression test_expr containing exactly the variables
    # regular_vars and fallback_vars, except possibly containing some
    # additional fixed vars, get a point at which test_expr is
    # well-defined according to the following process:
    # (1) try the origin
    # (2) try fixing fallback_vars at zero, and allow a solver to
    #     change regular_vars
    # (3) try allowing a solver to change all the vars
    #
    # If a point is found, return a ComponentMap x0_map from Var to the
    # found numeric value, and a ComponentSet used_vars of all Vars that
    # were given a nonzero value as part of that process. If no such
    # point can be found, raise a GDP_Error.
    #
    # If we have assume_fixed_vars_permanent=True, then the fixed vars
    # will not appear in regular_vars or fallback_vars, and we will not
    # try to change them for the purpose of finding a well-defined
    # point, nor will they appear in the x0_map. Otherwise, i.e.,
    # whenever they are actually passed in, they will be treated as
    # normal variables. We will restore the original variable values and
    # fixed statuses when we return successfully.
    def _get_well_defined_point(
        self, test_expr, regular_vars, fallback_vars, disj_name
    ):
        # First, see if test_expr is well-defined at the origin.
        x0_map = ComponentMap()
        orig_values = ComponentMap()
        orig_fixed = ComponentMap()
        orig_bounds = ComponentMap()
        for x in itertools.chain(regular_vars, fallback_vars):
            x0_map[x] = 0
            orig_values[x] = x.value
            orig_fixed[x] = x.fixed
            orig_bounds[x] = x.bounds
            x.bounds = (None, None)
        # Outer try-finally to ensure we always restore variable values,
        # bounds, and fixed statuses on exit (even if we or the solver
        # throw an exception)
        try:
            try:
                val = value(
                    EXPR.ExpressionReplacementVisitor(
                        substitute=x0_map
                    ).walk_expression(test_expr)
                )
                if math.isfinite(val):
                    return x0_map, ComponentSet()
            except (ValueError, ZeroDivisionError):  # ('math domain error')
                pass
            except Exception as e:  # can anything else be thrown here?
                logger.error(
                    "While trying to evaluate an expression, got unexpected exception type "
                    f"{e.__class__.__name__} (was prepared for success or a ValueError)."
                )
                raise
            logger.info(
                "A constraint function on a disjunct couldn't be evaluated at "
                "zero. Trying a solver-based heuristic to find a point that "
                "makes the constraints well-defined, which is necessary for "
                "hull. Note that the exact formulation obtained may not be "
                "fully deterministic, as it depend's on the solver's solution. "
                "If this is problematic, you can override using the "
                "`well_defined_points` option."
            )
            # Second, try making it well-defined by editing only the regular vars
            for x in fallback_vars:
                x.fix(0)
            for x in regular_vars:
                x.value = 0
                x.unfix()
            test_model = ConcreteModel()
            test_model.test_expr = Expression(expr=test_expr)
            test_model.obj = Objective(expr=0)
            # In case the solver can't deal with Vars it doesn't know about
            for x in itertools.chain(regular_vars, fallback_vars):
                test_model.add_component(
                    unique_component_name(test_model, x.name), Reference(x)
                )
            test_model.well_defined_cons = ConstraintList()
            _WellDefinedConstraintGenerator(
                cons_list=test_model.well_defined_cons
            ).walk_expression(test_expr)
            feasible = self._solve_for_first_feasible_solution(test_model)
            # Third, try again, but edit all the vars
            if not feasible:
                for x in fallback_vars:
                    x.unfix()
                feasible = self._solve_for_first_feasible_solution(test_model)
                if not feasible:
                    raise GDP_Error(
                        f"Unable to find a well-defined point on disjunction {disj_name}. "
                        "To carry out the hull transformation, each disjunction must have a "
                        "point at which every constraint function appearing in its "
                        "disjuncts is well-defined and finite. Please ensure such a point "
                        "actually exists, then if we still cannot find it, override our "
                        "search process using the `well_defined_points` option."
                    )
            # Found a point
            x0_map = ComponentMap()
            used_vars = ComponentSet()
            for x in itertools.chain(regular_vars, fallback_vars):
                x0_map[x] = value(x)
                if x0_map[x] != 0:
                    used_vars.add(x)
            return x0_map, used_vars
        finally:
            for x in itertools.chain(regular_vars, fallback_vars):
                x.value = orig_values[x]
                x.fixed = orig_fixed[x]
                x.bounds = orig_bounds[x]

    # Use gurobi_direct_minlp for the heuristic for now. It needs to be
    # a nonlinear solver.
    def _solve_for_first_feasible_solution(self, test_model):
        results = self._config.well_defined_points_heuristic_solver.solve(
            test_model, load_solutions=False
        )
        if results.solver.termination_condition is TerminationCondition.infeasible:
            return False
        if results.solver.status is not SolverStatus.ok:
            raise GDP_Error(f"Unexpected solver status {results.solver.status}.")
        test_model.solutions.load_from(results)
        return True

    def _transform_disjunctionData(
        self, obj, index, parent_disjunct, local_vars_by_disjunct
    ):
        # Hull reformulation doesn't work if this is an OR constraint. So if
        # xor is false, give up
        if not obj.xor:
            raise GDP_Error(
                "Cannot do hull reformulation for "
                "Disjunction '%s' with OR constraint. "
                "Must be an XOR!" % obj.name
            )
        # collect the Disjuncts we are going to transform now because we will
        # change their active status when we transform them, but we still need
        # this list after the fact.
        active_disjuncts = [disj for disj in obj.disjuncts if disj.active]

        # We put *all* transformed things on the parent Block of this
        # disjunction. We'll mark the disaggregated Vars as local, but beyond
        # that, we actually need everything to get transformed again as we go up
        # the nested hierarchy (if there is one)
        transBlock, xorConstraint = self._setup_transform_disjunctionData(
            obj, root_disjunct=None
        )

        disaggregationConstraint = transBlock.disaggregationConstraints
        disaggregationConstraintMap = (
            transBlock.private_data().disaggregation_constraint_map
        )
        disaggregatedVars = transBlock._disaggregatedVars
        disaggregated_var_bounds = transBlock._boundsConstraints

        # We first go through and collect all the variables that we are going to
        # disaggregate. We do this in its own pass because we want to know all
        # the Disjuncts that each Var appears in since that will tell us exactly
        # which diaggregated variables we need.
        var_order = ComponentSet()
        disjuncts_var_appears_in = ComponentMap()
        active_constraints = set()
        # For each disjunct in the disjunction, we will store a list of Vars
        # that need a disaggregated counterpart in that disjunct.
        disjunct_disaggregated_var_map = {}
        for disjunct in active_disjuncts:
            # create the key for each disjunct now
            disjunct_disaggregated_var_map[disjunct] = ComponentMap()
            for con in disjunct.component_data_objects(
                Constraint,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=Block,
            ):
                active_constraints.add(con)
                # [ESJ 02/14/2020] By default, we disaggregate fixed variables
                # on the philosophy that fixing is not a promise for the future
                # and we are mathematically wrong if we don't transform these
                # correctly and someone later unfixes them and keeps playing
                # with their transformed model. However, the user may have set
                # assume_fixed_vars_permanent to True in which case we will skip
                # them

                # Note that, because ComponentSets are ordered, we will
                # eventually disaggregate the vars in a deterministic order
                # (the order that we found them)
                for var in IdentifyVariableVisitor(
                    include_fixed=not self._config.assume_fixed_vars_permanent
                ).walk_expression(con.expr):
                    if var not in var_order:
                        var_order.add(var)
                        disjuncts_var_appears_in[var] = ComponentSet([disjunct])
                    else:
                        disjuncts_var_appears_in[var].add(disjunct)
        # Now, we will disaggregate all variables that are not explicitly
        # declared as being local. If we are moving up in a nested tree, we have
        # marked our own disaggregated variables as local, so they will not be
        # re-disaggregated.
        vars_to_disaggregate = {disj: ComponentSet() for disj in obj.disjuncts}
        all_vars_to_disaggregate = ComponentSet()
        # We will ignore variables declared as local in a Disjunct that don't
        # actually appear in any Constraints on that Disjunct, but in order to
        # do this, we will explicitly collect the set of local_vars in this
        # loop.
        local_vars = defaultdict(ComponentSet)
        all_local_vars = ComponentSet()
        # This set contains vars that potentially must be disaggregated,
        # but do not need to worry about global constraints, and can
        # safely be added to the local vars of any parent disjunct. This
        # is a superset of all_local_vars, and the extra elements are
        # members of all_vars_to_disaggregate.
        generalized_local_vars = ComponentSet()
        for var in var_order:
            disjuncts = disjuncts_var_appears_in[var]
            for disj in disjuncts:
                if disj in local_vars_by_disjunct:
                    if var in local_vars_by_disjunct[disj]:
                        if len(disjuncts) == 1:
                            local_vars[disj].add(var)
                            all_local_vars.add(var)
                        else:
                            vars_to_disaggregate[disj].add(var)
                            all_vars_to_disaggregate.add(var)
                        generalized_local_vars.add(var)
            if var not in generalized_local_vars:
                for disj in disjuncts:
                    # Not a local var, so we must disaggregate, even if
                    # it's only on one disjunct.
                    vars_to_disaggregate[disj].add(var)
                all_vars_to_disaggregate.add(var)
        # Find a well-defined point x_0. We need every constraint body
        # to successfully evaluate to something.
        if obj in self._config.well_defined_points:
            x0_map = self._config.well_defined_points[obj]
            offset_vars = ComponentSet()
            for x, val in x0_map.items():
                if val != 0:
                    offset_vars.add(x)
        else:
            x0_map, offset_vars = self._get_well_defined_point(
                test_expr=sum(con.body for con in active_constraints),
                regular_vars=all_vars_to_disaggregate,
                fallback_vars=all_local_vars,
                disj_name=obj.name,
            )
        transBlock.parent_block().private_data().well_defined_points_map[obj] = x0_map
        # Any var that got an offset cannot be local anymore, but it can
        # still be generalized local
        for var in offset_vars:
            if var in all_local_vars:
                var_disjunct = next(iter(disjuncts_var_appears_in[var]))
                local_vars[var_disjunct].remove(var)
                all_local_vars.remove(var)
                vars_to_disaggregate[var_disjunct].add(var)
                all_vars_to_disaggregate.add(var)

        # Now that we know who we need to disaggregate, we will do it
        # while we also transform the disjuncts.

        # Get the list of local variables for the parent Disjunct so that we can
        # add the disaggregated variables we're about to make to it:
        parent_local_var_list = self._get_local_var_list(parent_disjunct)
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var.get_associated_binary()
            if disjunct.active:
                self._transform_disjunct(
                    obj=disjunct,
                    transBlock=transBlock,
                    vars_to_disaggregate=vars_to_disaggregate[disjunct],
                    local_vars=local_vars[disjunct],
                    parent_local_var_suffix=parent_local_var_list,
                    parent_disjunct_local_vars=local_vars_by_disjunct[parent_disjunct],
                    disjunct_disaggregated_var_map=disjunct_disaggregated_var_map,
                    x0_map=x0_map,
                    offset_vars=offset_vars,
                )
        # The parent disjunct's local vars should include all of this
        # disjunction's local or generalized local vars.
        local_vars_by_disjunct[parent_disjunct].update(generalized_local_vars)

        xorConstraint.add(index, (or_expr, 1))
        # map the DisjunctionData to its XOR constraint to mark it as
        # transformed
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])

        # Now add the reaggregation constraints
        for var in all_vars_to_disaggregate:
            # If a var did not appear in every disjunct of the
            # disjunction, then we (intentionally) did not create a
            # complete set of disaggregated vars and corresponding
            # bounds constraints for it. This would cause the variable
            # to be forced to zero when no disjunct containing it is
            # selected. If the var were local, this would not matter,
            # but unless we were able to put it in
            # generalized_local_vars earlier, it is possible that it
            # could appear in other parts of the model. It is therefore
            # necessary that it be unconstrained when no disjunct
            # containing it is selected. We implement this by adding one
            # more disaggregated variable which becomes active if none
            # of the disjuncts containing the original var were
            # selected. Its only constraints are the bounds constraints.
            if (
                len(disjuncts_var_appears_in[var]) < len(active_disjuncts)
                and var not in generalized_local_vars
            ):
                # create one more disaggregated var
                idx = len(disaggregatedVars)
                disaggregated_var = disaggregatedVars[idx]
                # mark this as local because we won't re-disaggregate it if this
                # is a nested disjunction
                if parent_local_var_list is not None:
                    parent_local_var_list.append(disaggregated_var)
                local_vars_by_disjunct[parent_disjunct].add(disaggregated_var)
                var_free = 1 - sum(
                    disj.indicator_var.get_associated_binary()
                    for disj in disjuncts_var_appears_in[var]
                )
                self._declare_disaggregated_var_bounds(
                    original_var=var,
                    disaggregatedVar=disaggregated_var,
                    disjunct=obj,
                    bigmConstraint=disaggregated_var_bounds,
                    var_free_indicator=var_free,
                    x0_map=x0_map,
                    var_idx=idx,
                )
                original_var_info = var.parent_block().private_data()
                disaggregated_var_map = original_var_info.disaggregated_var_map

                # For every Disjunct the Var does not appear in, we want to map
                # that this new variable is its disaggreggated variable.
                for disj in active_disjuncts:
                    # Because we called _transform_disjunct above, we know that
                    # if this isn't transformed it is because it was cleanly
                    # deactivated, and we can just skip it.
                    if (
                        disj._transformation_block is not None
                        and disj not in disjuncts_var_appears_in[var]
                    ):
                        disaggregated_var_map[disj][var] = disaggregated_var

                # start the expression for the reaggregation constraint with
                # this var
                disaggregatedExpr = disaggregated_var
            else:
                disaggregatedExpr = 0
            for disjunct in disjuncts_var_appears_in[var]:
                disaggregatedExpr += disjunct_disaggregated_var_map[disjunct][var]

            cons_idx = len(disaggregationConstraint)
            # We always aggregate to the original var. If this is nested, this
            # constraint will be transformed again. (And if it turns out
            # everything in it is local, then that transformation won't actually
            # change the mathematical expression, so it's okay.
            disaggregationConstraint.add(
                cons_idx, var - x0_map[var] == disaggregatedExpr
            )
            # and update the map so that we can find this later. We index by
            # variable and the particular disjunction because there is a
            # different one for each disjunction
            disaggregationConstraintMap[var][obj] = disaggregationConstraint[cons_idx]

        # deactivate for the writers
        obj.deactivate()

    def _transform_disjunct(
        self,
        obj,
        transBlock,
        vars_to_disaggregate,
        local_vars,
        parent_local_var_suffix,
        parent_disjunct_local_vars,
        disjunct_disaggregated_var_map,
        x0_map,
        offset_vars,
    ):
        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)

        # Put the disaggregated variables all on their own block so that we can
        # isolate the name collisions and still have complete control over the
        # names on this block.
        relaxationBlock.disaggregatedVars = Block()

        # add the disaggregated variables and their bigm constraints
        # to the relaxationBlock
        for var in vars_to_disaggregate:
            disaggregatedVar = Var(within=Reals, initialize=var.value)
            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            disaggregatedVarName = unique_component_name(
                relaxationBlock.disaggregatedVars, var.getname(fully_qualified=True)
            )
            relaxationBlock.disaggregatedVars.add_component(
                disaggregatedVarName, disaggregatedVar
            )
            # mark this as local via the Suffix in case this is a partial
            # transformation:
            if parent_local_var_suffix is not None:
                parent_local_var_suffix.append(disaggregatedVar)
            # Record that it's local for our own bookkeeping in case we're in a
            # nested tree in *this* transformation
            parent_disjunct_local_vars.add(disaggregatedVar)

            # add the bigm constraint
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(
                disaggregatedVarName + "_bounds", bigmConstraint
            )

            self._declare_disaggregated_var_bounds(
                original_var=var,
                disaggregatedVar=disaggregatedVar,
                disjunct=obj,
                bigmConstraint=bigmConstraint,
                var_free_indicator=obj.indicator_var.get_associated_binary(),
                x0_map=x0_map,
            )
            # update the bigm constraint mappings
            data_dict = disaggregatedVar.parent_block().private_data()
            data_dict.bigm_constraint_map[disaggregatedVar][obj] = bigmConstraint
            disjunct_disaggregated_var_map[obj][var] = disaggregatedVar

        for var in local_vars:
            # we don't need to disaggregate, i.e., we can use this Var, but we
            # do need to set up its bounds constraints.

            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            conName = unique_component_name(
                relaxationBlock, var.getname(fully_qualified=False) + "_bounds"
            )
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(conName, bigmConstraint)

            self._declare_disaggregated_var_bounds(
                original_var=var,
                disaggregatedVar=var,
                disjunct=obj,
                bigmConstraint=bigmConstraint,
                var_free_indicator=obj.indicator_var.get_associated_binary(),
                x0_map=x0_map,  # trivial in this case
            )
            # update the bigm constraint mappings
            data_dict = var.parent_block().private_data()
            data_dict.bigm_constraint_map[var][obj] = bigmConstraint
            disjunct_disaggregated_var_map[obj][var] = var

        var_substitute_map = dict(
            (id(v), newV) for v, newV in disjunct_disaggregated_var_map[obj].items()
        )
        x0_substitute_map = dict(
            (id(v), x0_map[v])
            for v, newV in disjunct_disaggregated_var_map[obj].items()
        )

        # Transform each component within this disjunct. In particular,
        # call _transform_constraint on each constraint.
        self._transform_block_components(
            obj, obj, var_substitute_map, x0_substitute_map
        )

        # deactivate disjunct so writers can be happy
        obj._deactivate_without_fixing_indicator()

    def _declare_disaggregated_var_bounds(
        self,
        original_var,
        disaggregatedVar,
        disjunct,
        bigmConstraint,
        var_free_indicator,
        x0_map,
        var_idx=None,
    ):
        # For updating mappings:
        original_var_info = original_var.parent_block().private_data()
        disaggregated_var_map = original_var_info.disaggregated_var_map
        disaggregated_var_info = disaggregatedVar.parent_block().private_data()

        disaggregated_var_info.bigm_constraint_map[disaggregatedVar][disjunct] = {}

        if original_var.lb is None or original_var.ub is None:
            raise GDP_Error(
                "Variables that appear in disjuncts must be "
                "bounded in order to use the hull "
                "transformation! Missing bound for %s." % (original_var.name)
            )
        lb = original_var.lb - x0_map[original_var]
        ub = original_var.ub - x0_map[original_var]

        disaggregatedVar.setlb(min(0, lb))
        disaggregatedVar.setub(max(0, ub))

        if lb:
            lb_idx = 'lb'
            if var_idx is not None:
                lb_idx = (var_idx, 'lb')
            bigmConstraint.add(lb_idx, var_free_indicator * lb <= disaggregatedVar)
            disaggregated_var_info.bigm_constraint_map[disaggregatedVar][disjunct][
                'lb'
            ] = bigmConstraint[lb_idx]
        if ub:
            ub_idx = 'ub'
            if var_idx is not None:
                ub_idx = (var_idx, 'ub')
            bigmConstraint.add(ub_idx, disaggregatedVar <= ub * var_free_indicator)
            disaggregated_var_info.bigm_constraint_map[disaggregatedVar][disjunct][
                'ub'
            ] = bigmConstraint[ub_idx]

        # store the mappings from variables to their disaggregated selves on
        # the transformation block
        disaggregated_var_map[disjunct][original_var] = disaggregatedVar
        disaggregated_var_info.original_var_map[disaggregatedVar] = original_var

    def _get_local_var_list(self, parent_disjunct):
        # Add or retrieve Suffix from parent_disjunct so that, if this is
        # nested, we can use it to declare that the disaggregated variables are
        # local. We return the list so that we can add to it.
        local_var_list = None
        if parent_disjunct is not None:
            # This limits the cases that a user is allowed to name something
            # (other than a Suffix) 'LocalVars' on a Disjunct. But I am assuming
            # that the Suffix has to be somewhere above the disjunct in the
            # tree, so I can't put it on a Block that I own. And if I'm coopting
            # something of theirs, it may as well be here.
            self._get_local_var_suffix(parent_disjunct)
            if parent_disjunct.LocalVars.get(parent_disjunct) is None:
                parent_disjunct.LocalVars[parent_disjunct] = []
            local_var_list = parent_disjunct.LocalVars[parent_disjunct]

        return local_var_list

    def _transform_constraint(
        self, obj, disjunct, var_substitute_map, x0_substitute_map
    ):
        """Transform a single Constraint on a Disjunct.

        Applies the appropriate hull reformulation to each
        ``ConstraintData`` in ``obj``. When ``exact_hull_quadratic`` is
        enabled and the constraint body has polynomial degree 2, an exact
        hull formulation is used instead of the perspective function.

        Parameters
        ----------
        obj : Constraint
            The Constraint component to transform.
        disjunct : Disjunct
            The Disjunct that owns ``obj``.
        var_substitute_map : dict
            Mapping from ``id(original_var)`` to its disaggregated
            counterpart.
        zero_substitute_map : dict
            Mapping from ``id(original_var)`` to ``ZeroConstant``.
        """
        # we will put a new transformed constraint on the relaxation block.
        relaxationBlock = disjunct._transformation_block()
        constraint_map = relaxationBlock.private_data('pyomo.gdp')

        # We will make indexes from ({obj.local_name} x obj.index_set() x ['lb',
        # 'ub']), but don't bother construct that set here, as taking Cartesian
        # products is kind of expensive (and redundant since we have the
        # original model)
        newConstraint = relaxationBlock.transformedConstraints

        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue

            unique = len(newConstraint)
            name = c.local_name + "_%s" % unique

            # QuadraticRepnVisitor / LinearRepnVisitor treat fixed Vars as
            # constants.  When assume_fixed_vars_permanent is False (default),
            # fixed variables are still disaggregated, so we temporarily unfix
            # them so that the quadratic walk and variable substitutions all
            # see the full variable structure.
            to_refix = ComponentSet()
            if not self._config.assume_fixed_vars_permanent:
                for var in EXPR.identify_variables(c.body, include_fixed=True):
                    if var.fixed:
                        to_refix.add(var)
                        var.unfix()

            EPS = self._config.EPS
            mode = self._config.perspective_function
            exact_quad = self._config.exact_hull_quadratic

            # Analyze the body with QuadraticRepnVisitor (while vars are
            # still unfixed) to classify it as linear / quadratic / general
            # nonlinear and, when exact_hull_quadratic is enabled, reuse the
            # repn for the exact reformulation.
            var_map = {}
            var_order = {}
            visitor = QuadraticRepnVisitor(
                {},
                var_recorder=OrderedVarRecorder(
                    var_map, var_order, SortComponents.deterministic
                ),
            )
            qrepn = visitor.walk_expression(c.body)

            is_quadratic = bool(qrepn.quadratic)
            NL = is_quadratic or qrepn.nonlinear is not None
            use_exact_quad = exact_quad and is_quadratic and qrepn.nonlinear is None

            for var in to_refix:
                var.fix()

            if use_exact_quad:
                self._build_exact_quadratic_hull(
                    c,
                    y=disjunct.binary_indicator_var,
                    disjunct=disjunct,
                    relaxationBlock=relaxationBlock,
                    constraint_map=constraint_map,
                    var_substitute_map=var_substitute_map,
                    newConstraint=newConstraint,
                    name=name,
                    idx=i,
                    obj=obj,
                    qrepn=qrepn,
                    var_map=var_map,
                )
                continue

            # We need to evaluate the expression at the origin *before*
            # we substitute the expression variables with the
            # disaggregated variables
            if not NL or mode == "FurmanSawayaGrossmann":
                # Only permanently fixed vars are not being substituted
                # by the x0_substitute_map
                h_x0 = value(
                    clone_without_expression_components(
                        c.body, substitute=x0_substitute_map
                    )
                )

            y = disjunct.binary_indicator_var

            if NL:
                if mode == "LeeGrossmann":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (var, (subs / y) + x0_substitute_map[var])
                            for var, subs in var_substitute_map.items()
                        ),
                    )
                    expr = sub_expr * y
                elif mode == "GrossmannLee":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (var, (subs / (y + EPS)) + x0_substitute_map[var])
                            for var, subs in var_substitute_map.items()
                        ),
                    )
                    expr = (y + EPS) * sub_expr
                elif mode == "FurmanSawayaGrossmann":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (
                                var,
                                (subs / ((1 - EPS) * y + EPS)) + x0_substitute_map[var],
                            )
                            for var, subs in var_substitute_map.items()
                        ),
                    )
                    expr = ((1 - EPS) * y + EPS) * sub_expr - EPS * h_x0 * (1 - y)
                else:
                    raise RuntimeError("Unknown NL Hull mode")
            else:
                # For a linear constraint that looks like a^Tx + b <= c,
                # the transformed constraint will be a^Tv <= lambda (c -
                # b - a^Tx_0). Get the a^Tv here and note that b +
                # a^Tx_0 is exactly h_x0 from earlier, so we will have
                # it when we need it.
                #
                # Note: linear_repn.multiplier is always 1 when obtained
                # from LinearRepnVisitor.walk_expression so we do not
                # need to read it
                expr = sum(
                    coef * var_substitute_map[var]
                    for var, coef in qrepn.linear.items()
                    if coef != 0
                )

            if c.equality:
                if self._generate_debug_messages:
                    _name = c.getname(fully_qualified=True)
                    logger.debug("GDP(Hull): Transforming constraint '%s'", _name)
                if NL:
                    # NOTE: This nonlinear equality constraint is
                    # probably nonconvex, depending on your definition
                    # of nonconvex (it is never in the standard form for
                    # a convex problem unless the constraint body is a
                    # disguised affine function, but the feasible region
                    # may still be convex under weaker conditions,
                    # e.g. quasilinearity). But even so, this
                    # reformulation is still correct, so we will not
                    # complain to the user.
                    newConsExpr = expr == c.lower * y
                else:
                    if len(qrepn.linear) == 1:
                        var, coef = next(iter(qrepn.linear.items()))
                        # Second clause of this condition happens iff
                        # the constraint implies x = x0
                        if (
                            coef != 0
                            and (c.lower - qrepn.constant) / coef
                            == x0_substitute_map[var]
                        ):
                            v = var_substitute_map[var]
                            # Setting a variable to 0 in a disjunct is
                            # *very* common. We should recognize that in
                            # that structure, the disaggregated variable
                            # will also be fixed to 0. In the
                            # general-offset case, this happens when the
                            # equality constraint is of the form x =
                            # x_0. We're unlikely to hit this unless x_0
                            # is the origin or was passed manually, but
                            # that's fine - nonzero x_0 is already a
                            # rare special case.
                            v.fix(0)
                            # ESJ: If you ask where the transformed
                            # constraint is, the answer is
                            # nowhere. Really, it is in the bounds of
                            # this variable, so I'm going to return
                            # it. Alternatively we could return an empty
                            # list, but I think I like this better.
                            constraint_map.transformed_constraints[c].append(v)
                            # Reverse map also (this is strange)
                            constraint_map.src_constraint[v] = c
                            continue
                    newConsExpr = expr == (c.lower - h_x0) * y

                if obj.is_indexed():
                    newConstraint.add((name, i, 'eq'), newConsExpr)
                    # map the ConstraintDatas (we mapped the container above)
                    constraint_map.transformed_constraints[c].append(
                        newConstraint[name, i, 'eq']
                    )
                    constraint_map.src_constraint[newConstraint[name, i, 'eq']] = c
                else:
                    newConstraint.add((name, 'eq'), newConsExpr)
                    # map to the ConstraintData (And yes, for
                    # ScalarConstraints, this is overwriting the map to the
                    # container we made above, and that is what I want to
                    # happen. ScalarConstraints will map to lists. For
                    # IndexedConstraints, we can map the container to the
                    # container, but more importantly, we are mapping the
                    # ConstraintDatas to each other above)
                    constraint_map.transformed_constraints[c].append(
                        newConstraint[name, 'eq']
                    )
                    constraint_map.src_constraint[newConstraint[name, 'eq']] = c

                continue

            if c.lower is not None:
                if self._generate_debug_messages:
                    _name = c.getname(fully_qualified=True)
                    logger.debug("GDP(Hull): Transforming constraint " + "'%s'", _name)
                if NL:
                    newConsExpr = expr >= c.lower * y
                else:
                    newConsExpr = expr >= (c.lower - h_x0) * y

                if obj.is_indexed():
                    newConstraint.add((name, i, 'lb'), newConsExpr)
                    constraint_map.transformed_constraints[c].append(
                        newConstraint[name, i, 'lb']
                    )
                    constraint_map.src_constraint[newConstraint[name, i, 'lb']] = c
                else:
                    newConstraint.add((name, 'lb'), newConsExpr)
                    constraint_map.transformed_constraints[c].append(
                        newConstraint[name, 'lb']
                    )
                    constraint_map.src_constraint[newConstraint[name, 'lb']] = c

            if c.upper is not None:
                if self._generate_debug_messages:
                    _name = c.getname(fully_qualified=True)
                    logger.debug("GDP(Hull): Transforming constraint " + "'%s'", _name)
                if NL:
                    newConsExpr = expr <= c.upper * y
                else:
                    newConsExpr = expr <= (c.upper - h_x0) * y

                if obj.is_indexed():
                    newConstraint.add((name, i, 'ub'), newConsExpr)
                    # map (have to account for fact we might have created list
                    # above
                    constraint_map.transformed_constraints[c].append(
                        newConstraint[name, i, 'ub']
                    )
                    constraint_map.src_constraint[newConstraint[name, i, 'ub']] = c
                else:
                    newConstraint.add((name, 'ub'), newConsExpr)
                    constraint_map.transformed_constraints[c].append(
                        newConstraint[name, 'ub']
                    )
                    constraint_map.src_constraint[newConstraint[name, 'ub']] = c

        # deactivate now that we have transformed
        obj.deactivate()

    def _build_exact_quadratic_hull(
        self,
        c,
        y,
        disjunct,
        relaxationBlock,
        constraint_map,
        var_substitute_map,
        newConstraint,
        name,
        idx,
        obj,
        qrepn,
        var_map,
    ):
        """Build the exact hull reformulation for a single quadratic constraint.

        Implements the reformulation from Gusev & Bernal Neira (2025),
        arXiv:2508.16093.  For a constraint whose body is a quadratic of the
        form ``x'Qx + c'x + d``, this method constructs either the conic exact
        hull (when the quadratic is convex with respect to the bound
        direction) or the general exact hull (otherwise).

        Conic exact hull (convex case): introduces an auxiliary variable
        ``t >= 0`` and a rotated second-order cone constraint
        ``v'Q_psd v <= t * y``, then replaces the original bound with a
        linear constraint on ``t + c'v + d*y``.

        General exact hull (non-convex / equality case): directly
        substitutes the quadratic form to ``v'Qv + c'v*y + d*y**2``.

        Parameters
        ----------
        c : ConstraintData
            The individual constraint data object being transformed.
        y : Var
            The binary indicator variable for the parent disjunct.
        disjunct : Disjunct
            The Disjunct that owns the constraint.
        relaxationBlock : Block
            The transformation block for this disjunct.
        constraint_map : object
            Private data object tracking constraint mappings.
        var_substitute_map : dict
            Mapping from ``id(original_var)`` to disaggregated variable.
        newConstraint : Constraint
            The indexed Constraint container for transformed constraints.
        name : str
            Base name for the transformed constraint indices.
        idx : object
            The index of the constraint in its parent component.
        obj : Constraint
            The parent Constraint component (needed for ``is_indexed``).
        qrepn : QuadraticRepn
            Result of ``QuadraticRepnVisitor.walk_expression`` on the constraint
            body (walked by the caller with fixed Vars temporarily unfixed).
        var_map : dict
            The visitor's ``var_map``: maps ``id(Var)`` to ``Var`` objects.
        """
        const_term = qrepn.constant if qrepn.constant is not None else 0

        quad_items = sorted(
            qrepn.quadratic.items(), key=lambda item: (item[0][0], item[0][1])
        )

        if not numpy_available:
            raise GDP_Error(
                "exact_hull_quadratic requires NumPy for convexity checks. "
                "NumPy is not available in this environment."
            )

        # --- Build the symmetric Q matrix and determine convexity ---
        # Quadratic repn keys (vid_i, vid_j) are id(Var); assign contiguous
        # row/column indices for eigh. Off-diagonal repn coefficients are
        # split across Q[i,j] and Q[j,i] so Q matches the symmetric Hessian.
        var_to_idx = {}
        for (vid_i, vid_j), _coef in quad_items:
            if vid_i not in var_to_idx:
                var_to_idx[vid_i] = len(var_to_idx)
            if vid_j not in var_to_idx:
                var_to_idx[vid_j] = len(var_to_idx)
        n_vars = len(var_to_idx)
        Q = np.zeros((n_vars, n_vars))

        for (vid_i, vid_j), coef in quad_items:
            idx_i = var_to_idx[vid_i]
            idx_j = var_to_idx[vid_j]
            if vid_i == vid_j:
                Q[idx_i, idx_i] += coef
            else:
                Q[idx_i, idx_j] += 0.5 * coef
                Q[idx_j, idx_i] += 0.5 * coef

        numerical_tolerance = self._config.eigenvalue_tolerance
        eigenvalues, _ = np.linalg.eigh(Q)
        Q_is_psd = not np.any(eigenvalues < -numerical_tolerance)
        Q_is_nsd = not np.any(eigenvalues > numerical_tolerance)

        # Determine which bounds can use the conic formulation
        use_conic_upper = False
        use_conic_lower = False
        negate_for_conic = False

        if Q_is_psd and Q_is_nsd:
            # All eigenvalues lie within the tolerance band [-tol, tol].
            # In this numerically ambiguous case, we conservatively avoid the
            # conic reformulation and fall back to the general exact hull
            # reformulation.  When Q is simultaneously PSD and NSD (within
            # tolerance), both use_conic_upper and use_conic_lower would be
            # set for two-sided (range) constraints, but a single conic
            # expression built with negate_for_conic=True cannot correctly
            # serve both bounds.  Falling back to the general path avoids
            # this issue entirely.
            # Only warn for inequality constraints; equality constraints always
            # use the general exact hull path, so no fallback warning is needed.
            if not c.equality:
                max_abs_eigenvalue = float(np.max(np.abs(eigenvalues)))
                logger.warning(
                    "GDP(Hull): Constraint '%s' has quadratic terms, but all "
                    "eigenvalues of the Q matrix are within the "
                    "eigenvalue_tolerance band (largest eigenvalue by modulus: "
                    "%g). The conic reformulation cannot be applied; the "
                    "constraint will be handled by the general exact hull "
                    "reformulation instead. If this is not the expected behavior, "
                    "consider using a tighter (smaller) eigenvalue_tolerance.",
                    c.getname(fully_qualified=True),
                    max_abs_eigenvalue,
                )
        else:
            if c.upper is not None and not c.equality:
                if Q_is_psd:
                    use_conic_upper = True
            if c.lower is not None and not c.equality:
                if Q_is_nsd:
                    use_conic_lower = True
                    negate_for_conic = True

        # --- Decide which expression forms are needed ---
        need_non_convex = False
        if c.equality:
            need_non_convex = True
        if c.upper is not None and not use_conic_upper:
            need_non_convex = True
        if c.lower is not None and not use_conic_lower:
            need_non_convex = True

        non_conv_expr = None
        conic_expr_linear = None

        if need_non_convex:
            non_conv_expr = self._build_general_exact_hull_expr(
                qrepn, var_map, var_substitute_map, y, const_term
            )

        if use_conic_upper or use_conic_lower:
            conic_expr_linear = self._build_conic_exact_hull_expr(
                c,
                y,
                disjunct,
                relaxationBlock,
                constraint_map,
                qrepn,
                var_map,
                var_substitute_map,
                const_term,
                negate_for_conic,
            )

        # --- Equality constraints always use general exact hull ---
        if c.equality:
            if self._generate_debug_messages:
                _name = c.getname(fully_qualified=True)
                logger.debug("GDP(Hull): Transforming constraint '%s'", _name)
            newConsExpr = non_conv_expr == c.lower * y**2

            if obj.is_indexed():
                newConstraint.add((name, idx, 'eq'), newConsExpr)
                constraint_map.transformed_constraints[c].append(
                    newConstraint[name, idx, 'eq']
                )
                constraint_map.src_constraint[newConstraint[name, idx, 'eq']] = c
            else:
                newConstraint.add((name, 'eq'), newConsExpr)
                constraint_map.transformed_constraints[c].append(
                    newConstraint[name, 'eq']
                )
                constraint_map.src_constraint[newConstraint[name, 'eq']] = c
            return

        # --- Lower bound ---
        if c.lower is not None:
            if self._generate_debug_messages:
                _name = c.getname(fully_qualified=True)
                logger.debug("GDP(Hull): Transforming constraint '%s'", _name)

            if use_conic_lower:
                newConsExpr = conic_expr_linear <= -c.lower * y
            else:
                newConsExpr = non_conv_expr >= c.lower * y**2

            if obj.is_indexed():
                newConstraint.add((name, idx, 'lb'), newConsExpr)
                constraint_map.transformed_constraints[c].append(
                    newConstraint[name, idx, 'lb']
                )
                constraint_map.src_constraint[newConstraint[name, idx, 'lb']] = c
            else:
                newConstraint.add((name, 'lb'), newConsExpr)
                constraint_map.transformed_constraints[c].append(
                    newConstraint[name, 'lb']
                )
                constraint_map.src_constraint[newConstraint[name, 'lb']] = c

        # --- Upper bound ---
        if c.upper is not None:
            if self._generate_debug_messages:
                _name = c.getname(fully_qualified=True)
                logger.debug("GDP(Hull): Transforming constraint '%s'", _name)

            if use_conic_upper:
                newConsExpr = conic_expr_linear <= c.upper * y
            else:
                newConsExpr = non_conv_expr <= c.upper * y**2

            if obj.is_indexed():
                newConstraint.add((name, idx, 'ub'), newConsExpr)
                constraint_map.transformed_constraints[c].append(
                    newConstraint[name, idx, 'ub']
                )
                constraint_map.src_constraint[newConstraint[name, idx, 'ub']] = c
            else:
                newConstraint.add((name, 'ub'), newConsExpr)
                constraint_map.transformed_constraints[c].append(
                    newConstraint[name, 'ub']
                )
                constraint_map.src_constraint[newConstraint[name, 'ub']] = c

    def _build_general_exact_hull_expr(
        self, qrepn, var_map, var_substitute_map, y, const_term
    ):
        """Build the general exact hull expression for a quadratic constraint.

        Constructs the expression ``v'Qv + c'v*y + d*y**2`` where ``v`` are
        disaggregated variables, ``y`` is the indicator, ``c`` are linear
        coefficients, and ``d`` is the constant term.

        Parameters
        ----------
        qrepn : QuadraticRepn
            Result of ``QuadraticRepnVisitor.walk_expression`` on the constraint
            body.
        var_map : dict
            Maps ``id(Var)`` to ``Var`` for variables appearing in ``qrepn``.
        var_substitute_map : dict
            Mapping from ``id(original_var)`` to disaggregated variable.
        y : Var
            The binary indicator variable.
        const_term : float
            Constant term (``qrepn.constant``, normalized).

        Returns
        -------
        expression
            The general exact hull Pyomo expression.
        """
        expr = 0

        for (vid_i, vid_j), coef in qrepn.quadratic.items():
            var_i = var_map[vid_i]
            var_j = var_map[vid_j]
            v_i = var_substitute_map.get(id(var_i), var_i)
            v_j = var_substitute_map.get(id(var_j), var_j)
            if var_i is var_j:
                expr += coef * v_i**2
            else:
                expr += coef * v_i * v_j

        if qrepn.linear:
            for vid, coef in sorted(qrepn.linear.items(), key=lambda item: item[0]):
                var = var_map[vid]
                v = var_substitute_map.get(id(var), var)
                expr += coef * v * y

        if const_term:
            expr += const_term * y**2

        return expr

    def _build_conic_exact_hull_expr(
        self,
        c,
        y,
        disjunct,
        relaxationBlock,
        constraint_map,
        qrepn,
        var_map,
        var_substitute_map,
        const_term,
        negate_for_conic,
    ):
        """Build the conic exact hull expression for a convex quadratic.

        Creates an auxiliary variable ``t >= 0`` and a rotated second-order
        cone constraint ``v'Q_psd v <= t * y``, then returns the linear
        expression ``t + c'v + d*y`` (with signs adjusted for lower-bound
        constraints that required negation).

        Parameters
        ----------
        c : ConstraintData
            The constraint data being transformed.
        y : Var
            The binary indicator variable.
        disjunct : Disjunct
            The parent Disjunct.
        relaxationBlock : Block
            The transformation block for the disjunct.
        constraint_map : object
            Private data tracking constraint mappings.
        qrepn : QuadraticRepn
            Result of ``QuadraticRepnVisitor.walk_expression`` on the constraint
            body.
        var_map : dict
            Maps ``id(Var)`` to ``Var`` for variables appearing in ``qrepn``.
        var_substitute_map : dict
            Mapping from ``id(original_var)`` to disaggregated variable.
        const_term : float
            Constant term (``qrepn.constant``, normalized).
        negate_for_conic : bool
            If ``True``, coefficients are negated (used when the
            lower-bound constraint is reformulated by negation to obtain a
            PSD form).

        Returns
        -------
        expression
            The linear Pyomo expression ``t + c'v + d*y`` (or its negated
            variant) to be bounded by the constraint's RHS.
        """
        t = Var(domain=NonNegativeReals)
        t_name = unique_component_name(
            relaxationBlock,
            '_conic_aux_t_%s' % c.getname(fully_qualified=True, relative_to=disjunct),
        )
        relaxationBlock.add_component(t_name, t)
        relaxationBlock.private_data().exact_quadratic_aux_var_map[c] = t

        linear_expr = t

        if qrepn.linear:
            for vid, coef in sorted(qrepn.linear.items(), key=lambda item: item[0]):
                var = var_map[vid]
                v = var_substitute_map.get(id(var), var)
                actual_coef = -coef if negate_for_conic else coef
                linear_expr += actual_coef * v

        if const_term:
            actual_const = -const_term if negate_for_conic else const_term
            linear_expr += actual_const * y

        # Build rotated SOC: v'Q_psd v <= t * y
        # NOTE: For a general PSD matrix Q this is not in canonical rotated
        # second-order cone form (sum-of-squares <= product).  Some solvers or
        # writer interfaces may therefore treat the bilinear term t*y on the
        # right-hand side as a generic nonconvex product rather than
        # recognizing a rotated quadratic cone.  This formulation was chosen
        # deliberately based on computational testing with Gurobi and SCIP,
        # where the generic quadratic form overall performed better than
        # other formulations tested.
        quadratic_form = 0
        for (vid_i, vid_j), coef in qrepn.quadratic.items():
            var_i = var_map[vid_i]
            var_j = var_map[vid_j]
            v_i = var_substitute_map.get(id(var_i), var_i)
            v_j = var_substitute_map.get(id(var_j), var_j)
            actual_coef = -coef if negate_for_conic else coef
            if var_i is var_j:
                quadratic_form += actual_coef * v_i**2
            else:
                quadratic_form += actual_coef * v_i * v_j

        conic_constraint_name = unique_component_name(
            relaxationBlock,
            '_conic_constraint_%s'
            % c.getname(fully_qualified=True, relative_to=disjunct),
        )
        conic_constraint = Constraint(expr=quadratic_form <= t * y)
        relaxationBlock.add_component(conic_constraint_name, conic_constraint)

        constraint_map.transformed_constraints[c].append(conic_constraint)
        constraint_map.src_constraint[conic_constraint] = c

        return linear_expr

    def _get_local_var_suffix(self, disjunct):
        # If the Suffix is there, we will borrow it. If not, we make it. If it's
        # something else, we complain.
        localSuffix = disjunct.component("LocalVars")
        if localSuffix is None:
            disjunct.LocalVars = Suffix(direction=Suffix.LOCAL)
        else:
            if localSuffix.ctype is Suffix:
                return
            raise GDP_Error(
                "A component called 'LocalVars' is declared on "
                "Disjunct %s, but it is of type %s, not Suffix."
                % (disjunct.getname(fully_qualified=True), localSuffix.ctype)
            )

    def get_disaggregated_var(self, v, disjunct, raise_exception=True):
        """
        Returns the disaggregated variable corresponding to the Var v and the
        Disjunct disjunct.

        If v is a local variable, this method will return v.

        Parameters
        ----------
        v: a Var that appears in a constraint in a transformed Disjunct
        disjunct: a transformed Disjunct in which v appears
        """
        if disjunct._transformation_block is None:
            raise GDP_Error("Disjunct '%s' has not been transformed" % disjunct.name)
        msg = (
            "It does not appear '%s' is a "
            "variable that appears in disjunct '%s'" % (v.name, disjunct.name)
        )
        disaggregated_var_map = v.parent_block().private_data().disaggregated_var_map
        if v in disaggregated_var_map[disjunct]:
            return disaggregated_var_map[disjunct][v]
        else:
            if raise_exception:
                raise GDP_Error(msg)

    def get_src_var(self, disaggregated_var):
        """
        Returns the original model variable to which disaggregated_var
        corresponds.

        Parameters
        ----------
        disaggregated_var: a Var that was created by the hull
                           transformation as a disaggregated variable
                           (and so appears on a transformation block
                           of some Disjunct)
        """
        var_map = disaggregated_var.parent_block().private_data()
        if disaggregated_var in var_map.original_var_map:
            return var_map.original_var_map[disaggregated_var]
        raise GDP_Error(
            "'%s' does not appear to be a "
            "disaggregated variable" % disaggregated_var.name
        )

    # retrieves the disaggregation constraint for original_var resulting from
    # transforming disjunction
    def get_disaggregation_constraint(
        self, original_var, disjunction, raise_exception=True
    ):
        """
        Returns the disaggregation (re-aggregation?) constraint
        (which links the disaggregated variables to their original)
        corresponding to original_var and the transformation of disjunction.

        Parameters
        ----------
        original_var: a Var which was disaggregated in the transformation
                      of Disjunction disjunction
        disjunction: a transformed Disjunction containing original_var
        """
        for disjunct in disjunction.disjuncts:
            transBlock = disjunct.transformation_block
            if transBlock is not None:
                break
        if transBlock is None:
            raise GDP_Error(
                "Disjunction '%s' has not been properly "
                "transformed:"
                " None of its disjuncts are transformed." % disjunction.name
            )

        try:
            cons = (
                transBlock.parent_block()
                .private_data()
                .disaggregation_constraint_map[original_var][disjunction]
            )
        except Exception:
            if raise_exception:
                logger.error(
                    "It doesn't appear that '%s' is a variable that was "
                    "disaggregated by Disjunction '%s'"
                    % (original_var.name, disjunction.name)
                )
                raise
            return None
        while not cons.active:
            cons = self.get_transformed_constraints(cons)[0]
        return cons

    def get_var_bounds_constraint(self, v, disjunct=None):
        """
        Returns a dictionary mapping keys 'lb' and/or 'ub' to the Constraints that
        set a disaggregated variable to be within its lower and upper bounds
        (respectively) when its Disjunct is active and to be 0 otherwise.

        Parameters
        ----------
        v: a Var that was created by the hull transformation as a
           disaggregated variable (and so appears on a transformation
           block of some Disjunct)
        disjunct: (For nested Disjunctions) Which Disjunct in the
           hierarchy the bounds Constraint should correspond to.
           Optional since for non-nested models this can be inferred.
        """
        info = v.parent_block().private_data()
        if v in info.bigm_constraint_map:
            if len(info.bigm_constraint_map[v]) == 1:
                # Not nested, or it's at the top layer, so we're fine.
                return list(info.bigm_constraint_map[v].values())[0]
            elif disjunct is not None:
                # This is nested, so we need to walk up to find the active ones
                return info.bigm_constraint_map[v][disjunct]
            else:
                raise ValueError(
                    "It appears that the variable '%s' appears "
                    "within a nested GDP hierarchy, and no "
                    "'disjunct' argument was specified. Please "
                    "specify for which Disjunct the bounds "
                    "constraint for '%s' should be returned." % (v, v)
                )
        raise GDP_Error(
            "Either '%s' is not a disaggregated variable, or "
            "the disjunction that disaggregates it has not "
            "been properly transformed." % v.name
        )

    def get_exact_quadratic_aux_var(self, cons):
        """Return the conic exact-hull auxiliary variable for ``cons``.

        Parameters
        ----------
        cons : ConstraintData
            Original constraint on a Disjunct that may have been reformulated
            using the conic exact hull.

        Returns
        -------
        Var or None
            The auxiliary nonnegative ``t`` variable introduced by the conic
            exact hull for ``cons``, or ``None`` if that reformulation was not
            used for ``cons``.
        """
        disjunct = cons.parent_block()
        if disjunct is None or disjunct.ctype is not Disjunct:
            raise GDP_Error(
                "Constraint '%s' does not appear to belong to a Disjunct" % cons.name
            )
        relaxationBlock = disjunct.transformation_block
        if relaxationBlock is None:
            raise GDP_Error("Disjunct '%s' has not been transformed" % disjunct.name)
        return relaxationBlock.private_data().exact_quadratic_aux_var_map.get(
            cons, None
        )

    def get_transformed_constraints(self, cons):
        cons = super().get_transformed_constraints(cons)
        while not cons[0].active:
            transformed_cons = []
            for con in cons:
                transformed_cons += super().get_transformed_constraints(con)
            cons = transformed_cons
        return cons

    def get_well_defined_points_map(self, b):
        """
        Retrieve the well-defined points originally used to transform
        a Block. Format is a dict of ComponentMaps identical to
        that of the parameter well_defined_points.

        Parameters
        ----------
        b: a Block that was transformed by gdp.hull
        """
        return b.private_data().well_defined_points_map


@TransformationFactory.register(
    'gdp.chull',
    doc="[DEPRECATED] please use 'gdp.hull' to get the Hull transformation.",
)
@deprecated(
    "The 'gdp.chull' name is deprecated. "
    "Please use the more apt 'gdp.hull' instead.",
    logger='pyomo.gdp',
    version="5.7",
)
class _Deprecated_Name_Hull(Hull_Reformulation):
    def __init__(self):
        super(_Deprecated_Name_Hull, self).__init__()


# Walk the expression and, whenever encountering an expression that
# could fail to be well-defined, create a constraint that keeps it
# well-defined. This is done at the Pyomo level to make this capability
# more generic instead of needing to specially set up the options for
# each solver (plus, some solvers get rather buggy when used for this
# task).
class _WellDefinedConstraintGenerator(StreamBasedExpressionVisitor):
    def __init__(self, cons_list, **kwds):
        self.cons_list = cons_list
        super().__init__(**kwds)

    # Whenever we exit a node (entering would also be fine) check if it
    # has a restricted domain, and if it does add a corresponding
    # constraint
    def exitNode(self, node, data):
        if node.__class__ in _expr_handlers:
            for con in _expr_handlers[node.__class__](node):
                # note: con should never be a boolean True here, such
                # cases should have been filtered out during the handler
                # call
                self.cons_list.add(con)


# Epsilon for handling function domains with strict inequalities. This
# is a heuristic so it's not important for this to be tight.
EPS_HEURISTIC = 1e-4


def _handlePowExpression(node):
    base, exp = node.args
    # if base is not variable, nothing for us to do
    if base.__class__ in EXPR.native_types or not base.is_potentially_variable():
        return ()

    # If exp is a NPV nonnegative integer, there are no restrictions on
    # base. If exp is a NPV negative integer, base should not be
    # zero. If exp is a NPV nonnegative fraction, base should not be
    # negative. Otherwise, base should be strictly positive (as we can't
    # be sure that exp could not be negative or fractional).

    # Note: this is problematic for LP, but I don't want to potentially
    # invoke a MIP solve here, so replace "x is nonzero" with "x is >=
    # eps". It's a heuristic so this is not critical.
    if exp.__class__ in EXPR.native_types or not exp.is_potentially_variable():
        val = value(exp)
        if round(val) == val:
            if val >= 0:
                return ()
            else:
                # return base != 0
                return (base >= EPS_HEURISTIC,)
        elif val >= 0:
            return (base >= 0,)
    return (base >= EPS_HEURISTIC,)


def _handleDivisionExpression(node):
    # No division by zero. Dividing by an NPV is always allowed.
    arg = node.args[1]
    if arg.__class__ in EXPR.native_types or not arg.is_potentially_variable():
        return ()
    # Same LP vs MIP problem as before
    return (arg >= EPS_HEURISTIC,)


def _handleUnaryFunctionExpression(node):
    arg = node.args[0]
    if (
        node.name in _unary_functions_unrestricted
        or arg.__class__ in EXPR.native_types
        or not arg.is_potentially_variable()
    ):
        return ()
    elif node.name in _unary_function_handlers:
        return _unary_function_handlers[node.name](arg)
    else:
        raise GDP_Error(
            "Hull transformation base point heuristic: no domain "
            f"information available for unfamiliar unary function {node.name}"
        )


# Unary function handlers


def _handle_log(arg):
    return (arg >= EPS_HEURISTIC,)


def _handle_log10(arg):
    return (arg >= EPS_HEURISTIC,)


def _handle_sqrt(arg):
    return (arg >= 0,)


def _handle_asin(arg):
    return (arg >= -1, arg <= 1)


def _handle_acos(arg):
    return (arg >= -1, arg <= 1)


def _handle_tan(arg):
    # It can't be exactly pi/2 plus a multiple of pi. Rather difficult
    # to enforce, so make a conservative effort by instead keeping it in
    # (-pi/2, pi/2).
    return (arg >= -(math.pi / 2) + EPS_HEURISTIC, arg <= (math.pi / 2) - EPS_HEURISTIC)


def _handle_acosh(arg):
    return (arg >= 1,)


def _handle_atanh(arg):
    return (arg >= -1 + EPS_HEURISTIC, arg <= 1 - EPS_HEURISTIC)


# All expression types that can potentially be
# ill-defined:
_expr_handlers = {
    # You are on your own here
    # EXPR.ExternalFunctionExpression,
    EXPR.PowExpression: _handlePowExpression,
    EXPR.DivisionExpression: _handleDivisionExpression,
    EXPR.UnaryFunctionExpression: _handleUnaryFunctionExpression,
}

# All unary functions that can potentially be
# ill-defined:
_unary_function_handlers = {
    'log': _handle_log,
    'log10': _handle_log10,
    'sqrt': _handle_sqrt,
    'asin': _handle_asin,
    'acos': _handle_acos,
    'tan': _handle_tan,
    'acosh': _handle_acosh,
    'atanh': _handle_atanh,
}

# Unary functions that can never be ill-defined:
_unary_functions_unrestricted = {
    # Being a subclass of UnaryFunctionExpression, 'abs' does not end up here - this
    # error check is available only for direct users of UnaryFunctionExpression.
    'ceil',
    'floor',
    'exp',
    'sin',
    'cos',
    'sinh',
    'cosh',
    'tanh',
    'atan',
    'asinh',
}
