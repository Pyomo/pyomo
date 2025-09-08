#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
import logging
import math
import os
import threading
import enum

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    InEnum,
    PositiveInt,
    document_class_CONFIG,
)
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.common.dependencies import dill, dill_available, multiprocessing

from pyomo.core import (
    Block,
    ConcreteModel,
    Constraint,
    maximize,
    minimize,
    NonNegativeIntegers,
    Objective,
    SortComponents,
    Suffix,
    value,
    Any,
)
from pyomo.core.base import Reference, TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.util import target_list

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
    _BigM_MixIn,
    _convert_M_to_tuple,
    _warn_for_unused_bigM_args,
)
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.util import _to_dict
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.contrib.solver.common.base import SolverBase as NewSolverBase
from pyomo.contrib.solver.common.base import LegacySolverWrapper
from pyomo.repn import generate_standard_repn

from weakref import ref as weakref_ref


logger = logging.getLogger('pyomo.gdp.mbigm')

_trusted_solvers = {
    'gurobi',
    'cplex',
    'cbc',
    'glpk',
    'scip',
    'xpress',
    'mosek',
    'baron',
    'highs',
}

# Whether these are thread-local or at module scope makes no difference
# for 'spawn' or 'forkserver', but it matters if we run single-threaded,
# and possibly for edge cases of 'fork' (although it's not correct to
# use fork() here while having multiple threads anyway, making it moot
# in theory).
_thread_local = threading.local()
_thread_local.solver = None
_thread_local.model = None
_thread_local.config_use_primal_bound = None
_thread_local.in_progress = False


def Solver(val):
    if isinstance(val, str):
        return SolverFactory(val)
    if not hasattr(val, 'solve'):
        raise ValueError("Expected a string or solver object (with solve() method)")
    if isinstance(val, NewSolverBase) and not isinstance(val, LegacySolverWrapper):
        raise ValueError(
            "Please pass an old-style solver object, using the "
            "LegacySolverWrapper mechanism if necessary."
        )
    return val


class ProcessStartMethod(str, enum.Enum):
    spawn = 'spawn'
    fork = 'fork'
    forkserver = 'forkserver'


@TransformationFactory.register(
    'gdp.mbigm',
    doc="Relax disjunctive model using big-M terms specific to each disjunct",
)
@document_class_CONFIG(methods=['apply_to', 'create_using'])
class MultipleBigMTransformation(GDP_to_MIP_Transformation, _BigM_MixIn):
    """
    Implements the multiple big-M transformation from [TG15]_. Note that this
    transformation is no different than the big-M transformation for two-
    term disjunctions, but that it may provide a tighter relaxation for
    models containing some disjunctions with three or more terms.
    """

    #: Global class configuration;
    #: see :ref:`pyomo.gdp.plugins.multiple_bigm.MultipleBigMTransformation::CONFIG`.
    CONFIG = ConfigDict('gdp.mbigm')

    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be relaxed",
            doc="""
        This specifies the list of components to relax. If None (default), the
        entire model is transformed. Note that if the transformation is done out
        of place, the list of targets should be attached to the model before it
        is cloned, and the list will specify the targets on the cloned
        instance.""",
        ),
    )
    CONFIG.declare(
        'assume_fixed_vars_permanent',
        ConfigValue(
            default=False,
            domain=bool,
            description="If False, transformed models will still be valid "
            "after unfixing fixed Vars",
            doc="""
        Boolean indicating whether or not to transform so that
        the transformed model will still be valid when fixed Vars are
        unfixed.

        This is only relevant when the transformation will be calculating M
        values. If True, the transformation will calculate M values assuming
        that fixed variables will always be fixed to their current values. This
        means that if a fixed variable is unfixed after transformation, the
        transformed model is potentially no longer valid. By default, the
        transformation will assume fixed variables could be unfixed in the
        future and will use their bounds to calculate the M value rather than
        their value. Note that this could make for a weaker LP relaxation
        while the variables remain fixed.
        """,
        ),
    )
    CONFIG.declare(
        'solver',
        ConfigValue(
            default='gurobi',
            domain=Solver,
            description="A solver to use to solve the continuous subproblems for "
            "calculating the M values",
        ),
    )
    CONFIG.declare(
        'bigM',
        ConfigValue(
            default=None,
            domain=_to_dict,
            description="Big-M values to use while relaxing constraints",
            doc="""
        Big-M values to use while relaxing constraints.

        A user-specified dict or ComponentMap mapping tuples of Constraints
        and Disjuncts to Big-M values valid for relaxing the constraint if
        the Disjunct is chosen.

        Note: Unlike in the bigm transformation, we require the keys in this
        mapping specify the components the M value applies to exactly in order
        to avoid ambiguity. However, if the 'only_mbigm_bound_constraints'
        option is True, this argument can be used as it would be in the
        traditional bigm transformation for the non-bound constraints.
        """,
        ),
    )
    CONFIG.declare(
        'reduce_bound_constraints',
        ConfigValue(
            default=True,
            domain=bool,
            description="Combine constraints in multiple disjuncts that "
            "bound a single variable into a single constraint",
            doc="""
        Flag indicating whether or not to handle disjunctive
        constraints that bound a single variable in a single
        constraint, rather than one per Disjunct.

        Given the not-uncommon special structure:

        [l_1 <= x <= u_1] v [l_2 <= x <= u_2] v ... v [l_K <= x <= u_K],

        instead of applying the rote transformation that would create 2*K
        different constraints in the relaxation, we can write two constraints:

        x >= l_1*y_1 + l_2*y_2 + ... + l_K*y_k
        x <= u_1*y_1 + u_2*y_2 + ... + u_K*y_K.

        This relaxation is as tight and has fewer constraints. This option is
        a flag to tell the mbigm transformation to detect this structure and
        handle it specially. Note that this is a special case of the 'Hybrid
        Big-M Formulation' from [Vie15]_ that takes advantage of the common left-
        hand side matrix for disjunctive constraints that bound a single
        variable.

        Note that we do not use user-specified M values for these constraints
        when this flag is set to True: If tighter bounds exist then they
        they should be put in the constraints.
        """,
        ),
    )
    CONFIG.declare(
        'only_mbigm_bound_constraints',
        ConfigValue(
            default=False,
            domain=bool,
            description="If True, only transform univariate bound constraints.",
            doc="""
        Flag indicating if only bound constraints should be transformed
        with multiple-bigm, or if all the disjunctive constraints
        should.

        Sometimes it is only computationally advantageous to apply multiple-
        bigm to disjunctive constraints with the special structure:

        [l_1 <= x <= u_1] v [l_2 <= x <= u_2] v ... v [l_K <= x <= u_K],

        and transform other disjunctive constraints with the traditional
        big-M transformation. This flag is used to set the above behavior.

        Note that the reduce_bound_constraints flag must also be True when
        this flag is set to True.
        """,
        ),
    )
    CONFIG.declare(
        'threads',
        ConfigValue(
            default=None,
            domain=PositiveInt,
            description="Number of worker processes to use when estimating M values",
            doc="""
            Number of worker processes to use when estimating M values.

            If not specified, use up to the number of available cores, minus
            one. If set to 1, do not spawn processes, and revert to
            single-threaded operation.
            """,
        ),
    )
    CONFIG.declare(
        'process_start_method',
        ConfigValue(
            default=None,
            domain=InEnum(ProcessStartMethod),
            description="Start method used for spawning processes during M calculation",
            doc="""
            Start method used for spawning processes during M calculation.

            Options are the elements of the enum ProcessStartMethod, or equivalently the
            strings 'fork', 'spawn', or 'forkserver', or None. See the Python
            multiprocessing documentation for a full description of each of these. When
            None is passed, we determine a reasonable default. On POSIX, the default is
            'fork', unless we detect that Python has multiple threads at the time the
            process pool is created, in which case we instead use 'forkserver'. On
            Windows, the default and only possible option is 'spawn'. Note that if
            'spawn' or 'forkserver' are selected, we depend on the `dill` module for
            pickling, and model instances must be pickleable using `dill`. This option is
            ignored if `threads` is set to 1.
            """,
        ),
    )
    CONFIG.declare(
        'use_primal_bound',
        ConfigValue(
            default=False,
            domain=bool,
            description="When estimating M values, use the primal bound "
            "instead of the dual bound.",
            doc="""
            When estimating M values, use the primal bound instead of the dual bound.

            This is necessary when using a local solver such as ipopt, but be
            aware that interior feasible points for this subproblem do not give
            valid values for M. That is, in the presence of numerical error,
            this option will lead to a slightly wrong reformulation.
            """,
        ),
    )
    transformation_name = 'mbigm'

    def __init__(self):
        super().__init__(logger)
        self._arg_list = {}
        self._set_up_expr_bound_visitor()
        self.handlers[Suffix] = self._warn_for_active_suffix

    def _apply_to(self, instance, **kwds):
        # check for the rather implausible error case that
        # solver.solve() is a metasolver that indirectly calls this
        # transformation again
        if _thread_local.in_progress:
            raise GDP_Error("gdp.mbigm transformation cannot be called recursively")
        self.used_args = ComponentMap()
        with PauseGC():
            try:
                self._apply_to_impl(instance, **kwds)
            finally:
                self._restore_state()
                self.used_args.clear()
                self._arg_list.clear()
                self._expr_bound_visitor.leaf_bounds.clear()
                self._expr_bound_visitor.use_fixed_var_values_as_bounds = False
                _thread_local.model = None
                _thread_local.solver = None
                _thread_local.config_use_primal_bound = None
                _thread_local.in_progress = False

    def _apply_to_impl(self, instance, **kwds):
        _thread_local.in_progress = True
        self._process_arguments(instance, **kwds)
        if self._config.assume_fixed_vars_permanent:
            self._bound_visitor.use_fixed_var_values_as_bounds = True

        if (
            self._config.only_mbigm_bound_constraints
            and not self._config.reduce_bound_constraints
        ):
            raise GDP_Error(
                "The 'only_mbigm_bound_constraints' option is set "
                "to True, but the 'reduce_bound_constraints' "
                "option is not. This is not supported--please also "
                "set 'reduce_bound_constraints' to True if you "
                "only wish to transform the bound constraints with "
                "multiple bigm."
            )

        # filter out inactive targets and handle case where targets aren't
        # specified.
        targets = self._filter_targets(instance)
        # transform any logical constraints that might be anywhere on the stuff
        # we're about to transform. We do this before we preprocess targets
        # because we will likely create more disjunctive components that will
        # need transformation.
        self._transform_logical_constraints(instance, targets)
        # We don't allow nested, so it doesn't much matter which way we sort
        # this. But transforming from leaf to root makes the error checking for
        # complaining about nested smoother, so we do that. We have to transform
        # a Disjunction at a time because, more similarly to hull than bigm, we
        # need information from the other Disjuncts in the Disjunction.
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        preprocessed_targets = gdp_tree.reverse_topological_sort()

        arg_Ms = self._config.bigM if self._config.bigM is not None else {}
        self._transform_disjunctionDatas(
            instance, preprocessed_targets, arg_Ms, gdp_tree
        )

        # issue warnings about anything that was in the bigM args dict that we
        # didn't use
        _warn_for_unused_bigM_args(self._config.bigM, self.used_args, logger)

    def _transform_disjunctionDatas(
        self, instance, preprocessed_targets, arg_Ms, gdp_tree
    ):
        # We wish we could do this one Disjunction at a time, but we
        # also want to calculate the Ms in parallel. So we first iterate
        # the Disjunctions once to get a list of M calculation jobs,
        # then we calculate the Ms in parallel, then we return to a
        # single thread and iterate Disjunctions again to actually
        # transform the constraints.

        # To-do list in form (constraint, other_disjunct, unsuccessful_message, is_upper)
        jobs = []
        # map Disjunction -> set of its active Disjuncts
        active_disjuncts = ComponentMap()
        # set of Constraints processed during special handling of bound
        # constraints: we will deactivate these, but not until we're
        # done calculating Ms
        transformed_constraints = set()
        # Finished M values. If we are only doing the bound constraints,
        # we will skip all the calculation steps and pass these through
        # to transform_constraints.
        Ms = {}
        if self._config.only_mbigm_bound_constraints:
            Ms = arg_Ms
        # Disjunctions and their setup components. We will return to these after
        # calculating Ms.
        disjunction_setup = {}

        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                if gdp_tree.root_disjunct(t) is not None:
                    # We do not support nested because, unlike in regular bigM, the
                    # constraints are not fully relaxed when the exactly-one constraint
                    # is not enforced. (For example, in this model: [1 <= x <= 3, [1 <=
                    # y <= 5] v [6 <= y <= 10]] v [5 <= x <= 10, 15 <= y <= 20]), we
                    # would need to put the relaxed inner-disjunction constraints on the
                    # parent Disjunct and process them again. This means the order in
                    # which we transformed Disjuncts would change the calculated M
                    # values. This is crazy, so we skip it.
                    raise GDP_Error(
                        "Found nested Disjunction '%s'. The multiple bigm "
                        "transformation does not support nested GDPs. "
                        "Please flatten the model before calling the "
                        "transformation" % t.name
                    )
                if not t.xor:
                    # This transformation assumes it can relax constraints assuming that
                    # another Disjunct is chosen. If it could be possible to choose both
                    # then that logic might fail.
                    raise GDP_Error(
                        "Cannot do multiple big-M reformulation for "
                        "Disjunction '%s' with OR constraint.  "
                        "Must be an XOR!" % t.name
                    )

                # start doing transformation
                disjunction_setup[t] = (trans_block, algebraic_constraint) = (
                    self._setup_transform_disjunctionData(t, gdp_tree.root_disjunct(t))
                )
                # Unlike python set(), ComponentSet keeps a stable
                # ordering, so we use it for the sake of determinism.
                active_disjuncts[t] = ComponentSet(
                    disj for disj in t.disjuncts if disj.active
                )
                # this method returns the constraints transformed on this disjunct;
                # update because we are saving these from all disjuncts for later
                if self._config.reduce_bound_constraints:
                    transformed_constraints.update(
                        self._transform_bound_constraints(
                            active_disjuncts[t], trans_block, arg_Ms
                        )
                    )
                # Get the jobs to calculate missing M values for this Disjunction. We
                # skip this if we are only doing bound constraints, in which case Ms was
                # already set.
                if not self._config.only_mbigm_bound_constraints:
                    self._setup_jobs_for_disjunction(
                        t,
                        active_disjuncts,
                        transformed_constraints,
                        arg_Ms,
                        Ms,
                        jobs,
                        trans_block,
                    )
        # (Now exiting the DisjunctionDatas loop)
        if jobs:
            jobs_by_name = [
                (
                    constraint.getname(fully_qualified=True),
                    other_disjunct.getname(fully_qualified=True),
                    unsuccessful_solve_msg,
                    is_upper,
                )
                for (
                    constraint,
                    other_disjunct,
                    unsuccessful_solve_msg,
                    is_upper,
                ) in jobs
            ]
            threads = (
                self._config.threads
                if self._config.threads is not None
                # It would be better to use len(os.sched_getaffinity(0)),
                # but it is not available on all platforms.
                else os.cpu_count() - 1
            )
            if threads > 1:
                with self._setup_pool(threads, instance, len(jobs)) as pool:
                    results = pool.starmap(func=_calc_M, iterable=jobs_by_name)
                    pool.close()
                    pool.join()
            else:
                _thread_local.model = instance
                _thread_local.solver = self._config.solver
                _thread_local.config_use_primal_bound = self._config.use_primal_bound
                logger.info(f"Running {len(jobs)} jobs single-threaded.")
                results = itertools.starmap(_calc_M, jobs_by_name)
            deactivated = set()
            for (constraint, other_disjunct, _, is_upper), (
                M,
                disjunct_infeasible,
            ) in zip(jobs, results):
                if disjunct_infeasible:
                    if other_disjunct not in deactivated:
                        # If we made it here without an exception, the solver is on the
                        # trusted solvers list
                        logger.debug(
                            "Disjunct '%s' is infeasible, deactivating."
                            % other_disjunct.name
                        )
                        other_disjunct.deactivate()
                        active_disjuncts[gdp_tree.parent(other_disjunct)].remove(
                            other_disjunct
                        )
                        deactivated.add(other_disjunct)
                # Note that we can't just transform immediately because we might be
                # waiting on the other one of upper_M or lower_M.
                if is_upper:
                    Ms[constraint, other_disjunct] = (
                        Ms[constraint, other_disjunct][0],
                        M,
                    )
                else:
                    Ms[constraint, other_disjunct] = (
                        M,
                        Ms[constraint, other_disjunct][1],
                    )
                trans_block._mbm_values[constraint, other_disjunct] = Ms[
                    constraint, other_disjunct
                ]

        for con in transformed_constraints:
            con.deactivate()

        # Iterate the Disjunctions again and actually transform them
        for disjunction, (
            trans_block,
            algebraic_constraint,
        ) in disjunction_setup.items():
            or_expr = 0
            for disjunct in active_disjuncts[disjunction]:
                or_expr += disjunct.indicator_var.get_associated_binary()
                self._transform_disjunct(
                    disjunct, trans_block, active_disjuncts[disjunction], Ms
                )
            algebraic_constraint.add(disjunction.index(), or_expr == 1)
            # map the DisjunctionData to its XOR constraint to mark it as
            # transformed
            disjunction._algebraic_constraint = weakref_ref(
                algebraic_constraint[disjunction.index()]
            )
            disjunction.deactivate()

    def _setup_jobs_for_disjunction(
        self,
        disjunction,
        active_disjuncts,
        transformed_constraints,
        arg_Ms,
        Ms,
        jobs,
        trans_block,
    ):
        """
        Do the inner work setting up or skipping M calculation jobs for a
        disjunction. Mutate the parameters Ms, jobs, trans_block._mbm_values,
        and self.used_args.

        Args:
            self: self.used_args map from (constraint, disjunct) to M tuple
                updated with the keys used from arg_Ms
            disjunction: disjunction to set up the jobs for
            active_disjuncts: map from disjunctions to ComponentSets of active
                disjuncts
            transformed_constraints: set of already transformed Constraints
            arg_Ms: user-provided map from (constraint, disjunct) to M value
                or tuple
            Ms: working map from (constraint, disjunct) to M tuples to update
            jobs: working list of (constraint, other_disjunct,
                unsuccessful_solve_msg, is_upper) job tuples to update
            trans_block: working transformation block. Update the
                trans_block._mbigm_values map from (constraint, disjunct) to
                M tuples
        """
        for disjunct, other_disjunct in itertools.product(
            active_disjuncts[disjunction], active_disjuncts[disjunction]
        ):
            if disjunct is other_disjunct:
                continue

            for constraint in disjunct.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                sort=SortComponents.deterministic,
            ):
                if constraint in transformed_constraints:
                    continue
                # First check args
                if (constraint, other_disjunct) in arg_Ms:
                    (lower_M, upper_M) = _convert_M_to_tuple(
                        arg_Ms[constraint, other_disjunct], constraint, other_disjunct
                    )
                    self.used_args[constraint, other_disjunct] = (lower_M, upper_M)
                else:
                    (lower_M, upper_M) = (None, None)
                unsuccessful_solve_msg = (
                    "Unsuccessful solve to calculate M value to "
                    "relax constraint '%s' on Disjunct '%s' when "
                    "Disjunct '%s' is selected."
                    % (constraint.name, disjunct.name, other_disjunct.name)
                )

                if constraint.lower is not None and lower_M is None:
                    jobs.append(
                        (constraint, other_disjunct, unsuccessful_solve_msg, False)
                    )
                if constraint.upper is not None and upper_M is None:
                    jobs.append(
                        (constraint, other_disjunct, unsuccessful_solve_msg, True)
                    )

                Ms[constraint, other_disjunct] = (lower_M, upper_M)
                trans_block._mbm_values[constraint, other_disjunct] = (lower_M, upper_M)

    def _transform_disjunct(self, obj, trans_block, active_disjuncts, Ms):
        # We've already filtered out deactivated disjuncts, so we know obj is
        # active.

        # Make a relaxation block if we haven't already.
        relaxation_block = self._get_disjunct_transformation_block(obj, trans_block)

        # Transform everything on the disjunct
        self._transform_block_components(obj, obj, active_disjuncts, Ms)

        # deactivate disjunct so writers can be happy
        obj._deactivate_without_fixing_indicator()

    def _transform_constraint(self, obj, disjunct, active_disjuncts, Ms):
        # we will put a new transformed constraint on the relaxation block.
        relaxation_block = disjunct._transformation_block()
        constraint_map = relaxation_block.private_data('pyomo.gdp')
        trans_block = relaxation_block.parent_block()

        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name
        name = unique_component_name(
            relaxation_block, obj.getname(fully_qualified=False)
        )

        newConstraint = Constraint(Any)
        relaxation_block.add_component(name, newConstraint)

        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue

            if not self._config.only_mbigm_bound_constraints:
                transformed = constraint_map.transformed_constraints[c]
                if c.lower is not None:
                    rhs = sum(
                        Ms[c, disj][0] * disj.indicator_var.get_associated_binary()
                        for disj in active_disjuncts
                        if disj is not disjunct
                    )
                    newConstraint.add((i, 'lb'), c.body - c.lower >= rhs)
                    transformed.append(newConstraint[i, 'lb'])

                if c.upper is not None:
                    rhs = sum(
                        Ms[c, disj][1] * disj.indicator_var.get_associated_binary()
                        for disj in active_disjuncts
                        if disj is not disjunct
                    )
                    newConstraint.add((i, 'ub'), c.body - c.upper <= rhs)
                    transformed.append(newConstraint[i, 'ub'])
                for c_new in transformed:
                    constraint_map.src_constraint[c_new] = [c]
            else:
                lower = (None, None, None)
                upper = (None, None, None)

                if disjunct not in self._arg_list:
                    self._arg_list[disjunct] = self._get_bigM_arg_list(
                        self._config.bigM, disjunct
                    )
                arg_list = self._arg_list[disjunct]

                # first, we see if an M value was specified in the arguments.
                # (This returns None if not)
                lower, upper = self._get_M_from_args(c, Ms, arg_list, lower, upper)
                M = (lower[0], upper[0])

                # estimate if we don't have what we need
                if c.lower is not None and M[0] is None:
                    M = (self._estimate_M(c.body, c)[0] - c.lower, M[1])
                    lower = (M[0], None, None)
                if c.upper is not None and M[1] is None:
                    M = (M[0], self._estimate_M(c.body, c)[1] - c.upper)
                    upper = (M[1], None, None)
                self._add_constraint_expressions(
                    c,
                    i,
                    M,
                    disjunct.indicator_var.get_associated_binary(),
                    newConstraint,
                    constraint_map,
                )

            # deactivate now that we have transformed
            c.deactivate()

    def _transform_bound_constraints(self, active_disjuncts, trans_block, Ms):
        # first we're just going to find all of them
        bounds_cons = ComponentMap()
        lower_bound_constraints_by_var = ComponentMap()
        upper_bound_constraints_by_var = ComponentMap()
        transformed_constraints = set()
        for disj in active_disjuncts:
            for c in disj.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                sort=SortComponents.deterministic,
            ):
                repn = generate_standard_repn(c.body)
                if repn.is_linear() and len(repn.linear_vars) == 1:
                    # We can treat this as a bounds constraint
                    v = repn.linear_vars[0]
                    if v not in bounds_cons:
                        bounds_cons[v] = [{}, {}]
                    M = [None, None]
                    if c.lower is not None:
                        M[0] = (c.lower - repn.constant) / repn.linear_coefs[0]
                        if disj in bounds_cons[v][0]:
                            # this is a redundant bound, we need to keep the
                            # better one
                            M[0] = max(M[0], bounds_cons[v][0][disj])
                        bounds_cons[v][0][disj] = M[0]
                        if v in lower_bound_constraints_by_var:
                            lower_bound_constraints_by_var[v].add((c, disj))
                        else:
                            lower_bound_constraints_by_var[v] = {(c, disj)}
                    if c.upper is not None:
                        M[1] = (c.upper - repn.constant) / repn.linear_coefs[0]
                        if disj in bounds_cons[v][1]:
                            # this is a redundant bound, we need to keep the
                            # better one
                            M[1] = min(M[1], bounds_cons[v][1][disj])
                        bounds_cons[v][1][disj] = M[1]
                        if v in upper_bound_constraints_by_var:
                            upper_bound_constraints_by_var[v].add((c, disj))
                        else:
                            upper_bound_constraints_by_var[v] = {(c, disj)}
                    # Add the M values to the dictionary
                    trans_block._mbm_values[c, disj] = M

                    # We can't deactivate yet because we will still be solving
                    # this Disjunct when we calculate M values for non-bounds
                    # constraints. We track that it is transformed instead by
                    # adding it to this set.
                    transformed_constraints.add(c)

        # Now we actually construct the constraints. We do this separately so
        # that we can make sure that we have a term for every active disjunct in
        # the disjunction (falling back on the variable bounds if they are there
        transformed = trans_block.transformed_bound_constraints
        offset = len(transformed)
        for i, (v, (lower_dict, upper_dict)) in enumerate(bounds_cons.items()):
            lower_rhs = 0
            upper_rhs = 0
            for disj in active_disjuncts:
                relaxation_block = self._get_disjunct_transformation_block(
                    disj, trans_block
                )
                constraint_map = relaxation_block.private_data('pyomo.gdp')
                if len(lower_dict) > 0:
                    M = lower_dict.get(disj, None)
                    if M is None:
                        # substitute the lower bound if it has one
                        M = v.lb
                    if M is None:
                        raise GDP_Error(
                            "There is no lower bound for variable '%s', and "
                            "Disjunct '%s' does not specify one in its "
                            "constraints. The transformation cannot construct "
                            "the special bound constraint relaxation without "
                            "one of these." % (v.name, disj.name)
                        )
                    lower_rhs += M * disj.indicator_var.get_associated_binary()
                if len(upper_dict) > 0:
                    M = upper_dict.get(disj, None)
                    if M is None:
                        # substitute the upper bound if it has one
                        M = v.ub
                    if M is None:
                        raise GDP_Error(
                            "There is no upper bound for variable '%s', and "
                            "Disjunct '%s' does not specify one in its "
                            "constraints. The transformation cannot construct "
                            "the special bound constraint relaxation without "
                            "one of these." % (v.name, disj.name)
                        )
                    upper_rhs += M * disj.indicator_var.get_associated_binary()
            idx = i + offset
            if len(lower_dict) > 0:
                transformed.add((idx, 'lb'), v >= lower_rhs)
                constraint_map.src_constraint[transformed[idx, 'lb']] = []
                for c, disj in lower_bound_constraints_by_var[v]:
                    constraint_map.src_constraint[transformed[idx, 'lb']].append(c)
                    disj.transformation_block.private_data(
                        'pyomo.gdp'
                    ).transformed_constraints[c].append(transformed[idx, 'lb'])
            if len(upper_dict) > 0:
                transformed.add((idx, 'ub'), v <= upper_rhs)
                constraint_map.src_constraint[transformed[idx, 'ub']] = []
                for c, disj in upper_bound_constraints_by_var[v]:
                    constraint_map.src_constraint[transformed[idx, 'ub']].append(c)
                    # might already be here if it had an upper bound
                    disj_constraint_map = disj.transformation_block.private_data(
                        'pyomo.gdp'
                    )
                    disj_constraint_map.transformed_constraints[c].append(
                        transformed[idx, 'ub']
                    )

        return transformed_constraints

    def _setup_pool(self, threads, instance, num_jobs):
        method = (
            self._config.process_start_method
            if self._config.process_start_method is not None
            else (
                ProcessStartMethod.spawn
                if os.name == 'nt'
                else (
                    ProcessStartMethod.forkserver
                    if len(threading.enumerate()) > 1
                    else ProcessStartMethod.fork
                )
            )
        )
        logger.info(
            f"Running {num_jobs} jobs on {threads} worker "
            f"processes with process start method {method}."
        )
        if (
            method == ProcessStartMethod.spawn
            or method == ProcessStartMethod.forkserver
        ):
            if not dill_available:
                raise GDP_Error(
                    "Dill is required when spawning processes using "
                    "methods 'spawn' or 'forkserver', but it could "
                    "not be imported."
                )
            pool = multiprocessing.get_context(method.value).Pool(
                processes=threads,
                initializer=_setup_spawn,
                initargs=(
                    dill.dumps(instance),
                    dill.dumps(self._config.solver.__class__),
                    dill.dumps(self._config.solver.options),
                    self._config.use_primal_bound,
                ),
            )
        elif method == ProcessStartMethod.fork:
            _thread_local.model = instance
            _thread_local.solver = self._config.solver
            _thread_local.config_use_primal_bound = self._config.use_primal_bound
            pool = multiprocessing.get_context('fork').Pool(
                processes=threads, initializer=_setup_fork, initargs=()
            )
        return pool

    def _add_transformation_block(self, to_block):
        trans_block, new_block = super()._add_transformation_block(to_block)

        if new_block:
            # Will store M values as we transform
            trans_block._mbm_values = {}
            trans_block.transformed_bound_constraints = Constraint(
                NonNegativeIntegers, ['lb', 'ub']
            )
        return trans_block, new_block

    def _warn_for_active_suffix(self, suffix, disjunct, active_disjuncts, Ms):
        if suffix.local_name == 'BigM':
            logger.debug(
                "Found active 'BigM' Suffix on '{0}'. "
                "The multiple bigM transformation does not currently "
                "support specifying M's with Suffixes and is ignoring "
                "this Suffix.".format(disjunct.name)
            )
        elif suffix.local_name == 'LocalVars':
            # This is fine, but this transformation doesn't need anything from it
            pass
        else:
            raise GDP_Error(
                "Found active Suffix '{0}' on Disjunct '{1}'. "
                "The multiple bigM transformation does not "
                "support this Suffix.".format(suffix.name, disjunct.name)
            )

    # These are all functions to retrieve transformed components from
    # original ones and vice versa.

    def get_src_constraints(self, transformedConstraint):
        """Return the original Constraints whose transformed counterpart is
        transformedConstraint

        Parameters
        ----------
        transformedConstraint: Constraint, which must be a component on one of
        the BlockDatas in the relaxedDisjuncts Block of
        a transformation block
        """
        # This is silly, but we rename this function for multiple bigm because
        # transformed constraints have multiple source constraints.
        return super().get_src_constraint(transformedConstraint)

    def get_all_M_values(self, model):
        """Returns a dictionary mapping each constraint, disjunct pair (where
        the constraint is on a disjunct and the disjunct is in the same
        disjunction as that disjunct) to a tuple: (lower_M_value,
        upper_M_value), where either can be None if the constraint does not
        have a lower or upper bound (respectively).

        Parameters
        ----------
        model: A GDP model that has been transformed with multiple-BigM
        """
        all_ms = {}
        for disjunction in model.component_data_objects(
            Disjunction,
            active=None,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        ):
            if disjunction.algebraic_constraint is not None:
                trans_block = disjunction.algebraic_constraint.parent_block()
                # Don't necessarily assume all disjunctions were transformed
                # with multiple bigm...
                if hasattr(trans_block, "_mbm_values"):
                    all_ms.update(trans_block._mbm_values)

        return all_ms


# Things we call in subprocesses. These can't be member functions, or
# else we'd have to pickle `self`, which is problematic.
def _setup_spawn(model, solver_class, solver_options, use_primal_bound):
    # When using 'spawn' or 'forkserver', Python starts in a new
    # environment and executes only this file, so we need to manually
    # ensure necessary plugins are registered (even if the main process
    # has already registered them).
    import pyomo.environ

    _thread_local.model = dill.loads(model)
    _thread_local.solver = dill.loads(solver_class)(options=dill.loads(solver_options))
    _thread_local.config_use_primal_bound = use_primal_bound


def _setup_fork():
    # model and config_use_primal_bound were already properly set, but
    # remake the solver instead of using the passed argument. All these
    # processes are copies of the calling thread so the thread-local
    # still works.
    _thread_local.solver = _thread_local.solver.__class__(
        options=_thread_local.solver.options
    )


def _calc_M(constraint_name, other_disjunct_name, unsuccessful_message, is_upper):
    solver = _thread_local.solver
    model = _thread_local.model
    scratch = _get_scratch_block(
        model.find_component(constraint_name),
        model.find_component(other_disjunct_name),
        is_upper,
    )
    results = solver.solve(scratch, tee=False, load_solutions=False, keepfiles=False)
    termination_condition = results.solver.termination_condition
    if is_upper:
        incumbent = results.problem.lower_bound
        bound = results.problem.upper_bound
    else:
        incumbent = results.problem.upper_bound
        bound = results.problem.lower_bound
    if termination_condition is TerminationCondition.infeasible:
        # [2/18/24]: TODO: After the solver rewrite is complete, we will not
        # need this check since we can actually determine from the
        # termination condition whether or not the solver proved
        # infeasibility or just terminated at local infeasiblity. For now,
        # while this is not complete, it catches most of the solvers we
        # trust, and, unless someone is so pathological as to *rename* an
        # untrusted solver using a trusted solver name, it will never do the
        # *wrong* thing.
        if any(s in solver.name for s in _trusted_solvers):
            return (0, True)
        else:
            # This is a solver that might report
            # 'infeasible' for local infeasibility, so we
            # can't deactivate with confidence. To be
            # conservative, we'll just complain about
            # it. Post-solver-rewrite we will want to change
            # this so that we check for 'proven_infeasible'
            # and then we can abandon this hack
            raise GDP_Error(unsuccessful_message)
    elif termination_condition is not TerminationCondition.optimal:
        raise GDP_Error(unsuccessful_message)
    else:
        # NOTE: This transformation can be made faster by allowing the
        # solver a gap. As long as we have a bound, it's still valid
        # (though not as tight).
        #
        # We should use the dual bound here. The primal bound is
        # mathematically incorrect in the presence of numerical error,
        # but it's the best a local solver like ipopt can do, so we
        # allow it to be used by setting an option.
        if not _thread_local.config_use_primal_bound:
            M = bound
            if not math.isfinite(M):
                raise GDP_Error(
                    "Subproblem solved to optimality, but no finite dual bound was "
                    "obtained. If you would like to instead use the best obtained "
                    "solution, set the option use_primal_bound=True. This is "
                    "necessary when using a local solver such as ipopt, but be "
                    "aware that interior feasible points for this subproblem do "
                    "not give valid values for M in general."
                )
        else:
            M = incumbent
            if not math.isfinite(M):
                # Solved to optimality, but we did not find an incumbent
                # objective value. Try again by actually loading the
                # solution and evaluating the objective expression.
                try:
                    scratch.solutions.load_from(results)
                    M = value(scratch.obj)
                    if not math.isfinite(M):
                        raise ValueError()
                except Exception:
                    raise GDP_Error(
                        "`use_primal_bound` is enabled, but could not find a finite "
                        "objective value after optimal solve."
                    )
        return (M, False)


def _get_scratch_block(constraint, other_disjunct, is_upper):
    scratch = ConcreteModel()
    if is_upper:
        scratch.obj = Objective(expr=constraint.body - constraint.upper, sense=maximize)
    else:
        scratch.obj = Objective(expr=constraint.body - constraint.lower, sense=minimize)
    # Create a Block component (via Reference) that actually points to a
    # DisjunctData, as we want the writer to write it as if it were an
    # ordinary Block rather than getting any special
    # handling. DisjunctData inherits BlockData, so this should be
    # valid.
    scratch.disjunct_ref = Reference(other_disjunct, ctype=Block)

    # Add references to every Var that appears in an active constraint.
    # TODO: If the writers don't assume Vars are declared on the Block
    # being solved, we won't need this!
    seen = set()
    for constraint in scratch.component_data_objects(
        Constraint, active=True, sort=SortComponents.deterministic, descend_into=Block
    ):
        for var in EXPR.identify_variables(constraint.expr, include_fixed=True):
            if id(var) not in seen:
                seen.add(id(var))
                ref = Reference(var)
                scratch.add_component(unique_component_name(scratch, var.name), ref)
    return scratch
