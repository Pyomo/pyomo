#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Cutting plane-based GDP reformulation.

Implements a general cutting plane-based reformulation for linear and
convex GDPs.
"""
from __future__ import division

from pyomo.common.config import (ConfigBlock, ConfigValue, PositiveFloat,
                                 NonNegativeFloat, PositiveInt, In)
from pyomo.common.modeling import unique_component_name
from pyomo.core import ( Any, Block, Constraint, Objective, Param, Var,
                         SortComponents, Transformation, TransformationFactory,
                         value, NonNegativeIntegers, Reals, NonNegativeReals,
                         Suffix, ComponentMap )
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import ( verify_successful_solve, NORMAL,
                             clone_without_expression_components )

from pyomo.contrib.fme.fourier_motzkin_elimination import \
    Fourier_Motzkin_Elimination_Transformation

from six import iteritems

import logging

logger = logging.getLogger('pyomo.gdp.cuttingplane')

NAME_BUFFER = {}

def do_not_tighten(m):
    return m

def _get_constraint_exprs(constraints, hull_to_bigm_map):
    """Returns a list of expressions which are constrain.expr translated 
    into the bigm space, for each constraint in constraints.
    """
    cuts = []
    for cons in constraints:
        cuts.append(clone_without_expression_components( 
            cons.expr, substitute=hull_to_bigm_map))
    return cuts

def _constraint_tight(model, constraint, TOL):
    """
    Returns a list [a,b] where a is -1 if the lower bound is tight or
    slightly violated, b is 1 if the upper bound is tight of slightly
    violated, and [a,b]=[-1,1] if we have an exactly satisfied (or
    slightly violated) equality.
    """
    val = value(constraint.body)
    ans = [0, 0]
    if constraint.lower is not None:
        if val - value(constraint.lower) <= TOL:
            # tight or in violation of LB
            ans[0] -= 1

    if constraint.upper is not None:
        if value(constraint.upper) - val <= TOL:
            # tight or in violation of UB
            ans[1] += 1

    return ans

def _get_linear_approximation_expr(normal_vec, point):
    """Returns constraint linearly approximating constraint normal to normal_vec
    at point"""
    body = 0
    for coef, v in zip(point, normal_vec):
        body -= coef*v
    return body >= -sum(normal_vec[idx]*v.value for (idx, v) in
                       enumerate(point))

def _precompute_potentially_useful_constraints(transBlock_rHull,
                                               disaggregated_vars):
    instance_rHull = transBlock_rHull.model()
    constraints = transBlock_rHull.constraints_for_FME = []
    for constraint in instance_rHull.component_data_objects(
            Constraint,
            active=True,
            descend_into=Block,
            sort=SortComponents.deterministic):
        # we don't care about anything that does not involve at least one
        # disaggregated variable.
        repn = generate_standard_repn(constraint.body)
        for v in repn.linear_vars + repn.quadratic_vars + repn.nonlinear_vars:
            # ESJ: This is why disaggregated_vars is a ComponentSet
            if v in disaggregated_vars:
                constraints.append(constraint)
                break

def create_cuts_fme(transBlock_rHull, var_info, hull_to_bigm_map,
                    rBigM_linear_constraints, rHull_vars, disaggregated_vars,
                    norm, cut_threshold, zero_tolerance, integer_arithmetic,
                    constraint_tolerance):
    """Returns a cut which removes x* from the relaxed bigm feasible region.

    Finds all the constraints which are tight at xhat (assumed to be the 
    solution currently in instance_rHull), and calculates a composite normal
    vector by summing the vectors normal to each of these constraints. Then
    Fourier-Motzkin elimination is used to project the disaggregated variables
    out of the polyhedron formed by the composite normal and the collection 
    of tight constraints. This results in multiple cuts, of which we select
    one that cuts of x* by the greatest margin, as long as that margin is
    more than cut_threshold. If no cut satisfies the margin specified by 
    cut_threshold, we return None.

    Parameters
    -----------
    transBlock_rHull: transformation blcok on relaxed hull instance
    var_info: List of tuples (rBigM_var, rHull_var, xstar_param)
    hull_to_bigm_map: For expression substition, maps id(hull_var) to 
                      coresponding bigm var
    rBigM_linear_constraints: list of linear constraints in relaxed bigM
    rHull_vars: list of all variables in relaxed hull
    disaggregated_vars: ComponentSet of disaggregated variables in hull 
                        reformulation
    norm: norm used in the separation problem
    cut_threshold: Amount x* needs to be infeasible in generated cut in order
                   to consider the cut for addition to the bigM model.
    zero_tolerance: Tolerance at which a float will be treated as 0 during
                    Fourier-Motzkin elimination
    integer_arithmetic: boolean, whether or not to require Fourier-Motzkin
                        Elimination does integer arithmetic. Only possible 
                        when all data is integer.
    constraint_tolerance: Tolerance at which we will consider a constraint 
                          tight.
    """
    instance_rHull = transBlock_rHull.model()
    # In the first iteration, we will compute a list of constraints that could
    # ever be interesting: Everything that involves at least one disaggregated
    # variable.
    if transBlock_rHull.component("constraints_for_FME") is None:
        _precompute_potentially_useful_constraints( transBlock_rHull,
                                                    disaggregated_vars)

    tight_constraints = Block()
    conslist = tight_constraints.constraints = Constraint(
        NonNegativeIntegers)
    conslist.construct()
    something_interesting = False
    for constraint in transBlock_rHull.constraints_for_FME:
        multipliers = _constraint_tight(instance_rHull, constraint,
                                        constraint_tolerance)
        for multiplier in multipliers:
            if multiplier:
                something_interesting = True
                f = constraint.body
                firstDerivs = differentiate(f, wrt_list=rHull_vars)
                normal_vec = [multiplier*value(_) for _ in firstDerivs]
                # check if constraint is linear
                if f.polynomial_degree() == 1:
                    conslist[len(conslist)] = constraint.expr
                else: 
                    # we will use the linear approximation of this constraint at
                    # x_hat
                    conslist[len(conslist)] = _get_linear_approximation_expr(
                        normal_vec, rHull_vars)

    # NOTE: we now have all the tight Constraints (in the pyomo sense of the
    # word "Constraint"), but we are missing some variable bounds. The ones for
    # the disaggregated variables will be added by FME

    # It is possible that the separation problem returned a point in the
    # interior of the convex hull. It is also possible that the only active
    # constraints do not involve the disaggregated variables. In these
    # situations, there are not constraints from which to create a valid cut.
    if not something_interesting:
        return None

    tight_constraints.construct()
    logger.info("Calling FME transformation on %s constraints to eliminate"
                " %s variables" % (len(tight_constraints.constraints),
                                   len(disaggregated_vars)))
    TransformationFactory('contrib.fourier_motzkin_elimination').\
        apply_to(tight_constraints, vars_to_eliminate=disaggregated_vars,
                 zero_tolerance=zero_tolerance,
                 do_integer_arithmetic=integer_arithmetic,
                 projected_constraints_name="fme_constraints")
    fme_results = tight_constraints.fme_constraints
    projected_constraints = [cons for i, cons in iteritems(fme_results)]

    # we created these constraints with the variables from rHull. We
    # actually need constraints for BigM and rBigM now!
    cuts = _get_constraint_exprs(projected_constraints, hull_to_bigm_map)

    # We likely have some cuts that duplicate other constraints now. We will
    # filter them to make sure that they do in fact cut off x*. If that's the
    # case, we know they are not already in the BigM relaxation. Because they
    # came from FME, they are very likely redundant, so we'll keep the best one
    # we find
    best = 0
    best_cut = None
    cuts_to_keep = []
    for i, cut in enumerate(cuts):
        # x* is still in rBigM, so we can just remove this constraint if it
        # is satisfied at x*
        logger.info("FME: Post-processing cut %s" % cut)
        if value(cut):
            logger.info("FME:\t Doesn't cut off x*")
            continue
        # we have found a constraint which cuts of x* by some convincing amount
        # and is not already in rBigM. 
        cuts_to_keep.append(i)
        # We know cut is lb <= expr and that it's violated
        assert len(cut.args) == 2
        cut_off = value(cut.args[0]) - value(cut.args[1])
        if cut_off > cut_threshold and cut_off > best:
            best = cut_off
            best_cut = cut
            logger.info("FME:\t New best cut: Cuts off x* by %s." % best)

    # NOTE: this is not used right now, but it's not hard to imagine a world in
    # which we would want to keep multiple cuts from FME, so leaving it in for
    # now.
    cuts = [cuts[i] for i in cuts_to_keep]

    if best_cut is not None:
        return [best_cut]

    return None

def create_cuts_normal_vector(transBlock_rHull, var_info, hull_to_bigm_map,
                              rBigM_linear_constraints, rHull_vars,
                              disaggregated_vars, norm, cut_threshold,
                              zero_tolerance, integer_arithmetic,
                              constraint_tolerance):
    """Returns a cut which removes x* from the relaxed bigm feasible region.

    Ignores all parameters except var_info and cut_threshold, and constructs 
    a cut at x_hat, the projection of the relaxed bigM solution x* onto the hull,
    which is perpendicular to the vector from x* to x_hat.

    Note that this method will often lead to numerical difficulties since both
    x* and x_hat are solutions to optimization problems. To mitigate this,
    use some method of backing off the cut to make it a bit more conservative.

    Parameters
    -----------
    transBlock_rHull: transformation blcok on relaxed hull instance. Ignored by
                      this callback.
    var_info: List of tuples (rBigM_var, rHull_var, xstar_param)
    hull_to_bigm_map: For expression substition, maps id(hull_var) to 
                      coresponding bigm var. Ignored by this callback
    rBigM_linear_constraints: list of linear constraints in relaxed bigM.
                              Ignored by this callback.
    rHull_vars: list of all variables in relaxed hull. Ignored by this callback.
    disaggregated_vars: ComponentSet of disaggregated variables in hull 
                        reformulation. Ignored by this callback
    norm: The norm used in the separation problem, will be used to calculate
          the subgradient used to generate the cut
    cut_threshold: Amount x* needs to be infeasible in generated cut in order
                   to consider the cut for addition to the bigM model.
    zero_tolerance: Tolerance at which a float will be treated as 0 during
                    Fourier-Motzkin elimination. Ignored by this callback
    integer_arithmetic: Ignored by this callback (specifies FME use integer
                        arithmetic)
    constraint_tolerance: Ignored by this callback (specifies when constraints
                          are considered tight in FME)
    """
    cutexpr = 0
    if norm == 2:
        for x_rbigm, x_hull, x_star in var_info:
            cutexpr += (x_hull.value - x_star.value)*(x_rbigm - x_hull.value)
    elif norm == float('inf'):
        duals = transBlock_rHull.model().dual
        if len(duals) == 0:
            raise GDP_Error("No dual information in the separation problem! "
                            "To use the infinity norm and the "
                            "create_cuts_normal_vector method, you must use "
                            "a solver which provides dual information.")
        i = 0
        for x_rbigm, x_hull, x_star in var_info:
            # ESJ: We wrote this so duals will be nonnegative
            mu_plus = value(duals[transBlock_rHull.inf_norm_linearization[i]])
            mu_minus = value(duals[transBlock_rHull.inf_norm_linearization[i+1]])
            assert mu_plus >= 0
            assert mu_minus >= 0
            cutexpr += (mu_plus - mu_minus)*(x_rbigm - x_hull.value)
            i += 2

    # make sure we're cutting off x* by enough.
    if value(cutexpr) < -cut_threshold:
        return [cutexpr >= 0]
    logger.warning("Generated cut did not remove relaxed BigM solution by more "
                   "than the specified threshold of %s. Stopping cut "
                   "generation." % cut_threshold)
    return None

def back_off_constraint_with_calculated_cut_violation(cut, transBlock_rHull,
                                                      bigm_to_hull_map, opt,
                                                      stream_solver, TOL):
    """Calculates the maximum violation of cut subject to the relaxed hull
    constraints. Increases this violation by TOL (to account for optimality 
    tolerance in solving the problem), and, if it finds that cut can be violated
    up to this tolerance, makes it more conservative such that it no longer can.

    Parameters
    ----------
    cut: The cut to be made more conservative, a Constraint
    transBlock_rHull: the relaxed hull model's transformation Block
    bigm_to_hull_map: Dictionary mapping ids of bigM variables to the 
                      corresponding variables on the relaxed hull instance
    opt: SolverFactory object for solving the maximum violation problem
    stream_solver: Whether or not to set tee=True while solving the maximum
                   violation problem.
    TOL: An absolute tolerance to be added to the calculated cut violation,
         to account for optimality tolerance in the maximum violation problem
         solve.
    """
    instance_rHull = transBlock_rHull.model()
    logger.info("Post-processing cut: %s" % cut.expr)
    # Take a constraint. We will solve a problem maximizing its violation
    # subject to rHull. We will add some user-specified tolerance to that
    # violation, and then add that much padding to it if it can be violated.
    transBlock_rHull.separation_objective.deactivate()

    transBlock_rHull.infeasibility_objective = Objective(
        expr=clone_without_expression_components(cut.body,
                                                 substitute=bigm_to_hull_map))

    results = opt.solve(instance_rHull, tee=stream_solver, load_solutions=False)
    if verify_successful_solve(results) is not NORMAL:
        logger.warning("Problem to determine how much to "
                       "back off the new cut "
                       "did not solve normally. Leaving the constraint as is, "
                       "which could lead to numerical trouble%s" % (results,))
        # restore the objective
        transBlock_rHull.del_component(transBlock_rHull.infeasibility_objective)
        transBlock_rHull.separation_objective.activate()
        return
    instance_rHull.solutions.load_from(results)

    # we're minimizing, val is <= 0
    val = value(transBlock_rHull.infeasibility_objective) - TOL
    if val <= 0:
        logger.info("\tBacking off cut by %s" % val)
        cut._body += abs(val)
    # else there is nothing to do: restore the objective
    transBlock_rHull.del_component(transBlock_rHull.infeasibility_objective)
    transBlock_rHull.separation_objective.activate()

def back_off_constraint_by_fixed_tolerance(cut, transBlock_rHull,
                                           bigm_to_hull_map, opt, stream_solver,
                                           TOL):
    """Makes cut more conservative by absolute tolerance TOL

    Parameters
    ----------
    cut: the cut to be made more conservative, a Constraint
    transBlock_rHull: the relaxed hull model's transformation Block. Ignored by
                      this callback
    bigm_to_hull_map: Dictionary mapping ids of bigM variables to the 
                      corresponding variables on the relaxed hull instance.
                      Ignored by this callback.
    opt: SolverFactory object. Ignored by this callback
    stream_solver: Whether or not to set tee=True while solving. Ignored by
                   this callback
    TOL: An absolute tolerance to be added to make cut more conservative.
    """
    cut._body += TOL

@TransformationFactory.register('gdp.cuttingplane',
                                doc="Relaxes a linear disjunctive model by "
                                "adding cuts from convex hull to Big-M "
                                "reformulation.")
class CuttingPlane_Transformation(Transformation):
    """Relax convex disjunctive model by forming the bigm relaxation and then
    iteratively adding cuts from the hull relaxation (or the hull relaxation
    after some basic steps) in order to strengthen the formulation.

    Note that gdp.cuttingplane is not a structural transformation: If variables
    on the model are fixed, they will be treated as data, and unfixing them
    after transformation will very likely result in an invalid model.

    This transformation accepts the following keyword arguments:
    
    Parameters
    ----------
    solver : Solver name (as string) to use to solve relaxed BigM and separation
             problems
    solver_options : dictionary of options to pass to the solver
    stream_solver : Whether or not to display solver output
    verbose : Enable verbose output from cuttingplanes algorithm
    cuts_name : Optional name for the IndexedConstraint containing the projected
                cuts (must be a unique name with respect to the instance)
    minimum_improvement_threshold : Stopping criterion based on improvement in
                                    Big-M relaxation. This is the minimum 
                                    difference in relaxed BigM objective
                                    values between consecutive iterations
    separation_objective_threshold : Stopping criterion based on separation 
                                     objective. If separation objective is not 
                                     at least this large, cut generation will 
                                     terminate.
    cut_filtering_threshold : Stopping criterion based on effectiveness of the 
                              generated cut: This is the amount by which 
                              a cut must be violated at the relaxed bigM 
                              solution in order to be added to the bigM model
    max_number_of_cuts : The maximum number of cuts to add to the big-M model
    norm : norm to use in the objective of the separation problem
    tighten_relaxation : callback to modify the GDP model before the hull 
                         relaxation is taken (e.g. could be used to perform 
                         basic steps)
    create_cuts : callback to create cuts using the solved relaxed bigM and hull
                  problems
    post_process_cut : callback to perform post-processing on created cuts
    back_off_problem_tolerance : tolerance to use while post-processing
    zero_tolerance : Tolerance at which a float will be considered 0 when
                     using Fourier-Motzkin elimination to create cuts.
    do_integer_arithmetic : Whether or not to require Fourier-Motzkin elimination
                            to do integer arithmetic. Only possible when all
                            data is integer.
    tight_constraint_tolerance : Tolerance at which a constraint is considered
                                 tight for the Fourier-Motzkin cut generation
                                 procedure

    By default, the callbacks will be set such that the algorithm performed is
    as presented in [1], but with an additional post-processing procedure to
    reduce numerical error, which calculates the maximum violation of the cut 
    subject to the relaxed hull constraints, and then pads the constraint by 
    this violation plus an additional user-specified tolerance.

    In addition, the create_cuts_fme function provides an (exponential time)
    method of generating cuts which reduces numerical error (and can eliminate 
    it if all data is integer). It collects the hull constraints which are 
    tight at the solution of the separation problem. It creates a cut in the 
    extended space perpendicular to  a composite normal vector created by 
    summing the directions normal to these constraints. It then performs 
    fourier-motzkin elimination on the collection of constraints and the cut
    to project out the disaggregated variables. The resulting constraint which
    is most violated by the relaxed bigM solution is then returned.

    References
    ----------
        [1] Sawaya, N. W., Grossmann, I. E. (2005). A cutting plane method for 
        solving linear generalized disjunctive programming problems. Computers
        and Chemical Engineering, 29, 1891-1913 
    """

    CONFIG = ConfigBlock("gdp.cuttingplane")
    CONFIG.declare('solver', ConfigValue(
        default='ipopt',
        domain=str,
        description="""Solver to use for relaxed BigM problem and the separation
        problem""",
        doc="""
        This specifies the solver which will be used to solve LP relaxation
        of the BigM problem and the separation problem. Note that this solver
        must be able to handle a quadratic objective because of the separation
        problem.
        """
    ))
    CONFIG.declare('minimum_improvement_threshold', ConfigValue(
        default=0.01,
        domain=NonNegativeFloat,
        description="Threshold value for difference in relaxed bigM problem "
        "objectives used to decide when to stop adding cuts",
        doc="""
        If the difference between the objectives in two consecutive iterations is
        less than this value, the algorithm terminates without adding the cut
        generated in the last iteration.  
        """
    ))
    CONFIG.declare('separation_objective_threshold', ConfigValue(
        default=0.01,
        domain=NonNegativeFloat,
        description="Threshold value used to decide when to stop adding cuts: "
        "If separation problem objective is not at least this quantity, cut "
        "generation will terminate.",
        doc="""
        If the separation problem objective (distance between relaxed bigM 
        solution and its projection onto the relaxed hull feasible region)
        does not exceed this threshold, the algorithm will terminate.
        """
    ))
    CONFIG.declare('max_number_of_cuts', ConfigValue(
        default=100,
        domain=PositiveInt,
        description="The maximum number of cuts to add before the algorithm "
        "terminates.",
        doc="""
        If the algorithm does not terminate due to another criterion first,
        cut generation will stop after adding this many cuts.
        """
    ))
    CONFIG.declare('norm', ConfigValue(
        default=2,
        domain=In([2, float('inf')]),
        description="Norm to use in the separation problem: 2, or "
        "float('inf')",
        doc="""
        Norm used to calculate distance in the objective of the separation 
        problem which finds the nearest point on the hull relaxation region
        to the current solution of the relaxed bigm problem.

        Supported norms are the Euclidean norm (specify 2) and the infinity 
        norm (specify float('inf')). Note that the first makes the separation 
        problem objective quadratic and the latter makes it linear.
        """
    ))
    CONFIG.declare('verbose', ConfigValue(
        default=False,
        domain=bool,
        description="Flag to enable verbose output",
        doc="""
        If True, prints subproblem solutions, as well as potential and added cuts
        during algorithm.

        If False, only the relaxed BigM objective and minimal information about 
        cuts is logged.
        """
    ))
    CONFIG.declare('stream_solver', ConfigValue(
        default=False,
        domain=bool,
        description="""If true, sets tee=True for every solve performed over
        "the course of the algorithm"""
    ))
    CONFIG.declare('solver_options', ConfigBlock(
        implicit=True,
        description="Dictionary of solver options",
        doc="""
        Dictionary of solver options that will be set for the solver for both the
        relaxed BigM and separation problem solves.
        """
    ))
    CONFIG.declare('tighten_relaxation', ConfigValue(
        default=do_not_tighten,
        description="Callback which takes the GDP formulation and returns a "
        "GDP formulation with a tighter hull relaxation",
        doc="""
        Function which accepts the GDP formulation of the problem and returns
        a GDP formulation which the transformation will then take the hull
        reformulation of.

        Most typically, this callback would be used to apply basic steps before
        taking the hull reformulation, but anything which tightens the GDP can 
        be performed here.
        """
    ))
    CONFIG.declare('create_cuts', ConfigValue(
        default=create_cuts_normal_vector,
        description="Callback which generates a list of cuts, given the solved "
        "relaxed bigM and relaxed hull solutions. If no cuts can be "
        "generated, returns None",
        doc="""
        Callback to generate cuts to be added to the bigM problem based on 
        solutions to the relaxed bigM problem and the separation problem.

        Arguments
        ---------
        transBlock_rBigm: transformation block on relaxed bigM instance
        transBlock_rHull: transformation blcok on relaxed hull instance
        var_info: List of tuples (rBigM_var, rHull_var, xstar_param)
        hull_to_bigm_map: For expression substition, maps id(hull_var) to 
                          coresponding bigm var
        rBigM_linear_constraints: list of linear constraints in relaxed bigM
        rHull_vars: list of all variables in relaxed hull
        disaggregated_vars: ComponentSet of disaggregated variables in hull 
                            reformulation
        cut_threshold: Amount x* needs to be infeasible in generated cut in order
                       to consider the cut for addition to the bigM model.
        zero_tolerance: Tolerance at which a float will be treated as 0

        Returns
        -------
        list of cuts to be added to bigM problem (and relaxed bigM problem),
        represented as expressions using variables from the bigM model
        """
    ))
    CONFIG.declare('post_process_cut', ConfigValue(
        default=back_off_constraint_with_calculated_cut_violation,
        description="Callback which takes a generated cut and post processes "
        "it, presumably to back it off in the case of numerical error. Set to "
        "None if not post-processing is desired.",
        doc="""
        Callback to adjust a cut returned from create_cuts before adding it to
        the model, presumably to make it more conservative in case of numerical
        error.

        Arguments
        ---------
        cut: the cut to be made more conservative, a Constraint
        transBlock_rHull: the relaxed hull model's transformation Block.
        bigm_to_hull_map: Dictionary mapping ids of bigM variables to the 
                          corresponding variables on the relaxed hull instance.
        opt: SolverFactory object for subproblem solves in this procedure
        stream_solver: Whether or not to set tee=True while solving.
        TOL: A tolerance

        Returns
        -------
        None, modifies the cut in place
        """
    ))
    # back off problem tolerance (on top of the solver's (sometimes))
    CONFIG.declare('back_off_problem_tolerance', ConfigValue(
        default=1e-8,
        domain=NonNegativeFloat,
        description="Tolerance to pass to the post_process_cut callback.",
        doc="""
        Tolerance passed to the post_process_cut callback.

        Depending on the callback, different values could make sense, but 
        something on the order of the solver's optimality or constraint 
        tolerances is appropriate.
        """
    ))
    CONFIG.declare('cut_filtering_threshold', ConfigValue(
        default=0.001,
        domain=NonNegativeFloat,
        description="Tolerance used to decide if a cut removes x* from the "
        "relaxed BigM problem by enough to be added to the bigM problem.",
        doc="""
        Absolute tolerance used to decide whether to keep a cut. We require
        that, when evaluated at x* (the relaxed BigM optimal solution), the 
        cut be infeasible by at least this tolerance.
        """
    ))
    CONFIG.declare('zero_tolerance', ConfigValue(
        default=1e-9,
        domain=NonNegativeFloat,
        description="Tolerance at which floats are assumed to be 0 while "
        "performing Fourier-Motzkin elimination",
        doc="""
        Only relevant when create_cuts=create_cuts_fme, this sets the 
        zero_tolerance option for the Fourier-Motzkin elimination transformation.
        """
    ))
    CONFIG.declare('do_integer_arithmetic', ConfigValue(
        default=False,
        domain=bool,
        description="Only relevant if using Fourier-Motzkin Elimination (FME) "
        "and if all problem data is integer, requires FME transformation to "
        "perform integer arithmetic to eliminate numerical error.",
        doc="""
        Only relevant when create_cuts=create_cuts_fme and if all problem data 
        is integer, this sets the do_integer_arithmetic flag to true for the 
        FME transformation, meaning that the projection to the big-M space 
        can be done with exact precision.
        """
    ))
    CONFIG.declare('cuts_name', ConfigValue(
        default=None,
        domain=str,
        description="Optional name for the IndexedConstraint containing the "
        "projected cuts. Must be a unique name with respect to the "
        "instance.",
        doc="""
        Optional name for the IndexedConstraint containing the projected 
        constraints. If not specified, the cuts will be stored on a 
        private block created by the transformation, so if you want access 
        to them after the transformation, use this argument.

        Must be a string which is a unique component name with respect to the 
        Block on which the transformation is called.
        """
    ))
    CONFIG.declare('tight_constraint_tolerance', ConfigValue(
        default=1e-6, # Gurobi constraint tolerance
        domain=NonNegativeFloat,
        description="Tolerance at which a constraint is considered tight for "
        "the Fourier-Motzkin cut generation procedure.",
        doc="""
        For a constraint a^Tx <= b, the Fourier-Motzkin cut generation procedure
        will consider the constraint tight (and add it to the set of constraints
        being projected) when a^Tx - b is less than this tolerance. 

        It is recommended to set this tolerance to the constraint tolerance of
        the solver being used.
        """
    ))
    def __init__(self):
        super(CuttingPlane_Transformation, self).__init__()

    def _apply_to(self, instance, bigM=None, **kwds):
        original_log_level = logger.level
        log_level = logger.getEffectiveLevel()
        try:
            assert not NAME_BUFFER
            self._config = self.CONFIG(kwds.pop('options', {}))
            self._config.set_value(kwds)

            if self._config.verbose and log_level > logging.INFO:
                logger.setLevel(logging.INFO)
                self.verbose = True
            elif log_level <= logging.INFO:
                self.verbose = True
            else:
                self.verbose = False

            (instance_rBigM, cuts_obj, instance_rHull, var_info, 
             transBlockName) = self._setup_subproblems( instance, bigM,
                                                        self._config.\
                                                        tighten_relaxation)

            self._generate_cuttingplanes( instance_rBigM, cuts_obj,
                                          instance_rHull, var_info,
                                          transBlockName)

            # restore integrality
            TransformationFactory('core.relax_integer_vars').apply_to(instance,
                                                                      undo=True)
        finally:
            del self._config
            del self.verbose
            # clear the global name buffer
            NAME_BUFFER.clear()
            # restore logging level
            logger.setLevel(original_log_level)

    def _setup_subproblems(self, instance, bigM, tighten_relaxation_callback):
        # create transformation block
        transBlockName, transBlock = self._add_transformation_block(instance)

        # We store a list of all vars so that we can efficiently
        # generate maps among the subproblems

        transBlock.all_vars = list(v for v in instance.component_data_objects(
            Var,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic) if not v.is_fixed())

        # we'll store all the cuts we add together
        nm = self._config.cuts_name
        if nm is None:
            cuts_obj = transBlock.cuts = Constraint(NonNegativeIntegers)
        else:
            # check that this really is an available name
            if instance.component(nm) is not None:
                raise GDP_Error("cuts_name was specified as '%s', but this is "
                                "already a component on the instance! Please "
                                "specify a unique name." % nm)
            instance.add_component(nm, Constraint(NonNegativeIntegers))
            cuts_obj = instance.component(nm)

        # get bigM and hull relaxations
        bigMRelaxation = TransformationFactory('gdp.bigm')
        hullRelaxation = TransformationFactory('gdp.hull')
        relaxIntegrality = TransformationFactory('core.relax_integer_vars')

        #
        # Generate the Hull relaxation (used for the separation
        # problem to generate cutting planes)
        #
        tighter_instance = tighten_relaxation_callback(instance)
        instance_rHull = hullRelaxation.create_using(tighter_instance)
        relaxIntegrality.apply_to(instance_rHull,
                                  transform_deactivated_blocks=True)

        #
        # Reformulate the instance using the BigM relaxation (this will
        # be the final instance returned to the user)
        #
        bigMRelaxation.apply_to(instance, bigM=bigM)

        #
        # Generate the continuous relaxation of the BigM transformation. We'll
        # restore it at the end.
        #
        relaxIntegrality.apply_to(instance, transform_deactivated_blocks=True)

        #
        # Add the xstar parameter for the Hull problem
        #
        transBlock_rHull = instance_rHull.component(transBlockName)
        #
        # this will hold the solution to rbigm each time we solve it. We
        # add it to the transformation block so that we don't have to
        # worry about name conflicts.
        transBlock_rHull.xstar = Param( range(len(transBlock.all_vars)),
                                        mutable=True, default=0, within=Reals)

        # we will add a block that we will deactivate to use to store the
        # extended space cuts. We never need to solve these, but we need them to
        # be constructed for the sake of Fourier-Motzkin Elimination
        extendedSpaceCuts = transBlock_rHull.extendedSpaceCuts = Block()
        extendedSpaceCuts.deactivate()
        extendedSpaceCuts.cuts = Constraint(Any)

        #
        # Generate the mapping between the variables on all the
        # instances and the xstar parameter.
        #
        var_info = [
            (v, # this is the bigM variable
             transBlock_rHull.all_vars[i],
             transBlock_rHull.xstar[i])
            for i,v in enumerate(transBlock.all_vars)]

        # NOTE: we wait to add the separation objective to the rHull problem
        # because it is best to do it in the first iteration, so that we can
        # skip stale variables.

        return (instance, cuts_obj, instance_rHull, var_info, transBlockName)

    # this is the map that I need to translate my projected cuts and add
    # them to bigM
    def _create_hull_to_bigm_substitution_map(self, var_info):
        return dict((id(var_info[i][1]), var_info[i][0]) for i in
                    range(len(var_info)))

    # this map is needed to solve the back-off problem for post-processing
    def _create_bigm_to_hull_substition_map(self, var_info):
        return dict((id(var_info[i][0]), var_info[i][1]) for i in
                    range(len(var_info)))

    def _get_disaggregated_vars(self, hull):
        disaggregatedVars = ComponentSet()
        hull_xform = TransformationFactory('gdp.hull')
        for disjunction in hull.component_data_objects( Disjunction,
                                                        descend_into=(Disjunct,
                                                                      Block)):
            for disjunct in disjunction.disjuncts:
                if disjunct.transformation_block is not None:
                    transBlock = disjunct.transformation_block()
                    for v in transBlock.disaggregatedVars.\
                        component_data_objects(Var):
                        disaggregatedVars.add(v)
                
        return disaggregatedVars

    def _get_rBigM_obj_and_constraints(self, instance_rBigM):
        # We try to grab the first active objective. If there is more
        # than one, the writer will yell when we try to solve below. If
        # there are 0, we will yell here.
        rBigM_obj = next(instance_rBigM.component_data_objects(
            Objective, active=True), None)
        if rBigM_obj is None:
            raise GDP_Error("Cannot apply cutting planes transformation "
                            "without an active objective in the model!")

        #
        # Collect all of the linear constraints that are in the rBigM
        # instance. We will need these so that we can compare what we get from
        # FME to them and make sure we aren't adding redundant constraints to
        # the model. For convenience, we will make sure they are all in the form
        # lb <= expr (so we will break equality constraints)
        #
        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        rBigM_linear_constraints = []
        for cons in instance_rBigM.component_data_objects(
                Constraint,
                descend_into=Block,
                sort=SortComponents.deterministic,
                active=True):
            body = cons.body
            if body.polynomial_degree() != 1:
                # We will never get a nonlinear constraint out of FME, so we
                # don't risk it being identical to this one.
                continue

            # TODO: Guess this shouldn't have been private...
            rBigM_linear_constraints.extend(fme._process_constraint(cons))

        # [ESJ Aug 13 2020] NOTE: We actually don't need to worry about variable
        # bounds here because the FME transformation will take care of them
        # (i.e. convert those of the disaggregated variables to constraints for
        # the purposes of the projection.)

        return rBigM_obj, rBigM_linear_constraints

    def _generate_cuttingplanes( self, instance_rBigM, cuts_obj, instance_rHull,
                                 var_info, transBlockName):

        opt = SolverFactory(self._config.solver)
        stream_solver = self._config.stream_solver
        opt.options = dict(self._config.solver_options)

        improving = True
        prev_obj = None
        epsilon = self._config.minimum_improvement_threshold
        cuts = None

        transBlock_rHull = instance_rHull.component(transBlockName)

        rBigM_obj, rBigM_linear_constraints = self.\
                                              _get_rBigM_obj_and_constraints(
                                                  instance_rBigM)

        # Get list of all variables in the rHull model which we will use when
        # calculating the composite normal vector.
        rHull_vars = [i for i in instance_rHull.component_data_objects(
            Var,
            descend_into=Block,
            sort=SortComponents.deterministic)]

        # collect a list of disaggregated variables.
        disaggregated_vars = self._get_disaggregated_vars( instance_rHull)

        hull_to_bigm_map = self._create_hull_to_bigm_substitution_map(var_info)
        bigm_to_hull_map = self._create_bigm_to_hull_substition_map(var_info)
        xhat = ComponentMap()

        while (improving):
            # solve rBigM, solution is xstar
            results = opt.solve(instance_rBigM, tee=stream_solver,
                                load_solutions=False)
            if verify_successful_solve(results) is not NORMAL:
                logger.warning("Relaxed BigM subproblem "
                               "did not solve normally. Stopping cutting "
                               "plane generation.\n\n%s" % (results,))
                return
            instance_rBigM.solutions.load_from(results)

            rBigM_objVal = value(rBigM_obj)
            logger.warning("rBigM objective = %s" % (rBigM_objVal,))

            #
            # Add the separation objective to the hull subproblem if it's not
            # there already (so in the first iteration). We're waiting until now
            # to avoid it including variables that came back stale from the
            # rbigm solve.
            #
            if transBlock_rHull.component("separation_objective") is None:
                self._add_separation_objective(var_info, transBlock_rHull)
            
            # copy over xstar
            logger.info("x* is:")
            for x_rbigm, x_hull, x_star in var_info:
                if not x_rbigm.stale:
                    x_star.value = x_rbigm.value
                    # initialize the X values
                    x_hull.value = x_rbigm.value    
                if self.verbose:
                    logger.info("\t%s = %s" % 
                                (x_rbigm.getname(fully_qualified=True,
                                                 name_buffer=NAME_BUFFER),
                                 x_rbigm.value))

            # compare objectives: check absolute difference close to 0, relative
            # difference further from 0.
            if prev_obj is None:
                improving = True
            else:
                obj_diff = prev_obj - rBigM_objVal
                improving = ( abs(obj_diff) > epsilon if abs(rBigM_objVal) < 1
                             else abs(obj_diff/prev_obj) > epsilon )

            # solve separation problem to get xhat.
            results = opt.solve(instance_rHull, tee=stream_solver,
                                load_solutions=False)
            if verify_successful_solve(results) is not NORMAL:
                logger.warning("Hull separation subproblem "
                               "did not solve normally. Stopping cutting "
                               "plane generation.\n\n%s" % (results,))
                return
            instance_rHull.solutions.load_from(results)
            logger.warning("separation problem objective value: %s" %
                           value(transBlock_rHull.separation_objective))

            # save xhat to initialize rBigM with in the next iteration
            if self.verbose:
                logger.info("xhat is: ")
            for x_rbigm, x_hull, x_star in var_info:
                xhat[x_rbigm] = value(x_hull)
                if self.verbose:
                    logger.info("\t%s = %s" % 
                                (x_hull.getname(fully_qualified=True,
                                                name_buffer=NAME_BUFFER), 
                                 x_hull.value))

            # [JDS 19 Dec 18] Note: we check that the separation objective was
            # significantly nonzero.  If it is too close to zero, either the
            # rBigM solution was in the convex hull, or the separation vector is
            # so close to zero that the resulting cut is likely to have
            # numerical issues.
            if value(transBlock_rHull.separation_objective) < \
               self._config.separation_objective_threshold:
                logger.warning("Separation problem objective below threshold of"
                               " %s: Stopping cut generation." %
                               self._config.separation_objective_threshold)
                break

            cuts = self._config.create_cuts(transBlock_rHull, var_info,
                                            hull_to_bigm_map,
                                            rBigM_linear_constraints,
                                            rHull_vars, disaggregated_vars,
                                            self._config.norm,
                                            self._config.cut_filtering_threshold,
                                            self._config.zero_tolerance,
                                            self._config.do_integer_arithmetic,
                                            self._config.\
                                            tight_constraint_tolerance)
           
            # We are done if the cut generator couldn't return a valid cut
            if cuts is None:
                logger.warning("Did not generate a valid cut, stopping cut "
                               "generation.")
                break
            if not improving:
                logger.warning("Difference in relaxed BigM problem objective "
                               "values from past two iterations is below "
                               "threshold of %s: Stopping cut generation." %
                               epsilon)
                break

            for cut in cuts:
                # we add the cut to the model and then post-process it in place.
                cut_number = len(cuts_obj)
                logger.warning("Adding cut %s to BigM model." % (cut_number,))
                cuts_obj.add(cut_number, cut)
                if self._config.post_process_cut is not None:
                    self._config.post_process_cut(
                        cuts_obj[cut_number], transBlock_rHull,
                        bigm_to_hull_map, opt, stream_solver,
                        self._config.back_off_problem_tolerance)

            if cut_number + 1 == self._config.max_number_of_cuts:
                logger.warning("Reached maximum number of cuts.")
                break
                
            prev_obj = rBigM_objVal

            # Initialize rbigm with xhat (for the next iteration)
            for x_rbigm, x_hull, x_star in var_info:
                x_rbigm.value = xhat[x_rbigm]

    def _add_transformation_block(self, instance):
        # creates transformation block with a unique name based on name, adds it
        # to instance, and returns it.
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_cuttingplane_transformation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        return transBlockName, transBlock


    def _add_separation_objective(self, var_info, transBlock_rHull):
        # creates the separation objective. That is just adding an objective for
        # Euclidean norm, it means adding an auxilary variable to linearize the
        # L-infinity norm. We do this assuming that rBigM has been solved, and
        # if any variables come back stale, we leave them out of the separation
        # problem, as they aren't doing anything and they could cause numerical
        # issues later.

        # Deactivate any/all other objectives
        for o in transBlock_rHull.model().component_data_objects(Objective):
            o.deactivate()
        norm = self._config.norm
        to_delete = []

        if norm == 2:
            obj_expr = 0
            for i, (x_rbigm, x_hull, x_star) in enumerate(var_info):
                if not x_rbigm.stale:
                    obj_expr += (x_hull - x_star)**2
                else:
                    if self.verbose:
                        logger.info("The variable %s will not be included in "
                                    "the separation problem: It was stale in "
                                    "the rBigM solve." % x_rbigm.getname(
                                        fully_qualified=True,
                                        name_buffer=NAME_BUFFER))
                    to_delete.append(i)
        elif norm == float('inf'):
            u = transBlock_rHull.u = Var(domain=NonNegativeReals)
            inf_cons = transBlock_rHull.inf_norm_linearization = Constraint(
                NonNegativeIntegers)
            i = 0
            for j, (x_rbigm, x_hull, x_star) in enumerate(var_info):
                if not x_rbigm.stale:
                    # NOTE: these are written as >= constraints so that we know
                    # the duals will come back nonnegative.
                    inf_cons[i] = u  - x_hull >= - x_star
                    inf_cons[i+1] = u + x_hull >= x_star
                    i += 2
                else:
                    if self.verbose:
                        logger.info("The variable %s will not be included in "
                                    "the separation problem: It was stale in "
                                    "the rBigM solve." % x_rbigm.getname(
                                        fully_qualified=True,
                                        name_buffer=NAME_BUFFER))
                    to_delete.append(j)
            # we'll need the duals of these to get the subgradient
            self._add_dual_suffix(transBlock_rHull.model())
            obj_expr = u
        
        # delete the unneeded x_stars so that we don't add cuts involving
        # useless variables later.
        for i in sorted(to_delete, reverse=True):
            del var_info[i]

        # add separation objective to transformation block
        transBlock_rHull.separation_objective = Objective(expr=obj_expr)

    def _add_dual_suffix(self, rHull):
        # rHull is our model and we aren't giving it back (unless in the future
        # we we add a callback to do basic steps to it...), so we just check if
        # dual is there. If it's a Suffix, we'll borrow it. If it's something
        # else we'll rename it and add the Suffix.
        dual = rHull.component("dual")
        if dual is None:
            rHull.dual = Suffix(direction=Suffix.IMPORT)
        else:
            if dual.ctype is Suffix:
                return
            rHull.del_component(dual)
            rHull.dual = Suffix(direction=Suffix.IMPORT)
            rHull.add_component(unique_component_name(rHull, "dual"), dual)
