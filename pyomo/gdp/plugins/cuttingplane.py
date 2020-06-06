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
try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict


from pyomo.common.config import ConfigBlock, ConfigValue, PositiveFloat
from pyomo.common.modeling import unique_component_name
from pyomo.core import ( Any, Block, Constraint, Objective, Param, Var,
                         SortComponents, Transformation, TransformationFactory,
                         value, TransformationFactory, NonNegativeIntegers,
                         Reals )
from pyomo.core.expr import differentiate
from pyomo.core.base.component import ComponentUID
from pyomo.core.expr.current import identify_variables
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt import SolverFactory

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import ( verify_successful_solve, NORMAL, INFEASIBLE,
                             NONOPTIMAL, clone_without_expression_components )

from pyomo.contrib.fme.fourier_motzkin_elimination import \
    Fourier_Motzkin_Elimination_Transformation

from six import iterkeys, itervalues, iteritems
from numpy import isclose

import math
import logging
logger = logging.getLogger('pyomo.gdp.cuttingplane')

# DEBUG
from nose.tools import set_trace

@TransformationFactory.register('gdp.cuttingplane',
                                doc="Relaxes a linear disjunctive model by "
                                "adding cuts from convex hull to Big-M "
                                "relaxation.")
class CuttingPlane_Transformation(Transformation):
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
    CONFIG.declare('EPS', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description="Epsilon value used to decide when to stop adding cuts",
        doc="""
        If the difference between the objectives in two consecutive iterations is
        less than this value, the algorithm terminates without adding the cut
        generated in the last iteration.  """
    ))
    CONFIG.declare('stream_solver', ConfigValue(
        default=False,
        domain=bool,
        description="""If true, sets tee=True for every solve performed over
        "the course of the algorithm"""
    ))
    # TODO: Why not just switch to having them give you the SolverFactory?
    # 05/24: There are some complications with that (default gets created), but
    # it's still the answer.
    CONFIG.declare('solver_options', ConfigValue(
        default={},
        description="Dictionary of solver options",
        doc="""
        Dictionary of solver options that will be set for the solver for both the
        relaxed BigM and separation problem solves.
        """
    ))

    def __init__(self):
        super(CuttingPlane_Transformation, self).__init__()

    def _apply_to(self, instance, bigM=None, **kwds):
        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        (instance_rBigM, instance_rHull, var_info, 
         var_map, disaggregated_vars, 
         disaggregation_constraints, 
         rBigM_linear_constraints, transBlockName) = self._setup_subproblems(
             instance, bigM)

        self._generate_cuttingplanes( instance_rBigM, instance_rHull, var_info,
                                      var_map, disaggregated_vars,
                                      disaggregation_constraints,
                                      rBigM_linear_constraints, transBlockName)

        # restore integrality
        TransformationFactory('core.relax_integer_vars').apply_to(instance,
                                                                  undo=True)

    def _setup_subproblems(self, instance, bigM):
        # create transformation block
        transBlockName, transBlock = self._add_relaxation_block(
            instance,
            '_pyomo_gdp_cuttingplane_relaxation')

        # We store a list of all vars so that we can efficiently
        # generate maps among the subproblems

        # TODO: AAAAGH, we're going to have to make the fixed thing an option
        # here too!
        transBlock.all_vars = list(v for v in instance.component_data_objects(
            Var,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic) if not v.is_fixed())

        # we'll store all the cuts we add together
        transBlock.cuts = Constraint(NonNegativeIntegers)

        # get bigM and hull relaxations
        bigMRelaxation = TransformationFactory('gdp.bigm')
        hullRelaxation = TransformationFactory('gdp.hull')
        relaxIntegrality = TransformationFactory('core.relax_integer_vars')

        # HACK: for the current writers, we need to also apply gdp.reclassify so
        # that the indicator variables stay where they are in the big M model
        # (since that is what we are eventually going to solve after we add our
        # cuts).
        reclassify = TransformationFactory('gdp.reclassify')

        #
        # Generate the Hull relaxation (used for the separation
        # problem to generate cutting planes)
        #
        instance_rHull = hullRelaxation.create_using(instance,
                                                     targets=instance)
        # collect a list of disaggregated variables. We have to do this before
        # reclassify because we rely on Disjuncts still having a different
        # ctype.
        (disaggregated_vars, 
         disaggregation_constraints) = self._get_disaggregated_vars(
             instance_rHull)
        reclassify.apply_to(instance_rHull)
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

        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        #
        # Collect all of the linear constraints that are in the rBigM
        # instance. We will need these so that we can compare what we get from
        # FME to them and make sure we aren't adding redundant constraints to
        # the model. For convenience, we will make sure they are all in the form
        # lb <= expr (so we will break equality constraints)
        #
        rBigM_linear_constraints = []
        for cons in instance.component_data_objects(
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

        # [ESJ 3 June 2020] TODO: Do we need to also pull out variable bounds
        # here?? I think so, right??

        #
        # Add the xstar parameter for the Hull problem
        #
        transBlock_rHull = instance_rHull.component(transBlockName)
        #
        # this will hold the solution to rbigm each time we solve it. We
        # add it to the transformation block so that we don't have to
        # worry about name conflicts.
        transBlock_rHull.xstar = Param( range(len(transBlock.all_vars)),
                                        mutable=True, default=None,
                                        within=[None] | Reals)
        # we will add a block that we will deactivate to use to store the
        # extended space cuts. We never need to solve these, but we need them to
        # be contructed for the sake of Fourier-Motzkin Elimination
        extendedSpaceCuts = transBlock_rHull.extendedSpaceCuts = Block()
        extendedSpaceCuts.deactivate()
        extendedSpaceCuts.cuts = Constraint(Any)

        transBlock_rBigM = instance.component(transBlockName)

        #
        # Generate the mapping between the variables on all the
        # instances and the xstar parameter.
        #
        var_info = tuple(
            (v,
             transBlock_rBigM.all_vars[i],
             transBlock_rHull.all_vars[i],
             transBlock_rHull.xstar[i])
            for i,v in enumerate(transBlock.all_vars))

        # this is the map that I need to translate my projected cuts and add
        # them to bigM and rBigM.
        # [ESJ 5 March 2019] TODO: If I add xstar to this (or don't) can I just
        # replace var_info?
        var_map = ComponentMap((transBlock_rHull.all_vars[i], v)
                               for i,v in enumerate(transBlock.all_vars))

        #
        # Add the separation objective to the hull subproblem
        #
        self._add_separation_objective(var_info, transBlock_rHull)

        return (instance, instance_rHull, var_info, var_map, disaggregated_vars,
                disaggregation_constraints, rBigM_linear_constraints,
                transBlockName)

    def _get_disaggregated_vars(self, hull):
        # This function MUST be called *before* reclassify for this to work
        # (else we don't pick up any Disjuncts in the component_data_objects
        # call below.)
        disaggregatedVars = []
        disaggregationConstraints = ComponentSet()
        hull_xform = TransformationFactory('gdp.hull')
        for disjunction in hull.component_data_objects( Disjunction,
                                                        descend_into=(Disjunct,
                                                                      Block)):
            for disjunct in disjunction.disjuncts:
                if disjunct.transformation_block is not None:
                    transBlock = disjunct.transformation_block()
                    for v in transBlock.component_data_objects(Var):
                        disaggregatedVars.append(v)
                        disaggregationConstraints.add(
                            hull_xform.get_disaggregation_constraint(
                                hull_xform.get_src_var(v), disjunction))
                
        return disaggregatedVars, disaggregationConstraints

    def _generate_cuttingplanes( self, instance_rBigM, instance_rHull, var_info,
                                 var_map, disaggregated_vars,
                                 disaggregation_constraints,
                                 rBigM_linear_constraints, transBlockName):

        opt = SolverFactory(self._config.solver)
        stream_solver = self._config.stream_solver
        opt.options = self._config.solver_options

        improving = True
        prev_obj = float("inf")
        epsilon = self._config.EPS
        cuts = None

        transBlock_rBigM = instance_rBigM.component(transBlockName)
        transBlock_rHull = instance_rHull.component(transBlockName)

        # We try to grab the first active objective. If there is more
        # than one, the writer will yell when we try to solve below. If
        # there are 0, we will yell here.
        rBigM_obj = next(instance_rBigM.component_data_objects(
            Objective, active=True), None)
        if rBigM_obj is None:
            raise GDP_Error("Cannot apply cutting planes transformation "
                            "without an active objective in the model!")

        # Get list of all variables in the rHull model which we will use when
        # calculating the composite normal vector.
        rHull_vars = [i for i in instance_rHull.component_data_objects(
            Var,
            descend_into=Block,
            sort=SortComponents.deterministic)]

        while (improving):
            # solve rBigM, solution is xstar
            results = opt.solve(instance_rBigM, tee=stream_solver)
            if verify_successful_solve(results) is not NORMAL:
                logger.warning("GDP.cuttingplane: Relaxed BigM subproblem "
                               "did not solve normally. Stopping cutting "
                               "plane generation.\n\n%s" % (results,))
                return

            rBigM_objVal = value(rBigM_obj)
            logger.warning("gdp.cuttingplane: rBigM objective = %s"
                           % (rBigM_objVal,))

            # copy over xstar
            print("x* is:")
            for x_bigm, x_rbigm, x_hull, x_star in var_info:
                x_star.value = x_rbigm.value
                # initialize the X values
                x_hull.value = x_rbigm.value
                print("%s: %s" % (x_rbigm.name, x_rbigm.value))

            # compare objectives: check absolute difference close to 0, relative
            # difference further from 0.
            obj_diff = prev_obj - rBigM_objVal
            improving = math.isinf(obj_diff) or \
                        ( abs(obj_diff) > epsilon if abs(rBigM_objVal) < 1 else
                          abs(obj_diff/prev_obj) > epsilon )

            # solve separation problem to get xhat.
            opt.solve(instance_rHull, tee=stream_solver)
            if verify_successful_solve(results) is not NORMAL:
                logger.warning("GDP.cuttingplane: Hull separation subproblem "
                               "did not solve normally. Stopping cutting "
                               "plane generation.\n\n%s" % (results,))
                return
            # DEBUG:
            print("xhat is: ")
            for x_bigm, x_rbigm, x_hull, x_star in var_info:
                print("%s: %s" % (x_hull.name, x_hull.value))

            # [JDS 19 Dec 18] Note: we check that the separation objective was
            # significantly nonzero.  If it is too close to zero, either the
            # rBigM solution was in the convex hull, or the separation vector is
            # so close to zero that the resulting cut is likely to have
            # numerical issues.
            if abs(value(transBlock_rHull.separation_objective)) < epsilon:
                break

            cuts = self._create_cuts(var_info, var_map, disaggregated_vars,
                                     disaggregation_constraints, rHull_vars,
                                     instance_rHull, rBigM_linear_constraints,
                                     transBlock_rBigM, transBlock_rHull)
           
            # We are done if the cut generator couldn't return a valid cut
            if cuts is None or not improving:
                break

            for cut in cuts:
                cut_number = len(transBlock_rBigM.cuts)
                logger.warning("GDP.cuttingplane: Adding cut %s to BM model."
                               % (cut_number,))
                transBlock_rBigM.cuts.add(cut_number, cut)

            prev_obj = rBigM_objVal


    def _add_relaxation_block(self, instance, name):
        # creates transformation block with a unique name based on name, adds it
        # to instance, and returns it.
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_cuttingplane_transformation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        return transBlockName, transBlock


    def _add_separation_objective(self, var_info, transBlock_rHull):
        # Deactivate any/all other objectives
        for o in transBlock_rHull.model().component_data_objects(Objective):
            o.deactivate()

        obj_expr = 0
        for x_bigm, x_rbigm, x_hull, x_star in var_info:
            obj_expr += (x_hull - x_star)**2
        # add separation objective to transformation block
        transBlock_rHull.separation_objective = Objective(expr=obj_expr)


    def _create_cuts(self, var_info, var_map, disaggregated_vars,
                     disaggregation_constraints, rHull_vars, instance_rHull,
                     rBigM_linear_constraints, transBlock_rBigm,
                     transBlock_rHull):
        cut_number = len(transBlock_rBigm.cuts)
        logger.warning("gdp.cuttingplane: Creating (but not yet adding) cut %s."
                       % (cut_number,))

        # loop through all constraints in rHull and figure out which are active
        # or slightly violated. For each we will get the tangent plane at xhat
        # (which is x_hull below). We get the normal vector for each of these
        # tangent planes and sum them to get a composite normal. Our cut is then
        # the hyperplane normal to this composite through xbar (projected into
        # the original space).
        normal_vectors = []
        tight_constraints = Block()
        conslist = tight_constraints.constraints = Constraint(
            NonNegativeIntegers)
        conslist.construct()
        for constraint in instance_rHull.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                sort=SortComponents.deterministic):
            multiplier = self.constraint_tight(instance_rHull, constraint)
            if multiplier:
                f = constraint.body
                firstDerivs = differentiate(f, wrt_list=rHull_vars)
                normal_vec = [multiplier*value(_) for _ in firstDerivs]
                normal_vectors.append(normal_vec)
                # check if constraint is linear
                if f.polynomial_degree() == 1:
                    conslist[len(conslist)] = constraint.expr
                else: 
                    # we will use the linear approximation of this constraint at
                    # x_hat
                    conslist[len(conslist)] = self.get_linear_approximation_expr(
                        normal_vec, rHull_vars)
            # even if it was satisfied exactly, we need to grab the
            # disaggregation constraints in order to do the projection.
            elif constraint in disaggregation_constraints:
                conslist[len(conslist)] = constraint.expr

        # It is possible that the separation problem returned a point in
        # the interior of the convex hull.  It is also possible that the
        # only active constraints are (feasible) equality constraints.
        # in these situations, there are no normal vectors from which to
        # create a valid cut.
        if not normal_vectors:
            return None

        hull_xform = TransformationFactory('gdp.hull')

        composite_normal = list(
            sum(_) for _ in zip(*tuple(normal_vectors)) )
        composite_normal_map = ComponentMap(
            (v,n) for v,n in zip(rHull_vars, composite_normal))
        
        composite_cutexpr_Hull = 0
        for x_bigm, x_rbigm, x_hull, x_star in var_info:
            # make the cut in the Hull space with the Hull variables. We will
            # translate it all to BigM and rBigM later when we have projected
            # out the disaggregated variables
            composite_cutexpr_Hull += composite_normal_map[x_hull]*\
                                       (x_hull - x_hull.value)

        # expand the composite_cutexprs to be in the extended space
        vars_to_eliminate = ComponentSet()
        do_fme = False
        # add the part of the expression involving the disaggregated variables.
        for x_disaggregated in disaggregated_vars:
            normal_vec_component = composite_normal_map[x_disaggregated]
            composite_cutexpr_Hull += normal_vec_component*\
                                       (x_disaggregated - x_disaggregated.value)
            vars_to_eliminate.add(x_disaggregated)
            # check that at least one disaggregated variable appears in the
            # constraint. Else we don't need to do FME
            if not do_fme and normal_vec_component != 0:
                do_fme = True
    
        conslist[len(conslist)] = composite_cutexpr_Hull <= 0
        
        if do_fme:
            tight_constraints.construct()
            TransformationFactory('contrib.fourier_motzkin_elimination').\
                apply_to(tight_constraints, vars_to_eliminate=vars_to_eliminate)
            # I made this block, so I know they are here. Not that I won't hate
            # myself later for messing with private stuff.
            fme_results = tight_constraints._pyomo_contrib_fme_transformation.\
                          projected_constraints
            projected_constraints = [cons for i, cons in iteritems(fme_results)]
        else:
            # we didn't need to project, so it's the last guy we added.
            projected_constraints = [conslist[len(conslist) - 1]]

        # we created these constraints with the variables from rHull. We
        # actually need constraints for BigM and rBigM now!
        cuts = self.get_constraint_exprs(projected_constraints, var_map)

        # We likely have some cuts that duplicate other constraints now. We will
        # filter them to make sure that they do in fact cut off x*. If that's
        # the case, we know they are not already in the BigM relaxation.
        for i in sorted(range(len(cuts)), reverse=True):
            cut = cuts[i]
            # x* is still in rBigM, so we can just remove this constraint if it
            # is satisfied at x*
            print("hi there")
            print(cut)
            if value(cut):
                del cuts[i]
                continue
            # we have found a constraint which cuts of x* by some convincing
            # amount and is not already in rBigM, this has to be our cut and we
            # can stop. We know cut is lb <= expr and that it's violated
            assert len(cut.args) == 2
            # TODO: OK, we need a tolerance option here...
            print("maybe: ")
            print(cut)
            print(value(cut.args[0]) - value(cut.args[1]))
            if value(cut.args[0]) - value(cut.args[1]) > 0.001:
                return [cut]

        return None

    def get_constraint_exprs(self, constraints, var_map):
        cuts = []
        for cons in constraints:
            cuts.append(clone_without_expression_components( 
                cons.expr, substitute=dict((id(v), subs) for v, subs in
                                           iteritems(var_map))))
        return cuts
            
    def constraint_tight(self, model, constraint):
        val = value(constraint.body)
        ans = 0
        if constraint.lower is not None:
            if value(constraint.lower) >= val:
                # tight or in violation of LB
                ans -= 1

        if constraint.upper is not None:
            if value(constraint.upper) <= val:
                # tight or in violation of UB
                ans += 1

        return ans

    def get_linear_approximation_expr(self, normal_vec, point):
        body = 0
        for coef, v in zip(point, normal_vec):
            body -= coef*v
        return body >= -sum(normal_vec[idx]*v.value for (idx, v) in
                           enumerate(point))
