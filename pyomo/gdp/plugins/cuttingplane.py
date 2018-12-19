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
from pyomo.core import (
    Any, Block, Constraint, Objective, Param, Var, SortComponents,
    Transformation, TransformationFactory, value, TransformationFactory,
)
from pyomo.core.base.symbolic import differentiate
from pyomo.core.expr.current import identify_variables
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt import SolverFactory

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
    verify_successful_solve, NORMAL, INFEASIBLE, NONOPTIMAL
)

from six import iterkeys, itervalues
from numpy import isclose

import math
import logging
logger = logging.getLogger('pyomo.gdp.cuttingplane')

# DEBUG
from nose.tools import set_trace
# ESJ: so far this is a bad idea. Still need to try it in the direction of the
# optimal solution to rBigM though.
tiny_step = 1e-6

# TODO: this should be an option probably, right?
# do I have other options that won't be mad about the quadratic objective in the
# separation problem?
SOLVER = 'ipopt'
stream_solvers = False


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
        default=0.05,#TODO: this is an experiment... 0.01,
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

        instance_rBigM, instance_rCHull, var_info, transBlockName \
            = self._setup_subproblems(instance, bigM)

        self._generate_cuttingplanes(
            instance, instance_rBigM, instance_rCHull, var_info, transBlockName)


    def _setup_subproblems(self, instance, bigM):
        # create transformation block
        transBlockName, transBlock = self._add_relaxation_block(
            instance,
            '_pyomo_gdp_cuttingplane_relaxation')

        # We store a list of all vars so that we can efficiently
        # generate maps among the subproblems
        transBlock.all_vars = list(v for v in instance.component_data_objects(
            Var,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic) if not v.is_fixed())

        # we'll store all the cuts we add together
        transBlock.cuts = Constraint(Any)

        # get bigM and chull relaxations
        bigMRelaxation = TransformationFactory('gdp.bigm')
        chullRelaxation = TransformationFactory('gdp.chull')
        relaxIntegrality = TransformationFactory('core.relax_integrality')

        # HACK: for the current writers, we need to also apply gdp.reclassify so
        # that the indicator variables stay where they are in the big M model
        # (since that is what we are eventually going to solve after we add our
        # cuts).
        reclassify = TransformationFactory('gdp.reclassify')

        #
        # Generate the CHull relaxation (used for the separation
        # problem to generate cutting planes)
        #
        instance_rCHull = chullRelaxation.create_using(instance)
        # This relies on relaxIntegrality relaxing variables on deactivated
        # blocks, which should be fine.
        reclassify.apply_to(instance_rCHull)
        relaxIntegrality.apply_to(instance_rCHull)

        #
        # Reformulate the instance using the BigM relaxation (this will
        # be the final instance returned to the user)
        #
        bigMRelaxation.apply_to(instance, bigM=bigM)
        reclassify.apply_to(instance)

        #
        # Generate the continuous relaxation of the BigM transformation
        #
        instance_rBigM = relaxIntegrality.create_using(instance)

        #
        # Add the xstar parameter for the CHull problem
        #
        transBlock_rCHull = instance_rCHull.component(transBlockName)
        #
        # this will hold the solution to rbigm each time we solve it. We
        # add it to the transformation block so that we don't have to
        # worry about name conflicts.
        transBlock_rCHull.xstar = Param(
            range(len(transBlock.all_vars)), mutable=True, default=None)

        transBlock_rBigM = instance_rBigM.component(transBlockName)

        #
        # Generate the mapping between the variables on all the
        # instances and the xstar parameter
        #
        var_info = tuple(
            (v,
             transBlock_rBigM.all_vars[i],
             transBlock_rCHull.all_vars[i],
             transBlock_rCHull.xstar[i])
            for i,v in enumerate(transBlock.all_vars))

        #
        # Add the separation objective to the chull subproblem
        #
        self._add_separation_objective(var_info, transBlock_rCHull)

        return instance_rBigM, instance_rCHull, var_info, transBlockName


    def _generate_cuttingplanes(
            self, instance, instance_rBigM, instance_rCHull,
            var_info, transBlockName):

        opt = SolverFactory(self._config.solver)
        stream_solver = self._config.stream_solver
        opt.options = self._config.solver_options

        improving = True
        prev_obj = float("inf")
        epsilon = self._config.EPS
        cuts = None

        transBlock = instance.component(transBlockName)
        transBlock_rBigM = instance_rBigM.component(transBlockName)

        # We try to grab the first active objective. If there is more
        # than one, the writer will yell when we try to solve below. If
        # there are 0, we will yell here.
        rBigM_obj = next(instance_rBigM.component_data_objects(
            Objective, active=True), None)
        if rBigM_obj is None:
            raise GDP_Error("Cannot apply cutting planes transformation "
                            "without an active objective in the model!")

        # Get list of all variables in the rCHull model which we will use when
        # calculating the composite normal vector.
        rCHull_vars = [i for i in instance_rCHull.component_data_objects(
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
            # DEBUG
            #print("x*:")
            for x_bigm, x_rbigm, x_chull, x_star in var_info:
                x_star.value = x_rbigm.value
                # initialize the X values
                # ESJ: Isn't this initializing with an infeasible point?
                x_chull.value = x_rbigm.value
                # DEBUG
                #print("%s: %s" % (x_rbigm.name, x_star.value))

            # compare objectives: check absolute difference close to 0, relative
            # difference further from 0.
            obj_diff = prev_obj - rBigM_objVal
            improving = math.isinf(obj_diff) or \
                        ( abs(obj_diff) > epsilon if abs(rBigM_objVal) < 1 else
                          abs(obj_diff/prev_obj) > epsilon )

            # ESJ: This makes more sense to me. You should add the cut if you
            # improved this iteration. The case where you don't add it is when
            # you just discovered that you have done enough. But they way we had
            # it, if you have a problem where you can get one good cut, we got
            # 0.
            # if improving and cuts is not None:
            #     cut_number = len(transBlock.cuts)
            #     logger.warning("GDP.cuttingplane: Adding cut %s to BM model."
            #                    % (cut_number,))
            #     transBlock.cuts.add(cut_number, cuts['bigm'] <= 0)

            # solve separation problem to get xhat.
            opt.solve(instance_rCHull, tee=stream_solver)
            #print "Separation obj = %s" % (
            #    value(next(instance_rCHull.component_data_objects(
            #    Objective, active=True))),)

            # [JDS 19 Dec 18] Note: we should check that the separation
            # objective was significantly nonzero.  If it is too close
            # to zero, either the rBigM solution was in the convex hull,
            # or the separation vector is so close to zero that the
            # resulting cut is likely to have numerical issues.

            # DEBUG
            #print("x_hat:")
            #for x_hat in rCHull_vars:
            #    print("%s: %s" % (x_hat.name, x_hat.value))

            cuts = self._create_cuts(var_info, rCHull_vars, instance_rCHull,
                                     transBlock, transBlock_rBigM)
            # We are done if the cut generator couldn't return a valid cut
            if not cuts:
                break

            # add cut to rBigm
            transBlock_rBigM.cuts.add(len(transBlock_rBigM.cuts),
                                      cuts['rBigM'] <= 0)

            # DEBUG
            #print("adding this cut to rBigM:\n%s <= 0" % cuts['rBigM'])

            if improving:
                cut_number = len(transBlock.cuts)
                logger.warning("GDP.cuttingplane: Adding cut %s to BM model."
                               % (cut_number,))
                transBlock.cuts.add(cut_number, cuts['bigm'] <= 0)

            prev_obj = rBigM_objVal


    def _add_relaxation_block(self, instance, name):
        # creates transformation block with a unique name based on name, adds it
        # to instance, and returns it.
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_cuttingplane_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        return transBlockName, transBlock


    def _add_separation_objective(self, var_info, transBlock_rCHull):
        # Deactivate any/all other objectives
        for o in transBlock_rCHull.model().component_data_objects(Objective):
            o.deactivate()

        obj_expr = 0
        for x_bigm, x_rbigm, x_chull, x_star in var_info:
            obj_expr += (x_chull - x_star)**2
        # add separation objective to transformation block
        transBlock_rCHull.separation_objective = Objective(expr=obj_expr)


    def _create_cuts(self, var_info, rCHull_vars, instance_rCHull, transBlock,
                     transBlock_rBigm):
        cut_number = len(transBlock.cuts)
        logger.warning("gdp.cuttingplane: Creating (but not yet adding) cut %s."
                       % (cut_number,))
        # DEBUG
        # print("CURRENT SOLN (to separation problem):")
        # for var in rCHull_vars:
        #     print(var.name + '\t' + str(value(var)))

        # loop through all constraints in rCHull and figure out which are active
        # or slightly violated. For each we will get the tangent plane at xhat
        # (which is x_chull below). We get the normal vector for each of these
        # tangent planes and sum them to get a composite normal. Our cut is then
        # the hyperplane normal to this composite through xbar (projected into
        # the original space).
        normal_vectors = []
        # DEBUG
        # print("-------------------------------")
        # print("These constraints are tight:")
        #print "POINT: ", [value(_) for _ in rCHull_vars]
        for constraint in instance_rCHull.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                sort=SortComponents.deterministic):
            #print "   CON: ", constraint.expr
            multiplier = self.constraint_tight(instance_rCHull, constraint)
            if multiplier:
                # DEBUG
                # print(constraint.name)
                # print constraint.expr
                # get normal vector to tangent plane to this constraint at xhat
                #print "      TIGHT", multiplier
                f = constraint.body
                firstDerivs = differentiate(f, wrt_list=rCHull_vars)
                #print "     ", firstDerivs
                normal_vectors.append(
                    [multiplier*value(_) for _ in firstDerivs])
                #set_trace()

        # It is possible that the separation problem returned a point in
        # the interior of the convex hull.  It is also possible that the
        # only active constraints are (feasible) equality constraints.
        # in these situations, there are no normal vectors from which to
        # create a valid cut.
        if not normal_vectors:
            return None

        composite_normal = list(
            sum(_) for _ in zip(*tuple(normal_vectors)) )
        composite_normal_map = ComponentMap(
            (v,n) for v,n in zip(rCHull_vars, composite_normal))

        # DEBUG
        # print "COMPOSITE NORMAL, cut number %s" % cut_number
        # for x,v in composite_normal.iteritems():
        #     print(x.name + '\t' + str(v))

        # add a cut which is tangent to the composite normal at xhat:
        # (we are projecting out the disaggregated variables)
        composite_cutexpr_bigm = 0
        composite_cutexpr_rBigM = 0
        projection_cutexpr_bigm = 0
        projection_cutexpr_rBigM = 0
        # TODO: I don't think we need x_star in var_info anymore. Or maybe at
        # all?
        # DEBUG:
        #print composite_normal
        #print("FOR COMPARISON:\ncomposite\tx_hat - xstar")
        for x_bigm, x_rbigm, x_chull, x_star in var_info:
            composite_cutexpr_bigm \
                += composite_normal_map[x_chull]*(x_bigm - x_chull.value)
            composite_cutexpr_rBigM \
                += composite_normal_map[x_chull]*(x_rbigm - x_chull.value)

            # DEBUG: old way
            projection_cutexpr_bigm += 2*(x_star.value - x_chull.value)*\
                                       (x_bigm - x_chull.value)
            projection_cutexpr_rBigM += 2*(x_star.value - x_chull.value)*\
                                        (x_rbigm - x_chull.value)
            # DEBUG:
            # print("%s\t%s" %
            #       (composite_normal[x_chull], x_star.value - x_chull.value))

            # DEBUG: Let's try moving out along composite_normal
            # cutexpr_bigm += composite_normal[x_chull]*\
            #                 (x_bigm - (x_chull.value + \
            #                            tiny_step*composite_normal[x_chull]))
            # cutexpr_rBigM += composite_normal[x_chull]*\
            #                  (x_rbigm - (x_chull.value + \
            #                            tiny_step*composite_normal[x_chull]))

            # DEBUG some more: Let's try moving out along vector towards opt
            # solution to rBigM
            # backOff = x_star.value - x_chull.value
            # cutexpr_bigm += composite_normal[x_chull]*\
            #                 (x_bigm - (x_chull.value + tiny_step*backOff))
            # cutexpr_rBigM += composite_normal[x_chull]*\
            #                  (x_rbigm - (x_chull.value + tiny_step*backOff))

        #print "Composite normal cut"
        #print "   %s" % (composite_cutexpr_rBigM,)
        #print "Projection cut"
        #print "   %s" % (projection_cutexpr_rBigM,)
        # DEBUG
        # print("++++++++++++++++++++++++++++++++++++++++++")
        # print("So this is the cut expression:")
        # print(cutexpr_bigm)

        #return({'bigm': projection_cutexpr_bigm,
        #        'rBigM': projection_cutexpr_rBigM})
        return({'bigm': composite_cutexpr_bigm,
                'rBigM': composite_cutexpr_rBigM})


    def constraint_tight(self, model, constraint):
        val = value(constraint.body)
        ans = 0
        #print "    vals:", value(constraint.lower), val, value(constraint.upper)
        if constraint.lower is not None:
            if value(constraint.lower) >= val:
                ans += 1

            # if isclose(val, constraint.lower.value):
            #     if val > constraint.lower.value:
            #         difference = val - constraint.lower.value
            #         # print("DEBUG: We're in interior of constraint %s LB by %s"
            #         #       % (constraint.name, difference))
            #         # DEBUG: an experiment...
            #         #return False
            #     return -1
        if constraint.upper is not None:
            if value(constraint.upper) <= val:
                ans -= 1

            # if isclose(val, constraint.upper.value):
            #     if val < constraint.upper.value:
            #         difference = constraint.upper.value - val
            #         # print("DEBUG: We're in interior of constraint %s UB by %s"
            #         #       % (constraint.name, difference))
            #         #return False
            #     return 1

        return -1*ans
