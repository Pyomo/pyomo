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

from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Any, Block, Constraint, Objective, Param, Var, SortComponents,
    Transformation, TransformationFactory, value, TransformationFactory
)
from pyomo.opt import SolverFactory

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
    verify_successful_solve, NORMAL, INFEASIBLE, NONOPTIMAL
)

from six import iterkeys, itervalues

import math
import logging
logger = logging.getLogger('pyomo.gdp.cuttingplane')

# DEBUG
from nose.tools import set_trace

# TODO: this should be an option probably, right?
# do I have other options that won't be mad about the quadratic objective in the
# separation problem?
SOLVER = 'ipopt'
stream_solvers = False


@TransformationFactory.register('gdp.cuttingplane', doc="Relaxes a linear disjunctive model by "
          "adding cuts from convex hull to Big-M relaxation.")
class CuttingPlane_Transformation(Transformation):

    def __init__(self):
        super(CuttingPlane_Transformation, self).__init__()

    def _apply_to(self, instance, bigM=None, **kwds):
        options = kwds.pop('options', {})

        if kwds:
            logger.warning("GDP(CuttingPlanes): unrecognized keyword arguments:"
                           "\n%s" % ( '\n'.join(iterkeys(kwds)), ))
        if options:
            logger.warning("GDP(CuttingPlanes): unrecognized options:\n%s"
                        % ( '\n'.join(iterkeys(options)), ))

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
        # Generalte the CHull relaxation (used for the separation
        # problem to generate cutting planes
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

        opt = SolverFactory(SOLVER)

        improving = True
        iteration = 0
        prev_obj = float("inf")
        epsilon = 0.01

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

        while (improving):
            # solve rBigM, solution is xstar
            results = opt.solve(instance_rBigM, tee=stream_solvers)
            if verify_successful_solve(results) is not NORMAL:
                logger.warning("GDP.cuttingplane: Relaxed BigM subproblem "
                               "did not solve normally. Stopping cutting "
                               "plane generation.\n\n%s" % (results,))
                return

            rBigM_objVal = value(rBigM_obj)
            logger.warning("gdp.cuttingplane: rBigM objective = %s"
                           % (rBigM_objVal,))

            # copy over xstar
            for x_bigm, x_rbigm, x_chull, x_star in var_info:
                x_star.value = x_rbigm.value
                # initialize the X values
                x_chull.value = x_rbigm.value

            # solve separation problem to get xhat.
            results = opt.solve(instance_rCHull, tee=stream_solvers)
            if verify_successful_solve(results) is not NORMAL:
                logger.warning("GDP.cuttingplane: CHull separation subproblem "
                               "did not solve normally. Stopping cutting "
                               "plane generation.\n\n%s" % (results,))
                return

            self._add_cut(var_info, transBlock, transBlock_rBigM)

            # decide whether or not to keep going: check absolute difference
            # close to 0, relative difference further from 0.
            obj_diff = prev_obj - rBigM_objVal
            improving = math.isinf(obj_diff) or \
                        ( abs(obj_diff) > epsilon if abs(rBigM_objVal) < 1 else
                          abs(obj_diff/prev_obj) > epsilon )

            prev_obj = rBigM_objVal
            iteration += 1


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


    def _add_cut(self, var_info, transBlock, transBlock_rBigM):
        # add cut to BM and rBM
        cut_number = len(transBlock.cuts)
        logger.warning("gdp.cuttingplane: Adding cut %s to BM model."
                       % (cut_number,))

        cutexpr_bigm = 0
        cutexpr_rBigM = 0
        for x_bigm, x_rbigm, x_chull, x_star in var_info:
            # xhat = x_chull.value
            cutexpr_bigm += (
                x_chull.value - x_star.value)*(x_bigm - x_chull.value)
            cutexpr_rBigM += (
                x_chull.value - x_star.value)*(x_rbigm - x_chull.value)

        transBlock.cuts.add(cut_number, cutexpr_bigm >= 0)
        transBlock_rBigM.cuts.add(cut_number, cutexpr_rBigM >= 0)
