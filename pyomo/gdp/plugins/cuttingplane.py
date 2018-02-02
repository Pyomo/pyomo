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

from pyomo.util.modeling import unique_component_name
from pyomo.core import *
from pyomo.gdp import *
from pyomo.opt import SolverFactory
from pyomo.util.plugin import alias
from pyomo.core.base import Transformation

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

class CuttingPlane_Transformation(Transformation):

    alias('gdp.cuttingplane', doc="Relaxes a linear disjunctive model by "
          "adding cuts from convex hull to Big-M relaxation.")

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

        # create transformation block
        transBlock = self._add_relaxation_block(
            instance,
            '_pyomo_gdp_cuttingplane_relaxation')
        # we need to later find the transformation block on transformed versions
        # of this instance, so we are going to grab the cuid here.
        transBlockCUID = ComponentUID(transBlock)

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

        instance_rChull = chullRelaxation.create_using(instance)
        # This relies on relaxIntegrality relaxing variables on deactivated
        # blocks, which should be fine.
        reclassify.apply_to(instance_rChull)
        relaxIntegrality.apply_to(instance_rChull)

        bigMRelaxation.apply_to(instance, bigM=bigM)
        reclassify.apply_to(instance)
        instance_rBigm = relaxIntegrality.create_using(instance)

        self._cuttingplanes_transformation(instance, instance_rBigm,
                                           instance_rChull, transBlock,
                                           transBlockCUID)


    def _cuttingplanes_transformation(self, instance, instance_rBigm,
                                      instance_rChull, transBlock,
                                      transBlockCUID):
        opt = SolverFactory(SOLVER)

        improving = True
        iteration = 0
        prev_obj = float("inf")
        epsilon = 0.01

        transBlock_rChull = transBlockCUID.find_component(instance_rChull)
        transBlock_rBigm = transBlockCUID.find_component(instance_rBigm)

        for o in instance_rChull.component_data_objects(Objective):
            o.deactivate()

        # build map of components and their cuids. (I need cuids because I need
        # to be able to find the same component on the convex hull instance
        # later.)
        v_map = OrderedDict()
        for v in instance_rBigm.component_data_objects(
            Var,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
            ):
            v_map[id(v)] = (ComponentUID(v), v, len(v_map))

        self._add_separation_objective(v_map, instance_rChull, transBlock_rChull)

        # We try to grab the first active objective. If there is more than one,
        # the writer will yell when we try to solve below. If there are 0, we
        # will yell here.
        rBigM_obj = next(instance_rBigm.component_data_objects(
            Objective,
            active=True), None)
        if rBigM_obj is None:
            raise GDP_Error("Cannot apply cutting planes transformation "
                            "without an active objective in the model!")
        while (improving):
            # solve rBigm, solution is xstar
            opt.solve(instance_rBigm, tee=stream_solvers)
            rBigm_objVal = value(rBigM_obj)

            # copy over xstar
            for cuid, v, i in itervalues(v_map):
                transBlock_rChull.xstar[i] = value(v)

            # solve separation problem to get xhat.
            opt.solve(instance_rChull, tee=stream_solvers)

            self._add_cut(v_map, instance, instance_rChull, transBlock,
                          transBlock_rChull, transBlock_rBigm, iteration)

            # decide whether or not to keep going: check absolute difference
            # close to 0, relative difference further from 0.
            obj_diff = prev_obj - rBigm_objVal
            improving = math.isinf(obj_diff) or \
                        ( abs(obj_diff) > epsilon if abs(obj_diff) < 1 else
                          abs(obj_diff/prev_obj) > epsilon )

            prev_obj = rBigm_objVal
            iteration += 1


    def _add_relaxation_block(self, instance, name):
        # creates transformation block with a unique name based on name, adds it
        # to instance, and returns it.
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_cuttingplane_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        return transBlock


    def _add_separation_objective(self, v_map, instance_rChull,
                                  transBlock_rChull):
        # this will hold the solution to rbigm each time we solve it. We add it
        # to the transformation block so that we don't have to worry about name
        # conflicts.
        transBlock_rChull.xstar = Param(
            range(len(v_map)), mutable=True, default=None )

        obj_expr = 0
        for cuid, v, i in itervalues(v_map):
            x_star = transBlock_rChull.xstar[i]
            x = cuid.find_component(instance_rChull)
            obj_expr += (x - x_star)**2
        # add separation objective to transformation block
        transBlock_rChull.separation_objective = Objective(expr=obj_expr)


    def _add_cut(self, v_map, instance, instance_rChull, transBlock,
                 transBlock_rChull, transBlock_rBigm, cut_index):
        # add cut to BM and rBM
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Adding cut %s to BM model." % str(iteration))

        cutexpr_bigm = 0
        cutexpr_rBigm = 0
        # QUESTION: I think that this is not deterministic? We are getting the
        # variables in different orders when we build this constraint. Which is
        # awful for testing.
        for cuid, v, i in itervalues(v_map):
            xhat = cuid.find_component(instance_rChull).value
            xstar = transBlock_rChull.xstar[i].value
            x_bigm = cuid.find_component(instance)
            x_rBigm = v
            cutexpr_bigm += (xhat - xstar)*(x_bigm - xhat)
            cutexpr_rBigm += (xhat - xstar)*(x_rBigm - xhat)

        transBlock.cuts.add(cut_index, cutexpr_bigm >= 0)
        transBlock_rBigm.cuts.add(cut_index, cutexpr_rBigm >= 0)
