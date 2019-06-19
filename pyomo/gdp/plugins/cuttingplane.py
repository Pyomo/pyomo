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
from pyomo.core.base.component import ComponentUID
from pyomo.core.expr.current import identify_variables
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
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

        (instance_rBigM, instance_rCHull, var_info, var_map,
         disaggregated_var_info, rBigM_linear_constraints, 
         transBlockName) = self._setup_subproblems(
             instance, bigM)

        self._generate_cuttingplanes( instance, instance_rBigM, instance_rCHull,
                                      var_info, var_map, disaggregated_var_info,
                                      rBigM_linear_constraints, transBlockName)


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
        # Collect all of the linear constraints that are in the rBigM
        # instance. We will need these so that we can compare what we get from
        # FME to them and make sure we aren't adding redundant constraints to
        # the model. For convenience, we will make sure they are all in the form
        # lb <= expr (so we will break equality constraints)
        #
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
            
            std_repn = generate_standard_repn(body)
            cons_dict = {'lower': cons.lower,
                         'upper': cons.upper,
                         'body': std_repn
            }
            constraints_to_add = [cons_dict]
            if cons_dict['upper'] is not None:
                # if it has both bounds
                if cons_dict['lower'] is not None:
                    # copy the constraint and flip
                    leq_side = {'lower': -cons_dict['upper'],
                                'upper': None,
                                'body': generate_standard_repn(-1.0*body)}
                    constraints_to_add.append(leq_side)
                    cons_dict['upper'] = None

                elif cons_dict['lower'] is None:
                    # just flip the constraint
                    cons_dict['lower'] = -cons_dict['upper']
                    cons_dict['upper'] = None
                    cons_dict['body'].linear_coefs = (-1.0*coef for coef in \
                                                      cons_dict['body'].\
                                                      linear_coefs)

            # we will store the cut insuring that the constant in the body is
            # 0--we move all constants to the bounds
            for cons_dict in constraints_to_add:
                constant = cons_dict['body'].constant
                if constant != 0:
                    if cons_dict['lower'] is not None:
                        cons_dict['lower'] -= constant
                    if cons_dict['upper'] is not None:
                        cons_dict['upper'] -= constant
                    cons_dict['body'].constant = 0

            rBigM_linear_constraints.extend(constraints_to_add)            

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
        # we will add a block that we will deactivate to use to store the
        # extended space cuts. We never need to solve these, but we need them to
        # be contructed for the sake of Fourier-Motzkin Elimination
        extendedSpaceCuts = transBlock_rCHull.extendedSpaceCuts = Block()
        extendedSpaceCuts.deactivate()
        extendedSpaceCuts.cuts = Constraint(Any)

        transBlock_rBigM = instance_rBigM.component(transBlockName)

        # create a map which links all disaggregated variables to their
        # originals on both bigm and rBigm. We will use this to project the cut
        # from the extended space to the space of the bigM problem.
        disaggregatedVarMap = self._get_disaggregated_var_map(instance_rCHull,
                                                              instance,
                                                              instance_rBigM)

        #
        # Generate the mapping between the variables on all the
        # instances and the xstar parameter.
        #
        var_info = tuple(
            (v,
             transBlock_rBigM.all_vars[i],
             transBlock_rCHull.all_vars[i],
             transBlock_rCHull.xstar[i])
            for i,v in enumerate(transBlock.all_vars))

        # TODO: I don't know a better way to do this
        disaggregated_var_info = tuple(
            (v,
             disaggregatedVarMap[v]['bigm'],
             disaggregatedVarMap[v]['rBigm'])
            for v in disaggregatedVarMap.keys())

        # this is the map that I need to translate my projected cuts and add
        # them to bigM and rBigM.
        # [ESJ 5 March 2019] TODO: If I add xstar to this (or don't) can I just
        # replace var_info?
        var_map = ComponentMap((transBlock_rCHull.all_vars[i],
                                {'bigM': v,
                                 'rBigM': transBlock_rBigM.all_vars[i]})
                               for i,v in enumerate(transBlock.all_vars))

        #
        # Add the separation objective to the chull subproblem
        #
        self._add_separation_objective(var_info, transBlock_rCHull)

        return (instance_rBigM, instance_rCHull, var_info, var_map,
                disaggregated_var_info, rBigM_linear_constraints,
                transBlockName)

    def _get_disaggregated_var_map(self, chull, bigm, rBigm):
        disaggregatedVarMap = ComponentMap()
        # TODO: I guess technically I don't know that the transformation block
        # is named this... It could have a unique name, so I need to hunt that
        # down. (And then test that I do that correctly)
        for disjunct in chull._pyomo_gdp_chull_relaxation.relaxedDisjuncts.\
            values():
            for disaggregated_var, original in \
            disjunct._gdp_transformation_info['srcVars'].iteritems():
                orig_vars = disaggregatedVarMap.get(disaggregated_var)
                if orig_vars is None:
                    # TODO: this is probably expensive, but I don't have another
                    # idea... And I am only going to do it once
                    orig_cuid = ComponentUID(original)
                    disaggregatedVarMap[disaggregated_var] = \
                                    {'bigm': orig_cuid.find_component(bigm),
                                     'rBigm': orig_cuid.find_component(rBigm)}

        return disaggregatedVarMap

    def _generate_cuttingplanes( self, instance, instance_rBigM,
                                 instance_rCHull, var_info, var_map,
                                 disaggregated_var_info,
                                 rBigM_linear_constraints, transBlockName):

        opt = SolverFactory(self._config.solver)
        stream_solver = self._config.stream_solver
        opt.options = self._config.solver_options

        improving = True
        prev_obj = float("inf")
        epsilon = self._config.EPS
        cuts = None

        transBlock = instance.component(transBlockName)
        transBlock_rBigM = instance_rBigM.component(transBlockName)
        transBlock_rCHull = instance_rCHull.component(transBlockName)

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
            # print("x* is")
            for x_bigm, x_rbigm, x_chull, x_star in var_info:
                x_star.value = x_rbigm.value
                # initialize the X values
                x_chull.value = x_rbigm.value
                # DEBUG
                #print("\t%s: %s" % (x_rbigm.name, x_star.value))

            # compare objectives: check absolute difference close to 0, relative
            # difference further from 0.
            obj_diff = prev_obj - rBigM_objVal
            improving = math.isinf(obj_diff) or \
                        ( abs(obj_diff) > epsilon if abs(rBigM_objVal) < 1 else
                          abs(obj_diff/prev_obj) > epsilon )

            # solve separation problem to get xhat.
            opt.solve(instance_rCHull, tee=stream_solver)
            # DEBUG
            #print("x_hat is")
            # for x_hat in rCHull_vars:
            #    print("\t%s: %s" % (x_hat.name, x_hat.value))
            # print "Separation obj = %s" % (
            #    value(next(instance_rCHull.component_data_objects(
            #    Objective, active=True))),)

            # [JDS 19 Dec 18] Note: we should check that the separation
            # objective was significantly nonzero.  If it is too close
            # to zero, either the rBigM solution was in the convex hull,
            # or the separation vector is so close to zero that the
            # resulting cut is likely to have numerical issues.
            if abs(value(transBlock_rCHull.separation_objective)) < epsilon:
                # [ESJ 15 Feb 19] I think we just want to quit right, we're
                # going nowhere...?
                break

            cuts = self._create_cuts(var_info, var_map, disaggregated_var_info,
                                     rCHull_vars, instance_rCHull,
                                     rBigM_linear_constraints, transBlock,
                                     transBlock_rBigM, transBlock_rCHull)
           
            # We are done if the cut generator couldn't return a valid cut
            if not cuts:
                break

            # add cut to rBigm
            for cut in cuts['rBigM']:
                transBlock_rBigM.cuts.add(len(transBlock_rBigM.cuts), cut)

            # DEBUG
            #print("adding this cut to rBigM:\n%s <= 0" % cuts['rBigM'])

            if improving:
                for cut in cuts['bigM']:
                    cut_number = len(transBlock.cuts)
                    logger.warning("GDP.cuttingplane: Adding cut %s to BM model."
                                   % (cut_number,))
                    transBlock.cuts.add(cut_number, cut)

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


    def _create_cuts(self, var_info, var_map, disaggregated_var_info,
                     rCHull_vars, instance_rCHull, rBigM_linear_constraints,
                     transBlock, transBlock_rBigm, transBlock_rCHull):
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
        tight_constraints = []
        for constraint in instance_rCHull.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                sort=SortComponents.deterministic):
            # DEBUG
            #print "   CON: ", constraint.expr
            multiplier = self.constraint_tight(instance_rCHull, constraint)
            if multiplier:
                # DEBUG
                # print(constraint.name)
                # print constraint.expr
                # get normal vector to tangent plane to this constraint at xhat
                # print "      TIGHT", multiplier
                f = constraint.body
                firstDerivs = differentiate(f, wrt_list=rCHull_vars)
                #print "     ", firstDerivs
                normal_vec = [multiplier*value(_) for _ in firstDerivs]
                normal_vectors.append(normal_vec)
                # check if constraint is linear
                if f.polynomial_degree() == 1:
                    tight_constraints.append(
                        self.get_linear_constraint_repn(constraint))
                else: 
                    # we will use the linear approximation of this constraint at
                    # x_hat
                    tight_constraints.append(
                        self.get_linear_approximation_repn(normal_vec,
                                                           rCHull_vars))

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
        # for x,v in composite_normal_map.iteritems():
        #     print(x.name + '\t' + str(v))

        
        composite_cutexpr_CHull = 0
        for x_bigm, x_rbigm, x_chull, x_star in var_info:
            # make the cut in the CHull space with the CHull variables. We will
            # translate it all to BigM and rBigM later when we have projected
            # out the disaggregated variables
            composite_cutexpr_CHull += composite_normal_map[x_chull]*\
                                       (x_chull - x_chull.value)
            # DEBUG:
            # print("%s\t%s" %
            #       (composite_normal[x_chull], x_star.value - x_chull.value))

        # I am going to expand the composite_cutexprs to be in the extended space
        vars_to_eliminate = ComponentSet()
        do_fme = False
        for x_disaggregated, x_orig_bigm, x_orig_rBigm in disaggregated_var_info:
            normal_vec_component = composite_normal_map[x_disaggregated]
            composite_cutexpr_CHull += normal_vec_component*\
                                       (x_disaggregated - x_disaggregated.value)
            vars_to_eliminate.add(x_disaggregated)
            # check that at least one disaggregated variable appears in the
            # constraint. Else we don't need to do FME
            if not do_fme and normal_vec_component != 0:
                do_fme = True
    
        #print("The cut in extended space is: %s <= 0" % composite_cutexpr_CHull)
        cut_std_repn = generate_standard_repn(composite_cutexpr_CHull)
        cut_cons = {'lower': None, 
                    'upper': 0, 
                    'body': ComponentMap(zip(cut_std_repn.linear_vars,
                                             cut_std_repn.linear_coefs)),
                    'key_order': list(cut_std_repn.linear_vars)}
        cut_cons['body'][None] = value(cut_std_repn.constant)
        cut_cons['key_order'].append(None)
        tight_constraints.append(cut_cons)
        
        if do_fme:
            projected_constraints = self.fourier_motzkin_elimination(
                tight_constraints, vars_to_eliminate)
        else:
            projected_constraints = [cut_cons]

        # DEBUG:
        # print("These are the constraints we got from FME:")
        # for cons in projected_constraints:
        #     body = 0
        #     # We make sure that this loop happens in a deterministic order so
        #     # that the expression we produce is the same every time
        #     for var in cons['key_order']:
        #         val = cons['body'][var]
        #         body += val*var if var is not None else val
        #     print("\t%s <= %s <= %s" % (cons['lower'], body, cons['upper']))

        # we created these constraints with the variables from rCHull. We
        # actually need constraints for BigM and rBigM now!
        cuts = self.get_constraint_exprs(projected_constraints, var_map)

        # We likely have some cuts that duplicate other constraints now. We will
        # filter them to make sure that they do in fact cut off x* and that they
        # are not already in the BigM relaxation.
        print("The length to start is %s" % len(cuts['rBigM']))
        for i in sorted(range(len(cuts['rBigM'])), reverse=True):
            cut = cuts['rBigM'][i]
            # x* is still in rBigM, so we can just remove this constraint if it
            # is satisfied at x*
            if value(cut):
                del cuts['rBigM'][i]
                del cuts['bigM'][i]
                print("removed %s for being silly" % i)
                continue
            unique = True
            print("checking:")
            print(cut)
            # check that we don't already have this constraint in the model: 
            # I know that the constraints is LB <= expr, and we have put the
            # constraints in rBigM_linear_constraints in that form too.
            assert cut.nargs() == 2
            lb = cut.arg(0)
            cut_repn = generate_standard_repn(cut.arg(1))
            if cut_repn.constant != 0:
                lb -= cut_repn.constant
                cut_repn.constant = 0
            for cons in rBigM_linear_constraints:
                if i == 5:
                    set_trace()
                if lb == cons['lower'] and self.standard_repn_equals(
                        cut_repn, cons['body']):
                    del cuts['rBigM'][i]
                    del cuts['bigM'][i]
                    unique = False
                    print("removing %s because we already had it" % i)
                    break
            # if unique:
            #     # we have found a constraint which cuts of x* and is not already
            #     # in rBigM, this has to be out cut and we can stop.
            #     cuts['rBigM'] = [cuts['rBigM'][i]]
            #     cuts['bigM'] = [cuts['bigM'][i]]
            #     break

        assert len(cuts['rBigM']) == 1

        return(cuts)

    def get_constraint_exprs(self, constraints, var_map):
        #print("==========================\nBuilding actual expressions")
        cuts = {}
        cuts['rBigM'] = []
        cuts['bigM'] = []
        for cons in constraints:
            # DEBUG
            #print("cons:")
            body = 0
            for var in cons['key_order']:
                val = cons['body'][var]
                body += val*var if var is not None else val
            #print("\t%s <= %s <= %s" % (cons['lower'], body, cons['upper']))
            body_bigM = 0
            body_rBigM = 0
            # Check if this constraint actually has a body. If not, we don't
            # want it anyway--it's just a trivial thing coming out of FME
            trivial_constraint = True
            for var in cons['key_order']:
                coef = cons['body'][var]
                if var is None:
                    body_bigM += coef
                    body_rBigM += coef
                    continue
                # TODO: do I want almost equal here? I'm going to crash if I get
                # one of the disaggagregated variables... In case it didn't
                # quite cancel?
                if coef != 0:
                    body_bigM += coef*var_map[var]['bigM']
                    body_rBigM += coef*var_map[var]['rBigM']
                    trivial_constraint = False
            if trivial_constraint:
                continue
            if cons['lower'] is not None:
                cuts['rBigM'].append(cons['lower'] <= body_rBigM)
                cuts['bigM'].append(cons['lower'] <= body_bigM)
            elif cons['upper'] is not None:
                cuts['rBigM'].append(cons['upper'] >= body_rBigM)
                cuts['bigM'].append(cons['upper'] >= body_bigM)
        return cuts


    # assumes that constraints is a list of my own linear constraint repn (see
    # below)
    def fourier_motzkin_elimination(self, constraints, vars_to_eliminate):
        # First we will preprocess so that we have no equalities (break them
        # into two constraints). We will also make everything a geq constraint
        # to make life easier later.
        tmpConstraints = [cons for cons in constraints]
        for cons in tmpConstraints:
            if cons['lower'] is not None and cons['upper'] is not None:
                # make a copy to become the geq side 
                geq = {'lower': -cons['upper'],
                       'upper': None,
                       # I'm doing this so that I have a copy, not a reference:
                       'body': ComponentMap(
                           (var, -coef) for (var, coef) in cons['body'].items()),
                       'key_order': cons['key_order'] # this is a reference, but
                                                      # it's fine, I won't
                                                      # change it.
                }
                cons['upper'] = None
                constraints.append(geq)
            elif cons['upper'] is not None:
                constraints.remove(cons)
                constraints.append(
                    self.scalar_multiply_linear_constraint(cons, -1))

        vars_that_appear = ComponentSet()
        for cons in constraints:
            body = 0
            for var, val in cons['body'].items():
                #body += val*var if var is not None else val
                # We only need to eliminate variables that actually appear in
                # this set of constraints
                if var in vars_to_eliminate:
                    vars_that_appear.add(var)
            #print("\t%s <= %s <= %s" % (cons['lower'], body, cons['upper']))

        # we are done preprocessing, now we can actually do the recursion
        while vars_that_appear:
            the_var = vars_that_appear.pop()
            #print("DEBUG: we are eliminating %s" % the_var.name) 

            # we are 'reorganizing' the constraints, we will map the coefficient
            # of the_var from that constraint and the rest of the expression and
            # sorting based on whether we have the_var <= other stuff or vice
            # versa.
            leq_list = []
            geq_list = []
            waiting_list = []

            # sort our constraints, make it so leq constraints have coef of -1
            # on variable to eliminate, geq constraints have coef of 1 (so we
            # can add them) 
            # DEBUG 
            # print("CONSTRAINTS:")
            while(constraints):
                cons = constraints.pop()
                #for cons in constraints:
                # DEBUG
                # body = 0
                # for var, val in cons['body'].items():
                #     body += val*var if var is not None else val
                #print("\t%s <= %s <= %s" % (cons['lower'], body, cons['upper']))

                leaving_var_coef = cons['body'].get(the_var)
                if leaving_var_coef is None or leaving_var_coef == 0:
                    waiting_list.append(cons)
                    continue

                # at this point, we know that the constraint is a geq constraint
                assert cons['upper'] is None

                # NOTE: neither of the scalar multiplications below flip the
                # constraint. So we are sure to have only geq constraints
                # forever, which is exactly what we want.
                if leaving_var_coef < 0:
                    leq_list.append(self.scalar_multiply_linear_constraint(
                        cons, -1.0/leaving_var_coef))
                else:
                    geq_list.append(self.scalar_multiply_linear_constraint(
                        cons, 1.0/leaving_var_coef))

            # #print("Here be leq constraints:")
            # for cons in leq_list:
            #     body = 0
            #     for var, val in cons['body'].items():
            #         body += val*var if var is not None else val
            #     print("\t%s <= %s <= %s" % (cons['lower'], body, cons['upper']))

            # #print("Here be geq constraints:")
            # for cons in geq_list:
            #     body = 0
            #     for var, val in cons['body'].items():
            #         body += val*var if var is not None else val
            #     print("\t%s <= %s <= %s" % (cons['lower'], body, cons['upper']))

            for leq in leq_list:
                for geq in geq_list:
                    constraints.append(self.add_linear_constraints(leq, geq))

            # add back in the constraints that didn't have the variable we were
            # projecting out
            constraints.extend(waiting_list)

            #print("This is what we have now:")
            # for cons in constraints:
            #     body = 0
            #     for var, val in cons['body'].items():
            #         body += val*var if var is not None else val
                #print("\t%s <= %s <= %s" % (cons['lower'], body, cons['upper']))
            #return self.fm_elimination(constraints, vars_that_appear)
        return(constraints)

    # TODO: not sure this isn't already defined by == actually...
    def standard_repn_equals(self, repn1, repn2):
        if len(repn1.linear_coefs) != len(repn2.linear_coefs):
            return False
        dict1 = ComponentMap(zip(repn1.linear_vars, repn1.linear_coefs))
        dict2 = ComponentMap(zip(repn2.linear_vars, repn2.linear_coefs))
        for v, coef1 in dict1.items():
            coef2 = dict2.get(v)
            if coef2 is None:
                return False
            if coef1 != coef2:
                return False
        print("found equality!")
        return True
            
    def constraint_tight(self, model, constraint):
        val = value(constraint.body)
        ans = 0
        #print "    vals:", value(constraint.lower), val, value(constraint.upper)
        if constraint.lower is not None:
            if value(constraint.lower) >= val:
                # tight or in violation of LB
                ans -= 1

        if constraint.upper is not None:
            if value(constraint.upper) <= val:
                # tight or in violation of UB
                ans += 1

        return ans

    def get_linear_constraint_repn(self, cons):
        std_repn = generate_standard_repn(cons.body)
        cons_dict = {}
        cons_dict['lower'] = value(cons.lower)
        cons_dict['upper'] = value(cons.upper)
        cons_dict['body'] = ComponentMap(
            zip(std_repn.linear_vars, std_repn.linear_coefs))
        cons_dict['body'][None] = value(std_repn.constant)
        cons_dict['key_order'] = list(std_repn.linear_vars)
        cons_dict['key_order'].append(None)

        return cons_dict

    def get_linear_approximation_repn(self, normal_vec, point):
        # the constraint is normal_vec^T(point - point.values) <= 0
        cons_dict = {}
        cons_dict['lower'] = None
        cons_dict['upper'] = sum(normal_vec[idx]*v.value 
                                 for (idx, v) in enumerate(point))
        cons_dict['body'] = ComponentMap(zip(point, normal_vec))
        cons_dict['body'][None] = 0
        # need a copy of this, not a reference
        cons_dict['key_order'] = [v for v in point]
        cons_dict['key_order'].append(None)
        
        return cons_dict

    def add_linear_constraints(self, cons1, cons2):
        # creating a list of variables so we will have a deterministic ordering

        # This list is just a hack to have a version which does not have None in
        # it because we can't compare to it.
        cons1_vars = [var for var in cons1['key_order'] if var is not None]
        var_list = [var for var in cons1['key_order']] + \
                   [var for var in cons2['key_order'] if var is not None and
                    var not in cons1_vars]
        var_list.append(None)
        ans = {'lower': None, 'upper': None, 'body': ComponentMap(), 
               'key_order': var_list}
        all_vars = cons1['body'].items() + \
                   list(ComponentSet(cons2['body'].items()) - \
                        ComponentSet(cons1['body'].items()))
        for (var, coef) in all_vars:
            if var is None:
                ans['body'][None] = cons1['body'][None] + cons2['body'][None]
                continue
            #print var.name
            cons2_coef = cons2['body'].get(var)
            cons1_coef = cons1['body'].get(var)
            if cons2_coef is not None and cons1_coef is not None:
                ans['body'][var] = cons1_coef + cons2_coef
            elif cons1_coef is not None:
                ans['body'][var] = cons1_coef
            elif cons2_coef is not None:
                ans['body'][var] = cons2_coef

        bounds_good = False
        cons1_lower = cons1['lower']
        cons2_lower = cons2['lower']
        if cons1_lower is not None and cons2_lower is not None:
            ans['lower'] = cons1_lower + cons2_lower
            bounds_good = True

        cons1_upper = cons1['upper']
        cons2_upper = cons2['upper']
        if cons1_upper is not None and cons2_upper is not None:
            ans['upper'] = cons1_upper + cons2_upper
            bounds_good = True

        # in all other cases we don't actually want to add these constraints... I
        # mean, what we would actually do is multiply one of them be negative
        # one and then do it... But I guess I want to assume that I already did
        # that because in the context of FME, I already did
        if not bounds_good:
            raise RuntimeError("Trying to add a leq and geq constraint!")

        return ans

    def scalar_multiply_linear_constraint(self, cons, scalar):
        for var, coef in cons['body'].items():
            cons['body'][var] = coef*scalar

        if scalar >= 0:
            if cons['lower'] is not None:
                cons['lower'] *= scalar
            if cons['upper'] is not None:
                cons['upper'] *= scalar
        else:
            # we have to flip the constraint
            if cons['lower'] is not None:
                # TODO: This case can actually never happen right now because I
                # am preprocessing all the constraints in FME. But in general
                # you would need this. (This is true in a few other places too,
                # I don't know if I should remove those cases or leave them.)
                tmp_upper = cons['upper']
                cons['upper'] = scalar*cons['lower']
                cons['lower'] = None
                if tmp_upper is not None:
                    cons['lower'] = scalar*tmp_upper

            elif cons['upper'] is not None:
                tmp_upper = cons['upper']
                # we actually know that lower is None
                cons['upper'] = None
                cons['lower'] = scalar*tmp_upper

        return cons

