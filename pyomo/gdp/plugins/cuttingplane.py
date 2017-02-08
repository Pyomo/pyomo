# Implements cutting plane reformulation for linear, convex GDPs
from __future__ import division

from pyomo.core import *
from pyomo.gdp import *
from pyomo.opt import SolverFactory
from pyomo.util.plugin import alias
from pyomo.core.base import Transformation

from pyomo.environ import *

# DEBUG
import pdb

# do I have other options that won't be mad about the quadratic objective in the
# separation problem?
SOLVER = 'ipopt'
#MIPSOLVER = 'cbc'
stream_solvers = False

class CuttingPlane_Transformation(Transformation):
    
    #TODO: I just made this up...
    alias('gdp.cuttingplane', doc="Relaxes a linear disjunctive model by adding cuts from convex hull to Big-M relaxation.")

    def __init__(self):
        super(CuttingPlane_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        # generate bigM and chull relaxations
        bigMRelaxation = TransformationFactory('gdp.bigm')
        chullRelaxation = TransformationFactory('gdp.chull')
        relaxIntegrality = TransformationFactory('core.relax_integrality')

        instance_rChull = chullRelaxation.create_using(instance)
        relaxIntegrality.apply_to(instance_rChull)

        bigMRelaxation.apply_to(instance)
        instance_rBigm = relaxIntegrality.create_using(instance)

        opt = SolverFactory(SOLVER)

        improving = True
        iteration = 0
        prev_obj = float("inf")
        # TODO: What should this be again??
        epsilon = 0.1

        for o in instance_rChull.component_data_objects(Objective):
            o.deactivate()

        # build map of components and their IDs
        v_map = {}
        for v in instance_rBigm.component_data_objects(Var, descend_into=\
                                                       (Block, Disjunct)):
            # QUESTION: Why is the key the id?
            v_map[id(v)] = (ComponentUID(v), v, len(v_map))
        instance_rChull.xstar = Param(range(len(v_map)), mutable=True)
        
        # TODO: this doesn't work yet because indicator variables in bigm and
        # chull don't have the same cuid...
        obj_expr = 0
        for cuid, v, i in itervalues(v_map):
            # TODO: this is totally wrong, but I'm skipping indicator variables
            # for now because I can't get them with their cuid.
            if str(v).startswith("_gdp_relax_bigm."): continue
            x_star = instance_rChull.xstar[i]
            x = cuid.find_component(instance_rChull)
            # this breaks when we get to the indicator variables:
            # but if the indicator variables had the same cuid in both models,
            # it wouldn't.
            obj_expr += (x - x_star)**2
        instance_rChull.separation_objective = Objective(
            expr=obj_expr)

        while (improving):
            # solve rBigm, solution is xstar
            opt.solve(instance_rBigm, tee=stream_solvers)

            rBigm_objVal = value(list(instance_rBigm.component_data_objects(Objective))[0])

            # copy over xstar
            for cuid, v, i in itervalues(v_map):
                # TODO: same thing to avoid indicator var problem for the moment
                if str(v).startswith("_gdp_relax_bigm."): continue
                instance_rChull.xstar[i] = value(v)

            # Build objective expression for separation problem and save x* as 
            # a dictionary (variable name and index as key)
            # obj_expr = 0
            # x_star = {}
            # for v in instance_rBigm.component_objects(Var, active=True):
            #     var_name = str(v)
            #     # we don't want the indicator variables
            #     if not var_name.startswith("_gdp_relax_bigm."):
            #         varobject = getattr(instance_rBigm, var_name)
            #         sep_var = getattr(instance_rChull, var_name)
            #         for index in varobject:
            #             soln_value = varobject[index].value
            #             x_star[var_name + "[" + str(index) + "]"] = soln_value
            #             obj_expr += (sep_var[index] - soln_value)**2

            # get objective
            # obj_name = instance_rChull.component_objects(Objective, 
            #                                              active=True).next()
            # rChull_obj = getattr(instance_rChull, str(obj_name))
            # rChull_obj.set_value(expr=obj_expr)

            # solve separation problem to get xhat.
            opt.solve(instance_rChull, tee=stream_solvers)

            # add cut to BM and rBM
            print "Adding cut" + str(iteration) + " to BM model"
            cutexpr_bigm = 0
            cutexpr_rBigm = 0
            #for v in instance_rBigm.component_objects(Var, active=True):
            for vid, (cuid, v, i) in v_map.iteritems():
                # var_name = str(v)
                # TODO: same terrible thing for now:
                if str(v).startswith("_gdp_relax_bigm."): continue
                xhat = cuid.find_component(instance_rChull).value
                xstar = instance_rChull.xstar[i].value
                x_bigm = cuid.find_component(instance)
                x_rBigm = cuid.find_component(instance_rBigm)
                cutexpr_bigm += (xhat - xstar)*(x_bigm - xhat)
                cutexpr_rBigm += (xhat - xstar)*(x_rBigm - xhat)
                # rBigm_var = getattr(instance_rBigm, var_name)
                # bigm_var = getattr(instance, var_name)
                # xhat_var = getattr(instance_rChull, var_name)
                # for index in xhat_var:
                #     xhat_val = xhat_var[index].value
                #     norm_vec_val = xhat_val - x_star[var_name + "[" + \
                #                                      str(index) + "]"]
                #     cutexpr_bigm += norm_vec_val*(bigm_var[index] - xhat_val)
                #     cutexpr_rBigm += norm_vec_val*(rBigm_var[index] - xhat_val)

            instance.add_component("_cut" + str(iteration), 
                                        Constraint(expr=cutexpr_bigm >= 0))
            instance_rBigm.add_component("_cut" + str(iteration), 
                                         Constraint(expr=cutexpr_rBigm >= 0))

            # decide whether or not to keep going: check absolute difference close to 0,
            # relative difference further from 0.
            # TODO: is this right?
            obj_diff = prev_obj - rBigm_objVal
            if abs(obj_diff) < 1:
                improving = abs(obj_diff) > epsilon
            else:
                improving = abs(obj_diff/prev_obj) > epsilon
            # DEBUG
            print "prev_obj: " + str(prev_obj)
            print "rBigm_objVal: " + str(rBigm_objVal)
            print "prev_obj - rBigm_objVal: " + str(prev_obj - rBigm_objVal)
            prev_obj = rBigm_objVal
            iteration += 1

        # Last, we send off the bigm + cuts model to a MIP solver
        # But pyomo does this for us now...
        # print "Solving MIP"
        # mip_opt = SolverFactory(MIPSOLVER)
        # mip_opt.solve(instance, tee=stream_solvers)
