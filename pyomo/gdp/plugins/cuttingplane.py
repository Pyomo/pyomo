# Implements cutting plane reformulation for linear, convex GDPs

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
MIPSOLVER = 'cbc'
stream_solvers = True#False
#datFile = 'stripPacking_4BigM.dat'
datFile = 'stripPacking_12Chull.dat'

#########################################################################
# first I need a model. For now I am going to use my stripPacking one and
# I'll generalize later.
#########################################################################

# Strip-packing example from http://minlp.org/library/lib.php?lib=GDP

# This model packs a set of rectangles without rotation or overlap within a
# strip of a given width, minimizing the length of the strip.

# model = AbstractModel()

# model.RECTANGLES = Set(ordered=True)

# # height and width of each rectangle
# model.Heights = Param(model.RECTANGLES)
# model.Lengths = Param(model.RECTANGLES)

# # width of strip
# model.StripWidth = Param()

# # upperbound on length (default is sum of lengths of rectangles)
# def sumLengths(model):
#     return sum(model.Lengths[i] for i in model.RECTANGLES)
# model.LengthUB = Param(initialize=sumLengths)

# # rectangle relations
# model.RecRelations = Set(initialize=['LeftOf', 'RightOf', 'Above', 'Below'])

# # x and y coordinates of each of the rectangles
# model.x = Var(model.RECTANGLES, bounds=(0, model.LengthUB))
# model.y = Var(model.RECTANGLES, bounds=(0, model.StripWidth))

# # length of strip (this will be the objective)
# model.Lt = Var(within=NonNegativeReals)

# # generate the list of possible rectangle conflicts (which are any pair)
# def rec_pairs_filter(model, i, j):
#     return i < j
# model.RectanglePairs = Set(initialize=model.RECTANGLES * model.RECTANGLES,
#     dimen=2, filter=rec_pairs_filter)

# # strip length constraint
# def strip_ends_after_last_rec_rule(model, i):
#     return model.Lt >= model.x[i] + model.Lengths[i]
# model.strip_ends_after_last_rec = Constraint(model.RECTANGLES,
#     rule=strip_ends_after_last_rec_rule)

# # constraints to prevent rectangles from going off strip
# def no_recs_off_end_rule(model, i):
#     return 0 <= model.x[i] <= model.LengthUB - model.Lengths[i]
# model.no_recs_off_end = Constraint(model.RECTANGLES, rule=no_recs_off_end_rule)

# def no_recs_off_bottom_rule(model, i):
#     return model.Heights[i] <= model.y[i] <= model.StripWidth
# model.no_recs_off_bottom = Constraint(model.RECTANGLES,
#     rule=no_recs_off_bottom_rule)

# # Disjunctions to prevent overlap between rectangles
# def no_overlap_disjunct_rule(disjunct, i, j, recRelation):
#     model = disjunct.model()
#     # left
#     if recRelation == 'LeftOf':
#         disjunct.c = Constraint(expr=model.x[i] + model.Lengths[i] <= model.x[j])
#     # right
#     elif recRelation == 'RightOf':
#         disjunct.c = Constraint(expr=model.x[j] + model.Lengths[j] <= model.x[i])
#     # above
#     elif recRelation == 'Above':
#         disjunct.c = Constraint(expr=model.y[i] - model.Heights[i] >= model.y[j])
#     # below
#     elif recRelation == 'Below':
#         disjunct.c = Constraint(expr=model.y[j] - model.Heights[j] >= model.y[i])
#     else:
#         raise RuntimeError("Unrecognized rectangle relationship: %s" % recRelation)
# model.no_overlap_disjunct = Disjunct(model.RectanglePairs, model.RecRelations,
#     rule=no_overlap_disjunct_rule)

# def no_overlap(model, i, j):
#     return [model.no_overlap_disjunct[i, j, direction] for direction in model.RecRelations]
# model.disj = Disjunction(model.RectanglePairs, rule=no_overlap)

# # minimize length
# model.length = Objective(expr=model.Lt)


# # put data in the model
# instance = model.create_instance(datFile)


#######################################################################
# Now we have a model instance, this is the real beginning.
######################################################################

class CuttingPlane_Transformation(Transformation):
    
    #TODO: I just made this up...
    alias('gdp.cuttingplane', doc="Relaxes a linear disjunctive model by adding some cuts from convex hull to Big-M relaxation.")

    def __init__(self):
        super(CuttingPlane_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        # generate bigM and chull relaxations
        bigMRelaxation = TransformationFactory('gdp.bigm')
        chullRelaxation = TransformationFactory('gdp.chull')

        instance_bigm = bigMRelaxation.create_using(instance)
        instance_chull = chullRelaxation.create_using(instance)

        # relax integrality for both bigm and chull
        relaxIntegrality = TransformationFactory('core.relax_integrality')

        instance_rBigm = relaxIntegrality.create_using(instance_bigm)
        instance_rChull = relaxIntegrality.create_using(instance_chull)

        opt = SolverFactory(SOLVER)

        improving = True
        iteration = 0
        # this might be overkill, but I'm just making this up for the moment...
        prev_obj = float("inf")
        # TODO: I made up this number and I have no idea what I am doing...
        epsilon = 0.001

        while (improving):
            # solve rBigm
            results = opt.solve(instance_rBigm, tee=stream_solvers)
            # There is only one active objective, so we can pull it out this way:
            obj_name = instance_rBigm.component_objects(Objective, 
                                                        active=True).next()
            rBigm_obj = getattr(instance_rBigm, str(obj_name))
            rBigm_objVal = rBigm_obj.expr.value

            sep_name = "instance_rChull"

            # Build objective expression for separation problem and save x* as 
            # a dictionary (variable name and index as key)
            obj_expr = 0
            x_star = {}
            for v in instance_rBigm.component_objects(Var, active=True):
                var_name = str(v)
                # we don't want the indicator variables
                if not var_name.startswith("_gdp_relax_bigm."):
                    varobject = getattr(instance_rBigm, var_name)
                    sep_var = getattr(instance_rChull, var_name)
                    for index in varobject:
                        soln_value = varobject[index].value
                        x_star[var_name + "[" + str(index) + "]"] = soln_value
                        obj_expr += (sep_var[index] - soln_value)**2

            # get objective
            obj_name = instance_rChull.component_objects(Objective, 
                                                         active=True).next()
            rChull_obj = getattr(instance_rChull, str(obj_name))
            rChull_obj.set_value(expr=obj_expr)

            # solve separation problem to get xhat.
            opt.solve(instance_rChull, tee=stream_solvers)

            # add cut to BM and rBM
            print "Adding cut" + str(iteration) + " to BM model"
            cutexpr_bigm = 0
            cutexpr_rBigm = 0
            for v in instance_rBigm.component_objects(Var, active=True):
                var_name = str(v)
                # if it's not an indicator variable
                if not var_name.startswith("_gdp_relax_bigm."):
                    rBigm_var = getattr(instance_rBigm, var_name)
                    bigm_var = getattr(instance_bigm, var_name)
                    xhat_var = getattr(instance_rChull, var_name)
                    for index in xhat_var:
                        xhat_val = xhat_var[index].value
                        norm_vec_val = xhat_val - x_star[var_name + "[" + \
                                                         str(index) + "]"]
                        cutexpr_bigm += norm_vec_val*(bigm_var[index] - xhat_val)
                        cutexpr_rBigm += norm_vec_val*(rBigm_var[index] - xhat_val)

            instance_bigm.add_component("_cut" + str(iteration), 
                                        Constraint(expr=cutexpr_bigm >= 0))
            instance_rBigm.add_component("_cut" + str(iteration), 
                                         Constraint(expr=cutexpr_rBigm >= 0))

            # TODO: What IS "enough"? That's got to depend the problem, right?
            improving = prev_obj - rBigm_objVal > epsilon
            # DEBUG
            print "prev_obj: " + str(prev_obj)
            print "rBigm_objVal: " + str(rBigm_objVal)
            print "prev_obj - rBigm_objVal: " + str(prev_obj - rBigm_objVal)
            prev_obj = rBigm_objVal
            iteration += 1

        # Last, we send off the bigm + cuts model to a MIP solver
        print "Solving MIP"
        mip_opt = SolverFactory(MIPSOLVER)
        mip_opt.solve(instance_bigm, tee=stream_solvers)

        pdb.set_trace()
