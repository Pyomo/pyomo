import pyomo.environ
from pyomo.core import *
from pyomo.gdp import *

'''Problem from TODO
We are minimizing the cost of a design of a plant with parallel processing units and storage tanks
in between. We decide the number and volume of units, and the volume and location of the storage
tanks. The problem is convexified and has nonlinear objective and global constraints'''

model = AbstractModel()

# TODO: it looks like they set a bigM for each j. Which I need to look up how to do...
model.BigM = Suffix(direction=Suffix.LOCAL)
model.BigM[None] = 1000

##########
# Sets
##########

model.PRODUCTS = Set()
model.STAGES = Set()

# TODO: these aren't in the formulation??
#model.STORAGE_TANKS = Set()


###############
# Parameters
###############

model.HorizonTime = Param()
# alpha1
model.Alpha1 = Param()
model.Alpha2 = Param()
# TODO: this is about to be a mess. A bunch of these are hard-coded into the GAMS model...
model.Beta1 = Param()
model.Beta2 = Param()

model.ProductionAmount = Param(model.PRODUCTS)
model.ProductSizeFactor = Param(model.PRODUCTS, model.STAGES)
model.ProcessingTime = Param(model.PRODUCTS, model.STAGES)

# TODO: I'm still missing size factor indexed just by stages and size factor for intermediate storage

# TODO: bounds


################
# Variables
################

# TODO: right now these match the formulation. There are more in GAMS...

# unit size of stage j
model.volume = Var(model.STAGES)
# TODO: GAMS has a batch size indexed just by products that isn't in the formulation... I'm going
# to try to avoid it for the moment...
# batch size of product i at stage j
model.batchSize = Var(model.PRODUCTS, model.STAGES)
# TODO: this is different in GAMS... They index by stages too?
# cycle time of product i divided by batch size of product i
model.cycleTime = Var(model.PRODUCTS)
# number of units in parallel out-of-phase (or in phase) at stage j
model.unitsOutOfPhase = Var(model.STAGES)
model.unitsInPhase = Var(model.STAGES)
# TODO: what are we going to do as a boundary condition here? For that last stage?
# size of intermediate storage tank between stage j and j+1
model.storageTankSize = Var(model.STAGES)

# variables for convexified problem
model.volume_log = Var(model.STAGES)
model.batchSize_log = Var(model.PRODUCTS, model.STAGES)
model.cycleTime_log = Var(model.PRODUCTS)
model.unitsOutOfPhase_log = Var(model.STAGES)
model.unitsInPhase_log = Var(model.STAGES)
model.storageTankSize_log = Var(model.STAGES)

###############
# Objective
###############

# TODO: is there a pretty way to do the boundary condition? I just need the last sum to be over
# model.STAGES minus the last stage
def get_cost_rule(model):
    return model.Alpha1 * sum(exp(model.unitsInPhase_log[j] + model.unitsOutOfPhase_log[j] + \
                                          model.Beta1 * model.volume_log[j]) for j in model.STAGES) +\
        model.Alpha2 * sum(exp(model.Beta2 * model.storageTankSize_log[j]) for j in model.STAGES)
model.min_cost = Objective(rule=get_cost_rule)


##############
# Constraints
##############

def processing_capacity_rule(model, j, i):
    return model.volume_log[j] >= log(model.ProductSizeFactor[i, j]) + model.batchSize_log[i, j] - \
        model.unitsInPhase_log[j]
model.processing_capacity = Constraint(model.STAGES, model.PRODUCTS, rule=processing_capacity_rule)

def processing_time_rule(model, j, i):
    return model.cycleTime_log[i] >= log(model.ProcessingTime[i, j]) - model.batchSize_log[i, j] - \
        model.unitsOutOfPhase_log[j]
model.processing_time = Constraint(model.STAGES, model.PRODUCTS, rule=processing_time_rule)

def finish_in_time_rule(model):
    #TODO
