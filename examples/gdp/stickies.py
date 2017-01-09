import pyomo.environ
from pyomo.core import *
from pyomo.gdp import *

# Layout optimization for screening systems in waste paper recovery:
# Problem from http://www.minlp.org/library/problem/index.php?i=263&lib=GDP
# This problem is a design problem: When waste paper is recovered, a separator screen is used to
# separate the fiber from the stickies. Different reject rates can be set for different pieces
# of equipment to separate out the output with more fiber and the output with more stickies.
# The overall equipment layout affects the output purity and the flow rates. (We want more fiber
# and fewer stickies!)

model = AbstractModel()


model.BigM = Suffix(direction=Suffix.LOCAL)
model.BigM[None] = 7000

#######################
#Sets
#######################

# TODO: this is a set union and I think you can change that.
model.Components = Set()
model.GoodComponents = Set()
model.BadComponents = Set()
# total nodes in the system
model.Nodes = Set()
# possibe screens
# TODO: you can do this as a set union and you should, I think.
model.Screens = Set()

def screen_node_filter(model, s, n):
    return s != n
model.ScreenNodePairs = Set(initialize=model.Screens * model.Nodes, dimen=2, filter=screen_node_filter)

def screen_filter(model, s, sprime):
    return s != sprime
model.ScreenPairs = Set(initialize = model.Screens * model.Screens, dimen=2, filter=screen_filter)


######################
# Parameters
######################

# exponent coefficient for cost in screen s (alpha(s))
model.ExpScreenCostCoeff = Param(model.Screens)
# beta(s, k)
model.AcceptanceFactor = Param(model.Screens, model.Components)
# C_s^1
model.ScreenCostCoeff1 = Param(model.Screens)
# C_s^2
model.ScreenCostCoeff2 = Param(model.Screens, default=0)
# max percentage inlet stickies accepted in total flow (C_{st}^{up}, q(kb))
model.AcceptedLeftover = Param(model.BadComponents)
# F_j^0, m_src(k)
model.InitialFlow = Param(model.Components, default=0)

# bounds

# F_s^{in, lo} and F_s^{in, up}
def flow_ub_rule(model, s):
    return sum(model.InitialFlow[k] for k in model.Components)
model.FlowLB = Param(model.Screens)
model.FlowUB = Param(model.Screens, initialize=flow_ub_rule)

# Bounds on variables
model.InletComponentFlowLB = Param(model.Components, model.Nodes, default=0)
def component_flow_ub_rule(model, k, n):
    return model.InitialFlow[k]
model.InletComponentFlowUB = Param(model.Components, model.Nodes, initialize=component_flow_ub_rule)

# model.ScreenCostUB = Param(model.Screens)
# model.ScreenCostLB = Param(model.Screens, default=0)

model.RejectRateLB = Param(model.Screens)
model.RejectRateUB = Param(model.Screens)

model.RejectedComponentFlowLB = Param(model.Components, model.Screens, default=0)
def rejected_component_flow_bound(model, k, s):
    return model.InitialFlow[k]*(model.RejectRateUB[s]**model.AcceptanceFactor[s, k])
model.RejectedComponentFlowUB = Param(model.Components, model.Screens, initialize=rejected_component_flow_bound)

model.AcceptedComponentFlowLB = Param(model.Components, model.Screens, default=0)
def accepted_component_flow_bound(model, k, s):
    return model.InitialFlow[k]*(1 - model.RejectRateLB[s]**model.AcceptanceFactor[s, k])
model.AcceptedComponentFlowUB = Param(model.Components, model.Screens, initialize=accepted_component_flow_bound)

# constants for objective function (W^1, W^2, W^3)
model.FiberWeight = Param()
model.StickiesWeight = Param()
model.CostWeight = Param()


######################
# Variables
######################

# c_s, C(s), cost of selecting screen
# TODO: I did make trivial bounds for this, but since this problem is non-convex anyway, I guess
# we don't *need* them.
# def get_screen_cost_bounds(model, s):
#     return (model.ScreenCostLB[s], model.ScreenCostUB[s])
model.screenCost = Var(model.Screens, within=NonNegativeReals)#, bounds=get_screen_cost_bounds)

# total inlet flow into screen s (F_IN(s))
# QUESTION: In gams, the bounds are enforced all the time, I think. In the model formulation, 
# that constraint is in the disjunction. Does this matter?
def get_inlet_flow_bounds(model, s):
    return (model.FlowLB[s], model.FlowUB[s])
model.inletScreenFlow = Var(model.Screens, within=NonNegativeReals, bounds=get_inlet_flow_bounds)

# inlet flow of component j into node n, (f_{n,j}^I, M_IN
def get_inlet_component_flow_bounds(model, j, n):
    return (model.InletComponentFlowLB[j, n], model.InletComponentFlowUB[j, n])
model.inletComponentFlow = Var(model.Components, model.Nodes, within=NonNegativeReals, 
                               bounds=get_inlet_component_flow_bounds)

# accepted flow of component j from screen s
def get_accepted_component_flow_bounds(model, j, s):
    return (model.AcceptedComponentFlowLB[j, s], model.AcceptedComponentFlowUB[j, s])
model.acceptedComponentFlow = Var(model.Components, model.Screens, within=NonNegativeReals,
                                  bounds=get_accepted_component_flow_bounds)
# rejected flow of component j from screen s
def rej_component_flow_bounds(model, k, s):
    return (model.RejectedComponentFlowLB[k, s], model.RejectedComponentFlowUB[k, s])
model.rejectedComponentFlow = Var(model.Components, model.Screens, within=NonNegativeReals, 
                                  bounds=rej_component_flow_bounds)

# accepted flow of component j from screen s to node n
model.acceptedNodeFlow = Var(model.Components, model.Screens, model.Nodes, within=NonNegativeReals)
# rejected flow of component j from screen s to node n
model.rejectedNodeFlow = Var(model.Components, model.Screens, model.Nodes, within=NonNegativeReals)
# flow of component j from source to node n
model.flowFromSource = Var(model.Components, model.Nodes, within=NonNegativeReals)

# reject rate of screen s
def get_rej_rate_bounds(model, s):
    return (model.RejectRateLB[s], model.RejectRateUB[s])
model.rejectRate = Var(model.Screens, within=NonNegativeReals, bounds=get_rej_rate_bounds)


######################
# Objective
######################

def calc_cost_rule(model):
    lostFiberCost = model.FiberWeight * sum(model.inletComponentFlow[j, 'SNK'] \
                                           for j in model.GoodComponents)
    stickiesCost = model.StickiesWeight * sum(model.inletComponentFlow[j, 'PRD'] \
                                              for j in model.BadComponents)
    screenCost = model.CostWeight * sum(model.screenCost[s] for s in model.Screens)
    return lostFiberCost + stickiesCost + screenCost
model.min_cost = Objective(rule=calc_cost_rule)


######################
# Constraints
######################

def stickies_bound_rule(model, j):
    return sum(model.inletComponentFlow[j, 'PRD'] for j in model.BadComponents) <= \
        model.AcceptedLeftover[j] * model.InitialFlow[j]
model.stickies_bound = Constraint(model.BadComponents, rule=stickies_bound_rule)

def inlet_flow_rule(model, s, j):
    return model.inletComponentFlow[j, s] == model.acceptedComponentFlow[j, s] + \
        model.rejectedComponentFlow[j, s]
model.inlet_flow = Constraint(model.Screens, model.Components, rule=inlet_flow_rule)

def total_inlet_flow_rule(model, s):
    return model.inletScreenFlow[s] == sum(model.inletComponentFlow[j, s] for j in model.Components)
model.total_inlet_flow = Constraint(model.Screens, rule=total_inlet_flow_rule)

def inlet_flow_balance_rule(model, n, j):
    return model.inletComponentFlow[j, n] == model.flowFromSource[j, n] + \
        sum(model.acceptedNodeFlow[j, s, n] + model.rejectedNodeFlow[j, s, n] for s in model.Screens \
            if s != n)
model.inlet_flow_balance = Constraint(model.Nodes, model.Components, rule=inlet_flow_balance_rule)

def source_flow_rule(model, j):
    return model.InitialFlow[j] == sum(model.flowFromSource[j, n] for n in model.Nodes)
model.source_flow = Constraint(model.Components, rule=source_flow_rule)

#################
## Disjunctions
#################

def screen_disjunct_rule(disjunct, selectScreen, s):
    model = disjunct.model()
    def rejected_flow_rule(disjunct, j):
        return model.rejectedComponentFlow[j,s] == model.inletComponentFlow[j, s]*\
            (model.rejectRate[s]**model.AcceptanceFactor[s, j])

    if selectScreen:              
        # TODO: I think there is a typo in the formulation... They have f_{n,j}^I beow, but
        # GAMS has f_s, which makes more sense.
        disjunct.inlet_flow_bounds = Constraint(expr=model.FlowLB[s] <= model.inletScreenFlow[s] <= \
                                                model.FlowUB[s])
        disjunct.rejected_flow = Constraint(model.Components, rule=rejected_flow_rule)
        disjunct.screen_cost = Constraint(expr=model.screenCost[s] == \
                                          model.ScreenCostCoeff1[s]*\
                                          (model.inletScreenFlow[s]**model.ExpScreenCostCoeff[s]) + \
                                          model.ScreenCostCoeff2[s]*(1 - model.rejectRate[s]))
    else:
        disjunct.no_flow = Constraint(expr=model.inletScreenFlow[s] == 0)
        disjunct.no_cost = Constraint(expr=model.screenCost[s] == 0)
model.screen_selection_disjunct = Disjunct([0,1], model.Screens, rule=screen_disjunct_rule) 

def screen_disjunction_rule(model, s):
    return [model.screen_selection_disjunct[selectScreen, s] for selectScreen in [0,1]]
model.screen_disjunction = Disjunction(model.Screens, rule=screen_disjunction_rule) 


def accepted_flow_disjunct_rule(disjunct, s, n, acceptFlow):
    model = disjunct.model()
    def flow_balance_rule(disjunct, j):
        return model.acceptedNodeFlow[j, s, n] == model.acceptedComponentFlow[j, s]
    def no_flow_rule(disjunct, j):
        model = disjunct.model()
        return model.acceptedNodeFlow[j, s, n] == 0

    if acceptFlow:
        disjunct.flow_balance = Constraint(model.Components, rule=flow_balance_rule)
    else:
        disjunct.no_flow = Constraint(model.Components, rule=no_flow_rule)
model.flow_acceptance_disjunct = Disjunct(model.ScreenNodePairs, [0,1], rule=accepted_flow_disjunct_rule)

def flow_acceptance_disjunction_rule(model, s, n):
    return [model.flow_acceptance_disjunct[s, n, acceptFlow] for acceptFlow in [0,1]]
model.flow_acceptance_disjunction = Disjunction(model.ScreenNodePairs, rule=flow_acceptance_disjunction_rule)


def rejected_flow_disjunct_rule(disjunct, s, n, rejectFlow):
    model = disjunct.model()
    def flow_balance_rule(disjunct, j):
        return model.rejectedNodeFlow[j, s, n] == model.rejectedComponentFlow[j, s]
    def no_reject_rule(disjunct, j):
        return model.rejectedNodeFlow[j, s, n] == 0
    
    if rejectFlow:
        disjunct.flow_balance = Constraint(model.Components, rule=flow_balance_rule)
    else:
        disjunct.no_reject = Constraint(model.Components, rule=no_reject_rule)
model.flow_rejection_disjunct = Disjunct(model.ScreenNodePairs, [0, 1], rule=rejected_flow_disjunct_rule)

def rejected_flow_disjunction_rule(model, s, n):
    return [model.flow_rejection_disjunct[s, n, rejectFlow] for rejectFlow in [0, 1]]
model.flow_rejection_disjunction = Disjunction(model.ScreenNodePairs, rule=rejected_flow_disjunction_rule)


def flow_from_source_disjunct_rule(disjunct, n):
    model = disjunct.model()
    def sourceFlow_balance_rule(disjunct, j):
        return model.flowFromSource[j, n] == model.InitialFlow[j]
    def no_sourceFlow_rule(disjunct, j, nprime):
        print 'j: ' + str(j)
        print 'n: ' + str(n)
        print 'nprime: ' + str(nprime)
        #if nprime == n:
         #   return Constraint.Skip
        return model.flowFromSource[j, nprime] == 0

    disjunct.flow_balance = Constraint(model.Components, rule=sourceFlow_balance_rule)
    # TODO: the constraint below is the source of the BigM error 
    #disjunct.no_flow = Constraint(model.Components, model.Nodes-[n], rule=no_sourceFlow_rule)
model.flow_from_source_disjunct = Disjunct(model.Nodes, rule=flow_from_source_disjunct_rule)

def flow_from_source_disjunction_rule(model):
    return [model.flow_from_source_disjunct[n] for n in model.Nodes]
model.flow_from_source_disjunction = Disjunction(rule=flow_from_source_disjunction_rule)


######################
# Boolean Constraints
######################

# YA_{s,n} v YR_{s,n} implies Y_s
def flow_existence_rule1(model, s, n):
    return model.screen_selection_disjunct[1, s].indicator_var >= \
        model.flow_acceptance_disjunct[s, n, 1].indicator_var
model.flow_existence1 = Constraint(model.ScreenNodePairs, rule=flow_existence_rule1)

def flow_existence_rule2(model, s, n):
    return model.screen_selection_disjunct[1,s].indicator_var >= \
        model.flow_rejection_disjunct[s, n, 1].indicator_var
model.flow_existence2 = Constraint(model.ScreenNodePairs, rule = flow_existence_rule2)


# YA_{s,s'} v YR_{s',s} implies Y_s
def screen_flow_existence_rule1(model, s, sprime):
    return model.screen_selection_disjunct[1, s].indicator_var >= \
        model.flow_acceptance_disjunct[s, sprime, 1].indicator_var
model.flow_existence1 = Constraint(model.ScreenPairs, rule=screen_flow_existence_rule1)

def screen_flow_existence_rule2(model, s, sprime):
    return model.screen_selection_disjunct[1,s].indicator_var >= \
        model.flow_rejection_disjunct[sprime, s, 1].indicator_var
model.flow_existence2 = Constraint(model.ScreenPairs, rule=screen_flow_existence_rule2)


# YA_{s', s} XOR YA_{s, s'}
def accept_rule1(model, s, sprime):
    return 1 <= model.flow_acceptance_disjunct[s, sprime, 1].indicator_var + \
        model.flow_acceptance_disjunct[sprime, s, 1].indicator_var
model.accept1 = Constraint(model.ScreenPairs, rule=accept_rule1)

def accept_rule2(model, s, sprime):
    return 1 >= model.flow_acceptance_disjunct[s, sprime, 1].indicator_var - \
        model.flow_acceptance_disjunct[sprime, s, 1].indicator_var
model.accept2 = Constraint(model.ScreenPairs, rule=accept_rule2)

def accept_rule3(model, s, sprime):
    return 1 >= model.flow_acceptance_disjunct[sprime, s, 1].indicator_var - \
        model.flow_acceptance_disjunct[s, sprime, 1].indicator_var
#model.accept3 = Constraint(model.ScreenPairs, rule=accept_rule3)

def accept_rule4(model, s, sprime):
    return 1 <= 2 - model.flow_acceptance_disjunct[s, sprime, 1].indicator_var - \
        model.flow_acceptance_disjunct[sprime, s, 1].indicator_var
model.accept4 = Constraint(model.ScreenPairs, rule=accept_rule4)


# YA_{s,n} XOR YR_{s,n}
def accept_or_reject_rule1(model, s, n):
    return 1 <= model.flow_acceptance_disjunct[s, n, 1].indicator_var + \
        model.flow_rejection_disjunct[s, n, 1].indicator_var
#model.accept_or_reject1 = Constraint(model.ScreenNodePairs, rule=accept_or_reject_rule1)

def accept_or_reject_rule2(model, s, n):
    return 1 >= model.flow_acceptance_disjunct[s, n, 1].indicator_var - \
        model.flow_rejection_disjunct[s, n, 1].indicator_var
#model.accept_or_reject2 = Constraint(model.ScreenNodePairs, rule=accept_or_reject_rule2)

def accept_or_reject_rule3(model, s, n):
    return 1 >= model.flow_rejection_disjunct[s, n, 1].indicator_var - \
        model.flow_acceptance_disjunct[s, n, 1].indicator_var
#model.accept_or_reject3 = Constraint(model.ScreenNodePairs, rule=accept_or_reject_rule3)

def accept_or_reject_rule4(model, s, n):
    return 1 <= 2 - model.flow_acceptance_disjunct[s, n, 1].indicator_var - \
        model.flow_rejection_disjunct[s, n, 1].indicator_var
#model.accept_or_reject4 = Constraint(model.ScreenNodePairs, rule=accept_or_reject_rule4)
