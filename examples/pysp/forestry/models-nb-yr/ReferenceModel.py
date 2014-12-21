#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# fbv - Chile rules!
# dlw - base forest model - begin April 2009
# mnr - base forest model - revised May 1, 2009 - changed the notation of parameters to match those in paper.

#
# Imports (i.e., magical incantations)
#
from pyomo.core import *

#
# Model
#
model = AbstractModel()
model.name = "Forest Management Base Model"

#
# Implementation Sets (not in paper, used to make it happen)
# general nodes (not in paper)
model.Nodes = Set()

#
# Sets
#


# time Horizon \mathcal{T}
#def Time_init
#model.Times = Set(ordered=True, initialize=Time_init)
model.Times = Set(ordered=True)

# network nodes \mathcal{O}
model.OriginNodes = Set(within=model.Nodes)

# intersection nodes
model.IntersectionNodes = Set(within=model.Nodes)

# wood exit nodes
model.ExitNodes = Set(within=model.Nodes)

# harvest cells \mathcal{H}
model.HarvestCells = Set()

# harvest cells for origin o \mathcal{H}_o
model.HCellsForOrigin = Set(model.OriginNodes, within=model.HarvestCells)

# HERE'S MY BUG!!!!!!!!!!!!!!!!!!!
# origin nodes for cell
#def origin_node_for_harvest_cell_rule(model, h):
#   for o in model.OriginNodes:
#      if h in model.HCellsForOrigin[o]:
#         print "h,o=",h,o
#         return o
#         #return o()
#         #return str(o)
#model.COriginNodeForCellA = Set(model.HarvestCells, within=model.OriginNodes, rule=origin_node_for_harvest_cell_rule)
# SO I PRECALCULATED IT
model.COriginNodeForCell = Set(model.HarvestCells, within=model.OriginNodes)

# existing roads
model.ExistingRoads = Set(within=model.Nodes*model.Nodes) 

# potential roads
model.PotentialRoads = Set(within=model.Nodes*model.Nodes)

# Roads declaring
#model.AllRoads = model.ExistingRoads & model.PotentialRoads
model.AllRoads = Set(within=model.Nodes*model.Nodes)

# derived set
def aplus_arc_set_rule(model, v):
    return  [(i, j) for (i, j) in model.AllRoads if j == v]
model.Aplus = Set(model.Nodes, within=model.AllRoads, initialize=aplus_arc_set_rule)

# derived set
def aminus_arc_set_rule(model, v):
    return  [(i, j) for (i, j) in model.AllRoads if i == v]
model.Aminus = Set(model.Nodes, within=model.AllRoads, initialize=aminus_arc_set_rule)

# derived set for the "no isolated roads can be built" constraint
# if you would build the road, then one connecting road should be built around you. And because the roads its isolated, you need to sum over potential roads around you
def connected_potential_roads_rule(model, k, l):
    return  [ (i, j) for (i, j) in model.PotentialRoads if (i==k and j!=l) or (i==l and j!=k) or (j==k and i!=l) or (j==l and i!=k) ]
model.ConnectedPotentialRoads = Set( model.Nodes, model.Nodes, within=model.PotentialRoads, initialize=connected_potential_roads_rule)
# a PotentialRoad is not isolated if it connects to an ExitNode or an Existing Road
def potential_roads_connected_to_existing_roads_rule(model):
    set=[]
    for (k,l) in model.PotentialRoads:
        #if not (k in model.ExitNodes and l in model.OriginNodes) and not (l in model.ExitNodes and k in model.OriginNodes):
        if not (k in model.ExitNodes) and not (l in model.ExitNodes):
            for x in model.Nodes:
                if x!=k and x!=l:
                    #print "k,l,x=",k,l,x
                    if (x,k) in model.AllRoads or (x,l) in model.AllRoads or (k,x) in model.AllRoads or (l,x) in model.AllRoads:
                        #if not (k in model.OriginNodes and (l,x) in model.ExistingRoads) and not (l in model.OriginNodes and (k,x) in model.ExistingRoads):
                        if ((x,k) in model.ExistingRoads) or ((x,l) in model.ExistingRoads) or ((k,x) in model.ExistingRoads) or ((l,x) in model.ExistingRoads):
                            if (k,l) not in set:
                                set.append((k,l))
                                #print "set=",set
    return set
model.PotentialRoadsConnectedToExistingRoads = Set(within=model.PotentialRoads, initialize=potential_roads_connected_to_existing_roads_rule)
# isolated PotentialRoads
model.DisconnectedPotentialRoads = model.PotentialRoads - model.PotentialRoadsConnectedToExistingRoads

# derived set for the "no isolated lots can be harvest" constraint
# if a isolated lot is harvested then one potential road accesing it must be built around
def potential_roads_accesing_lot_h(model, h):
    return  [ (i, j) for (i, j) in model.PotentialRoads if (i in model.COriginNodeForCell[h]) or (j in model.COriginNodeForCell[h]) ]
model.PotentialRoadsAccesingLotH = Set( model.HarvestCells, within=model.PotentialRoads, initialize=potential_roads_accesing_lot_h )
# a isolated lot is not connected to an ExistingRoad
def lots_connected_to_existing_roads(model):
    set=[]
    for h in model.HarvestCells:
        for (i,j) in model.ExistingRoads:
            if (i in model.COriginNodeForCell[h]) or (j in model.COriginNodeForCell[h]):
                if h not in set:
                    set.append(h)
    return set
model.LotsConnectedToExistingRoads = Set(within=model.HarvestCells, initialize=lots_connected_to_existing_roads )
# isolated lots or HarvestCells
model.LotsNotConnectedToExistingRoads =  model.HarvestCells - model.LotsConnectedToExistingRoads


#
# Deterministic Parameters
#

# productivity of cell h if it is harvested in period t
model.a = Param(model.HarvestCells, model.Times)

# Area of cell h to be harvested
model.A = Param(model.HarvestCells)


# harvesting cost of one hectare of cell h in time t
model.P = Param(model.HarvestCells, model.Times)

# unit Production cost at origin o in time t
model.Q = Param(model.OriginNodes, model.Times)

# construction cost of one road in arc (k,l) in time t
model.C = Param(model.PotentialRoads, model.Times, default=0.0)

# unit transport cost through arc (k,l) in time t
model.D = Param(model.AllRoads, model.Times)

#
# Stochastic Parameters
#

# sale price at exit s in time period t
model.R = Param(model.ExitNodes, model.Times)

# upper and lower bounds on wood supplied
model.Zlb = Param(model.Times)
model.Zub = Param(model.Times)

# Yield Ratio
model.yr = Param(model.Times)

# flow capacity (smallest big M) 
def Umax_init(model):
    return sum([value(model.a[h, t]) * value(model.yr[t]) * value(model.A[h]) for h in model.HarvestCells for t in model.Times])
model.Umax = Param(initialize=Umax_init)
#
# Variables
#

# If cell h is harvested in period t
model.delta = Var(model.HarvestCells, model.Times, domain=Boolean)

# If road in arc (k,l) is built in period t
model.gamma = Var(model.PotentialRoads, model.Times, domain=Boolean)

# Flow of wood thransported through arc (k,l)
model.f = Var(model.AllRoads, model.Times, domain=NonNegativeReals)

# Supply of wood at exit s in period t
model.z = Var(model.ExitNodes, model.Times, domain=NonNegativeReals)

# Declared when changing the program for stochastics
model.AnoProfit = Var(model.Times)

#
# Constraints
#

# flow balance at origin nodes
def origin_flow_bal(model, o, t):
    ans1 = sum([model.a[h, t] * model.yr[t] * model.A[h] * model.delta[h, t] for h in model.HCellsForOrigin[o]])
    ans2 = sum([model.f[k, o, t] for (k, o) in model.Aplus[o]])
    ans3 = sum([model.f[o, k, t] for (o, k) in model.Aminus[o]])
    return (ans1 + ans2 - ans3) == 0.0
 
model.EnforceOriginFlowBalance = Constraint(model.OriginNodes, model.Times, rule=origin_flow_bal)

# flow balance at intersection nodes
def intersection_flow_bal(model, j, t):
    return (sum([model.f[k, j, t] for (k, j) in model.Aplus[j]]) - sum([model.f[j, k, t] for (j, k) in model.Aminus[j]])) == 0.0
 
model.EnforceIntersectionFlowBalance = Constraint(model.IntersectionNodes, model.Times, rule=intersection_flow_bal)

# flow balance at destination nodes
def destination_flow_bal(model, e, t):
    return (model.z[e, t] - sum([model.f[k, e, t] for (k, e) in model.Aplus[e]]) + sum([model.f[e, k, t] for (e, k) in model.Aminus[e]])) == 0.0
 
model.EnforceDestinationFlowBalance = Constraint(model.ExitNodes, model.Times, rule=destination_flow_bal)

# explicit divergence or supply equals demand or no inventory between periods
def divergence_flow_bal(model, t):
    return (sum([model.z[e, t] for e in model.ExitNodes]) - sum([model.a[h, t] * model.yr[t] * model.A[h] * model.delta[h, t] for h in model.HarvestCells])) == 0.0
 
model.EnforceDivergenceFlowBalance = Constraint(model.Times, rule=divergence_flow_bal)

# Wood Production Bounds
def wood_prod_bounds(model, t):
    return (model.Zlb[t], sum([model.z[e, t] for e in model.ExitNodes]), model.Zub[t])
 
model.EnforceWoodProductionBounds = Constraint(model.Times, rule=wood_prod_bounds)

# potential roads flow capacity 1 
def pot_road_flow_capacity(model, k, l, t):
    return (None, model.f[k, l, t] - sum([model.Umax * model.gamma[k, l, tau] for tau in model.Times if tau <= t]), 0.0)

model.EnforcePotentialRoadFlowCap = Constraint(model.PotentialRoads, model.Times, rule=pot_road_flow_capacity)

# each potential road can be built no more than once in the time horizon
def one_potential_road(model, k, l):
    return (None, sum([model.gamma[k ,l, t] for t in model.Times]), 1.0)
 
model.EnforceOnePotentialRoad = Constraint(model.PotentialRoads, rule=one_potential_road)

# existing roads flow capacity
def existing_roads_capacity(model, k, l, t):
    return (None, model.f[k, l, t], model.Umax)

model.EnforceExistingRoadCapacities = Constraint(model.ExistingRoads, model.Times, rule=existing_roads_capacity)

# additional road bound capacity
def additional_roads_capacity(model, k, l, t):
    return ( None, model.f[k, l, t] - sum([model.z[e, t] for e in model.ExitNodes]), 0.0 )

model.EnforceAdditionalRoadCapacities = Constraint(model.AllRoads, model.Times, rule=additional_roads_capacity)

# each cell can be harvested no more than once in the time horizon
def one_harvest(model, h):
    return (None, sum([model.delta[h,t] for t in model.Times]), 1.0)
 
model.EnforceOneHarvest = Constraint(model.HarvestCells, rule=one_harvest)

# no isolated roads can be built
def road_to_road_trigger(model, t, k, l):
    return ( None, sum([ model.gamma[ k, l, tt] for tt in model.Times if tt <= t ]) - sum([ model.gamma[ i, j, ttt] for ttt in model.Times if ttt <= t for (i,j) in model.ConnectedPotentialRoads[k,l] ]), 0.0 )
 
model.EnforceRoadToRoadTrigger = Constraint(model.Times, model.DisconnectedPotentialRoads, rule=road_to_road_trigger)

# no isolated lots can be harvest
def lot_to_road_trigger(model, t, h):
    return ( None, sum([ model.delta[ h, tt] for tt in model.Times if tt <= t ]) - sum([ model.gamma[ i, j, ttt] for ttt in model.Times if ttt <= t for (i,j) in model.PotentialRoadsAccesingLotH[h] ]), 0.0 )
 
model.EnforceLotToRoadTrigger = Constraint(model.Times, model.LotsNotConnectedToExistingRoads, rule=lot_to_road_trigger)

#
# Stage-specific profit computations
#

# awful but straigthforward way to set the discount factor
def hehe(t):
#    print "t=",t
    if t == "Ano1":
        return 1
    if t == "Ano2":
        return 0.9
    if t == "Ano3":
        return 0.81
    if t == "Ano4":
        return 0.729

def stage_profit_rule(model, t):
    # ans1 is T1 (i.e. wood sale revenue)
    ans1 = sum([model.R[e, t] * model.z[e, t] for e in model.ExitNodes])
    
    # ans2 is T2 (i.e. wood harvest cost)
    ans2 = sum([model.P[h, t] * model.A[h] * model.delta[h, t] for h in model.HarvestCells])
        
    # ans3 is T3 (i.e. production costs at origin nodes)
    ans3 = sum([model.a[h, t] * model.yr[t] * model.A[h] * model.delta[h,t] * model.Q[o, t] for o in model.OriginNodes for h in model.HCellsForOrigin[o]])
           
    # ans4 is T4 (i.e. potential road construction cost)
    ans4 = sum([model.C[i, j, t] * model.gamma[i, j, t] for (i,j) in model.PotentialRoads])
    
    # ans5 is T5 (i.e. wood transport cost)
    ans5 = sum([model.D[i, j, t] * model.f[i, j, t] for (i,j) in model.AllRoads])
       
    return (model.AnoProfit[t] - hehe(t)*(ans1 - ans2 - ans3 - ans4 - ans5)) == 0.0
#    return (model.AnoProfit[t] - (ans1 - ans2 - ans3 - ans4 - ans5)) == 0.0

model.ComputeStageProfit = Constraint(model.Times, rule=stage_profit_rule)

#
# PySP Auto-generated Objective
#
# maximize: sum of StageCostVariables
#
# A active scenario objective equivalent to that generated by PySP is
# included here for informational purposes.
def total_profit_rule(model):
    return summation(model.AnoProfit)
model.Production_Profit_Objective = Objective(rule=total_profit_rule, sense=maximize)

