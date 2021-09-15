import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *

def build_sp_model(yields):
    '''
    Code adapted from https://mpi-sppy.readthedocs.io/en/latest/examples.html#examples
    It specifies the extensive form of the two-stage stochastic programming with extra index for scenarios
    
    Arguments:
        yields: Yield information as a list, following the rank [wheat, corn, beets]
        
    Return: 
        model: farmer problem model 
    '''
    model = ConcreteModel()
    
    all_crops = ["WHEAT", "CORN", "BEETS"]
    purchase_crops = ["WHEAT", "CORN"]
    sell_crops = ["WHEAT", "CORN", "BEETS_FAVORABLE", "BEETS_UNFAVORABLE"]
    scenarios = ["ABOVE","AVERAGE","BELOW"]
    
    # Fields allocation
    model.X = Var(all_crops, within=NonNegativeReals)
    # How many tons of crops to purchase in each scenario
    model.Y = Var(purchase_crops, scenarios, within=NonNegativeReals)
    # How many tons of crops to sell in each scenario
    model.W = Var(sell_crops, scenarios, within=NonNegativeReals)

    # Objective function
    model.PLANTING_COST = 150 * model.X["WHEAT"] + 230 * model.X["CORN"] + 260 * model.X["BEETS"]
    model.PURCHASE_COST_ABOVE = 238 * model.Y["WHEAT", "ABOVE"] + 210 * model.Y["CORN","ABOVE"]
    model.SALES_REVENUE_ABOVE = (
        170 * model.W["WHEAT", "ABOVE"] + 150 * model.W["CORN","ABOVE"]
        + 36 * model.W["BEETS_FAVORABLE","ABOVE"] + 10 * model.W["BEETS_UNFAVORABLE","ABOVE"])
    
    model.PURCHASE_COST_AVE = 238 * model.Y["WHEAT", "AVERAGE"] + 210 * model.Y["CORN","AVERAGE"]
    model.SALES_REVENUE_AVE = (
        170 * model.W["WHEAT", "AVERAGE"] + 150 * model.W["CORN","AVERAGE"]
        + 36 * model.W["BEETS_FAVORABLE","AVERAGE"] + 10 * model.W["BEETS_UNFAVORABLE","AVERAGE"])
    
    model.PURCHASE_COST_BELOW = 238 * model.Y["WHEAT", "BELOW"] + 210 * model.Y["CORN","BELOW"]
    model.SALES_REVENUE_BELOW = (
        170 * model.W["WHEAT", "BELOW"] + 150 * model.W["CORN","BELOW"]
        + 36 * model.W["BEETS_FAVORABLE","BELOW"] + 10 * model.W["BEETS_UNFAVORABLE","BELOW"])
    
    model.OBJ = Objective(
        expr=model.PLANTING_COST + 1/3*(model.PURCHASE_COST_ABOVE + model.PURCHASE_COST_AVE + model.PURCHASE_COST_BELOW)
        - 1/3*(model.SALES_REVENUE_ABOVE + model.SALES_REVENUE_AVE + model.SALES_REVENUE_BELOW),
        sense=minimize
    )

    # Constraints
    model.CONSTR= ConstraintList()

    model.CONSTR.add(summation(model.X) <= 500)
    model.CONSTR.add(yields[0] * model.X["WHEAT"] + model.Y["WHEAT","AVERAGE"] - model.W["WHEAT","AVERAGE"] >= 200)
    model.CONSTR.add(yields[0]*1.2 * model.X["WHEAT"] + model.Y["WHEAT","ABOVE"] - model.W["WHEAT","ABOVE"] >= 200)
    model.CONSTR.add(yields[0]*0.8 * model.X["WHEAT"] + model.Y["WHEAT","BELOW"] - model.W["WHEAT","BELOW"] >= 200)
    
    model.CONSTR.add(yields[1] * model.X["CORN"] + model.Y["CORN","AVERAGE"] - model.W["CORN","AVERAGE"] >= 240)
    model.CONSTR.add(yields[1]*1.2 * model.X["CORN"] + model.Y["CORN","ABOVE"] - model.W["CORN","ABOVE"] >= 240)
    model.CONSTR.add(yields[1]*0.8 * model.X["CORN"] + model.Y["CORN","BELOW"] - model.W["CORN","BELOW"] >= 240)
    
    model.CONSTR.add(
        yields[2] * model.X["BEETS"] - model.W["BEETS_FAVORABLE","AVERAGE"] - model.W["BEETS_UNFAVORABLE","AVERAGE"] >= 0
    )
    model.CONSTR.add(
        yields[2]*1.2 * model.X["BEETS"] - model.W["BEETS_FAVORABLE","ABOVE"] - model.W["BEETS_UNFAVORABLE","ABOVE"] >= 0
    )
    model.CONSTR.add(
        yields[2]*0.8 * model.X["BEETS"] - model.W["BEETS_FAVORABLE","BELOW"] - model.W["BEETS_UNFAVORABLE","BELOW"] >= 0
    )
    
    
    model.W["BEETS_FAVORABLE","AVERAGE"].setub(6000)
    model.W["BEETS_FAVORABLE","ABOVE"].setub(6000)
    model.W["BEETS_FAVORABLE","BELOW"].setub(6000)

    return model


def build_block_model(yields):
    '''
    Code adapted from https://mpi-sppy.readthedocs.io/en/latest/examples.html#examples
    It defines each scenario as a block 
    
    Arguments:
        yields: Yield information as a list, following the rank [wheat, corn, beets]
        
    Return: 
        model: farmer problem model 
    '''
    model = ConcreteModel()
    
    # Define sets
    model.all_crops = Set(initialize=["WHEAT", "CORN", "BEETS"])
    model.purchase_crops = Set(initialize=["WHEAT", "CORN"])
    model.sell_crops = Set(initialize=["WHEAT", "CORN", "BEETS_FAVORABLE", "BEETS_UNFAVORABLE"])
    
    # define scenario 
    model.scenarios = Set(initialize=['ABOVE','AVERAGE','BELOW'])
    
    # Crops field allocation
    model.X = Var(model.all_crops, within=NonNegativeReals)

    def construct_block(b, s):
        b.Y = Var(model.purchase_crops, within=NonNegativeReals)
        b.W = Var(model.sell_crops, within=NonNegativeReals)
    
    model.lsb = Block(model.scenarios, rule=construct_block)

    # Objective function
    model.PLANTING_COST = 150 * model.X["WHEAT"] + 230 * model.X["CORN"] + 260 * model.X["BEETS"]
    
    model.PURCHASE_COST = 238*sum(model.lsb[s].W["WHEAT"] for s in model.scenarios) + 210*sum(model.lsb[s].Y["CORN"] for s in model.scenarios)
    
    model.SALES_REVENUE_ABOVE = (
    170*model.lsb['ABOVE'].W["WHEAT"] + 150*model.lsb['ABOVE'].W["CORN"]
        + 36*model.lsb['ABOVE'].W["BEETS_FAVORABLE"] + 10*model.lsb['ABOVE'].W["BEETS_UNFAVORABLE"]
    )
    
    model.SALES_REVENUE_AVERAGE = (
    170*model.lsb['AVERAGE'].W["WHEAT"] + 150*model.lsb['AVERAGE'].W["CORN"]
        + 36*model.lsb['AVERAGE'].W["BEETS_FAVORABLE"] + 10*model.lsb['AVERAGE'].W["BEETS_UNFAVORABLE"]
    )
    
    model.SALES_REVENUE_BELOW = (
    170*model.lsb['BELOW'].W["WHEAT"] + 150*model.lsb['BELOW'].W["CORN"]
        + 36*model.lsb['BELOW'].W["BEETS_FAVORABLE"] + 10*model.lsb['BELOW'].W["BEETS_UNFAVORABLE"]
    )
    
    # Maximize the Obj is to minimize the negative of the Obj
    model.OBJ = Objective(
        expr=model.PLANTING_COST + 1/3*model.PURCHASE_COST - 1/3*(model.SALES_REVENUE_ABOVE + 
                                                                  model.SALES_REVENUE_AVERAGE+
                                                                  model.SALES_REVENUE_BELOW), sense=minimize)


    # Constraints
    model.CONSTR= ConstraintList()

    model.CONSTR.add(summation(model.X) <= 500)
    model.CONSTR.add(yields[0] * model.X["WHEAT"] + model.lsb['AVERAGE'].Y["WHEAT"] - model.lsb['AVERAGE'].W["WHEAT"] >= 200)
    model.CONSTR.add(yields[0]*1.2 * model.X["WHEAT"] + model.lsb['ABOVE'].Y["WHEAT"] - model.lsb['ABOVE'].W["WHEAT"] >= 200)
    model.CONSTR.add(yields[0]*0.8 * model.X["WHEAT"] + model.lsb['BELOW'].Y["WHEAT"] - model.lsb['BELOW'].W["WHEAT"] >= 200)
    
    model.CONSTR.add(yields[1] * model.X["CORN"] + model.lsb['AVERAGE'].Y["CORN"] - model.lsb['AVERAGE'].W["CORN"] >= 240)
    model.CONSTR.add(yields[1]*1.2 * model.X["CORN"] + model.lsb['ABOVE'].Y["CORN"] - model.lsb['ABOVE'].W["CORN"] >= 240)
    model.CONSTR.add(yields[1]*0.8 * model.X["CORN"] + model.lsb['BELOW'].Y["CORN"] - model.lsb['BELOW'].W["CORN"] >= 240)
    
    
    model.CONSTR.add(yields[2] * model.X["BEETS"] - model.lsb['AVERAGE'].W["BEETS_FAVORABLE"] - model.lsb['AVERAGE'].W["BEETS_UNFAVORABLE"] >= 0)
    model.CONSTR.add(yields[2]*1.2 * model.X["BEETS"] - model.lsb['ABOVE'].W["BEETS_FAVORABLE"] - model.lsb['ABOVE'].W["BEETS_UNFAVORABLE"] >= 0)
    model.CONSTR.add(yields[2]*0.8 * model.X["BEETS"] - model.lsb['BELOW'].W["BEETS_FAVORABLE"] - model.lsb['BELOW'].W["BEETS_UNFAVORABLE"] >= 0)
    
    model.lsb['AVERAGE'].W["BEETS_FAVORABLE"].setub(6000)
    model.lsb['ABOVE'].W["BEETS_FAVORABLE"].setub(6000)
    model.lsb['BELOW'].W["BEETS_FAVORABLE"].setub(6000)

    return model

### calculate two-stage stochastic problem
yields_perfect = [2.5, 3, 20]
model = build_sp_model(yields_perfect)
solver = SolverFactory("ipopt")
solver.solve(model,tee=True)

profit_2stage = -value(model.OBJ)

print("===Optimal solutions of two-stage stochastic problem with extra index===")
print('Culture.         | ', 'Wheat |', 'Corn  |', 'Sugar Beets |')
print('Surface (acres)  | ', f'{value(model.X["WHEAT"]):.1f}', '|', 
      f'{value(model.X["CORN"]):.1f}', ' |',
       f'{value(model.X["BEETS"]):.1f}',' |')
print('First stage: s=1 (Above average)')
print('Culture.         | ', 'Wheat |', 'Corn  |', 'Sugar Beets |')
print('Yield (T)        | ', f'{value(model.X["WHEAT"])*yields_perfect[0]*1.2:.1f}', '|', 
      f'{value(model.X["CORN"])*yields_perfect[1]*1.2:.1f}', '|',
       f'{value(model.X["BEETS"])*yields_perfect[2]*1.2:.1f}','|')
print('Sales (T)        | ', f'{value(model.W["WHEAT","ABOVE"]):.1f}', '|', 
      f'{value(model.W["CORN","ABOVE"]):.1f}', '  |',
       f'{value(model.W["BEETS_FAVORABLE","ABOVE"]) + value(model.W["BEETS_UNFAVORABLE","ABOVE"]):.1f}','|')
print('Purchases (T)    | ', f'{value(model.Y["WHEAT","ABOVE"]):.1f}', '  |', 
      f'{value(model.Y["CORN","ABOVE"]):.1f}', '  |',
       '-','     |')

print('First stage: s=2 (Average average)')
print('Culture.         | ', 'Wheat |', 'Corn  |', 'Sugar Beets |')
print('Yield (T)        | ', f'{value(model.X["WHEAT"])*yields_perfect[0]:.1f}', '|', 
      f'{value(model.X["CORN"])*yields_perfect[1]:.1f}', '|',
       f'{value(model.X["BEETS"])*yields_perfect[2]:.1f}','|')
print('Sales (T)        | ', f'{value(model.W["WHEAT","AVERAGE"]):.1f}', '|', 
      f'{value(model.W["CORN","AVERAGE"]):.1f}', '  |',
       f'{value(model.W["BEETS_FAVORABLE","AVERAGE"]) + value(model.W["BEETS_UNFAVORABLE","AVERAGE"]):.1f}','|')
print('Purchases (T)    | ', f'{value(model.Y["WHEAT","AVERAGE"]):.1f}', '  |', 
      f'{value(model.Y["CORN","AVERAGE"]):.1f}', '  |',
       '-','     |')

print('First stage: s=3 (Below average)')
print('Culture.         | ', 'Wheat |', 'Corn  |', 'Sugar Beets |')
print('Yield (T)        | ', f'{value(model.X["WHEAT"])*yields_perfect[0]*0.8:.1f}', '|', 
      f'{value(model.X["CORN"])*yields_perfect[1]*0.8:.1f}', '|',
       f'{value(model.X["BEETS"])*yields_perfect[2]*0.8:.1f}','|')
print('Sales (T)        | ', f'{value(model.W["WHEAT","BELOW"]):.1f}', '|', 
      f'{value(model.W["CORN","BELOW"]):.1f}', '  |',
       f'{value(model.W["BEETS_FAVORABLE","BELOW"]) + value(model.W["BEETS_UNFAVORABLE","BELOW"]):.1f}','|')
print('Purchases (T)    | ', f'{value(model.Y["WHEAT","BELOW"]):.1f}', '  |', 
      f'{value(model.Y["CORN","BELOW"]):.1f}', '  |',
       '-','     |')
print('Overall profit: $',f"{profit_2stage:.1f}")

### calculate two-stage stochastic problem
yields_perfect = [2.5, 3, 20]
model = build_block_model(yields_perfect)
solver = SolverFactory("ipopt")
solver.solve(model,tee=True)

profit_2stage = -value(model.OBJ)


print("===Optimal solutions of two-stage stochastic problem with blocks===")
print('Culture.         | ', 'Wheat |', 'Corn  |', 'Sugar Beets |')
print('Surface (acres)  | ', f'{value(model.X["WHEAT"]):.1f}', '|', 
      f'{value(model.X["CORN"]):.1f}', ' |',
       f'{value(model.X["BEETS"]):.1f}',' |')
print('First stage: s=1 (Above average)')
print('Culture.         | ', 'Wheat |', 'Corn  |', 'Sugar Beets |')
print('Yield (T)        | ', f'{value(model.X["WHEAT"])*yields_perfect[0]*1.2:.1f}', '|', 
      f'{value(model.X["CORN"])*yields_perfect[1]*1.2:.1f}', '|',
       f'{value(model.X["BEETS"])*yields_perfect[2]*1.2:.1f}','|')

print('Sales (T)        | ', f'{value(model.lsb["ABOVE"].W["WHEAT"]):.1f}', '|', 
      f'{value(model.lsb["ABOVE"].W["CORN"]):.1f}', '|',
       f'{value(model.lsb["ABOVE"].W["BEETS_FAVORABLE"]) + value(model.lsb["ABOVE"].W["BEETS_UNFAVORABLE"]):.1f}','|')
print('Purchases (T)    | ', f'{value(model.lsb["ABOVE"].Y["WHEAT"]):.1f}', '|', 
      f'{value(model.lsb["ABOVE"].Y["CORN"]):.1f}', '  |',
       '-','     |')

print('First stage: s=2 (Average average)')
print('Culture.         | ', 'Wheat |', 'Corn  |', 'Sugar Beets |')
print('Yield (T)        | ', f'{value(model.X["WHEAT"])*yields_perfect[0]:.1f}', '|', 
      f'{value(model.X["CORN"])*yields_perfect[1]:.1f}', '|',
       f'{value(model.X["BEETS"])*yields_perfect[2]:.1f}','|')
print('Sales (T)        | ', f'{value(model.lsb["AVERAGE"].W["WHEAT"]):.1f}', '|', 
      f'{value(model.lsb["AVERAGE"].W["CORN"]):.1f}', '  |',
       f'{value(model.lsb["AVERAGE"].W["BEETS_FAVORABLE"]) + value(model.lsb["AVERAGE"].W["BEETS_UNFAVORABLE"]):.1f}','|')
print('Purchases (T)    | ', f'{value(model.lsb["AVERAGE"].Y["WHEAT"]):.1f}', '  |', 
      f'{value(model.lsb["AVERAGE"].Y["CORN"]):.1f}', '  |',
       '-','     |')

print('First stage: s=3 (Below average)')
print('Culture.         | ', 'Wheat |', 'Corn  |', 'Sugar Beets |')
print('Yield (T)        | ', f'{value(model.X["WHEAT"])*yields_perfect[0]*0.8:.1f}', '|', 
      f'{value(model.X["CORN"])*yields_perfect[1]*0.8:.1f}', '|',
       f'{value(model.X["BEETS"])*yields_perfect[2]*0.8:.1f}','|')
print('Sales (T)        | ', f'{value(model.lsb["BELOW"].W["WHEAT"]):.1f}', '|', 
      f'{value(model.lsb["BELOW"].W["CORN"]):.1f}', '  |',
       f'{value(model.lsb["BELOW"].W["BEETS_FAVORABLE"]) + value(model.lsb["BELOW"].W["BEETS_UNFAVORABLE"]):.1f}','|')
print('Purchases (T)    | ', f'{value(model.lsb["BELOW"].Y["WHEAT"]):.1f}', '  |', 
      f'{value(model.lsb["BELOW"].Y["CORN"]):.1f}', '  |',
       '-','     |')
print('Overall profit: $',f"{profit_2stage:.1f}")
