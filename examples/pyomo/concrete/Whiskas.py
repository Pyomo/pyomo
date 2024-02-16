#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.environ import *

# Creates a list of the Ingredients
Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']

# A dictionary of the costs of each of the Ingredients is created
costs = {
    'CHICKEN': 0.013,
    'BEEF': 0.008,
    'MUTTON': 0.010,
    'RICE': 0.002,
    'WHEAT': 0.005,
    'GEL': 0.001,
}

# A dictionary of the protein percent in each of the Ingredients is created
proteinPercent = {
    'CHICKEN': 0.100,
    'BEEF': 0.200,
    'MUTTON': 0.150,
    'RICE': 0.000,
    'WHEAT': 0.040,
    'GEL': 0.000,
}

# A dictionary of the fat percent in each of the Ingredients is created
fatPercent = {
    'CHICKEN': 0.080,
    'BEEF': 0.100,
    'MUTTON': 0.110,
    'RICE': 0.010,
    'WHEAT': 0.010,
    'GEL': 0.000,
}

# A dictionary of the fibre percent in each of the Ingredients is created
fibrePercent = {
    'CHICKEN': 0.001,
    'BEEF': 0.005,
    'MUTTON': 0.003,
    'RICE': 0.100,
    'WHEAT': 0.150,
    'GEL': 0.000,
}

# A dictionary of the salt percent in each of the Ingredients is created
saltPercent = {
    'CHICKEN': 0.002,
    'BEEF': 0.005,
    'MUTTON': 0.007,
    'RICE': 0.002,
    'WHEAT': 0.008,
    'GEL': 0.000,
}

model = ConcreteModel(name="The Whiskas Problem")

model.ingredient_vars = Var(
    Ingredients, bounds=(0, None), doc="The amount of each ingredient that is used"
)

model.obj = Objective(
    expr=sum(costs[i] * model.ingredient_vars[i] for i in Ingredients),
    doc="Total Cost of Ingredients per can",
)

model.c0 = Constraint(
    expr=sum(model.ingredient_vars[i] for i in Ingredients) == 100, doc="PercentagesSum"
)
model.c1 = Constraint(
    expr=sum(proteinPercent[i] * model.ingredient_vars[i] for i in Ingredients) >= 8.0,
    doc="ProteinRequirement",
)
model.c2 = Constraint(
    expr=sum(fatPercent[i] * model.ingredient_vars[i] for i in Ingredients) >= 6.0,
    doc="FatRequirement",
)
model.c3 = Constraint(
    expr=sum(fibrePercent[i] * model.ingredient_vars[i] for i in Ingredients) <= 2.0,
    doc="FibreRequirement",
)
model.c4 = Constraint(
    expr=sum(saltPercent[i] * model.ingredient_vars[i] for i in Ingredients) <= 0.4,
    doc="SaltRequirement",
)
