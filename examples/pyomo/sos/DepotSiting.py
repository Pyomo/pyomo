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

# a simple SOS example from "Modeling Building in Mathematical Programming", H. Paul Williams, 4th Edition, p. 166.

from pyomo.core import *

model = AbstractModel()

# the set of customers.
model.Customers = Set()

# the possible locations of the facility that you're trying to place.
model.Sites = Set()

# the cost of satisfying customer demand from each of the potential sites.
model.SatisfactionCost = Param(model.Customers, model.Sites, within=NonNegativeReals)

# indicators of which sites are selected. constraints ensure only one site is selected,
# and allow the binary integrality to be implicit.
model.SiteSelected = Var(model.Sites, bounds=(0, 1))

# ensure that only one of the site selected variables is non-zero.
model.SiteSelectedSOS = SOSConstraint(var=model.SiteSelected, sos=1)


# ensure that one of the sites is selected (enforce binary).
def enforce_site_selected_binary_rule(model):
    return sum_product(model.SiteSelected) == 1


model.EnforceSiteSelectedBinary = Constraint(rule=enforce_site_selected_binary_rule)


# the objective is to minimize the cost to satisfy all customers.
def minimize_cost_rule(model):
    return sum(
        [
            model.SatisfactionCost[c, s] * model.SiteSelected[s]
            for c in model.Customers
            for s in model.Sites
        ]
    )


model.MinimizeCost = Objective(rule=minimize_cost_rule, sense=minimize)
