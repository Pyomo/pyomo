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
Continuously stirred tank reactor model, based on
pyomo\examples\doc\pyomobook\nonlinear-ch\react_design\ReactorDesign.py
"""
import pandas as pd
from pyomo.environ import ConcreteModel, Var, PositiveReals, Objective, Constraint, maximize, SolverFactory

def reactor_design_model(data):
    
    # Create the concrete model
    model = ConcreteModel()
    
    # Rate constants
    model.k1 = Var(initialize = 5.0/6.0, within=PositiveReals) # min^-1
    model.k2 = Var(initialize = 5.0/3.0, within=PositiveReals) # min^-1
    model.k3 = Var(initialize = 1.0/6000.0, within=PositiveReals) # m^3/(gmol min)
    model.k1.fixed = True
    model.k2.fixed = True
    model.k3.fixed = True
    
    # Inlet concentration of A, gmol/m^3
    model.caf = Var(initialize = float(data['caf']), within=PositiveReals)
    model.caf.fixed = True
    
	# Space velocity (flowrate/volume)
    model.sv = Var(initialize = float(data['sv']), within=PositiveReals)
    model.sv.fixed = True
    
    # Outlet concentration of each component
    model.ca = Var(initialize = 5000.0, within=PositiveReals) 
    model.cb = Var(initialize = 2000.0, within=PositiveReals) 
    model.cc = Var(initialize = 2000.0, within=PositiveReals) 
    model.cd = Var(initialize = 1000.0, within=PositiveReals)
    
    # Objective
    model.obj = Objective(expr = model.cb, sense=maximize)
    
    # Constraints
    model.ca_bal = Constraint(expr = (0 == model.sv * model.caf \
                     - model.sv * model.ca - model.k1 * model.ca \
                     -  2.0 * model.k3 * model.ca ** 2.0))
    
    model.cb_bal = Constraint(expr=(0 == -model.sv * model.cb \
                     + model.k1 * model.ca - model.k2 * model.cb))
    
    model.cc_bal = Constraint(expr=(0 == -model.sv * model.cc \
                     + model.k2 * model.cb))
    
    model.cd_bal = Constraint(expr=(0 == -model.sv * model.cd \
                     + model.k3 * model.ca ** 2.0))
    
    return model

if __name__ == "__main__":
    
    # For a range of sv values, return ca, cb, cc, and cd
    results = []
    sv_values = [1.0 + v * 0.05 for v in range(1, 20)]
    caf = 10000
    for sv in sv_values:
        model = reactor_design_model({'caf': caf, 'sv': sv})
        solver = SolverFactory('ipopt')
        solver.solve(model)
        results.append([sv, caf, model.ca(), model.cb(), model.cc(), model.cd()])
    
    results = pd.DataFrame(results, columns=['sv', 'caf', 'ca', 'cb', 'cc', 'cd'])
    print(results)
    