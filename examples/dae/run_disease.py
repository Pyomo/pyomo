from pyomo.environ import *
from pyomo.dae import *
from disease_DAE import model

instance = model.create_instance('disease.dat')

discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(instance,nfe=520,ncp=3)

def _S_bar(model):
    return model.S_bar == sum(model.S[i] for i in model.TIME if i != 0)/(len(model.TIME)-1)
instance.con_S_bar = Constraint(rule=_S_bar)

solver=SolverFactory('ipopt')
results = solver.solve(instance,tee=True)

