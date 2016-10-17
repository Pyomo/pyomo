from pyomo.environ import SolverFactory, DataPortal
from DiseaseEstimation import model

model.pprint()

data = DataPortal(model=model)
data.load(filename='DiseaseEstimation.dat')
data.load(filename='DiseasePop.dat')

instance = model.create(data)
instance.pprint()

solver = SolverFactory("ipopt")
results = solver.solve(instance)

instance.display()
