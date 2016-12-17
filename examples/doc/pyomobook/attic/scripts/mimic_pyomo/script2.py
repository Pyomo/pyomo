from pyomo.core import DataPortal
from pyomo.opt import SolverFactory
from DiseaseEstimation import model

model.pprint()

# @modeldata:
modeldata = DataPortal(model=model)
modeldata.load(filename='DiseaseEstimation.dat')
modeldata.load(filename='DiseasePop.dat')
# @:modeldata

instance = model.create(modeldata)
instance.pprint()

opt = SolverFactory("ipopt")
results = opt.solve(instance)

results.write()
