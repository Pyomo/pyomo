# Mimic the pyomo script
from pyomo.core import *
from pyutilib.misc import Options

# set high level options that mimic pyomo comand line
options = Options()
options.model_file = 'DiseaseEstimation.py'
options.data_files = ['DiseaseEstimation.dat']
options.solver = 'ipopt'
options.solver_io = 'nl'
#options.keepfiles = True
#options.tee = True

# mimic the set of function calls done by pyomo command line
scripting.util.setup_environment(options)

# the following imports the model found in options.model_file,
# sets this to options.usermodel, and executes preprocessors
scripting.util.apply_preprocessing(options, parser=None)

# create the wrapper for the model, the data, the instance, and the options
model_data = scripting.util.create_model(options)
instance = model_data.instance

# solve
results, opt = scripting.util.apply_optimizer(options, instance)

# the following simply outputs the final time elapsed
scripting.util.finalize(options)

# load results into instance and print
instance.load(results)
display(instance)
