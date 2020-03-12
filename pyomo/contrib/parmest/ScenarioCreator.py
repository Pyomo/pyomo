# ScenariosCreator.py - Class to create and deliver scenarios using parmest
# DLW March 2020

import json
import pyomo.contrib.parmest.parmest as parmest
import pyomo.environ as pyo


class _ParmestScen(object):
    """
    local class to hold scenarios
    """

    def __init__(self, name, ThetaVals, probability):
        # ThetaVals[name]=val
        self.name = name  # might be ""
        self.ThetaVals = ThetaVals
        self.probility = probability

        
class ScenarioCreator(object):
    """ Create, deliver and perhaps store scenarios from parmest

    Args:
        pest (Estimator): the parmest object
        solvername (str): name of the solver (e.g. "ipopt")

    """

    def __init__(self, pest, solvername):
        self.pest = pest
        self.solvername = solvername
        self.experiment_numbers = pest._numbers_list
        self.Scenarios = list()  # list of _ParmestScen objects (often reset)


    def ScenariosFromExperiments(self):
        # Creates new self.Scenarios list using the experiments only.

        self.Scenarios = list()
        prob = 1. / len(self.pest._numbers_list)
        for exp_num in self.pest._numbers_list:
            print("Experiment number=", exp_num)
            model = self.pest._instance_creation_callback(exp_num, data)
            opt = pyo.SolverFactory(self.solvername)
            results = opt.solve(model)  # solves and updates model
            ## pyo.check_termination_optimal(results)
            ThetaVals = dict()
            for theta in self.pest.theta_names:
                tvar = eval('model.'+theta)
                tval = pyo.value(tvar)
                print("    theta, tval=", tvar, tval)
                ThetaVals[theta] = tval
            self.Scenarios.append(_ParmestScen("ExpScen"+str(exp_num), ThetaVals, prob))

if __name__ == "__main__":
    # quick test using semibatch
    import pyomo.contrib.parmest.examples.semibatch.semibatch as sb

    # Vars to estimate in parmest
    theta_names = ['k1', 'k2', 'E1', 'E2']

    # Data, list of dictionaries
    data = [] 
    for exp_num in range(10):
        fname = 'examples/semibatch/exp'+str(exp_num+1)+'.out'
        with open(fname,'r') as infile:
            d = json.load(infile)
            data.append(d)

    # Note, the model already includes a 'SecondStageCost' expression 
    # for sum of squared error that will be used in parameter estimation

    pest = parmest.Estimator(sb.generate_model, data, theta_names)
    
    scenmaker = ScenarioCreator(pest, "ipopt")

    scenmaker.ScenariosFromExperiments()
