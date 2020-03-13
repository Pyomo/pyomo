# ScenariosCreator.py - Class to create and deliver scenarios using parmest
# DLW March 2020

import json
import pyomo.contrib.parmest.parmest as parmest
import pyomo.environ as pyo


class ScenarioSet(object):
    """
    Class to hold scenario sets
    Args:
    name (str): name of the set (might be "")
    """

    def __init__(self, name):
        self.scens = list()  # use a df instead?
        self.name = name  #  might be ""

    def addone(self, scen):
        """ Add a scenario to the set
        Args:
            scen (_ParmestScen): the scenario to add
        """
        assert(isinstance(self.scens, list))
        self.scens.append(scen)

    def Concatwith(self, set1,  newname):
        """ Concatenate a set to this set and return a new set 
        Args: 
            set1 (ScenarioSet): to append to this
        Returns:
            a new ScenarioSet
        """
        assert(isinstance(self.scens, list))
        newlist = self.scens + set1.scens
        retval = ScenarioSet(newname)
        retval.scens = newlist
        

    def write_csv(self, filename):
        """ write a csv file with the scenarios in the set
        Args:
            filename (str): full path and full name of file
        """
        with open(filename, "w") as f:
            for s in self.scens:
                f.write("{}, {}".format(s.name, s.probability))
                for n,v in s.ThetaVals.items():
                    f.write(", {}, {}".format(n,v))
                f.write('\n')


class _ParmestScen(object):
    # private class to hold scenarios

    def __init__(self, name, ThetaVals, probability):
        # ThetaVals is a dict: ThetaVals[name]=val
        self.name = name  # might be ""
        assert(isinstance(ThetaVals, dict))
        self.ThetaVals = ThetaVals
        self.probability = probability

        
class ScenarioCreator(object):
    """ Create scenarios from parmest.

    Args:
        pest (Estimator): the parmest object
        solvername (str): name of the solver (e.g. "ipopt")

    """

    def __init__(self, pest, solvername):
        self.pest = pest
        self.solvername = solvername
        self.experiment_numbers = pest._numbers_list


    def ScenariosFromExperiments(self, addtoSet):
        """Creates new self.Scenarios list using the experiments only.
        Args:
            addtoSet (ScenarioSet): the scenarios will be added to this set
        Returns:
            a ScenarioSet
        """

        assert(isinstance(addtoSet, ScenarioSet))
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
            addtoSet.addone(_ParmestScen("ExpScen"+str(exp_num), ThetaVals, prob))
            

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

    experimentscens = ScenarioSet("Experiments")
    scenmaker.ScenariosFromExperiments(experimentscens)
    experimentscens.write_csv("delme_exp_csv.csv")
