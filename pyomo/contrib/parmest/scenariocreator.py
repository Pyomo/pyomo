#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# ScenariosCreator.py - Class to create and deliver scenarios using parmest
# DLW March 2020

import pyomo.environ as pyo


class ScenarioSet(object):
    """
    Class to hold scenario sets

    Args:
    name (str): name of the set (might be "")

    """

    def __init__(self, name):
        # Note: If there was a use-case, the list could be a dataframe.
        self._scens = list()  # use a df instead?
        self.name = name  #  might be ""


    def _firstscen(self):
        # Return the first scenario for testing and to get Theta names.
        assert(len(self._scens) > 0)
        return self._scens[0]


    def ScensIterator(self):
        """ Usage: for scenario in ScensIterator()"""
        return iter(self._scens)


    def ScenarioNumber(self, scennum):
        """ Returns the scenario with the given, zero-based number"""
        return self._scens[scennum]

    
    def addone(self, scen):
        """ Add a scenario to the set

        Args:
            scen (ParmestScen): the scenario to add
        """
        assert(isinstance(self._scens, list))
        self._scens.append(scen)

        
    def append_bootstrap(self, bootstrap_theta):
        """ Append a boostrap theta df to the scenario set; equally likely

        Args:
            boostrap_theta (dataframe): created by the bootstrap
        Note: this can be cleaned up a lot with the list becomes a df,
              which is why I put it in the ScenarioSet class.
        """
        assert(len(bootstrap_theta) > 0)
        prob = 1. / len(bootstrap_theta)

        # dict of ThetaVal dicts
        dfdict = bootstrap_theta.to_dict(orient='index')

        for index, ThetaVals in dfdict.items():
            name = "Boostrap"+str(index)
            self.addone(ParmestScen(name, ThetaVals, prob))


    def write_csv(self, filename):
        """ write a csv file with the scenarios in the set

        Args:
            filename (str): full path and full name of file
        """
        if len(self._scens) == 0:
            print ("Empty scenario set, not writing file={}".format(filename))
            return
        with open(filename, "w") as f:
            f.write("Name,Probability")
            for n in self._firstscen().ThetaVals.keys():
                f.write(",{}".format(n))
            f.write('\n')
            for s in self.ScensIterator():
                f.write("{},{}".format(s.name, s.probability))
                for v in s.ThetaVals.values():
                    f.write(",{}".format(v))
                f.write('\n')


class ParmestScen(object):
    """ A little container for scenarios; the Args are the attributes.

    Args:
        name (str): name for reporting; might be ""
        ThetaVals (dict): ThetaVals[name]=val
        probability (float): probability of occurance "near" these ThetaVals
    """

    def __init__(self, name, ThetaVals, probability):
        self.name = name
        assert(isinstance(ThetaVals, dict))
        self.ThetaVals = ThetaVals
        self.probability = probability

############################################################


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
            ##print("Experiment number=", exp_num)
            model = self.pest._instance_creation_callback(exp_num,
                                                        self.pest.callback_data)
            opt = pyo.SolverFactory(self.solvername)
            results = opt.solve(model)  # solves and updates model
            ## pyo.check_termination_optimal(results)
            ThetaVals = dict()
            for theta in self.pest.theta_names:
                tvar = eval('model.'+theta)
                tval = pyo.value(tvar)
                ##print("    theta, tval=", tvar, tval)
                ThetaVals[theta] = tval
            addtoSet.addone(ParmestScen("ExpScen"+str(exp_num), ThetaVals, prob))
            
    def ScenariosFromBoostrap(self, addtoSet, numtomake, seed=None):
        """Creates new self.Scenarios list using the experiments only.

        Args:
            addtoSet (ScenarioSet): the scenarios will be added to this set
            numtomake (int) : number of scenarios to create
        """

        assert(isinstance(addtoSet, ScenarioSet))

        bootstrap_thetas = self.pest.theta_est_bootstrap(numtomake, seed=seed)
        addtoSet.append_bootstrap(bootstrap_thetas)
