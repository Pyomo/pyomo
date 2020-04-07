# scenario creation example; DLW March 2020

import os
import json
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.semibatch.semibatch import generate_model
import pyomo.contrib.parmest.scenariocreator as sc

def main(dirname):
    """ dirname gives the location of the experiment input files"""
    # Semibatch Vars to estimate in parmest
    theta_names = ['k1', 'k2', 'E1', 'E2']

    # Semibatch data: list of dictionaries
    data = [] 
    for exp_num in range(10):
        fname = os.path.join(dirname, 'exp'+str(exp_num+1)+'.out')
        with open(fname,'r') as infile:
            d = json.load(infile)
            data.append(d)

    pest = parmest.Estimator(generate_model, data, theta_names)

    scenmaker = sc.ScenarioCreator(pest, "ipopt")

    ofile = "delme_exp.csv"
    print("Make one scenario per experiment and write to {}".format(ofile))
    experimentscens = sc.ScenarioSet("Experiments")
    scenmaker.ScenariosFromExperiments(experimentscens)
    ###experimentscens.write_csv(ofile)

    numtomake = 3
    print("\nUse the bootstrap to make {} scenarios and print.".format(numtomake))
    bootscens = sc.ScenarioSet("Bootstrap")
    scenmaker.ScenariosFromBoostrap(bootscens, numtomake)
    for s in bootscens.ScensIterator():
        print("{}, {}".format(s.name, s.probability))
        for n,v in s.ThetaVals.items():
            print("   {}={}".format(n, v))

if __name__ == "__main__":
    main(".")
