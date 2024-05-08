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

import json
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.semibatch.semibatch import SemiBatchExperiment
import pyomo.contrib.parmest.scenariocreator as sc


def main():

    # Data: list of dictionaries
    data = []
    file_dirname = dirname(abspath(str(__file__)))
    for exp_num in range(10):
        fname = join(file_dirname, 'exp' + str(exp_num + 1) + '.out')
        with open(fname, 'r') as infile:
            d = json.load(infile)
            data.append(d)

    # Create an experiment list
    exp_list = []
    for i in range(len(data)):
        exp_list.append(SemiBatchExperiment(data[i]))

    # View one model
    # exp0_model = exp_list[0].get_labeled_model()
    # exp0_model.pprint()

    pest = parmest.Estimator(exp_list)

    scenmaker = sc.ScenarioCreator(pest, "ipopt")

    # Make one scenario per experiment and write to a csv file
    output_file = "scenarios.csv"
    experimentscens = sc.ScenarioSet("Experiments")
    scenmaker.ScenariosFromExperiments(experimentscens)
    experimentscens.write_csv(output_file)

    # Use the bootstrap to make 3 scenarios and print
    bootscens = sc.ScenarioSet("Bootstrap")
    scenmaker.ScenariosFromBootstrap(bootscens, 3)
    for s in bootscens.ScensIterator():
        print("{}, {}".format(s.name, s.probability))
        for n, v in s.ThetaVals.items():
            print("   {}={}".format(n, v))


if __name__ == "__main__":
    main()
