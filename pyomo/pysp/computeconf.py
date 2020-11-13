#! /usr/bin/env python
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import random
import math
import time
import traceback

from pyomo.common.errors import ApplicationError

from pyomo.core import minimize
# this is a hack, in order to pick up the UndefinedData class. this is
# needed currently, as CPLEX is periodically barfing on cvar
# formulations, yielding an undefined gap. technically, the gap is
# defined and the solution is feasible, but a correct fix to the CPLEX
# plugin would yield a complete failure to solve cvar problems. see
# related hacks below, searching for CVARHACK.
from pyomo.opt import UndefinedData
from pyomo.common import pyomo_command
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.phinit import (construct_ph_options_parser,
                               GenerateScenarioTreeForPH,
                               PHAlgorithmBuilder,
                               PHFromScratch,
                               PHCleanup)
from pyomo.pysp.ef_writer_script import ExtensiveFormAlgorithm
from pyomo.pysp.phutils import _OLD_OUTPUT

from six import iteritems, iterkeys

# to avoid the pain of user lookup of parameter in t-tables, we
# provide decent coverage automatically.  feel free to add more
# values!!!! maps degrees-of-freedom to (alpha,t-statistic) pairs.

t_table_values = {

1 :  {0.25  : 1.000  ,0.1   : 3.078  ,0.05  : 6.314  ,0.025 : 12.706 ,0.01  : 31.821 ,0.005 : 63.657 ,0.001 : 318.309},
2 :  {0.25  : 0.816  ,0.1   : 1.886  ,0.05  : 2.920  ,0.025 : 4.303  ,0.01  : 6.965  ,0.005 : 9.925  ,0.001 : 22.327},
3 :  {0.25  : 0.765  ,0.1   : 1.638  ,0.05  : 2.353  ,0.025 : 3.182  ,0.01  : 4.541  ,0.005 : 5.841  ,0.001 : 10.215},
4 :  {0.25  : 0.741 , 0.1   : 1.533 , 0.05  : 2.132 , 0.025 : 2.776 , 0.01  : 3.747 , 0.005 : 4.604 , 0.001 : 7.173},
5 :  {0.25  : 0.727 , 0.1   : 1.476 , 0.05  : 2.015 , 0.025 : 2.571 , 0.01  : 3.365 , 0.005 : 4.032 , 0.001 : 5.893},
6 :  {0.25  : 0.718 , 0.1   : 1.440 , 0.05  : 1.943 , 0.025 : 2.447 , 0.01  : 3.143 , 0.005 : 3.707 , 0.001 : 5.208},
7 :  {0.25  : 0.711 , 0.1   : 1.415 , 0.05  : 1.895 , 0.025 : 2.365 , 0.01  : 2.998 , 0.005 : 3.499 , 0.001 : 4.785},
8 :  {0.25  : 0.706 , 0.1   : 1.397 , 0.05  : 1.860 , 0.025 : 2.306 , 0.01  : 2.896 , 0.005 : 3.355 , 0.001 : 4.501},
9 :  {0.25  : 0.703 , 0.1   : 1.383 , 0.05  : 1.833 , 0.025 : 2.262 , 0.01  : 2.821 , 0.005 : 3.250 , 0.001 : 4.297},
10 : {0.25  : 0.700 , 0.1   : 1.372 , 0.05  : 1.812 , 0.025 : 2.228 , 0.01  : 2.764 , 0.005 : 3.169 , 0.001 : 4.144},
11 : {0.25  : 0.697 , 0.1   : 1.363 , 0.05  : 1.796 , 0.025 : 2.201 , 0.01  : 2.718 , 0.005 : 3.106 , 0.001 : 4.025},
12 : {0.25  : 0.695 , 0.1   : 1.356 , 0.05  : 1.782 , 0.025 : 2.179 , 0.01  : 2.681 , 0.005 : 3.055 , 0.001 : 3.930},
13 : {0.25  : 0.694 , 0.1   : 1.350 , 0.05  : 1.771 , 0.025 : 2.160 , 0.01  : 2.650 , 0.005 : 3.012 , 0.001 : 3.852},
14 : {0.25  : 0.692 , 0.1   : 1.345 , 0.05  : 1.761 , 0.025 : 2.145 , 0.01  : 2.624 , 0.005 : 2.977 , 0.001 : 3.787},
15 : {0.25  : 0.691 , 0.1   : 1.341 , 0.05  : 1.753 , 0.025 : 2.131 , 0.01  : 2.602 , 0.005 : 2.947 , 0.001 : 3.733},
16 : {0.25  : 0.690 , 0.1   : 1.337 , 0.05  : 1.746 , 0.025 : 2.120 , 0.01  : 2.583 , 0.005 : 2.921 , 0.001 : 3.686},
17 : {0.25  : 0.689 , 0.1   : 1.333 , 0.05  : 1.740 , 0.025 : 2.110 , 0.01  : 2.567 , 0.005 : 2.898 , 0.001 : 3.646},
18 : {0.25  : 0.688 , 0.1   : 1.330 , 0.05  : 1.734 , 0.025 : 2.101 , 0.01  : 2.552 , 0.005 : 2.878 , 0.001 : 3.610},
19 : {0.25  : 0.688 , 0.1   : 1.328 , 0.05  : 1.729 , 0.025 : 2.093 , 0.01  : 2.539 , 0.005 : 2.861 , 0.001 : 3.579},
20 : {0.25  : 0.687 , 0.1   : 1.325 , 0.05  : 1.725 , 0.025 : 2.086 , 0.01  : 2.528 , 0.005 : 2.845 , 0.001 : 3.552},
21 : {0.25  : 0.686 , 0.1   : 1.323 , 0.05  : 1.721 , 0.025 : 2.080 , 0.01  : 2.518 , 0.005 : 2.831 , 0.001 : 3.527},
22 : {0.25  : 0.686 , 0.1   : 1.321 , 0.05  : 1.717 , 0.025 : 2.074 , 0.01  : 2.508 , 0.005 : 2.819 , 0.001 : 3.505},
23 : {0.25  : 0.685 , 0.1   : 1.319 , 0.05  : 1.714 , 0.025 : 2.069 , 0.01  : 2.500 , 0.005 : 2.807 , 0.001 : 3.485},
24 : {0.25  : 0.685 , 0.1   : 1.318 , 0.05  : 1.711 , 0.025 : 2.064 , 0.01  : 2.492 , 0.005 : 2.797 , 0.001 : 3.467},
25 : {0.25  : 0.684 , 0.1   : 1.316 , 0.05  : 1.708 , 0.025 : 2.060 , 0.01  : 2.485 , 0.005 : 2.787 , 0.001 : 3.450},
26 : {0.25  : 0.684 , 0.1   : 1.315 , 0.05  : 1.706 , 0.025 : 2.056 , 0.01  : 2.479 , 0.005 : 2.779 , 0.001 : 3.435},
27 : {0.25  : 0.684 , 0.1   : 1.314 , 0.05  : 1.703 , 0.025 : 2.052 , 0.01  : 2.473 , 0.005 : 2.771 , 0.001 : 3.421},
28 : {0.25  : 0.683 , 0.1   : 1.313 , 0.05  : 1.701 , 0.025 : 2.048 , 0.01  : 2.467 , 0.005 : 2.763 , 0.001 : 3.408},
29 : {0.25  : 0.683 , 0.1   : 1.311 , 0.05  : 1.699 , 0.025 : 2.045 , 0.01  : 2.462 , 0.005 : 2.756 , 0.001 : 3.396},
30 : {0.25  : 0.683 , 0.1   : 1.310 , 0.05  : 1.697 , 0.025 : 2.042 , 0.01  : 2.457 , 0.005 : 2.750 , 0.001 : 3.385},
31 : {0.25  : 0.682 , 0.1   : 1.309 , 0.05  : 1.696 , 0.025 : 2.040 , 0.01  : 2.453 , 0.005 : 2.744 , 0.001 : 3.375},
32 : {0.25  : 0.682 , 0.1   : 1.309 , 0.05  : 1.694 , 0.025 : 2.037 , 0.01  : 2.449 , 0.005 : 2.738 , 0.001 : 3.365},
33 : {0.25  : 0.682 , 0.1   : 1.308 , 0.05  : 1.692 , 0.025 : 2.035 , 0.01  : 2.445 , 0.005 : 2.733 , 0.001 : 3.356},
34 : {0.25  : 0.682 , 0.1   : 1.307 , 0.05  : 1.691 , 0.025 : 2.032 , 0.01  : 2.441 , 0.005 : 2.728 , 0.001 : 3.348},
35 : {0.25  : 0.682 , 0.1   : 1.306 , 0.05  : 1.690 , 0.025 : 2.030 , 0.01  : 2.438 , 0.005 : 2.724 , 0.001 : 3.340},
36 : {0.25  : 0.681 , 0.1   : 1.306 , 0.05  : 1.688 , 0.025 : 2.028 , 0.01  : 2.434 , 0.005 : 2.719 , 0.001 : 3.333},
37 : {0.25  : 0.681 , 0.1   : 1.305 , 0.05  : 1.687 , 0.025 : 2.026 , 0.01  : 2.431 , 0.005 : 2.715 , 0.001 : 3.326},
38 : {0.25  : 0.681 , 0.1   : 1.304 , 0.05  : 1.686 , 0.025 : 2.024 , 0.01  : 2.429 , 0.005 : 2.712 , 0.001 : 3.319},
39 : {0.25  : 0.681 , 0.1   : 1.304 , 0.05  : 1.685 , 0.025 : 2.023 , 0.01  : 2.426 , 0.005 : 2.708 , 0.001 : 3.313},
40 : {0.25  : 0.681 , 0.1   : 1.303 , 0.05  : 1.684 , 0.025 : 2.021 , 0.01  : 2.423 , 0.005 : 2.704 , 0.001 : 3.307},
41 : {0.25  : 0.681 , 0.1   : 1.303 , 0.05  : 1.683 , 0.025 : 2.020 , 0.01  : 2.421 , 0.005 : 2.701 , 0.001 : 3.301},
42 : {0.25  : 0.680 , 0.1   : 1.302 , 0.05  : 1.682 , 0.025 : 2.018 , 0.01  : 2.418 , 0.005 : 2.698 , 0.001 : 3.296},
43 : {0.25  : 0.680 , 0.1   : 1.302 , 0.05  : 1.681 , 0.025 : 2.017 , 0.01  : 2.416 , 0.005 : 2.695 , 0.001 : 3.291},
44 : {0.25  : 0.680 , 0.1   : 1.301 , 0.05  : 1.680 , 0.025 : 2.015 , 0.01  : 2.414 , 0.005 : 2.692 , 0.001 : 3.286},
45 : {0.25  : 0.680 , 0.1   : 1.301 , 0.05  : 1.679 , 0.025 : 2.014 , 0.01  : 2.412 , 0.005 : 2.690 , 0.001 : 3.281},
46 : {0.25  : 0.680 , 0.1   : 1.300 , 0.05  : 1.679 , 0.025 : 2.013 , 0.01  : 2.410 , 0.005 : 2.687 , 0.001 : 3.277},
47 : {0.25  : 0.680 , 0.1   : 1.300 , 0.05  : 1.678 , 0.025 : 2.012 , 0.01  : 2.408 , 0.005 : 2.685 , 0.001 : 3.273},
48 : {0.25  : 0.680 , 0.1   : 1.299 , 0.05  : 1.677 , 0.025 : 2.011 , 0.01  : 2.407 , 0.005 : 2.682 , 0.001 : 3.269},
49 : {0.25  : 0.680 , 0.1   : 1.299 , 0.05  : 1.677 , 0.025 : 2.010 , 0.01  : 2.405 , 0.005 : 2.680 , 0.001 : 3.265},
50 : {0.25  : 0.679 , 0.1   : 1.299 , 0.05  : 1.676 , 0.025 : 2.009 , 0.01  : 2.403 , 0.005 : 2.678 , 0.001 : 3.261}

}

def run(args=None):
    AllInOne = False
    # The value of AllInOne will be set to True for the "old"
    # computeconf (with fraction_for_solve) and will stay False for
    # the "new" computeconf (with MRP_directory_basename)

    try:
        conf_options_parser = construct_ph_options_parser("computeconf [options]")
        conf_options_parser.add_argument("--fraction-scenarios-for-solve",
                                       help="The fraction of scenarios that are allocated to finding a solution. Default is None.",
                                       action="store",
                                       dest="fraction_for_solve",
                                       type=float,
                                       default=None)
        conf_options_parser.add_argument("--number-samples-for-confidence-interval",
                                       help="The number of samples of scenarios that are allocated to the confidence inteval (n_g). Default is None.",
                                       action="store",
                                       dest="n_g",
                                       type=int,
                                       default=None)
        conf_options_parser.add_argument("--confidence-interval-alpha",
                                       help="The alpha level for the confidence interval. Default is 0.05",
                                       action="store",
                                       dest="confidence_interval_alpha",
                                       type=float,
                                       default=0.05)
        conf_options_parser.add_argument("--solve-xhat-with-ph",
                                       help="Perform xhat solve via PH rather than an EF solve. Default is False",
                                       action="store_true",
                                       dest="solve_xhat_with_ph",
                                       default=False)
        conf_options_parser.add_argument("--random-seed",
                                       help="Seed the random number generator used to select samples. Defaults to 0, indicating time seed will be used.",
                                       action="store",
                                       dest="random_seed",
                                       type=int,
                                       default=0)
        conf_options_parser.add_argument("--append-file",
                                       help="File to which summary run information is appended, for output tracking purposes.",
                                       action="store",
                                       dest="append_file",
                                       type=str,
                                       default=None)
        conf_options_parser.add_argument("--write-xhat-solution",
                                       help="Write xhat solutions (first stage variables only) to the append file? Defaults to False.",
                                       action="store_true",
                                       dest="write_xhat_solution",
                                       default=False)
        conf_options_parser.add_argument("--generate-weighted-cvar",
                                       help="Add a weighted CVaR term to the primary objective",
                                       action="store_true",
                                       dest="generate_weighted_cvar",
                                       default=False)
        conf_options_parser.add_argument("--cvar-weight",
                                       help="The weight associated with the CVaR term in the risk-weighted objective formulation. Default is 1.0. If the weight is 0, then *only* a non-weighted CVaR cost will appear in the EF objective - the expected cost component will be dropped.",
                                       action="store",
                                       dest="cvar_weight",
                                       type=float,
                                       default=1.0)
        conf_options_parser.add_argument("--risk-alpha",
                                       help="The probability threshold associated with cvar (or any future) risk-oriented performance metrics. Default is 0.95.",
                                       action="store",
                                       dest="risk_alpha",
                                       type=float,
                                       default=0.95)
        conf_options_parser.add_argument("--MRP-directory-basename",
                                       help="The basename for the replicate directories. It will be appended by the number of the group (loop over n_g). Default is None",
                                       action="store",
                                       dest="MRP_directory_basename",
                                       type=str,
                                       default=None)


        options = conf_options_parser.parse_args(args=args)
        # temporary hack
        options._ef_options = conf_options_parser._ef_options
        options._ef_options.import_argparse(options)
    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return _exc.code

    # seed the generator if a user-supplied seed is
    # provided. otherwise, python will seed from the current system
    # time.
    if options.random_seed > 0:
        random.seed(options.random_seed)

    start_time = time.time()
    if options.verbose:
        print("Importing model and scenario tree files")

    scenario_instance_factory = \
        ScenarioTreeInstanceFactory(options.model_directory,
                                    options.instance_directory)
    if _OLD_OUTPUT:
        print("Loading reference model and scenario tree")
    if options.verbose or options.output_times:
        print("Time to import model and scenario "
              "tree structure files=%.2f seconds"
              %(time.time() - start_time))

    try:

        scenario_tree = \
            scenario_instance_factory.generate_scenario_tree(
                downsample_fraction=options.scenario_tree_downsample_fraction,
                bundles=options.scenario_bundle_specification,
                random_bundles=options.create_random_bundles,
                random_seed=options.scenario_tree_random_seed,
                verbose=options.verbose)

        #
        # print the input tree for validation/information purposes.
        #
        if options.verbose:
            scenario_tree.pprint()

        #
        # validate the tree prior to doing anything serious
        #
        scenario_tree.validate()
        if options.verbose:
            print("Scenario tree is valid!")

        index_list, num_scenarios_for_solution, num_scenarios_per_sample = \
            partition_scenario_space(scenario_tree,
                                     options)

        #index_list = [0,3,5,7,1,4,6,8,2,9]
        #for ndx in index_list:
        #    print("%d: %s" % (ndx, scenario_tree._scenarios[ndx]._name))
        xhat_ph = find_candidate(scenario_instance_factory,
                                 index_list,
                                 num_scenarios_for_solution,
                                 scenario_tree,
                                 options)

        run_conf(scenario_instance_factory,
                 index_list,
                 num_scenarios_for_solution,
                 num_scenarios_per_sample,
                 scenario_tree,
                 xhat_ph,
                 options)

    finally:

        # delete temporary unarchived directories
        if scenario_instance_factory is not None:
            scenario_instance_factory.close()

def partition_scenario_space(full_scenario_tree,
                             options):

    if (options.MRP_directory_basename is not None) and \
       (options.fraction_for_solve is not None):
        raise RuntimeError("The two options --MRP-directory-"
                           "basename and --fraction-scenarios"
                           "-for-solve cannot both be set.")

    if options.MRP_directory_basename is None:
        AllInOne = True
        if options.fraction_for_solve is None:
            raise RuntimeError("Option --fraction-scenarios-"
                               "for-solve needs to be set.")
        if options.n_g is None:
            raise RuntimeError("Option --number-samples-for-"
                               "confidence-interval needs to be set.")

    print("Starting confidence interval calculation...")

    # randomly permute the indices to extract a subset to compute
    # xhat.
    scenario_count = len(full_scenario_tree._stages[-1]._tree_nodes)
    index_list = list(range(scenario_count))
    if AllInOne:
        random.shuffle(index_list)
        if options.verbose is True:
            print("Random permutation of the "
                  "scenario indices="+str(index_list))

    # figure out the scenario counts for both the xhat and confidence
    # interval computations.
    if AllInOne:
        num_scenarios_for_solution = \
            int(options.fraction_for_solve * scenario_count)
        n_g = options.n_g
        # 'n' in Morton's slides
        # integer division (cast to int required for python 3)
        num_scenarios_per_sample = \
            int((scenario_count - num_scenarios_for_solution) / n_g)
        wasted_scenarios = scenario_count - num_scenarios_for_solution - \
                           n_g * num_scenarios_per_sample
    else:
        num_scenarios_for_solution = scenario_count
        n_g = options.n_g
        ### num_scenario_per_sample
        biggest_scenario_number = 0
        scenariostructure_file = \
            open(os.path.join(options.MRP_directory_basename+"1",
                              "ScenarioStructure.dat"), "r")
        for line in scenariostructure_file:
            splitted_line = line.split(" ")
            if "Scenario" in str(splitted_line[0]):
                scenario_number_in_line = \
                    splitted_line[0].split("Scenario")[1]
                if scenario_number_in_line.isdigit():
                    if scenario_number_in_line > biggest_scenario_number:
                        biggest_scenario_number = int(scenario_number_in_line)
        num_scenarios_per_sample = biggest_scenario_number
        wasted_scenarios = scenario_count - num_scenarios_for_solution - \
                           n_g * num_scenarios_per_sample

    if num_scenarios_per_sample == 0:
        raise RuntimeError("Computed number of scenarios "
                           "per sample group equals 0 - "
                           +str(scenario_count - num_scenarios_for_solution)+
                           " scenarios cannot be divided into "
                           +str(n_g)+" groups!")

    print("Problem contains "+str(scenario_count)
          +" scenarios, of which "
          +str(num_scenarios_for_solution)
          +" will be used to find a solution xhat.")
    print("A total of "+str(n_g)+" groups of "
          +str(num_scenarios_per_sample)+
          " scenarios will be used to compute the "
          "confidence interval on xhat.")
    if wasted_scenarios > 0:
        print("A total of "+str(wasted_scenarios)
              +" scenarios will not be used.")

    if not options.solve_xhat_with_ph:
        if options.default_rho != "":
            raise ValueError("A default rho value can not be used "
                             "unless xhat is solved with ph")
        # provide a default rho value to avoid an error when
        # initializing the ph object for the xhat bundle
        # it will not be used
        options.default_rho = 1.0

    if len(full_scenario_tree._stages) > 2:
        raise RuntimeError("***ERROR: Confidence intervals are "
                           "available only for two stage stochastic "
                           "programs;"+str(len(full_scenario_tree._stages))+
                           " stages specified")
    return index_list, num_scenarios_for_solution, num_scenarios_per_sample

def find_candidate(scenario_instance_factory,
                   index_list,
                   num_scenarios_for_solution,
                   full_scenario_tree,
                   options):

    # create a ph object for finding the solution. we do this even if
    # we're solving the extensive form directly, mainly out of
    # convenience - we're leveraging the code in ph_for_bundle to
    # create the scenario tree and scenario instances.
    print("")
    print("Loading scenario instances and initializing "
          "scenario tree for xhat scenario bundle.")

    xhat_ph = None
    try:

        xhat_ph = ph_for_bundle(0,
                                num_scenarios_for_solution,
                                scenario_instance_factory,
                                full_scenario_tree,
                                index_list,
                                options)

        xhat_obj = None

        sense = xhat_ph._scenario_tree._scenarios[0]._objective_sense
        if sense == minimize:
            print("We are solving a MINIMIZATION problem.")
        else:
            print("We are solving a MAXIMIZATION problem.")

        if not options.solve_xhat_with_ph:
            print("Creating the xhat extensive form.")
            print("")
            print("Composite scenarios:")
            for scenario in xhat_ph._scenario_tree._scenarios:
                print(scenario._name)
            print("")

            if options.verbose:
                print("Time="+time.asctime())

            with ExtensiveFormAlgorithm(xhat_ph,
                                        options._ef_options,
                                        options_prefix="ef_") as ef:

                ef.build_ef()
                print("Solving the xhat extensive form.")
                # Instance preprocessing is managed within the
                # ph object automatically when required for a
                # solve. Since we are solving the instances
                # outside of the ph object, we will inform it
                # that it should complete the instance
                # preprocessing early
                xhat_ph._preprocess_scenario_instances()
                ef.solve(io_options=\
                         {'output_fixed_variable_bounds':
                          options.write_fixed_variables})
                xhat_obj = ef.objective
            """
            hatex_ef = create_ef_instance(
                xhat_ph._scenario_tree,
                verbose_output=options.verbose,
                generate_weighted_cvar=options.generate_weighted_cvar,
                cvar_weight=options.cvar_weight,
                risk_alpha=options.risk_alpha)

            if options.verbose:
                print("Time="+time.asctime())
            print("Solving the xhat extensive form.")

            # Instance preprocessing is managed within the ph object
            # automatically when required for a solve. Since we are
            # solving the instances outside of the ph object, we will
            # inform it that it should complete the instance preprocessing
            # early
            xhat_ph._preprocess_scenario_instances()

            ef_results = solve_ef(hatex_ef, options)

            if options.verbose:
                print("Loading extensive form solution.")
                print("Time="+time.asctime())

            # IMPT: the following method populates the _solution variables
            #       on the scenario tree nodes by forming an average of
            #       the corresponding variable values for all instances
            #       particpating in that node. if you don't do this, the
            #       scenario tree doesn't have the solution - and we need
            #       this below for variable fixing.
            if options.verbose:
                print("Computing extensive form solution from instances.")
                print("Time="+time.asctime())

            xhat_ph._scenario_tree.pullScenarioSolutionsFromInstances()
            xhat_ph._scenario_tree.snapshotSolutionFromScenarios()

            xhat_obj = xhat_ph._scenario_tree.findRootNode().computeExpectedNodeCost()
            """
            print("Extensive form objective value given xhat="
                  +str(xhat_obj))
        else:
            print("Solving for xhat via Progressive Hedging.")
            phretval = xhat_ph.solve()
            if phretval is not None:
                raise RuntimeError("No solution was obtained "
                                   "for scenario: "+phretval)
            # TBD - grab xhat_obj; complicated by the fact that PH may not
            #       have converged.
            # TBD - also not sure if PH calls
            #       snapshotSolutionFromAverages.
        print("The computation of xhat is complete - "
              "starting to compute confidence interval "
              "via sub-sampling.")

    finally:

        if xhat_ph is not None:

            # we are using the PHCleanup function for
            # convenience, but we need to prevent it
            # from shutting down the scenario_instance_factory
            # as it is managed outside this function
            xhat_ph._scenario_tree._scenario_instance_factory = None
            PHCleanup(xhat_ph)

    return xhat_ph

def run_conf(scenario_instance_factory,
             index_list,
             num_scenarios_for_solution,
             num_scenarios_per_sample,
             full_scenario_tree,
             xhat_ph,
             options):

    if options.MRP_directory_basename is None:
        AllInOne = True

    sense = xhat_ph._scenario_tree._scenarios[0]._objective_sense

    # in order to handle the case of scenarios that are not equally
    # likely, we will split the expectations for Gsupk
    # BUT we are going to assume that the groups themselves are
    # equally likely and just scale by n_g and n_g-1 for Gbar and VarG

    # really not always needed...
    # http://www.eecs.berkeley.edu/~mhoemmen/cs194/Tutorials/variance.pdf
    g_supk_of_xhat = []
    g_bar = 0
    sum_xstar_obj_given_xhat = 0
    n_g = options.n_g

    for k in range(1, n_g+1):

        gk_ph = None
        try:

            if AllInOne:

                start_index = num_scenarios_for_solution + \
                              (k-1)*num_scenarios_per_sample
                stop_index = start_index + num_scenarios_per_sample

                print("")
                print("Computing statistics for sample k="+str(k)+".")
                if options.verbose:
                    print("Bundle start index="+str(start_index)
                          +", stop index="+str(stop_index)+".")

                # compute this xstar solution for the EF associated with
                # sample k.

                print("Loading scenario instances and initializing "
                      "scenario tree for xstar scenario bundle.")

                gk_ph = ph_for_bundle(start_index,
                                      stop_index,
                                      scenario_instance_factory,
                                      full_scenario_tree,
                                      index_list,
                                      options)

            else:

                options.instance_directory = \
                    options.MRP_directory_basename+str(k)

                gk_ph = PHFromScratch(options)

            print("Creating the xstar extensive form.")
            print("")
            print("Composite scenarios:")
            for scenario in gk_ph._scenario_tree._scenarios:
                print (scenario._name)
            print("")
            gk_ef = ExtensiveFormAlgorithm(gk_ph,
                                           options._ef_options,
                                           options_prefix="ef_")
            gk_ef.build_ef()
            print("Solving the xstar extensive form.")
            # Instance preprocessing is managed within the
            # ph object automatically when required for a
            # solve. Since we are solving the instances
            # outside of the ph object, we will inform it
            # that it should complete the instance
            # preprocessing early
            gk_ph._preprocess_scenario_instances()
            gk_ef.solve(io_options=\
                        {'output_fixed_variable_bounds':
                         options.write_fixed_variables})
            xstar_obj = gk_ef.objective
            # assuming this is the absolute gap
            xstar_obj_gap = gk_ef.gap

            """
            gk_ef = create_ef_instance(gk_ph._scenario_tree,
                                       generate_weighted_cvar=options.generate_weighted_cvar,
                                       cvar_weight=options.cvar_weight,
                                       risk_alpha=options.risk_alpha)
            print("Solving the xstar extensive form.")

            # Instance preprocessing is managed within the ph object
            # automatically when required for a solve. Since we are
            # solving the instances outside of the ph object, we will
            # inform it that it should complete the instance preprocessing
            # early
            gk_ph._preprocess_scenario_instances()

            ef_results = solve_ef(gk_ef, options)

            # as in the computation of xhat, the following is required to form a
            # solution to the extensive form in the scenario tree itself.
            gk_ph._scenario_tree.pullScenarioSolutionsFromInstances()
            gk_ph._scenario_tree.snapshotSolutionFromScenarios()

            # extract the objective function value corresponding to the
            # xstar solution, along with any gap information.

            xstar_obj = gk_ph._scenario_tree.findRootNode().computeExpectedNodeCost()
            # assuming this is the absolute gap
            xstar_obj_gap = gk_ef.solutions[0].gap# ef_results.solution(0).gap
            """

            print("Sample extensive form objective value="+str(xstar_obj))


            # CVARHACK: if CPLEX barfed, keep trucking and bury our head
            # in the sand.
            if type(xstar_obj_gap) is UndefinedData:
                xstar_obj_bound = xstar_obj
                #EW#print("xstar_obj_bound= "+str(xstar_obj_bound))
            else:
                if sense == minimize:
                    xstar_obj_bound = xstar_obj - xstar_obj_gap
                else:
                    xstar_obj_bound = xstar_obj + xstar_obj_gap
                #EW#print("xstar_obj_bound= "+str(xstar_obj_bound))
                #EW#print("xstar_obj = "+str(xstar_obj))
                #EW#print("xstar_obj_gap = "+str(xstar_obj_gap))
            # TBD: ADD VERBOSE OUTPUT HERE

            # to get f(xhat) for this sample, fix the first-stage
            # variables and re-solve the extensive form.  note that the
            # fixing yields side-effects on the original gk_ef, but that
            # is fine as it isn't used after this point.
            print("Solving the extensive form given the xhat solution.")
            #xhat = pyomo.pysp.phboundbase.ExtractInternalNodeSolutionsforInner(xhat_ph)
            #
            # fix the first stage variables
            #
            gk_root_node = gk_ph._scenario_tree.findRootNode()
            #root_xhat = xhat[gk_root_node._name]
            root_xhat = xhat_ph._scenario_tree.findRootNode()._solution
            for variable_id in gk_root_node._standard_variable_ids:
                gk_root_node.fix_variable(variable_id,
                                          root_xhat[variable_id])

            # Push fixed variable statuses on instances (or
            # transmit to the phsolverservers), since we are not
            # calling the solve method on the ph object, we
            # need to do this manually
            gk_ph._push_fix_queue_to_instances()
            gk_ph._preprocess_scenario_instances()

            gk_ef.solve(io_options=\
                        {'output_fixed_variable_bounds':
                         options.write_fixed_variables})
            #ef_results = solve_ef(gk_ef, options)

            # we don't need the solution - just the objective value.
            #objective_name = "MASTER"
            #objective = gk_ef.find_component(objective_name)
            xstar_obj_given_xhat = gk_ef.objective

            print("Sample extensive form objective value given xhat="
                  +str(xstar_obj_given_xhat))

            #g_supk_of_xhat.append(xstar_obj_given_xhat - xstar_obj_bound)
            if sense == minimize:
                g_supk_of_xhat.append(xstar_obj_given_xhat - xstar_obj_bound)
            else:
                g_supk_of_xhat.append(- xstar_obj_given_xhat + xstar_obj_bound)
            g_bar += g_supk_of_xhat[k-1]
            sum_xstar_obj_given_xhat += xstar_obj_given_xhat

        finally:

            if gk_ph is not None:

                # we are using the PHCleanup function for
                # convenience, but we need to prevent it
                # from shutting down the scenario_instance_factory
                # as it is managed outside this function
                if gk_ph._scenario_tree._scenario_instance_factory is \
                   scenario_instance_factory:
                    gk_ph._scenario_tree._scenario_instance_factory = None
                PHCleanup(gk_ph)

    g_bar /= n_g
    # second pass for variance calculation (because we like storing
    # the g_supk)
    g_var = 0.0
    for k in range(0, n_g):
        print("g_supk_of_xhat[%d]=%12.6f"
              % (k+1, g_supk_of_xhat[k]))
        g_var = g_var + (g_supk_of_xhat[k] - g_bar) * \
                (g_supk_of_xhat[k] - g_bar)
    if n_g != 1:
        # sample var
        g_var = g_var / (n_g - 1)
    print("")
    print("Raw results:")
    print("g_bar= "+str(g_bar))
    print("g_stddev= "+str(math.sqrt(g_var)))
    print("Average f(xhat)= "+str(sum_xstar_obj_given_xhat / n_g))

    if n_g in t_table_values:
        print("")
        print("Results summary:")
        t_table_entries = t_table_values[n_g]
        for key in sorted(iterkeys(t_table_entries)):
            print("Confidence interval width for alpha="+str(key)
                  +" is "+str(g_bar + (t_table_entries[key] * \
                                       math.sqrt(g_var) / \
                                       math.sqrt(n_g))))
    else:
        print("No built-in t-table entries for "+str(n_g)
              +" degrees of freedom - cannot calculate confidence interval width")

    if options.write_xhat_solution:
        print("")
        print("xhat solution:")
        scenario_tree = xhat_ph._scenario_tree
        first_stage = scenario_tree._stages[0]
        root_node = first_stage._tree_nodes[0]
        for key, val in iteritems(root_node._solutions):
            for idx in val:
                if val[idx] != 0.0:
                    print("%s %s %s" % (str(key), str(idx), str(val[idx]())))

    scenario_count = len(full_scenario_tree._stages[-1]._tree_nodes)
    if options.append_file is not None:
        output_file = open(options.append_file, "a")
        output_file.write("\ninstancedirectory, "
                          +str(options.instance_directory)
                          +", seed, "+str(options.random_seed)
                          +", N, "+str(scenario_count)
                          +", hatn, "+str(num_scenarios_for_solution)
                          +", n_g, "+str(options.n_g)
                          +", Eoffofxhat, "
                          +str(sum_xstar_obj_given_xhat / n_g)
                          +", gbar, "+str(g_bar)+", sg, "
                          +str(math.sqrt(g_var))+", objforxhat, "
                          +str(xhat_obj)+", n,"
                          +str(num_scenarios_per_sample))

        if n_g in t_table_values:
            t_table_entries = t_table_values[n_g]
            for key in sorted(iterkeys(t_table_entries)):
                output_file.write(" , alpha="+str(key)+" , "
                                  +str(g_bar + (t_table_entries[key] * \
                                                math.sqrt(g_var) / \
                                                math.sqrt(n_g))))

        if options.write_xhat_solution:
            output_file.write(" , ")
            scenario_tree = xhat_ph._scenario_tree
            first_stage = scenario_tree._stages[0]
            root_node = first_stage._tree_nodes[0]
            for key, val in iteritems(root_node._solutions):
                for idx in val:
                    if val[idx] != 0.0:
                        output_file.write("%s %s %s"
                                          % (str(key),
                                             str(idx),
                                             str(val[idx]())))
        output_file.close()
        print("")
        print("Results summary appended to file="
              +options.append_file)

    xhat_ph.release_components()

#
# routine to create a down-sampled (bundled) scenario tree and the associated PH object.
#
def ph_for_bundle(bundle_start,
                  bundle_stop,
                  scenario_instance_factory,
                  full_scenario_tree,
                  index_list,
                  options):

    scenarios_to_bundle = []
    for i in range(bundle_start, bundle_stop):
        scenarios_to_bundle.append(full_scenario_tree.\
                                   _scenarios[index_list[i]]._name)

    ph = None
    try:

        scenario_tree = \
            GenerateScenarioTreeForPH(options,
                                      scenario_instance_factory,
                                      include_scenarios=scenarios_to_bundle)

        ph = PHAlgorithmBuilder(options, scenario_tree)

    except:

        if ph is not None:
            ph.release_components()

        raise

    return ph

#   print "DLW says: first stage vars are fixed; maybe we need to delete any constraints with only first stage vars due to precision issues"

#
# the main script routine starts here - and is obviously pretty simple!
#

@pyomo_command('computeconf', "Compute the confidence for a SP solution")
def main(args=None):
    # to import plugins
    import pyomo.environ

    try:
        run(args=args)
    except ValueError as str:
        print("VALUE ERROR:")
        print(str)
    except IOError as str:
        print("IO ERROR:")
        print(str)
    except ApplicationError as str:
        print("APPLICATION ERROR:")
        print(str)
    except RuntimeError as str:
        print("RUN-TIME ERROR:")
        print(str)
    except:
        print("Encountered unhandled exception")
        traceback.print_exc()
