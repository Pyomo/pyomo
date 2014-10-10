#! /usr/bin/env python

#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2010 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import os
import random
import math
import time

from pyomo.pysp.scenariotree import *
from pyomo.pysp.phinit import *
from pyomo.pysp.ph import *
from pyomo.pysp.ef import *

# this is a hack, in order to pick up the UndefinedData class. this is needed currently, as
# CPLEX is periodically barfing on cvar formulations, yielding an undefined gap. technically,
# the gap is defined and the solution is feasible, but a correct fix to the CPLEX plugin
# would yield a complete failure to solve cvar problems. see related hacks below, searching
# for CVARHACK.
from pyomo.opt.results.container import *

from pyomo.opt import SolverStatus, TerminationCondition, SolutionStatus
from pyomo.misc import pyomo_command

from six import iteritems, iterkeys, advance_iterator

# to avoid the pain of user lookup of parameter in t-tables, we provide decent coverage automatically.
# feel free to add more values!!!! maps degrees-of-freedom to (alpha,t-statistic) pairs.

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
    # The value of AllInOne will be set to True for the "old" computeconf (with fraction_for_solve) and will stay False for the "new" computeconf (with MRP_directory_basename)

    try:
        conf_options_parser = construct_ph_options_parser("computeconf [options]")
        conf_options_parser.add_option("--fraction-scenarios-for-solve",
                                       help="The fraction of scenarios that are allocated to finding a solution. Default is None.",
                                       action="store",
                                       dest="fraction_for_solve",
                                       type="float",
                                       default=None)
        conf_options_parser.add_option("--number-samples-for-confidence-interval",
                                       help="The number of samples of scenarios that are allocated to the confidence inteval (n_g). Default is None.",
                                       action="store",
                                       dest="n_g",
                                       type="int",
                                       default=None)
        conf_options_parser.add_option("--confidence-interval-alpha",
                                       help="The alpha level for the confidence interval. Default is 0.05",
                                       action="store",
                                       dest="confidence_interval_alpha",
                                       type="float",
                                       default=0.05)
        conf_options_parser.add_option("--solve-xhat-with-ph",
                                       help="Perform xhat solve via PH rather than an EF solve. Default is False",
                                       action="store_true",
                                       dest="solve_xhat_with_ph",
                                       default=False)
        conf_options_parser.add_option("--random-seed",
                                       help="Seed the random number generator used to select samples. Defaults to 0, indicating time seed will be used.",
                                       action="store",
                                       dest="random_seed",
                                       type="int",
                                       default=0)
        conf_options_parser.add_option("--append-file",
                                       help="File to which summary run information is appended, for output tracking purposes.",
                                       action="store",
                                       dest="append_file",
                                       type="string",
                                       default=None)
        conf_options_parser.add_option("--write-xhat-solution",
                                       help="Write xhat solutions (first stage variables only) to the append file? Defaults to False.",
                                       action="store_true",
                                       dest="write_xhat_solution",
                                       default=False)
        conf_options_parser.add_option("--generate-weighted-cvar",
                                       help="Add a weighted CVaR term to the primary objective",
                                       action="store_true",
                                       dest="generate_weighted_cvar",
                                       default=False)
        conf_options_parser.add_option("--cvar-weight",
                                       help="The weight associated with the CVaR term in the risk-weighted objective formulation. Default is 1.0. If the weight is 0, then *only* a non-weighted CVaR cost will appear in the EF objective - the expected cost component will be dropped.",
                                       action="store",
                                       dest="cvar_weight",
                                       type="float",
                                       default=1.0)
        conf_options_parser.add_option("--risk-alpha",
                                       help="The probability threshold associated with cvar (or any future) risk-oriented performance metrics. Default is 0.95.",
                                       action="store",
                                       dest="risk_alpha",
                                       type="float",
                                       default=0.95)
        conf_options_parser.add_option("--MRP-directory-basename",
                                       help="The basename for the replicate directories. It will be appended by the number of the group (loop over n_g). Default is None",
                                       action="store",
                                       dest="MRP_directory_basename",
                                       type="string",
                                       default=None)


        (options, args) = conf_options_parser.parse_args(args=args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return

    # seed the generator if a user-supplied seed is provided. otherwise,
    # python will seed from the current system time.
    if options.random_seed > 0:
        random.seed(options.random_seed)

    # import the reference model and create the scenario tree - no scenario instances yet.
    print("Loading reference model and scenario tree")
    scenario_instance_factory, full_scenario_tree = load_models(options)
    try:

        if (scenario_instance_factory is None) or (full_scenario_tree is None):
            raise RuntimeError("***ERROR: Failed to initialize model and/or the scenario tree data.")

        # load_model gets called again, so lets make sure unarchived directories are used
        options.model_directory = scenario_instance_factory._model_filename
        options.instance_directory = scenario_instance_factory._data_filename

        run_conf(scenario_instance_factory, full_scenario_tree, options)

    finally:

        # delete temporary unarchived directories
        if scenario_instance_factory is not None:
            scenario_instance_factory.close()

def run_conf(scenario_instance_factory, full_scenario_tree, options):

    if (options.MRP_directory_basename is not None) and (options.fraction_for_solve is not None):
        raise RuntimeError("The two options --MRP-directory-basename and --fraction-scenarios-for-solve cannot both be set.")

    if options.MRP_directory_basename is None:
        AllInOne = True
        if options.fraction_for_solve is None:
            raise RuntimeError("Option --fraction-scenarios-for-solve needs to be set.")
        if options.n_g is None:
            raise RuntimeError("Option --number-samples-for-confidence-interval needs to be set.")

    print("Starting confidence interval calculation...")

    scenario_count = len(full_scenario_tree._stages[-1]._tree_nodes)
    if len(full_scenario_tree._stages) > 2:
        raise RuntimeError("***ERROR: Confidence intervals are available only for two stage stochastic programs;"+str(len(full_scenario_tree._stages))+" stages specified")

    # randomly permute the indices to extract a subset to compute xhat.
    index_list = range(scenario_count)
    if AllInOne is True:
        random.shuffle(index_list)
        if options.verbose is True:
            print("Random permutation of the scenario indices="+str(index_list))

    # figure out the scenario counts for both the xhat and confidence interval computations.
    if AllInOne is True:
        num_scenarios_for_solution = int(options.fraction_for_solve * scenario_count)
        n_g = options.n_g
        num_scenarios_per_sample = int((scenario_count - num_scenarios_for_solution) / n_g) # 'n' in Morton's slides
        wasted_scenarios = scenario_count - num_scenarios_for_solution - n_g * num_scenarios_per_sample
    else:
        num_scenarios_for_solution = scenario_count
        n_g = options.n_g
        ### num_scenario_per_sample
        biggest_scenario_number = 0
        scenariostructure_file = open(options.MRP_directory_basename + "1" + os.sep + "ScenarioStructure.dat","r")
        for line in scenariostructure_file:
            splitted_line = line.split(" ")
            if "Scenario" in str(splitted_line[0]):
                scenario_number_in_line = splitted_line[0].split("Scenario")[1]
                if scenario_number_in_line.isdigit():
                    if scenario_number_in_line > biggest_scenario_number:
                        biggest_scenario_number = int(scenario_number_in_line)
        num_scenarios_per_sample = biggest_scenario_number
        wasted_scenarios = scenario_count - num_scenarios_for_solution - n_g * num_scenarios_per_sample

    if num_scenarios_per_sample is 0:
        raise RuntimeError("Computed number of scenarios per sample group equals 0 - "+str(scenario_count - num_scenarios_for_solution)+" scenarios cannot be divided into "+str(n_g)+" groups!")

    print("Problem contains "+str(scenario_count)+" scenarios, of which "+str(num_scenarios_for_solution)+" will be used to find a solution xhat.")
    print("A total of "+str(n_g)+" groups of "+str(num_scenarios_per_sample)+" scenarios will be used to compute the confidence interval on xhat.")
    if wasted_scenarios > 0:
        print("A total of "+str(wasted_scenarios)+" scenarios will not be used.")

    if not options.solve_xhat_with_ph:
        if options.default_rho != "":
            raise ValueError("A default rho value can not be used "
                             "unless xhat is solved with ph")
        # provide a default rho value to avoid an error when
        # initializing the ph object for the xhat bundle
        # it will not be used
        options.default_rho = 1.0

    # create a ph object for finding the solution. we do this even if
    # we're solving the extensive form directly, mainly out of
    # convenience - we're leveraging the code in ph_for_bundle to
    # create the scenario tree and scenario instances.
    print("")
    print("Loading scenario instances and initializing scenario tree for xhat scenario bundle.")
    xhat_ph = ph_for_bundle(0, num_scenarios_for_solution, scenario_instance_factory, full_scenario_tree, index_list, options)
    xhat_obj = None

    if find_active_objective(xhat_ph._scenario_tree._scenarios[0]._instance,safety_checks=True).is_minimizing():
        print("We are solving a MINIMIZATION problem.")
        sense = 'min'
    else:
        print("We are solving a MAXIMIZATION problem.")
        sense = 'max'

    if not options.solve_xhat_with_ph:
        print("Creating the xhat extensive form.")
        print("")
        print("Composite scenarios:")
        for scenario in xhat_ph._scenario_tree._scenarios:
            print (scenario._name)
        print("")

        if options.verbose is True:
            print("Time="+time.asctime())
        hatex_ef = create_ef_instance(xhat_ph._scenario_tree,
                                      xhat_ph._instances,
                                      verbose_output=options.verbose,
                                      generate_weighted_cvar=options.generate_weighted_cvar,
                                      cvar_weight=options.cvar_weight,
                                      risk_alpha=options.risk_alpha)
        if options.verbose is True:
            print("Time="+time.asctime())
        print("Solving the xhat extensive form.")

        # Instance preprocessing is managed within the ph object
        # automatically when required for a solve. Since we are
        # solving the instances outside of the ph object, we will
        # inform it that it should complete the instance preprocessing
        # early
        xhat_ph._preprocess_scenario_instances()

        ef_results = solve_ef(hatex_ef, xhat_ph._instances, options)

        if options.verbose is True:
            print("Loading extensive form solution.")
            print("Time="+time.asctime())
        hatex_ef.load(ef_results)
        # IMPT: the following method populates the _solution variables on the scenario tree
        #       nodes by forming an average of the corresponding variable values for all
        #       instances particpating in that node. if you don't do this, the scenario tree
        #       doesn't have the solution - and we need this below for variable fixing.
        if options.verbose is True:
            print("Computing extensive form solution from instances.")
            print("Time="+time.asctime())
        xhat_ph._scenario_tree.pullScenarioSolutionsFromInstances()
        xhat_ph._scenario_tree.snapshotSolutionFromScenarios()
        objective_name, objective = advance_iterator(iteritems(ef_results.solution(0).objective)) # take the first one - we don't know how to deal with multiple objectives.
        xhat_obj = float(objective.value)  ## DLW to JPW: we need the gap too
        """if sense == 'min':
           xhat_obj = float(ef_results.solution(0).objective['f'].value)  ## DLW to JPW: we need the gap too
        else:
           xhat_obj = -float(ef_results.solution(0).objective['f'].value)  ## DLW to JPW: we need the gap too"""
        print("Extensive form objective value given xhat="+str(xhat_obj))
    else:
        print("Solving for xhat via Progressive Hedging.")
        phretval = xhat_ph.solve()
        if phretval is not None:
            raise RuntimeError("No solution was obtained for scenario: "+phretval)
        # TBD - grab xhat_obj; complicated by the fact that PH may not have converged.
        # TBD - also note sure if PH calls snapshotSolutionFromAverages.
    print("The computation of xhat is complete - starting to compute confidence interval via sub-sampling.")

    # in order to handle the case of scenarios that are not equally likely, we will split the expectations for Gsupk
    # BUT we are going to assume that the groups themselves are equally likely and just scale by n_g and n_g-1 for Gbar and VarG
    g_supk_of_xhat = [] # really not always needed... http://www.eecs.berkeley.edu/~mhoemmen/cs194/Tutorials/variance.pdf
    g_bar = 0
    sum_xstar_obj_given_xhat = 0

    for k in range(1, n_g+1):
        if AllInOne is True:
            start_index = num_scenarios_for_solution + (k-1)*num_scenarios_per_sample
            stop_index = start_index + num_scenarios_per_sample 
            print("")
            print("Computing statistics for sample k="+str(k)+".")
            if options.verbose is True:
                print("Bundle start index="+str(start_index)+", stop index="+str(stop_index)+".")

            # compute this xstar solution for the EF associated with sample k.

            print("Loading scenario instances and initializing scenario tree for xstar scenario bundle.")
            gk_ph = ph_for_bundle(start_index, stop_index, scenario_instance_factory, full_scenario_tree, index_list, options)
        else:
            options.instance_directory = options.MRP_directory_basename+str(k)
            if os.path.isdir(options.instance_directory) is not True:
                raise RuntimeError("The instance directory "+str( options.instance_directory)+" does not exist.")

            scenario_instance_factory_for_soln, scenario_tree_for_soln = load_models(options)

            if (reference_model is None) or (scenario_tree_for_soln is None) or (scenario_tree_instance is None):
                raise RuntimeError("***ERROR: Failed to initialize reference model and/or the scenario tree.")

            gk_ph = create_ph_from_scratch(options, scenario_instance_factory_for_soln, scenario_tree_for_soln)

        print("Creating the xstar extensive form.")
        print("")
        print("Composite scenarios:")
        for scenario in gk_ph._scenario_tree._scenarios:
            print (scenario._name)
        print("")

        gk_ef = create_ef_instance(gk_ph._scenario_tree,
                                   gk_ph._instances,
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

        ef_results = solve_ef(gk_ef, gk_ph._instances, options)
        gk_ef.load(ef_results)

        # as in the computation of xhat, the following is required to form a
        # solution to the extensive form in the scenario tree itself.
        gk_ph._scenario_tree.pullScenarioSolutionsFromInstances()
        gk_ph._scenario_tree.snapshotSolutionFromScenarios()

        # extract the objective function value corresponding to the xstar solution, along with any gap information.

        objective_name, objective = advance_iterator(iteritems(ef_results.solution(0).objective)) # take the first one - we don't know how to deal with multiple objectives.
        xstar_obj = float(objective.value)  ## DLW to JPW: we need the gap too, and to add/subtract is as necessary.
        """if sense == 'min':
           xstar_obj = float(ef_results.solution(0).objective['f'].value)  ## DLW to JPW: we need the gap too, and to add/subtract is as necessary.
        else:
           xstar_obj = -float(ef_results.solution(0).objective['f'].value)  ## DLW to JPW: we need the gap too, and to add/subtract is as necessary."""

        print("Sample extensive form objective value="+str(xstar_obj))

        xstar_obj_gap = ef_results.solution(0).gap # assuming this is the absolute gap
        # CVARHACK: if CPLEX barfed, keep trucking and bury our head in the sand.
        if type(xstar_obj_gap) is UndefinedData:
            xstar_obj_bound = xstar_obj
            #EW#print("xstar_obj_bound= "+str(xstar_obj_bound))
        else:
            if sense == 'min':
                xstar_obj_bound = xstar_obj - xstar_obj_gap
            else:
                xstar_obj_bound = xstar_obj + xstar_obj_gap
            #EW#print("xstar_obj_bound= "+str(xstar_obj_bound))
            #EW#print("xstar_obj = "+str(xstar_obj))
            #EW#print("xstar_obj_gap = "+str(xstar_obj_gap))
        # TBD: ADD VERBOSE OUTPUT HERE

        # to get f(xhat) for this sample, fix the first-stage variables and re-solve the extensive form.
        # note that the fixing yields side-effects on the original gk_ef, but that is fine as it isn't
        # used after this point.
        print("Solving the extensive form given the xhat solution.")
        for scenario in  gk_ph._scenario_tree._scenarios:
            instance = gk_ph._instances[scenario._name]
            fix_first_stage_vars(xhat_ph._scenario_tree, instance)
            instance.preprocess()
        gk_ef.preprocess() # to account for the fixed variables in the scenario instances.
        ef_results = solve_ef(gk_ef, gk_ph._instances, options)

        gk_ef.load(ef_results)

        # we don't need the solution - just the objective value.
        objective_name = "MASTER"
        objective = gk_ef.find_component(objective_name)
        xstar_obj_given_xhat = objective()
        """if sense == 'min':
           xstar_obj_given_xhat = float(ef_results.solution(0).objective['f'].value)
        else:
           xstar_obj_given_xhat = -float(ef_results.solution(0).objective['f'].value)"""
        print("Sample extensive form objective value given xhat="+str(xstar_obj_given_xhat))

        #g_supk_of_xhat.append(xstar_obj_given_xhat - xstar_obj_bound)
        if sense == 'min':
            g_supk_of_xhat.append(xstar_obj_given_xhat - xstar_obj_bound)
        else:
            assert sense == 'max'
            g_supk_of_xhat.append(- xstar_obj_given_xhat + xstar_obj_bound)
        g_bar += g_supk_of_xhat[k-1]
        sum_xstar_obj_given_xhat += xstar_obj_given_xhat

    g_bar = g_bar / n_g
    # second pass for variance calculation (because we like storing the g_supk)
    g_var = 0
    for k in range(0, n_g):
        print("g_supk_of_xhat[%d]=%12.6f" % (k+1, g_supk_of_xhat[k]))
        g_var = g_var + (g_supk_of_xhat[k] - g_bar) * (g_supk_of_xhat[k] - g_bar)
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
            print("Confidence interval width for alpha="+str(key)+" is "+str(g_bar + (t_table_entries[key] * math.sqrt(g_var) / math.sqrt(n_g))))
    else:
        print("No built-in t-table entries for "+str(n_g)+" degrees of freedom - cannot calculate confidence interval width")

    if options.write_xhat_solution is True:
        print("")
        print("xhat solution:")
        scenario_tree = xhat_ph._scenario_tree
        first_stage = scenario_tree._stages[0]
        root_node = first_stage._tree_nodes[0]
        for key, val in iteritems(root_node._solutions):
            for idx in val:
                if val[idx] != 0.0:
                    print("%s %s %s" % (str(key), str(idx), str(val[idx]())))

    if options.append_file is not None:
        output_file = open(options.append_file, "a")
        output_file.write("\ninstancedirectory, "+str(options.instance_directory)+", seed, "+str(options.random_seed)+", N, "+str(scenario_count)+", hatn, "+str(num_scenarios_for_solution)+", n_g, "+str(options.n_g)+", Eoffofxhat, "+str(sum_xstar_obj_given_xhat / n_g)+", gbar, "+str(g_bar)+", sg, "+str(math.sqrt(g_var))+", objforxhat, "+str(xhat_obj)+", n,"+str(num_scenarios_per_sample))

        if n_g in t_table_values:
            t_table_entries = t_table_values[n_g]
            for key in sorted(iterkeys(t_table_entries)):
                output_file.write(" , alpha="+str(key)+" , "+str(g_bar + (t_table_entries[key] * math.sqrt(g_var) / math.sqrt(n_g))))

        if options.write_xhat_solution is True:
            output_file.write(" , ")
            scenario_tree = xhat_ph._scenario_tree
            first_stage = scenario_tree._stages[0]
            root_node = first_stage._tree_nodes[0]
            for key, val in iteritems(root_node._solutions):
                for idx in val:
                    if val[idx] != 0.0:
                        output_file.write("%s %s %s" % (str(key), str(idx), str(val[idx]())))
        output_file.close()
        print("")
        print("Results summary appended to file="+options.append_file)

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
        scenarios_to_bundle.append(full_scenario_tree._scenarios[index_list[i]]._name)

    if options.verbose is True:
        print("Creating PH object for scenario bundle="+str(scenarios_to_bundle))

    scenario_tree_for_soln = ScenarioTree(scenariotreeinstance=scenario_instance_factory._scenario_tree_instance,
                                          scenariobundlelist=scenarios_to_bundle)
    if scenario_tree_for_soln.validate() is False:
        raise RuntimeError("***ERROR: Bundled scenario tree is invalid!!!")

    ph = create_ph_from_scratch(options, scenario_instance_factory, scenario_tree_for_soln)
    return ph

#
# fixes the first stage variables in the input instance to the values
# of the solution indicated in the input scenario tree.
#
def fix_first_stage_vars(scenario_tree, instance):

    stage = scenario_tree._stages[0]
    root_node = stage._tree_nodes[0] # there should be only one root node!
    scenario_tree_var = root_node._solution
    nodeid_to_var_map = instance._ScenarioTreeSymbolMap.bySymbol

    for variable_id in root_node._variable_ids:
        
        variable = nodeid_to_var_map[variable_id]

        if variable.stale is False:
            
            fix_value = scenario_tree_var[variable_id]
            
            if isinstance(variable.domain, IntegerSet) or isinstance(variable.domain, BooleanSet):
                    
                fix_value = int(round(fix_value))

            variable.fixed = True
            variable.value = fix_value


#   print "DLW says: first stage vars are fixed; maybe we need to delete any constraints with only first stage vars due to precision issues"

#==============================================
def solve_ef(master_instance, scenario_instances, options):

    ef_solver = SolverFactory(options.solver_type)
    if ef_solver is None:
        raise ValueError("Failed to create solver of type="+options.solver_type+" for use in extensive form solve")
    if len(options.ef_solver_options) > 0:
        print("Initializing ef solver with options="+str(options.ef_solver_options))
        ef_solver.set_options("".join(options.ef_solver_options))
    if options.ef_mipgap is not None:
        if (options.ef_mipgap < 0.0) or (options.ef_mipgap > 1.0):
            raise ValueError("Value of the mipgap parameter for the EF solve must be on the unit interval; value specified=" + str(options.ef_mipgap))
        else:
            ef_solver.mipgap = options.ef_mipgap
    if options.keep_solver_files is True:
        ef_solver.keepfiles = True

    ef_solver_manager = SolverManagerFactory(options.solver_manager_type)
    if ef_solver is None:
        raise ValueError("Failed to create solver manager of type="+options.solver_type+" for use in extensive form solve")

    print("Solving extensive form.")
    if ef_solver.warm_start_capable():
        ef_action_handle = ef_solver_manager.queue(master_instance, 
                                                   opt=ef_solver, 
                                                   warmstart=False, 
                                                   tee=options.output_ef_solver_log, 
                                                   symbolic_solver_labels=options.symbolic_solver_labels)
    else:
        ef_action_handle = ef_solver_manager.queue(master_instance, 
                                                   opt=ef_solver, 
                                                   tee=options.output_ef_solver_log,
                                                   symbolic_solver_labels=options.symbolic_solver_labels)
    results = ef_solver_manager.wait_for(ef_action_handle)
    print("solve_ef() terminated.")

    # check the return code - if this is anything but "have a solution", we need to bail.
    if (results.solver.status == SolverStatus.ok) and \
            ((results.solver.termination_condition == TerminationCondition.optimal) or \
                 ((len(results.solution) > 0) and (results.solution(0).status == SolutionStatus.optimal))):                                                                  
        return results

    raise RuntimeError("Extensive form was infeasible!")

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
    except pyutilib.common.ApplicationError as str:
        print("APPLICATION ERROR:")
        print(str)
    except RuntimeError as str:
        print("RUN-TIME ERROR:")
        print(str)
    except:
        print("Encountered unhandled exception")
        traceback.print_exc()
