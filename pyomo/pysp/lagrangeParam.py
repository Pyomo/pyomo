#! /usr/bin/env python

# This is virgin parametric lagrange multiplier search
# no pre-processing and no gap-closing...just testing algorithm

import sys
import os
import inspect  # used for debug when Result was not returning correctly
import random
import math
import time
import datetime
import operator
import types
from pyomo.pysp.scenariotree import *
from pyomo.pysp.phinit import *
from pyomo.pysp.ph import *
from pyomo.pysp.ef import *
from pyomo.opt import SolverFactory

import pyomo.pysp.lagrangeutils as lagrUtil
##############################################################

def datetime_string():
   return "datetime = "+str(datetime.datetime.now())

###################################
def run(args=None):
###################################

   print("RUNNING - run args=%s\n" % str(args))

   import pyomo.modeling

   def LagrangeParametric(args=None):
      class Object(object): pass
      Result = Object()
      Result.status = 'LagrangeParam begins '+ datetime_string() + '...running new ph'
      def new_ph():
         scenario_instance_factory, scenario_tree = load_models(options)
         if scenario_instance_factory is None or scenario_tree is None:
            print("internal error in new_ph")
            exit(2)
         return create_ph_from_scratch(options, scenario_instance_factory, scenario_tree)

      blanks = "                          "  # used for formatting print statements
# options used
      betaMin       = options.beta_min
      betaMax       = options.beta_max
      betaTol       = options.beta_tol
      gapTol        = options.Lagrange_gap
      minProb       = options.min_prob
      maxIntervals  = options.max_intervals
      maxTime       = options.max_time
      IndVarName    = options.indicator_var_name
      multName      = options.lambda_parm_name
      CCStageNum    = options.stage_num
      csvPrefix     = options.csvPrefix
      verbosity     = options.verbosity
      verbosity = 2 # override for debug (= 3 to get super-debug)
      HGdebug = 0   # special debug (not public)
# local...may become option
      optTol = gapTol
####################################################################
      STARTTIME = time.time()

      Result.status = "options set"
      if verbosity > 1:
        print("From LagrangeParametric, status = %s\tSTARTTIME = %s" \
                % (str(getattr(Result,'status')), str(STARTTIME)))

      ph = new_ph()
      Result.ph = ph
      rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]   # use rootnode to loop over scenarios
      ReferenceInstance = ph._instances[rootnode._scenarios[0]._name]  # arbitrary scenario

      if find_active_objective(ph._scenario_tree._scenarios[0]._instance,safety_checks=True).is_minimizing():
         sense = 'min'
      else:
         sense = 'max'

      scenario_count = len(full_scenario_tree._stages[-1]._tree_nodes)
      if options.verbosity > 0: print("%s %s scenarios\n" % (str(sense),str(scenario_count)))

# initialize
      Result.status = 'starting at '+datetime_string()
      if verbosity > 0:
         print(Result.status)
      ScenarioList = []
      lambdaval = 0.
      lagrUtil.Set_ParmValue(ph, multName,lambdaval)
      sumprob = 0.
      minprob = 1.
      maxprob = 0.
      for scenario in rootnode._scenarios:
         instance = ph._instances[scenario._name]
         sname = scenario._name
         sprob = scenario._probability
         sumprob = sumprob + sprob
         minprob = min(minprob,sprob)
         maxprob = max(maxprob,sprob)
         ScenarioList.append([sname,sprob])
         getattr(instance,IndVarName).value = 0   # fixed = 0 to get PR point at b=0
         getattr(instance,IndVarName).fixed = True
      instance.preprocess()
      ScenarioList.sort(key=operator.itemgetter(1))   # sorts from min to max probability
      if verbosity > 0:
         print("probabilities sum to %f range: %f to %f" % (sumprob,minprob,maxprob))
      Result.ScenarioList = ScenarioList

# Write ScenarioList = name, probability in csv file sorted by probability
      outName = csvPrefix + 'ScenarioList.csv'
      print("writing to %s\n" % outName)
      outFile = file(outName,'w')
      for scenario in ScenarioList:
         print >>outFile, scenario[0]+ ", " +str(scenario[1])
      outFile.close()
      Result.ScenarioList = ScenarioList

      addstatus = 'Scenario List written to ' + csvPrefix+'ScenarioList.csv'
      Result.status = Result.status + '\n' + addstatus
      if verbosity > 0:
         print(addstatus)

      if verbosity > 0:
         print("solve begins %s\n" % datetime_string())
         print("\t- lambda = %f\n" % lambdaval)
      SolStat, zL = lagrUtil.solve_ph_code(ph, options)
      if verbosity > 0:
         print("solve ends %s\n" % datetime_string())
         print("\t- status = %s\n" % str(SolStat))
         print("\t- zL = %s\n" % str(zL))

      bL = Compute_ExpectationforVariable(ph, IndVarName, CCStageNum)
      if bL > 0:
         print("** bL = %s > 0 (all %s = 0)\n" % (str(bL), str(IndVarName)))
         return Result

      if verbosity > 0:  print("Initial optimal obj = %s for bL = %s\n" % (str(zL), str(bL)))

      for scenario in ScenarioList:
         sname = scenario[0]
         instance = ph._instances[sname]
         getattr(instance,IndVarName).value = 1  # fixed = 1 to get PR point at b=1
      instance.preprocess()

      if verbosity > 0:
        print("solve begins %s\n" % datetime_string())
        print("\t- lambda = %s\n" % str(lambdaval))
      SolStat, zU = lagrUtil.solve_ph_code(ph, options)
      if verbosity > 0:
        print("solve ends %s\n" % datetime_string())
        print("\t- status = %s\n" % str(SolStat))
        print("\t- zU = %s\n" % str(zU))
      if not SolStat[0:3] == 'opt':
         print(str(SolStat[0:3])+" is not 'opt'\n")
         addstatus = "** Solution is non-optimal...aborting"
         print(addstatus)
         Result.status = Result.status + "\n" + addstatus
         return Result

      bU = Compute_ExpectationforVariable(ph, IndVarName, CCStageNum)
      if bU < 1.- betaTol and verbosity > 0:
         print("** Warning:  bU = %s  < 1" % str(bU))

### enumerate points in PR space (all but one scenario)
#      Result.lbz = [ [0,bL,zL], [None,bU,zU] ]
#      for scenario in rootnode._scenarios:
#         sname = scenario._name
#         instance = ph._instances[sname]
#         print "excluding scenario",sname
#         getattr(instance,IndVarName).value = 0
#         print sname,"value =",getattr(instance,IndVarName).value,getattr(instance,IndVarName).fixed
#         SolStat, z = lagrUtil.solve_ph_code(ph, options)
#         b = Compute_ExpectationforVariable(ph, IndVarName, CCStageNum)
#         print "solve ends with status =",SolStat,"(b, z) =",b,z
#         getattr(instance,IndVarName).value = 1
#         Result.lbz.append([None,b,z])
#         for t in instance.TimePeriods:
#           print "Global at",t,"=",instance.posGlobalLoadGenerateMismatch[t].value, \
#                '-',instance.negGlobalLoadGenerateMismatch[t].value,"=",\
#                    instance.GlobalLoadGenerateMismatch[t].value,\
#               "\tDemand =",instance.TotalDemand[t].value, ",",\
#                "Reserve =",instance.ReserveRequirement[t].value
#
#      PrintPRpoints(Result.lbz)
#      return Result
#### end enumeration
########################################################################

      if verbosity > 1:
         print("We have bU = %s ...about to free all %s for %d scenarios\n" % \
                (str(bU), str(IndVarName), len(ScenarioList)))

      for scenario in ScenarioList:
         sname = scenario[0]
         instance = ph._instances[sname]
# free scenario selection variable
         getattr(instance,IndVarName).fixed = False
         instance.preprocess() # dlw indent
      if verbosity > 1:
         print("\tall %s freed; elapsed time = %f\n" % (str(IndVarName), time.time() - STARTTIME))

# initialize with the two endpoints
      Result.lbz = [ [0.,bL,zL], [None,bU,zU] ]
      Result.selections = [[], ScenarioList]
      NumIntervals = 1
      if verbosity > 0:
         print("Initial relative Lagrangian gap = %f maxIntervals = %d\n" % (1-zL/zU, maxIntervals))
         if verbosity > 1:
            print("entering while loop %s\n" % datetime_string())
         print("\n")

############ main loop to search intervals #############
########################################################
      while NumIntervals < maxIntervals:
         lapsedTime = time.time() - STARTTIME
         if lapsedTime > maxTime:
            addstatus = '** max time reached ' + str(lapsedTime)
            print(addstatus)
            Result.status = Result.status + '\n' + addstatus
            break
         if verbosity > 1:
            print("Top of while with %d intervals elapsed time = %f\n" % (NumIntervals, lapsedTime))
            PrintPRpoints(Result.lbz)

         lambdaval = None
### loop over PR points to find first unfathomed interval to search ###
         for PRpoint in range(1,len(Result.lbz)):
            if Result.lbz[PRpoint][0] == None:
# multiplier = None means interval with upper endpoint at PRpoint not fathomed
               bL = Result.lbz[PRpoint-1][1]
               zL = Result.lbz[PRpoint-1][2]
               bU = Result.lbz[PRpoint][1]
               zU = Result.lbz[PRpoint][2]
               lambdaval = (zU - zL) / (bU - bL)
               break

#############################
# Exited from the for loop
         if verbosity > 1:
            print("exited for loop with PRpoint = %s ...lambdaval = %f\n" % (PRpoint, lambdaval))
         if lambdaval == None: break # all intervals are fathomed

         if verbosity > 1: PrintPRpoints(Result.lbz)
         if verbosity > 0:
            print("Searching for b in [%s, %s] with %s = %f\n" % (str(round(bL,4)), str(round(bU,4)), multName, lambdaval))

# search interval (bL,bU)
         lagrUtil.Set_ParmValue(ph, multName,lambdaval)
         if verbosity > 0:
            print("solve begins %s\n" % datetime_string())
            print("\t- %s = %f\n" % (multName, lambdaval))

         #########################################################
         SolStat, Lagrangian = lagrUtil.solve_ph_code(ph, options)
         #########################################################
         if not SolStat[0:3] == 'opt':
            addstatus = "** Solution status " + SolStat + " is not optimal"
            print(addstatus)
            Result.status = Result.status + "\n" + addstatus
            return Result

         b = Compute_ExpectationforVariable(ph, IndVarName, CCStageNum)
         z = Lagrangian + lambdaval*b
         if verbosity > 0:
            print("solve ends %s" % datetime_string())
            print("\t- Lagrangian = %f\n" % Lagrangian)
            print("\t- b = %s\n" % str(b))
            print("\t- z = %s\n" % str(z))
            print("\n")

# We have PR point (b,z), which may be new or one of the endpoints
##################################################################

######### Begin tolerance tests ##########
# Test that b is in [bL,bU]
         if verbosity > 1: print("\ttesting b")
         if b < bL - betaTol or b > bU + betaTol:
            addstatus = "** fatal error: probability (= " + str(b) + \
                ") is outside interval, (" + str(bL) + ", " + str(bU) + ")"
            addstatus = addstatus + "\n\t(tolerance = " + str(betaTol) + ")"
            print(addstatus+'\n')
            Result.status = Result.status + addstatus
            return Result
# Test that z is in [zL,zU]
         if verbosity > 1: print("testing z\n")
# using optTol as absolute tolerance (not relative)
#   ...if we reconsider, need to allow negative z-values
         if z < zL - optTol or z > zU + optTol:
            addstatus = "** fatal error: obj (= " + str(z) + \
                ") is outside interval, (" + str(zL) + ", " + str(zU) + ")"
            print(addstatus+'\n')
            Result.status = Result.status + addstatus
            return Result

# Ok, we have (b,z) in [(bL,zL), (bU,zU)], at least within tolerances

         oldLagrangian = zL - lambdaval*bL
# ensure lambdaval set such that endpoints have same Lagrangian value
# (this is probably unnecessary, but check anyway)
         if abs(oldLagrangian - (zU - lambdaval*bU)) > optTol*abs(oldLagrangian):
            addstatus = "** fatal error: Lagrangian at (bL,zL) = " + \
                str(oldLagrangian) + " not= " + str(zU-lambdaval*bU) + \
                "\n\t(optTol = " + str(optTol) + ")"
            Result.status = Result.status + addstatus
            return Result

# no more fatal error tests...need to know if (b,z) is endpoint or new

         if verbosity > 1: print("No anomalies...testing if b = bL or bU")

# Test if endpoint is an alternative optimum of Lagrangian
# ...using optTol as *relative* tolerance
# (could use other reference values -- eg, avg or max of old and new Lagrangian values)
         refValue =  max( min( abs(oldLagrangian), abs(Lagrangian) ), 1.)
         alternativeOpt = abs( oldLagrangian - Lagrangian ) <= optTol*refValue

# alternativeOpt = True means we computed point (b,z) is alternative optimum such that:
#   case 1: (b,z) = endpoint, in which case we simply fathom [bL,bU] by setting PRpoint
#            to [lambdaval,bU,zU] (the numeric value of multiplier means fathomed)
#   case 2: (b,z) is new PR point on line segment, in which case we split into
#           [bL,b] and [b,bU], with both fathomed

         if verbosity > 1:
            print("oldLagrangian = %s" % str(oldLagrangian))
            if alternativeOpt: print(":= Lagrangian = %s\n" % str(Lagrangian ))
            else: print("> Lagrangian = %s\n" % str(Lagrangian))

         if alternativeOpt:
# setting multiplier of (bU,zU) to a numeric fathoms the interval [bL,bU]
            Result.lbz[PRpoint][0] = lambdaval

# test if (b,z) is an endpoint
         newPRpoint = abs(b-bL) > betaTol and abs(b-bU) > betaTol
         if not newPRpoint:
# ...(b,z) is NOT an endpoint (or sufficiently close), so split and fathom
            if verbosity > 1:
               print("\tnot an endpoint\tlbz = %s\n" % str(Result.lbz[PRpoint]))
            if verbosity > 0:
               print("Lagangian solution is new PR point on line segment of (" \
                  + str(bL) + ", " + str(bU) +")\n")
               print("\tsplitting (bL,bU) into (bL,b) and (b,bU), both fathomed\n")
# note:  else ==> b = bL or bU, so we do nothing, having already fathomed [bL,bU]

# (b,z) is new PR point, so split interval (still in while loop)
##########################################
# alternative optimum ==> split & fathom: (bL,b), (b,bU)
         if verbosity > 1:
            print("\talternativeOpt %f newPRpoint = %f\n" % (alternativeOpt, newPRpoint))
         if newPRpoint:
            NumIntervals += 1
            if alternativeOpt:
               if verbosity > 1: print("\tInsert [lambdaval,b,z] at %f\n" % PRpoint)
               Result.lbz = Insert([lambdaval,b,z],PRpoint,Result.lbz)
               addstatus = "Added PR point on line segment of envelope"
               if verbosity > 0: print(addstatus+'\n')
            else:
               if verbosity > 1: print("\tInsert [None,b,z] at %f\n" % PRpoint)
               Result.lbz = Insert([None,b,z],PRpoint,Result.lbz)
               addstatus = "new envelope extreme point added (interval split, not fathomed)"
            Result.status = Result.status + "\n" + addstatus

            if verbosity > 1:
               print("...after insertion:\n")
               PrintPRpoints(Result.lbz)

# get the selections of new point (ie, scenarios for which delta=1)
            Selections = []
            for scenario in ScenarioList:
               instance = ph._instances[scenario[0]]
               if getattr(instance,IndVarName).value == 1:
                  Selections.append(scenario)
            Result.selections = Insert(Selections,PRpoint,Result.selections)

            if verbosity > 0:
               print("Interval "+str(PRpoint)+", ["+str(bL)+", "+str(bU)+ \
                 "] split at ("+str(b)+", "+str(z)+")\n")
               print("\tnew PR point has "+str(len(Selections))+" selections\n")

            if verbosity > 1: print("test that selections list aligned with lbz\n")
            if not len(Result.lbz) == len(Result.selections):
               print("** fatal error: lbz not= selections\n")
               PrintPRpoints(Result.lbz)
               print("Result.selections:\n")
               for i in range(Result.selections): print("%d %f\n" % (i,Result.selections[i]))
               return Result

# ok, we have split and/or fathomed interval
         if NumIntervals >= maxIntervals:
# we are about to leave while loop due to...
            addstatus = "** terminating because number of intervals = " + \
                    str(NumIntervals) + " >= max = " + str(maxIntervals)
            if verbosity > 0: print(addstatus+'\n')
            Result.status = Result.status + "\n" + addstatus

# while loop continues
         if verbosity > 1:
            print("bottom of while loop\n")
            PrintPRpoints(Result.lbz)

###################################################
# end while NumIntervals < maxIntervals:
#     ^ this is indentation of while loop
################ end while loop ###################

      if verbosity > 1:  print("\nend while loop...setting multipliers\n")
      for i in range(1,len(Result.lbz)):
         db = Result.lbz[i][1] - Result.lbz[i-1][1]
         dz = Result.lbz[i][2] - Result.lbz[i-1][2]
         if dz > 0:
            Result.lbz[i][0] = dz/db
         else:
            #print "dz =",dz," at ",i,": ",Result.lbz[i]," -",Result.lbz[i-1]
            Result.lbz[i][0] = 0
      if verbosity > 0: PrintPRpoints(Result.lbz)

      addstatus = '\nLagrange multiplier search ends'+datetime_string()
      if verbosity > 0:
         print(addstatus+'\n')
      Result.status = Result.status + addstatus

      outName = csvPrefix + "PRoptimal.csv"
      outFile = file(outName,'w')
      if verbosity > 0: print("writing PR points to "+outName+'\n')
      for lbz in Result.lbz: print >>outFile, str(lbz[1])+ ", " +str(lbz[2])
      outFile.close()

      outName = csvPrefix + "OptimalSelections.csv"
      outFile = file(outName,'w')
      if verbosity > 0: print("writing optimal selections for each PR point to "+csvPrefix+'PRoptimal.csv\n')
      for selections in Result.selections:
         char = ""
         thisSelection = ""
         for slist in selections:
            if slist:
               thisSelection = thisSelection + char + slist[0]
               char = ","
         print >>outFile, thisSelection
      outFile.close()
      if verbosity > 0:
         print("\nReturning status:\n %s \n=======================\n" % Result.status)

################################
      if verbosity > 2:
         print("\nAbout to return...Result attributes: %d\n" % len(inspect.getmembers(Result)))
         for attr in inspect.getmembers(Result): print(attr[0])
         print("\n===========================================\n")
# LagrangeParametric ends here
      return Result
################################


####################################### start run ####################################

   AllInOne = False

########################
# options defined here
########################
   try:
      conf_options_parser = construct_ph_options_parser("lagrange [options]")
      conf_options_parser.add_option("--beta-min",
                                     help="The min beta level for the chance constraint. Default is 0",
                                     action="store",
                                     dest="beta_min",
                                     type="float",
                                     default=0.)
      conf_options_parser.add_option("--beta-max",
                                     help="The beta level for the chance constraint. Default is 1.",
                                     action="store",
                                     dest="beta_max",
                                     type="float",
                                     default=1.)
      conf_options_parser.add_option("--beta-tol",
                                     help="Tolerance for testing equality to beta. Default is 1e-5",
                                     action="store",
                                     dest="beta_tol",
                                     type="float",
                                     default=1e-5)
      conf_options_parser.add_option("--Lagrange-gap",
                                     help="The (relative) Lagrangian gap acceptable for the chance constraint. Default is 10^-4",
                                     action="store",
                                     type="float",
                                     dest="Lagrange_gap",
                                     default=0.0001)
      conf_options_parser.add_option("--min-prob",
                                     help="Tolerance for testing probability > 0. Default is 1e-9",
                                     action="store",
                                     dest="min_prob",
                                     type="float",
                                     default=1e-5)
      conf_options_parser.add_option("--max-intervals",
                                     help="The max number of intervals generated; if causes termination, non-fathomed intervals have multiplier=None.  Default = 100.",
                                     action="store",
                                     dest="max_intervals",
                                     type="int",
                                     default=100)
      conf_options_parser.add_option("--max-time",
                                     help="Maximum time (seconds). Default is 3600.",
                                     action="store",
                                     dest="max_time",
                                     type="float",
                                     default=3600)
      conf_options_parser.add_option("--lambda-parm-name",
                                     help="The name of the lambda parameter in the model. Default is lambdaMult",
                                     action="store",
                                     dest="lambda_parm_name",
                                     type="string",
                                     default="lambdaMult")
      conf_options_parser.add_option("--indicator-var-name",
                                     help="The name of the indicator variable for the chance constraint. The default is delta",
                                     action="store",
                                     dest="indicator_var_name",
                                     type="string",
                                     default="delta")
      conf_options_parser.add_option("--stage-num",
                                     help="The stage number of the CC indicator variable (number, not name). Default is 2",
                                     action="store",
                                     dest="stage_num",
                                     type="int",
                                     default=2)
      conf_options_parser.add_option("--csvPrefix",
                                     help="Output file name.  Default is ''",
                                     action="store",
                                     dest="csvPrefix",
                                     type="string",
                                     default='')
      conf_options_parser.add_option("--verbosity",
                                     help="verbosity=0 is no extra output, =1 is medium, =2 is debug, =3 super-debug. Default is 1.",
                                     action="store",
                                     dest="verbosity",
                                     type="int",
                                     default=1)
# The following needed for solve_ph_code in lagrangeutils
      conf_options_parser.add_option("--solve-with-ph",
                                     help="Perform solves via PH rather than an EF solve. Default is False",
                                     action="store_true",
                                     dest="solve_with_ph",
                                     default=False)
##HG: deleted params filed as deletedParam.py
#######################################################################################################

      (options, args) = conf_options_parser.parse_args(args=args)
   except SystemExit:
      # the parser throws a system exit if "-h" is specified - catch
      # it to exit gracefully.
      return

   # create the reference instances and the scenario tree - no scenario instances yet.
   if options.verbosity > 0:
               print("Loading reference model and scenario tree")
   scenario_instance_factory, full_scenario_tree = load_models(options)

   try:
      if (scenario_instance_factory is None) or (full_scenario_tree is None):
         raise RuntimeError, "***ERROR: Failed to initialize the model and/or scenario tree data."

      # load_model gets called again, so lets make sure unarchived directories are used
      options.model_directory = scenario_instance_factory._model_filename
      options.instance_directory = scenario_instance_factory._data_filename

########## Here is where multiplier search is called from run() ############
      Result = LagrangeParametric()
#####################################################################################
   finally:

      # delete temporary unarchived directories
      scenario_instance_factory.close()

   if options.verbosity > 0:
      print("\n===========================================\n")
      print("\nreturned from LagrangeParametric\n")
      if options.verbosity > 2:
         print("\nFrom run, Result should have status and ph objects...\n")
         for attr in inspect.getmembers(Result): print(attr)
         print("\n===========================================\n")

   try:
     status = Result.status
     print("status = "+str(Result.status))
   except:
     print("status not defined\n")
     sys.exit()

   try:
      lbz = Result.lbz
      PrintPRpoints(lbz)
      outFile = file(options.csvPrefix+"PRoptimal.csv",'w')
      for lbz in Result.lbz: print >>outFile, str(lbz[1])+ ", " +str(lbz[2])
      outFile.close()
   except:
      print("Result.lbz not defined\n")
      sys.exit()

   try:
      ScenarioList = Result.ScenarioList
      ScenarioList.sort(key=operator.itemgetter(1))
      outFile = file(options.csvPrefix+"ScenarioList.csv",'w')
      for scenario in ScenarioList: print >>outFile, scenario[0]+", "+str(scenario[1])
      outFile.close()
   except:
      print("Result.ScenarioList not defined\n")
      sys.exit()

########################### run ends here ##############################

########## begin function definitions ################
def Compute_ExpectationforVariable(ph, IndVarName, CCStageNum):
   SumSoFar = 0.0
   node_probability = 0.0
   stage = ph._scenario_tree._stages[CCStageNum-1]
   for tree_node in stage._tree_nodes:
      for scenario in tree_node._scenarios:
         instance = ph._instances[scenario._name]
         #print "scenario._probability:",scenario._probability
         node_probability += scenario._probability
         #print "node_probability:",node_probability
         #print "getattr(instance, IndVarName).value:",getattr(instance, IndVarName).value
         SumSoFar += scenario._probability * getattr(instance, IndVarName).value
         #print "SumSoFar:",SumSoFar
   return SumSoFar / node_probability

#######################################

def Insert(newpoint,location,List):
    newList = []
    for i in range(location): newList.append(List[i])
    newList.append(newpoint)
    for i in range(location,len(List)): newList.append(List[i])
    return newList

#######################################

def ismember(List,member):  # designed to test 1st member of each list in List (ie, 1st column)
   for i in List:
      if len(i[0]) == 0: continue   # in case list contains empty list
      if i[0] == member: return True
   return

#######################################

def putcommas(num):
   snum = str(num)
   decimal = snum.find('.')
   if decimal >= 0:
      frac = snum[decimal:]
      snum = snum[0:decimal]
   else: frac = ''
   if len(snum) < 4: return snum + frac
   else: return putcommas(snum[:len(snum)-3]) + "," + snum[len(snum)-3:len(snum)] + frac

#######################################

def HGdebugSolve(ph,options,ScenarioList):
   print("HGdebugSolve prints attributes after solving\n")
   SolStat, Lagrangian = lagrUtil.solve_ph_code(ph, options)
   IndVarName = options.indicator_var_name
   for scenario in ScenarioList:
       sname = scenario[0]
       instance = ph._instances[sname]
       if getattr(instance,IndVarName).fixed: stat = "fixed"
       else: stat = "free"
       lambdaval = getattr(instance, options.lambda_parm_name).value
       #print IndVarName+"("+sname+") =",getattr(instance,IndVarName).value, stat, "\tlambda =",lambdaval
   b = Compute_ExpectationforVariable(ph, IndVarName, options.stage_num)
   z = Lagrangian + lambdaval*b
   #print "L = ",z," - ",lambdaval,"x",b
   return SolStat, Lagrangian

def PrintPRpoints(PRlist):
   if len(PRlist) == 0:
      print("No PR points\n")
   else:
      print("%d PR points:\n" % len(PRlist))
      blanks = "                      "
      print("            lambda        beta-probability       min cost \n")
      for row in PRlist:
         b = round(row[1],4)
         z = round(row[2])
# lambda = row[0] could be float, string, or None
         sl = str(row[0])
         sl = blanks[0:20-len(sl)] + sl
         sb = str(b)
         sb = blanks[0:20-len(sb)] + sb
         sz = putcommas(z)
         sz = blanks[2:20-len(sz)] + sz
         #print sl+" "+sb+" "+sz
      print("==================================================================\n")
   return

###########
def ZeroOneIndexListsforVariable(ph, IndVarName, CCStageNum):
   # return lists across scenarios of the zero value scenarios and one value scenarios
   # for unindexed variable in the ph object for a stage (one based)
   # in this routine we trust that it is binary
   ZerosList = []
   OnesList = []

   stage = ph._scenario_tree._stages[CCStageNum-1]
   for tree_node in stage._tree_nodes:
      for scenario in tree_node._scenarios:
         instance = ph._instances[scenario._name]
         locval = getattr(instance, IndVarName).value
         #print locval
         if locval < 0.5:
            ZerosList.append(scenario)
         else:
            OnesList.append(scenario)
   return [ZerosList,OnesList]

###########
def PrintanIndexList(IndList):
   # show some useful information about an index list (note: indexes are scenarios)
   print("Zeros:\n")
   for i in IndList[0]:
      print(i._name+'\n')
   print("Ones:\n")
   for i in IndList[1]:
      print(i._name+'\n')

###########
def ReturnIndexListNames(IndList):
   ListNames=[[],[]]
   #print "Zeros:"
   for i in IndList[0]:
      ListNames[0].append(i._name)
   #print "Ones:"
   for i in IndList[1]:
      ListNames[1].append(i._name)
   return ListNames

#
# the main script routine starts here
#

if __name__ == "__main__":

   run()

# RESTORE THE BELOW ASAP
#try:
#    run()
#except ValueError, str:
#    print "VALUE ERROR:"
#    print str
#except IOError, str:
#    print "IO ERROR:"
#    print str
#except pyutilib.common.ApplicationError, str:
#    print "APPLICATION ERROR:"
#    print str
#except RuntimeError, str:
#    print "RUN-TIME ERROR:"
##    print str
#except:
#   print "Encountered unhandled exception", sys.exc_info()[0]
#   traceback.print_exc()
