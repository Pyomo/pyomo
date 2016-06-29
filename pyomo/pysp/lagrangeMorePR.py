#! /usr/bin/env python
#
# This reads PRoptimal.csv, which contains the sequence of PR points that defines the optimal response function, F^*
# Then, it computes more PR points and appends to PRmore.csv
#
#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import time
import datetime
import operator
import copy

from pyomo.opt import SolverManagerFactory

from pyomo.pysp.scenariotree.instance_factory import \
   ScenarioTreeInstanceFactory
from pyomo.pysp.phinit import (construct_ph_options_parser,
                               GenerateScenarioTreeForPH,
                               PHFromScratch)
from pyomo.pysp.phutils import find_active_objective
from pyomo.pysp import lagrangeutils as lagrUtil

def datetime_string():
   return "datetime = "+str(datetime.datetime.now())

###################################
def run(args=None):
###################################

   # to import plugins
   import pyomo.environ
   import pyomo.solvers.plugins.smanager.phpyro

   def LagrangeMorePR(args=None):
      print("lagrangeMorePR begins %s" % datetime_string())
      blanks = "                          "  # used for formatting print statements
      class Object(object): pass
      Result = Object()

# options used
      betaTol       = options.beta_tol          # tolerance used to separate b-values
      IndVarName    = options.indicator_var_name
      multName      = options.lambda_parm_name
      CCStageNum    = options.stage_num
      MaxMorePR     = options.max_number         # max PR points to be generated (above F^* with all delta fixed)
      MaxTime       = options.max_time           # max time before terminate
      csvPrefix = options.csvPrefix          # input filename prefix (eg, case name)
      probFileName  = options.probFileName       # name of file containing probabilities
##HG override
#      options.verbosity = 2
      verbosity     = options.verbosity

      Result.status = 'starting '+datetime_string()
      STARTTIME = time.time()

      ph = PHFromScratch(options)
      rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]   # use rootnode to loop over scenarios

      if find_active_objective(ph._scenario_tree._scenarios[0]._instance,safety_checks=True).is_minimizing():
         print("We are solving a MINIMIZATION problem.")
      else:
         print("We are solving a MAXIMIZATION problem.")

# initialize
      ScenarioList = []
      with open(csvPrefix+"ScenarioList.csv",'r') as inputFile:
         for line in inputFile.readlines():
            L = line.split(',')
            ScenarioList.append([L[0],float(L[1])])

      addstatus = str(len(ScenarioList))+' scenarios read from file: ' + csvPrefix+'ScenarioList.csv'
      if verbosity > 0: print(addstatus)
      Result.status = Result.status + '\n' + addstatus

      PRoptimal = []
      with open(csvPrefix+"PRoptimal.csv",'r') as inputFile:
         for line in inputFile.readlines():
            bzS = line.split(',')
            PRoptimal.append( [None, float(bzS[0]), float(bzS[1])] )

      addstatus = str(len(PRoptimal))+' PR points read from file: '+ csvPrefix+'PRoptimal.csv (envelope function)'
      if verbosity > 0:
         print(addstatus)
      Result.status = Result.status + '\n' + addstatus
# ensure PR points on envelope function are sorted by probability
      PRoptimal.sort(key=operator.itemgetter(1))

      PRoptimal[0][0] = 0   # initial lambda (for b=0)
      for p in range(1,len(PRoptimal)):
         dz = PRoptimal[p][2] - PRoptimal[p-1][2]
         db = PRoptimal[p][1] - PRoptimal[p-1][1]
         PRoptimal[p][0] = dz/db
      if verbosity > 0:
         PrintPRpoints(PRoptimal)
      Result.PRoptimal = PRoptimal

      lambdaval = 0.
      lagrUtil.Set_ParmValue(ph, options.lambda_parm_name,lambdaval)

      # IMPORTANT: Preprocess the scenario instances
      #            before fixing variables, otherwise they
      #            will be preprocessed out of the expressions
      #            and the output_fixed_variable_bounds option
      #            will have no effect when we update the
      #            fixed variable values (and then assume we
      #            do not need to preprocess again because
      #            of this option).
      ph._preprocess_scenario_instances()

## read scenarios to select for each PR point on envelope function
      with open(csvPrefix+"OptimalSelections.csv",'r') as inputFile:
         OptimalSelections = []
         for line in inputFile.readlines():
            if len(line) == 0: break # eof
            selections = line.split(',')
            L = len(selections)
            Ls = len(selections[L-1])
            selections[L-1] = selections[L-1][0:Ls-1]
            if verbosity > 1:
               print(str(selections))
            OptimalSelections.append(selections)

      Result.OptimalSelections = OptimalSelections

      addstatus = str(len(OptimalSelections)) + ' Optimal selections read from file: ' \
            + csvPrefix + 'OptimalSelections.csv'
      Result.status = Result.status + '\n' + addstatus

      if len(OptimalSelections) == len(PRoptimal):
         if verbosity > 0:
            print(addstatus)
      else:
         addstatus = addstatus + '\n** Number of selections not equal to number of PR points'
         print(addstatus)
         Result.status = Result.status + '\n' + addstatus
         print(str(OptimalSelections))
         print((PRoptimal))
         return Result

#####################################################################################

# get probabilities
      if probFileName is None:
# ...generate from widest gap regions
         PRlist = FindPRpoints(options, PRoptimal)
      else:
# ...read probabilities
         probList = []
         with open(probFileName,'r') as inputFile:
            if verbosity > 0:
               print("reading from probList = "+probFileName)
            for line in inputFile.readlines():  # 1 probability per line
               if len(line) == 0:
                  break
               prob = float(line)
               probList.append(prob)

         if verbosity > 0:
            print("\t "+str(len(probList))+" probabilities")
         if verbosity > 1:
            print(str(probList))
         PRlist = GetPoints(options, PRoptimal, probList)
         if verbosity > 1:
            print("PRlist:")
            for interval in PRlist:
               print(str(interval))

# We now have PRlist = [[i, b], ...], where b is in PRoptimal interval (i-1,i)
      addstatus = str(len(PRlist)) + ' probabilities'
      if verbosity > 0:
         print(addstatus)
      Result.status = Result.status + '\n' + addstatus

#####################################################################################

      lapsedTime = time.time() - STARTTIME
      addstatus = 'Initialize complete...lapsed time = ' + str(lapsedTime)
      if verbosity > 1:
         print(addstatus)
      Result.status = Result.status + '\n' + addstatus

#####################################################################################

      if verbosity > 1:
        print("\nlooping over Intervals to generate PR points by flipping heuristic")
      Result.morePR = []
      for interval in PRlist:
         lapsedTime = time.time() - STARTTIME
         if lapsedTime > MaxTime:
            addstatus = '** lapsed time = ' + str(lapsedTime) + ' > max time = ' + str(MaxTime)
            if verbosity > 0: print(addstatus)
            Result.status = Result.status + '\n' + addstatus
            break

         i = interval[0] # = PR point index
         b = interval[1] # = target probability to reach by flipping from upper endpoint
         bU = PRoptimal[i][1]   # = upper endpoint
         bL = PRoptimal[i-1][1] # = lower endpoint
         if verbosity > 1:
            print( "target probability = "+str(b)+" < bU = PRoptimal[" + str(i) + "][1]" \
                 " and > bL = PRoptimal["+str(i-1)+"][1]")
         if b < bL or b > bU:
            addstatus = '** probability = '+str(b) + ', not in gap interval: (' \
                + str(bL) + ', ' + str(bU) + ')'
            print(addstatus)
            print(str(PRoptimal))
            print(str(PRlist))
            Result.status = Result.status + '\n' + addstatus
            return Result

         if verbosity > 1:
            print( "i = "+str(i)+" : Starting with bU = "+str(bU)+" having "+ \
                str(len(OptimalSelections[i]))+ " selections:")
            print(str(OptimalSelections[i]))

# first fix all scenarios = 0
         for sname, sprob in ScenarioList:
            scenario = ph._scenario_tree.get_scenario(sname)
            lagrUtil.FixIndicatorVariableOneScenario(ph,
                                                     scenario,
                                                     IndVarName,
                                                     0)

# now fix optimal selections = 1
         for sname in OptimalSelections[i]:
            scenario = ph._scenario_tree.get_scenario(sname)
            lagrUtil.FixIndicatorVariableOneScenario(ph,
                                                     scenario,
                                                     IndVarName,
                                                     1)

# flip scenario selections from bU until we reach b (target probability)
         bNew = bU
         for sname, sprob in ScenarioList:
            scenario = ph._scenario_tree.get_scenario(sname)
            if bNew - sprob < b:
               continue
            instance = ph._instances[sname]
            if getattr(instance, IndVarName).value == 0:
               continue
            bNew = bNew - sprob
            # flipped scenario selection
            lagrUtil.FixIndicatorVariableOneScenario(ph,
                                                     scenario,
                                                     IndVarName,
                                                     0)
            if verbosity > 1:
               print("\tflipped "+sname+" with prob = "+str(sprob)+" ...bNew = "+str(bNew))

         if verbosity > 1:
            print("\tflipped selections reach "+str(bNew)+" >= target = "+str(b)+" (bL = "+str(bL)+")")
         if bNew <= bL + betaTol or bNew >= bU - betaTol:
            if verbosity > 0:
               print("\tNot generating PR point...flipping from bU failed")
            continue # to next interval in list

 # ready to solve to get cost for fixed scenario selections associated with probability = bNew

         if verbosity > 1:
# check that scenarios are fixed as they should be
            totalprob = 0.
            for scenario in ScenarioList:
               sname = scenario[0]
               sprob = scenario[1]
               instance = ph._instances[sname]
               print("fix "+sname+" = "+str(getattr(instance,IndVarName).value)+\
                  " is "+str(getattr(instance,IndVarName).fixed)+" probability = "+str(sprob))
               if getattr(instance,IndVarName).value == 1:
                  totalprob = totalprob + sprob
               lambdaval = getattr(instance, multName).value
            print("\ttotal probability = %f" % totalprob)

# solve (all delta fixed); lambda=0, so z = Lagrangian
         if verbosity > 0:
            print("solve begins %s" % datetime_string())
            print("\t- lambda = %f" % lambdaval)
         SolStat, z = lagrUtil.solve_ph_code(ph, options)
         b = Compute_ExpectationforVariable(ph, IndVarName, CCStageNum)
         if verbosity > 0:
            print("solve ends %s" % datetime_string())
            print("\t- SolStat = %s" % str(SolStat))
            print("\t- b = %s" % str(b))
            print("\t- z = %s" % str(z))
            print("(adding to more PR points)")

         Result.morePR.append([None,b,z])
         if verbosity > 1:
            PrintPRpoints(Result.morePR)
      ######################################################
      # end loop over target probabilities

      with open(csvPrefix+"PRmore.csv",'w') as outFile:
         for point in Result.morePR:
            outFile.write(str(point[1])+','+str(point[2]))

      addstatus = str(len(Result.morePR)) + ' PR points written to file: '+ csvPrefix + 'PRmore.csv'
      if verbosity > 0: print(addstatus)
      Result.status = Result.status + '\n' + addstatus
      addstatus = 'lapsed time = ' + putcommas(time.time() - STARTTIME)
      if verbosity > 0: print(addstatus)
      Result.status = Result.status + '\n' + addstatus

      return Result
################################
# LagrangeMorePR ends here
################################

#### start run ####

   AllInOne = False
#   VERYSTARTTIME=time.time()
#   print "##############VERYSTARTTIME:",str(VERYSTARTTIME-VERYSTARTTIME)

##########################
# options defined here
##########################
   try:
      conf_options_parser = construct_ph_options_parser("lagrange [options]")
      conf_options_parser.add_argument("--beta-min",
                                     help="The min beta level for the chance constraint. Default is None",
                                     action="store",
                                     dest="beta_min",
                                     type=float,
                                     default=None)
      conf_options_parser.add_argument("--beta-max",
                                     help="The beta level for the chance constraint. Default is None",
                                     action="store",
                                     dest="beta_max",
                                     type=float,
                                     default=None)
      conf_options_parser.add_argument("--min-prob",
                                     help="Tolerance for testing probability > 0. Default is 1e-5",
                                     action="store",
                                     dest="min_prob",
                                     type=float,
                                     default=1e-5)
      conf_options_parser.add_argument("--beta-tol",
                                     help="Tolerance for testing equality to beta. Default is 10^-2",
                                     action="store",
                                     dest="beta_tol",
                                     type=float,
                                     default=1e-2)
      conf_options_parser.add_argument("--Lagrange-gap",
                                     help="The (relative) Lagrangian gap acceptable for the chance constraint. Default is 10^-4.",
                                     action="store",
                                     type=float,
                                     dest="Lagrange_gap",
                                     default=0.0001)
      conf_options_parser.add_argument("--max-number",
                                     help="The max number of PR points. Default = 10.",
                                     action="store",
                                     dest="max_number",
                                     type=int,
                                     default=10)
      conf_options_parser.add_argument("--max-time",
                                     help="Maximum time (seconds). Default is 3600.",
                                     action="store",
                                     dest="max_time",
                                     type=float,
                                     default=3600)
      conf_options_parser.add_argument("--csvPrefix",
                                     help="Input file name prefix.  Default is ''",
                                     action="store",
                                     dest="csvPrefix",
                                     type=str,
                                     default="")
      conf_options_parser.add_argument("--lambda-parm-name",
                                     help="The name of the lambda parameter in the model. Default is lambdaMult",
                                     action="store",
                                     dest="lambda_parm_name",
                                     type=str,
                                     default="lambdaMult")
      conf_options_parser.add_argument("--indicator-var-name",
                                     help="The name of the indicator variable for the chance constraint. The default is delta",
                                     action="store",
                                     dest="indicator_var_name",
                                     type=str,
                                     default="delta")
      conf_options_parser.add_argument("--stage-num",
                                     help="The stage number of the CC indicator variable (number, not name). Default is 2",
                                     action="store",
                                     dest="stage_num",
                                     type=int,
                                     default=2)
      conf_options_parser.add_argument("--verbosity",
                                     help="verbosity=0 is no extra output, =1 is medium, =2 is debug, =3 super-debug. Default is 1.",
                                     action="store",
                                     dest="verbosity",
                                     type=int,
                                     default=1)
      conf_options_parser.add_argument("--prob-file",
                                     help="file name specifiying probabilities",
                                     action="store",
                                     dest="probFileName",
                                     type=str,
                                     default=None)
# The following needed for solve_ph_code in lagrangeutils
      conf_options_parser.add_argument("--solve-with-ph",
                                     help="Perform solves via PH rather than an EF solve. Default is False",
                                     action="store_true",
                                     dest="solve_with_ph",
                                     default=False)

################################################################

      options = conf_options_parser.parse_args(args=args)
      # temporary hack
      options._ef_options = conf_options_parser._ef_options
      options._ef_options.import_argparse(options)
   except SystemExit as _exc:
      # the parser throws a system exit if "-h" is specified - catch
      # it to exit gracefully.
      return _exc.code

   if options.verbose is True:
      print("Loading reference model and scenario tree")

   scenario_instance_factory = \
        ScenarioTreeInstanceFactory(options.model_directory,
                                    options.instance_directory)

   full_scenario_tree = \
            GenerateScenarioTreeForPH(options,
                                      scenario_instance_factory)

   solver_manager = SolverManagerFactory(options.solver_manager_type)
   if solver_manager is None:
      raise ValueError("Failed to create solver manager of "
                       "type="+options.solver_manager_type+
                       " specified in call to PH constructor")
   if isinstance(solver_manager,
                 pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):
      solver_manager.deactivate()
      raise ValueError("PHPyro can not be used as the solver manager")

   try:

      if (scenario_instance_factory is None) or (full_scenario_tree is None):
         raise RuntimeError("***ERROR: Failed to initialize the model and/or scenario tree data.")

      # load_model gets called again, so lets make sure unarchived directories are used
      options.model_directory = scenario_instance_factory._model_filename
      options.instance_directory = scenario_instance_factory._scenario_tree_filename

      scenario_count = len(full_scenario_tree._stages[-1]._tree_nodes)

      # create ph objects for finding the solution. we do this even if
      # we're solving the extensive form

      if options.verbose is True:
         print("Loading scenario instances and initializing scenario tree for full problem.")

########## Here is where multiplier search is called ############
      Result = LagrangeMorePR()
#####################################################################################
   finally:

      solver_manager.deactivate()
      # delete temporary unarchived directories
      scenario_instance_factory.close()

   print("\n====================  returned from LagrangeMorePR")
   print(str(Result.status))
   try:
     print("Envelope:")
     print(str(PrintPRpoints(Result.PRoptimal)))
     print("\nAdded:")
     PrintPRpoints(Result.morePR)
   except:
     print("from run:  PrintPRpoints failed")
     sys.exit()

# combine tables and sort by probability
   if len(Result.morePR) > 0:
     PRpoints = copy.deepcopy(Result.PRoptimal)
     for lbz in Result.morePR: PRpoints.append(lbz)
     print("Combined table of PR points (sorted):")
     PRpoints.sort(key=operator.itemgetter(1))
     print(str(PrintPRpoints(PRpoints)))

########################## functions defined here #############################

def FindPRpoints(options, PRoptimal):
# Find more PR points (above F^*)
   if options.verbosity > 1:
      print("entered FindPRpoints seeking %d points" % options.max_number)
   Intervals = []

# Find intervals, each with width > 2beta_tol, such that cdf[i] is near its midpoint for some i
   if options.verbosity > 1:
      print("Collecting intervals having width > %f" % 2*options.beta_tol)
   for i in range(1,len(PRoptimal)):
      width = PRoptimal[i][1] - PRoptimal[i-1][1]
      if width <= 2*options.beta_tol: continue
      midpoint = (PRoptimal[i][1] + PRoptimal[i-1][1]) / 2.
      Intervals.append( [i, width, midpoint] )
      if options.verbosity > 1:
         print("Interval: %d  width = %f  midpoint = %f" % (i, width, midpoint))
   Intervals.sort(key=operator.itemgetter(1),reverse=True)   # sorts from max to min width

   if options.verbosity > 1:
      print("%d Intervals:" % len(Intervals))
      for interval in Intervals:  print("\t %s" % str(interval))

   while len(Intervals) < options.max_number:
# split widest interval to have another PR point
      interval = Intervals[0]
      width = interval[1] # = width of interval
      if width < 2*options.beta_tol:
         status = 'greatest width = ' + str(width) + ' < 2*beta_tol = ' + str(2*options.beta_tol)
         print(status)
         if options.verbosity > 1: print("\t** break out of while")
         break
      i = interval[0] # = index of point in envelope function
      midpoint = interval[2]

      if options.verbosity > 1: print("splitting interval: %s" % str(interval))
      Intervals[0][1] = width/2.           # reduce width
      Intervals[0][2] = midpoint-width/4.  # new midpoint of left
      Intervals = Insert([i, width/2., midpoint+width/4.],    1, Intervals) # insert at top arbitrary choice
#                                     new midpoint of right
      Intervals.sort(key=operator.itemgetter(1),reverse=True)               # because we re-sort
      if options.verbosity > 1:
         print("Number of intervals = %d" % len(Intervals))
   if options.verbosity > 1:
      print("\n--- end while with %d intervals:" % len(Intervals))
      for interval in Intervals:
         print("\t%s" % str(interval))

   PRlist = []
   for interval in Intervals:
      PRlist.append( [interval[0],interval[2]] )
      #                                             |           = probability (= midpoint of Interval)
      #                                             = envelope index

   if options.verbosity > 1:
      print("\treturning PRlist:")
      for p in PRlist: print( "\t  %s" % str(p))
   return PRlist

#################################################################################

def GetPoints(options, PRoptimal, probList):
# Find gap intervals containing probability in probList
   PRlist = []
   for prob in probList:
      for i in range(1,len(PRoptimal)):
         if PRoptimal[i][1] >= prob: break
      PRlist.append([i,prob])  # i = index of upper value (bU)
   return PRlist

#################################################################################

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

def PrintPRpoints(PRlist):
   if len(PRlist) == 0:
      print("No PR points")
   else:
      print("%d PR points:" % len(PRlist))
      blanks = "                      "
      print("            lambda        beta-probability       min cost ")
      for row in PRlist:
         b = float(round(row[1],4))
         z = float(round(row[2]))
# lambda = row[0] could be float, string, or None
         sl = str(row[0])
         sl = blanks[0:20-len(sl)] + sl
         sb = str(b)
         sb = blanks[0:20-len(sb)] + sb
         sz = putcommas(z)
         sz = blanks[2:20-len(sz)] + sz
         print(sl+" "+sb+" "+sz)
      print("==================================================================")
   return

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
