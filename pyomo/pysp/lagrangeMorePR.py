#! /usr/bin/env python

# This reads PRoptimal.csv, which contains the sequence of PR points that defines the optimal response function, F^*
# Then, it computes more PR points and appends to PRmore.csv

import sys
import os
import random
import math 
import time 
import datetime
import operator
import types
from coopr.pysp.scenariotree import *
from coopr.pysp.phinit import *
from coopr.pysp.ph import *
from coopr.pysp.ef import *
from coopr.opt import SolverFactory

import coopr.pysp.lagrangeutils as lagrUtil
#import myfunctions as myfunc

def datetime_string():
   return "datetime = "+str(datetime.datetime.now())

###################################
def run(args=None):
###################################

   # to import plugins
   import coopr.environ

   def LagrangeMorePR(args=None):
      print("lagrangeMorePR begins %s\n" % datetime_string())
      blanks = "                          "  # used for formatting print statements
      class Object(object): pass
      Result = Object()
      def new_ph():
         scenario_instance_factory, scenario_tree = load_models(options)
         if scenario_instance_factory is None or scenario_tree is None:
            print("internal error in new_ph\n")
            exit(2)
         return create_ph_from_scratch(options,
                                       scenario_instance_factory,
                                       scenario_tree,
                                       solver_manager)

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

      ph = new_ph() 
      rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]   # use rootnode to loop over scenarios

      if find_active_objective(ph._scenario_tree._scenarios[0]._instance,safety_checks=True).is_minimizing():
         print("We are solving a MINIMIZATION problem.\n")
      else:
         print("We are solving a MAXIMIZATION problem.\n")
      
# initialize
      ScenarioList = []
      inputFile = file(csvPrefix+"ScenarioList.csv",'r')
      for line in inputFile.readlines():
         L = line.split(',')
         ScenarioList.append([L[0],float(L[1])])
      inputFile.close()
      addstatus = str(len(ScenarioList))+' scenarios read from file: ' + csvPrefix+'ScenarioList.csv'
      if verbosity > 0: print(addstatus+'\n')
      Result.status = Result.status + '\n' + addstatus

      PRoptimal = []
      inputFile = file(csvPrefix+"PRoptimal.csv",'r')
      for line in inputFile.readlines():
         bzS = line.split(',')
         PRoptimal.append( [None, float(bzS[0]), float(bzS[1])] )
      inputFile.close()
      addstatus = str(len(PRoptimal))+' PR points read from file: '+ csvPrefix+'PRoptimal.csv (envelope function)'
      if verbosity > 0: print(addstatus+'\n')
      Result.status = Result.status + '\n' + addstatus
# ensure PR points on envelope function are sorted by probability
      PRoptimal.sort(key=operator.itemgetter(1))  

      PRoptimal[0][0] = 0   # initial lambda (for b=0)
      for p in range(1,len(PRoptimal)):
         dz = PRoptimal[p][2] - PRoptimal[p-1][2]
         db = PRoptimal[p][1] - PRoptimal[p-1][1]
         PRoptimal[p][0] = dz/db
      if verbosity > 0: PrintPRpoints(PRoptimal)
      Result.PRoptimal = PRoptimal

      lambdaval = 0.
      lagrUtil.Set_ParmValue(ph, options.lambda_parm_name,lambdaval)

## read scenarios to select for each PR point on envelope function
      inputFile = file(csvPrefix+"OptimalSelections.csv",'r') 
      OptimalSelections = []
      for line in inputFile.readlines():
         if len(line) == 0: break # eof
         selections = line.split(',')
         L = len(selections)
         Ls = len(selections[L-1])
         selections[L-1] = selections[L-1][0:Ls-1]
         if verbosity > 1: print(str(selections)+'\n')
         OptimalSelections.append(selections)
      inputFile.close()
      Result.OptimalSelections = OptimalSelections

      addstatus = str(len(OptimalSelections)) + ' Optimal selections read from file: ' \
            + csvPrefix + 'OptimalSelections.csv'
      Result.status = Result.status + '\n' + addstatus

      if len(OptimalSelections) == len(PRoptimal):
         if verbosity > 0: print(addstatus+'\n')
      else:
         addstatus = addstatus + '\n** Number of selections not equal to number of PR points'
         print(addstatus+'\n')
         Result.status = Result.status + '\n' + addstatus
         print(str(OptimalSelections)+'\n')
         print((PRoptimal)+'\n')
         return Result

#####################################################################################

# get probabilities
      if probFileName == None:
# ...generate from widest gap regions
        PRlist = FindPRpoints(options, PRoptimal)
      else:
# ...read probabilities
        probList = []
        inputFile = file(probFileName,'r')
        if verbosity > 0: print("reading from probList = "+probFileName)
        for line in inputFile.readlines():  # 1 probability per line
           if len(line) == 0: break
           prob = float(line)
           probList.append(prob)
        inputFile.close()
        if verbosity > 0: print("\t "+str(len(probList))+" probabilities\n")
        if verbosity > 1: print(str(probList)+'\n')
        PRlist = GetPoints(options, PRoptimal, probList)
        if verbosity > 1: 
           print("PRlist:\n")
           for interval in PRlist: print(str(interval)+'\n')

# We now have PRlist = [[i, b], ...], where b is in PRoptimal interval (i-1,i)
      addstatus = str(len(PRlist)) + ' probabilities'
      if verbosity > 0: print(addstatus+'\n')
      Result.status = Result.status + '\n' + addstatus

#####################################################################################
                                         
      lapsedTime = time.time() - STARTTIME
      addstatus = 'Initialize complete...lapsed time = ' + str(lapsedTime)
      if verbosity > 1: print(addstatus+'\n')
      Result.status = Result.status + '\n' + addstatus
      
#####################################################################################

      if verbosity > 1: 
        print("\nlooping over Intervals to generate PR points by flipping heuristic\n")
      Result.morePR = []
      for interval in PRlist:
         lapsedTime = time.time() - STARTTIME
         if lapsedTime > MaxTime:
            addstatus = '** lapsed time = ' + str(lapsedTime) + ' > max time = ' + str(MaxTime)
            if verbosity > 0: print(addstatus+'\n')
            Result.status = Result.status + '\n' + addstatus
            break

         i = interval[0] # = PR point index
         b = interval[1] # = target probability to reach by flipping from upper endpoint
         bU = PRoptimal[i][1]   # = upper endpoint
         bL = PRoptimal[i-1][1] # = lower endpoint
         if verbosity > 1: 
            print( "target probability = "+str(b)+" < bU = PRoptimal[" + str(i) + "][1]" \
                 " and > bL = PRoptimal["+str(i-1)+"][1]\n")
         if b < bL or b > bU:
            addstatus = '** probability = '+str(b) + ', not in gap interval: (' \
                + str(bL) + ', ' + str(bU) + ')' 
            print(addstatus+'\n')
            print(str(PRoptimal)+'\n')
            print(str(PRlist)+'\n')
            Result.status = Result.status + '\n' + addstatus
            return Result

         if verbosity > 1: 
            print( "i = "+str(i)+" : Starting with bU = "+str(bU)+" having "+ \
                str(len(OptimalSelections[i]))+ " selections:\n")
            print(str(OptimalSelections[i])+'\n')

# first fix all scenarios = 0
         for scenario in ScenarioList:
            sname = scenario[0]
            instance = ph._instances[sname]
            getattr(instance, IndVarName).value = 0
            getattr(instance, IndVarName).fixed = True
# now fix optimal selections = 1
         for sname in OptimalSelections[i]:
            instance = ph._instances[sname]
            getattr(instance, IndVarName).value = 1

# flip scenario selections from bU until we reach b (target probability)
         bNew = bU
         for scenario in ScenarioList:
            sname = scenario[0]
            sprob = scenario[1]
            if bNew - sprob < b: continue
            instance = ph._instances[sname]
            if getattr(instance, IndVarName).value == 0: continue
            bNew = bNew - sprob
            getattr(instance, IndVarName).value = 0 # flipped scenario selection
            if verbosity > 1: print("\tflipped "+sname+" with prob = "+str(sprob)+" ...bNew = "+str(bNew)+'\n')

         if verbosity > 1:
            print("\tflipped selections reach "+str(bNew)+" >= target = "+str(b)+" (bL = "+str(bL)+")\n")
         if bNew <= bL + betaTol or bNew >= bU - betaTol: 
            if verbosity > 0:
               print("\tNot generating PR point...flipping from bU failed\n")
            continue # to next interval in list

 # ready to solve to get cost for fixed scenario selections associated with probability = bNew
         instance.preprocess() 

         if verbosity > 1:
# check that scenarios are fixed as they should be
            totalprob = 0.
            for scenario in ScenarioList:
               sname = scenario[0]
               sprob = scenario[1]
               instance = ph._instances[sname]
               print("fix "+sname+" = "+str(getattr(instance,IndVarName).value)+\
                  " is "+str(getattr(instance,IndVarName).fixed)+" probability = "+str(sprob)+'\n')
               if getattr(instance,IndVarName).value == 1: 
                  totalprob = totalprob + sprob
               lambdaval = getattr(instance, multName).value
            print("\ttotal probability = %f\n" % totalprob)

# solve (all delta fixed); lambda=0, so z = Lagrangian
         if verbosity > 0: 
            print("solve begins %s\n" % datetime_string())
            print("\t- lambda = %f\n" % lambdaval)
         SolStat, z = lagrUtil.solve_ph_code(ph, options)
         b = Compute_ExpectationforVariable(ph, IndVarName, CCStageNum)
         if verbosity > 0: 
            print("solve ends %s\n" % datetime_string())
            print("\t- SolStat = %s\n" % str(SolStat))
            print("\t- b = %s\n" % str(b))
            print("\t- z = %s\n" % str(z))
            print("(adding to more PR points)\n")

         Result.morePR.append([None,b,z])
         if verbosity > 1: PrintPRpoints(Result.morePR)
      ######################################################         
      # end loop over target probabilities 

      outFile = file(csvPrefix+"PRmore.csv",'w')
      for point in Result.morePR: 
         print >> outFile, str(point[1])+','+str(point[2])
      outFile.close() 

      addstatus = str(len(Result.morePR)) + ' PR points written to file: '+ csvPrefix + 'PRmore.csv'
      if verbosity > 0: print(addstatus+'\n')
      Result.status = Result.status + '\n' + addstatus 
      addstatus = 'lapsed time = ' + putcommas(time.time() - STARTTIME) 
      if verbosity > 0: print(addstatus+'\n')
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
      conf_options_parser.add_option("--beta-min",
                                     help="The min beta level for the chance constraint. Default is None",
                                     action="store",
                                     dest="beta_min",
                                     type="float",
                                     default=None)
      conf_options_parser.add_option("--beta-max",
                                     help="The beta level for the chance constraint. Default is None",
                                     action="store",
                                     dest="beta_max",
                                     type="float",
                                     default=None)
      conf_options_parser.add_option("--min-prob",
                                     help="Tolerance for testing probability > 0. Default is 1e-5",
                                     action="store",
                                     dest="min_prob",
                                     type="float",
                                     default=1e-5)
      conf_options_parser.add_option("--beta-tol",
                                     help="Tolerance for testing equality to beta. Default is 10^-2",
                                     action="store",
                                     dest="beta_tol",
                                     type="float",
                                     default=1e-2)
      conf_options_parser.add_option("--Lagrange-gap",
                                     help="The (relative) Lagrangian gap acceptable for the chance constraint. Default is 10^-4.",
                                     action="store",
                                     type="float",
                                     dest="Lagrange_gap",
                                     default=0.0001)
      conf_options_parser.add_option("--max-number",
                                     help="The max number of PR points. Default = 10.",
                                     action="store",
                                     dest="max_number",
                                     type="int",
                                     default=10)
      conf_options_parser.add_option("--max-time",
                                     help="Maximum time (seconds). Default is 3600.",
                                     action="store",
                                     dest="max_time",
                                     type="float",
                                     default=3600) 
      conf_options_parser.add_option("--csvPrefix",
                                     help="Input file name prefix.  Default is ''",
                                     action="store",
                                     dest="csvPrefix",
                                     type="string",
                                     default="")
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
      conf_options_parser.add_option("--verbosity",
                                     help="verbosity=0 is no extra output, =1 is medium, =2 is debug, =3 super-debug. Default is 1.",
                                     action="store",
                                     dest="verbosity",
                                     type="int",
                                     default=1)
      conf_options_parser.add_option("--prob-file",
                                     help="file name specifiying probabilities",
                                     action="store",
                                     dest="probFileName",
                                     type="string",
                                     default=None)
# The following needed for solve_ph_code in lagrangeutils
      conf_options_parser.add_option("--solve-with-ph",
                                     help="Perform solves via PH rather than an EF solve. Default is False",
                                     action="store_true",
                                     dest="solve_with_ph",
                                     default=False)

################################################################

      (options, args) = conf_options_parser.parse_args(args=args)
   except SystemExit:
      # the parser throws a system exit if "-h" is specified - catch
      # it to exit gracefully.
      return

   if options.verbose is True:
      print("Loading reference model and scenario tree\n")
   scenario_instance_factory, full_scenario_tree = load_models(options)

   solver_manager = SolverManagerFactory(options.solver_manager_type)
   if solver_manager is None:
      raise ValueError("Failed to create solver manager of "
                       "type="+options.solver_manager_type+
                       " specified in call to PH constructor")
   if isinstance(solver_manager, SolverManager_PHPyro):
      solver_manager.deactivate()
      raise ValueError("PHPyro can not be used as the solver manager")

   try:

      if (scenario_instance_factory is None) or (full_scenario_tree is None):
         raise RuntimeError, "***ERROR: Failed to initialize the model and/or scenario tree data."

      # load_model gets called again, so lets make sure unarchived directories are used
      options.model_directory = scenario_instance_factory._model_filename
      options.instance_directory = scenario_instance_factory._data_filename
   
      scenario_count = len(full_scenario_tree._stages[-1]._tree_nodes)

      # create ph objects for finding the solution. we do this even if
      # we're solving the extensive form

      if options.verbose is True:
         print("Loading scenario instances and initializing scenario tree for full problem.\n")

########## Here is where multiplier search is called ############
      Result = LagrangeMorePR()
##################################################################################### 
   finally:

      solver_manager.deactivate()
      # delete temporary unarchived directories
      scenario_instance_factory.close()

   print("\n====================  returned from LagrangeMorePR\n")
   print(str(Result.status)+'\n')
   try:
     print("Envelope:\n")
     print(str(PrintPRpoints(Result.PRoptimal))+'\n')
     print("\nAdded:\n")
     PrintPRpoints(Result.morePR)
   except:
     print("from run:  PrintPRpoints failed\n")
     sys.exit()   

# combine tables and sort by probability
   if len(Result.morePR) > 0:
     PRpoints = copy.deepcopy(Result.PRoptimal)
     for lbz in Result.morePR: PRpoints.append(lbz)
     print("Combined table of PR points (sorted):\n")
     PRpoints.sort(key=operator.itemgetter(1))
     print(str(PrintPRpoints(PRpoints))+'\n')

########################## functions defined here #############################

def FindPRpoints(options, PRoptimal):
# Find more PR points (above F^*)
   if options.verbosity > 1: 
      print("entered FindPRpoints seeking %d points\n" % options.max_number)
   Intervals = []

# Find intervals, each with width > 2beta_tol, such that cdf[i] is near its midpoint for some i
   if options.verbosity > 1: 
      print("Collecting intervals having width > %f\n" % 2*options.beta_tol)
   for i in range(1,len(PRoptimal)):
      width = PRoptimal[i][1] - PRoptimal[i-1][1] 
      if width <= 2*options.beta_tol: continue
      midpoint = (PRoptimal[i][1] + PRoptimal[i-1][1]) / 2.
      Intervals.append( [i, width, midpoint] )
      if options.verbosity > 1: 
         print("Interval: %d  width = %f  midpoint = %f\n" % (i, width, midpoint))
   Intervals.sort(key=operator.itemgetter(1),reverse=True)   # sorts from max to min width

   if options.verbosity > 1:
      print("%d Intervals:\n" % len(Intervals))
      for interval in Intervals:  print("\t %s\n" % str(interval))

   while len(Intervals) < options.max_number:
# split widest interval to have another PR point
      interval = Intervals[0]
      width = interval[1] # = width of interval
      if interval[1] < 2*options.beta_tol: 
         status = 'greatest width = ' + str(w) + ' < 2*beta_tol = ' + str(2*options.beta_tol)
         print(status+'\n')
         if options.verbosity > 1: print("\t** break out of while\n")
         break
      i = interval[0] # = index of point in envelope function
      midpoint = interval[2] 
        
      if options.verbosity > 1: print("splitting interval: %s\n" % str(interval))
      Intervals[0][1] = width/2.           # reduce width
      Intervals[0][2] = midpoint-width/4.  # new midpoint of left
      Intervals = Insert([i, width/2., midpoint+width/4.],    1, Intervals) # insert at top arbitrary choice
#                                     new midpoint of right
      Intervals.sort(key=operator.itemgetter(1),reverse=True)               # because we re-sort
      if options.verbosity > 1: print("Number of intervals = %d\n" % len(Intervals))
   if options.verbosity > 1: 
      print("\n--- end while with %d intervals:\n" % len(Intervals))
      for interval in Intervals:  print("\t%s\n" % str(interval))

   PRlist = []
   for interval in Intervals: PRlist.append( [interval[0],interval[2]] )
#                                             |           = probability (= midpoint of Interval)
#                                             = envelope index

   if options.verbosity > 1: 
      print("\treturning PRlist:\n")
      for p in PRlist: print( "\t  %s\n" % str(p))
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
      print("%s\n" % i._name)
   print("Ones:\n")
   for i in IndList[1]:
      print("%s\n" % i._name)

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
