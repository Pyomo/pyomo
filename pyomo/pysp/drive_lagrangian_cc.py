#! /usr/bin/env python
# code to drive lagrgangeutils.py for a test

#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2013 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import sys
import os
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

from pyomo.pysp.lagrangeutils import *
# from generalFunctions import *

#########################################
def run(args=None):
##########################================================#########
   # to import plugins
   import pyomo.modeling


   def partialLagrangeParametric(args=None):
      print("lagrangeParam begins ")
      blanks = "                          "  # used for formatting print statements
      class Object(object): pass
      Result = Object()
      def new_ph():
         scenario_instance_factory, scenario_tree = load_models(options)
         if scenario_instance_factory is None or scenario_tree is None:
            print("internal error in new_ph")
            exit(2)
         return create_ph_from_scratch(options, scenario_instance_factory, scenario_tree)

# options used
      IndVarName = options.indicator_var_name
      CCStageNum = options.stage_num
      alphaTol = options.alpha_tol
      MaxMorePR = options.MaxMorePR # option to include up to this many PR points above F^* with all delta fixed
      outputFilePrefix = options.outputFilePrefix

# We write ScenarioList = name, probability 
#          PRoptimal    = probability, min-cost, [selections]   
#          PRmore       = probability, min-cost, [selections]
# ================ sorted by probability ========================
#
# These can be read to avoid re-computing points

      ph = new_ph() 
      Result.ph = ph
      rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]   # use rootnode to loop over scenarios

      if find_active_objective(ph._scenario_tree._scenarios[0]._instance,safety_checks=True).is_minimizing():
         print("We are solving a MINIMIZATION problem.\n")
      else:
         print("We are solving a MAXIMIZATION problem.\n")
      
# initialize
      ScenarioList = []
      lambdaval = 0.
      Set_ParmValue(ph, options.lambda_parm_name,lambdaval)
      for scenario in rootnode._scenarios:
         instance = ph._instances[scenario._name]
         sname = scenario._name
         sprob = scenario._probability
         ScenarioList.append([sname,sprob])
         getattr(instance,IndVarName).value = 0
         getattr(instance,IndVarName).fixed = True
         #print "fixed",sname,"=0"
      instance.preprocess()
      ScenarioList.sort(key=operator.itemgetter(1))   # sorts from min to max probability
      outFile = file(outputFilePrefix+'ScenarioList.csv','w')
      for scenario in ScenarioList: 
          outFile.write(scenario[0]+ ", " +str(scenario[1])+"\n")
      outFile.close()
      Result.ScenarioList = ScenarioList
 
      print("lambda= "+str(lambdaval)+" ...run begins "+str(len(ScenarioList))+" scenarios")
      SolStat, zL = solve_ph_code(ph, options)
      print("\t...ends")
      bL = Compute_ExpectationforVariable(ph, IndVarName, CCStageNum)
      if bL > 0:
         print("** bL = "+str(bL)+"  > 0")
         return Result

      print("Initial cost = "+str(zL)+"  for bL = "+str(bL))

      for scenario in ScenarioList:
         sname = scenario[0]
         instance = ph._instances[sname]
         getattr(instance,IndVarName).value = 1
      instance.preprocess() 
      
      print("lambda= "+str(lambdaval)+" ...run begins")
      SolStat, zU = solve_ph_code(ph, options)
      print("\t...ends")
      bU = Compute_ExpectationforVariable(ph, IndVarName, CCStageNum)
      if bU < 1: 
            print("** bU = "+str(bU)+"  < 1")

      for scenario in ScenarioList:
         sname = scenario[0]
         instance = ph._instances[sname]
         getattr(instance,IndVarName).fixed = False
      instance.preprocess() 

      Result.lbz = [ [0,bL,zL], [None,bU,zU] ]
      Result.selections = [[], ScenarioList]
      NumIntervals = 1
      print("initial gap = "+str(1-zL/zU)+" \n")
      print("End of test; this is only a test.")

      return Result
################################
# LagrangeParametric ends here
################################
      
#### start run ####

   AllInOne = False
   
##########################
# options defined here
##########################
   try:
      conf_options_parser = construct_ph_options_parser("lagrange [options]")
      conf_options_parser.add_option("--alpha",
                                     help="The alpha level for the chance constraint. Default is 0.05",
                                     action="store",
                                     dest="alpha",
                                     type="float",
                                     default=0.05)
      conf_options_parser.add_option("--alpha-min",
                                     help="The min alpha level for the chance constraint. Default is None",
                                     action="store",
                                     dest="alpha_min",
                                     type="float",
                                     default=None)
      conf_options_parser.add_option("--alpha-max",
                                     help="The alpha level for the chance constraint. Default is None",
                                     action="store",
                                     dest="alpha_max",
                                     type="float",
                                     default=None)
      conf_options_parser.add_option("--min-prob",
                                     help="Tolerance for testing probability > 0. Default is 1e-5",
                                     action="store",
                                     dest="min_prob",
                                     type="float",
                                     default=1e-5)
      conf_options_parser.add_option("--alpha-tol",
                                     help="Tolerance for testing equality to alpha. Default is 1e-5",
                                     action="store",
                                     dest="alpha_tol",
                                     type="float",
                                     default=1e-5)
      conf_options_parser.add_option("--MaxMorePR",
                                     help="Generate up to this many additional PR points after response function. Default is 0",
                                     action="store",
                                     dest="MaxMorePR",
                                     type="int",
                                     default=0)
      conf_options_parser.add_option("--outputFilePrefix",
                                     help="Output file name.  Default is ''",
                                     action="store",
                                     dest="outputFilePrefix",
                                     type="string",
                                     default="")
      conf_options_parser.add_option("--stage-num",
                                     help="The stage number of the CC indicator variable (number, not name). Default is 2",
                                     action="store",
                                     dest="stage_num",
                                     type="int",
                                     default=2)
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
      conf_options_parser.add_option("--use-Loane-cuts",
                                     help="Add the Loane cuts if there is a gap. Default is False",
                                     action="store_true",
                                     dest="add_Loane_cuts",
                                     default=False)
      conf_options_parser.add_option("--fofx-var-name",
                                     help="(Loane) The name of the model's auxiliary variable that is constrained to be f(x). Default is fofox",
                                     action="store",
                                     dest="fofx_var_name",
                                     type="string",
                                     default="fofx")
      conf_options_parser.add_option("--solve-with-ph",
                                     help="Perform solves via PH rather than an EF solve. Default is False",
                                     action="store_true",
                                     dest="solve_with_ph",
                                     default=False)
      conf_options_parser.add_option("--skip-graph",
                                     help="Do not show the graph at the end. Default is False (i.e. show the graph)",
                                     action="store_true",
                                     dest="skip_graph",
                                     default=False)
      conf_options_parser.add_option("--write-xls",
                                     help="Write results into a xls file. Default is False",
                                     action="store_true",
                                     dest="write_xls",
                                     default=False)
      conf_options_parser.add_option("--skip-ExpFlip",
                                     help="Do not show the results for flipping the indicator variable for each scenario. Default is False (i.e. show the flipping-results)",
                                     action="store_true",
                                     dest="skip_ExpFlip",
                                     default=False)
      conf_options_parser.add_option("--HeurFlip",
                                     help="The number of solutions to evaluate after the heuristic. Default is 3. For 0 the heuristic flip gets skipped.",
                                     action="store",
                                     type="int",
                                     dest="HeurFlip",
                                     default=3)
      conf_options_parser.add_option("--HeurMIP",
                                     help="The mipgap for the scenariowise solves in the heuristic. Default is 0.0001",
                                     action="store",
                                     type="float",
                                     dest="HeurMIP",
                                     default=0.0001)
      conf_options_parser.add_option("--interactive",
                                     help="Enable interactive version of the code. Default is False.",
                                     action="store_true",                                  
                                     dest="interactive",
                                     default=False)
      conf_options_parser.add_option("--Lgap",
                                     help="The (relative) Lagrangian gap acceptable for the chance constraint. Default is 10^-4",
                                     action="store",
                                     type="float",
                                     dest="LagrangeGap",
                                     default=0.0001)  
      conf_options_parser.add_option("--lagrange-method",
                                     help="The Lagrange multiplier search method",
                                     action="store",           
                                     dest="lagrange_search_method",
                                     type="string",            
                                     default="tangential")          
      conf_options_parser.add_option("--max-lambda",
                                     help="The max value of the multiplier. Default=10^10",
                                     action="store",
                                     dest="max_lambda",
                                     type="float",
                                     default=10**10)
      conf_options_parser.add_option("--min-lambda",
                                     help="The min value of the multiplier. Default=0.0",
                                     action="store",
                                     dest="min_lambda",
                                     type="float",
                                     default=0)  
      conf_options_parser.add_option("--min-probability",
                                     help="The min value of scenario probability. Default=10^-15",
                                     action="store",
                                     dest="min_probability",
                                     type="float",
                                     default=10**(-15))

################################################################

      (options, args) = conf_options_parser.parse_args(args=args)
   except SystemExit:
      # the parser throws a system exit if "-h" is specified - catch
      # it to exit gracefully.
      return

   # load the reference model and create the scenario tree - no scenario instances yet.
   if options.verbose is True:
      print("Loading reference model and scenario tree")
   scenario_instance_factory, full_scenario_tree = load_models(options)


   try:

      if (scenario_instance_factory is None) or (full_scenario_tree is None):
         raise RuntimeError("***ERROR: Failed to initialize model and/or the scenario tree data.")

      # load_model gets called again, so lets make sure unarchived directories are used
      options.model_directory = scenario_instance_factory._model_filename
      options.instance_directory = scenario_instance_factory._data_filename

      scenario_count = len(full_scenario_tree._stages[-1]._tree_nodes)

      # create ph objects for finding the solution. we do this even if
      # we're solving the extensive form

      if options.verbose is True:
         print("Loading scenario instances and initializing scenario tree for full problem.")

########## Here is where multiplier search is called ############
      Result = partialLagrangeParametric()
##################################################################################### 

   finally:

      # delete temporary unarchived directories
      scenario_instance_factory.close()

   print("\nreturned from partialLagrangeParametric")
   
##########
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
      print(str(len(PRlist))+" PR points:")
      blanks = "                      "
      print("            lambda        beta-probability       min cost ")
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
         print(sl+" "+sb+" "+sz)
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
   print("Zeros:")
   for i in IndList[0]:
      print(i._name)
   print("Ones:")
   for i in IndList[1]:
      print(i._name)

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

def main(args=None):

   try:
      run(args)
   except ValueError:
      str = sys.exc_info()[1]
      print("VALUE ERROR:")
      print(str)
   except IOError:
      str = sys.exc_info()[1]
      print("IO ERROR:")
      print(str)
   except pyutilib.common.ApplicationError:
      str = sys.exc_info()[1]
      print("APPLICATION ERROR:")
      print(str)
   except RuntimeError:
      str = sys.exc_info()[1]
      print("RUN-TIME ERROR:")
      print(str)
   except:
      print("Encountered unhandled exception"+str(sys.exc_info()[0]))
      traceback.print_exc()

if __name__ == "__main__":

   run()
