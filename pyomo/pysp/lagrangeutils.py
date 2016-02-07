#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import os
import time

from pyomo.core import minimize
from pyomo.pysp.ef_writer_script import ExtensiveFormAlgorithm
from pyomo.pysp.phinit import run_ph
from pyomo.pysp.phutils import (reset_nonconverged_variables,
                                extractVariableNameAndIndex,
                                reset_stage_cost_variables)
from pyomo.pysp.solutionwriter import ISolutionWriterExtension
from pyomo.util.plugin import ExtensionPoint

# Tear the scenario instances off the ef instance when it is no longer required
# so warnings are not generated next time scenarios instances are placed inside
# a new ef instance
def _tear_down_ef(ef_instance, scenario_instances):
   for name in scenario_instances:
      ef_instance.del_component(name)

def solve_ph_code(ph, options):
   import pyomo.environ
   import pyomo.solvers.plugins.smanager.phpyro

   # consolidate the code to solve the problem for the "global" ph object
   # return a solver code (string from the solver if EF, "PH" if PH) and the objective fct val
   SolStatus = None

   ph._preprocess_scenario_instances()

   ObjectiveFctValue = \
      float('inf') if (ph._objective_sense is minimize) else float('-inf')
   SolStatus = None
   if not options.solve_with_ph:
      if options.verbose is True:
         print("Creating the extensive form.")
         print("Time="+time.asctime())

      with ExtensiveFormAlgorithm(ph,
                                  options._ef_options,
                                  prefix="ef_") as ef:
         ef.build_ef()
         failure = ef.solve(io_options=\
                            {'output_fixed_variable_bounds':
                             options.write_fixed_variables},
                            exception_on_failure=False)
         if not failure:
            ObjectiveFctValue = ef.objective
         SolStatus = str(ef.solver_status)

      """
      ef = create_ef_instance(ph._scenario_tree,
                              verbose_output=options.verbose)

      if options.verbose:
         print("Time="+time.asctime())
         print("Solving the extensive form.")

      ef_results = solve_ef(ef, options)

      SolStatus = str(ef_results.solver.status) 
      print("SolStatus="+SolStatus)
      if options.verbose is True:
         print("Loading extensive form solution.")
         print("Time="+time.asctime())
      ### If the solution is infeasible, we don't want to load the results
      ### It is up to the caller to decide what to do with non-optimal
      if SolStatus != "infeasible" and SolStatus != "unknown":

         # IMPT: the following method populates the _solution variables on the scenario tree
         #       nodes by forming an average of the corresponding variable values for all
         #       instances particpating in that node. if you don't do this, the scenario tree
         #       doesn't have the solution - and we need this below for variable bounding
         ph._scenario_tree.pullScenarioSolutionsFromInstances()
         ph._scenario_tree.snapshotSolutionFromScenarios()
         if options.verbose is True:
            print("SolStatus="+SolStatus)
            print("Time="+time.asctime())

      _tear_down_ef(ef, ph._instances)
      """
   else:
      if options.verbose:
         print("Solving via Progressive Hedging.")

      run_ph(options, ph)
      ph._scenario_tree.pullScenarioSolutionsFromInstances()
      root_node = ph._scenario_tree._stages[0]._tree_nodes[0]
      ObjectiveFctValue = root_node.computeExpectedNodeCost()
      SolStatus = "PH"
      """
      phretval = ph.solve()
      #print("--------->>>> "+str(phretval))
      SolStatus = "PH"
      if phretval is not None:
         print("Iteration zero solve was not successful for scenario: "+str(phretval))
         if options.verbose is True:
            print("Iteration zero solve was not successful for scenario: "+str(phretval))
         SolStatus = "PHFailAtScen"+str(phretval)

      # TBD - also not sure if PH calls snapshotSolutionFromAverages.
      if options.verbose is True:
         print("Done with PH solve.")

      ##begin copy from phinit
      solution_writer_plugins = ExtensionPoint(ISolutionWriterExtension)
      for plugin in solution_writer_plugins:
         plugin.write(ph._scenario_tree, "ph")

      # store the binding instance, if created, in order to load
      # the solution back into the scenario tree.
      binding_instance = None

      #
      # create the extensive form binding instance, so that we can either write or solve it (if specified).
      #
      if (options.write_ef) or (options.solve_ef):

        # The post-solve plugins may have done more variable
        # fixing. These should be pushed to the instance at this
        # point.
        print("Pushing fixed variable statuses to scenario instances")
        ph._push_all_node_fixed_to_instances()
        total_fixed_discrete_vars, total_fixed_continuous_vars = \
            ph.compute_fixed_variable_counts()
        print("Number of discrete variables fixed "
              "prior to ef creation="
              +str(total_fixed_discrete_vars)+
              " (total="+str(ph._total_discrete_vars)+")")
        print("Number of continuous variables fixed "
              "prior to ef creation="
              +str(total_fixed_continuous_vars)+
              " (total="+str(ph._total_continuous_vars)+")")

        print("Creating extensive form for remainder problem")
        ef_instance_start_time = time.time()
        binding_instance = create_ef_instance(ph._scenario_tree)

        ef_instance_end_time = time.time()
        print("Time to construct extensive form instance=%.2f seconds"
              % (ef_instance_end_time - ef_instance_start_time))

      ph._preprocess_scenario_instances()
      #
      # solve the extensive form and load the solution back into the PH scenario tree.
      # contents from the PH solve will obviously be over-written!
      #
      if options.write_ef:
         output_filename = os.path.expanduser(options.ef_output_file)
         # technically, we don't need the symbol map since we aren't solving it.
         print("Starting to write the extensive form")
         ef_write_start_time = time.time()
         symbol_map = write_ef(binding_instance,
                               output_filename,
                               symbolic_solver_labels=options.symbolic_solver_labels,
                               output_fixed_variable_bounds=options.write_fixed_variables)
         ef_write_end_time = time.time()
         print("Extensive form written to file="+output_filename)
         print("Time to write output file=%.2f seconds"
               % (ef_write_end_time - ef_write_start_time))

      if options.solve_ef:

         # set the value of each non-converged, non-final-stage variable to None -
         # this will avoid infeasible warm-stats.
         reset_nonconverged_variables(ph._scenario_tree, ph._instances)
         reset_stage_cost_variables(ph._scenario_tree, ph._instances)

         ef_results = solve_ef(binding_instance, options)

         print("Storing solution in scenario tree")
         ph._scenario_tree.pullScenarioSolutionsFromInstances()
         ph._scenario_tree.snapshotSolutionFromScenarios()

         ef_solve_end_time = time.time()
         print("Time to solve and load results for the "
               "extensive form=%.2f seconds"
               % (ef_solve_end_time - ef_solve_start_time))

         # print *the* metric of interest.
         print("")
         root_node = ph._scenario_tree._stages[0]._tree_nodes[0]
         print("***********************************************************************************************")
         print(">>>THE EXPECTED SUM OF THE STAGE COST VARIABLES="+str(root_node.computeExpectedNodeCost())+"<<<")
         print("***********************************************************************************************")
         print("")
         print("Extensive form solution:")
         ph._scenario_tree.pprintSolution()
         print("")
         print("Extensive form costs:")
         ph._scenario_tree.pprintCosts()

         solution_writer_plugins = ExtensionPoint(ISolutionWriterExtension)
         for plugin in solution_writer_plugins:
            plugin.write(ph._scenario_tree, "postphef")

      if binding_instance is not None:
         _tear_down_ef(binding_instance, ph._instances)
      """

   print("SolStatus="+str(SolStatus))
   if options.verbose:
      print("Time="+time.asctime())

   ## print "(using PySP Cost vars) ObjectiveFctValue=",ObjectiveFctValue
   return SolStatus, ObjectiveFctValue

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
      print(i._name+'\n')
   print("Ones:")
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

###########
def Set_ParmValue(ph, ParmName, NewVal):
   # set the value of the named parm in all instances (unindexed)
   rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]
   for scenario in rootnode._scenarios:
      instance = ph._instances[scenario._name]
      pm = getattr(instance, ParmName)
      if pm.is_indexed():
         for index in pm:
            pm[index].value = NewVal[index]
      else:
         pm.value = NewVal ##### dlw adds value :(  Jan 2014
      # required for advanced preprocessing that takes
      # place in PySP
      ph._problem_states.\
         user_constraints_updated[scenario._name] = True

###########
def Get_ParmValueOneScenario(ph, scenarioName, ParmName):
   instance = ph._instances[scenarioName]
   pm = getattr(instance, ParmName)
   values = []
   for index in pm:
      values.append(pm[index].value)
   return values

def PurifyIndVar(ph, IndVarName, tolZero=1.e-6):
   rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]
   for scenario in rootnode._scenarios:
      instance = ph._instances[scenario._name]
      pm = getattr(instance, IndVarName)
      for index in pm:
        delta = pm[index].value
        if   abs(delta)    < tolZero: delta = 0
        elif abs(delta-1.) < tolZero: delta = 1
        else:
            print("\n** delta["+str(index)+"," + scenario._name + "] = "+str(delta))
            print("\tnot within " + str(tolZero) + " of 0 or 1 ... causes sys.exit")
            print("**************************************************************\n")
            sys.exit()
        pm[index].value = delta

   return

def FreeAllIndicatorVariables(ph, IndVarName):
   for scenario in ph._scenario_tree._scenarios:
      FreeIndicatorVariableOneScenario(ph,
                                       scenario,
                                       IndVarName)

def FreeIndicatorVariableOneScenario(ph, scenario, IndVarName):
   instance = ph._instances[scenario._name]
   getattr(instance, IndVarName).free()
   # required for advanced preprocessing that takes
   # place in PySP
   ph._problem_states.\
      freed_variables[scenario._name].\
      append((IndVarName, None))

def FixAllIndicatorVariables(ph, IndVarName, value):
   for scenario in ph._scenario_tree._scenarios:
      FixIndicatorVariableOneScenario(ph,
                                      scenario,
                                      IndVarName,
                                      value)

def FixIndicatorVariableOneScenario(ph,
                                    scenario,
                                    IndVarName,
                                    fix_value):

   instance = ph._instances[scenario._name]
   getattr(instance, IndVarName).fix(fix_value)
   # required for advanced preprocessing that takes
   # place in PySP
   ph._problem_states.\
      fixed_variables[scenario._name].\
      append((IndVarName, None))

def FixFromLists(Lists, ph, IndVarName, CCStageNum):
   # fix variables from the lists
   stage = ph._scenario_tree._stages[CCStageNum-1]
   for tree_node in stage._tree_nodes:
      for scenario in tree_node._scenarios:
         instance = ph._instances[scenario._name]
         fix_value = None
         if scenario in Lists[0]:
            fix_value = 0
         elif scenario in Lists[1]:
            fix_value = 1
         if fix_value is not None:
            # we are assuming no index
            FixIndicatorVariableOneVariable(ph,
                                            scenario,
                                            IndVarName,
                                            fix_value)
   return

###########
def UseListsVariableThenSolve(Lists, ph, IndVarName, CCStageNum):
   # use lists across scenarios of to fix the variable value
   # then solve and return some stuff

   FixFromLists(Lists, ph, IndVarName, CCStageNum)
   LagrangianObj = solve_ph_code(ph)
   b = Compute_ExpectationforVariable(ph, IndVarName, 2)
   print("back from compute exp")

   return LagrangianObj, b

#==============================================
def AlgoExpensiveFlip(TrueForZeros, List, lambdaval, ph, IndVarName, CCStageNum):
   # Flip every delta of every scenario and solve with solve_ph_code(ph)
   # to find scenarios for which we will flip the indicator variable
   # returns list of scenarios sorted by the LagrObj
   if TrueForZeros is True:
      print("ExpensiveFlipAlgo: Fix ZerosToOnes and compute: ")
      value=1
   else:
      print("ExpensiveFlipAlgo: Fix OnesToZeros and compute: ")
      value=0
   D={}
   stage = ph._scenario_tree._stages[CCStageNum-1]
   for tree_node in stage._tree_nodes:
      for scenario in tree_node._scenarios:
         instance = ph._instances[scenario._name]
         if scenario._name in ReturnIndexListNames(List)[abs(value-1)]:
            FixIndicatorVariableOneScenario(ph,
                                            scenario,
                                            IndVarName,
                                            value)

            LagrObj = solve_ph_code(ph)
            b = Compute_ExpectationforVariable(ph, IndVarName, 2)
            z = LagrObj+(b*lambdaval)
            D[z]=[scenario,b]
            print("ObjVal w/ "+str(scenario._name)+
                  " delta flipped to "+str(value)+
                  " (LagrObj,b,z): "+str(LagrObj)+
                  " "+str(b)+" "+str(z))
            FixIndicatorVariableOneScenario(ph,
                                            scenario,
                                            IndVarName,
                                            abs(value-1))

   Dsort = []
   for key in sorted(D):
      Dsort.append(D[key])
   print("---SmallestObjValue:--->  "
         +str(sorted(D)[0])+"  with (LagrObj,b) = "
         +str(sorted(D)[0]-(lambdaval*Dsort[0][1]))+" "
         +str(Dsort[0][1])+"  at  "+str(Dsort[0][0]._name))
   if TrueForZeros is True:
      print("Back from ExpensiveFlipAlgo:Fix ZerosToOnes")
   else:
      print("Back from ExpensiveFlipAlgo:Fix OnesToZeros")
   return Dsort,sorted(D)[0],Dsort[0][1]

def Compute_ExpectationforVariable(ph, IndVarName, CCname, CCStageNum):
   rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]
   ReferenceInstance = ph._instances[rootnode._scenarios[0]._name]  # arbitrary scenario
   CC = getattr(ReferenceInstance,CCname)

   SumSoFar = {}
   for cc in CC: SumSoFar[cc] = 0.0
   node_probability = 0.0
   stage = ph._scenario_tree._stages[CCStageNum-1]
   for tree_node in stage._tree_nodes:
      for scenario in tree_node._scenarios:
         instance = ph._instances[scenario._name]
         node_probability += scenario._probability
         for cc in CC:
            deltaValue = getattr(instance,IndVarName)[cc].value
            SumSoFar[cc] += scenario._probability * deltaValue
   for cc in CC:
      SumSoFar[cc] = SumSoFar[cc] / node_probability
   return SumSoFar

def GetPenaltyCost(ph, IndVarName, multName):
# E[lambda*delta] contribution to cost
   rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]
   ReferenceInstance = ph._instances[rootnode._scenarios[0]._name]  # arbitrary scenario
   CC = ReferenceInstance.ChanceConstraints
   cost = 0.
   for scenario in rootnode._scenarios:
      instance = ph._instances[scenario._name]
      delta = getattr(instance, IndVarName)
      mult  = getattr(instance,multName)
      for index in delta:
        cost = cost + scenario._probability*delta[index].value*mult[index].value
   return cost
