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
import types
from pyomo.pysp.scenariotree import *
from pyomo.pysp.phinit import *
from pyomo.pysp.ph import *
from pyomo.pysp.ef import *
from pyomo.opt import SolverFactory

# Tear the scenario instances off the ef instance when it is no longer required
# so warnings are not generated next time scenarios instances are placed inside
# a new ef instance
def _tear_down_ef(ef_instance, scenario_instances):
   for name in scenario_instances:
      ef_instance.del_component(name)

def solve_ph_code(ph, options):
   # consolidate the code to solve the problem for the "global" ph object
   # return a solver code (string from the solver if EF, "PH" if PH) and the objective fct val
   SolStatus = None

   if options.solve_with_ph is False:
      if options.verbose is True:
         print("Creating the extensive form.")
         print("Time="+time.asctime())
      ef = create_ef_instance(ph._scenario_tree,
                              verbose_output=options.verbose)
      ef.preprocess()

      if options.verbose is True:
         print("Time="+time.asctime())
         print("Solving the extensive form.")

      ef_results = solve_ef(ef, ph._instances, options)
      
      SolStatus = str(ef_results.solution.status) ##HG: removed [0] to get full solution status string
      print("SolStatus="+SolStatus)
      if options.verbose is True:
         print("Loading extensive form solution.")
         print("Time="+time.asctime())
      ### If the solution is infeasible, we don't want to load the results
      ### It is up to the caller to decide what to do with non-optimal
      if SolStatus != "infeasible" and SolStatus != "unknown": 
         ef.load(ef_results)  
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
   else:
      if options.verbose is True:
         print("Solving via Progressive Hedging.")
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
        ph._push_fixed_to_instances()
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
        skip_canonical_repn = False
        if (options.solver_type == "asl") or (options.solver_type == "ipopt"):
          skip_canonical_repn = True
        binding_instance = create_ef_instance(ph._scenario_tree)
        binding_instance.preprocess()
        ef_instance_end_time = time.time()
        print("Time to construct extensive form instance=%.2f seconds" %(ef_instance_end_time - ef_instance_start_time))

      #
      # solve the extensive form and load the solution back into the PH scenario tree.
      # contents from the PH solve will obviously be over-written!
      #
      if options.write_ef:
         output_filename = os.path.expanduser(options.ef_output_file)
         # technically, we don't need the symbol map since we aren't solving it.
         print("Starting to write the extensive form")
         ef_write_start_time = time.time()
         symbol_map = write_ef(binding_instance, output_filename, symbolic_solver_labels=options.symbolic_solver_labels)
         ef_write_end_time = time.time()
         print("Extensive form written to file="+output_filename)
         print("Time to write output file=%.2f seconds" %(ef_write_end_time - ef_write_start_time))

      if options.solve_ef:

         # set the value of each non-converged, non-final-stage variable to None - 
         # this will avoid infeasible warm-stats.
         reset_nonconverged_variables(ph._scenario_tree, ph._instances)
         reset_stage_cost_variables(ph._scenario_tree, ph._instances)

         # create the solver plugin.
         ef_solver = SolverFactory(options.solver_type, solver_io=options.solver_io)
         if ef_solver is None:
            raise ValueError("Failed to create solver of type="+options.solver_type+" for use in extensive form solve")
         if options.keep_solver_files:
            ef_solver.keepFiles = True
         if len(options.ef_solver_options) > 0:
            print("Initializing ef solver with options="+str(options.ef_solver_options))
            ef_solver.set_options("".join(options.ef_solver_options))
         if options.ef_mipgap is not None:
            if (options.ef_mipgap < 0.0) or (options.ef_mipgap > 1.0):
               raise ValueError("Value of the mipgap parameter for the EF solve must be on the unit interval; value specified="+str(options.ef_mipgap))
            ef_solver.options.mipgap = float(options.ef_mipgap)

         # create the solver manager plugin.
         ef_solver_manager = SolverManagerFactory(options.ef_solver_manager_type)
         if ef_solver_manager is None:
            raise ValueError("Failed to create solver manager of type="+options.solver_type+" for use in extensive form solve")
         elif isinstance(ef_solver_manager, pyomo.plugins.smanager.phpyro.SolverManager_PHPyro):
            raise ValueError("Cannot solve an extensive form with solver manager type=phpyro")

         print("Queuing extensive form solve")
         ef_solve_start_time = time.time()
         if (options.disable_ef_warmstart) or (ef_solver.warm_start_capable() is False):        
            ef_action_handle = ef_solver_manager.queue(binding_instance, opt=ef_solver, tee=options.output_ef_solver_log)
         else:
            ef_action_handle = ef_solver_manager.queue(binding_instance, opt=ef_solver, tee=options.output_ef_solver_log, warmstart=True)            
         print("Waiting for extensive form solve")
         ef_results = ef_solver_manager.wait_for(ef_action_handle)

         print("Done with extensive form solve - loading results")
         binding_instance.load(ef_results)

         print("Storing solution in scenario tree")
         ph._scenario_tree.pullScenarioSolutionsFromInstances()
         ph._scenario_tree.snapshotSolutionFromScenarios()

         ef_solve_end_time = time.time()
         print("Time to solve and load results for the extensive form=%.2f seconds" %(ef_solve_end_time - ef_solve_start_time))

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

   # end copy from phinit
   ph._scenario_tree.pullScenarioSolutionsFromInstances()
   root_node = ph._scenario_tree._stages[0]._tree_nodes[0]
   ObjectiveFctValue = root_node.computeExpectedNodeCost() 
   if options.solve_with_ph is False:
      if str(ef_results.solution.status)[0] is not "o":
         ObjectiveFctValue=1e100
   else:
      if phretval is not None:
         ObjectiveFctValue=1e100
         #print "test"
   ## print "(using PySP Cost vars) ObjectiveFctValue=",ObjectiveFctValue
   return SolStatus, ObjectiveFctValue

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

      instance.preprocess() # needed to re-evaluate expressions

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
   instance.preprocess()
   return

def FixAllIndicatorVariables(ph,VarName,value):
   rootnode = ph._scenario_tree._stages[0]._tree_nodes[0]
   for scenario in rootnode._scenarios:
      instance = ph._instances[scenario._name]
      pm = getattr(instance, VarName)
      if pm.is_indexed():
         for index in pm:
            getattr(instance, VarName)[index].value = value
            getattr(instance, VarName)[index].fixed = True
      else:
            getattr(instance, VarName).value = value
            getattr(instance, VarName).fixed = True 
      instance.preprocess()
   return 

def UnfixParmValueOneScenario(ph, scenario, ParmName):
   instance = ph._instances[scenario._name]
   getattr(instance, ParmName).fixed = False
   instance.preprocess()
   
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
            Set_ParmValueOneScenarioAndFix(ph, scenario, IndVarName, fix_value)
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
            Set_ParmValueOneScenarioAndFix(ph, scenario, IndVarName, value)
            LagrObj = solve_ph_code(ph)
            b = Compute_ExpectationforVariable(ph, IndVarName, 2)
            z = LagrObj+(b*lambdaval)
            D[z]=[scenario,b]
            print("ObjVal w/ "+str(scenario._name)+" delta flipped to "+str(value)+" (LagrObj,b,z): "+str(LagrObj)+" "+str(b)+" "+str(z))
            Set_ParmValueOneScenarioAndFix(ph, scenario, IndVarName, abs(value-1))
   Dsort = []
   for key in sorted(D):
      Dsort.append(D[key])
   print("---SmallestObjValue:--->  "+str(sorted(D)[0])+"  with (LagrObj,b) = "+str(sorted(D)[0]-(lambdaval*Dsort[0][1]))+" "+str(Dsort[0][1])+"  at  "+str(Dsort[0][0]._name))
   if TrueForZeros is True:
      print("Back from ExpensiveFlipAlgo:Fix ZerosToOnes")
   else:
      print("Back from ExpensiveFlipAlgo:Fix OnesToZeros")
   return Dsort,sorted(D)[0],Dsort[0][1]


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
         raise ValueError("Value of the mipgap parameter for the EF solve must be on the unit interval; value specified="+str(options.ef_mipgap))
      else:
         ef_solver.mipgap = options.ef_mipgap
   if options.keep_solver_files is True:
      ef_solver.keepFiles = True                   
   
   ef_solver_manager = SolverManagerFactory(options.solver_manager_type)
   if ef_solver is None:
      raise ValueError("Failed to create solver manager of type="+options.solver_type+" for use in extensive form solve")

   if options.verbose:
      print("Solving extensive form.")
   if ef_solver.warm_start_capable():
      ef_action_handle = ef_solver_manager.queue(master_instance, opt=ef_solver, warmstart=False, tee=options.output_ef_solver_log, symbolic_solver_labels=options.symbolic_solver_labels)
   else:
      ef_action_handle = ef_solver_manager.queue(master_instance, opt=ef_solver, tee=options.output_ef_solver_log, symbolic_solver_labels=options.symbolic_solver_labels)      
   ef_results = ef_solver_manager.wait_for(ef_action_handle) 

   if options.verbose:
      print("solve_ef() finished.")
   return ef_results

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

