Demonstration of rapper Capabilities
====================================

We provide a series of examples intended to show different things that
can be done with rapper.

.. testsetup:: *

   import tempfile
   import sys
   import os
   import shutil
   import pyomo as pyomoroot

   tdir = tempfile.mkdtemp()    #TemporaryDirectory().name
   sys.path.insert(1,self.tdir)

   savecwd = os.getcwd()
   os.chdir(self.tdir)

   p = str(pyomoroot.__path__)
   l = p.find("'")
   r = p.find("'", l+1)
   pyomorootpath = p[l+1:r]
   farmpath = pyomorootpath + os.sep + ".." + os.sep + "examples" + \
              os.sep + "pysp" + os.sep + "farmer"
   farmpath = os.path.abspath(farmpath)
        
   farmer_concrete_file = farmpath + os.sep + \
                          "concrete" + os.sep + "ReferenceModel.py"

   shutil.copyfile(self.farmer_concrete_file,
                   self.tdir + os.sep + "ReferenceModel.py")
        
   abstract_tree = CreateAbstractScenarioTreeModel()
   shutil.copyfile(farmpath + os.sep +"scenariodata" + os.sep + "ScenarioStructure.dat",
                   self.tdir + os.sep + "ScenarioStructure.dat")
   farmer_concrete_tree = \
       abstract_tree.create_instance(self.tdir + os.sep + "ScenarioStructure.dat")

.. doctest::

   Imports:

   >>> import pyomo.pysp.util.rapper as rapper
   >>> import pyomo.pysp.plugins.csvsolutionwriter as csvw
   >>> from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
   >>> import pyomo.environ as pyo

   The next line establishes the solver to be used.
   
   >>> solvername = "gurobi"

   The next two lines show one way to create a concrete scenario tree. There are
   others that can be found in `pyomo.pysp.scenariotree.tree_structure_model`.

   >>> abstract_tree = CreateAbstractScenarioTreeModel()
   >>> concrete_tree = \
   >>>     abstract_tree.create_instance("ScenarioStructure.dat")


Emulate some aspects of `runef`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Create a `rapper` solver object assuming there is a
   file named `ReferenceModel.py` that has an appropriate
   `pysp_instance_creation_callback` function.

   >>> stsolver = rapper.StochSolver("ReferenceModel.py",
   >>>            tree_model = concrete_tree)

   This object has a `solve_ef` method (as well as a `solve_ph` method)
   
   >>> ef_sol = stsolver.solve_ef(solvername)

   The return status from the solver can be tested.

   >>> if ef_sol.solver.termination_condition != \
   >>>            pyo.TerminationCondition.optimal:
   >>>     print ("oops! not optimal:",ef_sol.solver.termination_condition)

   There is an iterator to loop over the root node solution:
   
   >>> for varname, varval in stsolver.root_Var_solution():
   >>>     print (varname, str(varval))

   There is also a function to compute compute the objective
   function value.
   
   >>> obj = stsolver.root_E_obj()
   >>> print ("Expecatation take over scenarios=", obj)

   Also, `stsolver.scenario_tree` has the solution (csvw is imported
   from PySP and is not part of `rapper`.)
   
   >>> csvw.write_csv_soln(stsolver.scenario_tree, "testcref")

Again, but with mip gap reported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
   Now we will solve the same problem again, but we cannot re-use the
   same `rapper.StochSolver` object in the same program so we must construct
   a new one; however, we can re-used the scenario tree.

   >>> stsolver = rapper.StochSolver("ReferenceModel.py",
   >>>            tree_model = concrete_tree)

   We add a solver option to get the mip gap
   
   >>> sopts = {"mipgap": 1} # I want a gap

   and we add the option to `solve_ef` to return the gap and
   the `tee` option to see the solver output as well.
   
   >>> res, gap = stsolver.solve_ef(solvername, sopts = sopts, tee=True, need_gap = True)
   >>> print ("ef gap=",gap)

PH
^^

   We will now do the same problem, but with PH and we will re-use the scenario
   tree in `tree_model` from the code above. We put sub-solver options in
   `sopts` and PH options (i.e., those that would provided to `runph`) 
   
   >>> sopts = {}
   >>> sopts['threads'] = 2
   >>> phopts = {}
   >>> phopts['--output-solver-log'] = None
   >>> phopts['--max-iterations'] = '3'

   >>> stsolver = rapper.StochSolver("ReferenceModel.py",
   >>>                               tree_model = concrete_tree,
   >>>                               phopts = phopts)

   The `solve_ph` method is similar to `solve_ef`, but requires
   a `default_rho` and accepts PH options:
   
   >>> ph = stsolver.solve_ph(subsolver = solvername, default_rho = 1,
   >>>                        phopts=phopts)

   With PH, it is important to be careful to distinguish x-bar from x-hat.
   
   >>> obj = stsolver.root_E_obj() # E[xbar]

   We can compute and x-hat (using the current PH options):
   
   >>> obj, xhat = rapper.xhat_from_ph(ph)

   There is a utility for obtaining the x-hat values:
   
   >>> for nodename, varname, varvalue in rapper.xhat_walker(xhat):
   >>>     print (nodename, varname, varvalue)
   
.. testcleanup:: *

   os.cwd(savecwd)
