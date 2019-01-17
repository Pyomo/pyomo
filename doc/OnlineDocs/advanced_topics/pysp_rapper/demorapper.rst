.. _demosect:

Demonstration of rapper Capabilities
====================================

..
   doctest:: I can't stop output from PySP so I can't test. And also:

   I think it is a bad idea to try to insist that output is the same
   every time this runs. I have other tests of this code, so it should
   be enough for the doctest just make sure there are no exceptions.

   I have tried +ELLIPSIS in various ways, but can't make it work, so
   I am testing as far as I can, then disabling.

In this section we provide a series of examples intended to show different things that
can be done with rapper.

.. testsetup:: *
	       
   import tempfile
   import sys
   import os
   import shutil
   import pyomo as pyomoroot

   tdir = tempfile.mkdtemp()    #TemporaryDirectory().name
   sys.path.insert(1, tdir)

   savecwd = os.getcwd()
   os.chdir(tdir)

   p = str(pyomoroot.__path__)
   l = p.find("'")
   r = p.find("'", l+1)
   pyomorootpath = p[l+1:r]
   farmpath = pyomorootpath + os.sep + ".." + os.sep + "examples" + os.sep + "pysp" + os.sep + "farmer"
   farmpath = os.path.abspath(farmpath)
        
   farmer_concrete_file = farmpath + os.sep + \
                          "concrete" + os.sep + "ReferenceModel.py"

   shutil.copyfile(farmer_concrete_file,
                   tdir + os.sep + "ReferenceModel.py")
        
   shutil.copyfile(farmpath + os.sep +"scenariodata" + os.sep + "ScenarioStructure.dat",
                   tdir + os.sep + "ScenarioStructure.dat")

.. doctest::

   Imports:

   >>> import pyomo.pysp.util.rapper as rapper
   >>> import pyomo.pysp.plugins.csvsolutionwriter as csvw
   >>> from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
   >>> import pyomo.environ as pyo

   The next line establishes the solver to be used.
   
   >>> solvername = "cplex"

   The next two lines show one way to create a concrete scenario tree. There are
   others that can be found in `pyomo.pysp.scenariotree.tree_structure_model`.

   >>> abstract_tree = CreateAbstractScenarioTreeModel()
   >>> concrete_tree = \
   ...     abstract_tree.create_instance("ScenarioStructure.dat")


Emulate some aspects of `runef`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Create a `rapper` solver object assuming there is a
   file named `ReferenceModel.py` that has an appropriate
   `pysp_instance_creation_callback` function.

   >>> stsolver = rapper.StochSolver("ReferenceModel.py",
   ...            tree_model = concrete_tree)

   This object has a `solve_ef` method (as well as a `solve_ph` method)
   
   >>> ef_sol = stsolver.solve_ef(solvername) # doctest: +SKIP

   The return status from the solver can be tested.

   >>> if ef_sol.solver.termination_condition != \ # doctest: +SKIP
   ...            pyo.TerminationCondition.optimal: # doctest: +SKIP
   ...     print ("oops! not optimal:",ef_sol.solver.termination_condition) # doctest: +SKIP

   There is an iterator to loop over the root node solution:
   
   >>> for varname, varval in stsolver.root_Var_solution(): # doctest: +SKIP
   ...    print (varname, str(varval)) # doctest: +SKIP

   There is also a function to compute compute the objective
   function value.
   
   >>> obj = stsolver.root_E_obj() # doctest: +SKIP
   >>> print ("Expecatation take over scenarios=", obj) # doctest: +SKIP
   
.. testoutput::
   :hide:
   :options: +ELLIPSIS

   Also, `stsolver.scenario_tree` has the solution (csvw is imported
   from PySP and is not part of `rapper`.)
   
   >>> csvw.write_csv_soln(stsolver.scenario_tree, "testcref") # doctest: +SKIP

Again, but with mip gap reported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
   Now we will solve the same problem again, but we cannot re-use the
   same `rapper.StochSolver` object in the same program so we must construct
   a new one; however, we can re-used the scenario tree.

   >>> stsolver = rapper.StochSolver("ReferenceModel.py", # doctest: +SKIP
   ...            tree_model = concrete_tree) # doctest: +SKIP

   We add a solver option to get the mip gap
   
   >>> sopts = {"mipgap": 1} # I want a gap

   and we add the option to `solve_ef` to return the gap and
   the `tee` option to see the solver output as well.
   
   >>> res, gap = stsolver.solve_ef(solvername, sopts = sopts, tee=True, need_gap = True) # doctest: +SKIP
   >>> print ("ef gap=",gap) # doctest: +SKIP

PH
^^

   We will now do the same problem, but with PH and we will re-use the scenario
   tree in `tree_model` from the code above. We put sub-solver options in
   `sopts` and PH options (i.e., those that would provided to `runph`) 
   Note that if options are passed to the constructor (and the solver);
   they are passed as a dictionary where options that do not have
   an argument have the data value `None`. The constructor really only
   needs to some options, such as those related to bundling.

   >>> sopts = {}
   >>> sopts['threads'] = 2
   >>> phopts = {}
   >>> phopts['--output-solver-log'] = None
   >>> phopts['--max-iterations'] = '3'

   >>> stsolver = rapper.StochSolver("ReferenceModel.py", 
   ...                               tree_model = concrete_tree, 
   ...                               phopts = phopts) 

   The `solve_ph` method is similar to `solve_ef`, but requires
   a `default_rho` and accepts PH options:
   
   >>> ph = stsolver.solve_ph(subsolver = solvername, default_rho = 1, # doctest: +SKIP
   ...                        phopts=phopts) # doctest: +SKIP

   With PH, it is important to be careful to distinguish x-bar from x-hat.
   
   >>> obj = stsolver.root_E_obj() # doctest: +SKIP

   We can compute and x-hat (using the current PH options):
   
   >>> obj, xhat = rapper.xhat_from_ph(ph) # doctest: +SKIP

   There is a utility for obtaining the x-hat values:
   
   >>> for nodename, varname, varvalue in rapper.xhat_walker(xhat): # doctest: +SKIP
   ...     print (nodename, varname, varvalue) # doctest: +SKIP
   
.. testcleanup:: *

   os.chdir(savecwd)


	     
