Abstract Constructor
====================

..
   doctest:: I can't stop output from PySP so I can't test. And also:

   I think it is a bad idea to try to insist that output is the same
   every time this runs. I have other tests of this code, so it should
   be enough for the doctest just make sure there are no exceptions.

   I have tried +ELLIPSIS in various ways, but can't make it work, so
   I am testing as far as I can, then disabling.

In :ref:`demosect` we provide a series of examples intended to show different things that can be done with rapper and the constructor is shown
for a `ConcreteModel`. The same capabilities are available for
an `AbstractModel` but the construction of the `rapper` object is
different as shown here.

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
        
   farmer_abstract_file = farmpath + os.sep + \
                          "models" + os.sep + "ReferenceModel.py"

   shutil.copyfile(farmer_abstract_file,
                   tdir + os.sep + "ReferenceModel.py")
        
   fromdir = farmpath + os.sep +"scenariodata" 

   for filename in os.listdir(fromdir):
      if filename.endswith(".dat"):
         src = str(fromdir + os.sep + filename)
         shutil.copyfile(src, tdir + os.sep + filename)

   ReferencePath = "."
   scenariodirPath = "."

   solvername = "cplex"

.. doctest::

   Import for constructor:

   >>> import pyomo.pysp.util.rapper as rapper

The next line constructs the `rapper` object that can be used
to emulate `runph` or `runef`.

   >>> stsolver = rapper.StochSolver(ReferencePath,
   ...                               fsfct = None,
   ...                               tree_model = scenariodirPath,
   ...                               phopts = None)
   
.. testcleanup:: *

   os.chdir(savecwd)


	     
