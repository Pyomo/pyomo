"""
A class for solving stochastic programs with data created by daps.
Author: David L. Woodruff, February 2017
"""

from pyomo.environ import *
from pyomo.pysp.scenariotree.instance_factory \
    import ScenarioTreeInstanceFactory
from pyomo.pysp.ef import create_ef_instance
import pyomo.pysp.phinit as phinit
import os

#==================================
class StochSolver:
    """
    A class for solving stochastic versions of concrete models.
    Inspired by the IDAES use case and by daps ability to create tree models.
    Author: David L. Woodruff, February 2017
    
    Members: scenario_tree: scenario tree object (that includes data)
             solve_ef: a function that solves the ef problem for the tree
             solve_serial_ph: in progress, March 2017
             solve_parallel_ph: tbd
    """
    def __init__(self, fsfile, tree_model = None):
        """
        inputs: 
          fsfile: is a file that contains the the scenario callback.
            We require a hard-wired function name in the file, which is
            "pysp_instance_creation_callback"
          tree_model: gives the tree as a concrete model
            if it is None, then look for a function in fsfile called
            "pysp_scenario_tree_model_callback" that will return it.
        """
        fsfile = fsfile.replace('.py','')  # import does not like .py

        scen_function = getattr(__import__(fsfile), \
                                "pysp_instance_creation_callback")

        if tree_model is None:
            tree_maker = getattr(__import__(fsfile), \
                                 "pysp_scenario_tree_model_callback")
            tree_model = tree_maker()

        scenario_instance_factory = \
            ScenarioTreeInstanceFactory(scen_function, tree_model)

        self.scenario_tree = \
            scenario_instance_factory.generate_scenario_tree() #verbose = True)
 
        instances = scenario_instance_factory. \
                    construct_instances_for_scenario_tree(self.scenario_tree)
        self.scenario_tree.linkInInstances(instances)        

    #=========================
    def solve_ef(self, subsolver, sopts = None, tee = False):
        """
        Solve the stochastic program directly using the extensive form.
        args:
        subsolver: the solver to call (e.g., 'ipopt')
        sopts: dictionary of solver options
        tee: the usual to indicate dynamic solver output to terminal.
        Update the scenario tree, populated with the solution.
        """
        
        ef_instance = create_ef_instance(self.scenario_tree, verbose_output=True)
        solver = SolverFactory(subsolver)
        if sopts is not None:
            for key in sopts:
                solver.options[key] = sopts[key]

        solver.solve(ef_instance, tee = tee)

        self.scenario_tree.pullScenarioSolutionsFromInstances()
        self.scenario_tree.snapshotSolutionFromScenarios() # update nodes

    #=========================
    def solve_serial_ph(self, subsolver, default_rho, phopts = None, sopts = None):
        # Solve the stochastic program given by this.scenario_tree using ph
        # subsolver: the solver to call (e.g., 'ipopt')
        # phopts: dictionary ph options
        # sopts: dictionary of subsolver options
        # Returns 

        ph = None
        parser = phinit.construct_ph_options_parser("")
        options = parser.parse_args(['--default-rho',str(default_rho)])
        ###!!!! tbd get options from argument !!!!! and delete next line
        ###try:

        ###scenario_tree = \
            ###phinit.GenerateScenarioTreeForPH(options,
                                 ### scenario_instance_factory)

        ph = phinit.PHAlgorithmBuilder(options, self.scenario_tree)

        ###except:
        ###    print ("Internal error: ph construction failed."
        ###    if ph is not None:
        ###        ph.release_components()
        ###    raise

        retval = ph.solve()
        if retval is not None:
            raise RuntimeError("ph Failure Encountered="+str(retval))
        print ("foobar of victory: HEY: get a solution writer; e.g., from phinit.py and/or return something")
