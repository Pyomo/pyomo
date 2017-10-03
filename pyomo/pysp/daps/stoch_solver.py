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

def _kwfromphopts(phopts):
    """
    This is really local to the StochSolver __init__ but
    I moved it way out to make the init more readable. The
    function takes the phopts dictionary and returns
    a kwargs dictionary suitable for a call to generate_scenario_tree.
    Note that only some options (i.e., bundle options) are needed
    when the tree is created. The rest can be passed in when the
    ph object is created.
    inputs:
        phopts: a ph options dictionary.
    return:
        kwargs: a dictionary suitable for a call to generate_scenario_tree.
    """
    kwargs = {}
    def dointpair(pho, fo):
        if pho in phopts:
            kwargs[fo] = int(phopts[pho])
        else:
            kwargs[fo] = None
    if phopts is not None:
        dointpair("--create-random-bundles", 'random_bundles')
        dointpair("--scenario-tree-seed", 'random_seed')
        if "--scenario-tree-downsample-fraction" in phopts:
            kwargs['downsample_fraction'] = \
                    float(phopts["--scenario-tree-downsample-fraction"])
        else:
            kwargs['downsample_fraction'] = None
            
        if "--scenario-bundle-specification" in phopts:
            kwargs['bundles'] = phopts["--scenario-tree-bundle-specification"]
        else:
            kwargs['bundles'] = None

    return kwargs


#==================================
class StochSolver:
    """
    A class for solving stochastic versions of concrete models.
    Inspired by the IDAES use case and by daps ability to create tree models.
    Author: David L. Woodruff, February 2017
    
    Members: scenario_tree: scenario tree object (that includes data)
             solve_ef: a function that solves the ef problem for the tree
             solve_ph: solves the problem in the tree using PH
    """
    def __init__(self, fsfile, tree_model = None, phopts = None):
        """
        inputs: 
          fsfile: is a file that contains the the scenario callback.
            We require a hard-wired function name in the file, which is
            "pysp_instance_creation_callback"
          tree_model: gives the tree as a concrete model
            if it is None, then look for a function in fsfile called
            "pysp_scenario_tree_model_callback" that will return it.
          phopts: dictionary of ph options; needed if there is bundling.
        
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

        kwargs = _kwfromphopts(phopts)
        self.scenario_tree = \
            scenario_instance_factory.generate_scenario_tree(**kwargs) #verbose = True)
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
    def solve_ph(self, subsolver, default_rho, phopts = None, sopts = None):
        """
        Solve the stochastic program given by this.scenario_tree using ph
        Update the scenario tree, populated with the solution.
        args:
            subsolver: the solver to call (e.g., 'ipopt')
            phopts: dictionary of ph options
            sopts: dictionary of subsolver options
        """

        ph = None

        # Build up the options for PH.
        parser = phinit.construct_ph_options_parser("")
        phargslist = ['--default-rho',str(default_rho)]
        phargslist.append('--solver')
        phargslist.append(str(subsolver))
        if phopts is not None:
            for key in phopts:
                phargslist.append(key)
                if phopts[key] is not None:
                    phargslist.append(phopts[key])
                    
        # Subproblem options go to PH as space-delimited, equals-separated pairs.
        if sopts is not None:
            soptstring = ""
            for key in sopts:
                soptstring += key + '=' + str(sopts[key]) + ' '
            phargslist.append('--scenario-solver-options')    
            phargslist.append(soptstring)
        phoptions = parser.parse_args(phargslist)

        # construct the PH solver object
        try:
            ph = phinit.PHAlgorithmBuilder(phoptions, self.scenario_tree)
        except:
            print ("Internal error: ph construction failed.")
            if ph is not None:
                ph.release_components()
            raise

        retval = ph.solve()
        if retval is not None:
            raise RuntimeError("ph Failure Encountered="+str(retval))
        # dlw May 2017: I am not sure if the next line is really needed
        ph.save_solution()
