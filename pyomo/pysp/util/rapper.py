"""
A class and some utilities to wrap PySP.
In particular to enable programmatic access to some of
the functionality in runef and runph for ConcreteModels
Author: David L. Woodruff, started February 2017
"""

import inspect
from pyomo.environ import *
from pyomo.pysp.scenariotree.instance_factory \
    import ScenarioTreeInstanceFactory
import pyomo.pysp.ef as pyspef # import (create_ef_instance, solve_ef)
from pyomo.pysp.ef_writer_script import ExtensiveFormAlgorithm
from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
from pyomo.pysp.scenariotree.instance_factory import \
                ScenarioTreeInstanceFactory

import pyomo.pysp.phinit as phinit
import os

def _optiondict_2_list(phopts, args_list = None):
    """ A little utility to change the format of options"""
    if args_list is None:
        args_list = []
    if phopts is not None:
        for key in phopts:
            args_list.append(key)
            if phopts[key] is not None:
                args_list.append(phopts[key])
    return args_list

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
        if pho in phopts and phopts[pho] is not None:
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
    """A class for solving stochastic versions of concrete models and
    abstract models.
    Inspired by the IDAES use case and by daps ability to create tree models.
    Author: David L. Woodruff, February 2017
    
    Args: 
      fsfile (str): is a path to the file that contains the scenario 
                    callback for concrete or the reference model for abstract.
      fsfct (str, or fct, or None): 
         str   callback function name in the file
         fct   callback function
         None  it is a AbstractModel
      tree_model (concrete model, or path): 
        gives the tree as a concrete model (which could be a fct)
        or path to AMPL data file.
      phopts: dictionary of ph options; needed during construction 
              if there is bundling.

    Attributes:
       scenario_tree: scenario tree object (that includes data)

    """
    def __init__(self, fsfile,
                 fsfct = None,
                 tree_model = None,
                 phopts = None):
        """Initialize a StochSolver object.
        """
        if fsfct is None:
            # Changed in October 2018: None implies AbstractModel
            args_list = _optiondict_2_list(phopts)
            parser = phinit.construct_ph_options_parser("")
            options = parser.parse_args(args_list)

            scenario_instance_factory = \
                ScenarioTreeInstanceFactory(fsfile, tree_model)

            try:
                self.scenario_tree = \
                    phinit.GenerateScenarioTreeForPH(options,
                                                     scenario_instance_factory)
            except:
                print ("ERROR in StochSolver called from",inspect.stack()[1][3])
                raise RuntimeError("fsfct is None, so assuming",
                      "AbstractModel but could not find all ingredients.")
                
        else:  # concrete model
            if  callable(fsfct):
                scen_function = fsfct
            else: # better be a string
                fsfile = fsfile.replace('.py','')  # import does not like .py
                # __import__ only gives the top level module
                # probably need to be dealing with modules installed via setup.py
                m = __import__(fsfile)
                for n in fsfile.split(".")[1:]:
                    m = getattr(m, n)
                scen_function = getattr(m, fsfct)

            if tree_model is None:
                treecbname = "pysp_scenario_tree_model_callback"
                tree_maker = getattr(m, treecbname)

                tree = tree_maker()
                if isinstance(tree, Pyo.ConcreteModel):
                    tree_model = tree
                else:
                    raise RuntimeError("The tree returned by",treecbname,
                                       "must be a ConcreteModel") 
                    
                scenario_instance_factory = ScenarioTreeInstanceFactory(scen_function, tree_model)

            else: 
                # DLW March 21: still not correct
                scenario_instance_factory = \
                    ScenarioTreeInstanceFactory(scen_function, tree_model)


            kwargs = _kwfromphopts(phopts)
            self.scenario_tree = \
                scenario_instance_factory.generate_scenario_tree(**kwargs) #verbose = True)
            instances = scenario_instance_factory. \
                        construct_instances_for_scenario_tree(self.scenario_tree)
            self.scenario_tree.linkInInstances(instances)        

    #=========================
    def make_ef(self, verbose=False):
        """ Make an ef object (used by solve_ef)
        
        Args:
            verbose (boolean): indicates verbosity to PySP for construction

        Returns:
            ef_instance: the ef object
        """
        return pyspef.create_ef_instance(self.scenario_tree, verbose_output=verbose)
    
    def solve_ef(self, subsolver, sopts = None, tee = False, need_gap = False):
        """Solve the stochastic program directly using the extensive form.
       
        Args:
            subsolver (str): the solver to call (e.g., 'ipopt')
            sopts (dict):  solver options
            tee (bool): indicates dynamic solver output to terminal.
            need_gap (bool): indicates the need for the optimality gap

        Returns: (`Pyomo solver result`, `float`)

                solve_result is the solver return value.

                absgap is the absolute optimality gap (might not be valid); only if requested      

        Note:
           Also update the scenario tree, populated with the solution.
           Also attach the full ef instance to the object. So you might want
           obj = pyo.value(stsolver.ef_instance.MASTER)
           This needs more work to deal with solver failure (dlw, March, 2018)

        """
        
        self.ef_instance = self.make_ef()
        solver = SolverFactory(subsolver)
        if sopts is not None:
            for key in sopts:
                solver.options[key] = sopts[key]

        if need_gap:
            solve_result = solver.solve(self.ef_instance, tee = tee, load_solutions=False)
            if len(solve_result.solution) > 0:
                absgap = solve_result.solution(0).gap
            else:
                absgap = None
            self.ef_instance.solutions.load_from(solve_result)
        else:
            solve_result = solver.solve(self.ef_instance, tee = tee)

        # note: the objective is probably called MASTER
        #print ("debug value(ef_instance.MASTER)=",value(ef_instance.MASTER))
        self.scenario_tree.pullScenarioSolutionsFromInstances()
        self.scenario_tree.snapshotSolutionFromScenarios() # update nodes
        if need_gap:
            return solve_result, absgap
        else:
            return solve_result

    #=========================
    def solve_ph(self, subsolver, default_rho, phopts = None, sopts = None):
        """Solve the stochastic program given by this.scenario_tree using ph

        Args:
            subsolver (str): the solver to call (e.g., 'ipopt')
            default_rho (float): the rho value to use by default
            phopts: dictionary of ph options (optional)
            sopts: dictionary of subsolver options (optional)

        Returns: the ph object

        Note:
            Updates the scenario tree, populated with the xbar values; 
            however, you probably want to do
            obj, xhat = ph.compute_and_report_inner_bound_using_xhat()
            where ph is the return value.

        """

        ph = None

        # Build up the options for PH.
        parser = phinit.construct_ph_options_parser("")
        phargslist = ['--default-rho',str(default_rho)]
        phargslist.append('--solver')
        phargslist.append(str(subsolver))
        phargslist = _optiondict_2_list(phopts, args_list = phargslist)
                    
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

        return ph

    #=========================
    def root_Var_solution(self):
        """Generator to loop over x-bar

        Yields:
            name, value pair for root node solution values
        """
        root_node = self.scenario_tree.findRootNode()
        for variable_id in sorted(root_node._variable_ids):
            var_name, index = root_node._variable_ids[variable_id]
            name = var_name
            if index is not None:
                name += "["+str(index)+"]"
            yield name, root_node._solution[variable_id]

    #=========================
    def root_E_obj(self):
        """post solve Expected cost of the solution in the scenario tree (xbar)

        Returns:
            float: the expected costs of the solution in the tree (xbar)
        """
        root_node = self.scenario_tree.findRootNode()
        return root_node.computeExpectedNodeCost()

#=========================
def xhat_from_ph(ph):
    """a service fuction to wrap a call to get xhat

    Args:
        ph: a post-solve ph object

    Returns: (float, object)

        float: the expected cost of the xhat solution for the scenarios

        xhat: an object with the solution tree
    """
    obj, xhat = ph.compute_and_report_inner_bound_using_xhat()
    return obj, xhat

#=========================
def xhat_walker(xhat):
    """A service generator to walk over a given xhat

    Args:
        xhat (dict): an xhat solution  (probably from xhat_from_ph)

    Yields:
        (nodename, varname, varvalue)
    """
    for nodename in xhat:
        for varname, varvalue in xhat[nodename].items():
            yield (nodename, varname, varvalue)


