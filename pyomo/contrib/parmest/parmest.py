import re
try:
    import numpy as np
    import pandas as pd
except:
    # some travis tests want to import, but not run much (dlw Oct 2018)
    print ("WARNING: numpy and/or pandas could not be imported.")
import importlib as im
import itertools
import types
import pyomo.environ as pyo
import pyomo.pysp.util.rapper as st
from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
from pyomo.opt import SolverFactory
import pyomo.contrib.parmest.mpi_utils as mpiu
import pyomo.contrib.parmest.ipopt_solver_wrapper as Carl

__version__ = 0.1

#=============================================
def _object_from_string(instance, vstr):
    """
    create a Pyomo object from a string; it is attached to instance
    args:
        instance: a concrete pyomo model
        vstr: a particular Var or Param (e.g. "pp.Keq_a[2]")
    output:
        the object 
    NOTE: We need to deal with blocks 
          and with indexes that might really be strings or ints
    """
    # pull off the index
    l = vstr.find('[')
    if l == -1:
        indexstr = None
        basestr = vstr
    else:
        r = vstr.find(']')
        indexstr = vstr[l+1:r]
        basestr = vstr[:l]
    # get the blocks and the name
    parts = basestr.split('.')
    name = parts[-1]
    retval = instance
    for i in range(len(parts)-1):
        retval = getattr(retval, parts[i])
    retval = getattr(retval, name)
    if indexstr is None:
        return retval
    # dlw jan 2018: TBD improve index handling... multiple indexes, e.g.
    try:
        indexstr = int(indexstr)  # hack...
    except:
        pass
    return retval[indexstr]

#=============================================
def _ef_ROOT_node_Object_from_string(efinstance, vstr):
    """
    Wrapper for _object_from_string for PySP extensive forms
    but only for Vars at the node named ROOT.
    DLW April 2018: needs work to generalized.
    """
    efvstr = "MASTER_BLEND_VAR_Node_ROOT["+vstr+"]"
    return _object_from_string(efinstance, efvstr)

#=============================================
###def _build_compdatalists(model, complist):
    # March 2018: not used
    """
    Convert a list of names of pyomo components (Var and Param)
    into two lists of so-called data objects found on model.

    args:
        model: ConcreteModel
        complist: pyo.Var and pyo.Param names in model
    return:
        vardatalist: a list of Vardata objects (perhaps empty)
        paramdatalist: a list of Paramdata objects or (perhaps empty)
    """
    """
    vardatalist = list()
    paramdatalist = list()

    if complist is None:
        raise RuntimeError("Internal: complist cannot be empty")
    # TBD: require a list (even if it there is only a single element
    
    for comp in complist:
        c = getattr(model, comp)
        if c.is_indexed() and isinstance(c, pyo.Var):
            vardatalist.extend([c[i] for i in sorted(c.keys())])
        elif isinstance(c, pyo.Var):
            vardatalist.append(c)
        elif c.is_indexed() and isinstance(c, pyo.Param):
            paramdatalist.extend([c[i] for i in sorted(c.keys())])
        elif isinstance(c, pyo.Param):
            paramdatalist.append(c)
        else:
            raise RuntimeError("Invalid component list entry= "+\
                               (str(c)) + " Expecting Param or Var")
    
    return vardatalist, paramdatalist
    """    
#=============================================
"""
  This is going to be called by PySP and it will call into
  the user's model's callback.
"""
def _pysp_instance_creation_callback(scenario_tree_model,
                                    scenario_name,
                                    node_names):
    """
    This is going to be called by PySP and it will call into
    the user's model's callback.

    Parameters:
    -----------
    scenario_tree_model: `pysp scenario tree`
        Standard pysp scenario tree, but with things tacked on:
        `CallbackModule` : `str` or `types.ModuleType`
        `CallbackFunction`: `str` or `callable`
        NOTE: if CallbackFunction is callable, you don't need a module.
    scenario_name: `str`
         `cb_data`: optional to pass through to user's callback function
        Scenario name should end with a number
    node_names: `None`
        Not used here 

    Returns:
    --------
    instance: `ConcreteModel`
        instantiated scenario

    Note:
    ----
    There is flexibility both in how the function is passed and its signature.
    """
    scen_num_str = re.compile(r'(\d+)$').search(scenario_name).group(1)
    scen_num = int(scen_num_str)
    basename = scenario_name[:-len(scen_num_str)] # to reconstruct name
    
    # The module and callback function names need to have been on tacked on the tree.
    # We allow a lot of flexibility in these things.
    if not hasattr(scenario_tree_model, "CallbackFunction"):
        raise RuntimeError(\
            "Internal Error: tree needs callback in parmest callback function")
    elif callable(scenario_tree_model.CallbackFunction):
        callback = scenario_tree_model.CallbackFunction
    else:
        cb_name = scenario_tree_model.CallbackFunction

        if not hasattr(scenario_tree_model, "CallbackModule"):
            raise RuntimeError(\
                "Internal Error: tree needs CallbackModule in parmest callback")
        else:
            modname = scenario_tree_model.CallbackModule

        if isinstance(modname, str):
            cb_module = im.import_module(modname, package=None)
        elif isinstance(modname, types.ModuleType):
            cb_module = modname
        else:
            print ("Internal Error: bad CallbackModule")
            raise

        try:
            callback = getattr(cb_module, cb_name)
        except:
            print ("Error getting function="+cb_name+" from module="+str(modname))
            raise
    
    if hasattr(scenario_tree_model, "BootList"):
        bootlist = scenario_tree_model.BootList
        #print ("debug in callback: using bootlist=",str(bootlist))
        # assuming bootlist itself is zero based
        exp_num = bootlist[scen_num]
    else:
        exp_num = scen_num

    scen_name = basename + str(exp_num)

    cb_data = scenario_tree_model.cb_data # cb_data might be None.

    # at least three signatures are supported. The first is preferred
    try:
        instance = callback(experiment_number = exp_num, cb_data = cb_data)
    except TypeError:
        try:
            instance = callback(scenario_tree_model, scen_name, node_names)
        except TypeError:  # deprecated signature?
            try:
                instance = callback(scen_name, node_names)
            except:
                print ("Failed to create instance using callback; TypeError+")
                raise
        except:
            print("Failed to create instance using callback.")
            raise

    if hasattr(scenario_tree_model, "ThetaVals"):
        thetavals = scenario_tree_model.ThetaVals

        # dlw august 2018: see mea code for more general theta
        for vstr in thetavals:
            object = _object_from_string(instance, vstr)
            if thetavals[vstr] is not None:
                #print ("Fixing",vstr,"at",str(thetavals[vstr]))
                object.fix(thetavals[vstr])
            else:
                #print ("Freeing",vstr)
                object.fixed = False

    return instance

#=============================================
def _treemaker(scenlist):
    """Makes a scenario tree (avoids dependence on daps)
    
    Parameters
    ---------- 
    scenlist (list of `int`): experiment (i.e. scenario) numbers

    Returns
    -------
    a `ConcreteModel` that is the scenario tree
    """

    num_scenarios = len(scenlist)
    m = CreateAbstractScenarioTreeModel()
    m.Stages.add('Stage1')
    m.Stages.add('Stage2')
    m.Nodes.add('RootNode')
    for i in scenlist:
        m.Nodes.add('LeafNode_Experiment'+str(i))
        m.Scenarios.add('Experiment'+str(i))
    m = m.create_instance()
    m.NodeStage['RootNode'] = 'Stage1'
    m.ConditionalProbability['RootNode'] = 1.0
    for node in m.Nodes:
        if node != 'RootNode':
            m.NodeStage[node] = 'Stage2'
            m.Children['RootNode'].add(node)
            m.Children[node].clear()
            m.ConditionalProbability[node] = 1.0/num_scenarios
            m.ScenarioLeafNode[node.replace('LeafNode_','')] = node

    return m

#=============================================
class ParmEstimator(object):
    """
    Stores inputs to the parameter estimations process.
    Provides API for getting the parameter estimates, distributions
    and confidence intervals.

    Parameters
    ----------
    gmodel_file : `string`
        Name of py file that has the gmodel_maker function
    gmodel_maker: `string`
        Name if function that makes a concrete pyomo model (gmodel) 
        that is solved for g for a given scenario
    qName: `string`
        The name of the `Expression` (or) `Var` in gmodel that has q 
        after optimization 
    numbers_list: `list` of `int`: 
        Numbers to name experiments (or Samples) 
        (indexes based on this are 1-based)
    thetalist: `list` of `string`
        List of component names (Vars or mutable Params) in gmodel.
    cb_data: `any` (optional)
        Data to be passed through to the callback function, can be of any type.
    tee: `bool` (optional)
        Indicates that ef solver output should be teed (default False)
    """
    def __init__(self, gmodel_file,
                 gmodel_maker, qName, numbers_list, thetalist,
                 cb_data = None,
                 tee=False):
        # NOTE: as of Dec 2017 this needs to be able to
        #       take a string with a number suffix....
        self.gmodel_file = gmodel_file
        self.gmodel_maker = gmodel_maker
        self.qName = qName
        self.numbers_list = numbers_list
        self.thetalist = thetalist
        self.cb_data = cb_data
        self.tee = tee
        self.diagnostic_mode = False

    def _set_diagnostic_mode(self, value):
        if type(value) != bool:
            raise ValueError("diagnostic_mode must be True or False")
        self._diagnostic_mode = value

    def _get_diagnostic_mode(self):
        return self._diagnostic_mode
    
    diagnostic_mode = property(_get_diagnostic_mode, _set_diagnostic_mode)
        
    #==========
    def Q_opt(self, ThetaVals=None, solver="ef_ipopt", bootlist=None):
        """
        Mainly for internal use.

        Set up all thetas as first stage Vars, return resulting theta
        values as well as the objective function value.

        NOTE: If thetavals is present it will be attached to the
        scenario tree so it can be used by the scenario creation
        callback.  Side note (feb 2018, dlw): if you later decide to
        construct the tree just once and reuse it, then remember to
        remove thetavals from it when none is desired.

        Parameters
        ----------
        ThetaVals: `dict` 
            A dictionary of theta values to fix in the pysp callback 
            (which has to grab them from the tree object and do the fixing)
        solver: `string`
            "ef_ipopt" or "k_aug". 
            Only ef is supported if ThetaVals is not None.
        bootlist: `list` of `int`
            The list is of scenario numbers for indirection used internally
            by bootstrap.
            The default is None and that is what driver authors should use.

        Returns
        -------
        objectiveval: `float`
            The objective function value
        thetavals: `dict`
            A dictionary of all values for theta
        Hessian: `dict`
            A dictionary of dictionaries for the Hessian.
            The Hessian is not returned if the solver is ef.
        """
        assert(solver != "k_aug" or ThetaVals == None)
        # Create a tree with dummy scenarios (callback will supply when needed).
        # Which names to use (i.e., numbers) depends on if it is for bootstrap.
        # (Bootstrap scenarios will use indirection through the bootlist)
        if bootlist is None:
            tree_model = _treemaker(self.numbers_list)
        else:
            tree_model = _treemaker(range(len(self.numbers_list)))
        stage1 = tree_model.Stages[1]
        stage2 = tree_model.Stages[2]
        tree_model.StageVariables[stage1] = self.thetalist
        tree_model.StageVariables[stage2] = []
        tree_model.StageCost[stage1] = "FirstStageCost"
        tree_model.StageCost[stage2] = "SecondStageCost"

        # Now attach things to the tree_model to pass them to the callback
        tree_model.CallbackModule = self.gmodel_file
        tree_model.CallbackFunction = self.gmodel_maker
        if ThetaVals is not None:
            tree_model.ThetaVals = ThetaVals
        if bootlist is not None:
            tree_model.BootList = bootlist
        tree_model.cb_data = self.cb_data  # None is OK
        """
        stsolver = st.StochSolver(fsfile = self.gmodel_file,
                                  fsfct = self.gmodel_maker,
                                  tree_model = tree_model)
        """
        stsolver = st.StochSolver(fsfile = "pyomo.contrib.parmest.parmest",
                                  fsfct = "_pysp_instance_creation_callback",
                                  tree_model = tree_model)
        if solver == "ef_ipopt":
            sopts = {}
            sopts['max_iter'] = 6000
            ef_sol = stsolver.solve_ef('ipopt', sopts=sopts, tee=self.tee)
            if self.diagnostic_mode:
                print('    solver termination condition=',
                       str(ef_sol.solver.termination_condition))

            # assume all first stage are thetas...
            thetavals = {}
            for name, solval in stsolver.root_Var_solution():
                 thetavals[name] = solval

            objval = stsolver.root_E_obj()

            return objval, thetavals
        
        elif solver == "k_aug":
            # Just hope for the best with respect to degrees of freedom.

            model = stsolver.make_ef()
            stream_solver = True
            ipopt = SolverFactory('ipopt')
            sipopt = SolverFactory('ipopt_sens')
            kaug = SolverFactory('k_aug')

            #: ipopt suffixes  REQUIRED FOR K_AUG!
            model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
            model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            model.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
            model.ipopt_zU_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)

            # declare the suffix to be imported by the solver
            model.red_hessian = pyo.Suffix(direction=pyo.Suffix.EXPORT)
            #: K_AUG SUFFIXES
            model.dof_v = pyo.Suffix(direction=pyo.Suffix.EXPORT) 
            model.rh_name = pyo.Suffix(direction=pyo.Suffix.IMPORT)

            for vstrindex in range(len(self.thetalist)):
                vstr = self.thetalist[vstrindex]
                varobject = _ef_ROOT_node_Object_from_string(model, vstr)
                varobject.set_suffix_value(model.red_hessian, vstrindex+1)
                varobject.set_suffix_value(model.dof_v, 1)
            
            #: rh_name will tell us which position the corresponding variable has on the reduced hessian text file.
            #: be sure to declare the suffix value (order)
            # dof_v is "degree of freedom variable"
            kaug.options["compute_inv"] = ""  #: if the reduced hessian is desired.
            #: please check the inv_.in file if the compute_inv option was used

            #: write some options for ipopt sens
            with open('ipopt.opt', 'w') as f:
                f.write('compute_red_hessian yes\n')  #: computes the reduced hessian (sens_ipopt)
                f.write('output_file my_ouput.txt\n')
                f.write('rh_eigendecomp yes\n')
                f.close()
            #: Solve
            sipopt.solve(model, tee=stream_solver)
            with open('ipopt.opt', 'w') as f:
                f.close()

            ipopt.solve(model, tee=stream_solver)

            model.ipopt_zL_in.update(model.ipopt_zL_out)
            model.ipopt_zU_in.update(model.ipopt_zU_out)

            #: k_aug
            print('k_aug \n\n\n')
            #m.write('problem.nl', format=ProblemFormat.nl)
            kaug.solve(model, tee=stream_solver)
            HessDict = {}
            thetavals = {}
            print('k_aug red_hess')
            with open('result_red_hess.txt', 'r') as f:
                lines = f.readlines()
            # asseble the return values
            objval = model.MASTER_OBJECTIVE_EXPRESSION.expr()
            for i in range(len(lines)):
                HessDict[self.thetalist[i]] = {}
                linein = lines[i]
                print(linein)
                parts = linein.split()
                for j in range(len(parts)):
                    HessDict[self.thetalist[i]][self.thetalist[j]] = \
                        float(parts[j])
                # Get theta value (there is probably a better way...)
                vstr = self.thetalist[i]
                varobject = _ef_ROOT_node_Object_from_string(model, vstr)
                thetavals[self.thetalist[i]] = pyo.value(varobject)
            return objval, thetavals, HessDict

        else:
            raise RuntimeError("Unknown solver in Q_Opt="+solver)
        
    #==========
    def theta_est(self, solver="ef_ipopt", bootlist=None):
        """return a theta estimate
        NOTE: To avoid risk, one should probably set all thetvals to None 
        and pass in the dict to Q_opt rather than call this function.

        Parameters
        ----------
        solver: `string`
            As of April 2018: "ef_ipopt" or "k_aug". 
            Default is "ef_ipopt".
        bootlist: `list` of `int`
            The list is of scenario numbers for indirection used by bootstrap.
            The default is None and that is what you driver users should use.

        Returns
        -------
        objectiveval: `float`
            The objective function value
        thetavals: `dict`
            A dictionary of all values for theta
        Hessian: `dict`
            A dictionary of dictionaries for the Hessian.
            The Hessian is not returned if the solver is ef.
        """

        return self.Q_opt(solver=solver, bootlist=bootlist)
            
    #==========
    def Q_at_theta(self, thetavals):
        """
        Return the objective function value with fixed theta values.
        
        Parameters
        ----------
        thetavals: `dict`
            A dictionary of theta values.

        Returns
        -------
        objectiveval: `float`
            The objective function value.
        thetavals: `dict`
            A dictionary of all values for theta that were input.
        solvertermination: `Pyomo TerminationCondition`
            Tries to return the "worst" solver status across the scenarios.
            pyo.TerminationCondition.optimal is the best and 
            pyo.TerminationCondition.infeasible is the worst.
        """

        optimizer = pyo.SolverFactory('ipopt')
        dummy_tree = lambda: None # empty object (we don't need a tree)
        dummy_tree.CallbackModule = self.gmodel_file
        dummy_tree.CallbackFunction = self.gmodel_maker
        dummy_tree.ThetaVals = thetavals
        dummy_tree.cb_data = self.cb_data
        
        if self.diagnostic_mode:
            print('    Compute Q_at_Theta=',str(thetavals))

        # start block of code to deal with models with no constraints
        # (ipopt will crash or complain on such problems without special care)
        instance = _pysp_instance_creation_callback(dummy_tree, "FOO1", None)    
        try: # deal with special problems so Ipopt will not crash
            first = next(instance.component_objects(pyo.Constraint, active=True))
        except:
            sillylittle = True 
        else:
            sillylittle = False
        # end block of code to deal with models with no constraints

        WorstStatus = pyo.TerminationCondition.optimal
        totobj = 0
        for snum in self.numbers_list:
            sname = "scenario_NODE"+str(snum)
            instance = _pysp_instance_creation_callback(dummy_tree,
                                                        sname, None)
            if not sillylittle:
                if self.diagnostic_mode:
                    print('      Experiment=',snum)
                    print ('     first solve with with special diagnostics wrapper')
                    status_obj, solved, iters, time, regu \
                        = Carl.ipopt_solve_with_stats(instance, optimizer, max_iter=500, max_cpu_time=120)
                    print ("   status_obj, solved, iters, time, regularization_stat=",
                           str(status_obj), str(solved), str(iters), str(time), str(regu))

                results = optimizer.solve(instance)
                if self.diagnostic_mode:
                    print ('standard solve solver termination condition=',
                            str(results.solver.termination_condition))

                if results.solver.termination_condition \
                   != pyo.TerminationCondition.optimal :
                    # DLW: Aug2018: not distinguishing "middlish" conditions
                    if WorstStatus != pyo.TerminationCondition.infeasible:
                        WorstStatus = results.solver.termination_condition
                    
            objobject = getattr(instance, self.qName)
            objval = pyo.value(objobject)
            totobj += objval
        retval = totobj / len(self.numbers_list) # -1??
        return retval, thetavals, WorstStatus

    #==========
    def _Estimate_Hessian(self, thetavals, epsilon=1e-1):
        """
        Unused as of August 2018
        Crude estimate of the Hessian of Q at thetavals

        Parameters
        ----------
        thetavals: `dict`
            A dictionary of values for theta

        Return
        ------
        FirstDeriv: `dict`
            Dictionary of scaled first differences
        HessianDict: `dict`
            Matrix (in dicionary form) of Hessian values
        """
        """
        def firstdiffer(tvals, tstr):
            tvals[tstr] = tvals[tstr] - epsilon / 2
            lval, foo, w = self.Q_at_theta(tvals)
            tvals[tstr] = tvals[tstr] + epsilon / 2
            rval, foo, w = self.Q_at_theta(tvals)
            tvals[tstr] = thetavals[tstr]
            return rval - lval

        # make a working copy of thetavals and get the Hessian dict started
        tvals = {}
        Hessian = {}
        for tstr in thetavals:
            tvals[tstr] = thetavals[tstr]
            Hessian[tstr] = {}
        
        # get "basline" first differences
        firstdiffs = {}
        for tstr in tvals:
            # TBD, dlw jan 2018: check for bounds on theta
            print ("debug firstdiffs for",tstr)
            firstdiffs[tstr] = firstdiffer(tvals, tstr)

        # now get the second differences
        # as of Jan 2018, do not assume symmetry so it can be "checked."
        for firstdim in tvals:
            for seconddim in tvals:
                print ("debug H for",firstdim,seconddim)
                tvals[seconddim] = thetavals[seconddim] + epsilon
                d2 = firstdiffer(tvals, firstdim)
                Hessian[firstdim][seconddim] = \
                        (d2 - firstdiffs[firstdim]) / (epsilon * epsilon) 
                tvals[seconddim] = thetavals[seconddim]

        FirstDeriv = {}
        for tstr in thetavals:
            FirstDeriv[tstr] = firstdiffs[tstr] / epsilon

        return FirstDeriv, Hessian
        """
    
    def bootstrap(self, N):
        """
        Run parameter estimation using N bootstap samples

        Parameters
        ----------
        N: `int`
            Number of bootstrap samples to draw

        Returns
        -------
        bootstrap_theta_list: `DataFrame`
            Samples and theta values from the bootstrap
        """
		
        bootstrap_theta = list()
        samplesize = len(self.numbers_list)  

        task_mgr = mpiu.ParallelTaskManager(N)
        global_bootlist = list()
        for i in range(N):
            j = unique_samples = 0
            while unique_samples <= len(self.thetalist):
                bootlist = np.random.choice(self.numbers_list,
                                            samplesize,
                                            replace=True)
                unique_samples = len(np.unique(bootlist))
                j += 1
                if j > N: # arbitrary timeout limit
                    raise RuntimeError("Internal error: timeout in bootstrap"+\
                                    " constructing a sample; possible hint:"+\
                                    " the dim of theta may be too close to N")
            global_bootlist.append((i, bootlist))

        local_bootlist = task_mgr.global_to_local_data(global_bootlist)

        for idx, bootlist in local_bootlist:
            #print('Bootstrap Run Number: ', idx + 1, ' out of ', N)
            objval, thetavals = self.theta_est(bootlist=bootlist)
            thetavals['samples'] = bootlist
            bootstrap_theta.append(thetavals)#, ignore_index=True)
        
        global_bootstrap_theta = task_mgr.allgather_global_data(bootstrap_theta)
        bootstrap_theta = pd.DataFrame(global_bootstrap_theta)
        #bootstrap_theta.set_index('samples', inplace=True)        

        return bootstrap_theta
        
    
    def likelihood_ratio(self, search_ranges=None):
        """
        Compute the likelihood ratio and return the entire mesh

        Parameters
        ----------
        search_ranges: `dictionary` of lists indexed by theta.
            Mesh points (might be optional in the future)

        Returns
        -------
        SSE: `DataFrame`
            Sum of squared errors values for the entire mesh unless
            some mesh points are infeasible, which are omitted.
        """

        ####
        def mesh_generator(search_ranges):
            # return the next theta point given by search_ranges
            """ from the web:
            def product_dict(**kwargs):
                keys = kwargs.keys()
                vals = kwargs.values()
                for instance in itertools.product(*vals):
                    yield dict(zip(keys, instance))
            """
            keys = search_ranges.keys()
            vals = search_ranges.values()
            for prod in itertools.product(*vals):
                yield dict(zip(keys, prod))

        # for parallel code we need to use lists and dicts in the loop
        all_SSE = list()
        global_mesh = list()
        MeshLen = 0
        for Theta in mesh_generator(search_ranges):
            MeshLen += 1
            global_mesh.append(Theta)
        task_mgr = mpiu.ParallelTaskManager(MeshLen)
        local_mesh = task_mgr.global_to_local_data(global_mesh)
        
        # walk over the mesh, using the objective function to get squared error
        for Theta in local_mesh:
            SSE, thetvals, worststatus = self.Q_at_theta(Theta)
            if worststatus != pyo.TerminationCondition.infeasible:
                 all_SSE.append(list(Theta.values()) + [SSE])
            # DLW, Aug2018: should we also store the worst solver status?
            
        global_all_SSE = task_mgr.allgather_global_data(all_SSE)
        dfcols = list(search_ranges.keys())+["SSE"]
        store_all_SSE = pd.DataFrame(data=global_all_SSE, columns=dfcols)

        return store_all_SSE
        
