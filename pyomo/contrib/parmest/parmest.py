import re
import importlib as im
import types
import json
try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except:
    pass

import pyomo.environ as pyo
import pyomo.pysp.util.rapper as st
from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
from pyomo.opt import SolverFactory

import pyomo.contrib.parmest.mpi_utils as mpiu
import pyomo.contrib.parmest.ipopt_solver_wrapper as ipopt_solver_wrapper
from pyomo.contrib.parmest.graphics import pairwise_plot

__version__ = 0.1

#=============================================
def _object_from_string(instance, vstr):
    """
    Create a Pyomo object from a string; it is attached to instance
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
    but only for Vars at the node named RootNode.
    DLW April 2018: needs work to be generalized.
    """
    efvstr = "MASTER_BLEND_VAR_RootNode["+vstr+"]"
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

def _pysp_instance_creation_callback(scenario_tree_model,
                                    scenario_name, node_names):
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
            print("Internal Error: bad CallbackModule")
            raise

        try:
            callback = getattr(cb_module, cb_name)
        except:
            print("Error getting function="+cb_name+" from module="+str(modname))
            raise
    
    if hasattr(scenario_tree_model, "BootList"):
        bootlist = scenario_tree_model.BootList
        #print("debug in callback: using bootlist=",str(bootlist))
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
                print("Failed to create instance using callback; TypeError+")
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
                #print("Fixing",vstr,"at",str(thetavals[vstr]))
                object.fix(thetavals[vstr])
            else:
                #print("Freeing",vstr)
                object.fixed = False

    return instance

#=============================================
def _treemaker(scenlist):
    """
    Makes a scenario tree (avoids dependence on daps)
    
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

    
def group_data(data, groupby_column_name, use_mean=None):
    """
    Group data by experiment/scenario
    
    Parameters
    ----------
    data: DataFrame
        Data
    groupby_column_name: strings
        Name of data column which contains experiment/scenario numbers
    use_mean: list of column names or None, optional
        Name of data columns which should be reduced to a single value per 
        experiment/scenario by taking the mean
        
    Returns
    ----------
    grouped_data: list of dictionaries
        Grouped data
    """
    grouped_data = []
    for exp_num, group in data.groupby(data[groupby_column_name]):
        d = {}
        for col in group.columns:
            if col in use_mean:
                d[col] = group[col].mean()
            else:
                d[col] = list(group[col])
        grouped_data.append(d)

    return grouped_data


class _SecondStateCostExpr(object):
    """
    Class to pass objective expression into the Pyomo model
    """
    def __init__(self, ssc_function, data):
        self._ssc_function = ssc_function
        self._data = data
    def __call__(self, model):
        return self._ssc_function(model, self._data)


class Estimator(object):
    """
    Parameter estimation class. Provides methods for parameter estimation, 
    bootstrap resampling, and likelihood ratio test.

    Parameters
    ----------
    model_function: function
        Function that generates an instance of the Pyomo model using 'data' 
        as the input argument
    data: pandas DataFrame, list of dictionaries, or list of json file names
        Data that is used to build an instance of the Pyomo model and build 
        the objective function
    theta_names: list of strings
        List of Vars to estimate
    obj_function: function, optional
        Function used to formulate parameter estimation objective, generally
        sum of squared error between measurments and model variables.  
        If no function is specified, the model is used 
        "as is" and should be defined with a "FirstStateCost" and 
        "SecondStageCost" expression that are used to build an objective 
        for pysp.
    tee: bool, optional
        Indicates that ef solver output should be teed
    diagnostic_mode: bool, optional
        if True, print diagnostics from the solver
    """
    def __init__(self, model_function, data, theta_names, obj_function=None, 
                 tee=False, diagnostic_mode=False):
        
        self.model_function = model_function
        self.callback_data = data
        self.theta_names = theta_names 
        self.obj_function = obj_function 
        self.tee = tee
        self.diagnostic_mode = diagnostic_mode
        
        self._second_stage_cost_exp = "SecondStageCost"
        self._numbers_list = list(range(len(data)))
        

    def _create_parmest_model(self, data):
        """
        Modify the Pyomo model for parameter estimation
        """
        from pyomo.core import Objective
        
        model = self.model_function(data)

        for theta in self.theta_names:
            try:
                var_validate = eval('model.'+theta)
                var_validate.fixed = False
            except:
                print(theta +'is not a variable')
        
        if self.obj_function:
            for obj in model.component_objects(Objective):
                obj.deactivate()
        
            def FirstStageCost_rule(model):
                return 0
            model.FirstStageCost = pyo.Expression(rule=FirstStageCost_rule)
            model.SecondStageCost = pyo.Expression(rule=_SecondStateCostExpr(self.obj_function, data))
            
            def TotalCost_rule(model):
                return model.FirstStageCost + model.SecondStageCost
            model.Total_Cost_Objective = pyo.Objective(rule=TotalCost_rule, sense=pyo.minimize)
        
        self.parmest_model = model
        
        return model
    
    
    def _instance_creation_callback(self, experiment_number=None, cb_data=None):
        
        # DataFrame
        if isinstance(cb_data, pd.DataFrame):
            # Keep single experiments in a Dataframe (not a Series)
            exp_data = cb_data.loc[experiment_number,:].to_frame().transpose() 
        
        # List of dictionaries OR list of json file names
        elif isinstance(cb_data, list):
            exp_data = cb_data[experiment_number]
            if isinstance(exp_data, dict):
                pass
            if isinstance(exp_data, str):
                try:
                    with open(exp_data,'r') as infile:
                        exp_data = json.load(infile)
                except:
                    print('Unexpected data format')
                    return
        else:
            print('Unexpected data format')
            return
        model = self._create_parmest_model(exp_data)
        
        return model
    

    def _Q_opt(self, ThetaVals=None, solver="ef_ipopt", bootlist=None):
        """
        Set up all thetas as first stage Vars, return resulting theta
        values as well as the objective function value.

        NOTE: If thetavals is present it will be attached to the
        scenario tree so it can be used by the scenario creation
        callback.  Side note (feb 2018, dlw): if you later decide to
        construct the tree just once and reuse it, then remember to
        remove thetavals from it when none is desired.
        """
        assert(solver != "k_aug" or ThetaVals == None)
        # Create a tree with dummy scenarios (callback will supply when needed).
        # Which names to use (i.e., numbers) depends on if it is for bootstrap.
        # (Bootstrap scenarios will use indirection through the bootlist)
        if bootlist is None:
            tree_model = _treemaker(self._numbers_list)
        else:
            tree_model = _treemaker(range(len(self._numbers_list)))
        stage1 = tree_model.Stages[1]
        stage2 = tree_model.Stages[2]
        tree_model.StageVariables[stage1] = self.theta_names
        tree_model.StageVariables[stage2] = []
        tree_model.StageCost[stage1] = "FirstStageCost"
        tree_model.StageCost[stage2] = "SecondStageCost"

        # Now attach things to the tree_model to pass them to the callback
        tree_model.CallbackModule = None
        tree_model.CallbackFunction = self._instance_creation_callback
        if ThetaVals is not None:
            tree_model.ThetaVals = ThetaVals
        if bootlist is not None:
            tree_model.BootList = bootlist
        tree_model.cb_data = self.callback_data  # None is OK

        stsolver = st.StochSolver(fsfile = "pyomo.contrib.parmest.parmest",
                                  fsfct = "_pysp_instance_creation_callback",
                                  tree_model = tree_model)
        if solver == "ef_ipopt":
            sopts = {}
            sopts['max_iter'] = 6000
            ef_sol = stsolver.solve_ef('ipopt', sopts=sopts, tee=self.tee)
            if self.diagnostic_mode:
                print('    Solver termination condition = ',
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

            for vstrindex in range(len(self.theta_names)):
                vstr = self.theta_names[vstrindex]
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
                HessDict[self.theta_names[i]] = {}
                linein = lines[i]
                print(linein)
                parts = linein.split()
                for j in range(len(parts)):
                    HessDict[self.theta_names[i]][self.theta_names[j]] = \
                        float(parts[j])
                # Get theta value (there is probably a better way...)
                vstr = self.theta_names[i]
                varobject = _ef_ROOT_node_Object_from_string(model, vstr)
                thetavals[self.theta_names[i]] = pyo.value(varobject)
            return objval, thetavals, HessDict

        else:
            raise RuntimeError("Unknown solver in Q_Opt="+solver)
        

    def _Q_at_theta(self, thetavals):
        """
        Return the objective function value with fixed theta values.
        
        Parameters
        ----------
        thetavals: dict
            A dictionary of theta values.

        Returns
        -------
        objectiveval: float
            The objective function value.
        thetavals: dict
            A dictionary of all values for theta that were input.
        solvertermination: Pyomo TerminationCondition
            Tries to return the "worst" solver status across the scenarios.
            pyo.TerminationCondition.optimal is the best and 
            pyo.TerminationCondition.infeasible is the worst.
        """

        optimizer = pyo.SolverFactory('ipopt')
        dummy_tree = lambda: None # empty object (we don't need a tree)
        dummy_tree.CallbackModule = None
        dummy_tree.CallbackFunction = self._instance_creation_callback
        dummy_tree.ThetaVals = thetavals
        dummy_tree.cb_data = self.callback_data
        
        if self.diagnostic_mode:
            print('    Compute objective at theta = ',str(thetavals))

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
        for snum in self._numbers_list:
            sname = "scenario_NODE"+str(snum)
            instance = _pysp_instance_creation_callback(dummy_tree,
                                                        sname, None)
            if not sillylittle:
                if self.diagnostic_mode:
                    print('      Experiment = ',snum)
                    print('     First solve with with special diagnostics wrapper')
                    status_obj, solved, iters, time, regu \
                        = ipopt_solver_wrapper.ipopt_solve_with_stats(instance, optimizer, max_iter=500, max_cpu_time=120)
                    print("   status_obj, solved, iters, time, regularization_stat = ",
                           str(status_obj), str(solved), str(iters), str(time), str(regu))

                results = optimizer.solve(instance)
                if self.diagnostic_mode:
                    print('standard solve solver termination condition=',
                            str(results.solver.termination_condition))

                if results.solver.termination_condition \
                   != pyo.TerminationCondition.optimal :
                    # DLW: Aug2018: not distinguishing "middlish" conditions
                    if WorstStatus != pyo.TerminationCondition.infeasible:
                        WorstStatus = results.solver.termination_condition
                    
            objobject = getattr(instance, self._second_stage_cost_exp)
            objval = pyo.value(objobject)
            totobj += objval
        retval = totobj / len(self._numbers_list) # -1??
        return retval, thetavals, WorstStatus


    def _Estimate_Hessian(self, thetavals, epsilon=1e-1):
        """
        Unused, Crude estimate of the Hessian of Q at thetavals

        Parameters
        ----------
        thetavals: dict
            A dictionary of values for theta

        Return
        ------
        FirstDeriv: dict
            Dictionary of scaled first differences
        HessianDict: dict
            Matrix (in dicionary form) of Hessian values
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
            print("Debug firstdiffs for ",tstr)
            firstdiffs[tstr] = firstdiffer(tvals, tstr)

        # now get the second differences
        # as of Jan 2018, do not assume symmetry so it can be "checked."
        for firstdim in tvals:
            for seconddim in tvals:
                print("Debug H for ",firstdim,seconddim)
                tvals[seconddim] = thetavals[seconddim] + epsilon
                d2 = firstdiffer(tvals, firstdim)
                Hessian[firstdim][seconddim] = \
                        (d2 - firstdiffs[firstdim]) / (epsilon * epsilon) 
                tvals[seconddim] = thetavals[seconddim]

        FirstDeriv = {}
        for tstr in thetavals:
            FirstDeriv[tstr] = firstdiffs[tstr] / epsilon

        return FirstDeriv, Hessian
    
    
    def theta_est(self, solver="ef_ipopt", bootlist=None): 
        """
        Run parameter estimation using all data

        Parameters
        ----------
        solver: string, optional
            "ef_ipopt" or "k_aug". Default is "ef_ipopt".

        Returns
        -------
        objectiveval: float
            The objective function value
        thetavals: dict
            A dictionary of all values for theta
        Hessian: dict
            A dictionary of dictionaries for the Hessian.
            The Hessian is not returned if the solver is ef.
        """
        return self._Q_opt(solver=solver, bootlist=bootlist)
    
    
    def theta_est_bootstrap(self, N, samplesize=None, replacement=True, seed=None, return_samples=False):
        """
        Run parameter estimation using N bootstap samples

        Parameters
        ----------
        N: int
            Number of bootstrap samples to draw from the data
        samplesize: int or None, optional
            Sample size, if None samplesize will be set to the number of experiments
        replacement: bool, optional
            Sample with or without replacement
        seed: int or None, optional
            Set the random seed
        return_samples: bool, optional
            Return a list of experiment numbers used in each bootstrap estimation
        
        Returns
        -------
        bootstrap_theta: DataFrame 
            Theta values for each bootstrap sample and (if return_samples = True) 
            the sample numbers used in each estimation
        """
        bootstrap_theta = list()
        
        if samplesize is None:
            samplesize = len(self._numbers_list)  
        if seed is not None:
            np.random.seed(seed)
            
        task_mgr = mpiu.ParallelTaskManager(N)
        global_bootlist = list()
        for i in range(N):
            j = unique_samples = 0
            while unique_samples <= len(self.theta_names):
                bootlist = np.random.choice(self._numbers_list,
                                            samplesize,
                                            replace=replacement)
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

        if not return_samples:
            del bootstrap_theta['samples']
                    
        return bootstrap_theta
    
    
    def objective_at_theta(self, theta_values):
        """
        Compute the objective over a range of theta values

        Parameters
        ----------
        theta_values: DataFrame, columns=theta_names
            Values of theta used to compute the objective
            
        Returns
        -------
        obj_at_theta: DataFrame
            Objective values for each theta value (infeasible solutions are 
            omitted).
        """
        # for parallel code we need to use lists and dicts in the loop
        theta_names = theta_values.columns
        all_thetas = theta_values.to_dict('records')
        task_mgr = mpiu.ParallelTaskManager(len(all_thetas))
        local_thetas = task_mgr.global_to_local_data(all_thetas)
        
        # walk over the mesh, return objective function
        all_obj = list()
        for Theta in local_thetas:
            obj, thetvals, worststatus = self._Q_at_theta(Theta)
            if worststatus != pyo.TerminationCondition.infeasible:
                 all_obj.append(list(Theta.values()) + [obj])
            # DLW, Aug2018: should we also store the worst solver status?
            
        global_all_obj = task_mgr.allgather_global_data(all_obj)
        dfcols = list(theta_names) + ['obj']
        obj_at_theta = pd.DataFrame(data=global_all_obj, columns=dfcols)

        return obj_at_theta
    
    
    def likelihood_ratio_test(self, obj_at_theta, obj_value, alpha, 
                              return_thresholds=False):
        """
        Compute the likelihood ratio for each value of alpha
        
        Parameters
        ----------
        obj_at_theta: DataFrame, columns = theta_names + 'obj'
            Objective values for each theta value (returned by 
            objective_at_theta)
            
        obj_value: float
            Objective value from parameter estimation using all data
        
        alpha: list
            List of alpha values to use in the chi2 test
        
        return_thresholds: bool, optional
            Return the threshold value for each alpha
            
        Returns
        -------
        LR: DataFrame 
            Objective values for each theta value along wit True or False for 
        thresholds: dictionary
            If return_threshold = True, the thresholds are also returned.
        """
        LR = obj_at_theta.copy()
        S = len(self.callback_data)
        thresholds = {}
        for a in alpha:
            chi2_val = stats.chi2.ppf(a, 2)
            thresholds[a] = obj_value * ((chi2_val / (S - 2)) + 1)
            LR[a] = LR['obj'] < thresholds[a]
        
        if return_thresholds:
            return LR, thresholds
        else:
            return LR
