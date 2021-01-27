#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import re
import importlib as im
import logging
import types
import json
from itertools import combinations

from pyomo.common.dependencies import (
    attempt_import,
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
    scipy, scipy_available,
)

import pyomo.environ as pyo
import pyomo.pysp.util.rapper as st
from pyomo.pysp.scenariotree import tree_structure
from pyomo.pysp.scenariotree.tree_structure_model import (
    CreateAbstractScenarioTreeModel
)
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID

import pyomo.contrib.parmest.mpi_utils as mpiu
import pyomo.contrib.parmest.ipopt_solver_wrapper as ipopt_solver_wrapper
import pyomo.contrib.parmest.graphics as graphics

parmest_available = numpy_available & pandas_available & scipy_available

inverse_reduced_hessian, inverse_reduced_hessian_available = attempt_import(
    'pyomo.contrib.interior_point.inverse_reduced_hessian')

logger = logging.getLogger(__name__)

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
    m = CreateAbstractScenarioTreeModel().create_instance()
    m.Stages.add('Stage1')
    m.Stages.add('Stage2')
    m.Nodes.add('RootNode')
    for i in scenlist:
        m.Nodes.add('LeafNode_Experiment'+str(i))
        m.Scenarios.add('Experiment'+str(i))
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
    Group data by scenario
    
    Parameters
    ----------
    data: DataFrame
        Data
    groupby_column_name: strings
        Name of data column which contains scenario numbers
    use_mean: list of column names or None, optional
        Name of data columns which should be reduced to a single value per 
        scenario by taking the mean
        
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
    Parameter estimation class

    Parameters
    ----------
    model_function: function
        Function that generates an instance of the Pyomo model using 'data' 
        as the input argument
    data: pandas DataFrame, list of dictionaries, or list of json file names
        Data that is used to build an instance of the Pyomo model and build 
        the objective function
    theta_names: list of strings
        List of Var names to estimate
    obj_function: function, optional
        Function used to formulate parameter estimation objective, generally
        sum of squared error between measurements and model variables.  
        If no function is specified, the model is used 
        "as is" and should be defined with a "FirstStateCost" and 
        "SecondStageCost" expression that are used to build an objective 
        for pysp.
    tee: bool, optional
        Indicates that ef solver output should be teed
    diagnostic_mode: bool, optional
        If True, print diagnostics from the solver
    solver_options: dict, optional
        Provides options to the solver (also the name of an attribute)
    """
    def __init__(self, model_function, data, theta_names, obj_function=None, 
                 tee=False, diagnostic_mode=False, solver_options=None):
        
        self.model_function = model_function
        self.callback_data = data

        if len(theta_names) == 0:
            self.theta_names = ['parmest_dummy_var']
        else:
            self.theta_names = theta_names 
            
        self.obj_function = obj_function 
        self.tee = tee
        self.diagnostic_mode = diagnostic_mode
        self.solver_options = solver_options
        
        self._second_stage_cost_exp = "SecondStageCost"
        self._numbers_list = list(range(len(data)))


    def _create_parmest_model(self, data):
        """
        Modify the Pyomo model for parameter estimation
        """
        from pyomo.core import Objective
        
        model = self.model_function(data)
        
        if (len(self.theta_names) == 1) and (self.theta_names[0] == 'parmest_dummy_var'):
            model.parmest_dummy_var = pyo.Var(initialize = 1.0)
            
        for i, theta in enumerate(self.theta_names):
            # First, leverage the parser in ComponentUID to locate the
            # component.  If that fails, fall back on the original
            # (insecure) use of 'eval'
            var_cuid = ComponentUID(theta)
            var_validate = var_cuid.find_component_on(model)
            if var_validate is None:
                logger.warning(
                    "theta_name[%s] (%s) was not found on the model",
                    (i, theta))
            else:
                try:
                    # If the component that was found is not a variable,
                    # this will generate an exception (and the warning
                    # in the 'except')
                    var_validate.fixed = False
                    # We want to standardize on the CUID string
                    # representation (which is what PySP will use
                    # internally)
                    self.theta_names[i] = repr(var_cuid)
                except:
                    logger.warning(theta + ' is not a variable')
        
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
    

    def _Q_opt(self, ThetaVals=None, solver="ef_ipopt",
               return_values=[], bootlist=None, calc_cov=False):
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

        try:
            # For structured models, it is important that we use the
            # updated version of the CUID representation.  PySP (for
            # backwards compatibility reasons, and so a ton of tests
            # don't have to be updated) still defaults to the old
            # representation.
            _cuidver = tree_structure.CUID_repr_version
            tree_structure.CUID_repr_version = 2
            stsolver = st.StochSolver(
                fsfile = "pyomo.contrib.parmest.parmest",
                fsfct = "_pysp_instance_creation_callback",
                tree_model = tree_model
            )
        finally:
            tree_structure.CUID_repr_version = _cuidver
                
        # Solve the extensive form with ipopt
        if solver == "ef_ipopt":
        
            # Generate the extensive form of the stochastic program using pysp
            self.ef_instance = stsolver.make_ef()

            # need_gap is a holdover from solve_ef in rapper.py. Would we ever want
            # need_gap = True with parmest?
            need_gap = False
            
            assert not (need_gap and self.calc_cov), "Calculating both the gap and reduced hessian (covariance) is not currently supported."

            if not calc_cov:
                # Do not calculate the reduced hessian

                solver = SolverFactory('ipopt')
                if self.solver_options is not None:
                    for key in self.solver_options:
                        solver.options[key] = self.solver_options[key]

                if need_gap:
                    solve_result = solver.solve(self.ef_instance, tee = self.tee, load_solutions=False)
                    if len(solve_result.solution) > 0:
                        absgap = solve_result.solution(0).gap
                    else:
                        absgap = None
                    self.ef_instance.solutions.load_from(solve_result)
                else:
                    solve_result = solver.solve(self.ef_instance, tee = self.tee)

            # The import error will be raised when we attempt to use
            # inv_reduced_hessian_barrier below.
            #
            #elif not asl_available:
            #    raise ImportError("parmest requires ASL to calculate the "
            #                      "covariance matrix with solver 'ipopt'")
            else:
                # parmest makes the fitted parameters stage 1 variables
                # thus we need to convert from var names (string) to 
                # Pyomo vars
                ind_vars = []
                for v in self.theta_names:

                    #ind_vars.append(eval('ef.'+v))
                    ind_vars.append(self.ef_instance.MASTER_BLEND_VAR_RootNode[v])
        
                # calculate the reduced hessian
                solve_result, inv_red_hes = \
                    inverse_reduced_hessian.inv_reduced_hessian_barrier(
                        self.ef_instance,
                        independent_variables= ind_vars,
                        solver_options=self.solver_options,
                        tee=self.tee)
            
            # Extract solution from pysp
            stsolver.scenario_tree.pullScenarioSolutionsFromInstances()
            stsolver.scenario_tree.snapshotSolutionFromScenarios() # update nodes
                                
            if self.diagnostic_mode:
                print('    Solver termination condition = ',
                       str(solve_result.solver.termination_condition))

            # assume all first stage are thetas...
            thetavals = {}
            for name, solval in stsolver.root_Var_solution():
                 thetavals[name] = solval

            objval = stsolver.root_E_obj()
            
            if calc_cov:
                # Calculate the covariance matrix
                
                # Extract number of data points considered
                n = len(self.callback_data)
                
                # Extract number of fitted parameters
                l = len(thetavals)
                
                # Assumption: Objective value is sum of squared errors
                sse = objval
                
                '''Calculate covariance assuming experimental observation errors are
                independent and follow a Gaussian 
                distribution with constant variance.
                
                The formula used in parmest was verified against equations (7-5-15) and
                (7-5-16) in "Nonlinear Parameter Estimation", Y. Bard, 1974.
                
                This formula is also applicable if the objective is scaled by a constant;
                the constant cancels out. (PySP scaled by 1/n because it computes an
                expected value.)
                '''
                cov = 2 * sse / (n - l) * inv_red_hes
                cov = pd.DataFrame(cov, index=thetavals.keys(), columns=thetavals.keys())
            
            if len(return_values) > 0:
                var_values = []
                for exp_i in self.ef_instance.component_objects(Block, descend_into=False):
                    vals = {}
                    for var in return_values:
                        exp_i_var = exp_i.find_component(str(var))
                        temp = [pyo.value(_) for _ in exp_i_var.itervalues()]
                        if len(temp) == 1:
                            vals[var] = temp[0]
                        else:
                            vals[var] = temp                    
                    var_values.append(vals)                    
                var_values = pd.DataFrame(var_values)
                if calc_cov:
                    return objval, thetavals, var_values, cov
                else:
                    return objval, thetavals, var_values

            if calc_cov:
                
                return objval, thetavals, cov
            else:
                return objval, thetavals
        
        # Solve with sipopt and k_aug
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

    def _get_sample_list(self, samplesize, num_samples, replacement=True):
        
        samplelist = list()
        
        if num_samples is None:
            # This could get very large
            for i, l in enumerate(combinations(self._numbers_list, samplesize)):
                samplelist.append((i, np.sort(l)))
        else:
            for i in range(num_samples):
                attempts = 0
                unique_samples = 0 # check for duplicates in each sample
                duplicate = False # check for duplicates between samples
                while (unique_samples <= len(self.theta_names)) and (not duplicate):
                    sample = np.random.choice(self._numbers_list,
                                                samplesize,
                                                replace=replacement)
                    sample = np.sort(sample).tolist()
                    unique_samples = len(np.unique(sample))
                    if sample in samplelist:
                        duplicate = True
                    
                    attempts += 1
                    if attempts > num_samples: # arbitrary timeout limit
                        raise RuntimeError("""Internal error: timeout constructing 
                                           a sample, the dim of theta may be too 
                                           close to the samplesize""")
    
                samplelist.append((i, sample))
            
        return samplelist
    
    def theta_est(self, solver="ef_ipopt", return_values=[], bootlist=None, calc_cov=False): 
        """
        Parameter estimation using all scenarios in the data

        Parameters
        ----------
        solver: string, optional
            "ef_ipopt" or "k_aug". Default is "ef_ipopt".
        return_values: list, optional
            List of Variable names used to return values from the model
        bootlist: list, optional
            List of bootstrap sample numbers, used internally when calling theta_est_bootstrap
        calc_cov: boolean, optional
            If True, calculate and return the covariance matrix (only for "ef_ipopt" solver)
            
        Returns
        -------
        objectiveval: float
            The objective function value
        thetavals: dict
            A dictionary of all values for theta
        variable values: pd.DataFrame
            Variable values for each variable name in return_values (only for solver='ef_ipopt')
        Hessian: dict
            A dictionary of dictionaries for the Hessian (only for solver='k_aug')
        cov: pd.DataFrame
            Covariance matrix of the fitted parameters (only for solver='ef_ipopt')
        """
        assert isinstance(solver, str)
        assert isinstance(return_values, list)
        assert isinstance(bootlist, (type(None), list))
        
        return self._Q_opt(solver=solver, return_values=return_values,
                           bootlist=bootlist, calc_cov=calc_cov)
    
    
    def theta_est_bootstrap(self, bootstrap_samples, samplesize=None, 
                            replacement=True, seed=None, return_samples=False):
        """
        Parameter estimation using bootstrap resampling of the data

        Parameters
        ----------
        bootstrap_samples: int
            Number of bootstrap samples to draw from the data
        samplesize: int or None, optional
            Size of each bootstrap sample. If samplesize=None, samplesize will be 
            set to the number of samples in the data
        replacement: bool, optional
            Sample with or without replacement
        seed: int or None, optional
            Random seed
        return_samples: bool, optional
            Return a list of sample numbers used in each bootstrap estimation
        
        Returns
        -------
        bootstrap_theta: DataFrame 
            Theta values for each sample and (if return_samples = True) 
            the sample numbers used in each estimation
        """
        assert isinstance(bootstrap_samples, int)
        assert isinstance(samplesize, (type(None), int))
        assert isinstance(replacement, bool)
        assert isinstance(seed, (type(None), int))
        assert isinstance(return_samples, bool)
        
        if samplesize is None:
            samplesize = len(self._numbers_list)  
        
        if seed is not None:
            np.random.seed(seed)
        
        global_list = self._get_sample_list(samplesize, bootstrap_samples, 
                                            replacement)

        task_mgr = mpiu.ParallelTaskManager(bootstrap_samples)
        local_list = task_mgr.global_to_local_data(global_list)

        # Reset numbers_list
        self._numbers_list =  list(range(samplesize))
        
        bootstrap_theta = list()
        for idx, sample in local_list:
            objval, thetavals = self.theta_est(bootlist=list(sample))
            thetavals['samples'] = sample
            bootstrap_theta.append(thetavals)
            
        # Reset numbers_list (back to original)
        self._numbers_list =  list(range(len(self.callback_data)))
        
        global_bootstrap_theta = task_mgr.allgather_global_data(bootstrap_theta)
        bootstrap_theta = pd.DataFrame(global_bootstrap_theta)       

        if not return_samples:
            del bootstrap_theta['samples']
            
        return bootstrap_theta
    
    
    def theta_est_leaveNout(self, lNo, lNo_samples=None, seed=None, 
                            return_samples=False):
        """
        Parameter estimation where N data points are left out of each sample

        Parameters
        ----------
        lNo: int
            Number of data points to leave out for parameter estimation
        lNo_samples: int
            Number of leave-N-out samples. If lNo_samples=None, the maximum 
            number of combinations will be used
        seed: int or None, optional
            Random seed
        return_samples: bool, optional
            Return a list of sample numbers that were left out
        
        Returns
        -------
        lNo_theta: DataFrame 
            Theta values for each sample and (if return_samples = True) 
            the sample numbers left out of each estimation
        """
        assert isinstance(lNo, int)
        assert isinstance(lNo_samples, (type(None), int))
        assert isinstance(seed, (type(None), int))
        assert isinstance(return_samples, bool)
        
        samplesize = len(self._numbers_list)-lNo

        if seed is not None:
            np.random.seed(seed)
        
        global_list = self._get_sample_list(samplesize, lNo_samples, replacement=False)
            
        task_mgr = mpiu.ParallelTaskManager(len(global_list))
        local_list = task_mgr.global_to_local_data(global_list)
        
        # Reset numbers_list
        self._numbers_list =  list(range(samplesize))
        
        lNo_theta = list()
        for idx, sample in local_list:
            objval, thetavals = self.theta_est(bootlist=list(sample))
            lNo_s = list(set(range(len(self.callback_data))) - set(sample))
            thetavals['lNo'] = np.sort(lNo_s)
            lNo_theta.append(thetavals)
        
        # Reset numbers_list (back to original)
        self._numbers_list =  list(range(len(self.callback_data)))
        
        global_bootstrap_theta = task_mgr.allgather_global_data(lNo_theta)
        lNo_theta = pd.DataFrame(global_bootstrap_theta)   
        
        if not return_samples:
            del lNo_theta['lNo']
                    
        return lNo_theta
    
    
    def leaveNout_bootstrap_test(self, lNo, lNo_samples, bootstrap_samples, 
                                     distribution, alphas, seed=None):
        """
        Leave-N-out bootstrap test to compare theta values where N data points are 
        left out to a bootstrap analysis using the remaining data, 
        results indicate if theta is within a confidence region
        determined by the bootstrap analysis

        Parameters
        ----------
        lNo: int
            Number of data points to leave out for parameter estimation
        lNo_samples: int
            Leave-N-out sample size. If lNo_samples=None, the maximum number 
            of combinations will be used
        bootstrap_samples: int:
            Bootstrap sample size
        distribution: string
            Statistical distribution used to define a confidence region,  
            options = 'MVN' for multivariate_normal, 'KDE' for gaussian_kde, 
            and 'Rect' for rectangular.
        alphas: list
            List of alpha values used to determine if theta values are inside 
            or outside the region.
        seed: int or None, optional
            Random seed
            
        Returns
        ----------
        List of tuples with one entry per lNo_sample:
            
        * The first item in each tuple is the list of N samples that are left 
          out.
        * The second item in each tuple is a DataFrame of theta estimated using 
          the N samples.
        * The third item in each tuple is a DataFrame containing results from 
          the bootstrap analysis using the remaining samples.
        
        For each DataFrame a column is added for each value of alpha which 
        indicates if the theta estimate is in (True) or out (False) of the 
        alpha region for a given distribution (based on the bootstrap results)
        """
        assert isinstance(lNo, int)
        assert isinstance(lNo_samples, (type(None), int))
        assert isinstance(bootstrap_samples, int)
        assert distribution in ['Rect', 'MVN', 'KDE']
        assert isinstance(alphas, list)
        assert isinstance(seed, (type(None), int))
        
        if seed is not None:
            np.random.seed(seed)
            
        data = self.callback_data.copy()
        
        global_list = self._get_sample_list(lNo, lNo_samples, replacement=False)
            
        results = []
        for idx, sample in global_list:
            
            # Reset callback_data and numbers_list
            self.callback_data = data.loc[sample,:] 
            self._numbers_list = self.callback_data.index
            obj, theta = self.theta_est()
            
            # Reset callback_data and numbers_list
            self.callback_data = data.drop(index=sample)
            self._numbers_list = self.callback_data.index
            bootstrap_theta = self.theta_est_bootstrap(bootstrap_samples)
            
            training, test = self.confidence_region_test(bootstrap_theta, 
                                    distribution=distribution, alphas=alphas, 
                                    test_theta_values=theta)
                
            results.append((sample, test, training))
        
        # Reset callback_data and numbers_list (back to original)
        self.callback_data = data
        self._numbers_list = self.callback_data.index
        
        return results
    
    
    def objective_at_theta(self, theta_values):
        """
        Objective value for each theta

        Parameters
        ----------
        theta_values: DataFrame, columns=theta_names
            Values of theta used to compute the objective
            
        Returns
        -------
        obj_at_theta: DataFrame
            Objective value for each theta (infeasible solutions are 
            omitted).
        """
        assert isinstance(theta_values, pd.DataFrame)
        
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
    
    
    def likelihood_ratio_test(self, obj_at_theta, obj_value, alphas, 
                              return_thresholds=False):
        """
        Likelihood ratio test to identify theta values within a confidence 
        region using the :math:`\chi^2` distribution
        
        Parameters
        ----------
        obj_at_theta: DataFrame, columns = theta_names + 'obj'
            Objective values for each theta value (returned by 
            objective_at_theta)
        obj_value: int or float
            Objective value from parameter estimation using all data
        alphas: list
            List of alpha values to use in the chi2 test
        return_thresholds: bool, optional
            Return the threshold value for each alpha
            
        Returns
        -------
        LR: DataFrame 
            Objective values for each theta value along with True or False for 
            each alpha
        thresholds: dictionary
            If return_threshold = True, the thresholds are also returned.
        """
        assert isinstance(obj_at_theta, pd.DataFrame)
        assert isinstance(obj_value, (int, float))
        assert isinstance(alphas, list)
        assert isinstance(return_thresholds, bool)
            
        LR = obj_at_theta.copy()
        S = len(self.callback_data)
        thresholds = {}
        for a in alphas:
            chi2_val = scipy.stats.chi2.ppf(a, 2)
            thresholds[a] = obj_value * ((chi2_val / (S - 2)) + 1)
            LR[a] = LR['obj'] < thresholds[a]
        
        if return_thresholds:
            return LR, thresholds
        else:
            return LR

    def confidence_region_test(self, theta_values, distribution, alphas, 
                               test_theta_values=None):
        """
        Confidence region test to determine if theta values are within a 
        rectangular, multivariate normal, or Gaussian kernel density distribution 
        for a range of alpha values
        
        Parameters
        ----------
        theta_values: DataFrame, columns = theta_names
            Theta values used to generate a confidence region 
            (generally returned by theta_est_bootstrap)
        distribution: string
            Statistical distribution used to define a confidence region,  
            options = 'MVN' for multivariate_normal, 'KDE' for gaussian_kde, 
            and 'Rect' for rectangular.
        alphas: list
            List of alpha values used to determine if theta values are inside 
            or outside the region.
        test_theta_values: dictionary or DataFrame, keys/columns = theta_names, optional
            Additional theta values that are compared to the confidence region
            to determine if they are inside or outside.
        
        Returns
        -------
        training_results: DataFrame 
            Theta value used to generate the confidence region along with True 
            (inside) or False (outside) for each alpha
        test_results: DataFrame 
            If test_theta_values is not None, returns test theta value along 
            with True (inside) or False (outside) for each alpha
        """
        assert isinstance(theta_values, pd.DataFrame)
        assert distribution in ['Rect', 'MVN', 'KDE']
        assert isinstance(alphas, list)
        assert isinstance(test_theta_values, (type(None), dict, pd.DataFrame))
        
        if isinstance(test_theta_values, dict):
            test_theta_values = pd.Series(test_theta_values).to_frame().transpose()
            
        training_results = theta_values.copy()
        
        if test_theta_values is not None:
            test_result = test_theta_values.copy()
        
        for a in alphas:
            
            if distribution == 'Rect':
                lb, ub = graphics.fit_rect_dist(theta_values, a)
                training_results[a] = ((theta_values > lb).all(axis=1) & \
                                  (theta_values < ub).all(axis=1))
                
                if test_theta_values is not None:
                    # use upper and lower bound from the training set
                    test_result[a] = ((test_theta_values > lb).all(axis=1) & \
                                  (test_theta_values < ub).all(axis=1))
                    
            elif distribution == 'MVN':
                dist = graphics.fit_mvn_dist(theta_values)
                Z = dist.pdf(theta_values)
                score = scipy.stats.scoreatpercentile(Z, (1-a)*100) 
                training_results[a] = (Z >= score)
                
                if test_theta_values is not None:
                    # use score from the training set
                    Z = dist.pdf(test_theta_values)
                    test_result[a] = (Z >= score) 
                
            elif distribution == 'KDE':
                dist = graphics.fit_kde_dist(theta_values)
                Z = dist.pdf(theta_values.transpose())
                score = scipy.stats.scoreatpercentile(Z, (1-a)*100) 
                training_results[a] = (Z >= score)
                
                if test_theta_values is not None:
                    # use score from the training set
                    Z = dist.pdf(test_theta_values.transpose())
                    test_result[a] = (Z >= score) 
                    
        if test_theta_values is not None:
            return training_results, test_result
        else:
            return training_results
