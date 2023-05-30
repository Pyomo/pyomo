#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#### Using mpi-sppy instead of PySP; May 2020
#### Adding option for "local" EF starting Sept 2020
#### Wrapping mpi-sppy functionality and local option Jan 2021, Feb 2021

# TODO: move use_mpisppy to a Pyomo configuration option
#
# False implies always use the EF that is local to parmest
use_mpisppy = True  # Use it if we can but use local if not.
if use_mpisppy:
    try:
        # MPI-SPPY has an unfortunate side effect of outputting
        # "[ 0.00] Initializing mpi-sppy" when it is imported.  This can
        # cause things like doctests to fail.  We will suppress that
        # information here.
        from pyomo.common.tee import capture_output

        with capture_output():
            import mpisppy.utils.sputils as sputils
    except ImportError:
        use_mpisppy = False  # we can't use it
if use_mpisppy:
    # These things should be outside the try block.
    sputils.disable_tictoc_output()
    import mpisppy.opt.ef as st
    import mpisppy.scenario_tree as scenario_tree
else:
    import pyomo.contrib.parmest.utils.create_ef as local_ef
    import pyomo.contrib.parmest.utils.scenario_tree as scenario_tree

import re
import importlib as im
import logging
import types
import json
from itertools import combinations

from pyomo.common.dependencies import (
    attempt_import,
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy,
    scipy_available,
)

import pyomo.environ as pyo

from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID

import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet

parmest_available = numpy_available & pandas_available & scipy_available

inverse_reduced_hessian, inverse_reduced_hessian_available = attempt_import(
    'pyomo.contrib.interior_point.inverse_reduced_hessian'
)

logger = logging.getLogger(__name__)


def ef_nonants(ef):
    # Wrapper to call someone's ef_nonants
    # (the function being called is very short, but it might be changed)
    if use_mpisppy:
        return sputils.ef_nonants(ef)
    else:
        return local_ef.ef_nonants(ef)


def _experiment_instance_creation_callback(
    scenario_name, node_names=None, cb_data=None
):
    """
    This is going to be called by mpi-sppy or the local EF and it will call into
    the user's model's callback.

    Parameters:
    -----------
    scenario_name: `str` Scenario name should end with a number
    node_names: `None` ( Not used here )
    cb_data : dict with ["callback"], ["BootList"],
              ["theta_names"], ["cb_data"], etc.
              "cb_data" is passed through to user's callback function
                        that is the "callback" value.
              "BootList" is None or bootstrap experiment number list.
                       (called cb_data by mpisppy)


    Returns:
    --------
    instance: `ConcreteModel`
        instantiated scenario

    Note:
    ----
    There is flexibility both in how the function is passed and its signature.
    """
    assert cb_data is not None
    outer_cb_data = cb_data
    scen_num_str = re.compile(r'(\d+)$').search(scenario_name).group(1)
    scen_num = int(scen_num_str)
    basename = scenario_name[: -len(scen_num_str)]  # to reconstruct name

    CallbackFunction = outer_cb_data["callback"]

    if callable(CallbackFunction):
        callback = CallbackFunction
    else:
        cb_name = CallbackFunction

        if "CallbackModule" not in outer_cb_data:
            raise RuntimeError(
                "Internal Error: need CallbackModule in parmest callback"
            )
        else:
            modname = outer_cb_data["CallbackModule"]

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
            print("Error getting function=" + cb_name + " from module=" + str(modname))
            raise

    if "BootList" in outer_cb_data:
        bootlist = outer_cb_data["BootList"]
        # print("debug in callback: using bootlist=",str(bootlist))
        # assuming bootlist itself is zero based
        exp_num = bootlist[scen_num]
    else:
        exp_num = scen_num

    scen_name = basename + str(exp_num)

    cb_data = outer_cb_data["cb_data"]  # cb_data might be None.

    # at least three signatures are supported. The first is preferred
    try:
        instance = callback(experiment_number=exp_num, cb_data=cb_data)
    except TypeError:
        raise RuntimeError(
            "Only one callback signature is supported: "
            "callback(experiment_number, cb_data) "
        )
        """
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
        """
    if hasattr(instance, "_mpisppy_node_list"):
        raise RuntimeError(f"scenario for experiment {exp_num} has _mpisppy_node_list")
    nonant_list = [
        instance.find_component(vstr) for vstr in outer_cb_data["theta_names"]
    ]
    if use_mpisppy:
        instance._mpisppy_node_list = [
            scenario_tree.ScenarioNode(
                name="ROOT",
                cond_prob=1.0,
                stage=1,
                cost_expression=instance.FirstStageCost,
                nonant_list=nonant_list,
                scen_model=instance,
            )
        ]
    else:
        instance._mpisppy_node_list = [
            scenario_tree.ScenarioNode(
                name="ROOT",
                cond_prob=1.0,
                stage=1,
                cost_expression=instance.FirstStageCost,
                scen_name_list=None,
                nonant_list=nonant_list,
                scen_model=instance,
            )
        ]

    if "ThetaVals" in outer_cb_data:
        thetavals = outer_cb_data["ThetaVals"]

        # dlw august 2018: see mea code for more general theta
        for vstr in thetavals:
            theta_cuid = ComponentUID(vstr)
            theta_object = theta_cuid.find_component_on(instance)
            if thetavals[vstr] is not None:
                # print("Fixing",vstr,"at",str(thetavals[vstr]))
                theta_object.fix(thetavals[vstr])
            else:
                # print("Freeing",vstr)
                theta_object.unfix()

    return instance


# =============================================
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
    m = scenario_tree.tree_structure_model.CreateAbstractScenarioTreeModel()
    m = m.create_instance()
    m.Stages.add('Stage1')
    m.Stages.add('Stage2')
    m.Nodes.add('RootNode')
    for i in scenlist:
        m.Nodes.add('LeafNode_Experiment' + str(i))
        m.Scenarios.add('Experiment' + str(i))
    m.NodeStage['RootNode'] = 'Stage1'
    m.ConditionalProbability['RootNode'] = 1.0
    for node in m.Nodes:
        if node != 'RootNode':
            m.NodeStage[node] = 'Stage2'
            m.Children['RootNode'].add(node)
            m.Children[node].clear()
            m.ConditionalProbability[node] = 1.0 / num_scenarios
            m.ScenarioLeafNode[node.replace('LeafNode_', '')] = node

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
    if use_mean is None:
        use_mean_list = []
    else:
        use_mean_list = use_mean

    grouped_data = []
    for exp_num, group in data.groupby(data[groupby_column_name]):
        d = {}
        for col in group.columns:
            if col in use_mean_list:
                d[col] = group[col].mean()
            else:
                d[col] = list(group[col])
        grouped_data.append(d)

    return grouped_data


class _SecondStageCostExpr(object):
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
    data: pd.DataFrame, list of dictionaries, list of dataframes, or list of json file names
        Data that is used to build an instance of the Pyomo model and build
        the objective function
    theta_names: list of strings
        List of Var names to estimate
    obj_function: function, optional
        Function used to formulate parameter estimation objective, generally
        sum of squared error between measurements and model variables.
        If no function is specified, the model is used
        "as is" and should be defined with a "FirstStageCost" and
        "SecondStageCost" expression that are used to build an objective.
    tee: bool, optional
        Indicates that ef solver output should be teed
    diagnostic_mode: bool, optional
        If True, print diagnostics from the solver
    solver_options: dict, optional
        Provides options to the solver (also the name of an attribute)
    """

    def __init__(
        self,
        model_function,
        data,
        theta_names,
        obj_function=None,
        tee=False,
        diagnostic_mode=False,
        solver_options=None,
    ):
        self.model_function = model_function

        assert isinstance(
            data, (list, pd.DataFrame)
        ), "Data must be a list or DataFrame"
        # convert dataframe into a list of dataframes, each row = one scenario
        if isinstance(data, pd.DataFrame):
            self.callback_data = [
                data.loc[i, :].to_frame().transpose() for i in data.index
            ]
        else:
            self.callback_data = data
        assert isinstance(
            self.callback_data[0], (dict, pd.DataFrame, str)
        ), "The scenarios in data must be a dictionary, DataFrame or filename"

        if len(theta_names) == 0:
            self.theta_names = ['parmest_dummy_var']
        else:
            self.theta_names = theta_names

        self.obj_function = obj_function
        self.tee = tee
        self.diagnostic_mode = diagnostic_mode
        self.solver_options = solver_options

        self._second_stage_cost_exp = "SecondStageCost"
        # boolean to indicate if model is initialized using a square solve
        self.model_initialized = False

    def _return_theta_names(self):
        """
        Return list of fitted model parameter names
        """
        # if fitted model parameter names differ from theta_names created when Estimator object is created
        if hasattr(self, 'theta_names_updated'):
            return self.theta_names_updated

        else:
            return (
                self.theta_names
            )  # default theta_names, created when Estimator object is created

    def _create_parmest_model(self, data):
        """
        Modify the Pyomo model for parameter estimation
        """
        model = self.model_function(data)

        if (len(self.theta_names) == 1) and (
            self.theta_names[0] == 'parmest_dummy_var'
        ):
            model.parmest_dummy_var = pyo.Var(initialize=1.0)

        # Add objective function (optional)
        if self.obj_function:
            for obj in model.component_objects(pyo.Objective):
                if obj.name in ["Total_Cost_Objective"]:
                    raise RuntimeError(
                        "Parmest will not override the existing model Objective named "
                        + obj.name
                    )
                obj.deactivate()

            for expr in model.component_data_objects(pyo.Expression):
                if expr.name in ["FirstStageCost", "SecondStageCost"]:
                    raise RuntimeError(
                        "Parmest will not override the existing model Expression named "
                        + expr.name
                    )
            model.FirstStageCost = pyo.Expression(expr=0)
            model.SecondStageCost = pyo.Expression(
                rule=_SecondStageCostExpr(self.obj_function, data)
            )

            def TotalCost_rule(model):
                return model.FirstStageCost + model.SecondStageCost

            model.Total_Cost_Objective = pyo.Objective(
                rule=TotalCost_rule, sense=pyo.minimize
            )

        # Convert theta Params to Vars, and unfix theta Vars
        model = utils.convert_params_to_vars(model, self.theta_names)

        # Update theta names list to use CUID string representation
        for i, theta in enumerate(self.theta_names):
            var_cuid = ComponentUID(theta)
            var_validate = var_cuid.find_component_on(model)
            if var_validate is None:
                logger.warning(
                    "theta_name[%s] (%s) was not found on the model", (i, theta)
                )
            else:
                try:
                    # If the component is not a variable,
                    # this will generate an exception (and the warning
                    # in the 'except')
                    var_validate.unfix()
                    self.theta_names[i] = repr(var_cuid)
                except:
                    logger.warning(theta + ' is not a variable')

        self.parmest_model = model

        return model

    def _instance_creation_callback(self, experiment_number=None, cb_data=None):
        # cb_data is a list of dictionaries, list of dataframes, OR list of json file names
        exp_data = cb_data[experiment_number]
        if isinstance(exp_data, (dict, pd.DataFrame)):
            pass
        elif isinstance(exp_data, str):
            try:
                with open(exp_data, 'r') as infile:
                    exp_data = json.load(infile)
            except:
                raise RuntimeError(f'Could not read {exp_data} as json')
        else:
            raise RuntimeError(f'Unexpected data format for cb_data={cb_data}')
        model = self._create_parmest_model(exp_data)

        return model

    def _Q_opt(
        self,
        ThetaVals=None,
        solver="ef_ipopt",
        return_values=[],
        bootlist=None,
        calc_cov=False,
        cov_n=None,
    ):
        """
        Set up all thetas as first stage Vars, return resulting theta
        values as well as the objective function value.

        """
        if solver == "k_aug":
            raise RuntimeError("k_aug no longer supported.")

        # (Bootstrap scenarios will use indirection through the bootlist)
        if bootlist is None:
            scenario_numbers = list(range(len(self.callback_data)))
            scen_names = ["Scenario{}".format(i) for i in scenario_numbers]
        else:
            scen_names = ["Scenario{}".format(i) for i in range(len(bootlist))]

        # tree_model.CallbackModule = None
        outer_cb_data = dict()
        outer_cb_data["callback"] = self._instance_creation_callback
        if ThetaVals is not None:
            outer_cb_data["ThetaVals"] = ThetaVals
        if bootlist is not None:
            outer_cb_data["BootList"] = bootlist
        outer_cb_data["cb_data"] = self.callback_data  # None is OK
        outer_cb_data["theta_names"] = self.theta_names

        options = {"solver": "ipopt"}
        scenario_creator_options = {"cb_data": outer_cb_data}
        if use_mpisppy:
            ef = sputils.create_EF(
                scen_names,
                _experiment_instance_creation_callback,
                EF_name="_Q_opt",
                suppress_warnings=True,
                scenario_creator_kwargs=scenario_creator_options,
            )
        else:
            ef = local_ef.create_EF(
                scen_names,
                _experiment_instance_creation_callback,
                EF_name="_Q_opt",
                suppress_warnings=True,
                scenario_creator_kwargs=scenario_creator_options,
            )
        self.ef_instance = ef

        # Solve the extensive form with ipopt
        if solver == "ef_ipopt":
            if not calc_cov:
                # Do not calculate the reduced hessian

                solver = SolverFactory('ipopt')
                if self.solver_options is not None:
                    for key in self.solver_options:
                        solver.options[key] = self.solver_options[key]

                solve_result = solver.solve(self.ef_instance, tee=self.tee)

            # The import error will be raised when we attempt to use
            # inv_reduced_hessian_barrier below.
            #
            # elif not asl_available:
            #    raise ImportError("parmest requires ASL to calculate the "
            #                      "covariance matrix with solver 'ipopt'")
            else:
                # parmest makes the fitted parameters stage 1 variables
                ind_vars = []
                for ndname, Var, solval in ef_nonants(ef):
                    ind_vars.append(Var)
                # calculate the reduced hessian
                (
                    solve_result,
                    inv_red_hes,
                ) = inverse_reduced_hessian.inv_reduced_hessian_barrier(
                    self.ef_instance,
                    independent_variables=ind_vars,
                    solver_options=self.solver_options,
                    tee=self.tee,
                )

            if self.diagnostic_mode:
                print(
                    '    Solver termination condition = ',
                    str(solve_result.solver.termination_condition),
                )

            # assume all first stage are thetas...
            thetavals = {}
            for ndname, Var, solval in ef_nonants(ef):
                # process the name
                # the scenarios are blocks, so strip the scenario name
                vname = Var.name[Var.name.find(".") + 1 :]
                thetavals[vname] = solval

            objval = pyo.value(ef.EF_Obj)

            if calc_cov:
                # Calculate the covariance matrix

                # Number of data points considered
                n = cov_n

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
                the constant cancels out. (was scaled by 1/n because it computes an
                expected value.)
                '''
                cov = 2 * sse / (n - l) * inv_red_hes
                cov = pd.DataFrame(
                    cov, index=thetavals.keys(), columns=thetavals.keys()
                )

            thetavals = pd.Series(thetavals)

            if len(return_values) > 0:
                var_values = []
                if len(scen_names) > 1:  # multiple scenarios
                    block_objects = self.ef_instance.component_objects(
                        Block, descend_into=False
                    )
                else:  # single scenario
                    block_objects = [self.ef_instance]
                for exp_i in block_objects:
                    vals = {}
                    for var in return_values:
                        exp_i_var = exp_i.find_component(str(var))
                        if (
                            exp_i_var is None
                        ):  # we might have a block such as _mpisppy_data
                            continue
                        # if value to return is ContinuousSet
                        if type(exp_i_var) == ContinuousSet:
                            temp = list(exp_i_var)
                        else:
                            temp = [pyo.value(_) for _ in exp_i_var.values()]
                        if len(temp) == 1:
                            vals[var] = temp[0]
                        else:
                            vals[var] = temp
                    if len(vals) > 0:
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

        else:
            raise RuntimeError("Unknown solver in Q_Opt=" + solver)

    def _Q_at_theta(self, thetavals, initialize_parmest_model=False):
        """
        Return the objective function value with fixed theta values.

        Parameters
        ----------
        thetavals: dict
            A dictionary of theta values.

        initialize_parmest_model: boolean
            If True: Solve square problem instance, build extensive form of the model for
            parameter estimation, and set flag model_initialized to True

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

        if len(thetavals) > 0:
            dummy_cb = {
                "callback": self._instance_creation_callback,
                "ThetaVals": thetavals,
                "theta_names": self._return_theta_names(),
                "cb_data": self.callback_data,
            }
        else:
            dummy_cb = {
                "callback": self._instance_creation_callback,
                "theta_names": self._return_theta_names(),
                "cb_data": self.callback_data,
            }

        if self.diagnostic_mode:
            if len(thetavals) > 0:
                print('    Compute objective at theta = ', str(thetavals))
            else:
                print('    Compute objective at initial theta')

        # start block of code to deal with models with no constraints
        # (ipopt will crash or complain on such problems without special care)
        instance = _experiment_instance_creation_callback("FOO0", None, dummy_cb)
        try:  # deal with special problems so Ipopt will not crash
            first = next(instance.component_objects(pyo.Constraint, active=True))
            active_constraints = True
        except:
            active_constraints = False
        # end block of code to deal with models with no constraints

        WorstStatus = pyo.TerminationCondition.optimal
        totobj = 0
        scenario_numbers = list(range(len(self.callback_data)))
        if initialize_parmest_model:
            # create dictionary to store pyomo model instances (scenarios)
            scen_dict = dict()

        for snum in scenario_numbers:
            sname = "scenario_NODE" + str(snum)
            instance = _experiment_instance_creation_callback(sname, None, dummy_cb)

            if initialize_parmest_model:
                # list to store fitted parameter names that will be unfixed
                # after initialization
                theta_init_vals = []
                # use appropriate theta_names member
                theta_ref = self._return_theta_names()

                for i, theta in enumerate(theta_ref):
                    # Use parser in ComponentUID to locate the component
                    var_cuid = ComponentUID(theta)
                    var_validate = var_cuid.find_component_on(instance)
                    if var_validate is None:
                        logger.warning(
                            "theta_name %s was not found on the model", (theta)
                        )
                    else:
                        try:
                            if len(thetavals) == 0:
                                var_validate.fix()
                            else:
                                var_validate.fix(thetavals[theta])
                            theta_init_vals.append(var_validate)
                        except:
                            logger.warning(
                                'Unable to fix model parameter value for %s (not a Pyomo model Var)',
                                (theta),
                            )

            if active_constraints:
                if self.diagnostic_mode:
                    print('      Experiment = ', snum)
                    print('     First solve with special diagnostics wrapper')
                    (
                        status_obj,
                        solved,
                        iters,
                        time,
                        regu,
                    ) = utils.ipopt_solve_with_stats(
                        instance, optimizer, max_iter=500, max_cpu_time=120
                    )
                    print(
                        "   status_obj, solved, iters, time, regularization_stat = ",
                        str(status_obj),
                        str(solved),
                        str(iters),
                        str(time),
                        str(regu),
                    )

                results = optimizer.solve(instance)
                if self.diagnostic_mode:
                    print(
                        'standard solve solver termination condition=',
                        str(results.solver.termination_condition),
                    )

                if (
                    results.solver.termination_condition
                    != pyo.TerminationCondition.optimal
                ):
                    # DLW: Aug2018: not distinguishing "middlish" conditions
                    if WorstStatus != pyo.TerminationCondition.infeasible:
                        WorstStatus = results.solver.termination_condition
                    if initialize_parmest_model:
                        if self.diagnostic_mode:
                            print(
                                "Scenario {:d} infeasible with initialized parameter values".format(
                                    snum
                                )
                            )
                else:
                    if initialize_parmest_model:
                        if self.diagnostic_mode:
                            print(
                                "Scenario {:d} initialization successful with initial parameter values".format(
                                    snum
                                )
                            )
                if initialize_parmest_model:
                    # unfix parameters after initialization
                    for theta in theta_init_vals:
                        theta.unfix()
                    scen_dict[sname] = instance
            else:
                if initialize_parmest_model:
                    # unfix parameters after initialization
                    for theta in theta_init_vals:
                        theta.unfix()
                    scen_dict[sname] = instance

            objobject = getattr(instance, self._second_stage_cost_exp)
            objval = pyo.value(objobject)
            totobj += objval

        retval = totobj / len(scenario_numbers)  # -1??
        if initialize_parmest_model and not hasattr(self, 'ef_instance'):
            # create extensive form of the model using scenario dictionary
            if len(scen_dict) > 0:
                for scen in scen_dict.values():
                    scen._mpisppy_probability = 1 / len(scen_dict)

            if use_mpisppy:
                EF_instance = sputils._create_EF_from_scen_dict(
                    scen_dict,
                    EF_name="_Q_at_theta",
                    # suppress_warnings=True
                )
            else:
                EF_instance = local_ef._create_EF_from_scen_dict(
                    scen_dict, EF_name="_Q_at_theta", nonant_for_fixed_vars=True
                )

            self.ef_instance = EF_instance
            # set self.model_initialized flag to True to skip extensive form model
            # creation using theta_est()
            self.model_initialized = True

            # return initialized theta values
            if len(thetavals) == 0:
                # use appropriate theta_names member
                theta_ref = self._return_theta_names()
                for i, theta in enumerate(theta_ref):
                    thetavals[theta] = theta_init_vals[i]()

        return retval, thetavals, WorstStatus

    def _get_sample_list(self, samplesize, num_samples, replacement=True):
        samplelist = list()

        scenario_numbers = list(range(len(self.callback_data)))

        if num_samples is None:
            # This could get very large
            for i, l in enumerate(combinations(scenario_numbers, samplesize)):
                samplelist.append((i, np.sort(l)))
        else:
            for i in range(num_samples):
                attempts = 0
                unique_samples = 0  # check for duplicates in each sample
                duplicate = False  # check for duplicates between samples
                while (unique_samples <= len(self._return_theta_names())) and (
                    not duplicate
                ):
                    sample = np.random.choice(
                        scenario_numbers, samplesize, replace=replacement
                    )
                    sample = np.sort(sample).tolist()
                    unique_samples = len(np.unique(sample))
                    if sample in samplelist:
                        duplicate = True

                    attempts += 1
                    if attempts > num_samples:  # arbitrary timeout limit
                        raise RuntimeError(
                            """Internal error: timeout constructing
                                           a sample, the dim of theta may be too
                                           close to the samplesize"""
                        )

                samplelist.append((i, sample))

        return samplelist

    def theta_est(
        self, solver="ef_ipopt", return_values=[], calc_cov=False, cov_n=None
    ):
        """
        Parameter estimation using all scenarios in the data

        Parameters
        ----------
        solver: string, optional
            Currently only "ef_ipopt" is supported. Default is "ef_ipopt".
        return_values: list, optional
            List of Variable names, used to return values from the model for data reconciliation
        calc_cov: boolean, optional
            If True, calculate and return the covariance matrix (only for "ef_ipopt" solver)
        cov_n: int, optional
            If calc_cov=True, then the user needs to supply the number of datapoints
            that are used in the objective function

        Returns
        -------
        objectiveval: float
            The objective function value
        thetavals: pd.Series
            Estimated values for theta
        variable values: pd.DataFrame
            Variable values for each variable name in return_values (only for solver='ef_ipopt')
        cov: pd.DataFrame
            Covariance matrix of the fitted parameters (only for solver='ef_ipopt')
        """
        assert isinstance(solver, str)
        assert isinstance(return_values, list)
        assert isinstance(calc_cov, bool)
        if calc_cov:
            assert isinstance(
                cov_n, int
            ), "The number of datapoints that are used in the objective function is required to calculate the covariance matrix"
            assert cov_n > len(
                self._return_theta_names()
            ), "The number of datapoints must be greater than the number of parameters to estimate"

        return self._Q_opt(
            solver=solver,
            return_values=return_values,
            bootlist=None,
            calc_cov=calc_cov,
            cov_n=cov_n,
        )

    def theta_est_bootstrap(
        self,
        bootstrap_samples,
        samplesize=None,
        replacement=True,
        seed=None,
        return_samples=False,
    ):
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
        bootstrap_theta: pd.DataFrame
            Theta values for each sample and (if return_samples = True)
            the sample numbers used in each estimation
        """
        assert isinstance(bootstrap_samples, int)
        assert isinstance(samplesize, (type(None), int))
        assert isinstance(replacement, bool)
        assert isinstance(seed, (type(None), int))
        assert isinstance(return_samples, bool)

        if samplesize is None:
            samplesize = len(self.callback_data)

        if seed is not None:
            np.random.seed(seed)

        global_list = self._get_sample_list(samplesize, bootstrap_samples, replacement)

        task_mgr = utils.ParallelTaskManager(bootstrap_samples)
        local_list = task_mgr.global_to_local_data(global_list)

        bootstrap_theta = list()
        for idx, sample in local_list:
            objval, thetavals = self._Q_opt(bootlist=list(sample))
            thetavals['samples'] = sample
            bootstrap_theta.append(thetavals)

        global_bootstrap_theta = task_mgr.allgather_global_data(bootstrap_theta)
        bootstrap_theta = pd.DataFrame(global_bootstrap_theta)

        if not return_samples:
            del bootstrap_theta['samples']

        return bootstrap_theta

    def theta_est_leaveNout(
        self, lNo, lNo_samples=None, seed=None, return_samples=False
    ):
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
        lNo_theta: pd.DataFrame
            Theta values for each sample and (if return_samples = True)
            the sample numbers left out of each estimation
        """
        assert isinstance(lNo, int)
        assert isinstance(lNo_samples, (type(None), int))
        assert isinstance(seed, (type(None), int))
        assert isinstance(return_samples, bool)

        samplesize = len(self.callback_data) - lNo

        if seed is not None:
            np.random.seed(seed)

        global_list = self._get_sample_list(samplesize, lNo_samples, replacement=False)

        task_mgr = utils.ParallelTaskManager(len(global_list))
        local_list = task_mgr.global_to_local_data(global_list)

        lNo_theta = list()
        for idx, sample in local_list:
            objval, thetavals = self._Q_opt(bootlist=list(sample))
            lNo_s = list(set(range(len(self.callback_data))) - set(sample))
            thetavals['lNo'] = np.sort(lNo_s)
            lNo_theta.append(thetavals)

        global_bootstrap_theta = task_mgr.allgather_global_data(lNo_theta)
        lNo_theta = pd.DataFrame(global_bootstrap_theta)

        if not return_samples:
            del lNo_theta['lNo']

        return lNo_theta

    def leaveNout_bootstrap_test(
        self, lNo, lNo_samples, bootstrap_samples, distribution, alphas, seed=None
    ):
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
            # Reset callback_data to only include the sample
            self.callback_data = [data[i] for i in sample]

            obj, theta = self.theta_est()

            # Reset callback_data to include all scenarios except the sample
            self.callback_data = [data[i] for i in range(len(data)) if i not in sample]

            bootstrap_theta = self.theta_est_bootstrap(bootstrap_samples)

            training, test = self.confidence_region_test(
                bootstrap_theta,
                distribution=distribution,
                alphas=alphas,
                test_theta_values=theta,
            )

            results.append((sample, test, training))

        # Reset callback_data (back to full data set)
        self.callback_data = data

        return results

    def objective_at_theta(self, theta_values=None, initialize_parmest_model=False):
        """
        Objective value for each theta

        Parameters
        ----------
        theta_values: pd.DataFrame, columns=theta_names
            Values of theta used to compute the objective

        initialize_parmest_model: boolean
            If True: Solve square problem instance, build extensive form of the model for
            parameter estimation, and set flag model_initialized to True


        Returns
        -------
        obj_at_theta: pd.DataFrame
            Objective value for each theta (infeasible solutions are
            omitted).
        """
        if len(self.theta_names) == 1 and self.theta_names[0] == 'parmest_dummy_var':
            pass  # skip assertion if model has no fitted parameters
        else:
            # create a local instance of the pyomo model to access model variables and parameters
            model_temp = self._create_parmest_model(self.callback_data[0])
            model_theta_list = []  # list to store indexed and non-indexed parameters
            # iterate over original theta_names
            for theta_i in self.theta_names:
                var_cuid = ComponentUID(theta_i)
                var_validate = var_cuid.find_component_on(model_temp)
                # check if theta in theta_names are indexed
                try:
                    # get component UID of Set over which theta is defined
                    set_cuid = ComponentUID(var_validate.index_set())
                    # access and iterate over the Set to generate theta names as they appear
                    # in the pyomo model
                    set_validate = set_cuid.find_component_on(model_temp)
                    for s in set_validate:
                        self_theta_temp = repr(var_cuid) + "[" + repr(s) + "]"
                        # generate list of theta names
                        model_theta_list.append(self_theta_temp)
                # if theta is not indexed, copy theta name to list as-is
                except AttributeError:
                    self_theta_temp = repr(var_cuid)
                    model_theta_list.append(self_theta_temp)
                except:
                    raise
            # if self.theta_names is not the same as temp model_theta_list,
            # create self.theta_names_updated
            if set(self.theta_names) == set(model_theta_list) and len(
                self.theta_names
            ) == set(model_theta_list):
                pass
            else:
                self.theta_names_updated = model_theta_list

        if theta_values is None:
            all_thetas = {}  # dictionary to store fitted variables
            # use appropriate theta names member
            theta_names = self._return_theta_names()
        else:
            assert isinstance(theta_values, pd.DataFrame)
            # for parallel code we need to use lists and dicts in the loop
            theta_names = theta_values.columns
            # # check if theta_names are in model
            for theta in list(theta_names):
                theta_temp = theta.replace("'", "")  # cleaning quotes from theta_names

                assert theta_temp in [
                    t.replace("'", "") for t in model_theta_list
                ], "Theta name {} in 'theta_values' not in 'theta_names' {}".format(
                    theta_temp, model_theta_list
                )
            assert len(list(theta_names)) == len(model_theta_list)

            all_thetas = theta_values.to_dict('records')

        if all_thetas:
            task_mgr = utils.ParallelTaskManager(len(all_thetas))
            local_thetas = task_mgr.global_to_local_data(all_thetas)
        else:
            if initialize_parmest_model:
                task_mgr = utils.ParallelTaskManager(
                    1
                )  # initialization performed using just 1 set of theta values
        # walk over the mesh, return objective function
        all_obj = list()
        if len(all_thetas) > 0:
            for Theta in local_thetas:
                obj, thetvals, worststatus = self._Q_at_theta(
                    Theta, initialize_parmest_model=initialize_parmest_model
                )
                if worststatus != pyo.TerminationCondition.infeasible:
                    all_obj.append(list(Theta.values()) + [obj])
                # DLW, Aug2018: should we also store the worst solver status?
        else:
            obj, thetvals, worststatus = self._Q_at_theta(
                thetavals={}, initialize_parmest_model=initialize_parmest_model
            )
            if worststatus != pyo.TerminationCondition.infeasible:
                all_obj.append(list(thetvals.values()) + [obj])

        global_all_obj = task_mgr.allgather_global_data(all_obj)
        dfcols = list(theta_names) + ['obj']
        obj_at_theta = pd.DataFrame(data=global_all_obj, columns=dfcols)
        return obj_at_theta

    def likelihood_ratio_test(
        self, obj_at_theta, obj_value, alphas, return_thresholds=False
    ):
        r"""
        Likelihood ratio test to identify theta values within a confidence
        region using the :math:`\chi^2` distribution

        Parameters
        ----------
        obj_at_theta: pd.DataFrame, columns = theta_names + 'obj'
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
        LR: pd.DataFrame
            Objective values for each theta value along with True or False for
            each alpha
        thresholds: pd.Series
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

        thresholds = pd.Series(thresholds)

        if return_thresholds:
            return LR, thresholds
        else:
            return LR

    def confidence_region_test(
        self, theta_values, distribution, alphas, test_theta_values=None
    ):
        """
        Confidence region test to determine if theta values are within a
        rectangular, multivariate normal, or Gaussian kernel density distribution
        for a range of alpha values

        Parameters
        ----------
        theta_values: pd.DataFrame, columns = theta_names
            Theta values used to generate a confidence region
            (generally returned by theta_est_bootstrap)
        distribution: string
            Statistical distribution used to define a confidence region,
            options = 'MVN' for multivariate_normal, 'KDE' for gaussian_kde,
            and 'Rect' for rectangular.
        alphas: list
            List of alpha values used to determine if theta values are inside
            or outside the region.
        test_theta_values: pd.Series or pd.DataFrame, keys/columns = theta_names, optional
            Additional theta values that are compared to the confidence region
            to determine if they are inside or outside.

        Returns
        training_results: pd.DataFrame
            Theta value used to generate the confidence region along with True
            (inside) or False (outside) for each alpha
        test_results: pd.DataFrame
            If test_theta_values is not None, returns test theta value along
            with True (inside) or False (outside) for each alpha
        """
        assert isinstance(theta_values, pd.DataFrame)
        assert distribution in ['Rect', 'MVN', 'KDE']
        assert isinstance(alphas, list)
        assert isinstance(
            test_theta_values, (type(None), dict, pd.Series, pd.DataFrame)
        )

        if isinstance(test_theta_values, (dict, pd.Series)):
            test_theta_values = pd.Series(test_theta_values).to_frame().transpose()

        training_results = theta_values.copy()

        if test_theta_values is not None:
            test_result = test_theta_values.copy()

        for a in alphas:
            if distribution == 'Rect':
                lb, ub = graphics.fit_rect_dist(theta_values, a)
                training_results[a] = (theta_values > lb).all(axis=1) & (
                    theta_values < ub
                ).all(axis=1)

                if test_theta_values is not None:
                    # use upper and lower bound from the training set
                    test_result[a] = (test_theta_values > lb).all(axis=1) & (
                        test_theta_values < ub
                    ).all(axis=1)

            elif distribution == 'MVN':
                dist = graphics.fit_mvn_dist(theta_values)
                Z = dist.pdf(theta_values)
                score = scipy.stats.scoreatpercentile(Z, (1 - a) * 100)
                training_results[a] = Z >= score

                if test_theta_values is not None:
                    # use score from the training set
                    Z = dist.pdf(test_theta_values)
                    test_result[a] = Z >= score

            elif distribution == 'KDE':
                dist = graphics.fit_kde_dist(theta_values)
                Z = dist.pdf(theta_values.transpose())
                score = scipy.stats.scoreatpercentile(Z, (1 - a) * 100)
                training_results[a] = Z >= score

                if test_theta_values is not None:
                    # use score from the training set
                    Z = dist.pdf(test_theta_values.transpose())
                    test_result[a] = Z >= score

        if test_theta_values is not None:
            return training_results, test_result
        else:
            return training_results
