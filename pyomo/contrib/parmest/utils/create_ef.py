#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# This software is distributed under the 3-clause BSD License.
# Copied with minor modifications from create_EF in mpisppy/utils/sputils.py
# from the mpi-sppy library (https://github.com/Pyomo/mpi-sppy).
"""
Note: parmest can make use of mpi-sppy to form the EF so that it could, if it
needed to, solve using a decomposition. To guard against loss of mpi-sppy,
we also have this "local" ability to form the EF.
"""

from pyomo.core.expr.numeric_expr import LinearExpression
import pyomo.environ as pyo
from pyomo.core import Objective


def get_objs(scenario_instance):
    """return the list of objective functions for scenario_instance"""
    scenario_objs = scenario_instance.component_data_objects(
        pyo.Objective, active=True, descend_into=True
    )
    scenario_objs = list(scenario_objs)
    if len(scenario_objs) == 0:
        raise RuntimeError("Scenario " + sname + " has no active objective functions.")
    if len(scenario_objs) > 1:
        print(
            "WARNING: Scenario",
            sname,
            "has multiple active "
            "objectives. Selecting the first objective for "
            "inclusion in the extensive form.",
        )
    return scenario_objs


def _models_have_same_sense(models):
    """Check if every model in the provided dict has the same objective sense.

    Input:
        models (dict) -- Keys are scenario names, values are Pyomo
            ConcreteModel objects.
    Returns:
        is_minimizing (bool) -- True if and only if minimizing. None if the
            check fails.
        check (bool) -- True only if all the models have the same sense (or
            no models were provided)
    Raises:
        ValueError -- If any of the models has either none or multiple
            active objectives.
    """
    if len(models) == 0:
        return True, True
    senses = [
        find_active_objective(scenario).is_minimizing() for scenario in models.values()
    ]
    sense = senses[0]
    check = all(val == sense for val in senses)
    if check:
        return (sense == pyo.minimize), check
    return None, check


def create_EF(
    scenario_names,
    scenario_creator,
    scenario_creator_kwargs=None,
    EF_name=None,
    suppress_warnings=False,
    nonant_for_fixed_vars=True,
):
    """Create a ConcreteModel of the extensive form.

    Args:
        scenario_names (list of str):
            Names for each scenario to be passed to the scenario_creator
            function.
        scenario_creator (callable):
            Function which takes a scenario name as its first argument and
            returns a concrete model corresponding to that scenario.
        scenario_creator_kwargs (dict, optional):
            Options to pass to `scenario_creator`.
        EF_name (str, optional):
            Name of the ConcreteModel of the EF.
        suppress_warnings (boolean, optional):
            If true, do not display warnings. Default False.
        nonant_for_fixed_vars (bool--optional): If True, enforces
            non-anticipativity constraints for all variables, including
            those which have been fixed. Default is True.

    Returns:
        EF_instance (ConcreteModel):
            ConcreteModel of extensive form with explicit
            non-anticipativity constraints.

    Note:
        If any of the scenarios produced by scenario_creator do not have a
        ._mpisppy_probability attribute, this function displays a warning, and assumes
        that all scenarios are equally likely.
    """
    if scenario_creator_kwargs is None:
        scenario_creator_kwargs = dict()
    scen_dict = {
        name: scenario_creator(name, **scenario_creator_kwargs)
        for name in scenario_names
    }

    if len(scen_dict) == 0:
        raise RuntimeError("create_EF() received empty scenario list")
    elif len(scen_dict) == 1:
        scenario_instance = list(scen_dict.values())[0]
        if not suppress_warnings:
            print("WARNING: passed single scenario to create_EF()")
        # special code to patch in ref_vars
        scenario_instance.ref_vars = dict()
        for node in scenario_instance._mpisppy_node_list:
            ndn = node.name
            nlens = {
                node.name: len(node.nonant_vardata_list)
                for node in scenario_instance._mpisppy_node_list
            }
            for i in range(nlens[ndn]):
                v = node.nonant_vardata_list[i]
                if (ndn, i) not in scenario_instance.ref_vars:
                    scenario_instance.ref_vars[(ndn, i)] = v
        # patch in EF_Obj
        scenario_objs = get_objs(scenario_instance)
        for obj_func in scenario_objs:
            obj_func.deactivate()
        obj = scenario_objs[0]
        sense = pyo.minimize if obj.is_minimizing() else pyo.maximize
        scenario_instance.EF_Obj = pyo.Objective(expr=obj.expr, sense=sense)

        return scenario_instance  #### special return for single scenario

    # Check if every scenario has a specified probability
    probs_specified = all(
        [hasattr(scen, '_mpisppy_probability') for scen in scen_dict.values()]
    )
    if not probs_specified:
        for scen in scen_dict.values():
            scen._mpisppy_probability = 1 / len(scen_dict)
        if not suppress_warnings:
            print(
                'WARNING: At least one scenario is missing _mpisppy_probability attribute.',
                'Assuming equally-likely scenarios...',
            )

    EF_instance = _create_EF_from_scen_dict(
        scen_dict, EF_name=EF_name, nonant_for_fixed_vars=True
    )
    return EF_instance


def _create_EF_from_scen_dict(scen_dict, EF_name=None, nonant_for_fixed_vars=True):
    """Create a ConcreteModel of the extensive form from a scenario
    dictionary.

    Args:
        scen_dict (dict): Dictionary whose keys are scenario names and
            values are ConcreteModel objects corresponding to each
            scenario.
        EF_name (str--optional): Name of the resulting EF model.
        nonant_for_fixed_vars (bool--optional): If True, enforces
            non-anticipativity constraints for all variables, including
            those which have been fixed. Default is True.

    Returns:
        EF_instance (ConcreteModel): ConcreteModel of extensive form with
            explicitly non-anticipativity constraints.

    Notes:
        The non-anticipativity constraints are enforced by creating
        "reference variables" at each node in the scenario tree (excluding
        leaves) and enforcing that all the variables for each scenario at
        that node are equal to the reference variables.

        This function is called directly when creating bundles for PH.

        Does NOT assume that each scenario is equally likely. Raises an
        AttributeError if a scenario object is encountered which does not
        have a ._mpisppy_probability attribute.

        Added the flag nonant_for_fixed_vars because original code only
        enforced non-anticipativity for non-fixed vars, which is not always
        desirable in the context of bundling. This allows for more
        fine-grained control.
    """
    is_min, clear = _models_have_same_sense(scen_dict)
    if not clear:
        raise RuntimeError(
            'Cannot build the extensive form out of models '
            'with different objective senses'
        )
    sense = pyo.minimize if is_min else pyo.maximize
    EF_instance = pyo.ConcreteModel(name=EF_name)
    EF_instance.EF_Obj = pyo.Objective(expr=0.0, sense=sense)

    # we don't strict need these here, but it allows for eliding
    # of single scenarios and bundles when convenient
    EF_instance._mpisppy_data = pyo.Block(name="For non-Pyomo mpi-sppy data")
    EF_instance._mpisppy_model = pyo.Block(
        name="For mpi-sppy Pyomo additions to the scenario model"
    )
    EF_instance._mpisppy_data.scenario_feasible = None

    EF_instance._ef_scenario_names = []
    EF_instance._mpisppy_probability = 0
    for sname, scenario_instance in scen_dict.items():
        EF_instance.add_component(sname, scenario_instance)
        EF_instance._ef_scenario_names.append(sname)
        # Now deactivate the scenario instance Objective
        scenario_objs = get_objs(scenario_instance)
        for obj_func in scenario_objs:
            obj_func.deactivate()
        obj_func = scenario_objs[0]  # Select the first objective
        try:
            EF_instance.EF_Obj.expr += (
                scenario_instance._mpisppy_probability * obj_func.expr
            )
            EF_instance._mpisppy_probability += scenario_instance._mpisppy_probability
        except AttributeError as e:
            raise AttributeError(
                "Scenario " + sname + " has no specified "
                "probability. Specify a value for the attribute "
                " _mpisppy_probability and try again."
            ) from e
    # Normalization does nothing when solving the full EF, but is required for
    # appropriate scaling of EFs used as bundles.
    EF_instance.EF_Obj.expr /= EF_instance._mpisppy_probability

    # For each node in the scenario tree, we need to collect the
    # nonanticipative vars and create the constraints for them,
    # which we do using a reference variable.
    ref_vars = dict()  # keys are _nonant_indices (i.e. a node name and a
    # variable number)

    ref_suppl_vars = dict()

    EF_instance._nlens = dict()

    nonant_constr = pyo.Constraint(pyo.Any, name='_C_EF_')
    EF_instance.add_component('_C_EF_', nonant_constr)

    nonant_constr_suppl = pyo.Constraint(pyo.Any, name='_C_EF_suppl')
    EF_instance.add_component('_C_EF_suppl', nonant_constr_suppl)

    for sname, s in scen_dict.items():
        nlens = {
            node.name: len(node.nonant_vardata_list) for node in s._mpisppy_node_list
        }

        for node_name, num_nonant_vars in nlens.items():  # copy nlens to EF
            if (
                node_name in EF_instance._nlens.keys()
                and num_nonant_vars != EF_instance._nlens[node_name]
            ):
                raise RuntimeError(
                    "Number of non-anticipative variables is "
                    "not consistent at node " + node_name + " in scenario " + sname
                )
            EF_instance._nlens[node_name] = num_nonant_vars

        nlens_ef_suppl = {
            node.name: len(node.nonant_ef_suppl_vardata_list)
            for node in s._mpisppy_node_list
        }

        for node in s._mpisppy_node_list:
            ndn = node.name
            for i in range(nlens[ndn]):
                v = node.nonant_vardata_list[i]
                if (ndn, i) not in ref_vars:
                    # create the reference variable as a singleton with long name
                    # xxxx maybe index by _nonant_index ???? rather than singleton VAR ???
                    ref_vars[(ndn, i)] = v
                # Add a non-anticipativity constraint, except in the case when
                # the variable is fixed and nonant_for_fixed_vars=False.
                elif (nonant_for_fixed_vars) or (not v.is_fixed()):
                    expr = LinearExpression(
                        linear_coefs=[1, -1],
                        linear_vars=[v, ref_vars[(ndn, i)]],
                        constant=0.0,
                    )
                    nonant_constr[(ndn, i, sname)] = (expr, 0.0)

            for i in range(nlens_ef_suppl[ndn]):
                v = node.nonant_ef_suppl_vardata_list[i]
                if (ndn, i) not in ref_suppl_vars:
                    # create the reference variable as a singleton with long name
                    # xxxx maybe index by _nonant_index ???? rather than singleton VAR ???
                    ref_suppl_vars[(ndn, i)] = v
                # Add a non-anticipativity constraint, expect in the case when
                # the variable is fixed and nonant_for_fixed_vars=False.
                elif (nonant_for_fixed_vars) or (not v.is_fixed()):
                    expr = LinearExpression(
                        linear_coefs=[1, -1],
                        linear_vars=[v, ref_suppl_vars[(ndn, i)]],
                        constant=0.0,
                    )
                    nonant_constr_suppl[(ndn, i, sname)] = (expr, 0.0)

    EF_instance.ref_vars = ref_vars
    EF_instance.ref_suppl_vars = ref_suppl_vars

    return EF_instance


def find_active_objective(pyomomodel):
    # return the only active objective or raise and error
    obj = list(
        pyomomodel.component_data_objects(Objective, active=True, descend_into=True)
    )
    if len(obj) != 1:
        raise RuntimeError(
            "Could not identify exactly one active "
            "Objective for model '%s' (found %d objectives)"
            % (pyomomodel.name, len(obj))
        )
    return obj[0]


def ef_nonants(ef):
    """An iterator to give representative Vars subject to non-anticipitivity
    Args:
        ef (ConcreteModel): the full extensive form model

    Yields:
        tree node name, full EF Var name, Var value
    """
    for key, val in ef.ref_vars.items():
        yield (key[0], val, pyo.value(val))
