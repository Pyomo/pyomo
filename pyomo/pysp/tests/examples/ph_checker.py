#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import shelve
import json
import os

# TODO:
# - check what weights are for fixed variables (should be zero)
# - check that fix queue from previous iter is always pushed

# don't use this anywhere but this file, it is ugly
def _update_exception_message(msg, e):
    # assert without any args means a coding error
    # in this file, otherwise we want to stack the assertion
    # messages in order to create more informative output
    if len(e.args):
        e.args = (msg+"\n"+e.args[0],)+e.args[1:]
        try:
            e.message = msg + "\n" + e.message
        except:
            # Not python 2
            pass

def assert_value_equals(x,y):
    if x != y:
        raise AssertionError("%s != %s\n" % (x,y))

def assert_float_gt(ub,lb,tol):
    if (ub is None) or (lb is None):
        raise AssertionError("%s !>= %s\n" % (ub,lb))
    if lb-ub > tol:
        raise AssertionError("%.17g !>= %.17g\n" % (ub,lb))

def assert_float_equals(x,y,tol):
    if (x is None) or (y is None):
        raise AssertionError("%s != %s\n" % (x,y))
    if abs(x-y) > tol:
        raise AssertionError("%.17g != %.17g\n" % (x,y))

def validate_probabilities(options, scenario_tree):

    for scenario_name, scenario in scenario_tree['scenarios'].items():
        scenario_probability = scenario['probability']
        test_probability = 1.0
        node_variables = set()
        # check that the probabilities on the scenario tree make sense
        for node_name in scenario['nodes']:
            test_probability *= \
                scenario_tree['nodes'][node_name]['conditional probability']
        try:
            assert_float_equals(scenario_probability,
                                test_probability,
                                options.absolute_tolerance)
        except AssertionError as e:
            msg = ""
            msg += ("Problem validating scenario tree probabilities "
                    "for scenario %s. Product of node conditional "
                    "probabilities does not match scenario probability.\n"
                    % (scenario_name))
            msg += ("Scenario Probability: %r\n" % (scenario_probability))
            msg += ("Node Conditional Product: %r\n" % (test_probability))
            for node_name in scenario['nodes']:
                msg += ("\t%s: %r\n"
                        % (node_name,
                           scenario_tree['nodes'][node_name]['conditional probability']))
            _update_exception_message(msg, e)
            raise

def validate_variable_set(options, scenario_tree, scenario_solutions, node_solutions):

    for scenario_name, scenario in scenario_tree['scenarios'].items():
        scenario_variable_set = \
            set(scenario_solutions[scenario_name]['variables'].keys())
        node_variable_set = set()
        for node_name in scenario['nodes']:
            node_variable_set.update(node_solutions[node_name]['variables'].keys())

        try:
            assert node_variable_set == scenario_variable_set
        except AssertionError as e:
            msg = ("Problem validating the set of reported node "
                   "variables against the set of reported scenario "
                   "variables. Sets do not match:\n")
            msg += ("Symmetric Difference: %s\n"
                    % (set.symmetric_difference(node_variable_set,
                                                scenario_variable_set)))
            raise AssertionError(msg)

def validate_leaf_stage_costs(options,
                              scenario_tree,
                              scenario_solutions,
                              node_solutions):

    # check that the stage cost of the last stage for scenario solutions
    # is equal to the expected node cost for the corresponding leaf node
    # solution
    for scenario_name, scenario in scenario_tree['scenarios'].items():
        scenario_sol = scenario_solutions[scenario_name]
        for stage_name, stage_cost in scenario_sol['stage costs'].items():
            found_one = False
            for node_name in scenario['nodes']:
                node = scenario_tree['nodes'][node_name]
                node_sol = node_solutions[node_name]
                if node['stage'] == stage_name:
                    found_one = True
                    if len(node['children']) == 0:
                        try:
                            assert_float_equals(stage_cost,
                                                node_sol['expected cost'],
                                                options.absolute_tolerance)
                        except AssertionError as e:
                            msg = ""
                            msg += ("Problem validating leaf stage cost for "
                                    "scenario %s. Expected node cost in leaf "
                                    "node solution does not match final stage "
                                    "cost in scenario solution.\n"
                                    % (scenario_name))
                            msg += ("Expected cost of node %s: %r\n"
                                    % (node_name, node_sol['expected cost']))
                            msg += ("Scenario cost of stage %s: %r\n"
                                    % (stage_name, stage_cost))
                            _update_exception_message(msg, e)
                            raise

            assert found_one

def validate_scenario_costs(options, scenario_tree, scenario_solutions):
    for scenario_name, scenario in scenario_tree['scenarios'].items():
        assert scenario_name == scenario['name']
        scenario_sol = scenario_solutions[scenario_name]

        try:
            assert scenario_sol['objective'] is not None, "value cannot be None"
            assert scenario_sol['cost'] is not None, "value cannot be None"
            assert scenario_sol['ph weight term'] is not None, "value cannot be None"
            assert scenario_sol['ph proximal term'] is not None, "value cannot be None"
            assert_float_equals(scenario_sol['objective'],
                                scenario_sol['cost']+\
                                scenario_sol['ph weight term']+\
                                scenario_sol['ph proximal term'],
                                options.absolute_tolerance)
        except AssertionError as e:
            msg = ""
            msg += ("Problem validating objective terms for "
                    "scenario %s. Sum of cost and ph terms does not "
                    "match objective\n" % (scenario_name))
            msg += ("Objective: %r\n" % (scenario_sol['objective']))
            msg += ("Cost: %r\n" % (scenario_sol['cost']))
            msg += ("PH Weight Term: %r\n" % (scenario_sol['ph weight term']))
            msg += ("PH Proximal Term: %r\n" % (scenario_sol['ph proximal term']))
            _update_exception_message(msg, e)
            raise

        try:
            assert_float_equals(scenario_sol['cost'],
                                sum(scenario_sol['stage costs'][stage_name] \
                                    for stage_name in scenario_tree['stages']),
                                options.absolute_tolerance)
        except AssertionError as e:
            msg = ""
            msg += ("Problem validating stage costs terms for "
                    "scenario %s. Sum of stage costs does not match "
                    "value of total cost term used in objective function.\n"
                    % (scenario_name))
            msg += ("Cost: %.17g\n" % (scenario_sol['cost']))
            for stage_name in scenario_tree['stages']:
                msg += ("%s Cost: %.17g\n" % (stage_name, scenario_sol['stage costs'][stage_name]))
            _update_exception_message(msg, e)
            raise

def validate_variable_statistics(options,
                                 scenario_tree,
                                 node_solutions,
                                 scenario_solutions):

    for node_name, node in scenario_tree['nodes'].items():
        try:
            assert node_name == node['name']
            node_sol = node_solutions[node_name]
            is_leaf_node = bool(len(node['children']) == 0)

            for scenario_name in node['scenarios']:
                scenario_sol = scenario_solutions[scenario_name]

            variable_wbar = dict.fromkeys(node_sol['variables'],0)
            variable_avg = dict.fromkeys(node_sol['variables'],0)
            variable_min = dict.fromkeys(node_sol['variables'],None)
            variable_max = dict.fromkeys(node_sol['variables'],None)
            variable_fixed_count = dict.fromkeys(node_sol['variables'],0)
            variable_fixed_value = \
                dict((variable_name,dict.fromkeys(node['scenarios'],None)) \
                     for variable_name in node_sol['variables'])

            for variable_name in node_sol['variables']:
                for scenario_name in node['scenarios']:
                    probability = scenario_tree['scenarios'][scenario_name]['probability']
                    scenario_sol = scenario_solutions[scenario_name]
                    scenario_variables_sol = scenario_sol['variables']
                    var_sol = scenario_sol['variables'][variable_name]
                    varval = var_sol['value']
                    variable_avg[variable_name] += probability*varval
                    if var_sol['weight'] is not None:
                        variable_wbar[variable_name] += \
                            probability*var_sol['weight']
                    else:
                        variable_wbar[variable_name] = None

                    if (variable_min[variable_name] is None) or \
                       (varval < variable_min[variable_name]):
                        variable_min[variable_name] = varval
                    if (variable_max[variable_name] is None) or \
                       (varval > variable_max[variable_name]):
                        variable_max[variable_name] = varval
                    variable_fixed_count[variable_name] += int(var_sol['fixed'])
                    if var_sol['fixed']:
                        variable_fixed_value[variable_name][scenario_name] = varval

            for variable_name, variable_sol in node_sol['variables'].items():
                scaled_avg = variable_avg[variable_name]/node['probability']
                # validate fixed count
                try:
                    if variable_sol['fixed']:
                        assert_value_equals(variable_fixed_count[variable_name],
                                            len(node['scenarios']))
                    else:
                        assert_value_equals(variable_fixed_count[variable_name],
                                            0)
                except AssertionError as e:
                    msg = ""
                    msg += ("Problem validating fixed status for variable %s. "
                            "The fixed status of this variable is not consistent "
                            "across all scenarios within this node.\n"
                            % (variable_name))
                    _update_exception_message(msg, e)
                    raise

                if variable_sol['fixed']:
                    scenario_name = None
                    scenario_value = None
                    try:
                        dict_items = list(variable_fixed_value[variable_name].items())
                        for (scen1_name, scen1_val), (scen2_name, scen2_val) in zip(dict_items, dict_items[1:]):
                            scenario_name = scen1_name
                            scenario_value = scen1_val
                            assert_float_equals(scen1_val,
                                                scen2_val,
                                                options.absolute_tolerance)
                        for scen_name, scen_val in dict_items:
                            scenario_name = scen_name
                            scenario_value = scen_val
                            assert_float_equals(scen_val,
                                                scaled_avg,
                                                options.absolute_tolerance)

                    except AssertionError as e:
                        msg = ""
                        msg += ("Problem validating fixed status for variable %s "
                                "in scenario %s. Fixed value does not match that of "
                                "other scenarios in this node or that of node average.\n"
                                % (variable_name, scenario_name))
                        msg += ("Scenario %s fixed value: %.17g\n"
                                % (scenario_name, scenario_value))
                        msg += ("Computed average from scenario solutions: %.17g\n"
                                % (scaled_avg))
                        _update_exception_message(msg, e)
                        raise
                    del scenario_name
                    del scenario_value

                # validate node "solution" (average)
                try:
                    assert_float_equals(variable_sol['solution'],
                                        scaled_avg,
                                        options.absolute_tolerance)
                except AssertionError as e:
                    msg = ""
                    msg += ("Problem validating node 'solution' for variable %s\n"
                            % (variable_name))
                    msg += ("Node 'solution' in solution: %.17g\n"
                            % variable_sol['solution'])
                    msg += ("Computed average from scenario solutions: %.17g\n"
                            % (scaled_avg))
                    _update_exception_message(msg, e)
                    raise

                if not is_leaf_node:
                    # validate node average
                    try:
                        assert_float_equals(variable_sol['average'],
                                            scaled_avg,
                                            options.absolute_tolerance)
                    except AssertionError as e:
                        msg = ""
                        msg += ("Problem validating node average for variable %s\n"
                                % (variable_name))
                        msg += ("Node average in solution: %.17g\n"
                                % variable_sol['average'])
                        msg += ("Computed average from scenario solutions: %.17g\n"
                                % (scaled_avg))
                        _update_exception_message(msg, e)
                        raise

                    # validate node max
                    try:
                        assert_float_equals(variable_sol['maximum'],
                                            variable_max[variable_name],
                                            options.absolute_tolerance)
                    except AssertionError as e:
                        msg = ""
                        msg += ("Problem validating node maximum for variable %s\n"
                                % (variable_name))
                        msg += ("Node maximum in solution: %.17g\n"
                                % variable_sol['maximum'])
                        msg += ("Computed maximum from scenario solutions: %.17g\n"
                                % (variable_max[variable_name]))
                        _update_exception_message(msg, e)
                        raise

                    # validate node min
                    try:
                        assert_float_equals(variable_sol['minimum'],
                                            variable_min[variable_name],
                                            options.absolute_tolerance)
                    except AssertionError as e:
                        msg = ""
                        msg += ("Problem validating node minimum for variable %s\n"
                                % (variable_name))
                        msg += ("Node minimum in solution: %r\n"
                                % variable_sol['minimum'])
                        msg += ("Computed minimum from scenario solutions: %r\n"
                                % (variable_min[variable_name]))
                        _update_exception_message(msg, e)
                        raise

                    # validate wbar
                    try:
                        assert_float_equals(variable_sol['wbar'],
                                            variable_wbar[variable_name],
                                            options.absolute_tolerance)
                    except AssertionError as e:
                        msg = ""
                        msg += ("Problem validating node wbar for variable %s\n"
                                % (variable_name))
                        msg += ("Node wbar in solution: %r\n"
                                % variable_sol['wbar'])
                        msg += ("Computed wbar from scenario solutions: %r\n"
                                % (variable_wbar[variable_name]))
                        _update_exception_message(msg, e)
                        raise

        except AssertionError as e:
            msg = ("Problem validating node %s" % (node_name))
            _update_exception_message(msg, e)
            raise

def validate_ph_objective_parameters(options,
                                     scenario_tree,
                                     scenario_solutions_previous,
                                     node_solutions_previous,
                                     scenario_solutions_current,
                                     node_solutions_current):

    # check that sum of weigth and rho and xbar from prev iter
    # with current variable sols compute the current sol's
    # ph objective terms
    computed_proximal_term = dict.fromkeys(scenario_tree['scenarios'],0.0)
    computed_weight_term = dict.fromkeys(scenario_tree['scenarios'],0.0)
    for node_name, node in scenario_tree['nodes'].items():
        assert node_name == node['name']
        prev_node_vars_sol = node_solutions_previous[node_name]['variables']
        curr_node_vars_sol = node_solutions_current[node_name]['variables']
        is_leaf_node = bool(len(node['children']) == 0)
        if not is_leaf_node:
            for variable_name, prev_node_variable_sol in prev_node_vars_sol.items():
                curr_node_variable_sol = curr_node_vars_sol[variable_name]
                derived = prev_node_variable_sol['derived']
                assert derived == curr_node_variable_sol['derived']
                if derived:
                    continue
                prev_var_xbar = prev_node_variable_sol['xbar']
                for scenario_name in node['scenarios']:
                    prev_scenario_sol = scenario_solutions_previous[scenario_name]
                    prev_var_scenario_sol = \
                        prev_scenario_sol['variables'][variable_name]
                    curr_scenario_sol = scenario_solutions_current[scenario_name]
                    curr_var_scenario_sol = \
                        curr_scenario_sol['variables'][variable_name]
                    computed_proximal_term[scenario_name] += \
                            prev_var_scenario_sol['rho']*0.5*\
                            (curr_var_scenario_sol['value'] - \
                             prev_var_xbar)**2
                    computed_weight_term[scenario_name] += \
                            prev_var_scenario_sol['weight']*\
                            curr_var_scenario_sol['value']

    for scenario_name, scenario in scenario_tree['scenarios'].items():
        curr_scenario_sol = scenario_solutions_current[scenario_name]
        try:
            if options.proximal_is_lb:
                assert_float_gt(computed_proximal_term[scenario_name],
                                curr_scenario_sol['ph proximal term'],
                                options.absolute_tolerance)
            elif options.proximal_is_ub:
                assert_float_gt(curr_scenario_sol['ph proximal term'],
                                computed_proximal_term[scenario_name],
                                options.absolute_tolerance)
            elif not options.disable_proximal_check:
                assert_float_equals(computed_proximal_term[scenario_name],
                                    curr_scenario_sol['ph proximal term'],
                                    options.absolute_tolerance)
        except AssertionError as e:
            msg = ""
            msg += ("Reported Proximal Term: %.17g\n"
                    % (curr_scenario_sol['ph proximal term']))
            msg += ("Computed Proximal Term: %.17g\n"
                    % (computed_proximal_term[scenario_name]))
            msg += ("Problem validating ph proximal term for "
                    "scenario %s. Value reported in current solution "
                    "does not %s value computed using ph parameters "
                    "reported in previous iteration with variable solution "
                    "reported in current iteration.\n"
                    % (scenario_name,
                       "correctly bound" if (options.proximal_is_lb or \
                                             options.proximal_is_ub)
                                         else
                       "match"))
            _update_exception_message(msg, e)
            raise
        try:
            assert_float_equals(computed_weight_term[scenario_name],
                                curr_scenario_sol['ph weight term'],
                                options.absolute_tolerance)
        except AssertionError as e:
            msg = ""
            msg += ("Reported Weight Term: %.17g\n"
                    % (curr_scenario_sol['ph weight term']))
            msg += ("Computed Weight Term: %.17g\n"
                    % (computed_weight_term[scenario_name]))
            msg += ("Problem validating ph weight term for "
                    "scenario %s. Value reported in current solution "
                    "does not match value computed using ph parameters "
                    "reported in previous iteration with variable solution "
                    "reported in current iteration.\n" % (scenario_name))
            _update_exception_message(msg, e)
            raise

def validate_ph(options):

    history = None
    try:
        with open(options.ph_history_filename) as f:
            history = json.load(f)
    except:
        history = None
        try:
            history = shelve.open(options.ph_history_filename,
                                  flag='r')
        except:
            history = None

    if history is None:
        raise RuntimeError("Unable to open ph history file as JSON "
                           "or python Shelve DB format")

    scenario_tree = history['scenario tree']

    try:
        validate_probabilities(options,
                               scenario_tree)
    except AssertionError as e:
        msg = "FAILURE:\n"
        msg += ("Problem validating probabilities for scenario "
                "tree in ph history from file: %s"
                % (options.ph_history_filename))
        _update_exception_message(msg, e)
        raise

    try:
        iter_keys = history['results keys']
    except KeyError:
        # we are using json format (which loads the entire file
        # anyway)
        iter_keys = list(history.keys())
        iter_keys.remove('scenario tree')
    iterations = sorted(int(k) for k in iter_keys)
    iterations = [str(k) for k in iterations]

    prev_iter_results = None
    for i in iterations:

        iter_results = history[i]

        convergence = iter_results['convergence']
        scenario_solutions = iter_results['scenario solutions']
        node_solutions = iter_results['node solutions']

        try:
            validate_variable_set(options,
                                  scenario_tree,
                                  scenario_solutions,
                                  node_solutions)
        except AssertionError as e:
            msg = "FAILURE:\n"
            msg += ("Problem validating variable set for iteration %s "
                    "of ph history in file: %s"
                    % (i, options.ph_history_filename))
            _update_exception_message(msg, e)
            raise


        try:
            validate_leaf_stage_costs(options,
                                      scenario_tree,
                                      scenario_solutions,
                                      node_solutions)
        except AssertionError as e:
            msg = "FAILURE:\n"
            msg += ("Problem validating leaf-stage costs for iteration %s "
                    "of ph history in file: %s"
                    % (i, options.ph_history_filename))
            _update_exception_message(msg, e)
            raise


        try:
            validate_scenario_costs(options, scenario_tree, scenario_solutions)
        except AssertionError as e:
            msg = "FAILURE:\n"
            msg += ("Problem validating scenario costs for iteration %s "
                    "of ph history in file: %s"
                    % (i, options.ph_history_filename))
            _update_exception_message(msg, e)
            raise


        try:
            validate_scenario_costs(options, scenario_tree, scenario_solutions)
        except AssertionError as e:
            msg = "FAILURE:\n"
            msg += ("Problem validating scenario costs for iteration %s "
                    "of ph history in file: %s"
                    % (i, options.ph_history_filename))
            _update_exception_message(msg, e)
            raise

        try:
            validate_variable_statistics(options,
                                         scenario_tree,
                                         node_solutions,
                                         scenario_solutions)
        except AssertionError as e:
            msg = "FAILURE:\n"
            msg += ("Problem validating variable statistics for iteration %s "
                    "of ph history in file: %s"
                    % (i, options.ph_history_filename))
            _update_exception_message(msg, e)
            raise

        if prev_iter_results is not None:
            try:
                validate_ph_objective_parameters(
                    options,
                    scenario_tree,
                    prev_iter_results['scenario solutions'],
                    prev_iter_results['node solutions'],
                    scenario_solutions,
                    node_solutions)
            except AssertionError as e:
                msg = "FAILURE:\n"
                msg += ("Problem validating ph objective parameters "
                        "for iteration %s of ph history in file: %s"
                        % (i, options.ph_history_filename))
                _update_exception_message(msg, e)
                raise
        prev_iter = i

    return 0

def construct_parser():

    import argparse
    parser = argparse.ArgumentParser(
        description=('Validate a JSON ph history output file that '
                     'was created the phhistoryextension plugin'))
    parser.add_argument('ph_history_filename', metavar='JSON_HISTORY', type=str,
                        action='store', default=None,
                        help=('JSON formatted output file to '
                              'validate'))
    parser.add_argument('-t','--tolerance',dest='absolute_tolerance',
                        action='store', default=1e-6, type=float,
                        help=('absolute tolerance used when comparing '
                              'floating point values (default: %(default)s)'))
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--proximal-term-bounds-below', dest='proximal_is_lb',
                        action='store_true', default=False,
                        help=('relax proximal term check from equality and '
                              'ensure that the solution represents a lower '
                              'bound'))
    group1.add_argument('--proximal-term-bounds-above', dest='proximal_is_ub',
                        action='store_true', default=False,
                        help=('relax proximal term check from equality and '
                              'ensure that the solution represents an upper '
                              'bound'))
    group1.add_argument('--disable-proximal-term-check', dest='disable_proximal_check',
                        action='store_true', default=False,
                        help=('skip validation of proximal term recomputation'))

    return parser

def main(args=None):

    parser = construct_parser()
    options = parser.parse_args(args=args)
    assert os.path.exists(options.ph_history_filename)
    return validate_ph(options)

if __name__ == "__main__":

    sys.exit(main(args=sys.argv[1:]))
