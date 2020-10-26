#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import json
import sys
import os

from pyomo.pysp.tests.examples.ph_checker import \
    _update_exception_message, \
    assert_value_equals, \
    assert_float_equals, \
    validate_probabilities, \
    validate_variable_set, \
    validate_leaf_stage_costs

def validate_nonanticipativity(options,
                               scenario_tree,
                               scenario_solutions,
                               node_solutions):

    # Check that variables with non-anticipativity constraints have
    # converged
    for node_name, node in scenario_tree['nodes'].items():
        assert node_name == node['name']
        node_sol = node_solutions[node_name]
        for scenario_name in node['scenarios']:
            scenario_sol = scenario_solutions[scenario_name]
            # variables, objective, stage costs
            for variable_name, variable_node_sol in node_sol['variables'].items():
                variable_scenario_sol = scenario_sol['variables'][variable_name]
                try:
                    assert_float_equals(variable_node_sol['solution'],
                                        variable_scenario_sol['value'],
                                        options.absolute_tolerance)
                except AssertionError as e:
                    msg = ""
                    msg += ("Problem validating nonanticipativity within "
                            "node %s for variable %s in scenario %s. "
                            "Variable value in scenario solution does not "
                            "match node solution.\n"
                            % (node_name, variable_name, scenario_name))
                    msg += ("Node value: %r\n"
                            % (variable_node_sol['solution']))
                    msg += ("Scenario value: %r\n"
                            % (variable_scenario_sol['value']))
                    _update_exception_message(msg, e)
                    raise
                try:
                    assert_value_equals(variable_node_sol['fixed'],
                                        variable_scenario_sol['fixed'])
                except AssertionError as e:
                    msg = ""
                    msg += ("Problem validating nonanticipativity within "
                            "node %s for variable %s in scenario %s. "
                            "Variable fixed status in scenario solution does not "
                            "match node fixed status.\n"
                            % (node_name, variable_name, scenario_name))
                    msg += ("Node fixed status: %s\n"
                            % (variable_node_sol['fixed']))
                    msg += ("Scenario fixed status: %s\n"
                            % (variable_scenario_sol['fixed']))
                    _update_exception_message(msg, e)
                    raise

def validate_expected_node_costs(options,
                                 scenario_tree,
                                 scenario_solutions,
                                 node_solutions):

    # Check that reported expected node costs make sense
    # and that the expected node cost of the root node
    # equals the ef objective function

    # sorted by time
    sorted_stages = \
        sorted(scenario_tree['stages'].values(),key=lambda x: x['order'])
    # leaf to root
    reverse_sorted_stages = reversed(sorted_stages)
    node_expected_costs = \
        dict((node_name,0.0) for node_name in scenario_tree['nodes'])
    for stage in reverse_sorted_stages:
        assert stage is scenario_tree['stages'][stage['name']]
        stage_name = stage['name']
        for node_name in stage['nodes']:
            node = scenario_tree['nodes'][node_name]
            assert node['name'] == node_name
            node_scenario_names = node['scenarios']
            
            single_scenario_name = node_scenario_names[0]
            single_scenario_stage_cost = \
                scenario_solutions[single_scenario_name]\
                ['stage costs'][stage_name]
            node_cost = 0.0
            for scenario_name in node_scenario_names:
                scenario = scenario_tree['scenarios'][scenario_name]
                scenario_stage_cost = \
                    scenario_solutions[scenario_name]['stage costs'][stage_name]
                node_cost += \
                    scenario['probability'] * \
                    scenario_solutions[scenario_name]['stage costs'][stage_name]

                try:
                    assert_float_equals(single_scenario_stage_cost,
                                        scenario_stage_cost,
                                        options.absolute_tolerance)
                except AssertionError as e:
                    msg = ""
                    msg += ("Problem validating stage costs "
                            "for stage %s of scenarios within node %s. "
                            "Not all scenarios report the same stage cost.\n"
                            % (stage_name, node_name))
                    msg += ("Scenario %s stage cost: %r\n"
                            % (scenario_name, scenario_stage_cost))
                    msg += ("Scenario %s stage cost: %r\n"
                            % (single_scenario_name, single_scenario_stage_cost))
                    _update_exception_message(msg, e)
                    raise

            node_cost /= node['probability']
            node_expected_costs[node_name] += node_cost
            for child_node_name in node['children']:
                child_node = scenario_tree['nodes'][child_node_name]
                node_expected_costs[node_name] += \
                    child_node['conditional probability'] * \
                    node_expected_costs[child_node_name]

                del child_node

            del node
            del node_scenario_names
            del node_cost
            del single_scenario_stage_cost

        del stage_name

    for node_name, expected_cost in node_expected_costs.items():
        node_sol = node_solutions[node_name]
        try:
            assert_float_equals(expected_cost,
                                node_sol['expected cost'],
                                options.absolute_tolerance)
        except AssertionError as e:
            msg = ""
            msg += ("Problem validating node expected cost for node %s\n"
                    % (node_name))
            msg += ("Node expected cost in solution: %r\n"
                    % (node_sol['expected cost']))
            msg += ("Computed average from stage costs: %r\n"
                    % (expected_cost))
            _update_exception_message(msg, e)
            raise

        del node_sol

    # get the root node
    assert len(scenario_tree['stages'][sorted_stages[0]['name']]['nodes']) == 1
    root_node_name = scenario_tree['stages'][sorted_stages[0]['name']]['nodes'][0]
    root_node_expected_cost = node_expected_costs[root_node_name]
    expected_scenario_objective = 0.0
    expected_scenario_cost1 = 0.0
    expected_scenario_cost2 = 0.0
    for scenario_name, scenario in scenario_tree['scenarios'].items():
        scenario_sol = scenario_solutions[scenario_name]
        expected_scenario_objective += \
            scenario['probability']*scenario_sol['objective']
        expected_scenario_cost1 += \
            scenario['probability']*scenario_sol['cost']
        expected_scenario_cost2 += \
            scenario['probability'] * \
            sum(stage_cost for stage_cost in scenario_sol['stage costs'].values())


    try:
        assert_float_equals(root_node_expected_cost,
                            expected_scenario_cost1,
                            options.absolute_tolerance)
    except AssertionError as e:
        msg = ""
        msg += ("Problem validating root node expected cost\n")
        msg += ("Root node expected cost in solution: %r\n"
                % (root_node_expected_cost))
        msg += ("Computed average from scenario costs in solution: %r\n"
                % (expected_scenario_cost1))
        _update_exception_message(msg, e)
        raise

    try:
        assert_float_equals(root_node_expected_cost,
                            expected_scenario_cost2,
                            options.absolute_tolerance)
    except AssertionError as e:
        msg = ""
        msg += ("Problem validating root node expected cost\n")
        msg += ("Root node expected cost in solution: %r\n"
                % (root_node_expected_cost))
        msg += ("Computed average from scenario stage costs in solution: %r\n"
                % (expected_scenario_cost2))
        _update_exception_message(msg, e)
        raise

    try:
        assert_float_equals(root_node_expected_cost,
                            expected_scenario_objective,
                            options.absolute_tolerance)
    except AssertionError as e:
        msg = ""
        msg += ("Problem validating root node expected cost\n")
        msg += ("Root node expected cost in solution: %r\n"
                % (root_node_expected_cost))
        msg += ("Computed average from scenario objectives in solution: %r\n"
                % (expected_scenario_objective))
        _update_exception_message(msg, e)
        raise

def validate_ef(options):

    results = None
    with open(options.ef_solution_filename) as f:
        results = json.load(f)

    scenario_tree = results['scenario tree']

    try:
        validate_probabilities(options,
                               scenario_tree)
    except AssertionError as e:
        msg = "FAILURE:\n"
        msg += ("Problem validating probabilities for scenario "
                "tree in ef solution from file: %s"
                % (options.ef_solution_filename))
        _update_exception_message(msg, e)
        raise

    scenario_solutions = results['scenario solutions']
    node_solutions = results['node solutions']

    assert set(scenario_tree['nodes'].keys()) \
        == set(node_solutions.keys())
    assert set(scenario_tree['scenarios']) \
        == set(scenario_solutions.keys())

    try:
        validate_variable_set(options,
                              scenario_tree,
                              scenario_solutions,
                              node_solutions)
    except AssertionError as e:
        msg = "FAILURE:\n"
        msg += ("Problem validating variable set "
                "in ef solution from file: %s"
                % (options.ef_solution_filename))
        _update_exception_message(msg, e)
        raise

    try:
        validate_nonanticipativity(options,
                                   scenario_tree,
                                   scenario_solutions,
                                   node_solutions)
    except AssertionError as e:
        msg = "FAILURE:\n"
        msg += ("Problem validating nonanticipativity "
                "in ef solution from file: %s"
                % (options.ef_solution_filename))
        _update_exception_message(msg, e)
        raise

    try:
        validate_leaf_stage_costs(options,
                                  scenario_tree,
                                  scenario_solutions,
                                  node_solutions)
    except AssertionError as e:
        msg = "FAILURE:\n"
        msg += ("Problem validating leaf-stage costs "
                "in ef solution from file: %s"
                % (options.ef_solution_filename))
        _update_exception_message(msg, e)
        raise

    try:
        validate_expected_node_costs(options,
                                     scenario_tree,
                                     scenario_solutions,
                                     node_solutions)
    except AssertionError as e:
        msg = "FAILURE:\n"
        msg += ("Problem validating expected node costs "
                "in ef solution from file: %s"
                % (options.ef_solution_filename))
        _update_exception_message(msg, e)
        raise

def construct_parser():

    import argparse
    parser = argparse.ArgumentParser(
        description=('Validate a JSON ef solution output file that '
                     'was created the jsonsolutionwriter plugin'))
    parser.add_argument('ef_solution_filename', metavar='JSON_SOLUTION', type=str,
                        action='store', default=None,
                        help=('JSON formatted output file to '
                              'validate'))
    parser.add_argument('-t','--tolerance',dest='absolute_tolerance',
                        action='store', default=1e-6, type=float,
                        help=('absolute tolerance used when comparing '
                              'floating point values (default: %(default)s)'))

    return parser


def main(args=None):

    parser = construct_parser()
    options = parser.parse_args(args=args)
    assert os.path.exists(options.ef_solution_filename)
    validate_ef(options)

if __name__ == "__main__":

    main(args=sys.argv[1:])
