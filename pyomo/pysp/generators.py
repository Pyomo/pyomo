#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from six import iteritems

# the intent of this file is to provide a collection of generator
# (iterables) functions, which embody certain highly repeated patterns
# of nested iteration in PySP. these generators should allow for
# simplification of the code base, and should generally improve
# maintainability.

# iterates over each stage (minus possibly the last, depending on the
# keyword arguments), each tree node in each stage, each variable in
# each tree node, and each component of each variable. returns a
# six-tuple, including:
# - the stage
# - the tree node
# - the variable name
# - the index of the variable
# - a sequence of (_VarData, probability) pairs for each scenario
#   instance at this node.
# - a boolean indicating whether the variable/index is fixed in all
#   scenario instance.
#   NOTE: We don't actually check for agreement, but we should.
# - a boolean indicating whether the variable/index is stale in all
#   scenario instances.

def scenario_tree_node_variables_generator(scenario_tree,
                                           includeDerivedVariables=True,
                                           includeLastStage=True,
                                           sort=False):

    if includeLastStage is False:
        stages_to_iterate = scenario_tree._stages[:-1]
    else:
        stages_to_iterate = scenario_tree._stages

    for stage in stages_to_iterate:

        tree_nodes = stage._tree_nodes
        if sort:
            tree_nodes = sorted(tree_nodes, key=lambda n: n.name)

        for tree_node in tree_nodes:

            variter = iteritems(tree_node._variable_datas)
            if sort:
                variter = sorted(variter, key=lambda x: x[0])

            for variable_id, variable_datas in variter:

                if (not includeDerivedVariables) and \
                   (variable_id in tree_node._derived_variable_ids):
                    continue

                # implicit assumption is that if a variable value is
                # fixed / stale in one scenario, it is fixed / stale
                # in all scenarios
                is_stale = False
                is_fixed = False

                instance_fixed_count = 0

                assert len(variable_datas)
                for var_data, scenario_probability in variable_datas:
                    if var_data.stale is True:
                        is_stale = True
                    if var_data.fixed is True:
                        instance_fixed_count += 1
                        is_fixed = True

                assert is_fixed == tree_node.is_variable_fixed(variable_id)
                if ((instance_fixed_count > 0) and \
                      (instance_fixed_count < len(tree_node._scenarios))):
                    variable_name, index = tree_node._variable_ids[variable_id]
                    raise RuntimeError("Variable="+variable_name+str(index)+" is "
                                       "fixed in "+str(instance_fixed_count)+" "
                                       "scenarios, which is less than the number "
                                       "of scenarios at tree node="+tree_node._name)

                yield (stage,
                       tree_node,
                       variable_id,
                       variable_datas,
                       is_fixed,
                       is_stale)

def scenario_tree_node_variables_generator_noinstances(scenario_tree,
                                                       includeDerivedVariables=True,
                                                       includeLastStage=True,
                                                       sort=False):

    if includeLastStage is False:
        stages_to_iterate = scenario_tree._stages[:-1]
    else:
        stages_to_iterate = scenario_tree._stages

    for stage in stages_to_iterate:

        tree_nodes = stage._tree_nodes
        if sort:
            tree_nodes = sorted(tree_nodes, key=lambda n: n.name)

        for tree_node in tree_nodes:

            variter = tree_node._variable_ids
            if sort:
                variter = sorted(variter)

            for variable_id in variter:

                if (not includeDerivedVariables) and \
                   (variable_id in tree_node._derived_variable_ids):
                    continue

                variable_values = []
                is_fixed = tree_node.is_variable_fixed(variable_id)
                is_stale = False
                instance_fixed_count = 0
                for scenario in tree_node._scenarios:
                    if scenario.is_variable_stale(tree_node, variable_id):
                        is_stale = True
                    if scenario.is_variable_fixed(tree_node, variable_id):
                        instance_fixed_count += 1
                    variable_values.append((scenario._x[tree_node._name][variable_id],
                                            scenario._probability))

                if ((instance_fixed_count > 0) and \
                      (instance_fixed_count < len(tree_node._scenarios))):
                    variable_name, index = tree_node._variable_ids[variable_id]
                    raise RuntimeError("Variable="+variable_name+str(index)+" is "
                                       "fixed in "+str(instance_fixed_count)+" "
                                       "scenarios, which is less than the number "
                                       "of scenarios at tree node="+tree_node._name)

                yield (stage,
                       tree_node,
                       variable_id,
                       variable_values,
                       is_fixed,
                       is_stale)
