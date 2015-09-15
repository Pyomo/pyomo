import os
import operator
import shutil

thisfile = os.path.abspath(__file__)
thisfile.replace(".pyc","").replace(".py","")

from pyomo.core.base.numvalue import value
from pyomo.core.base.block import _BlockData, SortComponents
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.expression import Expression
from pyomo.core.base.constraint import Constraint, _ConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.piecewise import Piecewise, _PiecewiseData
from pyomo.core.base.suffix import ComponentMap, Suffix
from pyomo.repn import LinearCanonicalRepn
from pyomo.repn import generate_canonical_repn
from pyomo.pysp.scenariotree.tree_structure import ScenarioTree
from pyomo.pysp.scenariotree.scenariotreemanager import \
    ScenarioTreeManagerSPPyro
from pyomo.pysp.scenariotree.scenariotreeserverutils import InvocationType

from six import iteritems, itervalues

# NOTES:
#  - Constants in the objective?
#  - Max variable name length
#  - Multi-stage

#
# PySP_StochasticParameters[param] = table
#
# PySP_StochasticRHS[constraint] = (True/False, True/False)
# PySP_StochasticRHS[constraint] = True
#
# PySP_StochasticMatrix[constraint] = [variable, ...]
#
# PySP_StochasticObjective[variable] = True
# PySP_StochasticObjective[None] = True
#
# PySP_StochasticVariableBounds[variable] = (True/False, True/False)
#

def map_constraint_stages(scenario, scenario_tree, LP_symbol_map):

    reference_model = scenario._instance

    assert len(scenario_tree._stages) == 2
    stage1 = scenario_tree._stages[0]
    stage2 = scenario_tree._stages[1]

    StageToConstraintMap = {}
    StageToConstraintMap[stage1._name] = []
    StageToConstraintMap[stage2._name] = []

    #
    # Keep output deterministic, there is enough to deal
    # with already
    #
    sortOrder = SortComponents.indices | SortComponents.alphabetical

    LP_byObject = LP_symbol_map.byObject
    # deal with the fact that the LP writer prepends constraint
    # names with things like 'c_e_', 'c_l_', etc depending on the
    # constraint bound type and will even split a constraint into
    # two constraints if it has two bounds
    LP_reverse_alias = \
        dict((symbol, []) for symbol in LP_symbol_map.bySymbol)
    for alias, obj_weakref in iteritems(LP_symbol_map.aliases):
        LP_reverse_alias[LP_byObject[id(obj_weakref())]].append(alias)

    # ** SORT POINT TO AVOID NON-DETERMINISTIC ROW ORDERING ***
    for _LP_aliases in itervalues(LP_reverse_alias):
        _LP_aliases.sort()

    #
    # Loop through constraints
    #
    for block in reference_model.block_data_objects(
            active=True,
            descend_into=True,
            sort=sortOrder):

        block_canonical_repn = getattr(block, "_canonical_repn", None)
        if block_canonical_repn is None:
            raise ValueError(
                "Unable to find _canonical_repn ComponentMap "
                "on block %s" % (block.cname(True)))

        piecewise_stage = None
        if isinstance(block, (Piecewise, _PiecewiseData)):
            piecewise_stage = stage1
            for vardata in block.referenced_variables():
                variable_node = \
                    scenario_tree.variableNode(vardata,
                                               reference_model)
                if variable_node._stage == stage2:
                    piecewise_stage = stage2
                else:
                    assert variable_node._stage == stage1

        for constraint_data in block.component_data_objects(
                Constraint,
                active=True,
                descend_into=False,
                sort=sortOrder):

            LP_name = LP_byObject[id(constraint_data)]
            # if it is a range constraint this will account for
            # that fact and hold and alias for each bound
            LP_aliases = LP_reverse_alias[LP_name]
            assert len(LP_aliases) > 0
            if piecewise_stage is None:
                constraint_node = scenario.constraintNode(
                    constraint_data,
                    canonical_repn=block_canonical_repn.get(constraint_data),
                    instance=reference_model)
                constraint_stage = constraint_node._stage
            else:
                constraint_stage = piecewise_stage

            StageToConstraintMap[constraint_stage._name].\
                append((LP_aliases, constraint_data))

    assert sorted(StageToConstraintMap.keys()) == \
        sorted([stage1._name, stage2._name])

    # sort each by name
    for key in StageToConstraintMap:
        StageToConstraintMap[key].sort(key=operator.itemgetter(0))

    return StageToConstraintMap

def map_variable_stages(scenario, scenario_tree, LP_symbol_map):

    reference_model = scenario._instance

    FirstStageVars = {}
    SecondStageVars = {}

    all_vars_cnt = 0
    for block in reference_model.block_data_objects(
            active=True,
            descend_into=True):

        all_vars_cnt += len(list(block.component_data_objects
                                 (Var, descend_into=False)))

    rootnode = scenario_tree.findRootNode()
    assert len(scenario_tree._stages) == 2
    stageone = scenario_tree._stages[0]
    stagetwo = scenario_tree._stages[1]
    stagetwo_node = scenario._node_list[-1]
    assert stagetwo_node._stage is stagetwo
    firststage_blended_variables = rootnode._standard_variable_ids
    LP_byObject = LP_symbol_map.byObject
    all_vars_on_tree = []

    for scenario_tree_id, vardata in \
          iteritems(reference_model.\
          _ScenarioTreeSymbolMap.bySymbol):
        try:
            LP_name = LP_byObject[id(vardata)]
        except:
            print(("FAILED ON VAR DATA= "+vardata.cname(True)))
            foobar
        if LP_name == "RHS":
            raise RuntimeError(
                "Congratulations! You have hit an edge case. The "
                "SMPS input format forbids variables from having "
                "the name 'RHS'. Please rename it")
        if scenario_tree_id in firststage_blended_variables:
            FirstStageVars[LP_name] = (vardata, scenario_tree_id)
        elif (scenario_tree_id in rootnode._derived_variable_ids) or \
             (scenario_tree_id in stagetwo_node._variable_ids):
            SecondStageVars[LP_name] = (vardata, scenario_tree_id)
        else:
            # More than two stages?
            assert False

        all_vars_on_tree.append(LP_name)

    for stage in scenario_tree._stages:
        cost_variable_name, cost_variable_index = \
            stage._cost_variable
        stage_cost_component = \
            reference_model.\
            find_component(cost_variable_name)
        if stage_cost_component.type() is not Expression:
            raise RuntimeError(
                "All StageCostVariables must be declared "
                "as Expression objects when using this tool")

    # Make sure every variable on the model has been
    # declared on the scenario tree
    if len(all_vars_on_tree) != all_vars_cnt:
        print("**** THERE IS A PROBLEM ****")
        print("Not all model variables are on the "
              "scenario tree. Investigating...")
        all_vars = set()
        for block in reference_model.block_data_objects(
                active=True,
                descend_into=True):
            all_vars.update(
                vardata.cname(True) \
                for vardata in block.component_data_objects
                (Var, descend_into=False))

        tree_vars = set()
        for scenario_tree_id, vardata in \
            iteritems(reference_model.\
                      _ScenarioTreeSymbolMap.bySymbol):
            tree_vars.add(vardata.cname(True))
        cost_vars = set()
        for stage in scenario_tree._stages:
            cost_variable_name, cost_variable_index = \
                stage._cost_variable
            stage_cost_component = \
                reference_model.\
                find_component(cost_variable_name)
            if stage_cost_component.type() is not Expression:
                cost_vars.add(
                    stage_cost_component[cost_variable_index].\
                    cname(True))

        print("Number of Scenario Tree Variables "
              "(found ddsip LP file): "+str(len(tree_vars)))
        print("Number of Scenario Tree Cost Variables "
               "(found ddsip LP file): "+str(len(cost_vars)))
        print("Number of Variables Found on Model: "
              +str(len(all_vars)))
        print("Variables Missing from Scenario Tree "
              "(or LP file):"+str(all_vars-tree_vars-cost_vars))

    # A necessary but not sufficient sanity check to make sure the
    # second stage variable sets are the same for all
    # scenarios. This is not required by pysp, but I think this
    # assumption is made in the rest of the code here
    for tree_node in stagetwo._tree_nodes:
        assert len(stagetwo_node._variable_ids) == \
            len(tree_node._variable_ids)

    assert len(scenario_tree._stages) == 2

    StageToVariableMap = {}
    StageToVariableMap[stageone._name] = \
        [(LP_name,
          FirstStageVars[LP_name][0],
          FirstStageVars[LP_name][1])
         for LP_name in sorted(FirstStageVars)]
    StageToVariableMap[stagetwo._name] = \
        [(LP_name,
          SecondStageVars[LP_name][0],
          SecondStageVars[LP_name][1])
         for LP_name in sorted(SecondStageVars)]

    return StageToVariableMap

def EXTERNAL_convert_explicit_setup(scenario_tree_manager,
                                    scenario_tree,
                                    scenario,
                                    output_directory,
                                    basename,
                                    io_options):
    import pyomo.environ
    import pyomo.repn.plugins.cpxlp
    assert os.path.exists(output_directory)

    reference_model = scenario._instance

    with pyomo.repn.plugins.cpxlp.ProblemWriter_cpxlp() as writer:
        lp_filename = os.path.join(output_directory,
                                   basename+".setup.lp."+scenario._name)
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        output_filename, LP_symbol_map = writer(reference_model,
                                                lp_filename,
                                                lambda x: True,
                                                io_options)
        assert output_filename == lp_filename

    StageToVariableMap = \
        map_variable_stages(scenario,
                            scenario_tree,
                            LP_symbol_map)

    StageToConstraintMap = \
        map_constraint_stages(scenario,
                              scenario_tree,
                              LP_symbol_map)

    assert len(scenario_tree._stages) == 2
    firststage = scenario_tree._stages[0]
    secondstage = scenario_tree._stages[1]

    #
    # Make sure the objective references all first stage variables.
    # We do this by directly modifying the canonical_repn of the
    # objective which the LP writer will reference next time we call
    # it. In addition, make that that the first second-stage variable
    # in our column ordering also appears in the objective so that
    # ONE_VAR_CONSTANT does not get identified as the first
    # second-stage variable.
    # ** Just do NOT preprocess again until we call the LP writer **
    #
    obj_canonical_repn_flag = \
        getattr(reference_model, "_gen_obj_canonical_repn", None)
    reference_model._gen_obj_canonical_repn = False
    canonical_repn = reference_model._canonical_repn
    obj_repn = canonical_repn[scenario._instance_objective]
    first_stage_varname_list = \
        [item[0] for item in StageToVariableMap[firststage._name]]
    if isinstance(obj_repn, LinearCanonicalRepn) and \
       (obj_repn.linear is not None):
        referenced_var_names = [LP_symbol_map.byObject[id(vardata)]
                                for vardata in obj_repn.variables]
        update_vars = []
        # add the first-stage variables (if not present)
        for LP_name in first_stage_varname_list:
            if LP_name not in referenced_var_names:
                update_vars.append(LP_symbol_map.bySymbol[LP_name])
        # add the first second-stage variable (if not present)
        if StageToVariableMap[secondstage._name][0][0] not in \
           referenced_var_names:
            update_vars.append(
                StageToVariableMap[secondstage._name][0][1])
        obj_repn.variables = list(obj_repn.variables) + \
                             update_vars
        obj_repn.linear = list(obj_repn.linear) + \
                          [0.0 for vardata in update_vars]
    else:
        raise RuntimeError("Unexpected objective representation")

    #
    # Create column (variable) ordering maps for LP files
    #
    column_order = ComponentMap()
    # first-stage variables
    for column_index, (LP_name, vardata, scenario_tree_id) \
          in enumerate(StageToVariableMap[firststage._name]):
        column_order[vardata] = column_index
    # second-stage variables
    for column_index, (LP_name, vardata, scenario_tree_id) \
        in enumerate(StageToVariableMap[secondstage._name],
                     len(column_order)):
        column_order[vardata] = column_index

    #
    # Create row (constraint) ordering maps for LP files
    #
    row_order = ComponentMap()
    # first-stage constraints
    for row_index, (LP_names, condata) \
          in enumerate(StageToConstraintMap[firststage._name]):
        row_order[condata] = row_index
    # second-stage constraints
    for row_index, (LP_names, condata) \
        in enumerate(StageToConstraintMap[secondstage._name],
                     len(row_order)):
        row_order[condata] = row_index

    #
    # Write the ordered LP file
    #
    with pyomo.repn.plugins.cpxlp.ProblemWriter_cpxlp() as writer:
        lp_filename = os.path.join(output_directory,
                                   basename+".lp."+scenario._name)
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        io_options = dict(io_options)
        io_options['column_order'] = column_order
        io_options['row_order'] = row_order
        output_filename, LP_symbol_map = writer(reference_model,
                                                lp_filename,
                                                lambda x: True,
                                                io_options)
        assert output_filename == lp_filename

    # Restore this PySP hack to its original value
    if obj_canonical_repn_flag is None:
        delattr(reference_model, "_gen_obj_canonical_repn")
    else:
        setattr(reference_model,
                "_gen_obj_canonical_repn",
                obj_canonical_repn_flag)

    #
    # Write the .tim file
    #
    with open(os.path.join(output_directory,
                           basename+".tim."+scenario._name),'w') as f:
        f.write('TIME\n')
        f.write('PERIODS\tLP\n')
        f.write('\t'+str(StageToVariableMap[firststage._name][0][0])+
                '\t'+str(LP_symbol_map.byObject\
                         [id(scenario._instance_objective)])+
                '\tTIME1\n')
        LP_names = StageToConstraintMap[secondstage._name][0][0]
        if len(LP_names) == 1:
            # equality constraint
            assert (LP_names[0].startswith('c_e_') or \
                    LP_names[0].startswith('c_l_') or \
                    LP_names[0].startswith('c_u_'))
            stage2_row_start = LP_names[0]
        else:
            # range constraint (assumed the LP writer outputs
            # the lower range constraint first)
            LP_names = sorted(LP_names)
            assert (LP_names[0].startswith('r_l_') or \
                    LP_names[0].startswith('r_u_'))
            stage2_row_start = LP_names[0]
        f.write('\t'+str(StageToVariableMap[secondstage._name][0][0])+
                '\t'+stage2_row_start+
                '\tTIME2\n')
        f.write('ENDATA\n')

    #
    # Write the .sto file
    #
    stochastic_rhs = reference_model.find_component('PySP_StochasticRHS')
    if (stochastic_rhs is not None) and \
       (not isinstance(stochastic_rhs, Suffix)):
        raise TypeError(
            "Object with name 'PySP_StochasticRHS' was found "
            "on model %s that is not of type 'Suffix'"
            % (reference_model.cname(True)))
    stochastic_matrix = \
        reference_model.find_component('PySP_StochasticMatrix')
    if (stochastic_matrix is not None) and \
       (not isinstance(stochastic_matrix, Suffix)):
        raise TypeError(
            "Object with name 'PySP_StochasticMatrix' was found "
            "on model %s that is not of type 'Suffix'"
            % (reference_model.cname(True)))
    stochastic_objective = \
        reference_model.find_component('PySP_StochasticObjective')
    if (stochastic_objective is not None) and \
       (not isinstance(stochastic_objective, Suffix)):
        raise TypeError(
            "Object with name 'PySP_StochasticObjective' was found "
            "on model %s that is not of type 'Suffix'"
            % (reference_model.cname(True)))
    stochastic_bounds = \
        reference_model.find_component('PySP_StochasticVariableBounds')
    if (stochastic_bounds is not None) and \
       (not isinstance(stochastic_bounds, Suffix)):
        raise TypeError(
            "Object with name 'PySP_StochasticVariableBounds' was "
            "found on model %s that is not of type 'Suffix'"
            % (reference_model.cname(True)))

    if (stochastic_bounds is not None):
        assert False

    constraint_LP_names = ComponentMap()
    for stage_name in StageToConstraintMap:
        for LP_names, constraint_data in StageToConstraintMap[stage_name]:
            constraint_LP_names[constraint_data] = LP_names

    if ((stochastic_rhs is None) or (len(stochastic_rhs) == 0)) and \
       ((stochastic_matrix is None) or (len(stochastic_matrix) == 0)) and \
       ((stochastic_objective is None) or (len(stochastic_objective) == 0)) and \
       ((stochastic_bounds is None) or (len(stochastic_bounds) == 0)):
        raise RuntimeError("No stochastic annotations found on model")

    with open(os.path.join(output_directory,
                           basename+".sto."+scenario._name),'w') as f:
        scenario_probability = None
        if hasattr(scenario_tree_manager,
                   "_uncompressed_scenario_tree"):
            scenario_probability = \
                scenario_tree_manager.\
                _uncompressed_scenario_tree.get_scenario\
                (scenario.name).probability
        else:
            scenario_probability = scenario._probability

        f.write(" BL BLOCK1 PERIOD2 %.17g\n" % (scenario_probability))
        #
        # Stochastic RHS
        #
        rhs_template = "    RHS    %s    %.17g\n"
        if stochastic_rhs is not None:
            sorted_names = [(constraint_data.cname(True),
                             constraint_data)
                            for constraint_data in stochastic_rhs]
            sorted_names.sort(key=operator.itemgetter(0))
            for cname, constraint_data in sorted_names:
                assert isinstance(constraint_data, _ConstraintData)
                include_bound = stochastic_rhs[constraint_data]
                # compute_values=False should prevent variables with a
                # zero coefficient from being eliminated from the
                # canonical representation
                constraint_repn = \
                    generate_canonical_repn(constraint_data.body,
                                            compute_values=False)
                assert isinstance(constraint_repn,
                                  LinearCanonicalRepn)
                body_constant = constraint_repn.constant
                if body_constant is None:
                    body_constant = 0.0
                #
                # **NOTE: In the code that follows we assume the LP
                #         writer always moves constraint body
                #         constants to the rhs
                #
                LP_names = constraint_LP_names[constraint_data]
                for con_label in LP_names:
                    if con_label.startswith('c_e_') or \
                       con_label.startswith('c_l_'):
                        assert include_bound is True
                        f.write(rhs_template %
                                (con_label,
                                 value(constraint_data.lower) - \
                                 value(body_constant)))
                    elif con_label.startswith('r_l_') :
                        assert len(include_bound) == 2
                        if include_bound[0] is True:
                            f.write(rhs_template %
                                    (con_label,
                                     value(constraint_data.lower) - \
                                     value(body_constant)))
                    elif con_label.startswith('c_u_'):
                        assert include_bound is True
                        f.write(rhs_template %
                                (con_label,
                                 value(constraint_data.upper) - \
                                 value(body_constant)))
                    elif con_label.startswith('r_u_'):
                        if include_bound[1] is True:
                            f.write(rhs_template %
                                    (con_label,
                                     value(constraint_data.upper) - \
                                     value(body_constant)))
                    else:
                        assert False

        #
        # Stochastic Matrix
        #
        matrix_template = "    %s    %s    %.17g\n"
        if stochastic_matrix is not None:
            sorted_names = [(constraint_data.cname(True),
                             constraint_data)
                            for constraint_data in stochastic_matrix]
            sorted_names.sort(key=operator.itemgetter(0))
            for cname, constraint_data in sorted_names:
                assert isinstance(constraint_data, _ConstraintData)
                # compute_values=False should prevent variables with a
                # zero coefficient from being eliminated from the
                # canonical representation
                constraint_repn = \
                    generate_canonical_repn(constraint_data.body,
                                            compute_values=False)
                assert isinstance(constraint_repn,
                                  LinearCanonicalRepn)
                assert len(constraint_repn.variables) > 0
                for var_data in stochastic_matrix[constraint_data]:
                    assert isinstance(var_data, _VarData)
                    var_coef = None
                    for var, coef in zip(constraint_repn.variables,
                                         constraint_repn.linear):
                        if var is var_data:
                            var_coef = coef
                            break
                    assert var_coef is not None
                    var_label = LP_symbol_map.byObject[id(var_data)]
                    LP_names = constraint_LP_names[constraint_data]
                    for con_label in LP_names:
                        f.write(matrix_template % (var_label,
                                                   con_label,
                                                   value(var_coef)))

        #
        # Stochastic Objective
        #
        obj_template = "    %s    %s    %.17g\n"
        if stochastic_objective is not None:
            # compute_values=False should prevent variables with a
            # zero coefficient from being eliminated from the
            # canonical representation
            objective_repn = \
                generate_canonical_repn(scenario._instance_objective.expr,
                                        compute_values=False)
            assert isinstance(objective_repn, LinearCanonicalRepn)
            sorted_names = [(var_data.cname(True), var_data)
                            for var_data in stochastic_objective]
            sorted_names.sort(key=operator.itemgetter(0))
            for cname, var_data in sorted_names:
                var_label = "ONE_VAR_CONSTANT"
                if var_data is None:
                    var_coef = objective_repn.constant
                    if var_coef is None:
                        var_coef = 0.0
                else:
                    assert isinstance(var_data, _VarData)
                    var_label = LP_symbol_map.byObject[id(var_data)]
                    var_coef = None
                    for var, coef in zip(objective_repn.variables,
                                         objective_repn.linear):
                        if var is var_data:
                            var_coef = coef
                            break
                assert var_coef is not None
                obj_label = LP_symbol_map.byObject[id(objective)]
                f.write(obj_template % (var_label,
                                        obj_label,
                                        value(var_coef)))

def convert_explicit(output_directory,
                     basename,
                     scenario_tree_manager,
                     io_options=None,
                     keep_scenario_files=False):
    import pyomo.environ
    import pyomo.solvers.plugins.smanager.phpyro

    if io_options is None:
        io_options = {}

    assert os.path.exists(output_directory)

    scenario_tree = scenario_tree_manager.scenario_tree

    if scenario_tree.contains_bundles():
        raise ValueError(
            "SMPS conversion does not yet handle bundles")

    scenario_directory = \
        os.path.abspath(os.path.join(output_directory,
                                     'scenario_files'))

    if not os.path.exists(scenario_directory):
        os.mkdir(scenario_directory)

    scenario_tree_manager.invoke_external_function(
        thisfile,
        "EXTERNAL_convert_explicit_setup",
        invocation_type=InvocationType.PerScenario,
        function_args=(scenario_directory,
                       basename,
                       io_options))

    #
    # Select on of the per-scenario .lp files as the master
    #
    lp_filename = os.path.join(output_directory, basename+".lp")
    rc = os.system(
        'cp '+os.path.join(scenario_directory,
                           (basename+".lp."+
                            scenario_tree._scenarios[0]._name))+
        ' '+lp_filename)
    assert not rc

    #
    # Select one of the per-scenario .tim files as the master
    # and verify that they all match
    #
    tim_filename = os.path.join(output_directory, basename+".tim")
    rc = os.system(
        'cp '+os.path.join(scenario_directory,
                           (basename+".tim."+
                            scenario_tree._scenarios[0]._name))+
        ' '+tim_filename)
    assert not rc
    for scenario in scenario_tree._scenarios:
        scenario_tim_filename = \
            os.path.join(scenario_directory,
                         basename+".tim."+scenario._name)
        rc = os.system('diff -q '+scenario_tim_filename+' '+
                       tim_filename)
        assert not rc

    #
    # Merge per-scenario the .sto files into one
    #
    sto_filename = os.path.join(output_directory, basename+".sto")
    with open(sto_filename, 'w') as f:
        f.write('STOCH '+basename+'\n')
        f.write('BLOCKS DISCRETE REPLACE\n')
    for scenario in scenario_tree._scenarios:
        scenario_sto_filename = \
            os.path.join(scenario_directory,
                         basename+".sto."+scenario._name)
        assert os.path.exists(scenario_sto_filename)
        rc = os.system(
            'cat '+scenario_sto_filename+" >> "+sto_filename)
        assert not rc
    with open(sto_filename, 'a+') as f:
        # make sure we are at the end of the file
        f.seek(0,2)
        f.write('ENDATA\n')

    if not keep_scenario_files:
        shutil.rmtree(scenario_directory, ignore_errors=True)

    print("Output saved to: "+output_directory)

def convert_implicit(output_directory,
                     basename,
                     scenario_instance_factory,
                     io_options=None,
                     keep_scenario_files=False):
    raise NotImplementedError("This functionality has not been fully implemented")
    """
    import pyomo.environ
    import pyomo.repn.plugins.cpxlp

    if io_options is None:
        io_options = {}

    assert os.path.exists(output_directory)

    stochastic_parameters = \
        reference_model.find_component('PySP_StochasticParameters')
    if not isinstance(stochastic_parameters, Suffix):
        raise TypeError("Object with name 'PySP_StochasticParameters' was found "
                        "on model %s that is not of type 'Suffix'"
                        % (reference_model.cname(True)))
    for param_data in stochastic_parameters:
        if not param_data.parent_component()._mutable:
            raise RuntimeError("Stochastic parameters must be mutable")
        if value(param_data) == 0:
            raise RuntimeError("Stochastic parameters should be initialized "
                               "with nonzero values to avoid issues due to sparsity "
                               "of Pyomo expressions. Please update the value of %s"
                               % (param_data.cname(True)))

    scenario_tree = ScenarioTree(scenariotreeinstance=scenario_tree_model,
                                 scenariobundlelist=[reference_model.name])
    scenario_tree.linkInInstances({reference_model.name: reference_model})

    reference_scenario = scenario_tree.get_scenario(reference_model.name)

    with pyomo.repn.plugins.cpxlp.ProblemWriter_cpxlp() as writer:
        lp_filename = os.path.join(output_directory, basename+".setup.lp")
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        output_filename, LP_symbol_map = writer(reference_model,
                                                lp_filename,
                                                lambda x: True,
                                                io_options)
        assert output_filename == lp_filename

    StageToVariableMap = \
        map_variable_stages(reference_scenario, scenario_tree, LP_symbol_map)

    StageToConstraintMap = \
        map_constraint_stages(reference_scenario, scenario_tree, LP_symbol_map)

    assert len(scenario_tree._stages) == 2
    firststage = scenario_tree._stages[0]
    secondstage = scenario_tree._stages[1]

    #
    # Make sure the objective references all first stage variables.
    # We do this by directly modifying the canonical_repn of the objective
    # which the LP writer will reference next time we call it. In addition,
    # make that that the first second-stage variable in our column
    # ordering also appears in the objective so that ONE_VAR_CONSTANT
    # does not get identified as the first second-stage variable.
    # ** Just do NOT preprocess again until we call the LP writer **
    #
    obj_canonical_repn_flag = \
        getattr(reference_model, "_gen_obj_canonical_repn", None)
    reference_model._gen_obj_canonical_repn = False
    canonical_repn = reference_model._canonical_repn
    obj_repn = canonical_repn[reference_scenario._instance_objective]
    first_stage_varname_list = \
        [item[0] for item in StageToVariableMap[firststage._name]]
    if isinstance(obj_repn, LinearCanonicalRepn) and \
       (obj_repn.linear is not None):
        referenced_var_names = [LP_symbol_map.byObject[id(vardata)]
                                for vardata in obj_repn.variables]
        update_vars = []
        # add the first-stage variables (if not present)
        for LP_name in first_stage_varname_list:
            if LP_name not in referenced_var_names:
                update_vars.append(LP_symbol_map.bySymbol[LP_name])
        # add the first second-stage variable (if not present)
        if StageToVariableMap[secondstage._name][0][0] not in \
           referenced_var_names:
            update_vars.append(StageToVariableMap[secondstage._name][0][1])
        obj_repn.variables = list(obj_repn.variables) + \
                             update_vars
        obj_repn.linear = list(obj_repn.linear) + \
                          [0.0 for vardata in update_vars]
    else:
        raise RuntimeError("Unexpected objective representation")

    #
    # Create column (variable) ordering maps for LP files
    #
    column_order = ComponentMap()
    # first-stage variables
    for column_index, (LP_name, vardata, scenario_tree_id) \
          in enumerate(StageToVariableMap[firststage._name]):
        column_order[vardata] = column_index
    # second-stage variables
    for column_index, (LP_name, vardata, scenario_tree_id) \
        in enumerate(StageToVariableMap[secondstage._name],
                     len(column_order)):
        column_order[vardata] = column_index

    #
    # Create row (constraint) ordering maps for LP files
    #
    row_order = ComponentMap()
    # first-stage constraints
    for row_index, (LP_names, condata) \
          in enumerate(StageToConstraintMap[firststage._name]):
        row_order[condata] = row_index
    # second-stage constraints
    for row_index, (LP_names, condata) \
        in enumerate(StageToConstraintMap[secondstage._name],
                     len(row_order)):
        row_order[condata] = row_index

    #
    # Write the ordered LP file
    #
    LP_symbol_map = None
    with pyomo.repn.plugins.cpxlp.ProblemWriter_cpxlp() as writer:
        lp_filename = os.path.join(output_directory, basename+".lp")
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        io_options = dict(io_options)
        io_options['column_order'] = column_order
        io_options['row_order'] = row_order
        output_filename, LP_symbol_map = writer(reference_model,
                                                lp_filename,
                                                lambda x: True,
                                                io_options)
        assert output_filename == lp_filename

    # Restore this PySP hack to its original value
    if obj_canonical_repn_flag is None:
        delattr(reference_model, "_gen_obj_canonical_repn")
    else:
        setattr(reference_model,
                "_gen_obj_canonical_repn",
                obj_canonical_repn_flag)

    #
    # Write the .tim file
    #
    with open(os.path.join(output_directory, basename+".tim"),'w') as f:
        f.write('TIME\n')
        f.write('PERIODS\tLP\n')
        f.write('\t'+str(StageToVariableMap[firststage._name][0][0])+
                '\t'+str(LP_symbol_map.byObject[id(reference_scenario._instance_objective)])+
                '\tTIME1\n')
        LP_names = StageToConstraintMap[secondstage._name][0][0]
        if len(LP_names) == 1:
            # equality constraint
            assert (LP_names[0].startswith('c_e_') or \
                    LP_names[0].startswith('c_l_') or \
                    LP_names[0].startswith('c_u_'))
            stage2_row_start = LP_names[0]
        else:
            # range constraint (assumed the LP writer outputs
            # the lower range constraint first)
            LP_names = sorted(LP_names)
            assert (LP_names[0].startswith('r_l_') or \
                    LP_names[0].startswith('r_u_'))
            stage2_row_start = LP_names[0]

        f.write('\t'+str(StageToVariableMap[secondstage._name][0][0])+
                '\t'+stage2_row_start+
                '\tTIME2\n')
        f.write('ENDATA\n')


    #
    # Write the (INDEP) .sto file
    #
    stochastic_rhs = reference_model.find_component('PySP_StochasticRHS')
    if (stochastic_rhs is not None) and \
       (not isinstance(stochastic_rhs, Suffix)):
        raise TypeError("Object with name 'PySP_StochasticRHS' was found "
                        "on model %s that is not of type 'Suffix'"
                        % (reference_model.cname(True)))
    stochastic_matrix = reference_model.find_component('PySP_StochasticMatrix')
    if (stochastic_matrix is not None) and \
       (not isinstance(stochastic_matrix, Suffix)):
        raise TypeError("Object with name 'PySP_StochasticMatrix' was found "
                        "on model %s that is not of type 'Suffix'"
                        % (reference_model.cname(True)))
    stochastic_objective = reference_model.find_component('PySP_StochasticObjective')
    if (stochastic_objective is not None) and \
       (not isinstance(stochastic_objective, Suffix)):
        raise TypeError("Object with name 'PySP_StochasticObjective' was found "
                        "on model %s that is not of type 'Suffix'"
                        % (reference_model.cname(True)))
    stochastic_bounds = reference_model.find_component('PySP_StochasticVariableBounds')
    if (stochastic_bounds is not None) and \
       (not isinstance(stochastic_bounds, Suffix)):
        raise TypeError("Object with name 'PySP_StochasticVariableBounds' was found "
                        "on model %s that is not of type 'Suffix'"
                        % (reference_model.cname(True)))

    if (stochastic_bounds is not None):
        # TODO
        assert False

    constraint_LP_names = ComponentMap()
    for stage_name in StageToConstraintMap:
        for LP_names, constraint_data in StageToConstraintMap[stage_name]:
            constraint_LP_names[constraint_data] = LP_names

    with open(os.path.join(output_directory, basename+".sto"),'w') as f:
        f.write('STOCH '+basename+'\n')
        f.write('INDEP              DISCRETE\n')

        #
        # Stochastic RHS
        #
        rhs_template = "    RHS    %s    %.17g    %.17g\n"
        if stochastic_rhs is not None:
            for constraint_data in stochastic_rhs:
                assert isinstance(constraint_data, _ConstraintData)
                param_data = stochastic_rhs[constraint_data]
                assert isinstance(param_data, _ParamData)
                constraint_repn = generate_canonical_repn(constraint_data.body,
                                                          compute_values=False)
                assert isinstance(constraint_repn, LinearCanonicalRepn)
                body_constant = constraint_repn.constant
                if body_constant is None:
                    body_constant = 0.0
                #
                # **NOTE: In the code that follows we assume the LP writer
                #         always moves constraint body constants to the rhs
                #
                LP_names = constraint_LP_names[constraint_data]
                if constraint_data.equality:
                    assert len(LP_names) == 1
                    con_label = LP_names[0]
                    assert con_label.startswith('c_e_')
                    # equality constraint
                    for param_value, probability in \
                          reference_model.PySP_StochasticParameters[param_data]:
                        param_data.value = param_value
                        f.write(rhs_template %
                                (con_label,
                                 value(constraint_data.lower) - value(body_constant),
                                 probability))

                elif ((constraint_data.lower is not None) and \
                      (constraint_data.upper is not None)):
                    # range constraint
                    assert len(LP_names) == 2
                    for con_label in LP_names:
                        if con_label.startswith('r_l_'):
                            # lower bound
                            for param_value, probability in \
                                  reference_model.PySP_StochasticParameters[param_data]:
                                param_data.value = param_value
                                f.write(rhs_template %
                                        (con_label,
                                         value(constraint_data.lower) - value(body_constant),
                                         probability))
                        else:
                            assert con_label.startswith('r_u_')
                            # upper_bound
                            for param_value, probability in \
                                  reference_model.PySP_StochasticParameters[param_data]:
                                param_data.value = param_value
                                f.write(rhs_template %
                                        (con_label,
                                         value(constraint_data.upper) - value(body_constant),
                                         param_value-body_constant,
                                         probability))

                elif constraint_data.lower is not None:
                    # lower bound
                    assert len(LP_names) == 1
                    con_label = LP_names[0]
                    assert con_label.startswith('c_l_')
                    for param_value, probability in \
                          reference_model.PySP_StochasticParameters[param_data]:
                        param_data.value = param_value
                        f.write(rhs_template %
                                (con_label,
                                 value(constraint_data.lower) - value(body_constant),
                                 probability))

                else:
                    # upper bound
                    assert constraint_data.upper is not None
                    assert len(LP_names) == 1
                    con_label = LP_names[0]
                    assert con_label.startswith('c_u_')
                    for param_value, probability in \
                          reference_model.PySP_StochasticParameters[param_data]:
                        param_data.value = param_value
                        f.write(rhs_template %
                                (con_label,
                                 value(constraint_data.upper) - value(body_constant),
                                 probability))

        #
        # Stochastic Matrix
        #
        matrix_template = "    %s    %s    %.17g    %.17g\n"
        if stochastic_matrix is not None:
            for constraint_data in stochastic_matrix:
                assert isinstance(constraint_data, _ConstraintData)
                # With the compute_values=False flag we should be able to update
                # stochastic parameter to obtain the new variable coefficent in the
                # constraint expression (while implicitly accounting for any other
                # constant terms that might be grouped into the coefficient)
                constraint_repn = generate_canonical_repn(constraint_data.body,
                                                          compute_values=False)
                assert isinstance(constraint_repn, LinearCanonicalRepn)
                assert len(constraint_repn.variables) > 0
                for var_data, param_data in stochastic_matrix[constraint_data]:
                    assert isinstance(var_data, _VarData)
                    assert isinstance(param_data, _ParamData)
                    var_coef = None
                    for var, coef in zip(constraint_repn.variables,
                                         constraint_repn.linear):
                        if var is var_data:
                            var_coef = coef
                            break
                    assert var_coef is not None
                    var_label = LP_symbol_map.byObject[id(var_data)]
                    LP_names = constraint_LP_names[constraint_data]
                    if len(LP_names) == 1:
                        assert (LP_names[0].startswith('c_e_') or \
                                LP_names[0].startswith('c_l_') or \
                                LP_names[0].startswith('c_u_'))
                        con_label = LP_names[0]
                        for param_value, probability in \
                              reference_model.PySP_StochasticParameters[param_data]:
                            param_data.value = param_value
                            f.write(matrix_template % (var_label,
                                                       con_label,
                                                       value(var_coef),
                                                       probability))
                    else:
                        # range constraint
                        for con_label in LP_names:
                            if con_label.startswith('r_l_'):
                                # lower bound
                                for param_value, probability in \
                                      reference_model.PySP_StochasticParameters[param_data]:
                                    param_data.value = param_value
                                    f.write(matrix_template % (var_label,
                                                               con_label,
                                                               value(var_coef),
                                                               probability))
                            else:
                                assert con_label.startswith('r_u_')
                                # upper_bound
                                for param_value, probability in \
                                      reference_model.PySP_StochasticParameters[param_data]:
                                    param_data.value = param_value
                                    f.write(matrix_template % (var_label,
                                                               con_label,
                                                               value(var_coef),
                                                               probability))

        #
        # Stochastic Objective
        #
        obj_template = "    %s    %s    %.17g    %.17g\n"
        if stochastic_objective is not None:
            assert len(stochastic_objective) == 1
            objective_data = stochastic_objective.keys()[0]
            assert objective_data == reference_model._instance_objective
            assert isinstance(objective_data, _ObjectiveData)
            # With the compute_values=False flag we should be able to update
            # stochastic parameter to obtain the new variable coefficent in the
            # objective expression (while implicitly accounting for any other
            # constant terms that might be grouped into the coefficient)
            objective_repn = generate_canonical_repn(objective_data.expr,
                                                     compute_values=False)
            assert isinstance(objective_repn, LinearCanonicalRepn)
            assert len(objective_repn.variables) > 0
            for var_data, param_data in stochastic_objective:
                assert isinstance(var_data, _VarData)
                assert isinstance(param_data, _ParamData)

                var_coef = None
                for var, coef in zip(objective_repn.variables, objective_repn.linear):
                    if var is var_data:
                        var_coef = coef
                        break
                assert var_coef is not None
                var_label = LP_symbol_map.byObject[id(var_data)]
                obj_label = LP_symbol_map.byObject[id(objective)]
                for param_value, probability in \
                      reference_model.PySP_StochasticParameters[param_data]:
                    param_data.value = param_value
                    f.write(obj_template % (var_label,
                                            obj_label,
                                            value(var_coef),
                                            probability))
        f.write('ENDATA\n')

        for scenario in scenario_tree.scenarios:

            #
            # Stochastic RHS
            #
            rhs_template = "    RHS    %s    %.17g    %.17g\n"
            if stochastic_rhs is not None:
                for constraint_data in stochastic_rhs:
                    assert isinstance(constraint_data, _ConstraintData)
                    param_data = stochastic_rhs[constraint_data]
                    assert isinstance(param_data, _ParamData)
                    constraint_repn = generate_canonical_repn(constraint_data.body,
                                                              compute_values=False)
                    assert isinstance(constraint_repn, LinearCanonicalRepn)
                    body_constant = constraint_repn.constant
                    if body_constant is None:
                        body_constant = 0.0
                    #
                    # **NOTE: In the code that follows we assume the LP writer
                    #         always moves constraint body constants to the rhs
                    #
                    LP_names = constraint_LP_names[constraint_data]
                    if constraint_data.equality:
                        assert len(LP_names) == 1
                        con_label = LP_names[0]
                        assert con_label.startswith('c_e_')
                        # equality constraint
                        for param_value, probability in \
                              reference_model.PySP_StochasticParameters[param_data]:
                            param_data.value = param_value
                            f.write(rhs_template %
                                    (con_label,
                                     value(constraint_data.lower) - value(body_constant),
                                     probability))

                    elif ((constraint_data.lower is not None) and \
                          (constraint_data.upper is not None)):
                        # range constraint
                        assert len(LP_names) == 2
                        for con_label in LP_names:
                            if con_label.startswith('r_l_'):
                                # lower bound
                                for param_value, probability in \
                                      reference_model.PySP_StochasticParameters[param_data]:
                                    param_data.value = param_value
                                    f.write(rhs_template %
                                            (con_label,
                                             value(constraint_data.lower) - value(body_constant),
                                             probability))
                            else:
                                assert con_label.startswith('r_u_')
                                # upper_bound
                                for param_value, probability in \
                                      reference_model.PySP_StochasticParameters[param_data]:
                                    param_data.value = param_value
                                    f.write(rhs_template %
                                            (con_label,
                                             value(constraint_data.upper) - value(body_constant),
                                             param_value-body_constant,
                                             probability))

                    elif constraint_data.lower is not None:
                        # lower bound
                        assert len(LP_names) == 1
                        con_label = LP_names[0]
                        assert con_label.startswith('c_l_')
                        for param_value, probability in \
                              reference_model.PySP_StochasticParameters[param_data]:
                            param_data.value = param_value
                            f.write(rhs_template %
                                    (con_label,
                                     value(constraint_data.lower) - value(body_constant),
                                     probability))

                    else:
                        # upper bound
                        assert constraint_data.upper is not None
                        assert len(LP_names) == 1
                        con_label = LP_names[0]
                        assert con_label.startswith('c_u_')
                        for param_value, probability in \
                              reference_model.PySP_StochasticParameters[param_data]:
                            param_data.value = param_value
                            f.write(rhs_template %
                                    (con_label,
                                     value(constraint_data.upper) - value(body_constant),
                                     probability))

            #
            # Stochastic Matrix
            #
            matrix_template = "    %s    %s    %.17g    %.17g\n"
            if stochastic_matrix is not None:
                for constraint_data in stochastic_matrix:
                    assert isinstance(constraint_data, _ConstraintData)
                    # With the compute_values=False flag we should be able to update
                    # stochastic parameter to obtain the new variable coefficent in the
                    # constraint expression (while implicitly accounting for any other
                    # constant terms that might be grouped into the coefficient)
                    constraint_repn = generate_canonical_repn(constraint_data.body,
                                                              compute_values=False)
                    assert isinstance(constraint_repn, LinearCanonicalRepn)
                    assert len(constraint_repn.variables) > 0
                    for var_data, param_data in stochastic_matrix[constraint_data]:
                        assert isinstance(var_data, _VarData)
                        assert isinstance(param_data, _ParamData)
                        var_coef = None
                        for var, coef in zip(constraint_repn.variables,
                                             constraint_repn.linear):
                            if var is var_data:
                                var_coef = coef
                                break
                        assert var_coef is not None
                        var_label = LP_symbol_map.byObject[id(var_data)]
                        LP_names = constraint_LP_names[constraint_data]
                        if len(LP_names) == 1:
                            assert (LP_names[0].startswith('c_e_') or \
                                    LP_names[0].startswith('c_l_') or \
                                    LP_names[0].startswith('c_u_'))
                            con_label = LP_names[0]
                            for param_value, probability in \
                                  reference_model.PySP_StochasticParameters[param_data]:
                                param_data.value = param_value
                                f.write(matrix_template % (var_label,
                                                           con_label,
                                                           value(var_coef),
                                                           probability))
                        else:
                            # range constraint
                            for con_label in LP_names:
                                if con_label.startswith('r_l_'):
                                    # lower bound
                                    for param_value, probability in \
                                          reference_model.\
                                          PySP_StochasticParameters[param_data]:
                                        param_data.value = param_value
                                        f.write(matrix_template % (var_label,
                                                                   con_label,
                                                                   value(var_coef),
                                                                   probability))
                                else:
                                    assert con_label.startswith('r_u_')
                                    # upper_bound
                                    for param_value, probability in \
                                          reference_model.\
                                          PySP_StochasticParameters[param_data]:
                                        param_data.value = param_value
                                        f.write(matrix_template % (var_label,
                                                                   con_label,
                                                                   value(var_coef),
                                                                   probability))

            #
            # Stochastic Objective
            #
            obj_template = "    %s    %s    %.17g    %.17g\n"
            if stochastic_objective is not None:
                assert len(stochastic_objective) == 1
                objective_data = stochastic_objective.keys()[0]
                assert objective_data == reference_model._instance_objective
                assert isinstance(objective_data, _ObjectiveData)
                # With the compute_values=False flag we should be able to update
                # stochastic parameter to obtain the new variable coefficent in the
                # objective expression (while implicitly accounting for any other
                # constant terms that might be grouped into the coefficient)
                objective_repn = generate_canonical_repn(objective_data.expr,
                                                         compute_values=False)
                assert isinstance(objective_repn, LinearCanonicalRepn)
                assert len(objective_repn.variables) > 0
                for var_data, param_data in stochastic_objective:
                    assert isinstance(var_data, _VarData)
                    assert isinstance(param_data, _ParamData)

                    var_coef = None
                    for var, coef in zip(objective_repn.variables,
                                         objective_repn.linear):
                        if var is var_data:
                            var_coef = coef
                            break
                    assert var_coef is not None
                    var_label = LP_symbol_map.byObject[id(var_data)]
                    obj_label = LP_symbol_map.byObject[id(objective)]
                    for param_value, probability in \
                          reference_model.PySP_StochasticParameters[param_data]:
                        param_data.value = param_value
                        f.write(obj_template % (var_label,
                                                obj_label,
                                                value(var_coef),
                                                probability))

    print("Output saved to: "+output_directory)
    """
