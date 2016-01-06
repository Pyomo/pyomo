#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import operator
import shutil
import filecmp

thisfile = os.path.abspath(__file__)
thisfile.replace(".pyc","").replace(".py","")

from pyomo.core.base.numvalue import value
from pyomo.core.base.block import (Block,
                                   _BlockData,
                                   SortComponents)
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.expression import Expression
from pyomo.core.base.objective import Objective, _ObjectiveData
from pyomo.core.base.constraint import Constraint, _ConstraintData
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.param import _ParamData
from pyomo.core.base.piecewise import Piecewise, _PiecewiseData
from pyomo.core.base.suffix import ComponentMap
from pyomo.repn import LinearCanonicalRepn
from pyomo.repn import generate_canonical_repn
from pyomo.pysp.scenariotree.tree_structure import ScenarioTree
from pyomo.pysp.scenariotree.manager import InvocationType
from pyomo.pysp.annotations import (locate_annotations,
                                    PySP_ConstraintStageAnnotation,
                                    PySP_StochasticRHSAnnotation,
                                    PySP_StochasticMatrixAnnotation,
                                    PySP_StochasticObjectiveAnnotation)

from six import iteritems, itervalues

# TODO:
#  - Fix stochastic objective section
#  - Implicit output
#  - check for variable first- / second-stageness in the different
#    annotations and report errors if appropriate.
#  - Count stochastic entries in matrix locations and report
#  - Handle cases of a variable not appearing in the
#    constraint expression but appear in the list of
#    variables with stochastic coefficients assigned to the
#    constraint using the suffix (an option or another
#    suffix to allow implicit zero coefficients in this case)
#  - Report max variable/constraint name length
#  - A better way to handle piecewise blocks would be to
#    allow users to declare stage variables on the scenario
#    tree by block name (see ticket)
#  - Generate .row, .col, and nonzero structure file for more robust
#    cross-scenario problem validation

# LONG TERM TODO:
#  - Write .cor file?
#  - Multi-stage?
#  - Quadratic constraints and objectives?
#     - For variables with both linear and quadratic terms, how
#       to distinguish between the two with model annotations?

def _safe_remove_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def _expand_annotation_entries(scenario,
                               ctype,
                               annotation,
                               check_value=None):
    if ctype is Constraint:
        ctype_data = _ConstraintData
    else:
        assert ctype is Objective
        ctype_data = _ObjectiveData

    annotation_data = annotation.data
    items = []
    component_ids = set()
    def _append(component, val):
        items.append((component, val))
        if id(component) in component_ids:
            raise RuntimeError(
                "(Scenario=%s): Component %s was assigned multiple declarations "
                "in annotation type %s. To correct this issue, ensure that "
                "multiple container components under which the component might "
                "be stored (such as a Block and an indexed Constraint) are not "
                "simultaneously set in this annotation." % (scenario.name,
                                                            component.cname(True),
                                                            annotation.__name__))
        component_ids.add(component)

    for component in annotation_data:
        if isinstance(component, ctype_data):
            if component.active:
                _append(component, annotation_data[component])
        elif isinstance(component, ctype):
            for index in component:
                obj = component[index]
                if obj.active:
                    _append(obj, annotation_data[component])
        elif isinstance(component, _BlockData):
            if component.active:
                for obj in component.component_data_objects(
                        ctype,
                        active=True,
                        descend_into=True):
                    _append(obj, annotation_data[component])
        elif isinstance(component, Block):
            for index in component:
                block = component[index]
                if block.active:
                    for obj in block.component_data_objects(
                            ctype,
                            active=True,
                            descend_into=True):
                        _append(obj, annotation_data[component])
        else:
            raise TypeError(
                "(Scenario=%s): Declarations in annotation type %s must be of type "
                "%s or Block. Invalid type: %s" % (scenario.name,
                                                   annotation.__name__,
                                                   ctype.__name__,
                                                   type(component)))
        if check_value is not None:
            check_value(component, annotation_data[component])

    return items

def map_constraint_stages(scenario,
                          scenario_tree,
                          LP_symbol_map,
                          constraint_stage_assignments=None):

    reference_model = scenario._instance

    assert len(scenario_tree.stages) == 2
    firststage = scenario_tree.stages[0]
    secondstage = scenario_tree.stages[1]

    StageToConstraintMap = {}
    StageToConstraintMap[firststage.name] = []
    StageToConstraintMap[secondstage.name] = []

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

        for constraint_data in block.component_data_objects(
                SOSConstraint,
                active=True,
                descend_into=False):
            raise TypeError("(Scenario=%s): SOSConstraints are not allowed with the "
                            "SMPS format. Invalid constraint: %s"
                            % (scenario.name, constraint_data.cname(True)))

        block_canonical_repn = getattr(block, "_canonical_repn", None)
        if block_canonical_repn is None:
            raise ValueError(
                "Unable to find _canonical_repn ComponentMap "
                "on block %s" % (block.cname(True)))

        piecewise_stage = None
        if isinstance(block, (Piecewise, _PiecewiseData)):
            piecewise_stage = firststage
            for vardata in block.referenced_variables():
                variable_node = \
                    scenario.variableNode(vardata,
                                          reference_model)
                if variable_node.stage == secondstage:
                    piecewise_stage = secondstage
                else:
                    assert variable_node.stage == firststage

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
                constraint_stage = constraint_node.stage
            else:
                constraint_stage = piecewise_stage

            if constraint_stage_assignments is not None:
                assigned_constraint_stage_id = constraint_stage_assignments.get(constraint_data, None)
                if assigned_constraint_stage_id is None:
                    raise ValueError(
                        "(Scenario=%s): The %s annotation type was found on the model but "
                        "no stage declaration was provided for constraint %s."
                        % (scenario.name,
                           PySP_ConstraintStageAnnotation.__name__,
                           constraint_data.cname(True)))
                elif assigned_constraint_stage_id == 1:
                    if constraint_stage is secondstage:
                        raise RuntimeError(
                            "(Scenario=%s): The %s annotation declared constraint %s as first-stage, "
                            "but this constraint contains references to second-stage variables."
                            % (scenario.name,
                               PySP_ConstraintStageAnnotation.__name__,
                               constraint_data.cname(True)))
                elif assigned_constraint_stage_id == 2:
                    # override the inferred stage-ness (whether or not it was first- or second-stage)
                    constraint_stage = secondstage
                else:
                    raise ValueError(
                        "(Scenario=%s): The %s annotation was declared with an invalid value (%s) "
                        "for constraint %s. Valid values are 1 or 2."
                        % (scenario.name,
                           PySP_ConstraintStageAnnotation.__name__,
                           assigned_constraint_stage_id,
                           constraint_data.cname(True)))

            StageToConstraintMap[constraint_stage.name].\
                append((LP_aliases, constraint_data))

    assert sorted(StageToConstraintMap.keys()) == \
        sorted([firststage.name, secondstage.name])

    # sort each by name
    for key in StageToConstraintMap:
        StageToConstraintMap[key].sort(key=operator.itemgetter(0))

    return StageToConstraintMap

def map_variable_stages(scenario, scenario_tree, LP_symbol_map):

    reference_model = scenario._instance

    FirstStageVars = {}
    SecondStageVars = {}

    all_vars_cnt = 0
    piecewise_blocks = []
    for block in reference_model.block_data_objects(
            active=True,
            descend_into=True):
        all_vars_cnt += len(list(block.component_data_objects
                                 (Var, descend_into=False)))
        if isinstance(block, (Piecewise, _PiecewiseData)):
            piecewise_blocks.append(block)

    rootnode = scenario_tree.findRootNode()
    assert len(scenario_tree.stages) == 2
    stageone = scenario_tree.stages[0]
    stagetwo = scenario_tree.stages[1]
    stagetwo_node = scenario.node_list[-1]
    assert stagetwo_node.stage is stagetwo
    firststage_blended_variables = rootnode._standard_variable_ids
    LP_byObject = LP_symbol_map.byObject
    all_vars_on_tree = []

    for scenario_tree_id, vardata in \
          iteritems(reference_model.\
          _ScenarioTreeSymbolMap.bySymbol):
        if vardata.is_expression():
            continue
        try:
            LP_name = LP_byObject[id(vardata)]
        except KeyError:
            raise ValueError("(Scenario=%s): Variable with name '%s' was declared "
                             "on the scenario tree but did not appear "
                             "in the reference scenario LP file."
                             % (scenario.name, vardata.cname(True)))
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

    for stage in scenario_tree.stages:
        cost_variable_name, cost_variable_index = \
            stage._cost_variable
        stage_cost_component = \
            reference_model.\
            find_component(cost_variable_name)
        if stage_cost_component.type() is not Expression:
            raise RuntimeError(
                "(Scenario=%s): All StageCost objects must be declared "
                "as Expression components when using this tool"
                % (scenario.name))

    # The *ONLY* case where we allow variables to exist on the
    # model that were not declared on the scenario tree is when
    # they are autogenerated by a Piecewise component

    # For now we just assume all auxiliary Piecewise variables
    # are SecondStage
    for block in piecewise_blocks:
        for vardata in block.component_data_objects(Var,
                                                    descend_into=False):
            LP_name = LP_byObject[id(vardata)]
            SecondStageVars[LP_name] = (vardata, scenario_tree_id)
            all_vars_on_tree.append(LP_name)

    # Make sure every variable on the model has been
    # declared on the scenario tree
    if len(all_vars_on_tree) != all_vars_cnt:
        print("**** THERE IS A PROBLEM ****")
        print("Not all model variables are on the "
              "scenario tree. Investigating...")
        all_vars = set()
        tmp_buffer = {}
        for block in reference_model.block_data_objects(
                active=True,
                descend_into=True):
            all_vars.update(
                vardata.cname(True, tmp_buffer) \
                for vardata in block.component_data_objects
                (Var, descend_into=False))

        tree_vars = set()
        for scenario_tree_id, vardata in \
            iteritems(reference_model.\
                      _ScenarioTreeSymbolMap.bySymbol):
            tree_vars.add(vardata.cname(True, tmp_buffer))
        cost_vars = set()
        for stage in scenario_tree.stages:
            cost_variable_name, cost_variable_index = \
                stage._cost_variable
            stage_cost_component = \
                reference_model.\
                find_component(cost_variable_name)
            if stage_cost_component.type() is not Expression:
                cost_vars.add(
                    stage_cost_component[cost_variable_index].\
                    cname(True, tmp_buffer))

        print("Number of Scenario Tree Variables "
              "(found in LP file): "+str(len(tree_vars)))
        print("Number of Scenario Tree Cost Variables "
               "(found in LP file): "+str(len(cost_vars)))
        print("Number of Variables Found on Model: "
              +str(len(all_vars)))
        print("Variables Missing from Scenario Tree "
              "(or LP file):"+str(all_vars-tree_vars-cost_vars))
        raise RuntimeError("(Scenario=%s): Failed verify that all model variables "
                           "have been declared on the scenario tree"
                           % (scenario.name))

    # A necessary but not sufficient sanity check to make sure the
    # second stage variable sets are the same for all
    # scenarios. This is not required by pysp, but I think this
    # assumption is made in the rest of the code here
    for tree_node in stagetwo._tree_nodes:
        assert len(stagetwo_node._variable_ids) == \
            len(tree_node._variable_ids)

    assert len(scenario_tree.stages) == 2

    StageToVariableMap = {}
    StageToVariableMap[stageone.name] = \
        [(LP_name,
          FirstStageVars[LP_name][0],
          FirstStageVars[LP_name][1])
         for LP_name in sorted(FirstStageVars)]
    StageToVariableMap[stagetwo.name] = \
        [(LP_name,
          SecondStageVars[LP_name][0],
          SecondStageVars[LP_name][1])
         for LP_name in sorted(SecondStageVars)]

    return StageToVariableMap

def _convert_explicit_setup(worker,
                            scenario,
                            output_directory,
                            basename,
                            io_options):
    import pyomo.environ
    import pyomo.repn.plugins.cpxlp
    assert os.path.exists(output_directory)

    scenario_tree = worker.scenario_tree
    reference_model = scenario._instance

    #
    # Check for model annotations
    #

    constraint_stage_annotation = locate_annotations(reference_model,
                                                     PySP_ConstraintStageAnnotation,
                                                     max_allowed=1)
    if len(constraint_stage_annotation) == 0:
        constraint_stage_annotation = None
    else:
        assert len(constraint_stage_annotation) == 1
        constraint_stage_annotation = constraint_stage_annotation[0][1]
    constraint_stage_assignments = None
    if constraint_stage_annotation is not None:
        constraint_stage_assignments = ComponentMap(
            _expand_annotation_entries(
                scenario,
                Constraint,
                constraint_stage_annotation,
                check_value=None))

    stochastic_rhs = locate_annotations(reference_model,
                                        PySP_StochasticRHSAnnotation,
                                        max_allowed=1)
    if len(stochastic_rhs) == 0:
        stochastic_rhs = None
    else:
        assert len(stochastic_rhs) == 1
        stochastic_rhs = stochastic_rhs[0][1]

    stochastic_matrix = locate_annotations(reference_model,
                                           PySP_StochasticMatrixAnnotation,
                                           max_allowed=1)
    if len(stochastic_matrix) == 0:
        stochastic_matrix = None
    else:
        assert len(stochastic_matrix) == 1
        stochastic_matrix = stochastic_matrix[0][1]

    stochastic_objective = locate_annotations(reference_model,
                                              PySP_StochasticObjectiveAnnotation,
                                              max_allowed=1)
    if len(stochastic_objective) == 0:
        stochastic_objective = None
    else:
        assert len(stochastic_objective) == 1
        stochastic_objective = stochastic_objective[0][1]

    if (stochastic_rhs is None) and \
       (stochastic_matrix is None) and \
       (stochastic_objective is None):
        raise RuntimeError("(Scenario=%s): No stochastic annotations found. SMPS "
                           "conversion requires at least one of the following "
                           "annotation types:\n - %s\n - %s\n - %s"
                           % (scenario.name,
                              PySP_StochasticRHSAnnotation.__name__,
                              PySP_StochasticMatrixAnnotation.__name__,
                              PySP_StochasticObjectiveAnnotation.__name__))

    #
    # Write the LP file once to obtain the symbol map
    #
    with pyomo.repn.plugins.cpxlp.ProblemWriter_cpxlp() as writer:
        lp_filename = os.path.join(output_directory,
                                   basename+".setup.lp."+scenario.name)
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
                              LP_symbol_map,
                              constraint_stage_assignments=constraint_stage_assignments)

    assert len(scenario_tree.stages) == 2
    firststage = scenario_tree.stages[0]
    secondstage = scenario_tree.stages[1]

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
        [item[0] for item in StageToVariableMap[firststage.name]]

    if isinstance(obj_repn, LinearCanonicalRepn) and \
       (obj_repn.linear is not None):
        referenced_var_names = set([LP_symbol_map.byObject[id(vardata)]
                                    for vardata in obj_repn.variables])
        obj_vars = list(obj_repn.variables)
        obj_coefs = list(obj_repn.linear)
        # add the first-stage variables (if not present)
        for LP_name in first_stage_varname_list:
            if LP_name not in referenced_var_names:
                obj_vars.append(LP_symbol_map.bySymbol[LP_name]())
                obj_coefs.append(0.0)
        # add the first second-stage variable (if not present)
        if StageToVariableMap[secondstage.name][0][0] not in \
           referenced_var_names:
            obj_vars.append(StageToVariableMap[secondstage.name][0][1])
            obj_coefs.append(0.0)
        obj_repn.variables = tuple(obj_vars)
        obj_repn.linear = tuple(obj_coefs)

    else:
        raise RuntimeError("(Scenario=%s): A linear objective is required for "
                           "conversion to SMPS format."
                           % (scenario.name))

    #
    # Create column (variable) ordering maps for LP files
    #
    column_order = ComponentMap()
    # first-stage variables
    for column_index, (LP_name, vardata, scenario_tree_id) \
        in enumerate(StageToVariableMap[firststage.name]):
        column_order[vardata] = column_index
    # second-stage variables
    for column_index, (LP_name, vardata, scenario_tree_id) \
        in enumerate(StageToVariableMap[secondstage.name],
                     len(column_order)):
        column_order[vardata] = column_index

    #
    # Create row (constraint) ordering maps for LP files
    #
    row_order = ComponentMap()
    # first-stage constraints
    for row_index, (LP_names, condata) \
        in enumerate(StageToConstraintMap[firststage.name]):
        row_order[condata] = row_index
    # second-stage constraints
    for row_index, (LP_names, condata) \
        in enumerate(StageToConstraintMap[secondstage.name],
                     len(row_order)):
        row_order[condata] = row_index

    #
    # Write the ordered LP file
    #
    with pyomo.repn.plugins.cpxlp.ProblemWriter_cpxlp() as writer:
        lp_filename = os.path.join(output_directory,
                                   basename+".lp."+scenario.name)
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        io_options = dict(io_options)
        io_options['column_order'] = column_order
        io_options['row_order'] = row_order
        io_options['force_objective_constant'] = True
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

    # re-generate these maps as the LP symbol map
    # is likely different
    StageToVariableMap = \
        map_variable_stages(scenario,
                            scenario_tree,
                            LP_symbol_map)

    StageToConstraintMap = \
        map_constraint_stages(scenario,
                              scenario_tree,
                              LP_symbol_map,
                              constraint_stage_assignments=constraint_stage_assignments)

    # generate a few data structures that are used
    # when writing the .sto file
    constraint_LP_names = ComponentMap(
        (constraint_data, LP_names) for stage_name in StageToConstraintMap
        for LP_names, constraint_data in StageToConstraintMap[stage_name])
    secondstage_constraint_ids = \
        set(id(constraint_data)
            for LP_names, constraint_data in StageToConstraintMap[secondstage.name])

    #
    # Write the explicit column ordering (variables) used
    # for the ordered LP file
    #
    with open(os.path.join(output_directory,
                           basename+".lp.col."+scenario.name),'w') as f_col:
        # first-stage variables
        for (LP_name, _, _) in StageToVariableMap[firststage.name]:
            f_col.write(LP_name+"\n")
        # second-stage variables
        for (LP_name, _, _) in StageToVariableMap[secondstage.name]:
            f_col.write(LP_name+"\n")

    #
    # Write the explicit row ordering (constraints) used
    # for the ordered LP file
    #
    with open(os.path.join(output_directory,
                           basename+".lp.row."+scenario.name),'w') as f_row:
        # the objective is always the first row in SMPS format
        f_row.write(LP_symbol_map.byObject[id(scenario._instance_objective)]+"\n")
        # first-stage constraints
        for (LP_names, _) in StageToConstraintMap[firststage.name]:
            # because range constraints are split into two
            # constraints (hopefully our ordering of the r_l_
            # and r_u_ forms is the same as the LP file!)
            for LP_name in LP_names:
                f_row.write(LP_name+"\n")
        # second-stage constraints
        for (LP_names, _) in StageToConstraintMap[secondstage.name]:
            # because range constraints are split into two
            # constraints (hopefully our ordering of the r_l_
            # and r_u_ forms is the same as the LP file!)
            for LP_name in LP_names:
                f_row.write(LP_name+"\n")

    #
    # Write the .tim file
    #
    with open(os.path.join(output_directory,
                           basename+".tim."+scenario.name),'w') as f_tim:
        f_tim.write('TIME '+basename+'\n')
        f_tim.write('PERIODS\tIMPLICIT\n')
        f_tim.write('\t%s' % (StageToVariableMap[firststage.name][0][0]))
        f_tim.write('\t%s\tTIME1\n'
                    % (LP_symbol_map.byObject[id(scenario._instance_objective)]))
        LP_names = StageToConstraintMap[secondstage.name][0][0]
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
        f_tim.write('\t%s' % (StageToVariableMap[secondstage.name][0][0]))
        f_tim.write('\t%s\tTIME2\n' % (stage2_row_start))
        f_tim.write('ENDATA\n')

    canonical_repn_cache = ComponentMap()
    #
    # Write the body of the .sto file
    #
    with open(os.path.join(output_directory,
                           basename+".sto.struct."+scenario.name),'w') as f_coords:
        with open(os.path.join(output_directory,
                           basename+".sto."+scenario.name),'w') as f_sto:
            constraint_name_buffer = {}
            objective_name_buffer = {}
            variable_name_buffer = {}
            scenario_probability = scenario.probability
            f_sto.write(" BL BLOCK1 PERIOD2 %.17g\n" % (scenario_probability))

            #
            # Stochastic RHS
            #
            def _check_rhs_value(component, val):
                # IMPT: the check for 'is' must not be changed to '=='
                if not ((val is True) or
                        ((type(val) is tuple) and
                         (len(val) == 2) and
                         (val in ((True,False),
                                  (True,True),
                                  (False,True))))):
                    raise ValueError(
                        "(Scenario=%s): Entries in the %s annotation type must be "
                        "assigned the value True or a 2-tuple with at least "
                        "one of the entries set to True. Invalid value (%s) "
                        "declared for entry %s."
                        % (scenario.name,
                           PySP_StochasticRHSAnnotation.__name__,
                           val,
                           component.cname(True)))
            rhs_template = "    RHS    %s    %.17g\n"
            if stochastic_rhs is not None:
                empty_rhs_annotation = False
                if len(stochastic_rhs.data) > 0:
                    empty_rhs_annotation = False
                    sorted_values = _expand_annotation_entries(
                        scenario,
                        Constraint,
                        stochastic_rhs,
                        check_value=_check_rhs_value)
                    sorted_values.sort(key=lambda x: x[0].cname(True, constraint_name_buffer))
                    if len(sorted_values) == 0:
                        raise RuntimeError(
                            "(Scenario=%s): The %s annotation was declared "
                            "with explicit entries but no active Constraint "
                            "objects were recovered from those entries."
                            % (scenario.name,
                               PySP_StochasticRHSAnnotation.__name__))
                else:
                    empty_rhs_annotation = True
                    sorted_values = tuple((constraint_data, stochastic_rhs.default)
                                          for _,constraint_data
                                          in StageToConstraintMap[secondstage.name])

                for constraint_data, include_bound in sorted_values:
                    assert isinstance(constraint_data, _ConstraintData)
                    if not empty_rhs_annotation:
                        # verify that this constraint was
                        # flagged by PySP or the user as second-stage
                        if id(constraint_data) not in secondstage_constraint_ids:
                            raise RuntimeError(
                                "(Scenario=%s): The constraint %s has been declared "
                                "in the %s annotation but it was not identified as "
                                "a second-stage constraint. To correct this issue, "
                                "either remove the constraint from this annotation "
                                "or manually declare it as second-stage using the "
                                "%s annotation."
                                % (scenario.name,
                                   constraint_data.cname(True),
                                   PySP_StochasticRHSAnnotation.__name__,
                                   PySP_ConstraintStageAnnotation.__name__))

                    if constraint_data in canonical_repn_cache:
                        constraint_repn = canonical_repn_cache[constraint_data]
                    else:
                        constraint_repn = \
                            generate_canonical_repn(constraint_data.body)
                    if not isinstance(constraint_repn, LinearCanonicalRepn):
                        raise RuntimeError("(Scenario=%s): Only linear constraints are "
                                           "accepted for conversion to SMPS format. "
                                           "Constraint %s is not linear."
                                           % (scenario.name,
                                              constraint_data.cname(True)))

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
                            assert (include_bound is True) or \
                                   (include_bound[0] is True)
                            f_sto.write(rhs_template %
                                        (con_label,
                                         value(constraint_data.lower) - \
                                         value(body_constant)))
                            f_coords.write("RHS %s\n" % (con_label))
                        elif con_label.startswith('r_l_') :
                            if (include_bound is True) or \
                               (include_bound[0] is True):
                                f_sto.write(rhs_template %
                                            (con_label,
                                             value(constraint_data.lower) - \
                                             value(body_constant)))
                                f_coords.write("RHS %s\n" % (con_label))
                        elif con_label.startswith('c_u_'):
                            assert (include_bound is True) or \
                                   (include_bound[1] is True)
                            f_sto.write(rhs_template %
                                        (con_label,
                                         value(constraint_data.upper) - \
                                         value(body_constant)))
                            f_coords.write("RHS %s\n" % (con_label))
                        elif con_label.startswith('r_u_'):
                            if (include_bound is True) or \
                               (include_bound[1] is True):
                                f_sto.write(rhs_template %
                                            (con_label,
                                             value(constraint_data.upper) - \
                                             value(body_constant)))
                                f_coords.write("RHS %s\n" % (con_label))
                        else:
                            assert False

            #
            # Stochastic Matrix
            #
            matrix_template = "    %s    %s    %.17g\n"
            if stochastic_matrix is not None:
                empty_matrix_annotation = False
                if len(stochastic_matrix.data) > 0:
                    empty_matrix_annotation = False
                    sorted_values = _expand_annotation_entries(
                        scenario,
                        Constraint,
                        stochastic_matrix,
                        check_value=None)
                    sorted_values.sort(key=lambda x: x[0].cname(True, constraint_name_buffer))
                    if len(sorted_values) == 0:
                        raise RuntimeError(
                            "(Scenario=%s): The %s annotation was declared "
                            "with explicit entries but no active Constraint "
                            "objects were recovered from those entries."
                            % (scenario.name,
                               PySP_StochasticRHSAnnotation.__name__))
                else:
                    empty_matrix_annotation = True
                    sorted_values = tuple((constraint_data,stochastic_matrix.default)
                                          for _,constraint_data
                                          in StageToConstraintMap[secondstage.name])

                for constraint_data, var_list in sorted_values:
                    assert isinstance(constraint_data, _ConstraintData)
                    if not empty_matrix_annotation:
                        # verify that this constraint was
                        # flagged by PySP or the user as second-stage
                        if id(constraint_data) not in secondstage_constraint_ids:
                            raise RuntimeError(
                                "(Scenario=%s): The constraint %s has been declared "
                                "in the %s annotation but it was not identified as "
                                "a second-stage constraint. To correct this issue, "
                                "either remove the constraint from this annotation "
                                "or manually declare it as second-stage using the "
                                "%s annotation."
                                % (scenario.name,
                                   constraint_data.cname(True),
                                   PySP_StochasticMatrixAnnotation.__name__,
                                   PySP_ConstraintStageAnnotation.__name__))
                    if constraint_data in canonical_repn_cache:
                        constraint_repn = canonical_repn_cache[constraint_data]
                    else:
                        constraint_repn = \
                            generate_canonical_repn(constraint_data.body)
                    if not isinstance(constraint_repn, LinearCanonicalRepn):
                        raise RuntimeError("(Scenario=%s): Only linear constraints are "
                                           "accepted for conversion to SMPS format. "
                                           "Constraint %s is not linear."
                                           % (scenario.name,
                                              constraint_data.cname(True)))
                    assert len(constraint_repn.variables) > 0
                    if var_list is None:
                        var_list = constraint_repn.variables
                    for var_data in var_list:
                        assert isinstance(var_data, _VarData)
                        assert not var_data.fixed
                        var_coef = None
                        for var, coef in zip(constraint_repn.variables,
                                             constraint_repn.linear):
                            if var is var_data:
                                var_coef = coef
                                break
                        if var_coef is None:
                            raise RuntimeError(
                                "(Scenario=%s): The coefficient for variable %s has "
                                "been marked as stochastic in constraint %s using "
                                "the %s annotation, but the variable does not appear"
                                " in the canonical constraint expression."
                                % (scenario.name,
                                   var_data.cname(True),
                                   constraint_data.cname(True),
                                   PySP_StochasticMatrixAnnotation.__name__))
                        var_label = LP_symbol_map.byObject[id(var_data)]
                        LP_names = constraint_LP_names[constraint_data]
                        for con_label in LP_names:
                            f_sto.write(matrix_template % (var_label,
                                                           con_label,
                                                           value(var_coef)))
                            f_coords.write("%s %s\n" % (var_label, con_label))

            #
            # Stochastic Objective
            #
            obj_template = "    %s    %s    %.17g\n"
            if stochastic_objective is not None:
                if len(stochastic_objective.data) > 0:
                    # temporarily reactivate the original
                    # user objective object
                    if scenario._instance_original_objective_object is not None:
                        assert not scenario._instance_original_objective_object.active
                        scenario._instance_original_objective_object.activate()
                        assert scenario._instance_objective.active
                        scenario._instance_objective.deactivate()
                    try:
                        sorted_values = _expand_annotation_entries(
                            scenario,
                            Objective,
                            stochastic_objective,
                            check_value=None,
                            name_buffer=objective_name_buffer)
                    finally:
                        if scenario._instance_original_objective_object is not None:
                            scenario._instance_original_objective_object.deactivate()
                            scenario._instance_objective.activate()

                    assert len(sorted_values) <= 1
                    if len(sorted_values) == 0:
                        raise RuntimeError(
                            "(Scenario=%s): The %s annotation was declared "
                            "with explicit entries but no active Objective "
                            "objects were recovered from those entries."
                            % (scenario.name,
                               PySP_StochasticObjectiveAnnotation.__name__))
                    objective, (objective_variables, include_constant) = \
                        sorted_values[0]
                    assert objective is scenario._instance_original_objective_object
                    objective = scenario._instance_objective
                else:
                    objective = scenario._instance_objective
                    objective_variables, include_constant = stochastic_objective.default

                objective_repn = \
                    generate_canonical_repn(objective.expr)
                assert isinstance(objective_repn, LinearCanonicalRepn)
                if objective_variables is None:
                    objective_variables = objective_repn.variables
                obj_label = LP_symbol_map.byObject[id(objective)]
                for var_data in objective_variables:
                    assert isinstance(var_data, _VarData)
                    var_coef = None
                    for var, coef in zip(objective_repn.variables,
                                         objective_repn.linear):
                        if var is var_data:
                            var_coef = coef
                            break
                    if var_coef is None:
                        raise RuntimeError(
                            "(Scenario=%s): The coefficient for variable %s has "
                            "been marked as stochastic in objective %s using "
                            "the %s annotation, but the variable does not appear"
                            " in the canonical objective expression."
                            % (scenario.name,
                               var_data.cname(True),
                               objective_data.cname(True),
                               PySP_StochasticObjectiveAnnotation.__name__))
                    var_label = LP_symbol_map.byObject[id(var_data)]
                    f_sto.write(obj_template % (var_label,
                                                obj_label,
                                                value(var_coef)))
                    f_coords.write("%s %s\n" % (var_label, obj_label))
                if include_constant:
                    obj_constant = objective_repn.constant
                    if obj_constant is None:
                        obj_constant = 0.0
                    f_sto.write(obj_template % ("ONE_VAR_CONSTANT",
                                                obj_label,
                                                obj_constant))
                    f_coords.write("%s %s\n" % ("ONE_VAR_CONSTANT", obj_label))

def convert_explicit(output_directory,
                     basename,
                     scenario_tree_manager,
                     io_options=None,
                     disable_consistency_checks=False,
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

    scenario_directory = os.path.join(output_directory,
                                      'scenario_files')

    if not os.path.exists(scenario_directory):
        os.mkdir(scenario_directory)

    scenario_tree_manager.invoke_function(
        "_convert_explicit_setup",
        thisfile,
        invocation_type=InvocationType.PerScenario,
        function_args=(scenario_directory,
                       basename,
                       io_options))

    reference_scenario = scenario_tree.scenarios[0]
    reference_scenario_name = reference_scenario.name

    # TODO: Out of laziness we are making shell calls to
    #       tools like 'cp' and 'cat'. Update the code to
    #       work on Windows.

    #
    # Copy the reference scenarios .lp, .lp.row, .lp.col,
    # and .tim files to the output directory. The consistency
    # checks will verify that these files match across scenarios
    #
    lp_filename = os.path.join(output_directory, basename+".lp")
    _safe_remove_file(lp_filename)
    rc = os.system(
        'cp '+os.path.join(scenario_directory,
                           (basename+".lp."+
                            reference_scenario_name))+
        ' '+lp_filename)
    assert not rc

    lp_row_filename = os.path.join(output_directory, basename+".lp.row")
    _safe_remove_file(lp_row_filename)
    rc = os.system(
        'cp '+os.path.join(scenario_directory,
                           (basename+".lp.row."+
                            reference_scenario_name))+
        ' '+lp_row_filename)
    assert not rc

    lp_col_filename = os.path.join(output_directory, basename+".lp.col")
    _safe_remove_file(lp_col_filename)
    rc = os.system(
        'cp '+os.path.join(scenario_directory,
                           (basename+".lp.col."+
                            reference_scenario_name))+
        ' '+lp_col_filename)
    assert not rc

    tim_filename = os.path.join(output_directory, basename+".tim")
    _safe_remove_file(tim_filename)
    rc = os.system(
        'cp '+os.path.join(scenario_directory,
                           (basename+".tim."+
                            reference_scenario_name))+
        ' '+tim_filename)
    assert not rc

    sto_struct_filename = os.path.join(output_directory, basename+".sto.struct")
    _safe_remove_file(sto_struct_filename)
    rc = os.system(
        'cp '+os.path.join(scenario_directory,
                           (basename+".sto.struct."+
                            reference_scenario_name))+
        ' '+sto_struct_filename)
    assert not rc

    #
    # Merge the per-scenario .sto files into one
    #
    sto_filename = os.path.join(output_directory, basename+".sto")
    _safe_remove_file(sto_filename)
    with open(sto_filename, 'w') as f:
        f.write('STOCH '+basename+'\n')
        f.write('BLOCKS DISCRETE REPLACE\n')
    for scenario in scenario_tree.scenarios:
        scenario_sto_filename = \
            os.path.join(scenario_directory,
                         basename+".sto."+scenario.name)
        assert os.path.exists(scenario_sto_filename)
        rc = os.system(
            'cat '+scenario_sto_filename+" >> "+sto_filename)
        assert not rc
    with open(sto_filename, 'a+') as f:
        # make sure we are at the end of the file
        f.seek(0,2)
        f.write('ENDATA\n')

    print("\nSMPS conversion complete")
    print("Output saved to: "+output_directory)
    if not disable_consistency_checks:
        print("\nStarting scenario structure consistency checks across scenario "
              "files stored in %s" % (scenario_directory))
        print("This may take some time. If this test is prohibitively slow or can "
              "not be executed on your system, it can be disabled by activating the "
              "disable_consistency_check option.")
        has_diff = False
        try:
            if not os.system('diff --help > /dev/null'):
                has_diff = True
            else:
                has_diff = False
        except:
            has_diff = False

        print(" - Checking row and column ordering...")
        for scenario in scenario_tree.scenarios:
            scenario_lp_row_filename = \
                os.path.join(scenario_directory,
                             basename+".lp.row."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_lp_row_filename+' '+
                               lp_row_filename)
            else:
                rc = not filecmp.cmp(scenario_lp_row_filename, lp_row_filename)
            if rc:
                raise ValueError(
                    "The LP row ordering indicated in file '%s' does not match that "
                    "for scenario %s indicated in file '%s'. This suggests that the "
                    "same constraint is being classified in different time stages "
                    "across scenarios. Consider manually declaring constraint "
                    "stages using the %s annotation if not already doing so, or "
                    "report this issue to the PySP developers."
                    % (lp_row_filename,
                       scenario.name,
                       scenario_lp_row_filename,
                       PySP_ConstraintStageAnnotation.__name__))

            scenario_lp_col_filename = \
                os.path.join(scenario_directory,
                             basename+".lp.col."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_lp_col_filename+' '+
                               lp_col_filename)
            else:
                rc = not filecmp.cmp(scenario_lp_col_filename, lp_col_filename)
            if rc:
                raise ValueError(
                    "The LP column ordering indicated in file '%s' does not match "
                    "that for scenario %s indicated in file '%s'. This suggests that"
                    " the set of variables on the model changes across scenarios. "
                    "This is not allowed by the SMPS format. If you feel this is a "
                    "developer error, please report this issue to the PySP "
                    "developers." % (lp_col_filename,
                                     scenario.name,
                                     scenario_lp_col_filename))

        print(" - Checking time-stage classifications...")
        for scenario in scenario_tree.scenarios:
            scenario_tim_filename = \
                os.path.join(scenario_directory,
                             basename+".tim."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_tim_filename+' '+
                               tim_filename)
            else:
                rc = not filecmp.cmp(scenario_tim_filename, tim_filename)
            if rc:
                raise ValueError(
                    "Main .tim file '%s' does not match .tim file for "
                    "scenario %s located at '%s'. This indicates there was a "
                    "problem translating the reference model to SMPS format. "
                    "Please make sure the problem structures are identical "
                    "over all scenarios (e.g., no. of variables, no. of constraints"
                    "), or report this issue to the PySP developers if you feel "
                    "that it is a developer error." % (tim_filename,
                                                       scenario.name,
                                                       scenario_tim_filename))

        print(" - Checking sparse locations of stochastic elements...")
        for scenario in scenario_tree.scenarios:
            scenario_sto_struct_filename = \
                os.path.join(scenario_directory,
                             basename+".sto.struct."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_sto_struct_filename+' '+
                               sto_struct_filename)
            else:
                rc = not filecmp.cmp(scenario_sto_struct_filename,
                                     sto_struct_filename)
            if rc:
                raise ValueError(
                    "The LP structure of stochastic entries indicated in file '%s' "
                    "does not match that for scenario %s indicated in file '%s'. "
                    "This suggests that the set of variables appearing in some "
                    "expression declared as stochastic is changing across scenarios."
                    " If you feel this is a developer error, please report this "
                    "issue to the PySP developers." % (sto_struct_filename,
                                                       scenario.name,
                                                       scenario_sto_struct_filename))

    if not keep_scenario_files:
        print("Cleaning temporary per-scenario files")
        shutil.rmtree(scenario_directory, ignore_errors=True)
    else:
        print("Temporary per-scenario files are retained in "
              "scenario_files subdirectory")

def convert_implicit(output_directory,
                     basename,
                     scenario_instance_factory,
                     io_options=None,
                     disable_consistency_checks=False,
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

    assert len(scenario_tree.stages) == 2
    firststage = scenario_tree.stages[0]
    secondstage = scenario_tree.stages[1]

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
        [item[0] for item in StageToVariableMap[firststage.name]]
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
        if StageToVariableMap[secondstage.name][0][0] not in \
           referenced_var_names:
            update_vars.append(StageToVariableMap[secondstage.name][0][1])
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
          in enumerate(StageToVariableMap[firststage.name]):
        column_order[vardata] = column_index
    # second-stage variables
    for column_index, (LP_name, vardata, scenario_tree_id) \
        in enumerate(StageToVariableMap[secondstage.name],
                     len(column_order)):
        column_order[vardata] = column_index

    #
    # Create row (constraint) ordering maps for LP files
    #
    row_order = ComponentMap()
    # first-stage constraints
    for row_index, (LP_names, condata) \
          in enumerate(StageToConstraintMap[firststage.name]):
        row_order[condata] = row_index
    # second-stage constraints
    for row_index, (LP_names, condata) \
        in enumerate(StageToConstraintMap[secondstage.name],
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
        f.write('\t'+str(StageToVariableMap[firststage.name][0][0])+
                '\t'+str(LP_symbol_map.byObject[id(reference_scenario._instance_objective)])+
                '\tTIME1\n')
        LP_names = StageToConstraintMap[secondstage.name][0][0]
        if len(LP_names) == 1:
            # equality constraint
            assert (LP_names[0].startswith('c_e_') or \
                    LP_names[0].startswith('c_l_') or \
                    LP_names[0].startswith('c_u_'))
            secondstage_row_start = LP_names[0]
        else:
            # range constraint (assumed the LP writer outputs
            # the lower range constraint first)
            LP_names = sorted(LP_names)
            assert (LP_names[0].startswith('r_l_') or \
                    LP_names[0].startswith('r_u_'))
            secondstage_row_start = LP_names[0]

        f.write('\t'+str(StageToVariableMap[secondstage.name][0][0])+
                '\t'+secondstage_row_start+
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
