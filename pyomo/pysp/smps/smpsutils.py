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
import copy

thisfile = os.path.abspath(__file__)
thisfile.replace(".pyc","").replace(".py","")

from pyomo.opt import WriterFactory
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

# LONG TERM TODO:
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
                                                            annotation.__class__.__name__))
        component_ids.add(id(component))

    for component in annotation_data:
        component_annotation_value = annotation_data[component]
        if isinstance(component, ctype_data):
            if component.active:
                _append(component, component_annotation_value)
        elif isinstance(component, ctype):
            for index in component:
                obj = component[index]
                if obj.active:
                    _append(obj, component_annotation_value)
        elif isinstance(component, _BlockData):
            if component.active:
                for obj in component.component_data_objects(
                        ctype,
                        active=True,
                        descend_into=True):
                    _append(obj, component_annotation_value)
        elif isinstance(component, Block):
            for index in component:
                block = component[index]
                if block.active:
                    for obj in block.component_data_objects(
                            ctype,
                            active=True,
                            descend_into=True):
                        _append(obj, component_annotation_value)
        else:
            raise TypeError(
                "(Scenario=%s): Declarations in annotation type %s must be of type "
                "%s or Block. Invalid type: %s" % (scenario.name,
                                                   annotation.__class__.__name__,
                                                   ctype.__name__,
                                                   type(component)))
        if check_value is not None:
            check_value(component, component_annotation_value)

    return items

def map_constraint_stages(scenario,
                          scenario_tree,
                          symbol_map,
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

    byObject = symbol_map.byObject
    # deal with the fact that the LP/MPS writer prepends constraint
    # names with things like 'c_e_', 'c_l_', etc depending on the
    # constraint bound type and will even split a constraint into
    # two constraints if it has two bounds
    reverse_alias = \
        dict((symbol, []) for symbol in symbol_map.bySymbol)
    for alias, obj_weakref in iteritems(symbol_map.aliases):
        reverse_alias[byObject[id(obj_weakref())]].append(alias)

    # ** SORT POINT TO AVOID NON-DETERMINISTIC ROW ORDERING ***
    for _aliases in itervalues(reverse_alias):
        _aliases.sort()

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

            symbol = byObject[id(constraint_data)]
            # if it is a range constraint this will account for
            # that fact and hold and alias for each bound
            aliases = reverse_alias[symbol]
            assert len(aliases) > 0
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
                append((aliases, constraint_data))

    assert sorted(StageToConstraintMap.keys()) == \
        sorted([firststage.name, secondstage.name])

    # sort each by name
    for key in StageToConstraintMap:
        StageToConstraintMap[key].sort(key=operator.itemgetter(0))

    return StageToConstraintMap

def map_variable_stages(scenario, scenario_tree, symbol_map):

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
    byObject = symbol_map.byObject
    all_vars_on_tree = []

    for scenario_tree_id, vardata in \
          iteritems(reference_model.\
          _ScenarioTreeSymbolMap.bySymbol):
        if vardata.is_expression():
            continue
        try:
            symbol = byObject[id(vardata)]
        except KeyError:
            raise ValueError("(Scenario=%s): Variable with name '%s' was declared "
                             "on the scenario tree but did not appear "
                             "in the reference scenario LP/MPS file."
                             % (scenario.name, vardata.cname(True)))
        if symbol == "RHS":
            raise RuntimeError(
                "Congratulations! You have hit an edge case. The "
                "SMPS input format forbids variables from having "
                "the name 'RHS'. Please rename it")
        if scenario_tree_id in firststage_blended_variables:
            FirstStageVars[symbol] = (vardata, scenario_tree_id)
        elif (scenario_tree_id in rootnode._derived_variable_ids) or \
             (scenario_tree_id in stagetwo_node._variable_ids):
            SecondStageVars[symbol] = (vardata, scenario_tree_id)
        else:
            # More than two stages?
            assert False

        all_vars_on_tree.append(symbol)

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
            symbol = byObject[id(vardata)]
            SecondStageVars[symbol] = (vardata, scenario_tree_id)
            all_vars_on_tree.append(symbol)

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
              "(found in LP/MPS file): "+str(len(tree_vars)))
        print("Number of Scenario Tree Cost Variables "
               "(found in LP/MPS file): "+str(len(cost_vars)))
        print("Number of Variables Found on Model: "
              +str(len(all_vars)))
        print("Variables Missing from Scenario Tree "
              "(or LP/MPS file):"+str(all_vars-tree_vars-cost_vars))
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
        [(symbol,
          FirstStageVars[symbol][0],
          FirstStageVars[symbol][1])
         for symbol in sorted(FirstStageVars)]
    StageToVariableMap[stagetwo.name] = \
        [(symbol,
          SecondStageVars[symbol][0],
          SecondStageVars[symbol][1])
         for symbol in sorted(SecondStageVars)]

    return StageToVariableMap

def _convert_explicit_setup(worker, scenario, *args, **kwds):
    reference_model = scenario._instance
    #
    # We will be tweaking the canonical_repn objects on objectives
    # and constraints, so cache anything related to this here so
    # that this function does not have any side effects on the
    # instance after returning
    #
    cached_attrs = []
    for block in reference_model.block_data_objects(
            active=True,
            descend_into=True):
        block_cached_attrs = {}
        if hasattr(block, "_gen_obj_canonical_repn"):
            block_cached_attrs["_gen_obj_canonical_repn"] = \
                block._gen_obj_canonical_repn
            del block._gen_obj_canonical_repn
        if hasattr(block, "_gen_con_canonical_repn"):
            block_cached_attrs["_gen_con_canonical_repn"] = \
                block._gen_con_canonical_repn
            del block._gen_con_canonical_repn
        if hasattr(block, "_canonical_repn"):
            block_cached_attrs["_canonical_repn"] = \
                block._canonical_repn
            del block._canonical_repn
        cached_attrs.append((block, block_cached_attrs))

    try:
        return _convert_explicit_setup_without_cleanup(
            worker, scenario, *args, **kwds)
    finally:
        for block, block_cached_attrs in cached_attrs:
            for name in block_cached_attrs:
                setattr(block, name, block_cached_attrs[name])

def _convert_explicit_setup_without_cleanup(worker,
                                            scenario,
                                            output_directory,
                                            basename,
                                            file_format,
                                            io_options):
    import pyomo.environ
    assert os.path.exists(output_directory)
    assert file_format in ('lp', 'mps')

    io_options = dict(io_options)
    scenario_tree = worker.scenario_tree
    reference_model = scenario._instance

    #
    # Check for model annotations
    #
    constraint_stage_annotation = locate_annotations(
        reference_model,
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

    stochastic_rhs = locate_annotations(
        reference_model,
        PySP_StochasticRHSAnnotation,
        max_allowed=1)
    if len(stochastic_rhs) == 0:
        stochastic_rhs = None
    else:
        assert len(stochastic_rhs) == 1
        stochastic_rhs = stochastic_rhs[0][1]

    stochastic_matrix = locate_annotations(
        reference_model,
        PySP_StochasticMatrixAnnotation,
        max_allowed=1)
    if len(stochastic_matrix) == 0:
        stochastic_matrix = None
    else:
        assert len(stochastic_matrix) == 1
        stochastic_matrix = stochastic_matrix[0][1]

    stochastic_objective = locate_annotations(
        reference_model,
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
        raise RuntimeError(
            "(Scenario=%s): No stochastic annotations found. SMPS "
            "conversion requires at least one of the following "
            "annotation types:\n - %s\n - %s\n - %s"
            % (scenario.name,
               PySP_StochasticRHSAnnotation.__name__,
               PySP_StochasticMatrixAnnotation.__name__,
               PySP_StochasticObjectiveAnnotation.__name__))

    #
    # Write the LP/MPS file once to obtain the symbol map
    #
    assert not hasattr(reference_model, "_canonical_repn")
    with WriterFactory(file_format) as writer:
        output_filename = \
            os.path.join(output_directory,
                         basename+".setup."+file_format+"."+scenario.name)
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        output_fname, symbol_map = writer(reference_model,
                                          output_filename,
                                          lambda x: True,
                                          io_options)
        assert output_fname == output_filename
    assert hasattr(reference_model, "_canonical_repn")

    StageToVariableMap = \
        map_variable_stages(scenario,
                            scenario_tree,
                            symbol_map)

    StageToConstraintMap = \
        map_constraint_stages(
            scenario,
            scenario_tree,
            symbol_map,
            constraint_stage_assignments=constraint_stage_assignments)

    assert len(scenario_tree.stages) == 2
    firststage = scenario_tree.stages[0]
    secondstage = scenario_tree.stages[1]

    # disable these as they do not need to be regenerated and
    # we will be modifiying them
    canonical_repn_cache = {}
    for block in reference_model.block_data_objects(
            active=True,
            descend_into=True):
        canonical_repn_cache[id(block)] = block._canonical_repn
        block._gen_obj_canonical_repn = False
        block._gen_con_canonical_repn = False

    #
    # Make sure the objective references all first stage variables.
    # We do this by directly modifying the canonical_repn of the
    # objective which the LP/MPS writer will reference next time we call
    # it. In addition, make sure that the first second-stage variable
    # in our column ordering also appears in the objective so that
    # ONE_VAR_CONSTANT does not get identified as the first
    # second-stage variable.
    # ** Just do NOT preprocess again until we call the writer **
    #
    objective_data = scenario._instance_objective
    assert objective_data is not None
    objective_block = objective_data.parent_block()
    objective_repn = canonical_repn_cache[id(objective_block)][objective_data]
    """
    original_objective_repn = copy.deepcopy(objective_repn)
    first_stage_varname_list = \
        [item[0] for item in StageToVariableMap[firststage.name]]
    if isinstance(objective_repn, LinearCanonicalRepn) and \
       (objective_repn.linear is not None):
        referenced_var_names = set([symbol_map.byObject[id(vardata)]
                                    for vardata in objective_repn.variables])
        obj_vars = list(objective_repn.variables)
        obj_coefs = list(objective_repn.linear)
        # add the first-stage variables (if not present)
        for symbol in first_stage_varname_list:
            if symbol not in referenced_var_names:
                obj_vars.append(symbol_map.bySymbol[symbol]())
                obj_coefs.append(0.0)
        # add the first second-stage variable (if not present),
        # this will make sure the ONE_VAR_CONSTANT variable
        # is not identified as the first second-stage variable
        # (but don't assume there is always a second stage variable)
        if len(StageToVariableMap[secondstage.name]) > 0:
            if StageToVariableMap[secondstage.name][0][0] not in \
               referenced_var_names:
                obj_vars.append(StageToVariableMap[secondstage.name][0][1])
                obj_coefs.append(0.0)
        objective_repn.variables = tuple(obj_vars)
        objective_repn.linear = tuple(obj_coefs)

    else:
        raise RuntimeError("(Scenario=%s): A linear objective is required for "
                           "conversion to SMPS format."
                           % (scenario.name))
    """

    #
    # Create column (variable) ordering maps for LP/MPS files
    #
    column_order = ComponentMap()
    # first-stage variables
    for column_index, (symbol, vardata, scenario_tree_id) \
        in enumerate(StageToVariableMap[firststage.name]):
        column_order[vardata] = column_index
    # second-stage variables
    for column_index, (symbol, vardata, scenario_tree_id) \
        in enumerate(StageToVariableMap[secondstage.name],
                     len(column_order)):
        column_order[vardata] = column_index

    #
    # Create row (constraint) ordering maps for LP/MPS files
    #
    row_order = ComponentMap()
    # first-stage constraints
    for row_index, (symbols, condata) \
        in enumerate(StageToConstraintMap[firststage.name]):
        row_order[condata] = row_index
    # second-stage constraints
    for row_index, (symbols, condata) \
        in enumerate(StageToConstraintMap[secondstage.name],
                     len(row_order)):
        row_order[condata] = row_index

    #
    # Write the ordered LP/MPS file
    #
    output_filename = os.path.join(output_directory,
                                   basename+"."+file_format+"."+scenario.name)
    with WriterFactory(file_format) as writer:
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        io_options['column_order'] = column_order
        io_options['row_order'] = row_order
        io_options['force_objective_constant'] = True
        output_fname, symbol_map = writer(reference_model,
                                          output_filename,
                                          lambda x: True,
                                          io_options)
        assert output_fname == output_filename

    # re-generate these maps as the LP/MPS symbol map
    # is likely different
    StageToVariableMap = map_variable_stages(
        scenario,
        scenario_tree,
        symbol_map)

    StageToConstraintMap = map_constraint_stages(
        scenario,
        scenario_tree,
        symbol_map,
        constraint_stage_assignments=constraint_stage_assignments)

    # generate a few data structures that are used
    # when writing the .sto file
    constraint_symbols = ComponentMap(
        (constraint_data, symbols) for stage_name in StageToConstraintMap
        for symbols, constraint_data in StageToConstraintMap[stage_name])
    secondstage_constraint_ids = \
        set(id(constraint_data) for symbols, constraint_data
            in StageToConstraintMap[secondstage.name])
    firststage_variable_ids = \
        set(id(variable_data) for symbol, variable_data, scenario_tree_id
            in StageToVariableMap[secondstage.name])

    #
    # Write the explicit column ordering (variables) used
    # for the ordered LP/MPS file
    #
    firststage_col_count = 0
    col_count = 0
    with open(os.path.join(output_directory,
                           basename+".col."+scenario.name),'w') as f_col:
        # first-stage variables
        for (symbol, _, _) in StageToVariableMap[firststage.name]:
            firststage_col_count += 1
            col_count += 1
            f_col.write(symbol+"\n")
        # second-stage variables
        for (symbol, _, _) in StageToVariableMap[secondstage.name]:
            col_count += 1
            f_col.write(symbol+"\n")
        col_count += 1
        f_col.write("ONE_VAR_CONSTANT\n")

    #
    # Write the explicit row ordering (constraints) used
    # for the ordered LP/MPS file
    #
    firststage_row_count = 0
    row_count = 0
    with open(os.path.join(output_directory,
                           basename+".row."+scenario.name),'w') as f_row:
        # the objective is always the first row in SMPS format
        f_row.write(symbol_map.byObject[id(objective_data)]+"\n")
        # first-stage constraints
        for (symbols, _) in StageToConstraintMap[firststage.name]:
            # because range constraints are split into two
            # constraints (hopefully our ordering of the r_l_
            # and r_u_ forms is the same as the LP/MPS file!)
            for symbol in symbols:
                firststage_row_count += 1
                row_count += 1
                f_row.write(symbol+"\n")
        # second-stage constraints
        for (symbols, _) in StageToConstraintMap[secondstage.name]:
            # because range constraints are split into two
            # constraints (hopefully our ordering of the r_l_
            # and r_u_ forms is the same as the LP/MPS file!)
            for symbol in symbols:
                f_row.write(symbol+"\n")
                row_count += 1
        f_row.write("c_e_ONE_VAR_CONSTANT")

    #
    # Write the .tim file
    #
    with open(os.path.join(output_directory,
                           basename+".tim."+scenario.name),'w') as f_tim:
        f_tim.write("TIME %s\n" % (basename))
        if file_format == 'mps':
            f_tim.write("PERIODS IMPLICIT\n")
            f_tim.write("    %s %s TIME1\n"
                        % (StageToVariableMap[firststage.name][0][0],
                           symbol_map.byObject[id(objective_data)]))
            symbols = StageToConstraintMap[secondstage.name][0][0]
            if len(symbols) == 1:
                # equality constraint
                assert (symbols[0].startswith('c_e_') or \
                        symbols[0].startswith('c_l_') or \
                        symbols[0].startswith('c_u_'))
                stage2_row_start = symbols[0]
            else:
                # range constraint (assumed the LP/MPS writer outputs
                # the lower range constraint first)
                symbols = sorted(symbols)
                assert (symbols[0].startswith('r_l_') or \
                        symbols[0].startswith('r_u_'))
                stage2_row_start = symbols[0]
            # don't assume there is always a second stage variable
            if len(StageToVariableMap[secondstage.name][0][0]) > 0:
                f_tim.write("    %s "
                            % (StageToVariableMap[secondstage.name][0][0]))
            else:
                f_tim.write("    ONE_VAR_CONSTANT ")
            f_tim.write("%s TIME2\n" % (stage2_row_start))
        else:
            assert file_format == "lp"
            f_tim.write("PERIODS EXPLICIT\n")
            f_tim.write("    TIME1\n")
            f_tim.write("    TIME2\n")
            line_template = "    %s %s\n"
            f_tim.write("ROWS\n")
            # the objective is always the first row in SMPS format
            f_tim.write(line_template
                        % (symbol_map.byObject[id(objective_data)],
                           "TIME1"))
            # first-stage constraints
            for (symbols, _) in StageToConstraintMap[firststage.name]:
                for symbol in symbols:
                    f_tim.write(line_template % (symbol, "TIME1"))
            # second-stage constraints
            for (symbols, _) in StageToConstraintMap[secondstage.name]:
                for symbol in symbols:
                    f_tim.write(line_template % (symbol, "TIME2"))
            f_tim.write(line_template % ("c_e_ONE_VAR_CONSTANT", "TIME2"))

            f_tim.write("COLS\n")
            # first-stage variables
            for (symbol, _, _) in StageToVariableMap[firststage.name]:
                f_tim.write(line_template % (symbol, "TIME1"))
            # second-stage variables
            for (symbol, _, _) in StageToVariableMap[secondstage.name]:
                f_tim.write(line_template % (symbol, "TIME2"))
            f_tim.write(line_template % ("ONE_VAR_CONSTANT", "TIME2"))

        f_tim.write("ENDATA\n")

    stochastic_lp_labels = set()
    stochastic_constraint_count = 0
    stochastic_secondstage_rhs_count = 0
    stochastic_firststagevar_constraint_count = 0
    stochastic_secondstagevar_constraint_count = 0
    stochastic_firststagevar_objective_count = 0
    stochastic_secondstagevar_objective_count = 0
    #
    # Write the body of the .sto file
    #
    #
    # **NOTE: In the code that follows we assume the LP/MPS
    #         writer always moves constraint body
    #         constants to the rhs and that the lower part
    #         of any range constraints are written before
    #         the upper part.
    #
    modified_constraint_lb = ComponentMap()
    modified_constraint_ub = ComponentMap()
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
                    sorted_values.sort(
                        key=lambda x: x[0].cname(True, constraint_name_buffer))
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

                    constraint_repn = \
                        canonical_repn_cache[id(constraint_data.parent_block())][constraint_data]
                    if not isinstance(constraint_repn, LinearCanonicalRepn):
                        raise RuntimeError("(Scenario=%s): Only linear constraints are "
                                           "accepted for conversion to SMPS format. "
                                           "Constraint %s is not linear."
                                           % (scenario.name,
                                              constraint_data.cname(True)))

                    body_constant = constraint_repn.constant
                    # We are going to rewrite the core problem file
                    # with all stochastic values set to zero. This will
                    # allow an easy test for missing user annotations.
                    constraint_repn.constant = 0
                    if body_constant is None:
                        body_constant = 0.0
                    symbols = constraint_symbols[constraint_data]
                    assert len(symbols) > 0
                    for con_label in symbols:
                        if con_label.startswith('c_e_') or \
                           con_label.startswith('c_l_'):
                            assert (include_bound is True) or \
                                   (include_bound[0] is True)
                            stochastic_lp_labels.add(con_label)
                            stochastic_secondstage_rhs_count += 1
                            f_sto.write(rhs_template %
                                        (con_label,
                                         value(constraint_data.lower) - \
                                         value(body_constant)))
                            f_coords.write("RHS %s\n" % (con_label))
                            # We are going to rewrite the core problem file
                            # with all stochastic values set to zero. This will
                            # allow an easy test for missing user annotations.
                            modified_constraint_lb[constraint_data] = \
                                constraint_data.lower
                            constraint_data._lower = 0
                            if con_label.startswith('c_e_'):
                                modified_constraint_ub[constraint_data] = \
                                    constraint_data.upper
                                constraint_data._upper = 0
                        elif con_label.startswith('r_l_') :
                            if (include_bound is True) or \
                               (include_bound[0] is True):
                                stochastic_lp_labels.add(con_label)
                                stochastic_secondstage_rhs_count += 1
                                f_sto.write(rhs_template %
                                            (con_label,
                                             value(constraint_data.lower) - \
                                             value(body_constant)))
                                f_coords.write("RHS %s\n" % (con_label))
                                # We are going to rewrite the core problem file
                                # with all stochastic values set to zero. This will
                                # allow an easy test for missing user annotations.
                                modified_constraint_lb[constraint_data] = \
                                    constraint_data.lower
                                constraint_data._lower = 0
                        elif con_label.startswith('c_u_'):
                            assert (include_bound is True) or \
                                   (include_bound[1] is True)
                            stochastic_lp_labels.add(con_label)
                            stochastic_secondstage_rhs_count += 1
                            f_sto.write(rhs_template %
                                        (con_label,
                                         value(constraint_data.upper) - \
                                         value(body_constant)))
                            f_coords.write("RHS %s\n" % (con_label))
                            # We are going to rewrite the core problem file
                            # with all stochastic values set to zero. This will
                            # allow an easy test for missing user annotations.
                            modified_constraint_ub[constraint_data] = \
                                constraint_data.upper
                            constraint_data._upper = 0
                        elif con_label.startswith('r_u_'):
                            if (include_bound is True) or \
                               (include_bound[1] is True):
                                stochastic_lp_labels.add(con_label)
                                stochastic_secondstage_rhs_count += 1
                                f_sto.write(rhs_template %
                                            (con_label,
                                             value(constraint_data.upper) - \
                                             value(body_constant)))
                                f_coords.write("RHS %s\n" % (con_label))
                                # We are going to rewrite the core problem file
                                # with all stochastic values set to zero. This will
                                # allow an easy test for missing user annotations.
                                modified_constraint_ub[constraint_data] = \
                                    constraint_data.upper
                                constraint_data._upper = 0
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
                    sorted_values.sort(
                        key=lambda x: x[0].cname(True, constraint_name_buffer))
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
                    constraint_repn = \
                        canonical_repn_cache[id(constraint_data.parent_block())][constraint_data]
                    if not isinstance(constraint_repn, LinearCanonicalRepn):
                        raise RuntimeError("(Scenario=%s): Only linear constraints are "
                                           "accepted for conversion to SMPS format. "
                                           "Constraint %s is not linear."
                                           % (scenario.name,
                                              constraint_data.cname(True)))
                    assert len(constraint_repn.variables) > 0
                    if var_list is None:
                        var_list = constraint_repn.variables
                    assert len(var_list) > 0
                    symbols = constraint_symbols[constraint_data]
                    stochastic_constraint_count += len(symbols)
                    # sort the variable list by the column ordering
                    # so that we have deterministic output
                    var_list = list(var_list)
                    var_list.sort(key=lambda _v: column_order[_v])
                    new_coefs = list(constraint_repn.linear)
                    for var_data in var_list:
                        assert isinstance(var_data, _VarData)
                        assert not var_data.fixed
                        var_coef = None
                        for i, (var, coef) in enumerate(zip(constraint_repn.variables,
                                                            constraint_repn.linear)):
                            if var is var_data:
                                var_coef = coef
                                # We are going to rewrite with core problem file
                                # with all stochastic values set to zero. This will
                                # allow an easy test for missing user annotations.
                                new_coefs[i] = 0
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
                        var_label = symbol_map.byObject[id(var_data)]
                        for con_label in symbols:
                            if id(var_data) in firststage_variable_ids:
                                stochastic_firststagevar_constraint_count += 1
                            else:
                                stochastic_secondstagevar_constraint_count += 1
                            stochastic_lp_labels.add(con_label)
                            f_sto.write(matrix_template % (var_label,
                                                           con_label,
                                                           value(var_coef)))
                            f_coords.write("%s %s\n" % (var_label, con_label))

                    constraint_repn.linear = tuple(new_coefs)

            stochastic_constraint_count = len(stochastic_lp_labels)

            #
            # Stochastic Objective
            #
            obj_template = "    %s    %s    %.17g\n"
            if stochastic_objective is not None:
                if len(stochastic_objective.data) > 0:
                    sorted_values = _expand_annotation_entries(
                        scenario,
                        Objective,
                        stochastic_objective,
                        check_value=None)
                    assert len(sorted_values) <= 1
                    if len(sorted_values) == 0:
                        raise RuntimeError(
                            "(Scenario=%s): The %s annotation was declared "
                            "with explicit entries but no active Objective "
                            "objects were recovered from those entries."
                            % (scenario.name,
                               PySP_StochasticObjectiveAnnotation.__name__))
                    objdata, (objective_variables, include_constant) = \
                        sorted_values[0]
                    assert objdata is objective_data
                else:
                    objective_variables, include_constant = \
                        stochastic_objective.default

                if not isinstance(objective_repn, LinearCanonicalRepn):
                    raise RuntimeError("(Scenario=%s): Only linear objectives are "
                                       "accepted for conversion to SMPS format. "
                                       "Objective %s is not linear."
                                       % (scenario.name,
                                          objective_data.cname(True)))
                if objective_variables is None:
                    objective_variables = objective_repn.variables
                stochastic_objective_label = symbol_map.byObject[id(objective_data)]
                # sort the variable list by the column ordering
                # so that we have deterministic output
                objective_variables = list(objective_variables)
                objective_variables.sort(key=lambda _v: column_order[_v])
                stochastic_lp_labels.add(stochastic_objective_label)
                assert (len(objective_variables) > 0) or include_constant
                new_coefs = list(objective_repn.linear)
                for var_data in objective_variables:
                    assert isinstance(var_data, _VarData)
                    var_coef = None
                    for i, (var, coef) in enumerate(zip(objective_repn.variables,
                                                        objective_repn.linear)):
                        if var is var_data:
                            var_coef = coef
                            # We are going to rewrite the core problem file
                            # with all stochastic values set to zero. This will
                            # allow an easy test for missing user annotations.
                            new_coefs[i] = 0
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
                    var_label = symbol_map.byObject[id(var_data)]
                    if id(var_data) in firststage_variable_ids:
                        stochastic_firststagevar_objective_count += 1
                    else:
                        stochastic_secondstagevar_objective_count += 1
                    f_sto.write(obj_template % (var_label,
                                                stochastic_objective_label,
                                                value(var_coef)))
                    f_coords.write("%s %s\n"
                                   % (var_label,
                                      stochastic_objective_label))

                objective_repn.linear = tuple(new_coefs)
                if include_constant:
                    obj_constant = objective_repn.constant
                    # We are going to rewrite the core problem file
                    # with all stochastic values set to zero. This will
                    # allow an easy test for missing user annotations.
                    objective_repn.constant = 0
                    if obj_constant is None:
                        obj_constant = 0.0
                    stochastic_secondstagevar_objective_count += 1
                    f_sto.write(obj_template % ("ONE_VAR_CONSTANT",
                                                stochastic_objective_label,
                                                obj_constant))
                    f_coords.write("%s %s\n"
                                   % ("ONE_VAR_CONSTANT",
                                      stochastic_objective_label))

    #
    # Write the deterministic part of the LP/MPS-file to its own
    # file for debugging purposes
    #
    reference_model_name = reference_model.name
    reference_model.name = "ZeroStochasticData"
    det_output_filename = \
        os.path.join(output_directory,
                     basename+"."+file_format+".det."+scenario.name)
    with WriterFactory(file_format) as writer:
        output_fname, symbol_map = writer(reference_model,
                                          det_output_filename,
                                          lambda x: True,
                                          io_options)
        assert output_fname == det_output_filename
    reference_model.name = reference_model_name

    # reset bounds on any constraints that were modified
    for constraint_data, lower in iteritems(modified_constraint_lb):
        constraint_data._lower = lower
    for constraint_data, upper in iteritems(modified_constraint_ub):
        constraint_data._upper = upper

    return (firststage_row_count,
            row_count,
            firststage_col_count,
            col_count,
            stochastic_constraint_count,
            stochastic_secondstage_rhs_count,
            stochastic_firststagevar_constraint_count,
            stochastic_secondstagevar_constraint_count,
            stochastic_firststagevar_objective_count,
            stochastic_secondstagevar_objective_count)

def convert_explicit(output_directory,
                     basename,
                     scenario_tree_manager,
                     core_format='mps',
                     io_options=None,
                     disable_consistency_checks=False,
                     keep_scenario_files=False,
                     keep_auxiliary_files=False):
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

    counts = scenario_tree_manager.invoke_function(
        "_convert_explicit_setup",
        thisfile,
        invocation_type=InvocationType.PerScenario,
        function_args=(scenario_directory,
                       basename,
                       core_format,
                       io_options))

    reference_scenario = scenario_tree.scenarios[0]
    reference_scenario_name = reference_scenario.name

    # TODO: Out of laziness we are making shell calls to
    #       tools like 'cp' and 'cat'. Update the code to
    #       work on Windows.

    #
    # Copy the reference scenario's core, row, col, and tim
    # to the output directory. The consistency checks will
    # verify that these files match across scenarios.
    #
    core_filename = os.path.join(output_directory,
                                 basename+"."+core_format)
    _safe_remove_file(core_filename)
    shutil.copy2(os.path.join(scenario_directory,
                             (basename+"."+core_format+"."+
                              reference_scenario_name)),
                core_filename)

    core_row_filename = os.path.join(output_directory,
                                   basename+".row")
    _safe_remove_file(core_row_filename)
    shutil.copy2(os.path.join(scenario_directory,
                              (basename+".row."+
                               reference_scenario_name)),
                 core_row_filename)

    core_col_filename = os.path.join(output_directory,
                                   basename+".col")
    _safe_remove_file(core_col_filename)
    shutil.copy2(os.path.join(scenario_directory,
                              (basename+".col."+
                               reference_scenario_name)),
                 core_col_filename)

    tim_filename = os.path.join(output_directory,
                                basename+".tim")
    _safe_remove_file(tim_filename)
    shutil.copy2(os.path.join(scenario_directory,
                              (basename+".tim."+
                               reference_scenario_name)),
                 tim_filename)

    sto_struct_filename = os.path.join(output_directory,
                                       basename+".sto.struct")
    _safe_remove_file(sto_struct_filename)
    shutil.copy2(os.path.join(scenario_directory,
                              (basename+".sto.struct."+
                               reference_scenario_name)),
                 sto_struct_filename)

    core_det_filename = os.path.join(output_directory,
                                     basename+"."+core_format+".det")
    _safe_remove_file(core_det_filename)
    shutil.copy2(os.path.join(scenario_directory,
                              (basename+"."+core_format+".det."+
                               reference_scenario_name)),
                 core_det_filename)

    #
    # Merge the per-scenario .sto files into one
    #
    sto_filename = os.path.join(output_directory,
                                basename+".sto")
    _safe_remove_file(sto_filename)
    with open(sto_filename, 'w') as fdst:
        fdst.write('STOCH '+basename+'\n')
        fdst.write('BLOCKS DISCRETE REPLACE\n')
        for scenario in scenario_tree.scenarios:
            scenario_sto_filename = \
                os.path.join(scenario_directory,
                             basename+".sto."+scenario.name)
            assert os.path.exists(scenario_sto_filename)
            with open(scenario_sto_filename, 'r') as fsrc:
                shutil.copyfileobj(fsrc, fdst)
        fdst.write('ENDATA\n')

    print("\nSMPS Conversion Complete")
    print("Output Saved To: "+os.path.relpath(output_directory))
    print("Basis Problem Information:")
    (firststage_row_count,
     row_count,
     firststage_col_count,
     col_count,
     stochastic_constraint_count,
     stochastic_secondstage_rhs_count,
     stochastic_firststagevar_constraint_count,
     stochastic_secondstagevar_constraint_count,
     stochastic_firststagevar_objective_count,
     stochastic_secondstagevar_objective_count) = counts[reference_scenario_name]
    print(" - Objective:")
    print("    - Stochastic Variable Coefficients: %d"
          % (stochastic_firststagevar_objective_count + \
             stochastic_secondstagevar_objective_count))
    print("        - First-Stage:  %d"
          % (stochastic_firststagevar_objective_count))
    print("        - Second-Stage: %d"
          % (stochastic_secondstagevar_objective_count))
    print(" - Constraint Matrix:")
    print("    - Columns: %d" % (col_count))
    print("        - First-Stage:  %d" % (firststage_col_count))
    print("        - Second-Stage: %d" % (col_count - firststage_col_count))
    print("    - Rows:    %d" % (row_count))
    print("        - First-Stage:  %d" % (firststage_row_count))
    print("        - Second-Stage: %d" % (row_count - firststage_row_count))
    print("    - Stochastic Second-Stage Rows: %d" % (stochastic_constraint_count))
    print("        - Stochastic Right-Hand-Sides:      %d"
          % (stochastic_secondstage_rhs_count))
    print("        - Stochastic Variable Coefficients: %d"
          % (stochastic_firststagevar_constraint_count + \
             stochastic_secondstagevar_constraint_count))
    print("            - First-Stage:  %d"
          % (stochastic_firststagevar_constraint_count))
    print("            - Second-Stage: %d"
          % (stochastic_secondstagevar_constraint_count))

    if not disable_consistency_checks:
        print("\nStarting scenario structure consistency checks across scenario "
              "files stored in %s." % (scenario_directory))
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
            scenario_core_row_filename = \
                os.path.join(scenario_directory,
                             basename+".row."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_core_row_filename+' '+
                               core_row_filename)
            else:
                rc = not filecmp.cmp(scenario_core_row_filename, core_row_filename)
            if rc:
                raise ValueError(
                    "The row ordering indicated in file '%s' does not match that "
                    "for scenario %s indicated in file '%s'. This suggests that the "
                    "same constraint is being classified in different time stages "
                    "across scenarios. Consider manually declaring constraint "
                    "stages using the %s annotation if not already doing so, or "
                    "report this issue to the PySP developers."
                    % (core_row_filename,
                       scenario.name,
                       scenario_core_row_filename,
                       PySP_ConstraintStageAnnotation.__name__))

            scenario_core_col_filename = \
                os.path.join(scenario_directory,
                             basename+".col."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_core_col_filename+' '+
                               core_col_filename)
            else:
                rc = not filecmp.cmp(scenario_core_col_filename, core_col_filename)
            if rc:
                raise ValueError(
                    "The column ordering indicated in file '%s' does not match "
                    "that for scenario %s indicated in file '%s'. This suggests that"
                    " the set of variables on the model changes across scenarios. "
                    "This is not allowed by the SMPS format. If you feel this is a "
                    "developer error, please report this issue to the PySP "
                    "developers." % (core_col_filename,
                                     scenario.name,
                                     scenario_core_col_filename))

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
                    "The structure of stochastic entries indicated in file '%s' "
                    "does not match that for scenario %s indicated in file '%s'. "
                    "This suggests that the set of variables appearing in some "
                    "expression declared as stochastic is changing across scenarios."
                    " If you feel this is a developer error, please report this "
                    "issue to the PySP developers." % (sto_struct_filename,
                                                       scenario.name,
                                                       scenario_sto_struct_filename))

        print(" - Checking deterministic sections in the core problem file...")
        for scenario in scenario_tree.scenarios:
            scenario_core_det_filename = \
                os.path.join(scenario_directory,
                             basename+"."+core_format+".det."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_core_det_filename+' '+
                               core_det_filename)
            else:
                rc = not filecmp.cmp(scenario_core_det_filename, core_det_filename)
            if rc:
                raise ValueError(
                    "One or more deterministic parts of the problem found in file '%s' do "
                    "not match those for scenario %s found in file %s. This suggests "
                    "that one or more locations of stochastic data have not been "
                    "been annotated on the reference Pyomo model. If this seems like "
                    "a tolerance issue or a developer error, please report this issue "
                    "to the PySP developers."
                    % (core_det_filename,
                       scenario.name,
                       scenario_core_det_filename))

    if not keep_auxiliary_files:
        _safe_remove_file(core_row_filename)
        _safe_remove_file(core_col_filename)
        _safe_remove_file(sto_struct_filename)
        _safe_remove_file(core_det_filename)

    if not keep_scenario_files:
        print("Cleaning temporary per-scenario files")
        for scenario in scenario_tree.scenarios:

            scenario_core_row_filename = \
                os.path.join(scenario_directory,
                             basename+".row."+scenario.name)
            assert os.path.exists(scenario_core_row_filename)
            _safe_remove_file(scenario_core_row_filename)

            scenario_core_col_filename = \
                os.path.join(scenario_directory,
                             basename+".col."+scenario.name)
            assert os.path.exists(scenario_core_col_filename)
            _safe_remove_file(scenario_core_col_filename)

            scenario_tim_filename = \
                os.path.join(scenario_directory,
                             basename+".tim."+scenario.name)
            assert os.path.exists(scenario_tim_filename)
            _safe_remove_file(scenario_tim_filename)

            scenario_sto_struct_filename = \
                os.path.join(scenario_directory,
                             basename+".sto.struct."+scenario.name)
            assert os.path.exists(scenario_sto_struct_filename)
            _safe_remove_file(scenario_sto_struct_filename)

            scenario_sto_filename = \
                os.path.join(scenario_directory,
                             basename+".sto."+scenario.name)
            assert os.path.exists(scenario_sto_filename)
            _safe_remove_file(scenario_sto_filename)

            scenario_core_det_filename = \
                os.path.join(scenario_directory,
                             basename+"."+core_format+".det."+scenario.name)
            assert os.path.exists(scenario_core_det_filename)
            _safe_remove_file(scenario_core_det_filename)

            scenario_core_setup_filename = \
                os.path.join(scenario_directory,
                             basename+".setup."+core_format+"."+scenario.name)
            assert os.path.exists(scenario_core_setup_filename)
            _safe_remove_file(scenario_core_setup_filename)

            scenario_core_filename = \
                os.path.join(scenario_directory,
                             basename+"."+core_format+"."+scenario.name)
            assert os.path.exists(scenario_core_filename)
            _safe_remove_file(scenario_core_filename)

        # only delete this directory if it is empty,
        # it might have previously existed and contains
        # user files
        if len(os.listdir(scenario_directory)) == 0:
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
