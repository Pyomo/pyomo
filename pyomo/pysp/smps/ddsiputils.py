#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import shutil
import filecmp
import logging

from pyomo.opt import WriterFactory
from pyomo.core.base.numvalue import value
from pyomo.core.base.block import (Block,
                                   _BlockData,
                                   SortComponents)
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.constraint import Constraint, _ConstraintData
from pyomo.core.base.suffix import ComponentMap
from pyomo.core.base import TextLabeler, NumericLabeler
from pyomo.repn import LinearCanonicalRepn
from pyomo.pysp.scenariotree.manager import InvocationType
from pyomo.pysp.annotations import (locate_annotations,
                                    _ConstraintStageAnnotation,
                                    StochasticConstraintBoundsAnnotation,
                                    StochasticConstraintBodyAnnotation,
                                    StochasticObjectiveAnnotation,
                                    StochasticVariableBoundsAnnotation)
from pyomo.pysp.smps.smpsutils import (map_variable_stages,
                                       map_constraint_stages,
                                       _safe_remove_file,
                                       _no_negative_zero,
                                       _deterministic_check_value,
                                       ProblemStats)

from six import iteritems, itervalues

thisfile = os.path.abspath(__file__)

logger = logging.getLogger('pyomo.pysp')

def _convert_external_setup(worker, scenario, *args, **kwds):
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
        return _convert_external_setup_without_cleanup(
            worker, scenario, *args, **kwds)
    except:
        logger.error("Failed to complete partial DDSIP conversion "
                     "for scenario: %s" % (scenario.name))
        raise
    finally:
        for block, block_cached_attrs in cached_attrs:
            for name in block_cached_attrs:
                setattr(block, name, block_cached_attrs[name])

def _convert_external_setup_without_cleanup(
        worker,
        scenario,
        output_directory,
        firststage_var_suffix,
        enforce_derived_nonanticipativity,
        io_options):
    import pyomo.environ
    assert os.path.exists(output_directory)

    io_options = dict(io_options)
    scenario_tree = worker.scenario_tree
    reference_model = scenario._instance
    firststage = scenario_tree.stages[0]
    secondstage = scenario_tree.stages[1]
    constraint_name_buffer = {}
    objective_name_buffer = {}
    variable_name_buffer = {}

    all_constraints = list(
        con for con in reference_model.component_data_objects(
            Constraint,
            active=True,
            descend_into=True))

    #
    # Check for model annotations
    #
    stochastic_rhs = locate_annotations(
        reference_model,
        StochasticConstraintBoundsAnnotation,
        max_allowed=1)
    if len(stochastic_rhs) == 0:
        stochastic_rhs = None
        stochastic_rhs_entries = {}
        empty_rhs_annotation = False
    else:
        assert len(stochastic_rhs) == 1
        stochastic_rhs = stochastic_rhs[0][1]
        if stochastic_rhs.has_declarations():
            empty_rhs_annotation = False
            stochastic_rhs_entries = stochastic_rhs.expand_entries()
            stochastic_rhs_entries.sort(
                key=lambda x: x[0].cname(True, constraint_name_buffer))
            if len(stochastic_rhs_entries) == 0:
                raise RuntimeError(
                    "The %s annotation was declared "
                    "with external entries but no active Constraint "
                    "objects were recovered from those entries."
                    % (StochasticConstraintBoundsAnnotation.__name__))
        else:
            empty_rhs_annotation = True
            stochastic_rhs_entries = tuple((con, stochastic_rhs.default)
                                           for con in all_constraints)


    stochastic_matrix = locate_annotations(
        reference_model,
        StochasticConstraintBodyAnnotation,
        max_allowed=1)
    if len(stochastic_matrix) == 0:
        stochastic_matrix = None
        stochastic_matrix_entries = {}
        empty_matrix_annotation = False
    else:
        assert len(stochastic_matrix) == 1
        stochastic_matrix = stochastic_matrix[0][1]
        if stochastic_matrix.has_declarations():
            empty_matrix_annotation = False
            stochastic_matrix_entries = stochastic_matrix.expand_entries()
            stochastic_matrix_entries.sort(
                key=lambda x: x[0].cname(True, constraint_name_buffer))
            if len(stochastic_matrix_entries) == 0:
                raise RuntimeError(
                    "The %s annotation was declared "
                    "with external entries but no active Constraint "
                    "objects were recovered from those entries."
                    % (StochasticConstraintBoundsAnnotation.__name__))
        else:
            empty_matrix_annotation = True
            stochastic_matrix_entries = tuple((con,stochastic_matrix.default)
                                              for con in all_constraints)

    stochastic_constraint_ids = set()
    stochastic_constraint_ids.update(id(con) for con,_
                                     in stochastic_rhs_entries)
    stochastic_constraint_ids.update(id(con) for con,_
                                     in stochastic_matrix_entries)

    stochastic_objective = locate_annotations(
        reference_model,
        StochasticObjectiveAnnotation,
        max_allowed=1)
    if len(stochastic_objective) == 0:
        stochastic_objective = None
    else:
        assert len(stochastic_objective) == 1
        stochastic_objective = stochastic_objective[0][1]

    stochastic_varbounds = locate_annotations(
        reference_model,
        StochasticVariableBoundsAnnotation)
    if len(stochastic_varbounds) > 0:
        raise ValueError(
            "The SMPS writer does not currently support "
            "stochastic variable bounds. Invalid annotation type: %s"
            % (StochasticVariableBoundsAnnotation.__name__))

    if (stochastic_rhs is None) and \
       (stochastic_matrix is None) and \
       (stochastic_objective is None):
        raise RuntimeError(
            "No stochastic annotations found. SMPS "
            "conversion requires at least one of the following "
            "annotation types:\n - %s\n - %s\n - %s"
            % (StochasticConstraintBoundsAnnotation.__name__,
               StochasticConstraintBodyAnnotation.__name__,
               StochasticObjectiveAnnotation.__name__))

    #
    # Write the LP file once to obtain the symbol map
    #
    assert not hasattr(reference_model, "_canonical_repn")
    output_filename = os.path.join(output_directory,
                                   scenario.name+".core.lp.setup")
    with WriterFactory("lp") as writer:
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        output_fname, symbol_map = writer(reference_model,
                                          output_filename,
                                          lambda x: True,
                                          io_options)
        assert output_fname == output_filename
    assert hasattr(reference_model, "_canonical_repn")
    _safe_remove_file(output_filename)

    StageToVariableMap = map_variable_stages(
        scenario,
        scenario_tree,
        symbol_map,
        enforce_derived_nonanticipativity=enforce_derived_nonanticipativity)
    firststage_variable_ids = \
        set(id(var) for symbol, var, scenario_tree_id
            in StageToVariableMap[firststage.name])
    secondstage_variable_ids = \
        set(id(var) for symbol, var, scenario_tree_id
            in StageToVariableMap[secondstage.name])

    StageToConstraintMap = \
        map_constraint_stages(
            scenario,
            scenario_tree,
            symbol_map,
            stochastic_constraint_ids,
            firststage_variable_ids,
            secondstage_variable_ids)
    secondstage_constraint_ids = \
        set(id(con) for symbols, con
            in StageToConstraintMap[secondstage.name])

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
    objective_object = scenario._instance_objective
    assert objective_object is not None
    objective_block = objective_object.parent_block()
    objective_repn = canonical_repn_cache[id(objective_block)][objective_object]

    #
    # Create column (variable) ordering maps for LP/MPS files
    #
    column_order = ComponentMap()
    firststage_variable_count = 0
    secondstage_variable_count = 0
    # first-stage variables
    for column_index, (symbol, var, scenario_tree_id) \
        in enumerate(StageToVariableMap[firststage.name]):
        column_order[var] = column_index
        firststage_variable_count += 1
    # second-stage variables
    for column_index, (symbol, var, scenario_tree_id) \
        in enumerate(StageToVariableMap[secondstage.name],
                     len(column_order)):
        column_order[var] = column_index
        secondstage_variable_count += 1
    # account for the ONE_VAR_CONSTANT second-stage variable
    # added by the LP writer
    secondstage_variable_count += 1

    #
    # Create row (constraint) ordering maps for LP/MPS files
    #
    firststage_constraint_count = 0
    secondstage_constraint_count = 0
    row_order = ComponentMap()
    # first-stage constraints
    for row_index, (symbols, con) \
        in enumerate(StageToConstraintMap[firststage.name]):
        row_order[con] = row_index
        firststage_constraint_count += len(symbols)
    # second-stage constraints
    for row_index, (symbols, con) \
        in enumerate(StageToConstraintMap[secondstage.name],
                     len(row_order)):
        row_order[con] = row_index
        secondstage_constraint_count += len(symbols)
    # account for the ONE_VAR_CONSTANT = 1 second-stage constraint
    # added by the LP writer
    secondstage_constraint_count += 1

    #
    # Create a custom labeler that allows DDSIP to identify
    # first-stage variables
    #
    if io_options.pop('symbolic_solver_labels', False):
        _labeler = TextLabeler()
    else:
        _labeler = NumericLabeler('x')
    labeler = lambda x: _labeler(x) + \
              (""
               if ((not isinstance(x, _VarData)) or \
                   (id(x) not in firststage_variable_ids)) else \
               firststage_var_suffix)

    #
    # Write the ordered LP/MPS file
    #
    output_filename = os.path.join(output_directory,
                                   scenario.name+".core.lp")
    with WriterFactory("lp") as writer:
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        assert 'labeler' not in io_options
        assert 'force_objective_constant' not in io_options
        io_options['column_order'] = column_order
        io_options['row_order'] = row_order
        io_options['force_objective_constant'] = True
        io_options['labeler'] = labeler
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
        symbol_map,
        enforce_derived_nonanticipativity=enforce_derived_nonanticipativity)

    StageToConstraintMap = map_constraint_stages(
        scenario,
        scenario_tree,
        symbol_map,
        stochastic_constraint_ids,
        firststage_variable_ids,
        secondstage_variable_ids)

    # generate a few data structures that are used
    # when writing the .sc files
    constraint_symbols = ComponentMap(
        (con, symbols) for stage_name in StageToConstraintMap
        for symbols, con in StageToConstraintMap[stage_name])

    #
    # Write the body of the .sc files
    #
    modified_constraint_lb = ComponentMap()
    modified_constraint_ub = ComponentMap()

    #
    # Stochastic RHS
    #
    # **NOTE: In the code that follows we assume the LP
    #         writer always moves constraint body
    #         constants to the rhs and that the lower part
    #         of any range constraints are written before
    #         the upper part.
    #
    stochastic_rhs_count = 0
    with open(os.path.join(output_directory,
                           scenario.name+".rhs.sc.struct"),'w') as f_rhs_struct:
        with open(os.path.join(output_directory,
                               scenario.name+".rhs.sc"),'w') as f_rhs:
            scenario_probability = scenario.probability
            rhs_struct_template = " %s\n"
            rhs_template = "  %.17g\n"
            f_rhs.write("scen\n%.17g\n"
                        % (_no_negative_zero(scenario_probability)))
            if stochastic_rhs is not None:
                for con, include_bound in stochastic_rhs_entries:
                    assert isinstance(con, _ConstraintData)
                    if not empty_rhs_annotation:
                        # verify that this constraint was
                        # flagged by PySP or the user as second-stage
                        if id(con) not in secondstage_constraint_ids:
                            raise RuntimeError(
                                "The constraint %s has been declared "
                                "in the %s annotation but it was not identified as "
                                "a second-stage constraint. To correct this issue, "
                                "either remove the constraint from this annotation "
                                "or manually declare it as second-stage using the "
                                "%s annotation."
                                % (con.cname(True),
                                   StochasticConstraintBoundsAnnotation.__name__,
                                   ConstraintStageAnnotation.__name__))

                    constraint_repn = \
                        canonical_repn_cache[id(con.parent_block())][con]
                    if not isinstance(constraint_repn, LinearCanonicalRepn):
                        raise RuntimeError("Only linear constraints are "
                                           "accepted for conversion to SMPS format. "
                                           "Constraint %s is not linear."
                                           % (con.cname(True)))

                    body_constant = constraint_repn.constant
                    # We are going to rewrite the core problem file
                    # with all stochastic values set to zero. This will
                    # allow an easy test for missing user annotations.
                    constraint_repn.constant = 0
                    if body_constant is None:
                        body_constant = 0.0
                    symbols = constraint_symbols[con]
                    assert len(symbols) > 0
                    for con_label in symbols:
                        if con_label.startswith('c_e_') or \
                           con_label.startswith('c_l_'):
                            assert (include_bound is True) or \
                                   (include_bound[0] is True)
                            stochastic_rhs_count += 1
                            f_rhs_struct.write(rhs_struct_template % (con_label))
                            f_rhs.write(rhs_template %
                                        (_no_negative_zero(
                                            value(con.lower) - \
                                            value(body_constant))))
                            # We are going to rewrite the core problem file
                            # with all stochastic values set to zero. This will
                            # allow an easy test for missing user annotations.
                            modified_constraint_lb[con] = con.lower
                            con._lower = _deterministic_check_value
                            if con_label.startswith('c_e_'):
                                modified_constraint_ub[con] = con.upper
                                con._upper = _deterministic_check_value
                        elif con_label.startswith('r_l_') :
                            if (include_bound is True) or \
                               (include_bound[0] is True):
                                stochastic_rhs_count += 1
                                f_rhs_struct.write(rhs_struct_template % (con_label))
                                f_rhs.write(rhs_template %
                                             (_no_negative_zero(
                                                 value(con.lower) - \
                                                 value(body_constant))))
                                # We are going to rewrite the core problem file
                                # with all stochastic values set to zero. This will
                                # allow an easy test for missing user annotations.
                                modified_constraint_lb[con] = con.lower
                                con._lower = _deterministic_check_value
                        elif con_label.startswith('c_u_'):
                            assert (include_bound is True) or \
                                   (include_bound[1] is True)
                            stochastic_rhs_count += 1
                            f_rhs_struct.write(rhs_struct_template % (con_label))
                            f_rhs.write(rhs_template %
                                        (_no_negative_zero(
                                            value(con.upper) - \
                                            value(body_constant))))
                            # We are going to rewrite the core problem file
                            # with all stochastic values set to zero. This will
                            # allow an easy test for missing user annotations.
                            modified_constraint_ub[con] = con.upper
                            con._upper = _deterministic_check_value
                        elif con_label.startswith('r_u_'):
                            if (include_bound is True) or \
                               (include_bound[1] is True):
                                stochastic_rhs_count += 1
                                f_rhs_struct.write(rhs_struct_template % (con_label))
                                f_rhs.write(rhs_template %
                                            (_no_negative_zero(
                                                value(con.upper) - \
                                                value(body_constant))))
                                # We are going to rewrite the core problem file
                                # with all stochastic values set to zero. This will
                                # allow an easy test for missing user annotations.
                                modified_constraint_ub[con] = con.upper
                                con._upper = _deterministic_check_value
                        else:
                            assert False

    #
    # Stochastic Matrix
    #
    stochastic_matrix_count = 0
    with open(os.path.join(output_directory,
                           scenario.name+".matrix.sc.struct"),'w') as f_mat_struct:
        with open(os.path.join(output_directory,
                               scenario.name+".matrix.sc"),'w') as f_mat:
            scenario_probability = scenario.probability
            matrix_struct_template = " %s %s\n"
            matrix_template = "  %.17g\n"
            f_mat.write("scen\n")
            if stochastic_matrix is not None:
                for con, var_list in stochastic_matrix_entries:
                    assert isinstance(con, _ConstraintData)
                    if not empty_matrix_annotation:
                        # verify that this constraint was
                        # flagged by PySP or the user as second-stage
                        if id(con) not in secondstage_constraint_ids:
                            raise RuntimeError(
                                "The constraint %s has been declared "
                                "in the %s annotation but it was not identified as "
                                "a second-stage constraint. To correct this issue, "
                                "either remove the constraint from this annotation "
                                "or manually declare it as second-stage using the "
                                "%s annotation."
                                % (con.cname(True),
                                   StochasticConstraintBodyAnnotation.__name__,
                                   ConstraintStageAnnotation.__name__))
                    constraint_repn = \
                        canonical_repn_cache[id(con.parent_block())][con]
                    if not isinstance(constraint_repn, LinearCanonicalRepn):
                        raise RuntimeError("Only linear constraints are "
                                           "accepted for conversion to SMPS format. "
                                           "Constraint %s is not linear."
                                           % (con.cname(True)))
                    assert len(constraint_repn.variables) > 0
                    if var_list is None:
                        var_list = constraint_repn.variables
                    assert len(var_list) > 0
                    symbols = constraint_symbols[con]
                    # sort the variable list by the column ordering
                    # so that we have deterministic output
                    var_list = list(var_list)
                    var_list.sort(key=lambda _v: column_order[_v])
                    new_coefs = list(constraint_repn.linear)
                    for var in var_list:
                        assert isinstance(var, _VarData)
                        assert not var.fixed
                        var_coef = None
                        for i, (_var, coef) in enumerate(zip(constraint_repn.variables,
                                                            constraint_repn.linear)):
                            if _var is var:
                                var_coef = coef
                                # We are going to rewrite with core problem file
                                # with all stochastic values set to zero. This will
                                # allow an easy test for missing user annotations.
                                new_coefs[i] = _deterministic_check_value
                                break
                        if var_coef is None:
                            raise RuntimeError(
                                "The coefficient for variable %s has "
                                "been marked as stochastic in constraint %s using "
                                "the %s annotation, but the variable does not appear"
                                " in the canonical constraint expression."
                                % (var.cname(True),
                                   con.cname(True),
                                   StochasticConstraintBodyAnnotation.__name__))
                        var_label = symbol_map.byObject[id(var)]
                        for con_label in symbols:
                            stochastic_matrix_count += 1
                            f_mat_struct.write(matrix_struct_template
                                              % (con_label, var_label))
                            f_mat.write(matrix_template
                                        % (_no_negative_zero(value(var_coef))))

                    constraint_repn.linear = tuple(new_coefs)

    #
    # Stochastic Objective
    #
    stochastic_cost_count = 0
    with open(os.path.join(output_directory,
                           scenario.name+".cost.sc.struct"),'w') as f_obj_struct:
        with open(os.path.join(output_directory,
                               scenario.name+".cost.sc"),'w') as f_obj:
            obj_struct_template = " %s\n"
            obj_template = "  %.17g\n"
            f_obj.write("scen\n")
            if stochastic_objective is not None:
                if stochastic_objective.has_declarations():
                    sorted_values = stochastic_objective.expand_entries()
                    assert len(sorted_values) <= 1
                    if len(sorted_values) == 0:
                        raise RuntimeError(
                            "The %s annotation was declared "
                            "with external entries but no active Objective "
                            "objects were recovered from those entries."
                            % (StochasticObjectiveAnnotation.__name__))
                    obj, (objective_variables, include_constant) = \
                        sorted_values[0]
                    assert obj is objective_object
                else:
                    objective_variables, include_constant = \
                        stochastic_objective.default

                if not isinstance(objective_repn, LinearCanonicalRepn):
                    raise RuntimeError("Only linear stochastic objectives are "
                                       "accepted for conversion to SMPS format. "
                                       "Objective %s is not linear."
                                       % (objective_object.cname(True)))
                if objective_variables is None:
                    objective_variables = objective_repn.variables
                stochastic_objective_label = symbol_map.byObject[id(objective_object)]
                # sort the variable list by the column ordering
                # so that we have deterministic output
                objective_variables = list(objective_variables)
                objective_variables.sort(key=lambda _v: column_order[_v])
                stochastic_lp_labels.add(stochastic_objective_label)
                assert (len(objective_variables) > 0) or include_constant
                new_coefs = list(objective_repn.linear)
                for var in objective_variables:
                    assert isinstance(var, _VarData)
                    var_coef = None
                    for i, (_var, coef) in enumerate(zip(objective_repn.variables,
                                                        objective_repn.linear)):
                        if _var is var:
                            var_coef = coef
                            # We are going to rewrite the core problem file
                            # with all stochastic values set to zero. This will
                            # allow an easy test for missing user annotations.
                            new_coefs[i] = _deterministic_check_value
                            break
                    if var_coef is None:
                        raise RuntimeError(
                            "The coefficient for variable %s has "
                            "been marked as stochastic in objective %s using "
                            "the %s annotation, but the variable does not appear"
                            " in the canonical objective expression."
                            % (var.cname(True),
                               objective_object.cname(True),
                               StochasticObjectiveAnnotation.__name__))
                    var_label = symbol_map.byObject[id(var)]
                    stochastic_cost_count += 1
                    f_obj_struct.write(obj_struct_template % (var_label))
                    f_obj.write(obj_template
                                % (_no_negative_zero(value(var_coef))))

                objective_repn.linear = tuple(new_coefs)
                if include_constant:
                    obj_constant = objective_repn.constant
                    # We are going to rewrite the core problem file
                    # with all stochastic values set to zero. This will
                    # allow an easy test for missing user annotations.
                    objective_repn.constant = _deterministic_check_value
                    if obj_constant is None:
                        obj_constant = 0.0
                    stochastic_cost_count += 1
                    f_obj_struct.write(obj_struct_template % ("ONE_VAR_CONSTANT"))
                    f_obj.write(obj_template % (_no_negative_zero(obj_constant)))

    #
    # Write the deterministic part of the LP/MPS-file to its own
    # file for debugging purposes
    #
    reference_model_name = reference_model.name
    reference_model.name = "ZeroStochasticData"
    det_output_filename = os.path.join(output_directory,
                                       scenario.name+".pysp_model.lp.det")
    with WriterFactory("lp") as writer:
        output_fname, symbol_map = writer(reference_model,
                                          det_output_filename,
                                          lambda x: True,
                                          io_options)
        assert output_fname == det_output_filename
    reference_model.name = reference_model_name

    # reset bounds on any constraints that were modified
    for con, lower in iteritems(modified_constraint_lb):
        con._lower = lower
    for con, upper in iteritems(modified_constraint_ub):
        con._upper = upper

    return (firststage_variable_count,
            secondstage_variable_count,
            firststage_constraint_count,
            secondstage_constraint_count,
            stochastic_cost_count,
            stochastic_rhs_count,
            stochastic_matrix_count)

def convert_external(output_directory,
                     firststage_var_suffix,
                     scenario_tree_manager,
                     enforce_derived_nonanticipativity=False,
                     io_options=None,
                     disable_consistency_checks=False,
                     keep_scenario_files=False,
                     keep_auxiliary_files=False,
                     verbose=False):
    import pyomo.environ
    import pyomo.solvers.plugins.smanager.phpyro

    if io_options is None:
        io_options = {}

    assert os.path.exists(output_directory)

    scenario_tree = scenario_tree_manager.scenario_tree

    if scenario_tree.contains_bundles():
        raise ValueError(
            "DDSIP conversion does not yet handle bundles")

    scenario_directory = os.path.join(output_directory,
                                      'scenario_files')

    if not os.path.exists(scenario_directory):
        os.mkdir(scenario_directory)

    counts = scenario_tree_manager.invoke_function(
        "_convert_external_setup",
        thisfile,
        invocation_type=InvocationType.PerScenario,
        function_args=(scenario_directory,
                       firststage_var_suffix,
                       enforce_derived_nonanticipativity,
                       io_options))

    reference_scenario = scenario_tree.scenarios[0]
    reference_scenario_name = reference_scenario.name
    (firststage_variable_count,
     secondstage_variable_count,
     firststage_constraint_count,
     secondstage_constraint_count,
     stochastic_cost_count,
     stochastic_rhs_count,
     stochastic_matrix_count) = counts[reference_scenario_name]

    #
    # Copy the reference scenario's core, row, col, and tim
    # to the output directory. The consistency checks will
    # verify that these files match across scenarios.
    #
    input_files = []
    lp_dst = os.path.join(output_directory, "core.lp")
    _safe_remove_file(lp_dst)
    lp_src = os.path.join(scenario_directory,
                          reference_scenario_name+".core.lp")
    shutil.copy2(lp_src, lp_dst)
    input_files.append(lp_dst)

    #
    # Merge the per-scenario .sc files into one
    #
    for basename in ["rhs.sc", "cost.sc", "matrix.sc"]:

        if basename == "cost.sc":
            if stochastic_cost_count == 0:
                continue
        elif basename == "matrix.sc":
            if stochastic_matrix_count == 0:
                continue
        else:
            assert basename == "rhs.sc"
            # Note: DDSIP requires that the RHS files always
            #       exists because it contains the scenario
            #       probabilities
            pass

        dst = os.path.join(output_directory, basename)
        input_files.append(dst)
        _safe_remove_file(dst)
        with open(dst, "w") as fdst:
            fdst.write("Names\n")
            assert reference_scenario is scenario_tree.scenarios[0]
            src = os.path.join(scenario_directory,
                               reference_scenario.name+"."+basename+".struct")
            assert os.path.exists(src)
            with open(src, "r") as fsrc:
                shutil.copyfileobj(fsrc, fdst)
            for scenario in scenario_tree.scenarios:
                src = os.path.join(scenario_directory,
                                   scenario.name+"."+basename)
                assert os.path.exists(src)
                with open(src, "r") as fsrc:
                    shutil.copyfileobj(fsrc, fdst)

    if verbose:
        print("\nDDSIP Conversion Complete")
        print("Output Saved To: "+os.path.relpath(output_directory))
        print("Basic Problem Information:")
        print(" - Variables:")
        print("   - First Stage: %d"
              % (firststage_variable_count))
        print("   - Second Stage: %d"
              % (secondstage_variable_count))
        print(" - Constraints:")
        print("   - First Stage: %d"
              % (firststage_constraint_count))
        print("   - Second Stage: %d"
              % (secondstage_constraint_count))
        print("   - Stoch. RHS Entries: %d"
              % (stochastic_rhs_count))
        print("   - Stoch. Matrix Entries: %d"
              % (stochastic_matrix_count))
        print(" - Objective:")
        print("    - Stoch. Cost Entries: %d"
              % (stochastic_cost_count))

    """
    if not disable_consistency_checks:
        if verbose:
            print("\nStarting scenario structure consistency checks "
                  "across scenario files stored in %s."
                  % (scenario_directory))
            print("This may take some time. If this test is "
                  "prohibitively slow or can not be executed on "
                  "your system, disable it by activating the "
                  "disable_consistency_check option.")
        has_diff = False
        try:
            if not os.system('diff --help > /dev/null'):
                has_diff = True
            else:
                has_diff = False
        except:
            has_diff = False
        if verbose:
            print(" - Checking row and column ordering...")
        for scenario in scenario_tree.scenarios:
            scenario_core_row_filename = \
                os.path.join(scenario_directory,
                             basename+".row."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_core_row_filename+' '+
                               core_row_filename)
            else:
                rc = not filecmp.cmp(scenario_core_row_filename,
                                     core_row_filename,
                                     shallow=False)
            if rc:
                raise ValueError(
                    "The row ordering indicated in file '%s' does not match "
                    "that for scenario %s indicated in file '%s'. This "
                    "suggests that the same constraint is being classified "
                    "in different time stages across scenarios. Consider "
                    "manually declaring constraint stages using the %s "
                    "annotation if not already doing so, or report this "
                    "issue to the PySP developers."
                    % (core_row_filename,
                       scenario.name,
                       scenario_core_row_filename,
                       ConstraintStageAnnotation.__name__))

            scenario_core_col_filename = \
                os.path.join(scenario_directory,
                             basename+".col."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_core_col_filename+' '+
                               core_col_filename)
            else:
                rc = not filecmp.cmp(scenario_core_col_filename,
                                     core_col_filename,
                                     shallow=False)
            if rc:
                raise ValueError(
                    "The column ordering indicated in file '%s' does not "
                    "match that for scenario %s indicated in file '%s'. "
                    "This suggests that the set of variables on the model "
                    "changes across scenarios. This is not allowed by the "
                    "SMPS format. If you feel this is a developer error, "
                    "please report this issue to the PySP developers."
                    % (core_col_filename,
                       scenario.name,
                       scenario_core_col_filename))

        if verbose:
            print(" - Checking time-stage classifications...")
        for scenario in scenario_tree.scenarios:
            scenario_tim_filename = \
                os.path.join(scenario_directory,
                             basename+".tim."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_tim_filename+' '+
                               tim_filename)
            else:
                rc = not filecmp.cmp(scenario_tim_filename,
                                     tim_filename,
                                     shallow=False)
            if rc:
                raise ValueError(
                    "Main .tim file '%s' does not match .tim file for "
                    "scenario %s located at '%s'. This indicates there was "
                    "a problem translating the reference model to SMPS "
                    "format. Please make sure the problem structure is "
                    "identical over all scenarios (e.g., no. of variables, "
                    "no. of constraints), or report this issue to the PySP "
                    "developers if you feel that it is a developer error."
                    % (tim_filename,
                       scenario.name,
                       scenario_tim_filename))

        if verbose:
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
                                     sto_struct_filename,
                                     shallow=False)
            if rc:
                raise ValueError(
                    "The structure of stochastic entries indicated in file "
                    "'%s' does not match that for scenario %s indicated in "
                    "file '%s'. This suggests that the set of variables "
                    "appearing in some expression declared as stochastic is "
                    "changing across scenarios. If you feel this is a "
                    "developer error, please report this issue to the PySP "
                    "developers." % (sto_struct_filename,
                                     scenario.name,
                                     scenario_sto_struct_filename))

        if verbose:
            print(" - Checking deterministic sections in the core "
                  "problem file...")
        for scenario in scenario_tree.scenarios:
            scenario_core_det_filename = \
                os.path.join(scenario_directory,
                             basename+"."+core_format+".det."+scenario.name)
            if has_diff:
                rc = os.system('diff -q '+scenario_core_det_filename+' '+
                               core_det_filename)
            else:
                rc = not filecmp.cmp(scenario_core_det_filename,
                                     core_det_filename,
                                     shallow=False)
            if rc:
                raise ValueError(
                    "One or more deterministic parts of the problem found "
                    "in file '%s' do not match those for scenario %s found "
                    "in file %s. This suggests that one or more locations "
                    "of stochastic data have not been been annotated on the "
                    "reference Pyomo model. If this seems like a tolerance "
                    "issue or a developer error, please report this issue "
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
        if verbose:
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
        if verbose:
            print("Temporary per-scenario files are retained in "
                  "scenario_files subdirectory")
        pass
    """

    return (ProblemStats(firststage_variable_count=firststage_variable_count,
                         secondstage_variable_count=secondstage_variable_count,
                         firststage_constraint_count=firststage_constraint_count,
                         secondstage_constraint_count=secondstage_constraint_count,
                         stochastic_cost_count=stochastic_cost_count,
                         stochastic_rhs_count=stochastic_rhs_count,
                         stochastic_matrix_count=stochastic_matrix_count,
                         scenario_count=len(scenario_tree.scenarios)),
            input_files)
