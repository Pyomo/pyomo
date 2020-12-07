#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import time
import sys
import argparse
import shutil
import filecmp
import logging

from pyomo.opt import WriterFactory
from pyomo.common.collections import ComponentMap
from pyomo.core.base.numvalue import value, as_numeric
from pyomo.core.base.var import _VarData
from pyomo.core.base.constraint import Constraint, _ConstraintData
from pyomo.core.base import TextLabeler, NumericLabeler
from pyomo.pysp.scenariotree.manager import InvocationType
from pyomo.pysp.annotations import (locate_annotations,
                                    StochasticConstraintBoundsAnnotation,
                                    StochasticConstraintBodyAnnotation,
                                    StochasticObjectiveAnnotation,
                                    StochasticVariableBoundsAnnotation)
from pyomo.pysp.convert.smps import (map_variable_stages,
                                     map_constraint_stages,
                                     build_repns,
                                     _safe_remove_file,
                                     _no_negative_zero,
                                     _deterministic_check_value,
                                     _deterministic_check_constant)
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_must_be_str)
from pyomo.pysp.scenariotree.manager import \
    (ScenarioTreeManagerClientSerial,
     ScenarioTreeManagerClientPyro)
from pyomo.pysp.util.misc import launch_command
import pyomo.environ

from six import iteritems

thisfile = os.path.abspath(__file__)

logger = logging.getLogger('pyomo.pysp')

def _convert_external_setup(worker, scenario, *args, **kwds):
    reference_model = scenario._instance
    #
    # We will be tweaking the standard_repn objects on objectives
    # and constraints, so cache anything related to this here so
    # that this function does not have any side effects on the
    # instance after returning
    #
    cached_attrs = []
    for block in reference_model.block_data_objects(
            active=True,
            descend_into=True):
        block_cached_attrs = {}
        if hasattr(block, "_gen_obj_repn"):
            block_cached_attrs["_gen_obj_repn"] = \
                block._gen_obj_repn
            del block._gen_obj_repn
        if hasattr(block, "_gen_con_repn"):
            block_cached_attrs["_gen_con_repn"] = \
                block._gen_con_repn
            del block._gen_con_repn
        if hasattr(block, "_repn"):
            block_cached_attrs["_repn"] = \
                block._repn
            del block._repn
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
            if hasattr(block, "_gen_obj_repn"):
                del block._gen_obj_repn
            if hasattr(block, "_gen_con_repn"):
                del block._gen_con_repn
            if hasattr(block, "_repn"):
                del block._repn
            for name in block_cached_attrs:
                setattr(block, name, block_cached_attrs[name])

def _convert_external_setup_without_cleanup(
        worker,
        scenario,
        output_directory,
        firststage_var_suffix,
        enforce_derived_nonanticipativity,
        io_options):
    assert os.path.exists(output_directory)

    io_options = dict(io_options)
    scenario_tree = worker.scenario_tree
    reference_model = scenario._instance
    rootnode = scenario_tree.findRootNode()
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
        if stochastic_rhs.has_declarations:
            empty_rhs_annotation = False
            stochastic_rhs_entries = stochastic_rhs.expand_entries()
            stochastic_rhs_entries.sort(
                key=lambda x: x[0].getname(True, constraint_name_buffer))
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
        if stochastic_matrix.has_declarations:
            empty_matrix_annotation = False
            stochastic_matrix_entries = stochastic_matrix.expand_entries()
            stochastic_matrix_entries.sort(
                key=lambda x: x[0].getname(True, constraint_name_buffer))
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
            "The DDSIP writer does not currently support "
            "stochastic variable bounds. Invalid annotation type: %s"
            % (StochasticVariableBoundsAnnotation.__name__))

    if (stochastic_rhs is None) and \
       (stochastic_matrix is None) and \
       (stochastic_objective is None):
        raise RuntimeError(
            "No stochastic annotations found. DDSIP "
            "conversion requires at least one of the following "
            "annotation types:\n - %s\n - %s\n - %s"
            % (StochasticConstraintBoundsAnnotation.__name__,
               StochasticConstraintBodyAnnotation.__name__,
               StochasticObjectiveAnnotation.__name__))

    assert not hasattr(reference_model, "_repn")
    repn_cache = build_repns(reference_model)
    assert hasattr(reference_model, "_repn")
    assert not reference_model._gen_obj_repn
    assert not reference_model._gen_con_repn
    # compute values
    for block_repns in repn_cache.values():
        for repn in block_repns.values():
            repn.constant = value(repn.constant)
            repn.linear_coefs = [value(c) for c in repn.linear_coefs]
            repn.quadratic_coefs = [value(c) for c in repn.quadratic_coefs]

    #
    # Write the LP file once to obtain the symbol map
    #
    output_filename = os.path.join(output_directory,
                                   scenario.name+".lp.setup")
    with WriterFactory("lp") as writer:
        assert 'column_order' not in io_options
        assert 'row_order' not in io_options
        output_fname, symbol_map = writer(reference_model,
                                          output_filename,
                                          lambda x: True,
                                          io_options)
        assert output_fname == output_filename
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

    #
    # Make sure the objective references all first stage variables.
    # We do this by directly modifying the _repn of the
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
    objective_repn = repn_cache[id(objective_block)][objective_object]

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
                                   scenario.name+".lp")
    symbols_filename = os.path.join(output_directory,
                                    scenario.name+".lp.symbols")
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
        # write the lp file symbol paired with the scenario
        # tree id for each variable in the root node
        with open(symbols_filename, "w") as f:
            st_symbol_map = reference_model._ScenarioTreeSymbolMap
            lines = []
            for id_ in sorted(rootnode._variable_ids):
                var = st_symbol_map.bySymbol[id_]
                if not var.is_expression_type():
                    lp_label = symbol_map.byObject[id(var)]
                    lines.append("%s %s\n" % (lp_label, id_))
            f.writelines(lines)

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
                                "remove the constraint from this annotation."
                                % (con.name,
                                   StochasticConstraintBoundsAnnotation.__name__))

                    constraint_repn = \
                        repn_cache[id(con.parent_block())][con]
                    if not constraint_repn.is_linear():
                        raise RuntimeError("Only linear constraints are "
                                           "accepted for conversion to DDSIP format. "
                                           "Constraint %s is not linear."
                                           % (con.name))

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
                            con._lower = _deterministic_check_constant
                            if con_label.startswith('c_e_'):
                                modified_constraint_ub[con] = con.upper
                                con._upper = _deterministic_check_constant
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
                                con._lower = _deterministic_check_constant
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
                            con._upper = _deterministic_check_constant
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
                                con._upper = _deterministic_check_constant
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
                                "remove the constraint from this annotation."
                                % (con.name,
                                   StochasticConstraintBodyAnnotation.__name__))
                    constraint_repn = \
                        repn_cache[id(con.parent_block())][con]
                    if not constraint_repn.is_linear():
                        raise RuntimeError("Only linear constraints are "
                                           "accepted for conversion to DDSIP format. "
                                           "Constraint %s is not linear."
                                           % (con.name))
                    assert len(constraint_repn.linear_vars) > 0
                    if var_list is None:
                        var_list = constraint_repn.linear_vars
                    assert len(var_list) > 0
                    symbols = constraint_symbols[con]
                    # sort the variable list by the column ordering
                    # so that we have deterministic output
                    var_list = list(var_list)
                    var_list.sort(key=lambda _v: column_order[_v])
                    new_coefs = list(constraint_repn.linear_coefs)
                    for var in var_list:
                        assert isinstance(var, _VarData)
                        assert not var.fixed
                        var_coef = None
                        for i, (_var, coef) in enumerate(zip(constraint_repn.linear_vars,
                                                             constraint_repn.linear_coefs)):
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
                                % (var.name,
                                   con.name,
                                   StochasticConstraintBodyAnnotation.__name__))
                        var_label = symbol_map.byObject[id(var)]
                        for con_label in symbols:
                            stochastic_matrix_count += 1
                            f_mat_struct.write(matrix_struct_template
                                              % (con_label, var_label))
                            f_mat.write(matrix_template
                                        % (_no_negative_zero(value(var_coef))))

                    constraint_repn.linear_coefs = tuple(new_coefs)

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
                if stochastic_objective.has_declarations:
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

                if not objective_repn.is_linear():
                    raise RuntimeError("Only linear stochastic objectives are "
                                       "accepted for conversion to DDSIP format. "
                                       "Objective %s is not linear."
                                       % (objective_object.name))
                if objective_variables is None:
                    objective_variables = objective_repn.linear_vars
                stochastic_objective_label = symbol_map.byObject[id(objective_object)]
                # sort the variable list by the column ordering
                # so that we have deterministic output
                objective_variables = list(objective_variables)
                objective_variables.sort(key=lambda _v: column_order[_v])
                assert (len(objective_variables) > 0) or include_constant
                new_coefs = list(objective_repn.linear_coefs)
                for var in objective_variables:
                    assert isinstance(var, _VarData)
                    var_coef = None
                    for i, (_var, coef) in enumerate(zip(objective_repn.linear_vars,
                                                         objective_repn.linear_coefs)):
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
                            % (var.name,
                               objective_object.name,
                               StochasticObjectiveAnnotation.__name__))
                    var_label = symbol_map.byObject[id(var)]
                    stochastic_cost_count += 1
                    f_obj_struct.write(obj_struct_template % (var_label))
                    f_obj.write(obj_template
                                % (_no_negative_zero(value(var_coef))))

                objective_repn.linear_coefs = tuple(new_coefs)
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
    reference_model._name = "ZeroStochasticData"
    det_output_filename = os.path.join(output_directory,
                                       scenario.name+".lp.det")
    with WriterFactory("lp") as writer:
        output_fname, symbol_map = writer(reference_model,
                                          det_output_filename,
                                          lambda x: True,
                                          io_options)
        assert output_fname == det_output_filename
    reference_model._name = reference_model_name

    # reset bounds on any constraints that were modified
    for con, lower in iteritems(modified_constraint_lb):
        con._lower = as_numeric(lower)
    for con, upper in iteritems(modified_constraint_ub):
        con._upper = as_numeric(upper)

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
    input_files = {}
    lp_dst = os.path.join(output_directory, "core.lp")
    _safe_remove_file(lp_dst)
    lp_src = os.path.join(scenario_directory,
                          reference_scenario_name+".lp")
    shutil.copy2(lp_src, lp_dst)
    input_files["core"] = lp_dst

    symbols_dst = os.path.join(output_directory, "core.lp.symbols")
    _safe_remove_file(symbols_dst)
    symbols_src = os.path.join(scenario_directory,
                               reference_scenario_name+".lp.symbols")
    shutil.copy2(symbols_src, symbols_dst)
    input_files["symbols"] = symbols_dst

    #
    # Merge the per-scenario .sc files into one
    #
    for _type in ["rhs", "cost", "matrix"]:

        basename = _type+".sc"
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

        dst = os.path.join(output_directory, basename)
        input_files[_type] = dst
        _safe_remove_file(dst)
        with open(dst, "w") as fdst:
            # Note: If the RHS file is going to be empty
            #       then we must leave out the "Names" line
            if not ((_type == "rhs") and
                    (stochastic_rhs_count == 0)):
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
            print(" - Checking structure in stochastic files...")
        for basename in ["rhs.sc.struct", "cost.sc.struct", "matrix.sc.struct"]:
            reference_struct_filename = os.path.join(
                scenario_directory,
                reference_scenario.name+"."+basename)
            for scenario in scenario_tree.scenarios:
                scenario_struct_filename = \
                    os.path.join(scenario_directory,
                                 scenario.name+"."+basename)
                if has_diff:
                    rc = os.system('diff -q '+scenario_struct_filename+' '+
                                   reference_struct_filename)
                else:
                    rc = not filecmp.cmp(scenario_struct_filename,
                                         reference_struct_filename,
                                         shallow=False)
                if rc:
                    raise ValueError(
                        "The structure indicated in file '%s' does not match "
                        "that for scenario %s indicated in file '%s'. This "
                        "suggests one or more locations of stachastic data "
                        "have not been annotated. If you feel this message is "
                        "in error, please report this issue to the PySP "
                        "developers."
                        % (reference_struct_filename,
                           scenario.name,
                           scenario_struct_filename))

        if verbose:
            print(" - Checking deterministic sections in the core "
                  "problem file...")
        reference_lp_det_filename = \
            os.path.join(scenario_directory,
                         reference_scenario.name+".lp.det")

        for scenario in scenario_tree.scenarios:
            scenario_lp_det_filename = \
                os.path.join(scenario_directory,
                             scenario.name+".lp.det")
            if has_diff:
                rc = os.system('diff -q '+scenario_lp_det_filename+' '+
                               reference_lp_det_filename)
            else:
                rc = not filecmp.cmp(scenario_lp_det_filename,
                                     reference_lp_det_filename,
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
                    % (reference_lp_det_filename,
                       scenario.name,
                       scenario_lp_det_filename))

    if not keep_scenario_files:
        if verbose:
            print("Cleaning temporary per-scenario files")

        for scenario in scenario_tree.scenarios:

            for basename in ["lp", "lp.det", "lp.symbols",
                             "matrix.sc", "matrix.sc.struct",
                             "cost.sc", "cost.sc.struct",
                             "rhs.sc", "rhs.sc.struct"]:
                scenario_filename = \
                    os.path.join(scenario_directory,
                                 scenario.name+"."+basename)
                assert os.path.exists(scenario_filename)
                _safe_remove_file(scenario_filename)

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

    config_filename = os.path.join(output_directory,
                                   "ddsip.config")
    with open(config_filename, 'w') as f:
        f.write("BEGIN \n\n\n")
        f.write("FIRSTCON "+str(firststage_constraint_count)+"\n")
        f.write("FIRSTVAR "+str(firststage_variable_count)+"\n")
        f.write("SECCON "+str(secondstage_constraint_count)+"\n")
        f.write("SECVAR "+str(secondstage_variable_count)+"\n")
        f.write("POSTFIX "+firststage_var_suffix+"\n")
        f.write("SCENAR "+str(len(scenario_tree.scenarios))+"\n")
        f.write("STOCRHS "+str(stochastic_rhs_count)+"\n")
        f.write("STOCCOST "+str(stochastic_cost_count)+"\n")
        f.write("STOCMAT "+str(stochastic_matrix_count)+"\n")
        f.write("\n\nEND\n")
    input_files["config"] = config_filename

    script_filename = \
        os.path.join(output_directory,
                     "ddsip.stdin")
    # hacked by DLW, November 2016: the model file is now
    # first and the config file is second. So ddsiputils
    # gets it almost right.
    with open(script_filename, "w") as f:
        f.write(os.path.relpath(input_files["core"], output_directory)+"\n")
        f.write(os.path.relpath(input_files["config"], output_directory)+"\n")
        assert "rhs" in input_files
        f.write(os.path.relpath(input_files["rhs"], output_directory)+"\n")
        if "cost" in input_files:
            f.write(os.path.relpath(input_files["cost"], output_directory)+"\n")
        if "matrix" in input_files:
            f.write(os.path.relpath(input_files["matrix"], output_directory)+"\n")
    input_files["script"] = script_filename

    return input_files

def convertddsip_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
    safe_register_common_option(options, "verbose")
    safe_register_common_option(options, "symbolic_solver_labels")
    safe_register_unique_option(
        options,
        "output_directory",
        PySPConfigValue(
            ".",
            domain=_domain_must_be_str,
            description=(
                "The directory in which all DDSIP files "
                "will be stored. Default is '.'."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "first_stage_suffix",
        PySPConfigValue(
            "__DDSIP_FIRSTSTAGE",
            domain=_domain_must_be_str,
            description=(
                "The suffix used to identify first-stage variables. "
                "Default: '__DDSIP_FIRSTSTAGE'"
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "enforce_derived_nonanticipativity",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Adds nonanticipativity constraints for variables flagged "
                "as derived within their respective time stage (except for "
                "the final time stage). The default behavior behavior is "
                "to treat derived variables as belonging to the final "
                "time stage."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "disable_consistency_checks",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Disables consistency checks that attempt to find issues "
                "with the DDSIP conversion. By default, these checks are run "
                "after conversion takes place and leave behind a temporary "
                "directory with per-scenario output files if the checks fail. "
                "This option is not recommended, but can be used if the "
                "consistency checks are prohibitively slow."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "keep_scenario_files",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Keeps around the per-scenario DDSIP files created for testing "
                "whether a conversion is valid (whether or not the validation "
                "checks are performed). These files can be useful for "
                "debugging purposes."
            ),
            doc=None,
            visibility=0))
    safe_register_common_option(options, "scenario_tree_manager")
    ScenarioTreeManagerClientSerial.register_options(options)
    ScenarioTreeManagerClientPyro.register_options(options)

    return options

#
# Convert a PySP scenario tree formulation to DDSIP input files
#

def run_convertddsip(options):

    if not os.path.exists(options.output_directory):
        os.makedirs(options.output_directory)

    start_time = time.time()

    io_options = {'symbolic_solver_labels':
                  options.symbolic_solver_labels}

    assert not options.compile_scenario_instances

    manager_class = None
    if options.scenario_tree_manager == 'serial':
        manager_class = ScenarioTreeManagerClientSerial
    elif options.scenario_tree_manager == 'pyro':
        manager_class = ScenarioTreeManagerClientPyro

    with manager_class(options) as scenario_tree_manager:
        scenario_tree_manager.initialize()
        files = convert_external(
            options.output_directory,
            options.first_stage_suffix,
            scenario_tree_manager,
            enforce_derived_nonanticipativity=\
            options.enforce_derived_nonanticipativity,
            io_options=io_options,
            disable_consistency_checks=\
            options.disable_consistency_checks,
            keep_scenario_files=options.keep_scenario_files,
            verbose=options.verbose)

    end_time = time.time()

    print("")
    print("Total execution time=%.2f seconds"
          % (end_time - start_time))

#
# the main driver routine for the convertddsip script.
#

def main(args=None):
    #
    # Top-level command that executes everything
    #

    #
    # Import plugins
    #
    import pyomo.environ

    #
    # Parse command-line options.
    #
    options = PySPConfigBlock()
    convertddsip_register_options(options)

    #
    # Prevent the compile_scenario_instances option from
    # appearing on the command line. This script relies on
    # the original constraints being present on the model
    #
    argparse_val = options.get('compile_scenario_instances')._argparse
    options.get('compile_scenario_instances')._argparse = None

    try:
        ap = argparse.ArgumentParser(prog='pyomo.pysp.convert.ddsip')
        options.initialize_argparse(ap)

        # restore the option so the class validation does not
        # raise an exception
        options.get('compile_scenario_instances')._argparse = argparse_val

        options.import_argparse(ap.parse_args(args=args))
    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(run_convertddsip,
                          options,
                          error_label="pyomo.pysp.convert.ddsip: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

if __name__ == "__main__":
    main(args=sys.argv[1:])
