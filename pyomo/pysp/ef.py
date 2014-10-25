import pyutilib
import sys
import tempfile
import shutil
import os
import time
import traceback
import copy
import gc
import weakref

from pyomo.pysp.scenariotree import *
from pyomo.pysp.convergence import *
from pyomo.pysp.ph import *
from pyomo.pysp.phutils import *

from pyomo.core.base import *

from pyomo.opt.results.solution import Solution

from pyutilib.misc import ArchiveReaderFactory, ArchiveReader

from six import iteritems, itervalues, advance_iterator

#
# a routine to create the extensive form, given an input scenario tree and instances.
# IMPT: unlike scenario instances, the extensive form instance is *not* self-contained.
#       in particular, it has binding constraints that cross the binding instance and
#       the scenario instances. it is up to the caller to keep track of which scenario
#       instances are associated with the extensive form. this might be something we
#       encapsulate at some later time.

#
# GAH: The following comment is still true, except that the scenario tree is no longer modified
#
# NOTE: if cvar terms are generated, then the input scenario tree is modified accordingly,
#       i.e., with the addition of the "eta" variable at the root node and the excess
#       variables at (for lack of a better place - they are per-scenario, but are not
#       blended) the second stage.
#

def create_ef_instance(scenario_tree,
                       ef_instance_name = "MASTER",
                       verbose_output = False,
                       generate_weighted_cvar = False,
                       cvar_weight = None,
                       risk_alpha = None,
                       cc_indicator_var_name=None,
                       cc_alpha=0.0):

    #
    # create the new and empty binding instance.
    #

    # scenario tree must be "linked" with a set of instances
    # to used this function
    scenario_instances = dict((scenario._name,scenario._instance) \
                              for scenario in scenario_tree._scenarios)

    binding_instance = ConcreteModel()
    binding_instance.name = ef_instance_name
    root_node = scenario_tree.findRootNode()

    opt_sense = minimize \
                if (scenario_tree._scenarios[0]._instance_objective.is_minimizing()) \
                   else maximize

    #
    # validate cvar options, if specified.
    #
    cvar_excess_vardatas = []
    if generate_weighted_cvar:
        if (cvar_weight is None) or (cvar_weight < 0.0):
            raise RuntimeError("Weight of CVaR term must be >= 0.0 - value supplied="+str(cvar_weight))
        if (risk_alpha is None) or (risk_alpha <= 0.0) or (risk_alpha >= 1.0):
            raise RuntimeError("CVaR risk alpha must be between 0 and 1, exclusive - value supplied="+str(risk_alpha))

        if verbose_output:
            print("Writing CVaR weighted objective")
            print("CVaR term weight="+str(cvar_weight))
            print("CVaR alpha="+str(risk_alpha))
            print("")

        # create the eta and excess variable on a per-scenario basis,
        # in addition to the constraint relating to the two.

        cvar_eta_variable_name = "CVAR_ETA_"+str(root_node._name)
        cvar_eta_variable = Var()
        binding_instance.add_component(cvar_eta_variable_name, cvar_eta_variable)

        excess_var_domain = NonNegativeReals if (opt_sense == minimize) else \
                            NonPositiveReals

        compute_excess_constraint = \
            binding_instance.COMPUTE_SCENARIO_EXCESS = \
                ConstraintList(noruleinit=True)

        for scenario in scenario_tree._scenarios:

            cvar_excess_variable_name = "CVAR_EXCESS_"+scenario._name
            cvar_excess_variable = Var(domain=excess_var_domain)
            binding_instance.add_component(cvar_excess_variable_name, cvar_excess_variable)

            compute_excess_expression = cvar_excess_variable
            compute_excess_expression -= scenario._instance_cost_expression
            compute_excess_expression += cvar_eta_variable
            if opt_sense == maximize:
                compute_excess_expression *= -1

            compute_excess_constraint.add((0.0, compute_excess_expression, None))

            cvar_excess_vardatas.append((cvar_excess_variable, scenario._probability))

    # the individual scenario instances are sub-blocks of the binding instance.
    for scenario in scenario_tree._scenarios:
        scenario_instance = scenario_instances[scenario._name]
        binding_instance.add_component(str(scenario._name), scenario_instance)
        # Now deactivate the scenario instance Objective since we are creating
        # a new master objective
        scenario._instance_objective.deactivate()

    # walk the scenario tree - create variables representing the
    # common values for all scenarios associated with that node, along
    # with equality constraints to enforce non-anticipativity.  also
    # create expected cost variables for each node, to be computed via
    # constraints/objectives defined in a subsequent pass. master
    # variables are created for all nodes but those in the last
    # stage. expected cost variables are, for no particularly good
    # reason other than easy coding, created for nodes in all stages.
    if verbose_output:
        print("Creating variables for master binding instance")

    _cmap = binding_instance.MASTER_CONSTRAINT_MAP = ComponentMap()
    for stage in scenario_tree._stages[:-1]: # skip the leaf stage

        for tree_node in stage._tree_nodes:

            # create the master blending variable and constraints for this node
            master_blend_variable_name = "MASTER_BLEND_VAR_"+str(tree_node._name)
            master_blend_constraint_name = "MASTER_BLEND_CONSTRAINT_"+str(tree_node._name)

            master_variable_index = Set(initialize=tree_node._standard_variable_ids,
                                        ordered=Set.SortedOrder,
                                        name=master_blend_variable_name+"_index")

            binding_instance.add_component(master_blend_variable_name+"_index",
                                           master_variable_index)

            # don't post non-anticipativity constraints for derived
            # variables, so here we index the master blending variable
            # for only the standard blended variable ids
            master_variable = Var(master_variable_index,
                                  name=master_blend_variable_name)

            binding_instance.add_component(master_blend_variable_name,
                                           master_variable)

            master_constraint = ConstraintList(name=master_blend_constraint_name,
                                               noruleinit=True)

            binding_instance.add_component(master_blend_constraint_name,
                                           master_constraint)

            tree_node_variable_datas = tree_node._variable_datas
            for variable_id in master_variable_index:

                master_vardata = master_variable[variable_id]

                vardatas = tree_node_variable_datas[variable_id]

                # Don't blend fixed variables
                if not tree_node.is_variable_fixed(variable_id):

                    for scenario_vardata, scenario_probability in vardatas:

                        _cmap[scenario_vardata] = master_constraint.add((master_vardata-scenario_vardata,0.0))

    if generate_weighted_cvar:

        cvar_cost_expression_name = "CVAR_COST_"+str(root_node._name)
        cvar_cost_expression = Expression(name=cvar_cost_expression_name)
        binding_instance.add_component(cvar_cost_expression_name, cvar_cost_expression)

    # create an expression to represent the expected cost at the root node
    binding_instance.EF_EXPECTED_COST = \
        Expression(initialize=sum(scenario._probability*scenario._instance_cost_expression \
                                  for scenario in scenario_tree._scenarios))

    opt_expression = \
        binding_instance.MASTER_OBJECTIVE_EXPRESSION = \
            Expression(initialize=binding_instance.EF_EXPECTED_COST)

    if generate_weighted_cvar:
        cvar_cost_expression_name = "CVAR_COST_"+str(root_node._name)
        cvar_cost_expression = \
            binding_instance.find_component(cvar_cost_expression_name)
        if cvar_weight == 0.0:
            # if the cvar weight is 0, then we're only
            # doing cvar - no mean.
            opt_expression.value = cvar_cost_expression
        else:
            opt_expression.value += cvar_weight * cvar_cost_expression

    binding_instance.MASTER = Objective(sense=opt_sense,
                                        expr=opt_expression)

    # CVaR requires the addition of a variable per scenario to
    # represent the cost excess, and a constraint to compute the cost
    # excess relative to eta.
    if generate_weighted_cvar:

        # add the constraint to compute the master CVaR variable value. iterate
        # over scenario instances to create the expected excess component first.
        cvar_cost_expression_name = "CVAR_COST_"+str(root_node._name)
        cvar_cost_expression = binding_instance.find_component(cvar_cost_expression_name)
        cvar_eta_variable_name = "CVAR_ETA_"+str(root_node._name)
        cvar_eta_variable = binding_instance.find_component(cvar_eta_variable_name)

        cost_expr = 1.0
        for scenario_excess_vardata, scenario_probability in cvar_excess_vardatas:
            cost_expr += (scenario_probability * scenario_excess_vardata)
        cost_expr /= (1.0 - risk_alpha)
        cost_expr += cvar_eta_variable

        cvar_cost_expression.value = cost_expr

    if cc_indicator_var_name is not None:
        if verbose_output is True:
            print("Creating chance constraint for indicator variable= "+cc_indicator_var_name)
            print( "with alpha= "+str(cc_alpha))
        if not isVariableNameIndexed(cc_indicator_var_name):
            cc_expression = 0  #??????
            for scenario in scenario_tree._scenarios:
                scenario_instance = scenario_instances[scenario._name]
                scenario_probability = scenario._probability
                cc_var = scenario_instance.find_component(cc_indicator_var_name)

                cc_expression += scenario_probability * cc_var

            def makeCCRule(expression):
                def CCrule(model):
                    return(1.0 - cc_alpha, cc_expression, None)
                return CCrule

            cc_constraint_name = "cc_"+cc_indicator_var_name
            cc_constraint = Constraint(name=cc_constraint_name, rule=makeCCRule(cc_expression))
            binding_instance.add_component(cc_constraint_name, cc_constraint)
        else:
            print("multiple cc not yet supported.")
            variable_name, index_template = extractVariableNameAndIndex(cc_indicator_var_name)

            # verify that the root variable exists and grab it.
            # NOTE: we are using whatever scenario happens to laying around... it might be better to use the reference
            variable = scenario_instance.find_component(variable_name)
            if variable is None:
                raise RuntimeError("Unknown variable="+variable_name+" referenced as the CC indicator variable.")

            # extract all "real", i.e., fully specified, indices matching the index template.
            match_indices = extractVariableIndices(variable, index_template)

            # there is a possibility that no indices match the input template.
            # if so, let the user know about it.
            if len(match_indices) == 0:
                raise RuntimeError("No indices match template="+str(index_template)+" for variable="+variable_name)

            # add the suffix to all variable values identified.
            for index in match_indices:
                variable_value = variable[index]

                cc_expression = 0  #??????
                for scenario in scenario_tree._scenarios:
                    scenario_instance = scenario_instances[scenario._name]
                    scenario_probability = scenario._probability
                    cc_var = scenario_instance.find_component(variable_name)[index]

                    cc_expression += scenario_probability * cc_var

                def makeCCRule(expression):
                    def CCrule(model):
                        return(1.0 - cc_alpha, cc_expression, None)
                    return CCrule

                indexasname = ''
                for c in str(index):
                   if c not in ' ,':
                      indexasname += c
                cc_constraint_name = "cc_"+variable_name+"_"+indexasname

                cc_constraint = Constraint(name=cc_constraint_name, rule=makeCCRule(cc_expression))
                binding_instance.add_component(cc_constraint_name, cc_constraint)

    return binding_instance

#
# write the EF binding instance and all sub-instances.
#

def write_ef(binding_instance,
             output_filename,
             symbolic_solver_labels=False,
             output_fixed_variable_bounds=False):

    # determine the output file type, and invoke the appropriate
    # writer.
    pieces = output_filename.rsplit(".",1)
    if len(pieces) != 2:
        raise RuntimeError("Could not determine suffix from output filename="+output_filename)
    ef_output_file_suffix = pieces[1]

    # create the output file.
    if ef_output_file_suffix == "lp":

        symbol_map = binding_instance.write(
            filename=output_filename,
            format=ProblemFormat.cpxlp,
            solver_capability=lambda x: True,
            io_options={"symbolic_solver_labels":symbolic_solver_labels,
                        "output_fixed_variable_bounds":output_fixed_variable_bounds})

    elif ef_output_file_suffix == "nl":

        symbol_map = binding_instance.write(
            filename=output_filename,
            format=ProblemFormat.nl,
            solver_capability=lambda x: True,
            io_options={"symbolic_solver_labels":symbolic_solver_labels,
                        "output_fixed_variable_bounds":output_fixed_variable_bounds})

    else:
        raise RuntimeError("Unknown file suffix="+ef_output_file_suffix+
                           " specified when writing extensive form")
