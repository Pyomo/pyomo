#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import Var, Block, Set, Objective, Constraint, SortComponents, SOSConstraint, Piecewise, BuildAction, Param, Any, Binary

from pyomo.repn.standard_repn import (preprocess_block_objectives,
                                      preprocess_block_constraints,
                                      preprocess_constraint)
from pyomo.opt import (UndefinedData,
                       undefined)

from six import iteritems, itervalues, string_types
from six.moves import xrange

def _preprocess(model, objective=True, constraints=True):
    objective_found = False
    if objective:
        for block in model.block_data_objects(active=True):
            for obj in block.component_data_objects(Objective,
                                                    active=True,
                                                    descend_into=False):
                objective_found = True
                preprocess_block_objectives(block)
                break
            if objective_found:
                break
    if constraints:
        for block in model.block_data_objects(active=True):
            preprocess_block_constraints(block)

_OLD_OUTPUT = True

def extract_solve_times(results, default=undefined):
    solve_time = default
    pyomo_solve_time = default
    # if the solver plugin doesn't populate the
    # user_time field, it is by default of type
    # UndefinedData - defined in pyomo.opt.results
    if hasattr(results.solver,"user_time") and \
       (not isinstance(results.solver.user_time,
                       UndefinedData)) and \
       (results.solver.user_time is not None):
        # the solve time might be a string, or might
        # not be - we eventually would like more
        # consistency on this front from the solver
        # plugins.
        solve_time = float(results.solver.user_time)
    elif hasattr(results.solver,"wallclock_time") and \
         (not isinstance(results.solver.wallclock_time,
                         UndefinedData))and \
         (results.solver.wallclock_time is not None):
        solve_time = float(results.solver.wallclock_time)
    elif hasattr(results.solver,"time"):
        solve_time = float(results.solver.time)

    if hasattr(results,"pyomo_solve_time"):
        pyomo_solve_time = results.pyomo_solve_time

    return solve_time, pyomo_solve_time

class BasicSymbolMap(object):

    def __init__(self):

        # maps object id()s to their assigned symbol.
        self.byObject = {}

        # maps assigned symbols to the corresponding objects.
        self.bySymbol = {}

    def getByObjectDictionary(self):
        return self.byObject

    def updateSymbols(self, data_stream):
        # check if the input is a generator / iterator,
        # if so, we need to copy since we use it twice
        if hasattr(data_stream, '__iter__') and \
           not hasattr(data_stream, '__len__'):
            obj_symbol_tuples = list(obj_symbol_tuples)
        self.byObject.update((id(obj), label) for obj,label in data_stream)
        self.bySymbol.update((label,obj) for obj,label in data_stream)

    def createSymbol(self, obj ,label):
        self.byObject[id(obj)] = label
        self.bySymbol[label] = obj

    def addSymbol(self, obj, label):
        self.byObject[id(obj)] = label
        self.bySymbol[label] = obj

    def getSymbol(self, obj):
        return self.byObject[id(obj)]

    def getObject(self, label):
        return self.bySymbol[label]

    def pprint(self, **kwds):
        print("BasicSymbolMap:")
        lines = [repr(label)+" <-> "+obj.name+" (id="+str(id(obj))+")"
                 for label, obj in iteritems(self.bySymbol)]
        print('\n'.join(sorted(lines)))
        print("")

#
# Creates a deterministic symbol map for ctypes on a Block. This
# allows convenient transmission of information to and from
# PHSolverServers and makes it easy to save solutions using a
# pickleable list (symbol,values) tuples
#
def create_block_symbol_maps(owner_block,
                             ctypes,
                             recursive=True,
                             update_new=False,
                             update_all=False):
    """
    Inputs:
      - owner_block: A constructed Pyomo Block
      -      ctypes: An iterable of ctypes (or single ctype) to
                     include in the _PHInstanceSymbolMaps dict
      -   recursive: Indicates whether to include ctypes contained on
                     subblocks in the given Block's SymbolMaps
      -  update_new: Update a possibly existing _PHInstanceSymbolMaps
                     dict on the owner_block with new ctypes. Any ctypes
                     given as inputs that already have existing SymbolMaps
                     will NOT be regenerated. **See: update_all
      -  update_all: Update a possibly existing _PHInstanceSymbolMaps
                     dict on the owner block with new ctypes,
                     regenerating any existing ctypes given as inputs.
    Outputs: A dictionary (keys are ctypes) of SymbolMaps is placed on
             the owner_block and is named _PHInstanceSymbolMaps
    """

    if owner_block.is_constructed() is False:
        raise ValueError("Failed to create _PHInstanceSymbolMap on Block %s. "\
                         "This Block has not been fully construced." % owner_block.name)

    # The ctypes input may be iterable or a single type.
    # Either way turn it into a tuple of type(s)
    try:
        ctypes = tuple(ct for ct in ctypes)
    except TypeError:
        ctypes = (ctypes,)

    # Add the _PHInstanceSymbolMaps dict to the owner_block and
    # and warn if one already exists as it will be overwritten
    phinst_sm_dict = getattr(owner_block,"_PHInstanceSymbolMaps",None)
    if phinst_sm_dict is not None:
        if (update_new is False) and (update_all is False):
            print("***WARNING - Attribute with name _PHInstanceSymbolMaps already exists " \
                  "on Block %s. This Attribute will be overwritten" % owner_block.name)
            phinst_sm_dict = owner_block._PHInstanceSymbolMaps = {}
        else:
            if type(phinst_sm_dict) is not dict:
                raise TypeError("Failed to update _PHInstanceSymbolMaps attribute "\
                                "on Block %s. Expected to find object with type %s "\
                                "but existing object is of type %s." % (owner_block.name,
                                                                        dict,
                                                                        type(phinst_sm_dict)))

    else:
        phinst_sm_dict = owner_block._PHInstanceSymbolMaps = {}

    ctypes_to_generate = []
    for ctype in ctypes:
        if (ctype not in phinst_sm_dict) or (update_all is True):
            ctypes_to_generate.append(ctype)

    # Create a SymbolMap for each ctype in the _PHInstanceSymbolMaps dict
    for ctype in ctypes_to_generate:
        phinst_sm_dict[ctype] = BasicSymbolMap()

    # Create the list of Blocks to iterator over. If this is recursive
    # turn the block_data_objects() generator into a list, as we need to sort by
    # keys (for indexed blocks) so this process remains deterministic,
    # and we don't want to sort more than necessary
    block_list = None
    if recursive is True:
        # FIXME: Why do you alphabetize the components by name here?  It
        # would be more efficient to use SortComponents.deterministic.
        # [JDS 12/31/14]
        block_list = tuple(owner_block.block_data_objects(active=True,
                                                          sort=SortComponents.alphabetizeComponentAndIndex))
    else:
        block_list = (owner_block,)

    for ctype in ctypes_to_generate:

        ctype_sm = phinst_sm_dict[ctype]
        bySymbol = ctype_sm.bySymbol
        cntr = 0
        for block in block_list:
            bySymbol.update(enumerate( \
                    block.component_data_objects(ctype,
                                                 descend_into=False,
                                                 sort=SortComponents.alphabetizeComponentAndIndex),
                    cntr))
            cntr += len(bySymbol)-cntr+1

        ctype_sm.byObject = dict((id(component_data),symbol) for symbol,component_data in iteritems(bySymbol))

#
# a utility to scan through a scenario tree and set the variable
# values to None for each variable not in the final stage that has not
# converged. necessary to avoid an illegal / infeasible warm-start for
# the extensive form solve.
#

def reset_nonconverged_variables(scenario_tree, scenario_instances):

   for stage in scenario_tree._stages[:-1]:

      for tree_node in stage._tree_nodes:

         for variable_id, var_datas in iteritems(tree_node._variable_datas):

             min_var_value = tree_node._minimums[variable_id]
             max_var_value = tree_node._maximums[variable_id]

             # TBD: THIS IS A HACK - GET THE THRESHOLD FROM SOMEWHERE AS AN INPUT ARGUMENT
             if (max_var_value - min_var_value) > 0.00001:

                 for var_data, probability in var_datas:

                     if not var_data.fixed:

                         var_data.value = None
                         var_data.stale = True

#
# a utility to clear all cost variables - these are easily derived by the solvers,
# and can/will lead to infeasibilities unless everything is perfectly blended.
# in which case, you don't need to write the EF.
#

def reset_stage_cost_variables(scenario_tree, scenario_instances):

    for stage in scenario_tree._stages:
        for tree_node in stage._tree_nodes:
            for cost_var_data, scenario_probability in tree_node._cost_variable_datas:
                if cost_var_data.is_expression_type() is False:
                    cost_var_data.value = None
                    cost_var_data.stale = True

#
# a utility to clear the value of any PHQUADPENALTY* variables in the instance. these are
# associated with linearization, and if they are not cleared, can interfere with warm-starting
# due to infeasibilities.
#

def reset_linearization_variables(instance):

    for variable_name, variable in iteritems(instance.component_map(Var, active=True)):
        if variable_name.startswith("PHQUADPENALTY"):
            for var_value in itervalues(variable):
                var_value.value = None
                var_value.stale = True

#
# a simple utility function to pretty-print an index tuple into a [x,y] form string.
#

_nontuple = string_types + (int, float)
def indexToString(index):

    if index is None:
        return ''

    # if the input type is a string or an int, then this isn't a tuple!
    # TODO: Why aren't we just checking for tuple?
    if isinstance(index, _nontuple):
        return "["+str(index)+"]"

    result = "["
    for i in range(0,len(index)):
        result += str(index[i])
        if i != len(index) - 1:
            result += ","
    result += "]"
    return result

#
# a simple utility to determine if a variable name contains an index specification.
# in other words, is the reference to a complete variable (e.g., "foo") - which may
# or may not be indexed - or a specific index or set of indices (e.g., "foo[1]" or
# or "foo[1,*]".
#

def isVariableNameIndexed(variable_name):

    left_bracket_count = variable_name.count('[')
    right_bracket_count = variable_name.count(']')

    if (left_bracket_count == 1) and (right_bracket_count == 1):
        return True
    elif (left_bracket_count == 1) or (right_bracket_count == 1):
        raise ValueError("Illegally formed variable name="+variable_name+"; if indexed, variable names must contain matching left and right brackets")
    else:
        return False

#
# takes a string indexed of the form "('foo', 'bar')" and returns a proper tuple ('foo','bar')
#

def tupleizeIndexString(index_string):

    index_string=index_string.lstrip('(')
    index_string=index_string.rstrip(')')
    pieces = index_string.split(',')
    return_index = ()
    for piece in pieces:
        piece = piece.strip()
        piece = piece.lstrip('\'')
        piece = piece.rstrip('\'')
        transformed_component = None
        try:
            transformed_component = int(piece)
        except ValueError:
            transformed_component = piece
        return_index = return_index + (transformed_component,)

    # IMPT: if the tuple is a singleton, return the element itself.
    if len(return_index) == 1:
        return return_index[0]
    else:
        return return_index

#
# related to above, extract the index from the variable name.
# will throw an exception if the variable name isn't indexed.
# the returned variable name is a string, while the returned
# index is a tuple. integer values are converted to integers
# if the conversion works!
#

def extractVariableNameAndIndex(variable_name):

    if not isVariableNameIndexed(variable_name):
        raise ValueError(
            "Non-indexed variable name passed to "
            "function extractVariableNameAndIndex()")

    pieces = variable_name.split('[')
    name = pieces[0].strip()
    full_index = pieces[1].rstrip(']')

    # even nested tuples in pyomo are "flattened" into
    # one-dimensional tuples. to accomplish flattening
    # replace all parens in the string with commas and
    # proceed with the split.
    full_index = full_index.replace("(",",").replace(")",",")
    indices = full_index.split(',')

    return_index = ()

    for index in indices:

        # unlikely, but strip white-space from the string.
        index=index.strip()

        # if the tuple contains nested tuples, then the nested
        # tuples have single quotes - "'" characters - around
        # strings. remove these, as otherwise you have an
        # illegal index.
        index = index.replace("\'","")

        # if the index is an integer, make it one!
        transformed_index = None
        try:
            transformed_index = int(index)
        except ValueError:
            transformed_index = index
        return_index = return_index + (transformed_index,)

    # IMPT: if the tuple is a singleton, return the element itself.
    if len(return_index) == 1:
        return name, return_index[0]
    else:
        return name, return_index

#
# determine if the input index is an instance of the template,
# which may or may not contain wildcards.
#

def indexMatchesTemplate(index, index_template):

    # if the input index is not a tuple, make it one.
    # ditto with the index template. one-dimensional
    # indices in pyomo are not tuples, but anything
    # else is.

    if type(index) != tuple:
        index = (index,)
    if type(index_template) != tuple:
        index_template = (index_template,)

    if len(index) != len(index_template):
        return False

    for i in xrange(0,len(index_template)):
        if index_template[i] == '*':
            # anything matches
            pass
        else:
            if index_template[i] != index[i]:
                return False

    return True

#
# given a component (the real object, not the name) and an
# index template, "shotgun" the index and see which variable
# indices match the template. the cardinality could be >
# 1 if slices are specified, e.g., [*,1].
#
# NOTE: This logic can be expensive for scenario trees with many
#       nodes, and for variables with many indices. thus, the
#       logic behind the indexMatchesTemplate utility above
#       is in-lined in an efficient way here.
#

def extractComponentIndices(component, index_template):

    component_index_dimension = component.dim()

    # do special handling for the case where the component is
    # not indexed, i.e., of dimension 0. for scalar components,
    # the match template can be the empty string, or - more
    # commonly, given that the empty string is hard to specify
    # in the scenario tree input data - a single wildcard character.
    if component_index_dimension == 0:
       if (index_template != '') and (index_template != "*"):
          raise RuntimeError(
              "Index template=%r specified for scalar object=%s"
              % (index_template, component.name))
       return [None]

    # from this point on, we're dealing with an indexed component.
    if index_template == "":
        return [ndx for ndx in component]

    # if the input index template is not a tuple, make it one.
    # one-dimensional indices in pyomo are not tuples, but
    # everything else is.
    if type(index_template) != tuple:
        index_template = (index_template,)

    if component_index_dimension != len(index_template):
        raise RuntimeError(
            "The dimension of index template=%s (%s) does match "
            "the dimension of component=%s (%s)"
            % (index_template,
               len(index_template),
               component.name,
               component_index_dimension))

    # cache for efficiency
    iterator_range = [i for i,match_str in enumerate(index_template)
                      if match_str != '*']

    if len(iterator_range) == 0:
        return list(component)
    elif len(iterator_range) == component_index_dimension:
        if (len(index_template) == 1) and \
           (index_template[0] in component):
            return index_template
        elif index_template in component:
            return [index_template]
        else:
            raise ValueError(
                "The index %s is not valid for component named: %s"
                % (str(tuple(index_template)), component.name))

    result = []

    for index in component:

        # if the input index is not a tuple, make it one for processing
        # purposes. however, return the original index always.
        if component_index_dimension == 1:
           modified_index = (index,)
        else:
           modified_index = index

        match_found = True # until proven otherwise
        for i in iterator_range:
            if index_template[i] != modified_index[i]:
                match_found = False
                break

        if match_found is True:
            result.append(index)

    return result

#
# method to eliminate constraints from an input instance.
# TBD: really need to generalize the name, as we also cull
#      build actions and other stuff.
#
def cull_constraints_from_instance(model, constraints_to_retain):

    for block in model.block_data_objects(active=True):
        for constraint_name, constraint in iteritems(block.component_map(Constraint)):
            if constraint_name not in constraints_to_retain:
                block.del_component(constraint_name)
        # Piecewise is a derived Block, so we have to look for it by sub-type.
        for constraint_name, constraint in iteritems(block.component_map(Block)):
            if isinstance(constraint, Piecewise) and \
               (constraint_name not in constraints_to_retain):
                block.del_component(constraint_name)
        for constraint_name, constraint in iteritems(block.component_map(SOSConstraint)):
            if constraint_name not in constraints_to_retain:
                block.del_component(constraint_name)
        for action_name, action in iteritems(block.component_map(BuildAction)):
            if action_name not in constraints_to_retain:
                block.del_component(action_name)
        # Prevent any new constraints from being declared (e.g. during
        # construction of nested block models)
        block._suppress_ctypes.add(Constraint)
        block._suppress_ctypes.add(SOSConstraint)
        block._suppress_ctypes.add(Piecewise)
        block._suppress_ctypes.add(BuildAction)

def update_all_rhos(instances, scenario_tree, rho_value=None, rho_scale=None):

    assert not ((rho_value is not None) and (rho_scale is not None))

    for stage in scenario_tree._stages[:-1]:

        for tree_node in stage._tree_nodes:

            for scenario in tree_node._scenarios:

                rho = scenario._rho[tree_node._name]

                for variable_id in tree_node._variable_ids:

                    if rho_value is not None:
                        rho[variable_id] = rho_value
                    else:
                        rho[variable_id] *= rho_scale


# creates all PH parameters for a problem instance, given a scenario tree
# (to identify the relevant variables), a default rho (simply for initialization),
# and a boolean indicating if quadratic penalty terms are to be linearized.
# returns a list of any created variables, specifically when linearizing -
# this is useful to clean-up reporting.

def create_ph_parameters(instance, scenario_tree, default_rho, linearizing_penalty_terms):

    new_penalty_variable_names = []

    # gather all variables, and their corresponding indices, that are referenced
    # in any tree node corresponding to this scenario - with the exception of the
    # leaf tree node (where PH doesn't blend variables).
    instance_variables = {}  # map between variable names and sets of indices

    scenario = scenario_tree.get_scenario(instance.name)
    if scenario == None:
        raise RuntimeError("Scenario corresponding to instance name="
                           +instance.name+" not present in scenario tree "
                           "- could not create PH parameters for instance")

    for tree_node in scenario._node_list[:-1]:

        new_w_parameter_name = "PHWEIGHT_"+str(tree_node._name)
        new_rho_parameter_name = "PHRHO_"+str(tree_node._name)
        if linearizing_penalty_terms > 0:
            new_penalty_term_variable_name = \
                "PHQUADPENALTY_"+str(tree_node._name)

        nodal_index_set_name = "PHINDEX_"+str(tree_node._name)
        nodal_index_set = instance.find_component(nodal_index_set_name)
        # TODO: This function requires calling
        # create_nodal_ph_paramters first
        assert nodal_index_set is not None

        ### dlw Jan 2014 nochecking=True, mutable=True)
        scenario._w[tree_node._name].update(
            dict.fromkeys(scenario._w[tree_node._name],0.0))
        new_w_parameter = \
            Param(nodal_index_set,
                  name=new_w_parameter_name,
                  initialize=scenario._w[tree_node._name],
                  mutable=True)
        ### dlw Jan 2014 nochecking=True, mutable=True)
        scenario._rho[tree_node._name].update(
            dict.fromkeys(scenario._rho[tree_node._name],default_rho))
        new_rho_parameter = \
            Param(nodal_index_set,
                  name=new_rho_parameter_name,
                  initialize=scenario._rho[tree_node._name],
                  domain=Any,
                  mutable=True)
        if linearizing_penalty_terms > 0:
            new_penalty_term_variable = \
                Var(nodal_index_set,
                    name=new_penalty_term_variable_name,
                    bounds=(0.0,None),
                    initialize=0.0)

        instance.add_component(new_w_parameter_name,new_w_parameter)
        instance.add_component(new_rho_parameter_name,new_rho_parameter)
        if linearizing_penalty_terms > 0:
            instance.add_component(new_penalty_term_variable_name, new_penalty_term_variable)
            new_penalty_variable_names.append(new_penalty_term_variable_name)

    return new_penalty_variable_names


# creates all PH node parameters for all instance, given a scenario
# tree

def create_nodal_ph_parameters(scenario_tree):

    for stage in scenario_tree._stages[:-1]:

        for tree_node in stage._tree_nodes:

            new_nodal_index_set_name = "PHINDEX_"+str(tree_node._name)
            new_xbar_parameter_name = "PHXBAR_"+str(tree_node._name)
            new_blend_parameter_name = "PHBLEND_"+str(tree_node._name)

            # only create nodal index sets for non-derived variables.
            new_nodal_index_set = Set(name=new_nodal_index_set_name,
                                      initialize=list(tree_node._standard_variable_ids))

            for scenario in tree_node._scenarios:
                instance = scenario._instance

                # avoid the warnings generated by adding Set to
                # multiple components, and learn to live with the fact
                # that these Params will point to some arbitrary
                # instance as their "parent" in the end
                new_nodal_index_set._parent = None

                # Add the shared parameter to the instance
                instance.add_component(new_nodal_index_set_name,
                                       new_nodal_index_set)

            ### dlw Jan 2014 nochecking=True, mutable=True)
            new_xbar_parameter = Param(new_nodal_index_set,
                                       name=new_xbar_parameter_name,
                                       default=0.0,
                                       mutable=True)
            ### dlw Jan 2014 nochecking=True, mutable=True)
            new_blend_parameter = Param(new_nodal_index_set,
                                        name=new_blend_parameter_name,
                                        within=Binary,
                                        default=False,
                                        mutable=True)

            for scenario in tree_node._scenarios:
                instance = scenario._instance

                # avoid the warnings generated by adding Param to
                # multiple components, and learn to live with the fact
                # that these Params will point to some arbitrary
                # instance as their "parent" in the end
                new_xbar_parameter._parent = None
                new_blend_parameter._parent = None

                # Add the shared parameter to the instance
                instance.add_component(new_xbar_parameter_name,
                                       new_xbar_parameter)
                instance.add_component(new_blend_parameter_name,
                                       new_blend_parameter)

            new_xbar_parameter.store_values(0.0)
            tree_node._xbars.update(dict.fromkeys(tree_node._xbars,0.0))
            new_blend_parameter.store_values(1)
            tree_node._blend.update(dict.fromkeys(tree_node._blend,1))

#
# Extracts an active objective from the instance (top-level only).
# Works with index objectives that may have all but one index
# deactivated. safety_checks=True asserts that exactly ONE active objective
# is found on the top-level instance.
#

def find_active_objective(instance, safety_checks=False):

    if safety_checks is False:
        for objective_data in instance.component_data_objects(Objective,
                                                              active=True,
                                                              descend_into=True):
            # Return the first active objective encountered
            return objective_data
    else:
        objectives = []
        for objective_data in instance.component_data_objects(Objective,
                                                              active=True,
                                                              descend_into=True):
            objectives.append(objective_data)
        if len(objectives) > 1:
            names = [o.name for o in objectives]
            raise AssertionError("More than one active objective was "
                                 "found on instance %s: %s"
                                 % (instance.name, names))
        if len(objectives) > 0:
            return objectives[0]
    return None

def preprocess_scenario_instance(scenario_instance,
                                 instance_variables_fixed,
                                 instance_variables_freed,
                                 instance_user_constraints_modified,
                                 instance_ph_constraints_modified,
                                 instance_ph_constraints,
                                 instance_objective_modified,
                                 preprocess_fixed_variables,
                                 solver):

    # TODO: Does this import need to be delayed because
    #       it is in a plugins subdirectory
    from pyomo.solvers.plugins.solvers.persistent_solver import \
        PersistentSolver

    persistent_solver_in_use = isinstance(solver, PersistentSolver)

    if (not instance_objective_modified) and \
       (not instance_variables_fixed) and \
       (not instance_variables_freed) and \
       (not instance_ph_constraints_modified) and \
       (not instance_user_constraints_modified):

        # the condition of "nothing modified" should only be triggered
        # at PH iteration 0. instances are already preprocessed
        # following construction, and there isn't any augmentation of
        # the objective function yet.
        return

    if instance_objective_modified:
        # if only the objective changed, there is minimal work to do.
        _preprocess(scenario_instance,
                    objective=True,
                    constraints=False)

        if persistent_solver_in_use:
            active_objective_datas = []
            for active_objective_data in scenario_instance.component_data_objects(Objective,
                                                                                  active=True,
                                                                                  descend_into=True):
                active_objective_datas.append(active_objective_data)
            if len(active_objective_datas) > 1:
                raise RuntimeError("Multiple active objectives identified for scenario=%s" % scenario_instance._name)
            elif len(active_objective_datas) == 1:
                solver.set_objective(active_objective_datas[0])

    if (instance_variables_fixed or instance_variables_freed) and \
       (preprocess_fixed_variables):

        _preprocess(scenario_instance)

        # We've preprocessed the entire instance, no point in checking
        # anything else
        return

    if (instance_variables_fixed or instance_variables_freed) and \
       (persistent_solver_in_use):
        # it can be the case that the solver plugin no longer has an
        # instance compiled, depending on what state the solver plugin
        # is in relative to the instance.  if this is the case, just
        # don't compile the variable bounds.
        if solver.has_instance():
            variables_to_change = \
                instance_variables_fixed + instance_variables_freed
            for var_name, var_index in variables_to_change:
                solver.update_var(scenario_instance.find_component(var_name)[var_index])

    if instance_user_constraints_modified:

        _preprocess(scenario_instance,
                    objective=False,
                    constraints=True)

    # TBD: Should this be an an if below - both user and ph constraints
    #      could be modified at the same time, no?
    elif instance_ph_constraints_modified:

        # only pre-process the piecewise constraints
        idMap = {}
        for constraint_name in instance_ph_constraints:
            preprocess_constraint(
                scenario_instance,
                getattr(scenario_instance, constraint_name),
                idMap=idMap)


# TBD: doesn't do much now... - SHOULD PROPAGATE FLAGS FROM _preprocess_scenario_instances...

def preprocess_bundle_instance(bundle_instance,
                               solver):

    # TODO: Does this import need to be delayed because
    #       it is in a plugins subdirectory
    from pyomo.solvers.plugins.solvers.persistent_solver import \
        PersistentSolver

    persistent_solver_in_use = isinstance(solver, PersistentSolver)

    if persistent_solver_in_use:
        active_objective_datas = []
        for active_objective_data in bundle_instance.component_data_objects(Objective,
                                                                              active=True,
                                                                              descend_into=True):
            active_objective_datas.append(active_objective_data)
        if len(active_objective_datas) > 1:
            raise RuntimeError("Multiple active objectives identified for bundle=%s" % bundle_instance._name)
        elif len(active_objective_datas) == 1:
            solver.set_objective(active_objective_datas[0])

def reset_ph_plugins(ph):
    for ph_plugin in ph._ph_plugins:
        ph_plugin.reset(ph)
