#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Problem Writer for BARON .bar Format Files
#

import logging
import itertools
from pyutilib.math import infinity

import pyomo.util.plugin
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter
from pyomo.core.base import (SortComponents,
                             SymbolMap,
                             AlphaNumTextLabeler,
                             NumericLabeler,
                             BooleanSet, Constraint,
                             IntegerSet, Objective,
                             Var, Param)
from pyomo.core.base.numvalue import is_fixed, value
from pyomo.core.base.set_types import *
#CLH: EXPORT suffixes "constraint_types" and "branching_priorities"
#     pass their respective information to the .bar file
from pyomo.core.base.suffix import active_export_suffix_generator
from pyomo.repn import LinearCanonicalRepn

from six import iteritems, StringIO, iterkeys
from six.moves import xrange

logger = logging.getLogger('pyomo.core')

# TODO: The to_string function is handy, but the fact that
#       it calls .name under the hood for all components
#       everywhere they are used will present ENORMOUS
#       overhead for components that have a large index set.
#       It might be worth adding an extra keyword to that
#       function that takes a "labeler" or "symbol_map" for
#       writing non-expression components.

# TODO: Is the precision used by to_string for writing
#       numeric values suitable for output to a solver?
#       In the LP and NL writer we used %.17g for all
#       numbers. This does get used in this writer
#       but not for numbers appearing in the objective
#       or constraints (which are written from to_string)

# TODO: The variables section needs to be written after the
#       objectives and constraint expressions have been
#       queried. We only want to include the variables that
#       were actually used. This affects the .stale flag on
#       variables when loading results.

# TODO: Add support for output_fixed_variable_bounds

class ProblemWriter_bar(AbstractProblemWriter):

    #pyomo.util.plugin.alias('baron_writer')
    pyomo.util.plugin.alias('bar', 'Generate the corresponding BARON BAR file.')

    def __init__(self):

        AbstractProblemWriter.__init__(self, ProblemFormat.bar)

        #Copied from cpxlp.py:
        # Keven Hunter made a nice point about using %.16g in his attachment
        # to ticket #4319. I am adjusting this to %.17g as this mocks the
        # behavior of using %r (i.e., float('%r'%<number>) == <number>) with
        # the added benefit of outputting (+/-). The only case where this
        # fails to mock the behavior of %r is for large (long) integers (L),
        # which is a rare case to run into and is probably indicative of
        # other issues with the model.
        # *** NOTE ***: If you use 'r' or 's' here, it will break code that
        #               relies on using '%+' before the formatting character
        #               and you will need to go add extra logic to output
        #               the number's sign.
        self._precision_string = '.17g'

    def _get_bound(self, exp):
        if exp is None:
            return None
        if is_fixed(exp):
            return value(exp)
        raise ValueError("non-fixed bound: " + str(exp))

    def __call__(self,
                 model,
                 output_filename,
                 solver_capability,
                 io_options):

        # Make sure not to modify the user's dictionary, they may be
        # reusing it outside of this call
        io_options = dict(io_options)

        # NOTE: io_options is a simple dictionary of keyword-value
        #       pairs specific to this writer.
        symbolic_solver_labels = \
            io_options.pop("symbolic_solver_labels", False)
        labeler = io_options.pop("labeler", None)

        # How much effort do we want to put into ensuring the
        # LP file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)

        sorter = SortComponents.unsorted
        if file_determinism >= 1:
            sorter = sorter | SortComponents.indices
            if file_determinism >= 2:
                sorter = sorter | SortComponents.alphabetical

        # TODO
        #output_fixed_variable_bounds = \
        #    io_options.pop("output_fixed_variable_bounds", False)

        # Skip writing constraints whose body section is fixed (i.e.,
        # no variables)
        skip_trivial_constraints = \
            io_options.pop("skip_trivial_constraints", False)

        # Note: Baron does not allow specification of runtime
        #       option outside of this file, so we add support
        #       for them here
        solver_options = io_options.pop("solver_options", {})

        if len(io_options):
            raise ValueError(
                "ProblemWriter_baron_writer passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(io_options)))

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError("Baron problem writer: Using both the "
                             "'symbolic_solver_labels' and 'labeler' "
                             "I/O options is forbidden")

        if output_filename is None:
            output_filename = model.name + ".bar"

        output_file=open(output_filename, "w")

        # Process the options. Rely on baron to catch
        # and reset bad option values
        output_file.write("OPTIONS {\n")
        summary_found = False
        if len(solver_options):
            for key, val in iteritems(solver_options):
                if (key.lower() == 'summary'):
                    summary_found = True
                if key.endswith("Name"):
                    output_file.write(key+": \""+str(val)+"\";\n")
                else:
                    output_file.write(key+": "+str(val)+";\n")
        if not summary_found:
            # The 'summary option is defaulted to 0, so that no
            # summary file is generated in the directory where the
            # user calls baron. Check if a user explicitly asked for
            # a summary file.
            output_file.write("Summary: 0;\n")
        output_file.write("}\n\n")

        if symbolic_solver_labels:
            labeler = AlphaNumTextLabeler()
        elif labeler is None:
            labeler = NumericLabeler('x')

        symbol_map = SymbolMap()
        sm_bySymbol = symbol_map.bySymbol
        referenced_variable_ids = set()

        #cache frequently called functions
        create_symbol_func = SymbolMap.createSymbol
        create_symbols_func = SymbolMap.createSymbols
        alias_symbol_func = SymbolMap.alias

        # Cache the list of model blocks so we don't have to call
        # model.block_data_objects() many many times, which is slow
        # for indexed blocks
        all_blocks_list = list(model.block_data_objects(active=True,
                                                        sort=sorter,
                                                        descend_into=True))
        active_components_data_var = {}
        for block in all_blocks_list:
            tmp = active_components_data_var[id(block)] = \
                  list(obj for obj in block.component_data_objects(Var,
                                                                   active=True,
                                                                   sort=sorter,
                                                                   descend_into=False))
            create_symbols_func(symbol_map, tmp, labeler)

            # GAH: Not sure this is necessary, and also it would break for
            #      non-mutable indexed params so I am commenting out for now.
            #for param_data in active_components_data(block, Param, sort=sorter):
                #instead of checking if param_data._mutable:
                #if not param_data.is_constant():
                #    create_symbol_func(symbol_map, param_data, labeler)

        symbol_map_variable_ids = set(symbol_map.byObject.keys())
        object_symbol_dictionary = symbol_map.byObject

        def _skip_trivial(constraint_data):
            if skip_trivial_constraints:
                if isinstance(constraint_data, LinearCanonicalRepn):
                    if constraint_data.variables is None:
                        return True
                else:
                    if constraint_data.body.polynomial_degree() == 0:
                        return True
            return False

        #
        # Check for active suffixes to export
        #
        r_o_eqns = []
        c_eqns = []
        l_eqns = []
        branching_priorities_suffixes = []
        for block in all_blocks_list:
            for name, suffix in active_export_suffix_generator(block):
                if name == 'branching_priorities':
                    branching_priorities_suffixes.append(suffix)
                elif name == 'constraint_types':
                    for constraint_data, constraint_type in iteritems(suffix):
                        if not _skip_trivial(constraint_data):
                            if constraint_type.lower() == 'relaxationonly':
                                r_o_eqns.append(constraint_data)
                            elif constraint_type.lower() == 'convex':
                                c_eqns.append(constraint_data)
                            elif constraint_type.lower() == 'local':
                                l_eqns.append(constraint_data)
                            else:
                                raise ValueError(
                                    "A suffix '%s' contained an invalid value: %s\n"
                                    "Choices are: [relaxationonly, convex, local]"
                                    % (suffix.name, constraint_type))
                else:
                    raise ValueError(
                        "The BARON writer can not export suffix with name '%s'. "
                        "Either remove it from block '%s' or deactivate it."
                        % (block.name, name))

        non_standard_eqns = r_o_eqns + c_eqns + l_eqns

        # GAH 1/5/15: Substituting all non-alphanumeric characters for underscore
        #             in labeler so this manual update should no longer be needed
        #
        # If the text labeler is used, correct the labels to be
        # baron-allowed variable names
        # Change '(' and ')' to '__'
        # This way, for simple variable names like 'x(1_2)' --> 'x__1_2__'
        # FIXME: 7/21/14 This may break if users give variable names
        #        with two or more underscores together
        #if symbolic_solver_labels:
        #    for key,label in iteritems(object_symbol_dictionary):
        #        label = label.replace('(','___')
        #        object_symbol_dictionary[key] = label.replace(')','__')

        #
        # BINARY_VARIABLES, INTEGER_VARIABLES, POSITIVE_VARIABLES, VARIABLES
        #

        BinVars = []
        IntVars = []
        PosVars = []
        Vars = []
        for block in all_blocks_list:
            for var_data in active_components_data_var[id(block)]:

                if isinstance(var_data.domain, BooleanSet):
                    TypeList = BinVars
                elif isinstance(var_data.domain, IntegerSet):
                    TypeList = IntVars
                elif isinstance(var_data.domain, RealSet) and \
                     (var_data.lb is not None) and \
                     (var_data.lb >= 0):
                    TypeList = PosVars
                else:
                    TypeList = Vars

                var_name = object_symbol_dictionary[id(var_data)]
                #if len(var_name) > 15:
                #    logger.warning(
                #        "Variable symbol '%s' for variable %s exceeds maximum "
                #        "character limit for BARON. Solver may fail"
                #        % (var_name, var_data.name))

                TypeList.append(var_name)

        if len(BinVars) > 0:
            output_file.write('BINARY_VARIABLES ')
            for var_name in BinVars[:-1]:
                output_file.write(str(var_name)+', ')
            output_file.write(str(BinVars[-1])+';\n\n')
        if len(IntVars) > 0:
            output_file.write('INTEGER_VARIABLES ')
            for var_name in IntVars[:-1]:
                output_file.write(str(var_name)+', ')
            output_file.write(str(IntVars[-1])+';\n\n')

        output_file.write('POSITIVE_VARIABLES ')
        output_file.write('ONE_VAR_CONST__')
        for var_name in PosVars:
            output_file.write(', '+str(var_name))
        output_file.write(';\n\n')

        if len(Vars) > 0:
            output_file.write('VARIABLES ')
            for var_name in Vars[:-1]:
                output_file.write(str(var_name)+', ')
            output_file.write(str(Vars[-1])+';\n\n')

        #
        # LOWER_BOUNDS
        #

        LowerBoundHeader = False
        for block in all_blocks_list:
            for var_data in active_components_data_var[id(block)]:
                if var_data.fixed:
                    var_data_lb = var_data.value
                else:
                    var_data_lb = var_data.lb
                    if var_data_lb == -infinity:
                        var_data_lb = None

                if var_data_lb is not None:
                    if LowerBoundHeader is False:
                        output_file.write("LOWER_BOUNDS{\n")
                        LowerBoundHeader = True
                    name_to_output = object_symbol_dictionary[id(var_data)]
                    lb_string_template = '%s: %'+self._precision_string+';\n'
                    output_file.write(lb_string_template
                                      % (name_to_output, var_data_lb))

        if LowerBoundHeader:
            output_file.write("}\n\n")

        #
        # UPPER_BOUNDS
        #

        UpperBoundHeader = False
        for block in all_blocks_list:
            for var_data in active_components_data_var[id(block)]:
                if var_data.fixed:
                    var_data_ub = var_data.value
                else:
                    var_data_ub = var_data.ub
                    if var_data_ub == infinity:
                        var_data_ub = None

                if var_data_ub is not None:
                    if UpperBoundHeader is False:
                        output_file.write("UPPER_BOUNDS{\n")
                        UpperBoundHeader = True
                    name_to_output = object_symbol_dictionary[id(var_data)]
                    ub_string_template = '%s: %'+self._precision_string+';\n'
                    output_file.write(ub_string_template
                                      % (name_to_output, var_data_ub))

        if UpperBoundHeader:
            output_file.write("}\n\n")

        #
        # BRANCHING_PRIORITIES
        #

        # Specifyig priorities requires that the pyomo model has established an
        # EXTERNAL, float suffix called 'branching_priorities' on the model
        # object, indexed by the relevant variable
        BranchingPriorityHeader = False
        for suffix in branching_priorities_suffixes:
            for var_data, priority in iteritems(suffix):
                if priority is not None:
                    if not BranchingPriorityHeader:
                        output_file.write('BRANCHING_PRIORITIES{\n')
                        BranchingPriorityHeader = True
                    name_to_output = object_symbol_dictionary[id(var_data)]
                    output_file.write(name_to_output+': '+str(priority)+';\n')

        if BranchingPriorityHeader:
            output_file.write("}\n\n")

        #
        # EQUATIONS
        #

        #Equation Declaration
        n_roeqns = len(r_o_eqns)
        n_ceqns = len(c_eqns)
        n_leqns = len(l_eqns)
        eqns = []

        # Alias the constraints by declaration order since Baron does not
        # include the constraint names in the solution file. It is important
        # that this alias not clash with any real constraint labels, hence
        # the use of the ".c<integer>" template. It is not possible to declare
        # a component having this type of name when using standard syntax.
        # There are ways to do it, but it is unlikely someone will.
        order_counter = 0
        alias_template = ".c%d"
        output_file.write('EQUATIONS ')
        output_file.write("c_e_FIX_ONE_VAR_CONST__")
        order_counter += 1
        for block in all_blocks_list:

            for constraint_data in block.component_data_objects(Constraint,
                                                                active=True,
                                                                sort=sorter,
                                                                descend_into=False):

                if (not _skip_trivial(constraint_data)) and \
                   (constraint_data not in non_standard_eqns):

                    eqns.append(constraint_data)

                    con_symbol = \
                        create_symbol_func(symbol_map, constraint_data, labeler)
                    assert not con_symbol.startswith('.')
                    assert con_symbol != "c_e_FIX_ONE_VAR_CONST__"

                    alias_symbol_func(symbol_map,
                                      constraint_data,
                                      alias_template % order_counter)
                    output_file.write(", "+str(con_symbol))
                    order_counter += 1

        output_file.write(";\n\n")

        if n_roeqns > 0:
            output_file.write('RELAXATION_ONLY_EQUATIONS ')
            for i, constraint_data in enumerate(r_o_eqns):
                con_symbol = create_symbol_func(symbol_map, constraint_data, labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != "c_e_FIX_ONE_VAR_CONST__"
                alias_symbol_func(symbol_map,
                                  constraint_data,
                                  alias_template % order_counter)
                if i == n_roeqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                order_counter += 1

        if n_ceqns > 0:
            output_file.write('CONVEX_EQUATIONS ')
            for i, constraint_data in enumerate(c_eqns):
                con_symbol = create_symbol_func(symbol_map, constraint_data, labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != "c_e_FIX_ONE_VAR_CONST__"
                alias_symbol_func(symbol_map,
                                  constraint_data,
                                  alias_template % order_counter)
                if i == n_ceqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                order_counter += 1

        if n_leqns > 0:
            output_file.write('LOCAL_EQUATIONS ')
            for i, constraint_data in enumerate(l_eqns):
                con_symbol = create_symbol_func(symbol_map, constraint_data, labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != "c_e_FIX_ONE_VAR_CONST__"
                alias_symbol_func(symbol_map,
                                  constraint_data,
                                  alias_template % order_counter)
                if i == n_leqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                order_counter += 1

        # Create a dictionary of baron variable names to match to the
        # strings that constraint.to_string() prints. An important
        # note is that the variable strings are padded by spaces so
        # that whole variable names are recognized, and simple
        # variable names are not identified inside longer names.
        # Example: ' x[1] ' -> ' x3 '
        #FIXME: 7/18/14 CLH: This may cause mistakes if spaces in
        #                    variable names are allowed
        vstring_to_bar_dict = {}
        pstring_to_bar_dict = {}
        for block in all_blocks_list:

            for var_data in active_components_data_var[id(block)]:
                variable_stream = StringIO()
                var_data.to_string(ostream=variable_stream, verbose=False)
                variable_string = variable_stream.getvalue()

                variable_string = ' '+variable_string+' '
                vstring_to_bar_dict[variable_string] = \
                    ' '+object_symbol_dictionary[id(var_data)]+' '

            for param in block.component_objects(Param, active=True):
                if param._mutable and param.is_indexed():
                    param_data_iter = \
                        (param_data for index, param_data in iteritems(param))
                elif not param.is_indexed():
                    param_data_iter = iter([param])
                else:
                    param_data_iter = iter([])

                for param_data in param_data_iter:
                    param_stream = StringIO()
                    param.to_string(ostream=param_stream, verbose=False)
                    param_string = param_stream.getvalue()

                    param_string = ' '+param_string+' '
                    pstring_to_bar_dict[param_string] = ' '+str(param_data())+' '

        # Equation Definition
        string_template = '%'+self._precision_string
        output_file.write('c_e_FIX_ONE_VAR_CONST__:  ONE_VAR_CONST__  == 1;\n');
        for constraint_data in itertools.chain(eqns,
                                               r_o_eqns,
                                               c_eqns,
                                               l_eqns):

            #########################
            #CLH: The section below is kind of a hack-y way to use
            #     the expr.to_string function to print
            #     expressions. A stream is created, writen to, and
            #     then the string is recovered and stored in
            #     eqn_body. Then the variable names are converted
            #     to match the variable names that are used in the
            #     bar file.

            # Fill in the body of the equation
            body_string_buffer = StringIO()

            constraint_data.body.to_string(ostream=body_string_buffer,
                                           verbose=False)
            eqn_body = body_string_buffer.getvalue()

            # First, pad the equation so that if there is a
            # variable name at the start or end of the equation,
            # it can still be identified as padded with spaces.

            # Second, change pyomo's ** to baron's ^, also with
            # padding so that variable can always be found with
            # space around them

            # Third, add more padding around multiplication. Pyomo
            # already has spaces between variable on variable
            # multiplication, but not for constants on variables
            eqn_body = ' '+eqn_body+' '
            eqn_body = eqn_body.replace('**',' ^ ')
            eqn_body = eqn_body.replace('*', ' * ')


            #
            # FIXME: The following block of code is extremely inefficient.
            #        We are looping through every parameter and variable in
            #        the model each time we write a constraint expression.
            #
            ################################################
            vnames = [(variable_string, bar_string)
                      for variable_string, bar_string in iteritems(vstring_to_bar_dict)
                      if variable_string in eqn_body]
            for variable_string, bar_string in vnames:
                eqn_body = eqn_body.replace(variable_string, bar_string)
            for param_string, bar_string in iteritems(pstring_to_bar_dict):
                eqn_body = eqn_body.replace(param_string, bar_string)
            referenced_variable_ids.update(
                id(sm_bySymbol[bar_string.strip()]())
                for variable_string, bar_string in vnames)
            ################################################

            if len(vnames) == 0:
                assert not skip_trivial_constraints
                eqn_body += "+ 0 * ONE_VAR_CONST__ "

            # 7/29/14 CLH:
            #FIXME: Baron doesn't handle many of the
            #       intrinsic_functions available in pyomo. The
            #       error message given by baron is also very
            #       weak.  Either a function here to re-write
            #       unallowed expressions or a way to track solver
            #       capability by intrinsic_expression would be
            #       useful.
            ##########################

            con_symbol = object_symbol_dictionary[id(constraint_data)]
            output_file.write(str(con_symbol) + ': ')

            # Fill in the left and right hand side (constants) of
            #  the equations

            # Equality constraint
            if constraint_data.equality:
                eqn_lhs = ''
                eqn_rhs = ' == ' + \
                          str(string_template
                              % self._get_bound(constraint_data.upper))

            # Greater than constraint
            elif constraint_data.upper is None:
                eqn_rhs = ' >= ' + \
                          str(string_template
                              % self._get_bound(constraint_data.lower))
                eqn_lhs = ''

            # Less than constraint
            elif constraint_data.lower is None:
                eqn_rhs = ' <= ' + \
                          str(string_template
                              % self._get_bound(constraint_data.upper))
                eqn_lhs = ''

            # Double-sided constraint
            elif (constraint_data.upper is not None) and \
                 (constraint_data.lower is not None):
                eqn_lhs = str(string_template
                              % self._get_bound(constraint_data.lower)) + \
                          ' <= '
                eqn_rhs = ' <= ' + \
                          str(string_template
                              % self._get_bound(constraint_data.upper))

            eqn_string = eqn_lhs + eqn_body + eqn_rhs + ';\n'
            output_file.write(eqn_string)

        #
        # OBJECTIVE
        #

        output_file.write("\nOBJ: ")

        n_objs = 0
        for block in all_blocks_list:

            for objective_data in block.component_data_objects(Objective,
                                                               active=True,
                                                               sort=sorter,
                                                               descend_into=False):

                n_objs += 1
                if n_objs > 1:
                    raise ValueError("The BARON writer has detected multiple active "
                                     "objective functions on model %s, but "
                                     "currently only handles a single objective."
                                     % (model.name))

                # create symbol
                create_symbol_func(symbol_map, objective_data, labeler)
                alias_symbol_func(symbol_map, objective_data, "__default_objective__")

                if objective_data.is_minimizing():
                    output_file.write("minimize ")
                else:
                    output_file.write("maximize ")

                #FIXME 7/18/14 See above, constraint writing
                #              section. Will cause problems if there
                #              are spaces in variables
                # Similar to the constraints section above, the
                # objective is generated from the expr.to_string
                # function.
                obj_stream = StringIO()
                objective_data.expr.to_string(ostream=obj_stream, verbose=False)

                obj_string = ' '+obj_stream.getvalue()+' '
                obj_string = obj_string.replace('**',' ^ ')
                obj_string = obj_string.replace('*', ' * ')

                #
                # FIXME: The following block of code is extremely inefficient.
                #        We are looping through every parameter and variable in
                #        the model each time we write an expression.
                #
                ################################################
                vnames = [(variable_string, bar_string)
                          for variable_string, bar_string in iteritems(vstring_to_bar_dict)
                          if variable_string in obj_string]
                for variable_string, bar_string in vnames:
                    obj_string = obj_string.replace(variable_string, bar_string)
                for param_string, bar_string in iteritems(pstring_to_bar_dict):
                    obj_string = obj_string.replace(param_string, bar_string)
                referenced_variable_ids.update(
                    id(sm_bySymbol[bar_string.strip()]())
                    for variable_string, bar_string in vnames)
                ################################################

        output_file.write(obj_string+";\n\n")

        #
        # STARTING_POINT
        #
        output_file.write('STARTING_POINT{\nONE_VAR_CONST__: 1;\n')
        string_template = '%s: %'+self._precision_string+';\n'
        for block in all_blocks_list:
            for var_data in active_components_data_var[id(block)]:
                starting_point = var_data.value
                if starting_point is not None:
                    var_name = object_symbol_dictionary[id(var_data)]
                    output_file.write(string_template % (var_name, starting_point))

        output_file.write('}\n\n')

        output_file.close()

        # Clean up the symbol map to only contain variables referenced
        # in the active constraints
        vars_to_delete = symbol_map_variable_ids - referenced_variable_ids
        sm_byObject = symbol_map.byObject
        for varid in vars_to_delete:
            symbol = sm_byObject[varid]
            del sm_byObject[varid]
            del sm_bySymbol[symbol]

        del symbol_map_variable_ids
        del referenced_variable_ids

        return output_filename, symbol_map

