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

import pyomo.util.plugin
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter
from pyomo.core.base import SortComponents
from pyomo.core.base import SymbolMap, AlphaNumTextLabeler, NumericLabeler
from pyomo.core.base import BooleanSet, Constraint, IntegerSet
from pyomo.core.base.objective import Objective
from pyomo.core.base import Constraint, Var, Param
from pyomo.core.base.set_types import *
#CLH: EXPORT suffixes "constraint_types" and "branching_priorities"
#     pass their respective information to the .bar file
from pyomo.core.base.suffix import active_export_suffix_generator

from six import iteritems, StringIO
from six.moves import xrange

logger = logging.getLogger('pyomo.core')

# TODO: The to_string function is handy, but the fact that
#       it calls .cname(True) under the hood for all components
#       everywhere they are used will present ENORMOUS
#       overhead components that have a large index set.
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
#       were actually used. Tthis affects the .stale flag on
#       variables when loading results.

# TODO: Add support for output_fixed_variable_bounds

class ProblemWriter_bar(AbstractProblemWriter):

    pyomo.util.plugin.alias('baron_writer')
    pyomo.util.plugin.alias('bar')

    def __init__(self):

        AbstractProblemWriter.__init__(self, ProblemFormat.bar)

        #Coppied from cpxlp.py:
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

    def __call__(self, model, output_filename, solver_capability, io_options):

        # NOTE: io_options is a simple dictionary of keyword-value pairs
        #       specific to this writer.

        symbolic_solver_labels = io_options.pop("symbolic_solver_labels", False)
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

        #output_fixed_variable_bounds = io_options.pop("output_fixed_variable_bounds", False)

        if io_options:
            logger.warn(
                "ProblemWriter_baron_writer passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(io_options)))

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError("Baron problem writer: Using both the "
                             "'symbolic_solver_labels' and 'labeler' "
                             "I/O options is forbidden")

        if output_filename is None:
            output_filename = model.name + ".bar"

        output_file=open(output_filename, "a")

        #*******Begin section copied from BARON.py**********
        #print('*********************************************')
        #print('Executing bar writer in baron_writer.py')
        #print('*********************************************')

        if symbolic_solver_labels:
            labeler = AlphaNumTextLabeler()
        elif labeler is None:
            labeler = NumericLabeler('x')

        #
        # Populate the symbol_map
        #

        symbol_map = SymbolMap(model)

        #cache frequently called functions
        create_symbol_func = SymbolMap.createSymbol
        create_symbols_func = SymbolMap.createSymbols
        alias_symbol_func = SymbolMap.alias

        # Cache the list of model blocks so we don't have to call
        # model.block_data_objects() many many times, which is slow for
        # indexed blocks
        all_blocks_list = list(model.block_data_objects(active=True, sort=sorter))

        # Cache component iteration lists just in case sorting is involved
        active_components_data_var = {}
        active_components_data_con = {}
        active_components_data_obj = {}
        for block in all_blocks_list:

            active_components_data_obj[id(block)] = \
                list(block.component_data_objects(Objective, active=True, sort=sorter, descend_into=False))
            create_symbols_func(symbol_map,
                                active_components_data_obj[id(block)],
                                labeler)

            active_components_data_con[id(block)] = \
                list(block.component_data_objects(Constraint, active=True, sort=sorter, descend_into=False))
            create_symbols_func(symbol_map,
                                active_components_data_con[id(block)],
                                labeler)

            active_components_data_var[id(block)] = \
                list(block.component_data_objects(Var, active=True, sort=sorter, descend_into=False))
            create_symbols_func(symbol_map,
                                active_components_data_var[id(block)],
                                labeler)

            # GAH: Not sure this is necessary, and also it would break for
            #      non-mutable indexed params so I am commenting out for now.
            #for param_data in active_components_data(block, Param, sort=sorter):
                #instead of checking if param_data._mutable:
                #if not param_data.is_constant():
                #    create_symbol_func(symbol_map, param_data, labeler)

        object_symbol_dictionary = symbol_map.getByObjectDictionary()

        # GAH 1/5/15: Substituting all non-alphanumeric characters for underscore
        #             in labeler so this manual update should no longer be needed
        #
        # If the text labeler is used, correct the labels to be baron-allowed variable names
        # Change '(' and ')' to '__'
        # This way, for simple variable names like 'x(1_2)' --> 'x__1_2__'
        # FIXME: 7/21/14  This may break if users give variable names with two or more underscores together
        #if symbolic_solver_labels:
        #    for key,label in iteritems(object_symbol_dictionary):
        #        label = label.replace('(','___')
        #        object_symbol_dictionary[key] = label.replace(')','__')

        #
        # BINARY_VARIABLES, INTEGER_VARIABLES, POSITIVE_VARIABLES, VARIABLES
        #

        nbv = 0
        niv = 0
        npv = 0
        nv = 0

        BinVars = []
        IntVars = []
        PosVars = []
        Vars = []

        for block in all_blocks_list:
            for var_data in active_components_data_var[id(block)]:

                if isinstance(var_data.domain, BooleanSet):
                    nbv += 1
                    TypeList = BinVars
                elif isinstance(var_data.domain, IntegerSet):
                    niv += 1
                    TypeList = IntVars
                elif isinstance(var_data.domain, RealSet) and \
                     (var_data.lb is not None) and \
                     (var_data.lb >= 0):
                    npv += 1
                    TypeList = PosVars
                else:
                    nv += 1
                    TypeList = Vars

                var_name = object_symbol_dictionary[id(var_data)]
                TypeList.append(var_name)

        if nbv != 0:
            output_file.write('BINARY_VARIABLES ')
            for var_name in BinVars[:-1]:
                output_file.write(str(var_name)+', ')
            var_name = BinVars[nbv-1]
            output_file.write(str(var_name)+';\n\n')
        if niv != 0:
            output_file.write('INTEGER_VARIABLES ')
            for var_name in IntVars[:-1]:
                output_file.write(str(var_name)+', ')
            var_name = IntVars[niv-1]
            output_file.write(str(var_name)+';\n\n')
        if npv != 0:
            output_file.write('POSITIVE_VARIABLES ')
            for var_name in PosVars[:-1]:
                output_file.write(str(var_name)+', ')
            var_name = PosVars[npv-1]
            output_file.write(str(var_name)+';\n\n')
        if nv != 0:
            output_file.write('VARIABLES ')
            for var_name in Vars[:-1]:
                output_file.write(str(var_name)+', ')
            var_name = Vars[nv-1]
            output_file.write(str(var_name)+';\n\n')

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

        for block in all_blocks_list:
            for name,suffix in active_export_suffix_generator(block):
                if name == 'branching_priorities':
                    for var_data in active_components_data_var[id(block)]:
                        priority = suffix.get(var_data)
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

        # Equation Counting
        eqns = []
        r_o_eqns = []
        c_eqns = []
        l_eqns = []
        EquationHeader = False
        for block in all_blocks_list:
            for constraint_data in active_components_data_con[id(block)]:

                #FIXME: CLH, 7/18/14: Not sure if the code for .trivial
                #                     is up-to-date and needs to here.
                #GAH 1/5/15: The .active flag is checked by the active_components_data
                #            generator for this loop. The .trivial flag is set after calling
                #            model.preprocess() (which the baron writer does not require).
                #            Pyomo writer/solver plugins are inconsistent as to which writers required
                #            preprocessing (cpxlp, CPLEXDirect, gurobi_direct), which writers
                #            automatically preprocess (ampl), and which writers ignore
                #            preprocessing (baron).
                #
                #            It's a subtle issue, but when scripts involve combinations of operations that
                #            add/fix/free variables, add/remove/activate/deactivate constraints,
                #            cache/restore solutions, and make use of multiple solver interfaces,
                #            these differences in the solver plugins are easily overlooked and garbage
                #            results ensue without warning.
                #
                #            This is not a bug, but should be part of a future design discussion

                #if constraint_data.parent_component().trivial:
                #    continue
                #if not constraint_data.active:
                #    continue

                flag = False
                for name,suffix in active_export_suffix_generator(block):
                    if name == 'constraint_types':
                        constraint_type = suffix.get(constraint_data)
                        if constraint_type is None:
                            flag = True
                            eqns.append(constraint_data)
                        elif constraint_type.lower() == 'relaxationonly':
                            flag = True
                            r_o_eqns.append(constraint_data)
                        elif constraint_type.lower() == 'convex':
                            flag = True
                            c_eqns.append(constraint_data)
                        elif constraint_type.lower() == 'local':
                            flag = True
                            l_eqns.append(constraint_data)
                        else:
                            logger.warn('A constraint_types suffix was not '
                                        'recognized: %s' % constraint_type)
                if not flag:
                    eqns.append(constraint_data)

        #Equation Declaration
        n_eqns = len(eqns)
        n_roeqns = len(r_o_eqns)
        n_ceqns = len(c_eqns)
        n_leqns = len(l_eqns)

        # Alias the constraints by declaration order since Baron does not
        # include the constraint names in the solution file. It is important
        # that this alias not clash with any real constraint labels, hence
        # the use of the ".c<integer>" template. It is not possible to declare
        # a component having this type of name when using standard syntax.
        # There are ways to do it, but it is unlikely someone will.
        order_counter = 0
        alias_template = ".c%d"
        if n_eqns > 0:
            output_file.write('EQUATIONS ')
            for i, constraint_data in enumerate(eqns):
                con_symbol = object_symbol_dictionary[id(constraint_data)]
                if i == n_eqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                assert not con_symbol.startswith('.')
                alias_symbol_func(symbol_map,
                                  constraint_data,
                                  alias_template % order_counter)
                order_counter += 1

        if n_roeqns > 0:
            output_file.write('RELAXATION_ONLY_EQUATIONS ')
            for i, constraint_data in enumerate(r_o_eqns):
                con_symbol = object_symbol_dictionary[id(constraint_data)]
                if i == n_roeqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                assert not con_symbol.startswith('.')
                alias_symbol_func(symbol_map,
                                  constraint_data,
                                  alias_template % order_counter)
                order_counter += 1

        if n_ceqns > 0:
            output_file.write('CONVEX_EQUATIONS ')
            for i, constraint_data in enumerate(c_eqns):
                con_symbol = object_symbol_dictionary[id(constraint_data)]
                if i == n_ceqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                assert not con_symbol.startswith('.')
                alias_symbol_func(symbol_map,
                                  constraint_data,
                                  alias_template % order_counter)
                order_counter += 1

        if n_leqns > 0:
            output_file.write('LOCAL_EQUATIONS ')
            for i, constraint_data in enumerate(l_eqns):
                con_symbol = object_symbol_dictionary[id(constraint_data)]
                if i == n_leqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                assert not con_symbol.startswith('.')
                alias_symbol_func(symbol_map,
                                  constraint_data,
                                  alias_template % order_counter)
                order_counter += 1

        # Create a dictionary of baron variable names to match to the
        # strings that constraint.to_string() prints. An important
        # note is that the variable strings are padded by spaces so
        # that whole variable names are recognized, and simple
        # variable names are not identified inside longer names.
        # Example: ' x[1] ' -> ' x3 '
        #FIXME: 7/18/14 CLH: This may cause mistakes if spaces in
        #                    variable names are allowed
        string_to_bar_dict = {}
        for block in all_blocks_list:

            for var_data in active_components_data_var[id(block)]:
                variable_stream = StringIO()
                var_data.to_string(ostream=variable_stream, verbose=False)
                variable_string = variable_stream.getvalue()

                variable_string = ' '+variable_string+' '
                string_to_bar_dict[variable_string] = \
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
                    string_to_bar_dict[param_string] = ' '+str(param_data())+' '

        # Equation Definition
        for block in all_blocks_list:
            for constraint_data in active_components_data_con[id(block)]:

                #FIXME: 7/18/14 CLH: same as above, not sure if
                #                    .trivial is necessary anymore
                #GAH 1/5/15: see comment above

                #if constraint_data.parent_component().trivial:
                #    continue
                #if not constraint_data.active:
                #    continue

                con_symbol = object_symbol_dictionary[id(constraint_data)]
                label = str(con_symbol) + ': '
                output_file.write(label)

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

                for variable_string in string_to_bar_dict.iterkeys():
                    eqn_body = eqn_body.replace(variable_string,
                                                string_to_bar_dict[variable_string])

                #FIXME: 7/29/14 CLH: Baron doesn't handle many of the intrinsic_functions available
                #                    in pyomo. The error message given by baron is also very weak.
                #                    Either a function here to re-write unallowed expressions or
                #                    a way to track solver capability by intrinsic_expression would
                #                    be useful.
                #
                ##########################

                string_template = '%'+self._precision_string

                # Fill in the left and right hand side (constants) of
                #  the equations

                # Equality constraint
                if constraint_data.equality:
                    eqn_lhs = ''
                    eqn_rhs = ' == ' + \
                              str(string_template % constraint_data.upper())

                # Greater than constraint
                elif constraint_data.upper is None:
                    eqn_rhs = ' >= ' + \
                              str(string_template % constraint_data.lower())
                    eqn_lhs = ''

                # Less than constraint
                elif constraint_data.lower is None:
                    eqn_rhs = ' <= ' + \
                              str(string_template % constraint_data.upper())
                    eqn_lhs = ''

                # Double-sided constraint
                elif (constraint_data.upper is not None) and \
                     (constraint_data.lower is not None):
                    eqn_lhs = str(string_template % constraint_data.lower()) + \
                              ' <= '
                    eqn_rhs = ' <= ' + \
                              str(string_template % constraint_data.upper())

                eqn_string = eqn_lhs + eqn_body + eqn_rhs + ';\n'
                output_file.write(eqn_string)

        #
        # OBJECTIVE
        #

        output_file.write("\nOBJ: ")

        n_objs = 0
        for block in all_blocks_list:
            for objective_data in active_components_data_obj[id(block)]:

                n_objs += 1
                if n_objs > 1:
                    raise ValueError("The BARON writer has detected multiple active "
                                     "objective functions on model %s, but "
                                     "currently only handles a single objective."
                                     % (model.cname(True)))

                alias_symbol_func(symbol_map,
                                  objective_data,
                                  "__default_objective__")

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

                for variable_string in string_to_bar_dict.iterkeys():
                    obj_string = obj_string.replace(
                        variable_string,
                        string_to_bar_dict[variable_string])

        output_file.write(obj_string+";\n\n")

        #
        # STARTING_POINT
        #
        starting_point_list = []
        for block in all_blocks_list:
            for var_data in active_components_data_var[id(block)]:
                starting_point = var_data.value
                if starting_point is not None:
                    starting_point_list.append((var_data,starting_point))

        if len(starting_point_list) > 0:
            output_file.write('STARTING_POINT{\n')
            for variable,starting_value in starting_point_list:
                var_name = object_symbol_dictionary[id(variable)]
                string_template = '%s: %'+self._precision_string+';\n'
                output_file.write(string_template % (var_name, starting_value))
            output_file.write('}\n\n')


        #*******End section copied from BARON.py************

        output_file.close()

        return output_filename, symbol_map

