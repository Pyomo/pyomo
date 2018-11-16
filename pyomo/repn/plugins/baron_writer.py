#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Problem Writer for BARON .bar Format Files
#

import logging
import itertools
from six import iteritems, StringIO, iterkeys
from six.moves import xrange
from pyutilib.math import isclose

from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.expr.numvalue import is_fixed, value, native_numeric_types, native_types
from pyomo.core.expr import current as EXPR
from pyomo.core.base import (SortComponents,
                             SymbolMap,
                             AlphaNumericTextLabeler,
                             NumericLabeler,
                             BooleanSet, Constraint,
                             IntegerSet, Objective,
                             Var, Param)
from pyomo.core.base.set_types import *
#CLH: EXPORT suffixes "constraint_types" and "branching_priorities"
#     pass their respective information to the .bar file
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock

logger = logging.getLogger('pyomo.core')


#
# A visitor pattern that creates a string for an expression
# that is compatible with the BARON syntax.
#
class ToBaronVisitor(EXPR.ExpressionValueVisitor):

    def __init__(self, variables, smap):
        super(ToBaronVisitor, self).__init__()
        self.variables = variables
        self.smap = smap

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        tmp = []
        for i,val in enumerate(values):
            arg = node._args_[i]

            if arg is None:
                tmp.append('Undefined')
            elif arg.__class__ in native_numeric_types:
                tmp.append(val)
            elif arg.__class__ in native_types:
                tmp.append("'{0}'".format(val))
            elif arg.is_variable_type():
                tmp.append(val)
            elif arg.is_expression_type() and node._precedence() < arg._precedence():
                tmp.append("({0})".format(val))
            else:
                tmp.append(val)

        if node.__class__ is EXPR.LinearExpression:
            for v in node.linear_vars:
                self.variables.add(id(v))

        if node.__class__ is EXPR.ProductExpression:
            return "{0} * {1}".format(tmp[0], tmp[1])
        elif node.__class__ is EXPR.MonomialTermExpression:
            if tmp[0] == '-1':
                # It seems dumb to construct a temporary NegationExpression object
                # Should we copy the logic from that function here?
                return EXPR.NegationExpression._to_string(EXPR.NegationExpression((None,)), [tmp[1]], None, self.smap, True)
            else:
                return "{0} * {1}".format(tmp[0], tmp[1])
        elif node.__class__ is EXPR.PowExpression:
            return "{0} ^ {1}".format(tmp[0], tmp[1])
        elif node.__class__ is EXPR.UnaryFunctionExpression and node.name == "sqrt":
            return "{0} ^ 0.5".format(tmp[0])
        else:
            return node._to_string(tmp, None, self.smap, True)

    def visiting_potential_leaf(self, node):
        """ 
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        #print("ISLEAF")
        #print(node.__class__)
        if node is None:
            return True, None

        if node.__class__ in native_types:
            return True, str(node)

        if node.is_variable_type():
            if node.fixed:
                return True, str(value(node))
            self.variables.add(id(node))
            label = self.smap.getSymbol(node)
            return True, label

        if not node.is_expression_type():
            return True, str(value(node))

        return False, None


def expression_to_string(expr, variables, labeler=None, smap=None):
    if labeler is not None:
        if smap is None:
            smap = SymbolMap()
        smap.default_labeler = labeler
    visitor = ToBaronVisitor(variables, smap)
    return visitor.dfs_postorder_stack(expr)



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

@WriterFactory.register('bar', 'Generate the corresponding BARON BAR file.')
class ProblemWriter_bar(AbstractProblemWriter):

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

    def _write_equations_section(self,
                                 model,
                                 output_file,
                                 all_blocks_list,
                                 active_components_data_var,
                                 symbol_map,
                                 c_labeler,
                                 output_fixed_variable_bounds,
                                 skip_trivial_constraints,
                                 sorter):

        referenced_variable_ids = set()

        def _skip_trivial(constraint_data):
            if skip_trivial_constraints:
                if constraint_data._linear_canonical_form:
                    repn = constraint_data.canonical_form()
                    if (repn.variables is None) or \
                       (len(repn.variables) == 0):
                        return True
                elif constraint_data.body.polynomial_degree() == 0:
                    return True
            return False

        #
        # Check for active suffixes to export
        #
        if isinstance(model, IBlock):
            suffix_gen = lambda b: ((suf.storage_key, suf) \
                                    for suf in pyomo.core.kernel.suffix.\
                                    export_suffix_generator(b,
                                                            active=True,
                                                            descend_into=False))
        else:
            suffix_gen = lambda b: pyomo.core.base.suffix.\
                         active_export_suffix_generator(b)
        r_o_eqns = []
        c_eqns = []
        l_eqns = []
        branching_priorities_suffixes = []
        for block in all_blocks_list:
            for name, suffix in suffix_gen(block):
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

                if (not constraint_data.has_lb()) and \
                   (not constraint_data.has_ub()):
                    assert not constraint_data.equality
                    continue # non-binding, so skip

                if (not _skip_trivial(constraint_data)) and \
                   (constraint_data not in non_standard_eqns):

                    eqns.append(constraint_data)

                    con_symbol = symbol_map.createSymbol(constraint_data, c_labeler)
                    assert not con_symbol.startswith('.')
                    assert con_symbol != "c_e_FIX_ONE_VAR_CONST__"

                    symbol_map.alias(constraint_data,
                                      alias_template % order_counter)
                    output_file.write(", "+str(con_symbol))
                    order_counter += 1

        output_file.write(";\n\n")

        if n_roeqns > 0:
            output_file.write('RELAXATION_ONLY_EQUATIONS ')
            for i, constraint_data in enumerate(r_o_eqns):
                con_symbol = symbol_map.createSymbol(constraint_data, c_labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != "c_e_FIX_ONE_VAR_CONST__"
                symbol_map.alias(constraint_data,
                                  alias_template % order_counter)
                if i == n_roeqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                order_counter += 1

        if n_ceqns > 0:
            output_file.write('CONVEX_EQUATIONS ')
            for i, constraint_data in enumerate(c_eqns):
                con_symbol = symbol_map.createSymbol(constraint_data, c_labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != "c_e_FIX_ONE_VAR_CONST__"
                symbol_map.alias(constraint_data,
                                  alias_template % order_counter)
                if i == n_ceqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')
                order_counter += 1

        if n_leqns > 0:
            output_file.write('LOCAL_EQUATIONS ')
            for i, constraint_data in enumerate(l_eqns):
                con_symbol = symbol_map.createSymbol(constraint_data, c_labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != "c_e_FIX_ONE_VAR_CONST__"
                symbol_map.alias(constraint_data,
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
        if isinstance(model, IBlock):
            mutable_param_gen = lambda b: \
                                b.components(ctype=Param,
                                             descend_into=False)
        else:
            def mutable_param_gen(b):
                for param in block.component_objects(Param):
                    if param._mutable and param.is_indexed():
                        param_data_iter = \
                            (param_data for index, param_data
                             in iteritems(param))
                    elif not param.is_indexed():
                        param_data_iter = iter([param])
                    else:
                        param_data_iter = iter([])

                    for param_data in param_data_iter:
                        yield param_data

        if False:
            #
            # This was part of a merge from master that caused
            # test failures.  But commenting this out didn't cause additional failures!?!
            #
            vstring_to_var_dict = {}
            vstring_to_bar_dict = {}
            pstring_to_bar_dict = {}
            _val_template = ' %'+self._precision_string+' '
            for block in all_blocks_list:
                for var_data in active_components_data_var[id(block)]:
                    variable_stream = StringIO()
                    var_data.to_string(ostream=variable_stream, verbose=False)
                    variable_string = variable_stream.getvalue()
                    variable_string = ' '+variable_string+' '
                    vstring_to_var_dict[variable_string] = var_data
                    if output_fixed_variable_bounds or (not var_data.fixed):
                        vstring_to_bar_dict[variable_string] = \
                            ' '+object_symbol_dictionary[id(var_data)]+' '
                    else:
                        assert var_data.value is not None
                        vstring_to_bar_dict[variable_string] = \
                            (_val_template % (var_data.value,))

                for param_data in mutable_param_gen(block):
                    param_stream = StringIO()
                    param_data.to_string(ostream=param_stream, verbose=False)
                    param_string = param_stream.getvalue()

                    param_string = ' '+param_string+' '
                    pstring_to_bar_dict[param_string] = \
                        (_val_template % (param_data(),))

        # Equation Definition
        string_template = '%'+self._precision_string
        output_file.write('c_e_FIX_ONE_VAR_CONST__:  ONE_VAR_CONST__  == 1;\n');
        for constraint_data in itertools.chain(eqns,
                                               r_o_eqns,
                                               c_eqns,
                                               l_eqns):

            variables = set()
            #print(symbol_map.byObject.keys())
            eqn_body = expression_to_string(constraint_data.body, variables, smap=symbol_map)
            #print(symbol_map.byObject.keys())
            referenced_variable_ids.update(variables)

            if len(variables) == 0:
                assert not skip_trivial_constraints
                eqn_body += " + 0 * ONE_VAR_CONST__ "

            # 7/29/14 CLH:
            #FIXME: Baron doesn't handle many of the
            #       intrinsic_functions available in pyomo. The
            #       error message given by baron is also very
            #       weak.  Either a function here to re-write
            #       unallowed expressions or a way to track solver
            #       capability by intrinsic_expression would be
            #       useful.
            ##########################

            con_symbol = symbol_map.byObject[id(constraint_data)]
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
            elif not constraint_data.has_ub():
                eqn_rhs = ' >= ' + \
                          str(string_template
                              % self._get_bound(constraint_data.lower))
                eqn_lhs = ''

            # Less than constraint
            elif not constraint_data.has_lb():
                eqn_rhs = ' <= ' + \
                          str(string_template
                              % self._get_bound(constraint_data.upper))
                eqn_lhs = ''

            # Double-sided constraint
            elif constraint_data.has_lb() and \
                 constraint_data.has_ub():
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
                symbol_map.createSymbol(objective_data, c_labeler)
                symbol_map.alias(objective_data, "__default_objective__")

                if objective_data.is_minimizing():
                    output_file.write("minimize ")
                else:
                    output_file.write("maximize ")

                variables = set()
                #print(symbol_map.byObject.keys())
                obj_string = expression_to_string(objective_data.expr, variables, smap=symbol_map)
                #print(symbol_map.byObject.keys())
                referenced_variable_ids.update(variables)


        output_file.write(obj_string+";\n\n")
        #referenced_variable_ids.update(symbol_map.byObject.keys())

        return referenced_variable_ids, branching_priorities_suffixes

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

        output_fixed_variable_bounds = \
            io_options.pop("output_fixed_variable_bounds", False)

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
            v_labeler = AlphaNumericTextLabeler()
            c_labeler = AlphaNumericTextLabeler()
        elif labeler is None:
            v_labeler = NumericLabeler('x')
            c_labeler = NumericLabeler('c')

        symbol_map = SymbolMap()
        symbol_map.default_labeler = v_labeler
        #sm_bySymbol = symbol_map.bySymbol

        # Cache the list of model blocks so we don't have to call
        # model.block_data_objects() many many times, which is slow
        # for indexed blocks
        all_blocks_list = list(model.block_data_objects(active=True,
                                                        sort=sorter,
                                                        descend_into=True))
        active_components_data_var = {}
        #for block in all_blocks_list:
        #    tmp = active_components_data_var[id(block)] = \
        #          list(obj for obj in block.component_data_objects(Var,
        #                                                           sort=sorter,
        #                                                           descend_into=False))
        #    create_symbols_func(symbol_map, tmp, labeler)

            # GAH: Not sure this is necessary, and also it would break for
            #      non-mutable indexed params so I am commenting out for now.
            #for param_data in active_components_data(block, Param, sort=sorter):
                #instead of checking if param_data._mutable:
                #if not param_data.is_constant():
                #    create_symbol_func(symbol_map, param_data, labeler)

        #symbol_map_variable_ids = set(symbol_map.byObject.keys())
        #object_symbol_dictionary = symbol_map.byObject

        #
        # Go through the objectives and constraints and generate
        # the output so that we can obtain the set of referenced
        # variables.
        #
        equation_section_stream = StringIO()
        referenced_variable_ids, branching_priorities_suffixes = \
            self._write_equations_section(
                model,
                equation_section_stream,
                all_blocks_list,
                active_components_data_var,
                symbol_map,
                c_labeler,
                output_fixed_variable_bounds,
                skip_trivial_constraints,
                sorter)

        #
        # BINARY_VARIABLES, INTEGER_VARIABLES, POSITIVE_VARIABLES, VARIABLES
        #

        BinVars = []
        IntVars = []
        PosVars = []
        Vars = []
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]()

            if var_data.is_continuous():
                if var_data.has_lb() and \
                   (self._get_bound(var_data.lb) >= 0):
                    TypeList = PosVars
                else:
                    TypeList = Vars
            elif var_data.is_binary():
                TypeList = BinVars
            elif var_data.is_integer():
                TypeList = IntVars
            else:
                assert False
            TypeList.append(name)

        if len(BinVars) > 0:
            BinVars.sort()
            output_file.write('BINARY_VARIABLES ')
            output_file.write(", ".join(BinVars))
            output_file.write(';\n\n')

        if len(IntVars) > 0:
            IntVars.sort()
            output_file.write('INTEGER_VARIABLES ')
            output_file.write(", ".join(IntVars))
            output_file.write(';\n\n')

        PosVars.append('ONE_VAR_CONST__')
        PosVars.sort()
        output_file.write('POSITIVE_VARIABLES ')
        output_file.write(", ".join(PosVars))
        output_file.write(';\n\n')

        if len(Vars) > 0:
            Vars.sort()
            output_file.write('VARIABLES ')
            output_file.write(", ".join(Vars))
            output_file.write(';\n\n')

        #
        # LOWER_BOUNDS
        #

        lbounds = {}
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]()

            if var_data.fixed:
                if output_fixed_variable_bounds:
                    var_data_lb = var_data.value
                else:
                    var_data_lb = None
            else:
                var_data_lb = None
                if var_data.has_lb():
                    var_data_lb = self._get_bound(var_data.lb)

            if var_data_lb is not None:
                name_to_output = symbol_map.getSymbol(var_data)
                lb_string_template = '%s: %'+self._precision_string+';\n'
                lbounds[name_to_output] = lb_string_template % (name_to_output, var_data_lb)

        if len(lbounds) > 0:
            output_file.write("LOWER_BOUNDS{\n")
            output_file.write("".join( lbounds[key] for key in sorted(lbounds.keys()) ) )
            output_file.write("}\n\n")
        lbounds = None

        #
        # UPPER_BOUNDS
        #

        ubounds = {}
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]()

            if var_data.fixed:
                if output_fixed_variable_bounds:
                    var_data_ub = var_data.value
                else:
                    var_data_ub = None
            else:
                var_data_ub = None
                if var_data.has_ub():
                    var_data_ub = self._get_bound(var_data.ub)

            if var_data_ub is not None:
                name_to_output = symbol_map.getSymbol(var_data)
                ub_string_template = '%s: %'+self._precision_string+';\n'
                ubounds[name_to_output] = ub_string_template % (name_to_output, var_data_ub)

        if len(ubounds) > 0:
            output_file.write("UPPER_BOUNDS{\n")
            output_file.write("".join( ubounds[key] for key in sorted(ubounds.keys()) ) )
            output_file.write("}\n\n")
        ubounds = None

        #
        # BRANCHING_PRIORITIES
        #

        # Specifying priorities requires that the pyomo model has established an
        # EXTERNAL, float suffix called 'branching_priorities' on the model
        # object, indexed by the relevant variable
        BranchingPriorityHeader = False
        for suffix in branching_priorities_suffixes:
            for var_data, priority in iteritems(suffix):
                if id(var_data) not in referenced_variable_ids:
                    continue
                if priority is not None:
                    if not BranchingPriorityHeader:
                        output_file.write('BRANCHING_PRIORITIES{\n')
                        BranchingPriorityHeader = True
                    name_to_output = symbol_map.getSymbol(var_data)
                    output_file.write(name_to_output+': '+str(priority)+';\n')

        if BranchingPriorityHeader:
            output_file.write("}\n\n")

        #
        # Now write the objective and equations section
        #
        output_file.write(equation_section_stream.getvalue())

        #
        # STARTING_POINT
        #
        output_file.write('STARTING_POINT{\nONE_VAR_CONST__: 1;\n')
        tmp = {}
        string_template = '%s: %'+self._precision_string+';\n'
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]()

            starting_point = var_data.value
            if starting_point is not None:
                var_name = symbol_map.getSymbol(var_data)
                tmp[var_name] = string_template % (var_name, starting_point)

        output_file.write("".join( tmp[key] for key in sorted(tmp.keys()) ))
        output_file.write('}\n\n')

        output_file.close()

        return output_filename, symbol_map

