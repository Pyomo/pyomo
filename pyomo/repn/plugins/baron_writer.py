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

import itertools
import logging
import math
from six import iteritems, StringIO

from pyomo.common.collections import OrderedSet
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.expr.numvalue import (
    value, native_numeric_types, native_types, nonpyomo_leaf_types,
)
from pyomo.core.expr import current as EXPR
from pyomo.core.base import (SortComponents,
                             SymbolMap,
                             ShortNameLabeler,
                             NumericLabeler,
                             Constraint,
                             Objective,
                             Param)
from pyomo.core.base.component import ActiveComponent
#CLH: EXPORT suffixes "constraint_types" and "branching_priorities"
#     pass their respective information to the .bar file
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.repn.util import valid_expr_ctypes_minlp, \
    valid_active_ctypes_minlp, ftoa

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
                tmp.append('Undefined')                 # TODO: coverage
            else:
                parens = False
                if val and val[0] in '-+':
                    parens = True
                elif arg.__class__ in native_numeric_types:
                    pass
                elif arg.__class__ in nonpyomo_leaf_types:
                    val = "'{0}'".format(val)
                elif arg.is_expression_type():
                    if node._precedence() < arg._precedence():
                        parens = True
                    elif node._precedence() == arg._precedence():
                        if i == 0:
                            parens = node._associativity() != 1
                        elif i == len(node._args_)-1:
                            parens = node._associativity() != -1
                        else:
                            parens = True
                if parens:
                    tmp.append("({0})".format(val))
                else:
                    tmp.append(val)

        if node.__class__ in EXPR.NPV_expression_types:
            return ftoa(value(node))

        if node.__class__ is EXPR.LinearExpression:
            for v in node.linear_vars:
                self.variables.add(id(v))

        if node.__class__ in {
                EXPR.ProductExpression, EXPR.MonomialTermExpression}:
            if tmp[0] in node._to_string.minus_one:
                return "- {0}".format(tmp[1])
            if tmp[0] in node._to_string.one:
                return tmp[1]
            return "{0} * {1}".format(tmp[0], tmp[1])
        elif node.__class__ is EXPR.PowExpression:
            x,y = node.args
            if type(x) not in native_types and not x.is_fixed() and \
               type(y) not in native_types and not y.is_fixed():
                # Per the BARON manual, x ^ y is allowed as long as x
                # and y are not both variables
                return "exp(({1}) * log({0}))".format(tmp[0], tmp[1])
            else:
                return "{0} ^ {1}".format(tmp[0], tmp[1])
        elif node.__class__ is EXPR.UnaryFunctionExpression:
            if node.name == "sqrt":
                return "{0} ^ 0.5".format(tmp[0])
            elif node.name == 'log10':
                return "{0} * log({1})".format(math.log10(math.e), tmp[0])
            elif node.name in {'exp','log'}:
                return node._to_string(tmp, None, self.smap, True)
            else:
                raise RuntimeError(
                    'The BARON .BAR format does not support the unary '
                    'function "%s".' % (node.name,))
        elif node.__class__ is EXPR.AbsExpression:
            return "({0} ^ 2) ^ 0.5".format(tmp[0])
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
            return True, ftoa(node)

        if node.is_expression_type():
            # we will descend into this, so type checking will happen later
            return False, None

        if node.is_component_type():
            _ctype = node.ctype
            if _ctype not in valid_expr_ctypes_minlp:
                # Make sure all components in active constraints
                # are basic ctypes we know how to deal with.
                raise RuntimeError(
                    "Unallowable component '%s' of type %s found in an active "
                    "constraint or objective.\nThe GAMS writer cannot export "
                    "expressions with this component type."
                    % (node.name, _ctype.__name__))

        if node.is_variable_type():
            if node.fixed:
                return True, ftoa(value(node))
            else:
                self.variables.add(id(node))
                label = self.smap.getSymbol(node)
                return True, label

        return True, ftoa(value(node))


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

@WriterFactory.register('bar', 'Generate the corresponding BARON BAR file.')
class ProblemWriter_bar(AbstractProblemWriter):

    def __init__(self):

        AbstractProblemWriter.__init__(self, ProblemFormat.bar)

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

        referenced_variable_ids = OrderedSet()

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
                    if param.mutable and param.is_indexed():
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
                            ftoa(var_data.value)

                for param_data in mutable_param_gen(block):
                    param_stream = StringIO()
                    param_data.to_string(ostream=param_stream, verbose=False)
                    param_string = param_stream.getvalue()

                    param_string = ' '+param_string+' '
                    pstring_to_bar_dict[param_string] = ftoa(param_data())

        # Equation Definition
        output_file.write('c_e_FIX_ONE_VAR_CONST__:  ONE_VAR_CONST__  == 1;\n');
        for constraint_data in itertools.chain(eqns,
                                               r_o_eqns,
                                               c_eqns,
                                               l_eqns):

            variables = OrderedSet()
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
                eqn_rhs = ' == ' + ftoa(constraint_data.upper)

            # Greater than constraint
            elif not constraint_data.has_ub():
                eqn_rhs = ' >= ' + ftoa(constraint_data.lower)
                eqn_lhs = ''

            # Less than constraint
            elif not constraint_data.has_lb():
                eqn_rhs = ' <= ' + ftoa(constraint_data.upper)
                eqn_lhs = ''

            # Double-sided constraint
            elif constraint_data.has_lb() and \
                 constraint_data.has_ub():
                eqn_lhs = ftoa(constraint_data.lower) + \
                          ' <= '
                eqn_rhs = ' <= ' + ftoa(constraint_data.upper)

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

                variables = OrderedSet()
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

        # Make sure there are no strange ActiveComponents. The expression
        # walker will handle strange things in constraints later.
        model_ctypes = model.collect_ctypes(active=True)
        invalids = set()
        for t in (model_ctypes - valid_active_ctypes_minlp):
            if issubclass(t, ActiveComponent):
                invalids.add(t)
        if len(invalids):
            invalids = [t.__name__ for t in invalids]
            raise RuntimeError(
                "Unallowable active component(s) %s.\nThe BARON writer cannot "
                "export models with this component type." %
                ", ".join(invalids))

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
            # Note that the Var and Constraint labelers must use the
            # same labeler, so that we can correctly detect name
            # collisions (which can arise when we truncate the labels to
            # the max allowable length.  BARON requires all identifiers
            # to start with a letter.  We will (randomly) choose "s_"
            # (for 'shortened')
            v_labeler = c_labeler = ShortNameLabeler(
                15, prefix='s_', suffix='_', caseInsensitive=True,
                legalRegex='^[a-zA-Z]')
        elif labeler is None:
            v_labeler = NumericLabeler('x')
            c_labeler = NumericLabeler('c')
        else:
            v_labeler = c_labeler = labeler

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
                #instead of checking if param_data.mutable:
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
                if var_data.has_lb() and (value(var_data.lb) >= 0):
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
                    var_data_lb = ftoa(var_data.value)
                else:
                    var_data_lb = None
            else:
                var_data_lb = None
                if var_data.has_lb():
                    var_data_lb = ftoa(var_data.lb)

            if var_data_lb is not None:
                name_to_output = symbol_map.getSymbol(var_data)
                lbounds[name_to_output] = '%s: %s;\n' % (
                    name_to_output, var_data_lb)

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
                    var_data_ub = ftoa(var_data.value)
                else:
                    var_data_ub = None
            else:
                var_data_ub = None
                if var_data.has_ub():
                    var_data_ub = ftoa(var_data.ub)

            if var_data_ub is not None:
                name_to_output = symbol_map.getSymbol(var_data)
                ubounds[name_to_output] = '%s: %s;\n' % (
                    name_to_output, var_data_ub)

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
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]()

            starting_point = var_data.value
            if starting_point is not None:
                var_name = symbol_map.getSymbol(var_data)
                tmp[var_name] = "%s: %s;\n" % (
                    var_name, ftoa(starting_point))

        output_file.write("".join( tmp[key] for key in sorted(tmp.keys()) ))
        output_file.write('}\n\n')

        output_file.close()

        return output_filename, symbol_map

