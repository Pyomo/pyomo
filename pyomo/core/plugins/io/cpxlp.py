#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________

#
# Problem Writer for CPLEX LP Format Files
#

import logging
import math
import weakref

from six import iterkeys, itervalues, iteritems, advance_iterator, StringIO
from six.moves import xrange, zip

from coopr.opt import ProblemFormat
from coopr.opt.base import AbstractProblemWriter
from coopr.pyomo.base import SymbolMap, BasicSymbolMap, TextLabeler, NumericLabeler
from coopr.pyomo.base import BooleanSet, Constraint, ConstraintList, expr, IntegerSet, Component
from coopr.pyomo.base import active_components, active_components_data
from coopr.pyomo.base import Var, value, label_from_name, NumericConstant, ComponentMap
from coopr.pyomo.base.sos import SOSConstraint
from coopr.pyomo.base.objective import Objective, minimize, maximize
from coopr.pyomo.expr import canonical_degree, LinearCanonicalRepn

from coopr.core.plugin import alias
from pyutilib.misc import tostr, PauseGC

logger = logging.getLogger('coopr.pyomo')

class ProblemWriter_cpxlp(AbstractProblemWriter):

    alias('cpxlp')
    alias('lp')

    def __init__(self):

        AbstractProblemWriter.__init__(self, ProblemFormat.cpxlp)

        # the LP writer is responsible for tracking which variables are
        # referenced in constraints, so that one doesn't end up with a
        # zillion "unreferenced variables" warning messages. stored at
        # the object level to avoid additional method arguments.
        # dictionary of id(_VarData)->_VarData.
        self._referenced_variable_ids = {}

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
        output_fixed_variable_bounds = io_options.pop("output_fixed_variable_bounds", False)
        labeler = io_options.pop("labeler", None)
        # How much effort do we want to put into ensuring the
        # LP file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)
        if io_options:
            logger.warn(
                "ProblemWriter_cpxlp passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(io_options)))

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError("ProblemWriter_cpxlp: Using both the "
                             "'symbolic_solver_labels' and 'labeler' "
                             "I/O options is forbidden")

        if symbolic_solver_labels:
            labeler = TextLabeler()
        elif labeler is None:
            labeler = NumericLabeler('x')

        # when sorting, there are a non-trivial number of temporary objects
        # created. these all yield non-circular references, so disable GC -
        # the overhead is non-trivial, and because references are non-circular,
        # everything will be collected immediately anyway.
        suspend_gc = PauseGC()

        # clear the collection of referenced variables.
        self._referenced_variable_ids.clear()

        if output_filename is None:
            output_filename = model.name + ".lp"

        output_file=open(output_filename, "w")
        symbol_map = self._print_model_LP(model,
                                          output_file,
                                          solver_capability,
                                          labeler,
                                          output_fixed_variable_bounds,
                                          file_determinism=file_determinism)
        output_file.close()

        self._referenced_variable_ids.clear()

        return output_filename, symbol_map

    def _get_bound(self, exp):

        if exp.is_fixed():
            return exp()
        else:
            raise ValueError("ERROR: non-fixed bound: " + str(exp))

    def _print_expr_linear(self, x, output_file, object_symbol_dictionary, variable_symbol_dictionary, is_objective):

        """
        Return a expression as a string in LP format.

        Note that this function does not handle any differences in LP format
        interpretation by the solvers (e.g. CPlex vs GLPK).  That decision is
        left up to the caller.

        required arguments:
          x: A Pyomo linear encoding of an expression to write in LP format
        """

        # Currently, it appears that we only need to print the constant
        # offset term for objectives.
        print_offset = is_objective

        constant_term = x[0]
        linear_terms = x[1]

        name_to_coefficient_map = {}

        for coefficient, vardata in linear_terms:

            var_id = id(vardata)
            self._referenced_variable_ids[var_id] = vardata

            name = variable_symbol_dictionary[var_id]

            # due to potential disabling of expression simplification,
            # variables might appear more than once - condense coefficients.
            name_to_coefficient_map[name] = coefficient + name_to_coefficient_map.get(name,0.0)

        sorted_names = sorted(iterkeys(name_to_coefficient_map))

        string_template = '%+'+self._precision_string+' %s\n'
        for name in sorted_names:

            coefficient = name_to_coefficient_map[name]

            output_file.write(string_template % (coefficient, name))

        if print_offset and (constant_term != 0.0):

            output_file.write(string_template % (constant_term, 'ONE_VAR_CONSTANT'))

        return constant_term


    def _print_expr_canonical(self, x, output_file, object_symbol_dictionary, variable_symbol_dictionary, is_objective):

        """
        Return a expression as a string in LP format.

        Note that this function does not handle any differences in LP format
        interpretation by the solvers (e.g. CPlex vs GLPK).  That decision is
        left up to the caller.

        required arguments:
          x: A Pyomo canonical expression to write in LP format
        """

        # cache - this is referenced numerous times.
        if isinstance(x, LinearCanonicalRepn):
            var_hashes = None # not needed
        else:
            var_hashes = x[-1]

        #
        # Linear
        #
        linear_coef_string_template = '%+'+self._precision_string+' %s\n'
        if isinstance(x, LinearCanonicalRepn) and (x.linear is not None):

            # the 99% case is when the input instance is a linear canonical expression, so the exception should be rare.
            for i in xrange(0,len(x.linear)):
                var_coef = x.linear[i]
                vardata = x.variables[i]
                self._referenced_variable_ids[id(vardata)] = vardata

            sorted_names = [(variable_symbol_dictionary[id(x.variables[i])], x.linear[i]) for i in xrange(0,len(x.linear))]
            sorted_names.sort()

            for name, coef in sorted_names:
                output_file.write(linear_coef_string_template % (coef, name))
        elif 1 in x:

            for var_hash in x[1]:
                vardata = var_hashes[var_hash]
                self._referenced_variable_ids[id(vardata)] = vardata

            sorted_names = [(variable_symbol_dictionary[id(var_hashes[var_hash])], var_coefficient) for var_hash, var_coefficient in iteritems(x[1])]
            sorted_names.sort()

            for name, coef in sorted_names:
                output_file.write(linear_coef_string_template % (coef, name))

        #
        # Quadratic
        #
        quad_coef_string_template = '%+'+self._precision_string+' '
        if canonical_degree(x) is 2:

            # first, make sure there is something to output - it is possible for all
            # terms to have coefficients equal to 0.0, in which case you don't want
            # to get into the bracket notation at all.
            # NOTE: if the coefficient is really 0.0, it should be preprocessed out by
            #       the canonial expression generator!
            found_nonzero_term = False # until proven otherwise
            for var_hash, var_coefficient in iteritems(x[2]):
                for var in var_hash:
                    vardata = var_hashes[var]

                if math.fabs(var_coefficient) != 0.0:
                    found_nonzero_term = True
                    break

            if found_nonzero_term:

                output_file.write("+ [\n")

                num_output = 0

                for var_hash in sorted(iterkeys(x[2])):

                    coefficient = x[2][var_hash]

                    if is_objective:
                        coefficient *= 2

                    # times 2 because LP format requires /2 for all the quadratic
                    # terms /of the objective only/.  Discovered the last bit thru
                    # trial and error.  Obnoxious.
                    # Ref: ILog CPlex 8.0 User's Manual, p197.

                    output_file.write(quad_coef_string_template % coefficient)
                    term_variables = []

                    for var in var_hash:
                        vardata = var_hashes[var]
                        self._referenced_variable_ids[id(vardata)] = vardata
                        name = variable_symbol_dictionary[id(vardata)]
                        term_variables.append(name)

                    if len(term_variables) == 2:
                        output_file.write("%s * %s" % (term_variables[0],term_variables[1]))
                    else:
                        output_file.write("%s ^ 2" % term_variables[0])
                    output_file.write("\n")

                output_file.write("]")

                if is_objective:
                    output_file.write(' / 2\n')
                    # divide by 2 because LP format requires /2 for all the quadratic
                    # terms.  Weird.  Ref: ILog CPlex 8.0 User's Manual, p197
                else:
                    output_file.write("\n")


        #
        # Constant offset
        #
        if isinstance(x, LinearCanonicalRepn):
            constant = x.constant
        else:
            if 0 in x:
                constant = x[0][None]
            else:
                constant = None

        if constant is not None:
            offset = constant
        else:
            offset=0.0

        # Currently, it appears that we only need to print the constant
        # offset term for objectives.
        obj_string_template = '%+'+self._precision_string+' %s\n'
        if is_objective and offset != 0.0:
            output_file.write(obj_string_template % (offset, 'ONE_VAR_CONSTANT'))

        #
        # Return constant offset
        #
        return offset

    def printSOS(self, symbol_map, labeler, variable_symbol_map, soscondata, output_file):
        """
        Prints the SOS constraint associated with the _SOSConstraintData object
        """

        sos_items = soscondata.get_items()
        level = soscondata.get_level()

        if len(sos_items) == 0:
            return

        output_file.write('%s: S%s::\n' % (symbol_map.getSymbol(soscondata,labeler), level))

        sos_template_string = "%s:%"+self._precision_string+"\n"
        for vardata, weight in sos_items:
            if vardata.fixed:
                raise RuntimeError("SOSConstraint '%s' includes a fixed variable '%s'. "
                                   "This is currently not supported. Deactive this constraint "
                                   "in order to proceed" % (soscondata.cname(True), vardata.cname(True)))
            self._referenced_variable_ids[id(vardata)] = vardata
            output_file.write(sos_template_string % (variable_symbol_map.getSymbol(vardata), weight) )

    #
    # a simple utility to pass through each variable and constraint
    # in the input model and populate the symbol map accordingly. once
    # we know all of the objects in the model, we can then "freeze"
    # the contents and use more efficient query mechanisms on simple
    # dictionaries - and avoid checking and function call overhead.
    #
    def _populate_symbol_map(self, model, symbol_map, labeler, variable_symbol_map, file_determinism=1):

        # NOTE: we use createSymbol instead of getSymbol because we know
        # whether or not the symbol exists, and don't want to the overhead
        # of error/duplicate checking.

        # cache frequently called functions
        create_symbol_func = SymbolMap.createSymbol
        create_symbols_func = SymbolMap.createSymbols
        alias_symbol_func = SymbolMap.alias
        variable_label_pairs = []

        # nested by block
        objective_list = []
        constraint_list = []
        # flat
        sosconstraint_list = []
        variable_list = []
        sort_kwds = {}
        if file_determinism >= 1:
            sort_kwds['sort_by_keys'] = True
        if file_determinism >= 2:
            sort_kwds['sort_by_names'] = True
        for block in model.all_blocks(**sort_kwds):

            block_objective_list = []
            for objective_data in active_components_data(block, Objective, **sort_kwds):
                block_objective_list.append(objective_data)
                create_symbol_func(symbol_map, objective_data, labeler)
            objective_list.append((block, block_objective_list))

            block_constraint_list = []
            for constraint_data in active_components_data(block, Constraint, **sort_kwds):
                block_constraint_list.append(constraint_data)
                constraint_data_symbol = create_symbol_func(symbol_map,
                                                            constraint_data,
                                                            labeler)
                if constraint_data._equality:
                    label = 'c_e_' + constraint_data_symbol + '_'
                    alias_symbol_func(symbol_map, constraint_data, label)
                else:
                    if constraint_data.lower is not None:
                        if constraint_data.upper is not None:
                            alias_symbol_func(symbol_map,
                                              constraint_data,
                                              'r_l_' + constraint_data_symbol + '_')
                            alias_symbol_func(symbol_map,
                                              constraint_data,
                                              'r_u_' + constraint_data_symbol + '_')
                        else:
                            label = 'c_l_' + constraint_data_symbol + '_'
                            alias_symbol_func(symbol_map, constraint_data, label)
                    elif constraint_data.upper is not None:
                        label = 'c_u_' + constraint_data_symbol + '_'
                        alias_symbol_func(symbol_map, constraint_data, label)
            constraint_list.append((block,block_constraint_list))

            for condata in active_components_data(block, SOSConstraint, **sort_kwds):
                sosconstraint_list.append(condata)
                create_symbol_func(symbol_map, condata, labeler)

            for vardata in active_components_data(block, Var, **sort_kwds):
                variable_list.append(vardata)
                variable_label_pairs.append(
                    (vardata,create_symbol_func(symbol_map, vardata, labeler)))

        variable_symbol_map.updateSymbols(variable_label_pairs)

        return objective_list, constraint_list, sosconstraint_list, variable_list

    def _print_model_LP(self,
                        model,
                        output_file,
                        solver_capability,
                        labeler,
                        output_fixed_variable_bounds,
                        file_determinism=1):

        symbol_map = SymbolMap(model)
        variable_symbol_map = BasicSymbolMap()

        # populate the symbol map in a single pass.
        objective_list, constraint_list, sosconstraint_list, variable_list \
            = self._populate_symbol_map(model, symbol_map, labeler, variable_symbol_map, file_determinism=file_determinism)

        # and extract the information we'll need for rapid labeling.
        object_symbol_dictionary = symbol_map.getByObjectDictionary()
        variable_symbol_dictionary = variable_symbol_map.byObject

        # cache - these are called all the time.
        print_expr_linear = self._print_expr_linear
        print_expr_canonical = self._print_expr_canonical

        # print the model name and the source, so we know roughly where
        # it came from.
        #
        # NOTE: this *must* use the "\* ... *\" comment format: the GLPK
        # LP parser does not correctly handle other formats (notably, "%").
        output_file.write(
            "\\* Source Pyomo model name=%s *\\\n\n" % (model.name,) )

        #
        # Objective
        #

        supports_quadratic_objective = solver_capability('quadratic_objective')

        numObj = 0
        onames = []
        #for block in model.all_blocks(sort_by_keys=True, sort_by_names=True):
        for block, block_objectives in objective_list:

            block_canonical_repn = getattr(block,"canonical_repn",None)

            if len(block_objectives):
                if block_canonical_repn is None:
                    raise ValueError("No canonical_repn ComponentMap was found on "
                                     "block with name %s. Did you forget to preprocess?"
                                     % (block.cname(True)))

            for objective_data in block_objectives:

                numObj += 1
                onames.append(objective_data.cname())
                if numObj > 1:
                    msg = "More than one active objective defined for input model '%s'; " \
                          'Cannot write legal LP file\n'                           \
                          'Objectives: %s'
                    raise ValueError(
                        #msg % ( model.name,', '.join("'%s'" % x.cname(True) for x in _obj) ))
                        msg % ( model.name,' '.join(onames)))
                #
                symbol_map.alias(objective_data, '__default_objective__')
                if objective_data.is_minimizing():
                    output_file.write("min \n")
                else:
                    output_file.write("max \n")
                #

                obj_data_repn = block_canonical_repn.get(objective_data)
                if obj_data_repn is None:
                    raise ValueError("No entry found in canonical_repn ComponentMap on "
                                     "block %s for active objective with name %s. "
                                     "Did you forget to preprocess?"
                                     % (block.cname(True), objective_data.cname(True)))

                degree = canonical_degree(obj_data_repn)
                #
                if degree == 0:
                    print("Warning: Constant objective detected, replacing " +
                          "with a placeholder to prevent solver failure.")
                    output_file.write(object_symbol_dictionary[id(objective_data)] + ": +0.0 ONE_VAR_CONSTANT\n")
                        # Skip the remaining logic of the section
                else:
                    if degree == 2:
                        if not supports_quadratic_objective:
                            raise RuntimeError(
                                'Selected solver is unable to handle objective functions with quadratic terms. ' \
                                'Objective at issue: %s.' % objective_data.cname())
                    elif degree != 1:
                        msg  = "Cannot write legal LP file.  Objective '%s' "  \
                               'has nonlinear terms that are not quadratic.' % objective_data.cname(True)
                        raise RuntimeError(msg)

                    output_file.write(object_symbol_dictionary[id(objective_data)]+':\n')

                    offset = print_expr_canonical( obj_data_repn,
                                                   output_file,
                                                   object_symbol_dictionary,
                                                   variable_symbol_dictionary,
                                                   True )

        if numObj == 0:
            msg = "ERROR: No objectives defined for input model '%s'; "    \
                  ' cannot write legal LP file'
            raise ValueError(msg % str( model.name ))


        # Constraints
        #
        # If there are no non-trivial constraints, you'll end up with an empty
        # constraint block. CPLEX is OK with this, but GLPK isn't. And
        # eliminating the constraint block (i.e., the "s.t." line) causes GLPK
        # to whine elsewhere. Output a warning if the constraint block is empty,
        # so users can quickly determine the cause of the solve failure.

        output_file.write("\n")
        output_file.write("s.t.\n")
        output_file.write("\n")

        have_nontrivial = False

        supports_quadratic_constraint = solver_capability('quadratic_constraint')

        # FIXME: This is a hack to get nested blocks working...
        eq_string_template = "= %"+self._precision_string+'\n'
        geq_string_template = ">= %"+self._precision_string+'\n\n'
        leq_string_template = "<= %"+self._precision_string+'\n\n'
        #for block in model.all_blocks(sort_by_keys=True, sort_by_names=True):
        for block, block_constraints in constraint_list:

            block_canonical_repn = getattr(block,"canonical_repn",None)
            block_lin_body = getattr(block,"lin_body",None)

            if len(block_constraints):
                have_nontrivial=True
                if (block_canonical_repn is None) and (block_lin_body is None):
                    raise ValueError("No canonical_repn ComponentMap was found on "
                                     "block with name %s. Did you forget to preprocess?"
                                     % (block.cname(True)))

            for constraint_data in block_constraints:

                # if expression trees have been linearized, then the canonical
                # representation attribute on the constraint data object will
                # be equal to None.
                constraint_data_repn = block_canonical_repn.get(constraint_data)
                lin_body = None
                if constraint_data_repn is None:
                    lin_body = block_lin_body.get(constraint_data)
                    if lin_body is None:
                        raise ValueError("No entry found in canonical_repn ComponentMap on "
                                         "block %s for active constraint with name %s. "
                                         "Did you forget to preprocess?"
                                         % (block.cname(True), constraint_data.cname(True)))

                if constraint_data_repn is not None:

                    degree = canonical_degree(constraint_data_repn)

                    # There are conditions, e.g., when fixing variables, under which
                    # a constraint block might be empty.  Ignore these, for both
                    # practical reasons and the fact that the CPLEX LP format
                    # requires a variable in the constraint body.  It is also
                    # possible that the body of the constraint consists of only a
                    # constant, in which case the "variable" of
                    if degree == 0:
                        # this happens *all* the time in many applications,
                        # including PH - so suppress the warning.
                        #
                        #msg = 'WARNING: ignoring constraint %s[%s] which is ' \
                            #      'constant'
                        #print msg % (str(C),str(index))
                        continue

                    if degree == 2:
                        if not supports_quadratic_constraint:
                            msg  = 'Solver unable to handle quadratic expressions.'\
                                   "  Constraint at issue: '%s%%s'"
                            msg %= constraint.name
                            if index is None:
                                msg %= ''
                            else:
                                msg %= '%s' % ( str(index)
                                                .replace('(', '[')
                                                .replace(')', ']')
                                            )

                            raise ValueError(msg)

                    elif degree != 1:
                        msg = "Cannot write legal LP file.  Constraint '%s%s' "   \
                              'has a body with nonlinear terms.'
                        if index is None:
                            msg %= ( constraint.name, '')
                        else:
                            msg %= ( constraint.name, '[%s]' % index )
                        raise ValueError(msg)

                con_symbol = object_symbol_dictionary[id(constraint_data)]
                if constraint_data._equality:
                    label = 'c_e_' + con_symbol + '_'
                    output_file.write(label+':\n')
                    try:
                        if lin_body is not None:
                            offset = print_expr_linear(lin_body, output_file, object_symbol_dictionary, variable_symbol_dictionary, False)
                        else:
                            offset = print_expr_canonical(constraint_data_repn, output_file, object_symbol_dictionary, variable_symbol_dictionary, False)
                    except:
                        raise RuntimeError(
                            "Failed to write constraint=%s to LP file - was it preprocessed correctly?" % constraint_data.cname())
                    bound = constraint_data.lower
                    bound = self._get_bound(bound) - offset
                    output_file.write(eq_string_template%bound)
                    output_file.write("\n")
                else:
                    if constraint_data.lower is not None:
                        if constraint_data.upper is not None:
                            label = 'r_l_' + con_symbol + '_'
                        else:
                            label = 'c_l_' + con_symbol + '_'
                        output_file.write(label+':\n')
                        try:
                            if lin_body is not None:
                                offset = print_expr_linear(lin_body, output_file, object_symbol_dictionary, variable_symbol_dictionary, False)
                            else:
                                offset = print_expr_canonical(constraint_data_repn, output_file, object_symbol_dictionary, variable_symbol_dictionary, False)
                        except:
                            raise RuntimeError(
                                "Failed to write constraint=%s to LP file - was it preprocessed correctly?" % constraint_data.cname())
                        bound = constraint_data.lower
                        bound = self._get_bound(bound) - offset
                        output_file.write(geq_string_template%bound)
                    if constraint_data.upper is not None:
                        if constraint_data.lower is not None:
                            label = 'r_u_' + con_symbol + '_'
                        else:
                            label = 'c_u_' + con_symbol + '_'
                        output_file.write(label+':\n')
                        try:
                            if lin_body is not None:
                                offset = print_expr_linear(lin_body, output_file, object_symbol_dictionary, variable_symbol_dictionary, False)
                            else:
                                offset = print_expr_canonical(constraint_data_repn, output_file, object_symbol_dictionary, variable_symbol_dictionary, False)
                        except:
                            raise RuntimeError(
                                "Failed to write constraint=%s to LP file - was it preprocessed correctly?" % constraint_data.cname())
                        bound = constraint_data.upper
                        bound = self._get_bound(bound) - offset
                        output_file.write(leq_string_template%bound)

        if not have_nontrivial:
            print('WARNING: Empty constraint block written in LP format '  \
                  '- solver may error')

        # the CPLEX LP format doesn't allow constants in the objective (or
        # constraint body), which is a bit silly.  To avoid painful
        # book-keeping, we introduce the following "variable", constrained
        # to the value 1.  This is used when quadratic terms are present.
        # worst-case, if not used, is that CPLEX easily pre-processes it out.
        prefix = ""
        output_file.write('%sc_e_ONE_VAR_CONSTANT: \n' % prefix)
        output_file.write('%sONE_VAR_CONSTANT = 1.0\n' % prefix)
        output_file.write("\n")

        # SOS constraints
        #
        # For now, we write out SOS1 and SOS2 constraints in the cplex format
        #
        # All Component objects are stored in model._component, which is a
        # dictionary of {class: {objName: object}}.
        #
        # Consider the variable X,
        #
        #   model.X = Var(...)
        #
        # We print X to CPLEX format as X(i,j,k,...) where i, j, k, ... are the
        # indices of X.
        #
        SOSlines = StringIO()
        sos1 = solver_capability("sos1")
        sos2 = solver_capability("sos2")
        writtenSOS = False
        #for block in model.all_blocks(sort_by_keys=True, sort_by_names=True):
        for soscondata in sosconstraint_list:
            level = soscondata.get_level()
            if (level == 1 and not sos1) or (level == 2 and not sos2) or (level > 2):
                raise ValueError("Solver does not support SOS level %s constraints"
                                 % (level,) )
            if writtenSOS == False:
                SOSlines.write("SOS\n")
                writtenSOS = True
            # This updates the referenced_variable_ids, just in case
            # there is a variable that only appears in an
            # SOSConstraint, in which case this needs to be known
            # before we write the "bounds" section (Cplex does not
            # handle this correctly, Gurobi does)
            self.printSOS(symbol_map, labeler, variable_symbol_map, soscondata, SOSlines)

        #
        # Bounds
        #

        output_file.write("bounds \n")

        # Scan all variables even if we're only writing a subset of them.
        # required because we don't store maps by variable type currently.

        # FIXME: This is a hack to get nested blocks working...
        lb_string_template = "%"+self._precision_string+" <= "
        ub_string_template = " <= %"+self._precision_string+"\n"
        # Track the number of integer and binary variables, so you can
        # output their status later.
        integer_vars = []
        binary_vars = []
        for vardata in variable_list:

            # TODO: We could just loop over the set of items in
            #       self._referenced_variable_ids, except this is
            #       a dictionary that is hashed by id(vardata)
            #       which would make the bounds section
            #       nondeterministic (bad for unit testing)
            if id(vardata) not in self._referenced_variable_ids:
                continue

            if vardata.fixed:
                if not output_fixed_variable_bounds:
                    raise ValueError("Encountered a fixed variable (%s) inside an active objective "
                                     "or constraint expression on model %s, which is usually indicative of "
                                     "a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' "
                                     "to suppress this error and fix the variable by overwriting its bounds in "
                                     "the LP file."
                                     % (vardata.cname(True),model.cname(True),))
                if vardata.value is None:
                    raise ValueError("Variable cannot be fixed to a value of None.")
                vardata_lb = vardata.value
                vardata_ub = vardata.value
            else:
                vardata_lb = vardata.lb
                vardata_ub = vardata.ub

            name_to_output = variable_symbol_dictionary[id(vardata)]

            # track the number of integer and binary variables, so we know whether
            # to output the general / binary sections below.
            if vardata.is_integer():
                integer_vars.append(name_to_output)
            elif vardata.is_binary():
                binary_vars.append(name_to_output)
            elif not vardata.is_continuous():
                raise TypeError("Invalid domain type for variable with name '%s'. "
                                "Variable is not continuous, integer, or binary."
                                % ( vardata.cname(True),) )

            # in the CPLEX LP file format, the default variable
            # bounds are 0 and +inf.  These bounds are in
            # conflict with Pyomo, which assumes -inf and +inf
            # (which we would argue is more rational).
            output_file.write("   ")
            if vardata_lb is not None:
                output_file.write(lb_string_template % value(vardata_lb))
            else:
                output_file.write(" -inf <= ")
            if name_to_output == "e":
                msg = 'Attempting to write variable with name' \
                      "'e' in a CPLEX LP formatted file - "    \
                      'will cause a parse failure due to '     \
                      'confusion with numeric values '         \
                      'expressed in scientific notation'
                raise ValueError(msg)
            output_file.write(name_to_output)
            if vardata_ub is not None:
                output_file.write(ub_string_template % value(vardata_ub))
            else:
                output_file.write(" <= +inf\n")

        if len(integer_vars) > 0:

            output_file.write("general\n")
            for var_name in integer_vars:
                output_file.write('  %s\n' % var_name)

        if len(binary_vars) > 0:

            output_file.write("binary\n")
            for var_name in binary_vars:
                output_file.write('  %s\n' % var_name)


        # Write the SOS section
        output_file.write(SOSlines.getvalue())

        #
        # wrap-up
        #
        output_file.write("end \n")

        # Clean up the symbol map to only contain variables referenced
        # in the active constraints **Note**: warm start method may
        # rely on this for choosing the set of potential warm start
        # variables
        vars_to_delete = set(variable_symbol_map.byObject.keys())-set(self._referenced_variable_ids.keys())
        sm_byObject = symbol_map.byObject
        sm_bySymbol = symbol_map.bySymbol
        var_sm_byObject = variable_symbol_map.byObject
        for varid in vars_to_delete:
            symbol = var_sm_byObject[varid]
            del sm_byObject[varid]
            del sm_bySymbol[symbol]
        del variable_symbol_map

        return symbol_map
