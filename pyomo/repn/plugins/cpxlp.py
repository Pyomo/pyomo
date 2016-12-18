#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Problem Writer for CPLEX LP Format Files
#

import logging
import math
import operator

from six import iterkeys, iteritems, StringIO
from six.moves import xrange

from pyutilib.math import infinity
from pyutilib.misc import PauseGC
import pyomo.util.plugin
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter
from pyomo.core.base import \
    (SymbolMap, TextLabeler,
     NumericLabeler, Constraint, SortComponents,
     Var, value,
     SOSConstraint, Objective,
     ComponentMap, is_fixed)
from pyomo.repn import (generate_canonical_repn,
                        canonical_degree,
                        LinearCanonicalRepn)

logger = logging.getLogger('pyomo.core')

def _no_negative_zero(val):
    """Make sure -0 is never output. Makes diff tests easier."""
    if val == 0:
        return 0
    return val

class ProblemWriter_cpxlp(AbstractProblemWriter):

    pyomo.util.plugin.alias('cpxlp', 'Generate the corresponding CPLEX LP file')
    pyomo.util.plugin.alias('lp', 'Generate the corresponding CPLEX LP file')

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

    def __call__(self,
                 model,
                 output_filename,
                 solver_capability,
                 io_options):

        # Make sure not to modify the user's dictionary,
        # they may be reusing it outside of this call
        io_options = dict(io_options)

        # Skip writing constraints whose body section is
        # fixed (i.e., no variables)
        skip_trivial_constraints = \
            io_options.pop("skip_trivial_constraints", False)

        # Use full Pyomo component names in the LP file rather
        # than shortened symbols (slower, but useful for debugging).
        symbolic_solver_labels = \
            io_options.pop("symbolic_solver_labels", False)

        output_fixed_variable_bounds = \
            io_options.pop("output_fixed_variable_bounds", False)

        # If False, unused variables will not be included in
        # the LP file. Otherwise, include all variables in
        # the bounds sections.
        include_all_variable_bounds = \
            io_options.pop("include_all_variable_bounds", False)

        labeler = io_options.pop("labeler", None)

        # How much effort do we want to put into ensuring the
        # LP file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)

        # user defined orderings for variable and constraint
        # output
        row_order = io_options.pop("row_order", None)
        column_order = io_options.pop("column_order", None)

        # make sure the ONE_VAR_CONSTANT variable appears in
        # the objective even if the constant part of the
        # objective is zero
        force_objective_constant = \
            io_options.pop("force_objective_constant", False)

        if len(io_options):
            raise ValueError(
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

        # clear the collection of referenced variables.
        self._referenced_variable_ids.clear()

        if output_filename is None:
            output_filename = model.name + ".lp"

        # when sorting, there are a non-trivial number of
        # temporary objects created. these all yield
        # non-circular references, so disable GC - the
        # overhead is non-trivial, and because references
        # are non-circular, everything will be collected
        # immediately anyway.
        with PauseGC() as pgc:
            with open(output_filename, "w") as output_file:
                symbol_map = self._print_model_LP(
                    model,
                    output_file,
                    solver_capability,
                    labeler,
                    output_fixed_variable_bounds=output_fixed_variable_bounds,
                    file_determinism=file_determinism,
                    row_order=row_order,
                    column_order=column_order,
                    skip_trivial_constraints=skip_trivial_constraints,
                    force_objective_constant=force_objective_constant,
                    include_all_variable_bounds=include_all_variable_bounds)

        self._referenced_variable_ids.clear()

        return output_filename, symbol_map

    def _get_bound(self, exp):
        if exp is None:
            return None
        if is_fixed(exp):
            return value(exp)
        raise ValueError("non-fixed bound: " + str(exp))

    def _print_expr_canonical(self,
                              x,
                              output_file,
                              object_symbol_dictionary,
                              variable_symbol_dictionary,
                              is_objective,
                              column_order,
                              force_objective_constant=False):

        """
        Return a expression as a string in LP format.

        Note that this function does not handle any differences in LP format
        interpretation by the solvers (e.g. CPlex vs GLPK).  That decision is
        left up to the caller.

        required arguments:
          x: A Pyomo canonical expression to write in LP format
        """
        assert (not force_objective_constant) or (is_objective)

        # cache - this is referenced numerous times.
        if isinstance(x, LinearCanonicalRepn):
            var_hashes = None # not needed
        else:
            var_hashes = x[-1]

        #
        # Linear
        #
        linear_coef_string_template = '%+'+self._precision_string+' %s\n'
        if isinstance(x, LinearCanonicalRepn):

            #
            # optimization (these might be generated on the fly)
            #
            coefficients = x.linear
            if coefficients is not None:
                variables = x.variables

                # the 99% case is when the input instance is a linear
                # canonical expression, so the exception should be rare.
                for vardata in variables:
                    self._referenced_variable_ids[id(vardata)] = vardata

                if column_order is None:
                    sorted_names = [(variable_symbol_dictionary[id(variables[i])],
                                     coefficients[i])
                                    for i in xrange(0,len(coefficients))]
                    sorted_names.sort()
                else:
                    sorted_names = [(variables[i], coefficients[i])
                                    for i in xrange(0,len(coefficients))]
                    sorted_names.sort(key=lambda _x: column_order[_x[0]])
                    sorted_names = [(variable_symbol_dictionary[id(var)], coef)
                                    for var, coef in sorted_names]

                for name, coef in sorted_names:
                    output_file.write(linear_coef_string_template % (coef, name))

            elif not is_objective:
                # If we made it to here we are outputing
                # trivial constraints place 0 *
                # ONE_VAR_CONSTANT on this side of the
                # constraint for the benefit of solvers like
                # Glpk that cannot parse an LP file without
                # a variable on the left hand side.
                output_file.write(linear_coef_string_template
                                  % (0, 'ONE_VAR_CONSTANT'))

        elif 1 in x:

            for var_hash in x[1]:
                vardata = var_hashes[var_hash]
                self._referenced_variable_ids[id(vardata)] = vardata

            if column_order is None:
                sorted_names = [(variable_symbol_dictionary[id(var_hashes[var_hash])],
                                 var_coefficient)
                                for var_hash, var_coefficient in iteritems(x[1])]
                sorted_names.sort()
            else:
                sorted_names = [(var_hashes[var_hash], var_coefficient)
                                for var_hash, var_coefficient in iteritems(x[1])]
                sorted_names.sort(key=lambda _x: column_order[_x[0]])
                sorted_names = [(variable_symbol_dictionary[id(var)], coef)
                                for var, coef in sorted_names]

            for name, coef in sorted_names:
                output_file.write(linear_coef_string_template % (coef, name))

        #
        # Quadratic
        #
        quad_coef_string_template = '%+'+self._precision_string+' '
        if canonical_degree(x) == 2:

            # first, make sure there is something to output
            # - it is possible for all terms to have
            # coefficients equal to 0.0, in which case you
            # don't want to get into the bracket notation at
            # all.
            # NOTE: if the coefficient is really 0.0, it
            #       should be preprocessed out by the
            #       canonial expression generator!
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

                var_hashes_order = list(iterkeys(x[2]))
                # sort by the sorted tuple of symbols (or column assignments)
                # for the variables appearing in the term
                if column_order is None:
                    var_hashes_order.sort(
                        key=lambda term: \
                          sorted(variable_symbol_dictionary[id(var_hashes[vh])]
                                 for vh in term))
                else:
                    var_hashes_order.sort(
                        key=lambda term: sorted(column_order[var_hashes[vh]]
                                                for vh in term))

                for var_hash in var_hashes_order:

                    coefficient = x[2][var_hash]

                    if is_objective:
                        coefficient *= 2

                    # times 2 because LP format requires /2 for all the quadratic
                    # terms /of the objective only/.  Discovered the last bit thru
                    # trial and error.  Obnoxious.
                    # Ref: ILog CPlex 8.0 User's Manual, p197.

                    output_file.write(quad_coef_string_template % coefficient)
                    term_variables = []

                    var_hash_order = list(iterkeys(var_hash))
                    # sort by symbols (or column assignments)
                    if column_order is None:
                        var_hash_order.sort(
                            key=lambda vh: \
                              variable_symbol_dictionary[id(var_hashes[vh])])
                    else:
                        var_hash_order.sort(
                            key=lambda vh: column_order[var_hashes[vh]])

                    # sort the term for consistent output
                    for var in var_hash_order:
                        vardata = var_hashes[var]
                        self._referenced_variable_ids[id(vardata)] = vardata
                        name = variable_symbol_dictionary[id(vardata)]
                        term_variables.append(name)

                    if len(term_variables) == 2:
                        output_file.write("%s * %s"
                                          % (term_variables[0], term_variables[1]))
                    else:
                        output_file.write("%s ^ 2" % (term_variables[0]))
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

        # Currently, it appears that we only need to print
        # the constant offset term for objectives.
        obj_string_template = '%+'+self._precision_string+' %s\n'
        if is_objective and (force_objective_constant or (offset != 0.0)):
            output_file.write(obj_string_template
                              % (offset, 'ONE_VAR_CONSTANT'))

        #
        # Return constant offset
        #
        return offset

    def printSOS(self,
                 symbol_map,
                 labeler,
                 variable_symbol_map,
                 soscondata,
                 output_file):
        """
        Prints the SOS constraint associated with the _SOSConstraintData object
        """
        if soscondata.num_variables() == 0:
            return

        sos_items = list(soscondata.get_items())
        level = soscondata.level

        output_file.write('%s: S%s::\n'
                          % (symbol_map.getSymbol(soscondata,labeler), level))

        sos_template_string = "%s:%"+self._precision_string+"\n"
        for vardata, weight in sos_items:
            if vardata.fixed:
                raise RuntimeError(
                    "SOSConstraint '%s' includes a fixed variable '%s'. This is "
                    "currently not supported. Deactive this constraint in order to "
                    "proceed." % (soscondata.name, vardata.name))
            self._referenced_variable_ids[id(vardata)] = vardata
            output_file.write(sos_template_string
                              % (variable_symbol_map.getSymbol(vardata),
                                 weight))

    def _print_model_LP(self,
                        model,
                        output_file,
                        solver_capability,
                        labeler,
                        output_fixed_variable_bounds=False,
                        file_determinism=1,
                        row_order=None,
                        column_order=None,
                        skip_trivial_constraints=False,
                        force_objective_constant=False,
                        include_all_variable_bounds=False):

        symbol_map = SymbolMap()
        variable_symbol_map = SymbolMap()
        # NOTE: we use createSymbol instead of getSymbol because we
        #       know whether or not the symbol exists, and don't want
        #       to the overhead of error/duplicate checking.
        # cache frequently called functions
        create_symbol_func = SymbolMap.createSymbol
        create_symbols_func = SymbolMap.createSymbols
        alias_symbol_func = SymbolMap.alias
        variable_label_pairs = []

        # populate the symbol map in a single pass.
        #objective_list, constraint_list, sosconstraint_list, variable_list \
        #    = self._populate_symbol_map(model,
        #                                symbol_map,
        #                                labeler,
        #                                variable_symbol_map,
        #                                file_determinism=file_determinism)
        sortOrder = SortComponents.unsorted
        if file_determinism >= 1:
            sortOrder = sortOrder | SortComponents.indices
            if file_determinism >= 2:
                sortOrder = sortOrder | SortComponents.alphabetical

        #
        # Create variable symbols (and cache the block list)
        #
        all_blocks = []
        variable_list = []
        for block in model.block_data_objects(active=True,
                                              sort=sortOrder):

            all_blocks.append(block)

            for vardata in block.component_data_objects(
                    Var,
                    active=True,
                    sort=sortOrder,
                    descend_into=False):

                variable_list.append(vardata)
                variable_label_pairs.append(
                    (vardata,create_symbol_func(symbol_map,
                                                vardata,
                                                labeler)))

        variable_symbol_map.addSymbols(variable_label_pairs)

        # and extract the information we'll need for rapid labeling.
        object_symbol_dictionary = symbol_map.byObject
        variable_symbol_dictionary = variable_symbol_map.byObject

        # cache - these are called all the time.
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

        supports_quadratic_objective = \
            solver_capability('quadratic_objective')

        numObj = 0
        onames = []
        for block in all_blocks:

            gen_obj_canonical_repn = \
                getattr(block, "_gen_obj_canonical_repn", True)

            # Get/Create the ComponentMap for the repn
            if not hasattr(block,'_canonical_repn'):
                block._canonical_repn = ComponentMap()
            block_canonical_repn = block._canonical_repn

            for objective_data in block.component_data_objects(
                    Objective,
                    active=True,
                    sort=sortOrder,
                    descend_into=False):

                numObj += 1
                onames.append(objective_data.name)
                if numObj > 1:
                    raise ValueError(
                        "More than one active objective defined for input "
                        "model '%s'; Cannot write legal LP file\n"
                        "Objectives: %s" % (model.name, ' '.join(onames)))

                create_symbol_func(symbol_map,
                                   objective_data,
                                   labeler)

                symbol_map.alias(objective_data, '__default_objective__')
                if objective_data.is_minimizing():
                    output_file.write("min \n")
                else:
                    output_file.write("max \n")

                if gen_obj_canonical_repn:
                    canonical_repn = \
                        generate_canonical_repn(objective_data.expr)
                    block_canonical_repn[objective_data] = canonical_repn
                else:
                    canonical_repn = block_canonical_repn[objective_data]

                degree = canonical_degree(canonical_repn)

                if degree == 0:
                    logger.warning("Constant objective detected, replacing "
                          "with a placeholder to prevent solver failure.")
                    force_objective_constant = True
                elif degree == 2:
                    if not supports_quadratic_objective:
                        raise RuntimeError(
                            "Selected solver is unable to handle "
                            "objective functions with quadratic terms. "
                            "Objective at issue: %s."
                            % objective_data.name)
                elif degree != 1:
                    raise RuntimeError(
                        "Cannot write legal LP file.  Objective '%s' "
                        "has nonlinear terms that are not quadratic."
                        % objective_data.name)

                output_file.write(
                    object_symbol_dictionary[id(objective_data)]+':\n')

                offset = print_expr_canonical(
                    canonical_repn,
                    output_file,
                    object_symbol_dictionary,
                    variable_symbol_dictionary,
                    True,
                    column_order,
                    force_objective_constant=force_objective_constant)

        if numObj == 0:
            raise ValueError(
                "ERROR: No objectives defined for input model '%s'; "
                " cannot write legal LP file" % str(model.name))

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

        def constraint_generator():
            for block in all_blocks:

                gen_con_canonical_repn = \
                    getattr(block, "_gen_con_canonical_repn", True)

                # Get/Create the ComponentMap for the repn
                if not hasattr(block,'_canonical_repn'):
                    block._canonical_repn = ComponentMap()
                block_canonical_repn = block._canonical_repn

                for constraint_data in block.component_data_objects(
                        Constraint,
                        active=True,
                        sort=sortOrder,
                        descend_into=False):

                    if isinstance(constraint_data, LinearCanonicalRepn):
                        canonical_repn = constraint_data
                    else:
                        if gen_con_canonical_repn:
                            canonical_repn = generate_canonical_repn(constraint_data.body)
                            block_canonical_repn[constraint_data] = canonical_repn
                        else:
                            canonical_repn = block_canonical_repn[constraint_data]

                    yield constraint_data, canonical_repn

        if row_order is not None:
            sorted_constraint_list = list(constraint_generator())
            sorted_constraint_list.sort(key=lambda x: row_order[x[0]])
            def yield_all_constraints():
                for constraint_data, canonical_repn in sorted_constraint_list:
                    yield constraint_data, canonical_repn
        else:
            yield_all_constraints = constraint_generator

        # FIXME: This is a hack to get nested blocks working...
        eq_string_template = "= %"+self._precision_string+'\n'
        geq_string_template = ">= %"+self._precision_string+'\n\n'
        leq_string_template = "<= %"+self._precision_string+'\n\n'
        for constraint_data, canonical_repn in yield_all_constraints():
            have_nontrivial = True

            degree = canonical_degree(canonical_repn)

            #
            # Write constraint
            #

            # There are conditions, e.g., when fixing variables, under which
            # a constraint block might be empty.  Ignore these, for both
            # practical reasons and the fact that the CPLEX LP format
            # requires a variable in the constraint body.  It is also
            # possible that the body of the constraint consists of only a
            # constant, in which case the "variable" of
            if degree == 0:
                if skip_trivial_constraints:
                    continue
            elif degree == 2:
                if not supports_quadratic_constraint:
                    raise ValueError(
                        "Solver unable to handle quadratic expressions. Constraint"
                        " at issue: '%s'" % (constraint_data.name))
            elif degree != 1:
                raise ValueError(
                    "Cannot write legal LP file.  Constraint '%s' has a body "
                    "with nonlinear terms." % (constraint_data.name))

            # Create symbol
            con_symbol = create_symbol_func(symbol_map, constraint_data, labeler)

            if constraint_data.equality:
                label = 'c_e_' + con_symbol + '_'
                alias_symbol_func(symbol_map, constraint_data, label)
                output_file.write(label+':\n')
                offset = print_expr_canonical(canonical_repn,
                                              output_file,
                                              object_symbol_dictionary,
                                              variable_symbol_dictionary,
                                              False,
                                              column_order)
                bound = constraint_data.lower
                bound = self._get_bound(bound) - offset
                output_file.write(eq_string_template
                                  % (_no_negative_zero(bound)))
                output_file.write("\n")
            else:
                if constraint_data.lower is not None:
                    if constraint_data.upper is not None:
                        label = 'r_l_' + con_symbol + '_'
                    else:
                        label = 'c_l_' + con_symbol + '_'
                    alias_symbol_func(symbol_map, constraint_data, label)
                    output_file.write(label+':\n')
                    offset = print_expr_canonical(canonical_repn,
                                                  output_file,
                                                  object_symbol_dictionary,
                                                  variable_symbol_dictionary,
                                                  False,
                                                  column_order)
                    bound = constraint_data.lower
                    bound = self._get_bound(bound) - offset
                    output_file.write(geq_string_template
                                      % (_no_negative_zero(bound)))
                if constraint_data.upper is not None:
                    if constraint_data.lower is not None:
                        label = 'r_u_' + con_symbol + '_'
                    else:
                        label = 'c_u_' + con_symbol + '_'
                    alias_symbol_func(symbol_map, constraint_data, label)
                    output_file.write(label+':\n')
                    offset = print_expr_canonical(canonical_repn,
                                                  output_file,
                                                  object_symbol_dictionary,
                                                  variable_symbol_dictionary,
                                                  False,
                                                  column_order)
                    bound = constraint_data.upper
                    bound = self._get_bound(bound) - offset
                    output_file.write(leq_string_template
                                      % (_no_negative_zero(bound)))

        if not have_nontrivial:
            logger.warning('Empty constraint block written in LP format '  \
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
        for block in all_blocks:

            for soscondata in block.component_data_objects(
                    SOSConstraint,
                    active=True,
                    sort=sortOrder,
                    descend_into=False):

                create_symbol_func(symbol_map, soscondata, labeler)

                level = soscondata.level
                if (level == 1 and not sos1) or \
                   (level == 2 and not sos2) or \
                   (level > 2):
                    raise ValueError(
                        "Solver does not support SOS level %s constraints" % (level))
                if writtenSOS == False:
                    SOSlines.write("SOS\n")
                    writtenSOS = True
                # This updates the referenced_variable_ids, just in case
                # there is a variable that only appears in an
                # SOSConstraint, in which case this needs to be known
                # before we write the "bounds" section (Cplex does not
                # handle this correctly, Gurobi does)
                self.printSOS(symbol_map,
                              labeler,
                              variable_symbol_map,
                              soscondata,
                              SOSlines)

        #
        # Bounds
        #

        output_file.write("bounds\n")

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
            if (not include_all_variable_bounds) and \
               (id(vardata) not in self._referenced_variable_ids):
                continue

            if vardata.fixed:
                if not output_fixed_variable_bounds:
                    raise ValueError(
                        "Encountered a fixed variable (%s) inside an active "
                        "objective or constraint expression on model %s, which is "
                        "usually indicative of a preprocessing error. Use the "
                        "IO-option 'output_fixed_variable_bounds=True' to suppress "
                        "this error and fix the variable by overwriting its bounds "
                        "in the LP file." % (vardata.name, model.name))
                if vardata.value is None:
                    raise ValueError("Variable cannot be fixed to a value of None.")
                vardata_lb = value(vardata.value)
                vardata_ub = value(vardata.value)
            else:
                vardata_lb = self._get_bound(vardata.lb)
                vardata_ub = self._get_bound(vardata.ub)

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
                                % (vardata.name))

            # in the CPLEX LP file format, the default variable
            # bounds are 0 and +inf.  These bounds are in
            # conflict with Pyomo, which assumes -inf and +inf
            # (which we would argue is more rational).
            output_file.write("   ")
            if (vardata_lb is not None) and (vardata_lb != -infinity):
                output_file.write(lb_string_template
                                  % (_no_negative_zero(vardata_lb)))
            else:
                output_file.write(" -inf <= ")
            if name_to_output == "e":
                raise ValueError(
                    "Attempting to write variable with name 'e' in a CPLEX LP "
                    "formatted file will cause a parse failure due to confusion with "
                    "numeric values expressed in scientific notation")

            output_file.write(name_to_output)
            if (vardata_ub is not None) and (vardata_ub != infinity):
                output_file.write(ub_string_template
                                  % (_no_negative_zero(vardata_ub)))
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
        output_file.write("end\n")

        # Clean up the symbol map to only contain variables referenced
        # in the active constraints **Note**: warm start method may
        # rely on this for choosing the set of potential warm start
        # variables
        vars_to_delete = set(variable_symbol_map.byObject.keys()) - \
                         set(self._referenced_variable_ids.keys())
        sm_byObject = symbol_map.byObject
        sm_bySymbol = symbol_map.bySymbol
        var_sm_byObject = variable_symbol_map.byObject
        for varid in vars_to_delete:
            symbol = var_sm_byObject[varid]
            del sm_byObject[varid]
            del sm_bySymbol[symbol]
        del variable_symbol_map

        return symbol_map
