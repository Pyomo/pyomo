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
# Problem Writer for GAMS Format Files
#

from six import StringIO, string_types, iteritems
from six.moves import xrange

from pyutilib.misc import PauseGC

from pyomo.core.base import (
    SymbolMap, AlphaNumericTextLabeler, NumericLabeler,
    Block, Constraint, Expression, Objective, Var, Set, RangeSet, Param,
    minimize, Suffix, SortComponents, Connector)

from pyomo.core.base.component import ComponentData
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter
import pyomo.util.plugin

from pyomo.core.kernel.component_block import IBlockStorage
from pyomo.core.kernel.component_interface import ICategorizedObject
from pyomo.core.kernel.numvalue import is_fixed, value, as_numeric

import logging

logger = logging.getLogger('pyomo.core')

def _get_bound(exp):
    if exp is None:
        return None
    if is_fixed(exp):
        return value(exp)
    raise ValueError("non-fixed bound or weight: " + str(exp))


# HACK: Temporary check for Connectors in active constriants.
# This should be removed after the writer is moved to an
# explicit GAMS-specific expression walker for generating the
# constraint strings.
import pyomo.core.base.expr_coopr3 as coopr3
from pyomo.core.kernel.numvalue import native_types
from pyomo.core.base.connector import _ConnectorData
def _check_for_connectors(con):
    _stack = [ ([con.body], 0, 1) ]
    while _stack:
        _argList, _idx, _len = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if type(_sub) in native_types:
                pass
            elif _sub.is_expression():
                _stack.append(( _argList, _idx, _len ))
                if type(_sub) is coopr3._ProductExpression:
                    if _sub._denominator:
                        _stack.append(
                            (_sub._denominator, 0, len(_sub._denominator)) )
                    _argList = _sub._numerator
                else:
                    _argList = _sub._args
                _idx = 0
                _len = len(_argList)
            elif isinstance(_sub, _ConnectorData):
                raise TypeError(
                    "Constraint '%s' body contains unexpanded connectors"
                    % (con.name,))


class ProblemWriter_gams(AbstractProblemWriter):
    pyomo.util.plugin.alias('gams', 'Generate the corresponding GAMS file')

    def __init__(self):
        AbstractProblemWriter.__init__(self, ProblemFormat.gams)

    def __call__(self,
                 model,
                 output_filename,
                 solver_capability,
                 io_options):
        """
        output_filename:
            Name of file to write GAMS model to. Optionally pass a file-like
            stream and the model will be written to that instead.
        io_options:
            warmstart=False:
                Warmstart by initializing model's variables to their values.
            symbolic_solver_labels=False:
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
            labeler=None:
                Custom labeler. Incompatible with symbolic_solver_labels.
            solver=None:
                If None, GAMS will use default solver for model type.
            mtype=None:
                Model type. If None, will chose from lp, nlp, mip, and minlp.
            add_options=None:
                List of additional lines to write directly
                into model file before the solve statement.
                For model attributes, <model name> is GAMS_MODEL.
            skip_trivial_constraints=False:
                Skip writing constraints whose body section is fixed
            file_determinism=1:
                How much effort do we want to put into ensuring the
                GAMS file is written deterministically for a Pyomo model:
                   0 : None
                   1 : sort keys of indexed components (default)
                   2 : sort keys AND sort names (over declaration order)
            put_results=None:
                Filename for optionally writing solution values and
                marginals to (put_results).dat, and solver statuses
                to (put_results + 'stat').dat.
        """

        # Make sure not to modify the user's dictionary,
        # they may be reusing it outside of this call
        io_options = dict(io_options)

        # Use full Pyomo component names rather than
        # shortened symbols (slower, but useful for debugging).
        symbolic_solver_labels = io_options.pop("symbolic_solver_labels", False)

        # Custom labeler option. Incompatible with symbolic_solver_labels.
        labeler = io_options.pop("labeler", None)

        # If None, GAMS will use default solver for model type.
        solver = io_options.pop("solver", None)

        # If None, will chose from lp, nlp, mip, and minlp.
        mtype = io_options.pop("mtype", None)

        # Lines to add before solve statement.
        add_options = io_options.pop("add_options", None)

        # Skip writing constraints whose body section is
        # fixed (i.e., no variables)
        skip_trivial_constraints = \
            io_options.pop("skip_trivial_constraints", False)

        # How much effort do we want to put into ensuring the
        # GAMS file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)
        sorter_map = {0:SortComponents.unsorted,
                      1:SortComponents.deterministic,
                      2:SortComponents.sortBoth}
        sort = sorter_map[file_determinism]

        # Warmstart by initializing model's variables to their values.
        warmstart = io_options.pop("warmstart", False)

        # Filename for optionally writing solution values and marginals
        # Set to True by GAMSSolver
        put_results = io_options.pop("put_results", None)

        if len(io_options):
            raise ValueError(
                "ProblemWriter_gams passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s"
                            % (k,v) for k,v in iteritems(io_options)))

        if solver is not None:
            if solver.upper() not in valid_solvers:
                raise ValueError("ProblemWriter_gams passed unrecognized "
                                 "solver: %s" % solver)

        if mtype is not None:
            valid_mtypes = set([
                'lp', 'qcp', 'nlp', 'dnlp', 'rmip', 'mip', 'rmiqcp', 'rminlp',
                'miqcp', 'minlp', 'rmpec', 'mpec', 'mcp', 'cns', 'emp'])
            if mtype.lower() not in valid_mtypes:
                raise ValueError("ProblemWriter_gams passed unrecognized "
                                 "model type: %s" % mtype)
            if (solver is not None and
                mtype.upper() not in valid_solvers[solver.upper()]):
                raise ValueError("ProblemWriter_gams passed solver (%s) "
                                 "unsuitable for given model type (%s)"
                                 % (solver, mtype))

        if output_filename is None:
            output_filename = model.name + ".gms"

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError("ProblemWriter_gams: Using both the "
                             "'symbolic_solver_labels' and 'labeler' "
                             "I/O options is forbidden")

        if symbolic_solver_labels:
            var_labeler = con_labeler = AlphaNumericTextLabeler()
        elif labeler is None:
            var_labeler = NumericLabeler('x')
            con_labeler = NumericLabeler('c')
        else:
            var_labeler = con_labeler = labeler

        var_list = []
        symbolMap = SymbolMap()

        def var_recorder(obj):
            ans = var_labeler(obj)
            var_list.append(ans)
            return ans

        def var_label(obj):
            if obj.is_fixed():
                return str(value(obj))
            return symbolMap.getSymbol(obj, var_recorder)

        # when sorting, there are a non-trivial number of
        # temporary objects created. these all yield
        # non-circular references, so disable GC - the
        # overhead is non-trivial, and because references
        # are non-circular, everything will be collected
        # immediately anyway.
        with PauseGC() as pgc:
            try:
                if isinstance(output_filename, string_types):
                    output_file = open(output_filename, "w")
                else:
                    # Support passing of stream such as a StringIO
                    # on which to write the model file
                    output_file = output_filename
                self._write_model(
                    model=model,
                    output_file=output_file,
                    solver_capability=solver_capability,
                    var_list=var_list,
                    var_label=var_label,
                    symbolMap=symbolMap,
                    con_labeler=con_labeler,
                    sort=sort,
                    skip_trivial_constraints=skip_trivial_constraints,
                    warmstart=warmstart,
                    solver=solver,
                    mtype=mtype,
                    add_options=add_options,
                    put_results=put_results
                )
            finally:
                if isinstance(output_filename, string_types):
                    output_file.close()

        return output_filename, symbolMap

    def _write_model(self,
                     model,
                     output_file,
                     solver_capability,
                     var_list,
                     var_label,
                     symbolMap,
                     con_labeler,
                     sort,
                     skip_trivial_constraints,
                     warmstart,
                     solver,
                     mtype,
                     add_options,
                     put_results):
        constraint_names = []
        ConstraintIO = StringIO()
        linear = True
        linear_degree = set([0,1])

        # Sanity check: all active components better be things we know
        # how to deal with, plus Suffix if solving
        valid_ctypes = set([
            Block, Constraint, Expression, Objective, Param,
            Set, RangeSet, Var, Suffix, Connector ])
        model_ctypes = model.collect_ctypes(active=True)
        if not model_ctypes.issubset(valid_ctypes):
            invalids = [t.__name__ for t in (model_ctypes - valid_ctypes)]
            raise RuntimeError(
                "Unallowable component(s) %s.\nThe GAMS writer cannot "
                "export models with this component type" %
                ", ".join(invalids))

        # HACK: Temporary check for Connectors in active constriants.
        # This should be removed after the writer is moved to an
        # explicit GAMS-specific expression walker for generating the
        # constraint strings.
        has_Connectors = Connector in model_ctypes

        # Walk through the model and generate the constraint definition
        # for all active constraints.  Any Vars / Expressions that are
        # encountered will be added to the var_list due to the labeler
        # defined above.
        for con in model.component_data_objects(Constraint,
                                                active=True,
                                                sort=sort):

            if (not con.has_lb()) and \
               (not con.has_ub()):
                assert not con.equality
                continue # non-binding, so skip

            # HACK: Temporary check for Connectors in active constriants.
            if has_Connectors:
                _check_for_connectors(con)

            con_body = as_numeric(con.body)
            if skip_trivial_constraints and con_body.is_fixed():
                continue
            if linear:
                if con_body.polynomial_degree() not in linear_degree:
                    linear = False

            body = StringIO()
            con_body.to_string(body, labeler=var_label)
            cName = symbolMap.getSymbol(con, con_labeler)
            if con.equality:
                constraint_names.append('%s' % cName)
                ConstraintIO.write('%s.. %s =e= %s ;\n' % (
                    constraint_names[-1],
                    body.getvalue(),
                    _get_bound(con.upper)
                ))
            else:
                if con.has_lb():
                    constraint_names.append('%s_lo' % cName)
                    ConstraintIO.write('%s.. %s =l= %s ;\n' % (
                        constraint_names[-1],
                        _get_bound(con.lower),
                        body.getvalue()
                    ))
                if con.has_ub():
                    constraint_names.append('%s_hi' % cName)
                    ConstraintIO.write('%s.. %s =l= %s ;\n' % (
                        constraint_names[-1],
                        body.getvalue(),
                        _get_bound(con.upper)
                    ))

        obj = list(model.component_data_objects(Objective,
                                                active=True,
                                                sort=sort))
        if len(obj) != 1:
            raise RuntimeError(
                "GAMS writer requires exactly one active objective (found %s)"
                % (len(obj)))
        obj = obj[0]
        if linear:
            if obj.expr.polynomial_degree() not in linear_degree:
                linear = False
        oName = symbolMap.getSymbol(obj, con_labeler)
        body = StringIO()
        obj.expr.to_string(body, labeler=var_label)
        constraint_names.append(oName)
        ConstraintIO.write('%s.. GAMS_OBJECTIVE =e= %s ;\n' % (
            oName,
            body.getvalue()
        ))

        # Categorize the variables that we found
        categorized_vars = Categorizer(var_list, symbolMap)

        # Write the GAMS model
        # $offdigit ignores extra precise digits instead of erroring
        output_file.write("$offdigit\n\n")
        output_file.write("EQUATIONS\n\t")
        output_file.write("\n\t".join(constraint_names))
        if categorized_vars.binary:
            output_file.write(";\n\nBINARY VARIABLES\n\t")
            output_file.write("\n\t".join(categorized_vars.binary))
        if categorized_vars.ints:
            output_file.write(";\n\nINTEGER VARIABLES")
            output_file.write("\n\t")
            output_file.write("\n\t".join(categorized_vars.ints))
        if categorized_vars.positive:
            output_file.write(";\n\nPOSITIVE VARIABLES\n\t")
            output_file.write("\n\t".join(categorized_vars.positive))
        output_file.write(";\n\nVARIABLES\n\tGAMS_OBJECTIVE\n\t")
        output_file.write("\n\t".join(categorized_vars.reals))
        output_file.write(";\n\n")

        for line in ConstraintIO.getvalue().splitlines():
            if '**' in line:
                # Investigate power functions for an integer exponent, in which
                # case replace with power(x, int) function to improve domain
                # issues.
                line = replace_power(line)
            if len(line) > 80000:
                line = split_long_line(line)
            output_file.write(line + "\n")

        output_file.write("\n")

        warn_int_bounds = False
        for category, var_name in categorized_vars:
            var = symbolMap.getObject(var_name)
            if category == 'positive':
                if var.has_ub():
                    output_file.write("%s.up = %s;\n" %
                                      (var_name, _get_bound(var.ub)))
            elif category == 'ints':
                if not var.has_lb():
                    warn_int_bounds = True
                    # GAMS doesn't allow -INF lower bound for ints
                    logger.warning("Lower bound for integer variable %s set "
                                   "to -1.0E+100." % var.name)
                    output_file.write("%s.lo = -1.0E+100;\n" % (var_name))
                elif value(var.lb) != 0:
                    output_file.write("%s.lo = %s;\n" %
                                      (var_name, _get_bound(var.lb)))
                if not var.has_ub():
                    warn_int_bounds = True
                    # GAMS has an option value called IntVarUp that is the
                    # default upper integer bound, which it applies if the
                    # integer's upper bound is INF. This option maxes out at
                    # 2147483647, so we can go higher by setting the bound.
                    logger.warning("Upper bound for integer variable %s set "
                                   "to +1.0E+100." % var.name)
                    output_file.write("%s.up = +1.0E+100;\n" % (var_name))
                else:
                    output_file.write("%s.up = %s;\n" %
                                      (var_name, _get_bound(var.ub)))
            elif category == 'binary':
                if var.has_lb() and value(var.lb) != 0:
                    output_file.write("%s.lo = %s;\n" %
                                      (var_name, _get_bound(var.lb)))
                if var.has_ub() and value(var.ub) != 1:
                    output_file.write("%s.up = %s;\n" %
                                      (var_name, _get_bound(var.ub)))
            elif category == 'reals':
                if var.has_lb():
                    output_file.write("%s.lo = %s;\n" %
                                      (var_name, _get_bound(var.lb)))
                if var.has_ub():
                    output_file.write("%s.up = %s;\n" %
                                      (var_name, _get_bound(var.ub)))
            else:
                raise KeyError('Category %s not supported' % category)
            if warmstart and var.value is not None:
                output_file.write("%s.l = %s;\n" % (var_name, var.value))
            if var.is_fixed():
                # This probably doesn't run, since all fixed vars are by default
                # replaced with their value and not assigned a symbol.
                # But leave this here in case we change handling of fixed vars
                assert var.value is not None, "Cannot fix variable at None"
                output_file.write("%s.fx = %s;\n" % (var_name, var.value))

        if warn_int_bounds:
            logger.warning(
                "GAMS requires finite bounds for integer variables. 1.0E100 "
                "is as extreme as GAMS will define, and should be enough to "
                "appear unbounded. If the solver cannot handle this bound, "
                "explicitly set a smaller bound on the pyomo model, or try a "
                "different GAMS solver.")

        model_name = "GAMS_MODEL"
        output_file.write("\nMODEL %s /all/ ;\n" % model_name)

        if mtype is None:
            mtype =  ('lp','nlp','mip','minlp')[
                (0 if linear else 1) +
                (2 if (categorized_vars.binary or categorized_vars.ints)
                 else 0)]

        if solver is not None:
            if mtype.upper() not in valid_solvers[solver.upper()]:
                raise ValueError("ProblemWriter_gams passed solver (%s) "
                                 "unsuitable for model type (%s)"
                                 % (solver, mtype))
            output_file.write("option %s=%s;\n" % (mtype, solver))

        if add_options is not None:
            output_file.write("\n* START USER ADDITIONAL OPTIONS\n")
            for line in add_options:
                output_file.write('\n' + line)
            output_file.write("\n\n* END USER ADDITIONAL OPTIONS\n\n")

        output_file.write(
            "SOLVE %s USING %s %simizing GAMS_OBJECTIVE;\n\n"
            % ( model_name,
                mtype,
                'min' if obj.sense == minimize else 'max'))

        # Set variables to store certain statuses and attributes
        stat_vars = ['MODELSTAT', 'SOLVESTAT', 'OBJEST', 'OBJVAL', 'NUMVAR',
                     'NUMEQU', 'NUMDVAR', 'NUMNZ', 'ETSOLVE']
        output_file.write("Scalars MODELSTAT 'model status', "
                          "SOLVESTAT 'solve status';\n")
        output_file.write("MODELSTAT = %s.modelstat;\n" % model_name)
        output_file.write("SOLVESTAT = %s.solvestat;\n\n" % model_name)

        output_file.write("Scalar OBJEST 'best objective', "
                          "OBJVAL 'objective value';\n")
        output_file.write("OBJEST = %s.objest;\n" % model_name)
        output_file.write("OBJVAL = %s.objval;\n\n" % model_name)

        output_file.write("Scalar NUMVAR 'number of variables';\n")
        output_file.write("NUMVAR = %s.numvar\n\n" % model_name)

        output_file.write("Scalar NUMEQU 'number of equations';\n")
        output_file.write("NUMEQU = %s.numequ\n\n" % model_name)

        output_file.write("Scalar NUMDVAR 'number of discrete variables';\n")
        output_file.write("NUMDVAR = %s.numdvar\n\n" % model_name)

        output_file.write("Scalar NUMNZ 'number of nonzeros';\n")
        output_file.write("NUMNZ = %s.numnz\n\n" % model_name)

        output_file.write("Scalar ETSOLVE 'time to execute solve statement';\n")
        output_file.write("ETSOLVE = %s.etsolve\n\n" % model_name)

        if put_results is not None:
            results = put_results + '.dat'
            output_file.write("\nfile results /'%s'/;" % results)
            output_file.write("\nresults.nd=15;")
            output_file.write("\nresults.nw=21;")
            output_file.write("\nput results;")
            output_file.write("\nput 'SYMBOL  :  LEVEL  :  MARGINAL' /;")
            for var in var_list:
                output_file.write("\nput %s %s.l %s.m /;" % (var, var, var))
            for con in constraint_names:
                output_file.write("\nput %s %s.l %s.m /;" % (con, con, con))
            output_file.write("\nput GAMS_OBJECTIVE GAMS_OBJECTIVE.l "
                              "GAMS_OBJECTIVE.m;\n")

            statresults = put_results + 'stat.dat'
            output_file.write("\nfile statresults /'%s'/;" % statresults)
            output_file.write("\nstatresults.nd=15;")
            output_file.write("\nstatresults.nw=21;")
            output_file.write("\nput statresults;")
            output_file.write("\nput 'SYMBOL   :   VALUE' /;")
            for stat in stat_vars:
                output_file.write("\nput '%s' %s /;\n" % (stat, stat))


class Categorizer(object):
    """Class for representing categorized variables.

    Given a list of variable names and a symbol map, categorizes the variable
    names into the categories: binary, ints, positive and reals.

    """

    def __init__(self, var_list, symbol_map):
        self.binary = []
        self.ints = []
        self.positive = []
        self.reals = []

        # categorize variables
        for var in var_list:
            v = symbol_map.getObject(var)
            if v.is_binary():
                self.binary.append(var)
            elif v.is_integer():
                if (v.has_lb() and (value(v.lb) >= 0)) and \
                   (v.has_ub() and (value(v.ub) <= 1)):
                    self.binary.append(var)
                else:
                    self.ints.append(var)
            elif value(v.lb) == 0:
                self.positive.append(var)
            else:
                self.reals.append(var)

    def __iter__(self):
        """Iterate over all variables.

        Yield a tuple containing the variables category and its name.
        """
        for category in ['binary', 'ints', 'positive', 'reals']:
            var_list = getattr(self, category)
            for var_name in var_list:
                yield category, var_name


def split_terms(line):
    """
    Take line from GAMS model file and return list of terms split by space
    but grouping together parentheses-bound expressions.
    """
    terms = []
    begin = 0
    inparens = 0
    for i in xrange(len(line)):
        if line[i] == '(':
            inparens += 1
        elif line[i] == ')':
            assert inparens > 0, "Unexpected close parenthesis ')'"
            inparens -= 1
        elif not inparens:
            if line[i] == ' ':
                if i > begin:
                    terms.append(line[begin:i])
                begin = i + 1
            elif (line[i] == '/' or
                  (line[i] in ('+', '-') and not (line[i-1] == 'e' and
                                                  line[i-2].isdigit())) or
                  (line[i] == '*' and line[i-1] != '*' and line[i+1] != '*')):
                # Keep power functions together
                if i > begin:
                    terms.append(line[begin:i])
                terms.append(line[i])
                begin = i + 1
    assert inparens == 0, "Missing close parenthesis in line '%s'" % line
    if begin < len(line):
        terms.append(line[begin:len(line)])
    return terms


def split_args(term):
    """
    Split a term by the ** operator but keep parenthesis-bound
    expressions grouped togeter.
    """
    args = []
    begin = 0
    inparens = 0
    for i in xrange(len(term)):
        if term[i] == '(':
            inparens += 1
        elif term[i] == ')':
            assert inparens > 0, "Unexpected close parenthesis ')'"
            inparens -= 1
        elif not inparens and term[i:i + 2] == '**':
            assert i > begin, "Invalid syntax around '**' operator"
            args.append(term[begin:i])
            begin = i + 2
    assert inparens == 0, "Missing close parenthesis in term '%s'" % term
    args.append(term[begin:len(term)])
    return args


def replace_power(line):
    new_line = ''
    for term in split_terms(line):
        if '**' in term:
            args = split_args(term)
            for i in xrange(len(args)):
                if '**' in args[i]:
                    first_paren = args[i].find('(')
                    assert ((first_paren != -1) and (args[i][-1] == ')')), (
                        "Assumed arg '%s' was a parenthesis-bound expression "
                        "or function" % args[i])
                    arg = args[i][first_paren + 1:-1]
                    args[i] = '%s( %s )' % (args[i][:first_paren],
                                            replace_power(arg))
            try:
                if float(args[-1]) == int(float(args[-1])):
                    term = ''
                    for arg in args[:-2]:
                        term += arg + '**'
                    term += 'power(%s, %s)' % (args[-2], args[-1])
            except ValueError:
                term = ''
                for arg in args[:-1]:
                    term += arg + '**'
                term += args[-1]
        new_line += term + ' '
    # Remove trailing space
    return new_line[:-1]


def split_long_line(line):
    """
    GAMS has an 80,000 character limit for lines, so split as many
    times as needed so as to not have illegal lines.
    """
    new_lines = ''
    while len(line) > 80000:
        i = 80000
        while line[i] != ' ':
            # Walk backwards to find closest space,
            # where it is safe to split to a new line
            i -= 1
        new_lines += line[:i] + '\n'
        line = line[i + 1:]
    new_lines += line
    return new_lines


valid_solvers = {
'ALPHAECP': ['MINLP','MIQCP'],
'AMPL': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'],
'ANTIGONE': ['NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'BARON': ['LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'BDMLP': ['LP','MIP','RMIP'],
'BDMLPD': ['LP','RMIP'],
'BENCH': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'BONMIN': ['MINLP','MIQCP'],
'BONMINH': ['MINLP','MIQCP'],
'CBC': ['LP','MIP','RMIP'],
'COINBONMIN': ['MINLP','MIQCP'],
'COINCBC': ['LP','MIP','RMIP'],
'COINCOUENNE': ['NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'COINIPOPT': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'COINOS': ['LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'COINSCIP': ['MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'CONOPT': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'CONOPT3': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'CONOPT4': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'CONOPTD': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'CONVERT': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'CONVERTD': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'],
'COUENNE': ['NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'CPLEX': ['LP','MIP','RMIP','QCP','MIQCP','RMIQCP'],
'CPLEXD': ['LP','MIP','RMIP','QCP','MIQCP','RMIQCP'],
'CPOPTIMIZER': ['MIP','MINLP','MIQCP'],
'DE': ['EMP'],
'DECIS': ['EMP'],
'DECISC': ['LP'],
'DECISM': ['LP'],
'DICOPT': ['MINLP','MIQCP'],
'DICOPTD': ['MINLP','MIQCP'],
'EXAMINER': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'EXAMINER2': ['LP','MIP','RMIP','NLP','MCP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'GAMSCHK': ['LP','MIP','RMIP','NLP','MCP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'GLOMIQO': ['QCP','MIQCP','RMIQCP'],
'GUROBI': ['LP','MIP','RMIP','QCP','MIQCP','RMIQCP'],
'GUSS': ['LP', 'MIP', 'NLP', 'MCP', 'CNS', 'DNLP', 'MINLP', 'QCP', 'MIQCP'],
'IPOPT': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'IPOPTH': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'JAMS': ['EMP'],
'KESTREL': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'],
'KNITRO': ['LP','RMIP','NLP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'LGO': ['LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'],
'LGOD': ['LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'],
'LINDO': ['LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'],
'LINDOGLOBAL': ['LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'LINGO': ['LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP'],
'LOCALSOLVER': ['MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'LOGMIP': ['EMP'],
'LS': ['LP','RMIP'],
'MILES': ['MCP'],
'MILESE': ['MCP'],
'MINOS': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'MINOS5': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'MINOS55': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'MOSEK': ['LP','MIP','RMIP','NLP','DNLP','RMINLP','QCP','MIQCP','RMIQCP'],
'MPECDUMP': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'],
'MPSGE': [],
'MSNLP': ['NLP','DNLP','RMINLP','QCP','RMIQCP'],
'NLPEC': ['MCP','MPEC','RMPEC'],
'OQNLP': ['NLP', 'DNLP', 'MINLP', 'QCP', 'MIQCP'],
'OS': ['LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'OSICPLEX': ['LP','MIP','RMIP'],
'OSIGUROBI': ['LP','MIP','RMIP'],
'OSIMOSEK': ['LP','MIP','RMIP'],
'OSISOPLEX': ['LP','RMIP'],
'OSIXPRESS': ['LP','MIP','RMIP'],
'PATH': ['MCP','CNS'],
'PATHC': ['MCP','CNS'],
'PATHNLP': ['LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'],
'PYOMO': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'],
'QUADMINOS': ['LP'],
'SBB': ['MINLP','MIQCP'],
'SCENSOLVER': ['LP','MIP','RMIP','NLP','MCP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'SCIP': ['MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'SNOPT': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'SOPLEX': ['LP','RMIP'],
'XA': ['LP','MIP','RMIP'],
'XPRESS': ['LP','MIP','RMIP','QCP','MIQCP','RMIQCP']
}
