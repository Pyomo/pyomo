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

from io import StringIO

from pyomo.common.gc_manager import PauseGC
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import (
    value, as_numeric, native_types, native_numeric_types,
    nonpyomo_leaf_types,
)
from pyomo.core.base import (
    SymbolMap, ShortNameLabeler, NumericLabeler, Constraint, 
    Objective, Var, minimize, SortComponents)
from pyomo.core.base.component import ActiveComponent
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.repn.util import valid_expr_ctypes_minlp, \
    valid_active_ctypes_minlp, ftoa

import logging

logger = logging.getLogger('pyomo.core')

_legal_unary_functions = {
    'ceil','floor','exp','log','log10','sqrt',
    'sin','cos','tan','asin','acos','atan','sinh','cosh','tanh',
}
_arc_functions = {'acos','asin','atan'}
_dnlp_functions = {'ceil','floor','abs'}
_zero_one = {0, 1}
#
# A visitor pattern that creates a string for an expression
# that is compatible with the GAMS syntax.
#
class ToGamsVisitor(EXPR.ExpressionValueVisitor):

    def __init__(self, smap, treechecker, output_fixed_variables=False):
        super(ToGamsVisitor, self).__init__()
        self.smap = smap
        self.treechecker = treechecker
        self.is_discontinuous = False
        self.output_fixed_variables = output_fixed_variables

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        tmp = []
        for i,val in enumerate(values):
            arg = node._args_[i]

            parens = False
            if val[0] in '-+':
                # Note: This is technically only necessary for i > 0
                parens = True
            elif arg.__class__ in native_types:
                pass
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
                tmp.append("(" + val + ")")
            else:
                tmp.append(val)

        if node.__class__ is EXPR.PowExpression:
            # If the exponent is a positive integer, use the power() function.
            # Otherwise, use the ** operator.
            exponent = node.arg(1)
            if (exponent.__class__ in native_numeric_types and
                    exponent == int(exponent)):
                return "power({0}, {1})".format(tmp[0], tmp[1])
            else:
                return "{0} ** {1}".format(tmp[0], tmp[1])
        elif node.__class__ is EXPR.UnaryFunctionExpression:
            if node.name not in _legal_unary_functions:
                raise RuntimeError(
                    "GAMS files cannot represent the unary function %s"
                    % ( node.name, ))
            if node.name in _dnlp_functions:
                self.is_discontinuous = True
            if node.name in _arc_functions:
                return "arc{0}({1})".format(node.name[1:], tmp[0])
            else:
                return node._to_string(tmp, None, self.smap, True)
        elif node.__class__ is EXPR.AbsExpression:
            self.is_discontinuous = True
            return node._to_string(tmp, None, self.smap, True)
        else:
            return node._to_string(tmp, None, self.smap, True)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_types:
            try:
                return True, ftoa(node)
            except TypeError:
                return True, repr(node)

        if node.is_expression_type():
            # Special handling if NPV and semi-NPV types:
            if not node.is_potentially_variable():
                return True, ftoa(value(node))
            if node.__class__ is EXPR.MonomialTermExpression:
                return True, self._monomial_to_string(node)
            if node.__class__ is EXPR.LinearExpression:
                return True, self._linear_to_string(node)
            # we will descend into this, so type checking will happen later
            if node.is_component_type():
                self.treechecker(node)
            return False, None

        if node.is_component_type():
            if node.ctype not in valid_expr_ctypes_minlp:
                # Make sure all components in active constraints
                # are basic ctypes we know how to deal with.
                raise RuntimeError(
                    "Unallowable component '%s' of type %s found in an active "
                    "constraint or objective.\nThe GAMS writer cannot export "
                    "expressions with this component type."
                    % (node.name, node.ctype.__name__))
            if node.ctype is not Var:
                # For these, make sure it's on the right model. We can check
                # Vars later since they don't disappear from the expressions
                self.treechecker(node)

        if node.is_fixed() and not (
                self.output_fixed_variables and node.is_potentially_variable()):
            return True, ftoa(value(node))
        else:
            assert node.is_variable_type()
            return True, self.smap.getSymbol(node)

    def _monomial_to_string(self, node):
        const, var = node.args
        const = value(const)
        if var.is_fixed() and not self.output_fixed_variables:
            return ftoa(const * var.value)
        # Special handling: ftoa is slow, so bypass _to_string when this
        # is a trivial term
        if const in {-1, 1}:
            if const < 0:
                return '-' + self.smap.getSymbol(var)
            else:
                return self.smap.getSymbol(var)
        return node._to_string((ftoa(const), self.smap.getSymbol(var)),
                               False, self.smap, True)

    def _linear_to_string(self, node):
        iter_ = iter(node.args)
        values = []
        if node.constant:
            next(iter_)
            values.append(ftoa(node.constant))
        values.extend(map(self._monomial_to_string, iter_))
        return node._to_string(values, False, self.smap, True)


def expression_to_string(expr, treechecker, labeler=None, smap=None,
                         output_fixed_variables=False):
    if labeler is not None:
        if smap is None:
            smap = SymbolMap()
        smap.default_labeler = labeler
    visitor = ToGamsVisitor(smap, treechecker, output_fixed_variables)
    expr_str = visitor.dfs_postorder_stack(expr)
    return expr_str, visitor.is_discontinuous


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
        self.fixed = []

        # categorize variables
        for var in var_list:
            v = symbol_map.getObject(var)
            if v.is_fixed():
                self.fixed.append(var)
            elif v.is_continuous():
                if v.lb == 0:
                    self.positive.append(var)
                else:
                    self.reals.append(var)
            elif v.is_integer():
                if all(bnd in _zero_one for bnd in v.bounds):
                    self.binary.append(var)
                else:
                    self.ints.append(var)
            else:
                raise RuntimeError(
                    "Cannot output variable to GAMS: effective variable "
                    "domain is not in {Reals, Integers, Binary}")

    def __iter__(self):
        """Iterate over all variables.

        Yield a tuple containing the variables category and its name.
        """
        for category in ['binary', 'ints', 'positive', 'reals']:
            var_list = getattr(self, category)
            for var_name in var_list:
                yield category, var_name


class StorageTreeChecker(object):
    def __init__(self, model):
        # blocks are hashable so we can use a normal set
        self.tree = {model}
        self.model = model
        # add everything above the model
        pb = self.parent_block(model)
        while pb is not None:
            self.tree.add(pb)
            pb = self.parent_block(pb)

    def __call__(self, comp, exception_flag=True):
        if comp is self.model:
            return True

        # walk up tree until there are no more parents
        seen = set()
        pb = self.parent_block(comp)
        while pb is not None:
            if pb in self.tree:
                self.tree.update(seen)
                return True
            seen.add(pb)
            pb = self.parent_block(pb)

        if exception_flag:
            self.raise_error(comp)
        else:
            return False

    def parent_block(self, comp):
        if isinstance(comp, ICategorizedObject):
            parent = comp.parent
            while (parent is not None) and \
                  (not parent._is_heterogeneous_container):
                parent = parent.parent
            return parent
        else:
            return comp.parent_block()

    def raise_error(self, comp):
        raise RuntimeError(
            "GAMS writer: found component '%s' not on same model tree.\n"
            "All components must have the same parent model." % comp.name)


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
            if i < 0:
                raise RuntimeError(
                    "Found an 80,000+ character string with no spaces")
            i -= 1
        new_lines += line[:i] + '\n'
        # the space will be the first character in the next line,
        # so that the line doesn't start with the comment character '*'
        line = line[i:]
    new_lines += line
    return new_lines


@WriterFactory.register('gams', 'Generate the corresponding GAMS file')
class ProblemWriter_gams(AbstractProblemWriter):

    def __init__(self):
        AbstractProblemWriter.__init__(self, ProblemFormat.gams)

    def __call__(self,
                 model,
                 output_filename,
                 solver_capability,
                 io_options):
        """
        Write a model in the GAMS modeling language format.

        Keyword Arguments
        -----------------
        output_filename: str
            Name of file to write GAMS model to. Optionally pass a file-like
            stream and the model will be written to that instead.
        io_options: dict
            - warmstart=True
                Warmstart by initializing model's variables to their values.
            - symbolic_solver_labels=False
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
            - labeler=None
                Custom labeler. Incompatible with symbolic_solver_labels.
            - solver=None
                If None, GAMS will use default solver for model type.
            - mtype=None
                Model type. If None, will chose from lp, nlp, mip, and minlp.
            - add_options=None
                List of additional lines to write directly
                into model file before the solve statement.
                For model attributes, <model name> is GAMS_MODEL.
            - skip_trivial_constraints=False
                Skip writing constraints whose body section is fixed.
            - output_fixed_variables=False
                If True, output fixed variables as variables; otherwise,
                output numeric value.
            - file_determinism=1
                | How much effort do we want to put into ensuring the
                | GAMS file is written deterministically for a Pyomo model:
                |     0 : None
                |     1 : sort keys of indexed components (default)
                |     2 : sort keys AND sort names (over declaration order)
            - put_results=None
                Filename for optionally writing solution values and
                marginals.  If put_results_format is 'gdx', then GAMS
                will write solution values and marginals to
                GAMS_MODEL_p.gdx and solver statuses to
                {put_results}_s.gdx.  If put_results_format is 'dat',
                then solution values and marginals are written to
                (put_results).dat, and solver statuses to (put_results +
                'stat').dat.
            - put_results_format='gdx'
                Format used for put_results, one of 'gdx', 'dat'.

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

        # Improved GAMS calling options
        solprint = io_options.pop("solprint", "off")
        limrow = io_options.pop("limrow", 0)
        limcol = io_options.pop("limcol", 0)
        solvelink = io_options.pop("solvelink", 5)

        # Lines to add before solve statement.
        add_options = io_options.pop("add_options", None)

        # Skip writing constraints whose body section is
        # fixed (i.e., no variables)
        skip_trivial_constraints = \
            io_options.pop("skip_trivial_constraints", False)

        # Output fixed variables as variables
        output_fixed_variables = \
            io_options.pop("output_fixed_variables", False)

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
        warmstart = io_options.pop("warmstart", True)

        # Filename for optionally writing solution values and marginals
        # Set to True by GAMSSolver
        put_results = io_options.pop("put_results", None)
        put_results_format = io_options.pop("put_results_format", 'gdx')
        assert put_results_format in ('gdx','dat')

        if len(io_options):
            raise ValueError(
                "GAMS writer passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s"
                            % (k,v) for k,v in io_options.items()))

        if solver is not None and solver.upper() not in valid_solvers:
            raise ValueError(
                "GAMS writer passed unrecognized solver: %s" % solver)

        if mtype is not None:
            valid_mtypes = set([
                'lp', 'qcp', 'nlp', 'dnlp', 'rmip', 'mip', 'rmiqcp', 'rminlp',
                'miqcp', 'minlp', 'rmpec', 'mpec', 'mcp', 'cns', 'emp'])
            if mtype.lower() not in valid_mtypes:
                raise ValueError("GAMS writer passed unrecognized "
                                 "model type: %s" % mtype)
            if (solver is not None and
                mtype.upper() not in valid_solvers[solver.upper()]):
                raise ValueError("GAMS writer passed solver (%s) "
                                 "unsuitable for given model type (%s)"
                                 % (solver, mtype))

        if output_filename is None:
            output_filename = model.name + ".gms"

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError("GAMS writer: Using both the "
                             "'symbolic_solver_labels' and 'labeler' "
                             "I/O options is forbidden")

        if symbolic_solver_labels:
            # Note that the Var and Constraint labelers must use the
            # same labeler, so that we can correctly detect name
            # collisions (which can arise when we truncate the labels to
            # the max allowable length.  GAMS requires all identifiers
            # to start with a letter.  We will (randomly) choose "s_"
            # (for 'shortened')
            var_labeler = con_labeler = ShortNameLabeler(
                60, prefix='s_', suffix='_', caseInsensitive=True,
                legalRegex='^[a-zA-Z]')
        elif labeler is None:
            var_labeler = NumericLabeler('x')
            con_labeler = NumericLabeler('c')
        else:
            var_labeler = con_labeler = labeler

        var_list = []

        def var_recorder(obj):
            ans = var_labeler(obj)
            try:
                if obj.is_variable_type():
                    var_list.append(ans)
            except:
                pass
            return ans

        def var_label(obj):
            #if obj.is_fixed():
            #    return str(value(obj))
            return symbolMap.getSymbol(obj, var_recorder)

        symbolMap = SymbolMap(var_label)

        # when sorting, there are a non-trivial number of
        # temporary objects created. these all yield
        # non-circular references, so disable GC - the
        # overhead is non-trivial, and because references
        # are non-circular, everything will be collected
        # immediately anyway.
        with PauseGC() as pgc:
            try:
                if isinstance(output_filename, str):
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
                    output_fixed_variables=output_fixed_variables,
                    warmstart=warmstart,
                    solver=solver,
                    mtype=mtype,
                    solprint=solprint,
                    limrow=limrow,
                    limcol=limcol,
                    solvelink=solvelink,
                    add_options=add_options,
                    put_results=put_results,
                    put_results_format=put_results_format,
                )
            finally:
                if isinstance(output_filename, str):
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
                     output_fixed_variables,
                     warmstart,
                     solver,
                     mtype,
                     solprint,
                     limrow,
                     limcol,
                     solvelink,
                     add_options,
                     put_results,
                     put_results_format,
                 ):
        constraint_names = []
        ConstraintIO = StringIO()
        linear = True
        linear_degree = set([0,1])
        dnlp = False

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
                "Unallowable active component(s) %s.\nThe GAMS writer cannot "
                "export models with this component type." %
                ", ".join(invalids))

        tc = StorageTreeChecker(model)

        # Walk through the model and generate the constraint definition
        # for all active constraints.  Any Vars / Expressions that are
        # encountered will be added to the var_list due to the labeler
        # defined above.
        for con in model.component_data_objects(Constraint,
                                                active=True,
                                                sort=sort):

            if not con.has_lb() and not con.has_ub():
                assert not con.equality
                continue # non-binding, so skip

            con_body = as_numeric(con.body)
            if skip_trivial_constraints and con_body.is_fixed():
                continue
            if linear:
                if con_body.polynomial_degree() not in linear_degree:
                    linear = False

            cName = symbolMap.getSymbol(con, con_labeler)
            con_body_str, con_discontinuous = expression_to_string(
                con_body, tc, smap=symbolMap,
                output_fixed_variables=output_fixed_variables
            )
            dnlp |= con_discontinuous
            if con.equality:
                constraint_names.append('%s' % cName)
                ConstraintIO.write('%s.. %s =e= %s ;\n' % (
                    constraint_names[-1],
                    con_body_str,
                    ftoa(con.upper)
                ))
            else:
                if con.has_lb():
                    constraint_names.append('%s_lo' % cName)
                    ConstraintIO.write('%s.. %s =l= %s ;\n' % (
                        constraint_names[-1],
                        ftoa(con.lower),
                        con_body_str,
                    ))
                if con.has_ub():
                    constraint_names.append('%s_hi' % cName)
                    ConstraintIO.write('%s.. %s =l= %s ;\n' % (
                        constraint_names[-1],
                        con_body_str,
                        ftoa(con.upper)
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
        obj_expr_str, obj_discontinuous = expression_to_string(
            obj.expr, tc, smap=symbolMap,
            output_fixed_variables=output_fixed_variables,
        )
        dnlp |= obj_discontinuous
        oName = symbolMap.getSymbol(obj, con_labeler)
        constraint_names.append(oName)
        ConstraintIO.write('%s.. GAMS_OBJECTIVE =e= %s ;\n' % (
            oName,
            obj_expr_str,
        ))

        # Categorize the variables that we found
        categorized_vars = Categorizer(var_list, symbolMap)

        # Write the GAMS model
        output_file.write("$offlisting\n")
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
        output_file.write("\n\t".join(
            categorized_vars.reals + categorized_vars.fixed
        ))
        output_file.write(";\n\n")

        for var in categorized_vars.fixed:
            output_file.write("%s.fx = %s;\n" % (
                var, ftoa(value(symbolMap.getObject(var)))
            ))
        output_file.write("\n")

        for line in ConstraintIO.getvalue().splitlines():
            if len(line) > 80000:
                line = split_long_line(line)
            output_file.write(line + "\n")

        output_file.write("\n")

        warn_int_bounds = False
        for category, var_name in categorized_vars:
            var = symbolMap.getObject(var_name)
            tc(var)
            lb, ub = var.bounds
            if category == 'positive':
                if ub is not None:
                    output_file.write("%s.up = %s;\n" %
                                      (var_name, ftoa(ub)))
            elif category == 'ints':
                if lb is None:
                    warn_int_bounds = True
                    # GAMS doesn't allow -INF lower bound for ints
                    logger.warning("Lower bound for integer variable %s set "
                                   "to -1.0E+100." % var.name)
                    output_file.write("%s.lo = -1.0E+100;\n" % (var_name))
                elif lb != 0:
                    output_file.write("%s.lo = %s;\n" % (var_name, ftoa(lb)))
                if ub is None:
                    warn_int_bounds = True
                    # GAMS has an option value called IntVarUp that is the
                    # default upper integer bound, which it applies if the
                    # integer's upper bound is INF. This option maxes out at
                    # 2147483647, so we can go higher by setting the bound.
                    logger.warning("Upper bound for integer variable %s set "
                                   "to +1.0E+100." % var.name)
                    output_file.write("%s.up = +1.0E+100;\n" % (var_name))
                else:
                    output_file.write("%s.up = %s;\n" % (var_name, ftoa(ub)))
            elif category == 'binary':
                if lb != 0:
                    output_file.write("%s.lo = %s;\n" % (var_name, ftoa(lb)))
                if ub != 1:
                    output_file.write("%s.up = %s;\n" % (var_name, ftoa(ub)))
            elif category == 'reals':
                if lb is not None:
                    output_file.write("%s.lo = %s;\n" % (var_name, ftoa(lb)))
                if ub is not None:
                    output_file.write("%s.up = %s;\n" % (var_name, ftoa(ub)))
            else:
                raise KeyError('Category %s not supported' % category)
            if warmstart and var.value is not None:
                output_file.write("%s.l = %s;\n" %
                                  (var_name, ftoa(var.value)))

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
            if mtype == 'nlp' and dnlp:
                mtype = 'dnlp'

        if solver is not None:
            if mtype.upper() not in valid_solvers[solver.upper()]:
                raise ValueError("GAMS writer passed solver (%s) "
                                 "unsuitable for model type (%s)"
                                 % (solver, mtype))
            output_file.write("option %s=%s;\n" % (mtype, solver))

        output_file.write("option solprint=%s;\n" % solprint)
        output_file.write("option limrow=%d;\n" % limrow)
        output_file.write("option limcol=%d;\n" % limcol)
        output_file.write("option solvelink=%d;\n" % solvelink)
        
        if put_results is not None and put_results_format == 'gdx':
            output_file.write("option savepoint=1;\n")

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
            if put_results_format == 'gdx':
                output_file.write("\nexecute_unload '%s_s.gdx'" % put_results)
                for stat in stat_vars:
                    output_file.write(", %s" % stat)
                output_file.write(";\n")
            else:
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

valid_solvers = {
'ALPHAECP': {'MINLP','MIQCP'},
'AMPL': {'LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'},
'ANTIGONE': {'NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'BARON': {'LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'BDMLP': {'LP','MIP','RMIP'},
'BDMLPD': {'LP','RMIP'},
'BENCH': {'LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'BONMIN': {'MINLP','MIQCP'},
'BONMINH': {'MINLP','MIQCP'},
'CBC': {'LP','MIP','RMIP'},
'COINBONMIN': {'MINLP','MIQCP'},
'COINCBC': {'LP','MIP','RMIP'},
'COINCOUENNE': {'NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'COINIPOPT': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'COINOS': {'LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'COINSCIP': {'MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'CONOPT': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'CONOPT3': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'CONOPT4': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'CONOPTD': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'CONVERT': {'LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'CONVERTD': {'LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'},
'COUENNE': {'NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'CPLEX': {'LP','MIP','RMIP','QCP','MIQCP','RMIQCP'},
'CPLEXD': {'LP','MIP','RMIP','QCP','MIQCP','RMIQCP'},
'CPOPTIMIZER': {'MIP','MINLP','MIQCP'},
'DE': {'EMP'},
'DECIS': {'EMP'},
'DECISC': {'LP'},
'DECISM': {'LP'},
'DICOPT': {'MINLP','MIQCP'},
'DICOPTD': {'MINLP','MIQCP'},
'EXAMINER': {'LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'EXAMINER2': {'LP','MIP','RMIP','NLP','MCP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'GAMSCHK': {'LP','MIP','RMIP','NLP','MCP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'GLOMIQO': {'QCP','MIQCP','RMIQCP'},
'GUROBI': {'LP','MIP','RMIP','QCP','MIQCP','RMIQCP'},
'GUSS': {'LP', 'MIP', 'NLP', 'MCP', 'CNS', 'DNLP', 'MINLP', 'QCP', 'MIQCP'},
'IPOPT': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'IPOPTH': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'JAMS': {'EMP'},
'KESTREL': {'LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'},
'KNITRO': {'LP','RMIP','NLP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'LGO': {'LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'},
'LGOD': {'LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'},
'LINDO': {'LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'},
'LINDOGLOBAL': {'LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'LINGO': {'LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP'},
'LOCALSOLVER': {'MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'LOGMIP': {'EMP'},
'LS': {'LP','RMIP'},
'MILES': {'MCP'},
'MILESE': {'MCP'},
'MINOS': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'MINOS5': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'MINOS55': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'MOSEK': {'LP','MIP','RMIP','NLP','DNLP','RMINLP','QCP','MIQCP','RMIQCP'},
'MPECDUMP': {'LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'},
'MPSGE': {},
'MSNLP': {'NLP','DNLP','RMINLP','QCP','RMIQCP'},
'NLPEC': {'MCP','MPEC','RMPEC'},
'OQNLP': {'NLP', 'DNLP', 'MINLP', 'QCP', 'MIQCP'},
'OS': {'LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'OSICPLEX': {'LP','MIP','RMIP'},
'OSIGUROBI': {'LP','MIP','RMIP'},
'OSIMOSEK': {'LP','MIP','RMIP'},
'OSISOPLEX': {'LP','RMIP'},
'OSIXPRESS': {'LP','MIP','RMIP'},
'PATH': {'MCP','CNS'},
'PATHC': {'MCP','CNS'},
'PATHNLP': {'LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'},
'PYOMO': {'LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'},
'QUADMINOS': {'LP'},
'SBB': {'MINLP','MIQCP'},
'SCENSOLVER': {'LP','MIP','RMIP','NLP','MCP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'SCIP': {'MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'},
'SHOT': {'MINLP','MIQCP'},
'SNOPT': {'LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'},
'SOPLEX': {'LP','RMIP'},
'XA': {'LP','MIP','RMIP'},
'XPRESS': {'LP','MIP','RMIP','QCP','MIQCP','RMIQCP'}
}
