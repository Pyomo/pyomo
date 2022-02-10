#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
import sys
from collections import deque
from operator import itemgetter, attrgetter

from pyomo.common.backports import nullcontext
from pyomo.common.config import ConfigBlock, ConfigValue, InEnum
from pyomo.common.errors import DeveloperError
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer

from pyomo.core.expr.current import (
    NegationExpression, ProductExpression, DivisionExpression,
    PowExpression, AbsExpression, UnaryFunctionExpression,
    MonomialTermExpression, LinearExpression, SumExpression,
    EqualityExpression, InequalityExpression, RangedExpression,
    Expr_ifExpression, ExternalFunctionExpression,
    native_types, value,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.base import (
    Block, Objective, Constraint, Var, Param, Expression, ExternalFunction,
    Suffix, SymbolMap, NameLabeler, SortComponents, minimize,
)
from pyomo.core.base.block import SortComponents
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
import pyomo.core.kernel as kernel
from pyomo.opt import WriterFactory

from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env

if sys.version_info[:2] >= (3,7):
    _deterministic_dict = dict
else:
    from pyomo.common.collections import OrderedDict
    _deterministic_dict = OrderedDict

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
###


class _CONSTANT(object): pass
class _MONOMIAL(object): pass
class _GENERAL(object): pass

class FileDeterminism(enum.IntEnum):
    NONE = 0
    SORT_INDICES = 1
    SORT_SYMBOLS = 2

def _activate_nl_writer_version(n):
    """DEBUGGING TOOL to switch the "default" NL writer"""
    doc = WriterFactory.doc('nl')
    WriterFactory.unregister('nl')
    WriterFactory.register('nl', doc)(WriterFactory.get_class(f'nl_v{n}'))

def identify_unrecognized_components(model, active=True, valid=set()):
    assert active in (True, None)
    unrecognized = {}
    for block in model.block_data_objects(active=active,
                                          descend_into=True,
                                          sort=SortComponents.unsorted):
        local_ctypes = block.collect_ctypes(active=None, descend_into=False)
        for ctype in local_ctypes - valid:
            # TODO: we should rethink the definition of "active" for
            # Components that are not subclasses of ActiveComponent
            if not issubclass(ctype, ActiveComponent):
                continue
            unrecognized.setdefault(ctype, []).extend(
                block.component_data_objects(
                    ctype=ctype,
                    active=active,
                    descend_into=False,
                    sort=SortComponents.unsorted))
    return {k:v for k,v in unrecognized.items() if v}

@WriterFactory.register(
    'nl_v2', 'Generate the corresponding AMPL NL file (version 2).')
class NLWriter(object):
    CONFIG = ConfigBlock('nlwriter')
    CONFIG.declare('show_section_timing', ConfigValue(
        default=False,
        domain=bool,
        description='Print timing after writing each section of the NL file',
    ))
    CONFIG.declare('skip_trivial_constraints', ConfigValue(
        default=False,
        domain=bool,
        description='Skip writing constraints whose body is constant'
    ))
    CONFIG.declare('file_determinism', ConfigValue(
        default=FileDeterminism.NONE,
        domain=InEnum(FileDeterminism),
        description='How much effort to ensure file is deterministic',
        doc="""
        How much effort do we want to put into ensuring the
        NL file is written deterministically for a Pyomo model:
            NONE (0) : None
            SORT_INDICES (1) : sort keys of indexed components (default)
            SORT_SYMBOLS (2) : sort keys AND sort names (over declaration order)
        """
    ))
    CONFIG.declare('symbolic_solver_labels', ConfigValue(
        default=False,
        domain=bool,
        description='Write the corresponding .row and .col files',
    ))


    def __init__(self):
        self.config = self.CONFIG()

    def __call__(self, model, filename, solver_capability, io_options):
        if filename is None:
            filename = model.name + ".nl"
        elif not filename.lower().endswith('.nl'):
            filename += '.nl'
        row_fname = filename[:-2] + 'row'
        col_fname = filename[:-2] + 'col'

        config = self.config(io_options)
        if config.symbolic_solver_labels:
            _open = lambda fname: open(fname, 'w')
        else:
            _open = nullcontext
        with open(filename, 'w') as FILE, \
             _open(row_fname) as ROWFILE, \
             _open(col_fname) as COLFILE:
            symbol_map, amplfuncs = self.write(
                model, FILE, ROWFILE, COLFILE, config=config)
        # Historically, the NL writer communicated the external function
        # libraries back to the ASL interface through the PYOMO_AMPLFUNC
        # environment variable.
        set_pyomo_amplfunc_env(amplfuncs)
        # The ProblemWriter callable interface returns the filename that
        # was generated and the symbol_map
        return filename, symbol_map

    def write(self, model, ostream, rowstream=None, colstream=None, **options):
        config = options.pop('config', self.config)(options)

        unknown = identify_unrecognized_components(model, active=True, valid={
            Block, Objective, Constraint, Var, Param, Expression,
            ExternalFunction,
            # FIXME: Non-active components should not report as Active
            Set, RangeSet, Port,
            # TODO: Suffix, Piecewise, SOSConstraint, Complementarity
            Suffix,
        })
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the NL writer does not know how to process:\n\t%s" %
                (model.name, "\n\t".join("%s:\n\t\t%s" % (
                    k, "\n\t\t".join(map(attrgetter('name'), v)))
                    for k, v in unknown.items())))

        _impl = _NLWriter_impl(ostream, rowstream, colstream, config)
        # Pause the GC, as the walker that generates the compiled NL
        # representation generates (and disposes of) a large number of
        # small objects.
        with PauseGC():
            return _impl.write(model)

def _RANGE_TYPE(lb, ub):
    if lb == ub:
        if lb is None:
            return 3 # -inf <= c <= inf
        else:
            return 4 # L == c == U
    elif lb is None:
        return 1 # c <= U
    elif ub is None:
        return 2 # L <= c
    else:
        return 0 # L <= c <= U

class _NLWriter_impl(object):

    def __init__(self, ostream, rowstream, colstream, config):
        self.ostream = ostream
        self.rowstream = rowstream
        self.colstream = colstream
        self.config = config
        self.symbolic_solver_labels = config.symbolic_solver_labels
        if self.symbolic_solver_labels:
            self.template = text_nl_debug_template
        else:
            self.template = text_nl_template
        self.subexpression_cache = {}
        self.subexpression_order = []
        self.external_functions = {}
        self.used_named_expressions = set()
        self.var_map = _deterministic_dict()
        self.visitor = AMPLRepnVisitor(
            self.template,
            self.subexpression_cache,
            self.subexpression_order,
            self.external_functions,
            self.var_map,
            self.used_named_expressions,
        )
        self.next_V_line_id = 0

    def write(self, model):
        try:
            assert AMPLRepn.ActiveVisitor is None
            AMPLRepn.ActiveVisitor = self.visitor
            return self._write_impl(model)
        finally:
            assert AMPLRepn.ActiveVisitor is self.visitor
            AMPLRepn.ActiveVisitor = None

    def _write_impl(self, model):
        timer = TicTocTimer()

        sorter = SortComponents.unsorted
        if self.config.file_determinism >= FileDeterminism.SORT_INDICES:
            sorter = sorter | SortComponents.indices
            if self.config.file_determinism  >= FileDeterminism.SORT_SYMBOLS:
                sorter = sorter | SortComponents.alphabetical

        # Caching some frequently-used objects into the locals()
        symbolic_solver_labels = self.symbolic_solver_labels
        visitor = self.visitor
        ostream = self.ostream
        var_map = self.var_map

        if self.config.file_determinism > FileDeterminism.NONE:
            # We will pre-gather the variables so that their order
            # matches the file_determinism flag.
            #
            # This is a little cumbersome, but is implemented this way
            # for consistency with the original NL writer.  Note that
            # Vars that appear twice (e.g., through a Reference) will be
            # sorted with the LAST occurance.
            for var in model.component_data_objects(
                    Var, descend_into=True, sort=sorter):
                var_map[id(var)] = var

        #
        # Tabulate the model expressions
        #
        objectives = []
        linear_objs = []
        for obj_comp in model.component_objects(
                Objective, active=True, descend_into=True, sort=sorter):
            try:
                obj_vals = obj_comp.values()
            except AttributeError:
                # kernel does not define values() for scalar objectives
                obj_vals = (obj_comp,)
            for obj in obj_vals:
                if not obj.active:
                    continue
                expr = visitor.walk_expression((obj.expr, obj, 1))
                if expr.nonlinear:
                    objectives.append((obj, expr))
                else:
                    linear_objs.append((obj, expr))
            timer.toc(f'Objective {obj_comp.name}')
        # Order the objectives, moving all nonlinear objectives to
        # the beginning
        n_nonlinear_objs = len(objectives)
        objectives.extend(linear_objs)
        n_objs = len(objectives)

        constraints = []
        linear_cons = []
        n_ranges = 0
        n_equality = 0
        for con_comp in model.component_objects(
                Constraint, active=True, descend_into=True, sort=sorter):
            try:
                con_vals = con_comp.values()
            except AttributeError:
                # kernel does not define values() for scalar constraints
                con_vals = (con_comp,)
            for con in con_vals:
                if not con.active:
                    continue
                expr = visitor.walk_expression((con.body, con, 0))
                lb = con.lb
                if lb is not None:
                    lb = repr(lb - expr.const)
                ub = con.ub
                if ub is not None:
                    ub = repr(ub - expr.const)
                _type = _RANGE_TYPE(lb, ub)
                if _type == 4:
                    n_equality += 1
                elif _type == 0:
                    n_ranges += 1
                if expr.nonlinear:
                    constraints.append((con, expr, _type, lb, ub))
                else:
                    linear_cons.append((con, expr, _type, lb, ub))
            timer.toc(f'Constraint {con_comp.name}')
        # Order the constraints, moving all nonlinear constraints to
        # the beginning
        n_nonlinear_cons = len(constraints)
        constraints.extend(linear_cons)
        n_cons = len(constraints)

        #
        # Collect constraints and objectives into the groupings
        # necessary for AMPL
        #
        # For efficiency, we will do everything with ids (and not the
        # var objects themselves)
        #

        # linear contribution by (constraint, objective) component.
        # Keys are component id(), Values are dicts mapping variable
        # id() to linear coefficient.  All nonzeros in the component
        # (variables appearing in the linear and/or nonlinear
        # subexpressions) will appear in the dict.
        linear_by_comp = {}

        # We need to categorize the named subexpressions first so that
        # we know their linear / nonlinear vars when we encounter them
        # in constraints / objectives
        self._categorize_vars(
            map(self.subexpression_cache.__getitem__,
                filter(self.used_named_expressions.__contains__,
                       self.subexpression_order)),
            linear_by_comp
        )
        n_subexpressions = self._count_subexpression_occurances()
        timer.toc('subexpressions')

        obj_vars_linear, obj_vars_nonlinear, obj_nnz_by_var \
            = self._categorize_vars(objectives, linear_by_comp)
        timer.toc('objectives')

        con_vars_linear, con_vars_nonlinear, con_nnz_by_var \
            = self._categorize_vars(constraints, linear_by_comp)
        timer.toc('constraints')

        n_lcons = 0 # We do not yet support logical constraints

        # obj_vars = obj_vars_linear.union(obj_vars_nonlinear)
        # con_vars = con_vars_linear.union(con_vars_nonlinear)
        # all_vars = con_vars.union(obj_vars)
        obj_vars = obj_vars_linear | obj_vars_nonlinear
        con_vars = con_vars_linear | con_vars_nonlinear
        all_vars = con_vars | obj_vars
        n_vars = len(all_vars)

        continuous_vars = set()
        binary_vars = set()
        integer_vars = set()
        for _id in all_vars:
            v = var_map[_id]
            if v.is_continuous():
                continuous_vars.add(_id)
            elif v.is_binary():
                binary_vars.add(_id)
            elif v.is_integer():
                integer_vars.add(_id)
            else:
                raise ValueError(
                    f"Variable '{v.name}' has a domain that is not Real, "
                    f"Integer, or Binary: Cannot write a legal NL file.")
        discrete_vars = binary_vars | integer_vars

        nonlinear_vars = con_vars_nonlinear | obj_vars_nonlinear
        linear_only_vars = (con_vars_linear | obj_vars_linear) - nonlinear_vars

        self.column_order = column_order = {
            _id: i for i, _id in enumerate(var_map)
        }
        variables = []
        #
        both_vars_nonlinear = con_vars_nonlinear & obj_vars_nonlinear
        if both_vars_nonlinear:
            variables.extend(sorted(
                both_vars_nonlinear & continuous_vars,
                key=column_order.__getitem__))
            variables.extend(sorted(
                both_vars_nonlinear & discrete_vars,
                key=column_order.__getitem__))
        #
        con_only_nonlinear_vars = con_vars_nonlinear - both_vars_nonlinear
        if con_only_nonlinear_vars:
            variables.extend(sorted(
                con_only_nonlinear_vars & continuous_vars,
                key=column_order.__getitem__))
            variables.extend(sorted(
                con_only_nonlinear_vars & discrete_vars,
                key=column_order.__getitem__))
        #
        obj_only_nonlinear_vars = obj_vars_nonlinear - both_vars_nonlinear
        if obj_vars_nonlinear:
            variables.extend(sorted(
                obj_only_nonlinear_vars & continuous_vars,
                key=column_order.__getitem__))
            variables.extend(sorted(
                obj_only_nonlinear_vars & discrete_vars,
                key=column_order.__getitem__))
        #
        if linear_only_vars:
            variables.extend(sorted(
                linear_only_vars - discrete_vars,
                key=column_order.__getitem__))
            variables.extend(sorted(
                linear_only_vars & binary_vars,
                key=column_order.__getitem__))
            variables.extend(sorted(
                linear_only_vars & integer_vars,
                key=column_order.__getitem__))
        assert len(variables) == n_vars
        timer.toc(f'{len(variables)} variables, {len(constraints)} constraints'
                  f' [{n_cons-n_nonlinear_cons} L, {n_nonlinear_cons} NL]')
        # Fill in the variable list and update the new column order
        for idx, _id in enumerate(variables):
            v = var_map[_id]
            column_order[_id] = idx
            lb, ub = v.bounds
            if lb is not None:
                lb = repr(lb)
            if ub is not None:
                ub = repr(ub)
            variables[idx] = (v, _id, _RANGE_TYPE(lb, ub), lb, ub)
        timer.toc("var bounds")

        # Now that the row/column ordering is resolved, create the labels
        symbol_map = SymbolMap()
        symbol_map.addSymbols(
            (info[0], f"v{idx}") for idx, info in enumerate(variables)
        )
        symbol_map.addSymbols(
            (info[0], f"c{idx}") for idx, info in enumerate(constraints)
        )
        symbol_map.addSymbols(
            (info[0], f"o{idx}") for idx, info in enumerate(objectives)
        )
        timer.toc("symbols")

        if symbolic_solver_labels:
            labeler = NameLabeler()
            row_labels = [labeler(info[0]) for info in constraints] \
                         + [labeler(info[0]) for info in objectives]
            row_comments = [f'\t#{lbl}' for lbl in row_labels]
            col_labels = [labeler(info[0]) for info in variables]
            col_comments = [f'\t#{lbl}' for lbl in col_labels]
            self.var_id_to_nl = {
                info[1]: f'{var_idx}{col_comments[var_idx]}'
                for var_idx, info in enumerate(variables)
            }
            # Write out the .row and .col data
            if self.rowstream is not None:
                self.rowstream.write('\n'.join(row_labels))
                self.rowstream.write('\n')
            if self.colstream is not None:
                self.colstream.write('\n'.join(col_labels))
                self.colstream.write('\n')
        else:
            row_labels = row_comments = [''] * (n_cons + n_objs)
            col_labels = col_comments = [''] * len(variables)
            self.var_id_to_nl = {
                info[1]: var_idx for var_idx, info in enumerate(variables)
            }

        timer.toc("row/col labels & comments")
        #
        # Print Header
        #
        # LINE 1
        #
        ostream.write("g3 1 1 0\t# problem %s\n" % (model.name,))
        #
        # LINE 2
        #
        ostream.write(
            " %d %d %d %d %d \t"
            "# vars, constraints, objectives, ranges, eqns\n"
            % ( n_vars,
                n_cons,
                n_objs,
                n_ranges,
                n_equality,
            ))
        #
        # LINE 3
        #
        ostream.write(
            " %d %d %d %d %d %d \t"
            "# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n"
            % ( n_nonlinear_cons,
                n_nonlinear_objs,
                0, # ccons_lin,
                0, # ccons_nonlin,
                0, # ccons_nd,
                0, # ccons_nzlb,
            ))
        #
        # LINE 4
        #
        ostream.write(" 0 0\t# network constraints: nonlinear, linear\n")
        #
        # LINE 5
        #
        ostream.write(
            " %d %d %d \t"
            "# nonlinear vars in constraints, objectives, both\n"
            % ( len(con_vars_nonlinear),
                # Note that the objectives entry appears to be reported
                # by AMPL as the total number of nonlinear variables
                #len(obj_vars_nonlinear),
                len(nonlinear_vars),
                len(both_vars_nonlinear),
            ))

        #
        # LINE 6
        #
        ostream.write(
            " 0 %d 0 1\t"
            "# linear network variables; functions; arith, flags\n"
            % ( len(self.external_functions),
            ))
        #
        # LINE 7
        #
        ostream.write(
            " %d %d %d %d %d \t"
            "# discrete variables: binary, integer, nonlinear (b,c,o)\n"
            % ( len(con_vars_linear.intersection(binary_vars)),
                len(con_vars_linear.intersection(integer_vars)),
                len(both_vars_nonlinear.intersection(discrete_vars)),
                len(con_vars_nonlinear.intersection(discrete_vars)),
                len(obj_vars_nonlinear.intersection(discrete_vars)),
            ))
        #
        # LINE 8
        #
        # objective info computed above
        ostream.write(
            " %d %d \t# nonzeros in Jacobian, obj. gradient\n"
            % ( sum(con_nnz_by_var.values()),
                len(obj_vars),
            ))
        #
        # LINE 9
        #
        ostream.write(
            " %d %d\t# max name lengths: constraints, variables\n"
            % ( max(map(len, row_labels)),
                max(map(len, col_labels)),
            ))
        #
        # LINE 10
        #
        ostream.write(" %d %d %d %d %d\t# common exprs: b,c,o,c1,o1\n"
                      % tuple(n_subexpressions))

        #
        # "F" lines (external function definitions)
        #
        amplfunc_libraries = set()
        for fid, fcn in sorted(self.external_functions.values()):
            amplfunc_libraries.add(fcn._library)
            ostream.write("F%d 1 -1 %s\n" % (fid, fcn._function))

        #
        # "S" lines (suffixes)
        #

        #
        # "V" lines (common subexpressions)
        #
        # per "writing .nl files", common subexpressions appearing in
        # more than one constraint/objective come first, then
        # subexpressions that only appear in one place come immediately
        # before the C/O line that references it.
        single_use_subexpressions = {}
        self.next_V_line_id = n_vars
        for _id in self.subexpression_order:
            if _id not in self.used_named_expressions:
                continue
            cache = self.subexpression_cache[_id]
            if cache[2][2]:
                # substitute expression directly into expression trees
                # and do NOT emit the V line
                pass
            elif 0 in cache[2] or None not in cache[2]:
                self._write_v_line(_id, 0)
            else:
                target_expr = tuple(filter(None, cache[2]))[0]
                if target_expr not in single_use_subexpressions:
                    single_use_subexpressions[target_expr] = []
                single_use_subexpressions[target_expr].append(_id)
        #
        # "C" lines (constraints: nonlinear expression)
        #
        for row_idx, info in enumerate(constraints):
            for _id in single_use_subexpressions.get(id(info[0]), ()):
                self._write_v_line(_id, row_idx)
            ostream.write(f'C{row_idx}{row_comments[row_idx]}\n')
            self._write_nl_expression(info[1], False)

        #
        # "O" lines (objectives: nonlinear expression)
        #
        for obj_idx, info in enumerate(objectives):
            for _id in single_use_subexpressions.get(id(info[0]), ()):
                self._write_v_line(_id, n_cons + n_lcons + obj_idx)
            lbl = row_comments[n_cons + obj_idx]
            sense = 0 if info[0].sense == minimize else 1
            ostream.write(f'O{obj_idx} {sense}{lbl}\n')
            self._write_nl_expression(info[1], True)

        #
        # "d" lines (dual initialization)
        #

        #
        # "x" lines (variable initialization)
        #
        _init_lines = [
            f'{var_idx} {info[0].value!r}{col_comments[var_idx]}\n'
            for var_idx, info in enumerate(variables)
            if info[0].value is not None
        ]
        ostream.write('x%d%s\n' % (
            len(_init_lines),
            "\t# initial guess" if symbolic_solver_labels else '',
        ))
        ostream.write(''.join(_init_lines))

        #
        # "r" lines (constraint bounds)
        #
        ostream.write('r%s\n' % (
            "\t#%d ranges (rhs's)" % len(constraints)
            if symbolic_solver_labels else '',
        ))
        # _bound_writer = {
        #     0: lambda i, c: ostream.write(f"0 {i[3]} {i[4]}{c}\n"),
        #     1: lambda i, c: ostream.write(f"1 {i[4]}{c}\n"),
        #     2: lambda i, c: ostream.write(f"2 {i[3]}{c}\n"),
        #     3: lambda i, c: ostream.write(f"3{c}\n"),
        #     4: lambda i, c: ostream.write(f"4 {i[3]}{c}\n"),
        # }
        for row_idx, info in enumerate(constraints):
            # _bound_writer[info[2]](info, row_comments[row_idx])
            ###
            i = info[2]
            if i == 4:   # ==
                ostream.write(f"4 {info[3]}{row_comments[row_idx]}\n")
            elif i == 1: # body <= ub
                ostream.write(f"1 {info[4]}{row_comments[row_idx]}\n")
            elif i == 2: # lb <= body
                ostream.write(f"2 {info[3]}{row_comments[row_idx]}\n")
            elif i == 0: # lb <= body <= ub
                ostream.write(f"0 {info[3]} {info[4]}{row_comments[row_idx]}\n")
            else: # i == 3; unbounded
                ostream.write(f"3{row_comments[row_idx]}\n")

        #
        # "b" lines (variable bounds)
        #
        ostream.write('b%s\n' % (
            "\t#%d bounds (on variables)" % len(variables)
            if symbolic_solver_labels else '',
        ))
        for var_idx, info in enumerate(variables):
            # _bound_writer[info[2]](info, col_comments[var_idx])
            ###
            i = info[2]
            if i == 0: # lb <= body <= ub
                ostream.write(f"0 {info[3]} {info[4]}{col_comments[var_idx]}\n")
            elif i == 2: # lb <= body
                ostream.write(f"2 {info[3]}{col_comments[var_idx]}\n")
            elif i == 1: # body <= ub
                ostream.write(f"1 {info[4]}{col_comments[var_idx]}\n")
            elif i == 4:   # ==
                ostream.write(f"4 {info[3]}{col_comments[var_idx]}\n")
            else: # i == 3; unbounded
                ostream.write(f"3{col_comments[var_idx]}\n")

        #
        # "k" lines (column offsets in Jacobian NNZ)
        #
        ostream.write('k%d%s\n' % (
            len(variables) - 1,
            "\t#intermediate Jacobian column lengths"
            if symbolic_solver_labels else '',
        ))
        ktot = 0
        for var_idx, info in enumerate(variables[:-1]):
            ktot += con_nnz_by_var.get(info[1], 0)
            ostream.write(f"{ktot}\n")

        #
        # "J" lines (non-empty terms in the Jacobian)
        #
        for row_idx, info in enumerate(constraints):
            linear = info[1].linear
            ostream.write(f'J{row_idx} {len(linear)}{row_comments[row_idx]}\n')
            for _id in sorted(linear.keys(), key=column_order.__getitem__):
                ostream.write(
                    f'{column_order[_id]} {linear[_id]!r}\n'
                )

        #
        # "G" lines (non-empty terms in the Objective)
        #
        for obj_idx, info in enumerate(objectives):
            linear = info[1].linear
            ostream.write(
                f'G{obj_idx} {len(linear)}{row_comments[obj_idx + n_cons]}\n')
            for _id in sorted(linear.keys(), key=column_order.__getitem__):
                ostream.write(
                    f'{column_order[_id]} {linear[_id]!r}\n'
                )

        timer.toc("written")
        return symbol_map, sorted(amplfunc_libraries)


    def _categorize_vars(self, comp_list, linear_by_comp):
        """Categorize compiled expression vars into linear and nonlinear

        This routine takes an iterable of compiled component expression
        infos and returns the sets of variables appearing lineary and
        nonlinearly in those components.

        This routine has a number of side effects:

          - the ``linear_by_comp`` dict is updated to contain the set of
            nonzeros for each component in the ``comp_list``

          - the expr_info (the second element in each tuple in
            ``comp_list``) is "compiled": the ``linear`` attribute is
            converted from a list of coef, var_id terms (potentially with
            duplicate entries) into a dict that maps var id to
            coefficients

        Returns
        -------
        all_linear_vars: set
            set of all vars that only appear linearly in the compiled
            component expression infos

        all_nonlinear_vars: set
            set of all vars that appear nonlinearly in the compiled
            component expression infos

        nnz_by_var: dict
            Count of the number of components that each var appears in.

        """
        all_linear_vars = set()
        all_nonlinear_vars = set()
        nnz_by_var = {}

        for comp_info in comp_list:
            expr_info = comp_info[1]
            # Note: mult will be 1 here: it is either cleared by
            # finalizeResult, or this is a named expression, in which
            # case the mult was reset within handle_named_expression_node
            #assert expr_info.mult == 1
            #
            # Process the linear portion of this component
            if expr_info.linear:
                if expr_info.linear.__class__ is list:
                    linear = {}
                    for v, c in expr_info.linear:
                        if v in linear:
                            linear[v] += c
                        else:
                            linear[v] = c
                    expr_info.linear = linear
                linear_vars = set(expr_info.linear)
                all_linear_vars.update(linear_vars)
            # else:
            #     # NOTE: we only create the linear_vars set if there
            #     # are linear vars: the use of linear_vars below is
            #     # guarded by 'if expr_info.linear'
            #     linear_vars = set()

            # Process the nonlinear portion of this component
            if expr_info.nonlinear:
                nonlinear_vars = set()
                for _id in expr_info.nonlinear[1]:
                    if _id in linear_by_comp:
                        nonlinear_vars.update(linear_by_comp[_id].keys())
                    else:
                        nonlinear_vars.add(_id)
                # Recreate nz if this component has both linear and
                # nonlinear components.
                if expr_info.linear:
                    # Ensure any variables that only appear nonlinearly
                    # in the expression have 0's in the linear dict
                    for i in nonlinear_vars - linear_vars:
                        expr_info.linear[i] = 0
                else:
                    # All variables are nonlinear; generate the linear
                    # dict with all zeros
                    expr_info.linear = dict.fromkeys(nonlinear_vars, 0)
                all_nonlinear_vars.update(nonlinear_vars)

            # Update the count of components that each variable appears in
            #nnz_by_var.update(nz)
            for v in expr_info.linear:
                if v in nnz_by_var:
                    nnz_by_var[v] += 1
                else:
                    nnz_by_var[v] = 1
            # Record all nonzero variable ids for this component
            linear_by_comp[id(comp_info[0])] = expr_info.linear
        # Linear models (or objectives) are common.  Avoid the set
        # difference if possible
        if all_nonlinear_vars:
            all_linear_vars -= all_nonlinear_vars
        return all_linear_vars, all_nonlinear_vars, nnz_by_var

    def _count_subexpression_occurances(self):
        # We now need to go through the subexpression cache and update
        # the flag for nested subexpressions used by multiple components
        # (the walker can only update the flag in subexpressions
        # appearing explicitly in the tree, so we now need to propagate
        # this usage info into subexpressions nested in other
        # subexpressions).
        #
        # We need to walk twice: once to sort out the use in Constraints
        # and once to sort out the use in Objectives
        for idx in (0, 1):
            cache = self.subexpression_cache
            for id_ in self.subexpression_order:
                src_id = cache[id_][2][idx]
                if src_id is None:
                    continue
                # This expression is used by this component type
                # (constraint or objective); ensure that all
                # subexpressions (recursively) used by this expression
                # are also marked as being used by this component type
                queue = [id_]
                while queue:
                    info = cache[queue.pop()]
                    if not info[1].nonlinear:
                        # Subexpressions can only appear in the
                        # nonlinear terms.  If there are none, then we
                        # are done.
                        continue
                    for subid in info[1].nonlinear[1]:
                        # Check if this "id" (normally a var id, but
                        # could be a subexpression id) is a
                        # subexpression id
                        if subid not in cache:
                            continue
                        # Check if we need to update this subexpression:
                        # either it has never been marked as being used
                        # by this component type, or else it was used by
                        # a different id.  If we need to update the
                        # flag, then do so and recurse into it
                        target = cache[subid][2]
                        if (target[idx] is None
                            or (target[idx] and target[idx] != src_id)):
                            target[idx] = src_id
                            queue.append(subid)
        # Now we can reliably know where nested subexpressions are used.
        # Group them into:
        #   [ used in both objectives and constraints,
        #     used by more than one constraint (but no objectives),
        #     used by more than one objective (but no constraints),
        #     used by one constraint,
        #     used by one objective ]
        n_subexpressions = [0]*5
        for info in map(itemgetter(2), self.subexpression_cache.values()):
            if info[2]:
                pass
            elif info[1] is None:
                n_subexpressions[3 if info[0] else 1] += 1
            elif info[0] is None:
                n_subexpressions[4 if info[1] else 2] += 1
            else:
                n_subexpressions[0] += 1
        return n_subexpressions

    def _write_nl_expression(self, repn, include_const):
        #assert repn.mult == 1
        if repn.nonlinear:
            nl, args = repn.nonlinear
            if include_const and repn.const:
                # Add the constant to the NL expression.  AMPL adds the
                # constant as the second argument, so we will too.
                nl = self.template.binary_sum + nl + (
                    self.template.const % repn.const)
            self.ostream.write(
                nl % tuple(map(self.var_id_to_nl.__getitem__, args))
            )
        elif include_const:
            self.ostream.write(self.template.const % repn.const)
        else:
            self.ostream.write(self.template.const % 0)

    def _write_v_line(self, expr_id, k):
        ostream = self.ostream
        column_order = self.column_order
        info = self.subexpression_cache[expr_id]
        if self.symbolic_solver_labels:
            lbl = '\t#%s' % info[0].name
        else:
            lbl = ''
        self.var_id_to_nl[expr_id] = f"{self.next_V_line_id}{lbl}"
        linear = info[1].linear
        ostream.write(f'V{self.next_V_line_id} {len(linear)} {k}{lbl}\n')
        for _id in sorted(linear, key=column_order.__getitem__):
            ostream.write(f'{column_order[_id]} {linear[_id]!r}\n')
        self._write_nl_expression(info[1], True)
        self.next_V_line_id += 1


class NLFragment(object):
    """This is a mock "component" for the nl portion of a named Expression.

    It is used internally in the writer when requesting symbolic solver
    labels so that we can generate meaningful names for the nonlinear
    portion of an Expression component.

    """
    __slots__ = ('_repn', '_node')

    def __init__(self, repn, node):
        self._repn = repn
        self._node = node

    @property
    def name(self):
        return 'nl(' + self._node.name + ')'


class AMPLRepn(object):
    __slots__ = ('nl', 'mult', 'const', 'linear', 'nonlinear')

    ActiveVisitor = None

    def __init__(self, const, linear, nonlinear):
        self.nl = None
        self.mult = 1
        self.const = const
        self.linear = linear
        self.nonlinear = nonlinear

    def compile_repn(self, visitor, prefix='', args=None):
        template = visitor.template
        if self.mult != 1:
            prefix += template.multiplier % self.mult
        if self.nl is not None:
            nl, nl_args = self.nl
            visitor.used_named_expressions.update(nl_args)
            if prefix:
                nl = prefix + nl
            if args is not None and args is not nl_args:
                args.extend(nl_args)
            else:
                args = list(nl_args)
            return nl, args

        if args is None:
            args = []
        nterms = 0
        if self.const:
            nterms += 1
            nl_sum = template.const % self.const
        else:
            nl_sum = ''
        if self.linear:
            nterms += len(self.linear)
            nl_sum += ''.join(
                template.var if c == 1 else template.monomial % c
                for c in map(itemgetter(1), self.linear))
            args.extend(map(itemgetter(0), self.linear))
        if self.nonlinear:
            if self.nonlinear.__class__ is list:
                nterms += len(self.nonlinear)
                nl_sum += ''.join(map(itemgetter(0), self.nonlinear))
                deque(map(args.extend, map(itemgetter(1), self.nonlinear)),
                      maxlen=0)
            else:
                nterms += 1
                nl_sum += self.nonlinear[0]
                args.extend(self.nonlinear[1])

        if nterms > 2:
            return prefix + (template.nary_sum % nterms) + nl_sum, args
        elif nterms == 2:
            return prefix + template.binary_sum + nl_sum, args
        elif nterms == 1:
            return prefix + nl_sum, args
        else: # nterms == 0
            return prefix + (template.const % 0), []

    def compile_nonlinear_fragment(self, template):
        args = []
        nterms = len(self.nonlinear)
        nl_sum = ''.join(map(itemgetter(0), self.nonlinear))
        deque(map(args.extend, map(itemgetter(1), self.nonlinear)),
              maxlen=0)

        if nterms > 2:
            self.nonlinear = (template.nary_sum % nterms) + nl_sum, args
        elif nterms == 2:
            self.nonlinear = template.binary_sum + nl_sum, args
        elif nterms == 1:
            self.nonlinear = nl_sum, args
        else: # nterms == 0
            self.nonlinear = None

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use an AMPLRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

        """
        #assert self.mult == 1
        _type = other[0]
        if _type is _MONOMIAL:
            self.linear.append(other[1:])
        elif _type is _GENERAL:
            other = other[1]
            if other.mult != 1:
                if other.nonlinear and other.nonlinear.__class__ is list:
                    other.compile_nonlinear_fragment(
                        self.ActiveVisitor.template)
                mult = other.mult
                self.const += mult * other.const
                if other.linear:
                    self.linear.extend((v, c*mult) for v, c in other.linear)
                if other.nonlinear:
                    if mult == -1:
                        prefix = self.ActiveVisitor.template.negation
                    else:
                        prefix = self.ActiveVisitor.template.multiplier % mult
                    self.nonlinear.append(
                        (prefix + other.nonlinear[0], other.nonlinear[1])
                    )
            else:
                self.const += other.const
                if other.linear:
                    self.linear.extend(other.linear)
                if other.nonlinear:
                    if other.nonlinear.__class__ is list:
                        self.nonlinear.extend(other.nonlinear)
                    else:
                        self.nonlinear.append(other.nonlinear)
        elif _type is _CONSTANT:
            self.const += other[1]


def _create_strict_inequality_map(vars_):
    vars_['strict_inequality_map'] = {
        True: vars_['less_than'],
        False: vars_['less_equal'],
        (True, True): (vars_['less_than'], vars_['less_than']),
        (True, False): (vars_['less_than'], vars_['less_equal']),
        (False, True): (vars_['less_equal'], vars_['less_than']),
        (False, False): (vars_['less_equal'], vars_['less_equal']),
    }


class text_nl_debug_template(object):
    unary = {
        'log':    'o43\t#log\n',
        'log10':  'o42\t#log10\n',
        'sin':    'o41\t#sin\n',
        'cos':    'o46\t#cos\n',
        'tan':    'o38\t#tan\n',
        'sinh':   'o40\t#sinh\n',
        'cosh':   'o45\t#cosh\n',
        'tanh':   'o37\t#tanh\n',
        'asin':   'o51\t#asin\n',
        'acos':   'o53\t#acos\n',
        'atan':   'o49\t#atan\n',
        'exp':    'o44\t#exp\n',
        'sqrt':   'o39\t#sqrt\n',
        'asinh':  'o50\t#asinh\n',
        'acosh':  'o52\t#acosh\n',
        'atanh':  'o47\t#atanh\n',
        'ceil':   'o14\t#ceil\n',
        'floor':  'o13\t#floor\n',
    }

    binary_sum = 'o0\t# +\n'
    product = 'o2\t# *\n'
    division = 'o3\t# /\n'
    pow = 'o5\t#^\n'
    abs = 'o15\t# abs\n'
    negation = 'o16\t# -\n'
    nary_sum = 'o54\t# sumlist\n%d\t# (n)\n'
    exprif = 'o35\t# if\n'
    and_expr = 'o21\t# and\n'
    less_than = 'o22\t# lt\n'
    less_equal = 'o23\t# le\n'
    equality = 'o24\t# eq\n'
    external_fcn = 'f%d %d\n'
    var = 'v%s\n'
    const = 'n%r\n'
    string = 'h%d:%s\n'
    monomial = product + const + var.replace('%', '%%')
    multiplier = product + const

    _create_strict_inequality_map(vars())

def _strip_template_comments(vars_, base_):
    vars_['unary'] = {k: v[:v.find('\t#')]+'\n'
             for k, v in base_.unary.items()}
    for k, v in base_.__dict__.items():
        if type(v) is str and '\t#' in v:
            v_lines = v.split('\n')
            for i, l in enumerate(v_lines):
                comment_start = l.find('\t#')
                if comment_start >= 0:
                    v_lines[i] = l[:comment_start]
            vars_[k] = '\n'.join(v_lines)


# The "standard" text mode template is the debugging template with the
# comments removed
class text_nl_template(text_nl_debug_template):
    _strip_template_comments(vars(), text_nl_debug_template)
    _create_strict_inequality_map(vars())


def node_result_to_amplrepn(data):
    if data[0] is _GENERAL:
        return data[1]
    elif data[0] is _MONOMIAL:
        if data[2]:
            return AMPLRepn(0, (data[1:],), None)
        else:
            return AMPLRepn(0, None, None)
    elif data[0] is _CONSTANT:
        return AMPLRepn(data[1], None, None)
    else:
        raise DeveloperError("unknown result type")

def handle_negation_node(visitor, node, arg1):
    if arg1[0] is _MONOMIAL:
        return (_MONOMIAL, arg1[1], -1*arg1[2])
    elif arg1[0] is _GENERAL:
        arg1[1].mult *= -1
        return arg1
    elif arg1[0] is _CONSTANT:
        return (_CONSTANT, -1*arg1[1])
    else:
        raise RuntimeError("%s: %s" % (type(arg1[0]), arg1))

def handle_product_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        arg2, arg1 = arg1, arg2
    if arg1[0] is _CONSTANT:
        mult = arg1[1]
        if not mult:
            # simplify multiplication by 0 (if arg2 is zero, the
            # simplification happens implicitly when we evaluate the
            # constant below)
            return arg1
        if mult == 1:
            return arg2
        elif arg2[0] is _MONOMIAL:
            return (_MONOMIAL, arg2[1], mult*arg2[2])
        elif arg2[0] is _GENERAL:
            arg2[1].mult *= mult
            return arg2
        elif arg2[0] is _CONSTANT:
            return (_CONSTANT, mult*arg2[1])
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.product)
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_division_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        div = arg2[1]
        if div == 1:
            return arg1
        if arg1[0] is _MONOMIAL:
            return (_MONOMIAL, arg1[1], arg1[2]/div)
        elif arg1[0] is _GENERAL:
            arg1[1].mult /= div
            return arg1
        elif arg1[0] is _CONSTANT:
            return (_CONSTANT, arg1[1]/div)
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.division)
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_pow_node(visitor, node, arg1, arg2):
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.pow)
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_abs_node(visitor, node, arg1):
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.abs)
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_unary_node(visitor, node, arg1):
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.unary[node.name])
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_exprif_node(visitor, node, arg1, arg2, arg3):
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.exprif)
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    nonlin = node_result_to_amplrepn(arg3).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_equality_node(visitor, node, arg1, arg2):
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.equality)
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_inequality_node(visitor, node, arg1, arg2):
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.strict_inequality_map[node.strict])
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    op = visitor.template.strict_inequality_map[node.strict]
    nl, args = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.and_expr + op[0])
    nl2, args2 = node_result_to_amplrepn(arg2).compile_repn(visitor)
    nl += nl2 + op[1] + nl2
    args.extend(args2)
    args.extend(args2)
    nonlin = node_result_to_amplrepn(arg3).compile_repn(visitor, nl, args)
    return (_GENERAL, AMPLRepn(0, None, nonlin))

def handle_named_expression_node(visitor, node, arg1):
    #return arg1
    _id = id(node)
    # Note that while named subexpressions ('defined variables' in the
    # ASL NL file vernacular) look like variables, they are not allowed
    # to appear in the 'linear' portion of a constraint / objective
    # definition.  We will return this as a "var" template, but
    # wrapped in the nonlinear portion of the expression tree.
    repn = node_result_to_amplrepn(arg1)

    # When converting this shared subexpression to a (nonlinear)
    # node, we want to just reference this subexpression:
    repn.nl = (visitor.template.var, (_id,))

    # A local copy of the expression source list.  This will be updated
    # later if the same Expression node is encountered in another
    # expression tree.
    expression_source = list(visitor.active_expression_source)

    if repn.nonlinear:
        # As we will eventually need the compiled form of any nonlinear
        # expression, we will go ahead and compile it here.  We do not
        # do the same for the linear component as we will only need the
        # linear component compiled to a dict if we are emitting the
        # original (linear + nonlinear) V line (which will not happen if
        # the V line is part of a larger linear operator).
        if repn.nonlinear.__class__ is list:
            repn.compile_nonlinear_fragment(visitor.template)

        if repn.linear:
            # If this expession has both linear and nonlinear
            # components, we will follow the ASL convention and break
            # the named subexpression into two named subexpressions: one
            # that is only the nonlinear component and one that has the
            # const/linear component (and references the first).  This
            # will allow us to propagate linear coefficients up from
            # named subexpressions when appropriate.
            sub_node = NLFragment(repn, node)
            sub_id = id(sub_node)
            sub_repn = AMPLRepn(0, None, repn.nonlinear)
            sub_repn.nl = (visitor.template.var, (sub_id,))
            # See below for the meaning of this tuple
            visitor.subexpression_cache[sub_id] = (
                sub_node, sub_repn, list(expression_source),
            )
            repn.nonlinear = sub_repn.nl
            # It is important that the NL subexpression comes before the
            # main named expression:
            visitor.subexpression_order.append(sub_id)
            # The nonlinear identifier is *always* used
            visitor.used_named_expressions.add(sub_id)
    else:
        repn.nonlinear = None
        if repn.linear:
            if (not repn.const and len(repn.linear) == 1
                and repn.linear[0][1] == 1):
                # This Expression holds only a variable (multiplied by
                # 1).  Do not emit this as a named variable and instead
                # just inject the variable where this expression is
                # used.
                repn.nl = None
                expression_source[2] = True
        else:
            # This Expression holds only a constant.  Do not emit this
            # as a named variable and instead just inject the constant
            # where this expression is used.
            repn.nl = None
            expression_source[2] = True

    if repn.mult != 1:
        mult = repn.mult
        repn.mult = 1
        repn.const *= mult
        if repn.linear:
            repn.linear = [(v, c*mult) for v, c in repn.linear]
        if repn.nonlinear:
            if mult == -1:
                prefix = visitor.template.negation
            else:
                prefix = visitor.template.multiplier % mult
            repn.nonlinear = prefix + repn.nonlinear[0], repn.nonlinear[1]

    visitor.subexpression_cache[_id] = (
        # 0: the "component" that generated this expression ID
        node,
        # 1: the common subexpression (to be written out)
        repn,
        # 2: the (single) component that uses this subexpression.  This
        # is a 3-tuple [con_id, obj_id, substitute_expression].  If the
        # expression is used by 1 constraint / objective, then the id is
        # set to 0.  If it is not used by any, then it is None.
        # substitue_expression is a bool indicating if this named
        # subexpression tree should be directly substituted into any
        # expression tree that references this node (i.e., do NOT emit
        # the V line).
        expression_source,
    )
    visitor.subexpression_order.append(_id)
    ans = AMPLRepn(repn.const, repn.linear, repn.nonlinear)
    ans.nl = repn.nl
    return (_GENERAL, ans)

def handle_external_function_node(visitor, node, *args):
    func = node._fcn._function
    if func in visitor.external_functions:
        if node._fcn._library != visitor.external_functions[func][1]._library:
            raise RuntimeError(
                "The same external function name (%s) is associated "
                "with two different libraries (%s through %s, and %s "
                "through %s).  The ASL solver will fail to link "
                "correctly." %
                (func,
                 visitor.external_byFcn[func]._library,
                 visitor.external_byFcn[func]._library.name,
                 node._fcn._library,
                 node._fcn.name))
    else:
        visitor.external_functions[func] = (
            len(visitor.external_functions),
            node._fcn,
        )
    nonlin = node_result_to_amplrepn(args[0]).compile_repn(
        visitor, visitor.template.external_fcn % (
            visitor.external_functions[func][0], len(args)))
    for arg in args[1:]:
        nonlin = node_result_to_amplrepn(arg[0]).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


_operator_handles = {
    NegationExpression: handle_negation_node,
    ProductExpression: handle_product_node,
    DivisionExpression: handle_division_node,
    PowExpression: handle_pow_node,
    AbsExpression: handle_abs_node,
    UnaryFunctionExpression: handle_unary_node,
    Expr_ifExpression: handle_exprif_node,
    EqualityExpression: handle_equality_node,
    InequalityExpression: handle_inequality_node,
    RangedExpression: handle_ranged_inequality_node,
    _GeneralExpressionData: handle_named_expression_node,
    ScalarExpression: handle_named_expression_node,
    kernel.expression.expression: handle_named_expression_node,
    ExternalFunctionExpression: handle_external_function_node,
    # These are handled explicitly in beforeChild():
    # LinearExpression: handle_linear_expression,
    # SumExpression: handle_sum_expression,
    # MonomialTermExpression: handle_monomial_term,
}


def _before_native(visitor, child):
    return False, (_CONSTANT, child)

def _before_non_expression(visitor, child):
    if child.is_fixed():
        return False, (_CONSTANT, child())
    else:
        _id = id(child)
        if _id not in visitor.var_map:
            visitor.var_map[_id] = child
        return False, (_MONOMIAL, _id, 1)

def _before_npv(visitor, child):
    # _id = id(child)
    # if _id in visitor.value_cache:
    #     child = visitor.value_cache[_id]
    # else:
    #     child = visitor.value_cache[_id] = child()
    return False, (_CONSTANT, child())

def _before_monomial(visitor, child):
    #
    # The following are performance optimizations for common
    # situations (Monomial terms and Linear expressions)
    #
    arg1, arg2 = child._args_
    if arg1.__class__ not in native_types:
        # _id = id(arg1)
        # if _id in visitor.value_cache:
        #     arg1 = visitor.value_cache[_id]
        # else:
        #     arg1 = visitor.value_cache[_id] = arg1()
        arg1 = arg1()
    if arg2.is_fixed():
        return False, (_CONSTANT, arg1 * arg2())
    else:
        _id = id(arg2)
        if _id not in visitor.var_map:
            visitor.var_map[_id] = arg2
        return False, (_MONOMIAL, _id, arg1)

def _before_linear(visitor, child):
    # Because we are going to modify the LinearExpression in this
    # walker, we need to make a copy of the LinearExpression from
    # the original expression tree.
    var_map = visitor.var_map
    const = child.constant
    linear = []
    for v, c in zip(child.linear_vars, child.linear_coefs):
        if c.__class__ not in native_types:
            c = c()
        if v.is_fixed():
            const += c * v()
        else:
            _id = id(v)
            if _id not in var_map:
                var_map[_id] = v
            linear.append((_id, c))
    return False, (_GENERAL, AMPLRepn(const, linear, None))

def _before_named_expression(visitor, child):
    _id = id(child)
    if _id in visitor.subexpression_cache:
        obj, repn, info = visitor.subexpression_cache[_id]
        info[visitor.active_expression_source_idx] = 0
        ans = AMPLRepn(repn.const, repn.linear, repn.nonlinear)
        ans.nl = repn.nl
        return False, (_GENERAL, ans)
    else:
        return True, None

def _before_general_expression(visitor, child):
    return True, None


# Register an initial set of known expression types with the "before
# child" expression handler lookup table.
_before_child_handlers = {
    _type: _before_native for _type in native_types
}
# general operators
for _type in _operator_handles:
    _before_child_handlers[_type] = _before_general_expression
# named subexpressions
for _type in (_GeneralExpressionData, ScalarExpression,
              kernel.expression.expression):
    _before_child_handlers[_type] = _before_named_expression
# Special linear / summation expressions
_before_child_handlers[MonomialTermExpression] = _before_monomial
_before_child_handlers[LinearExpression] = _before_linear
_before_child_handlers[SumExpression] = _before_general_expression

class AMPLRepnVisitor(StreamBasedExpressionVisitor):

    def __init__(self, template, subexpression_cache, subexpression_order,
                 external_functions, var_map, used_named_expressions):
        super().__init__()
        self.template = template
        self.subexpression_cache = subexpression_cache
        self.subexpression_order = subexpression_order
        self.external_functions = external_functions
        self.active_expression_source = None
        self.var_map = var_map
        self.used_named_expressions = used_named_expressions
        #self.value_cache = {}

    def initializeWalker(self, expr):
        expr, src, src_idx = expr
        self.active_expression_source = [None, None, False]
        self.active_expression_source[src_idx] = id(src)
        self.active_expression_source_idx = src_idx
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        try:
            return _before_child_handlers[child.__class__](self, child)
        except KeyError:
            self._register_new_before_child_processor(child)
        return _before_child_handlers[child.__class__](self, child)

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if node.__class__ is SumExpression:
            return node.args, AMPLRepn(0, [], [])
        else:
            return node.args, []

    def exitNode(self, node, data):
        if data.__class__ is AMPLRepn:
            return (_GENERAL, data)
        #
        # General expressions...
        #
        if all(arg[0] is _CONSTANT for arg in data):
            return (
                _CONSTANT, node._apply_operation(list(map(
                    itemgetter(1), data)))
            )
        return _operator_handles[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        ans = node_result_to_amplrepn(result)
        if ans.nl:
            self.used_named_expressions.update(ans.nl[1])
            ans.const = 0
            ans.linear = None
            ans.nonlinear = ans.nl
            ans.nl = None
        if ans.nonlinear.__class__ is list:
            if ans.nonlinear:
                ans.compile_nonlinear_fragment(self.template)
            else:
                ans.nonlinear = None
        linear = {}
        if ans.mult != 1:
            mult = ans.mult
            ans.mult = 1
            ans.const *= mult
            if ans.linear:
                for v, c in ans.linear:
                    if v in linear:
                        linear[v] += mult * c
                    else:
                        linear[v] = mult * c
            if ans.nonlinear:
                if mult == -1:
                    prefix = self.template.negation
                else:
                    prefix = self.template.multiplier % mult
                ans.nonlinear = prefix + ans.nonlinear[0], ans.nonlinear[1]
        elif ans.linear:
            for v, c in ans.linear:
                if v in linear:
                    linear[v] += c
                else:
                    linear[v] = c
        ans.linear = linear
        #
        self.active_expression_source = None
        return ans

    def _register_new_before_child_processor(self, child):
        handlers = _before_child_handlers
        child_type = child.__class__
        if child_type in native_types:
            handlers[child_type] = _before_native
        elif not child.is_expression_type():
            handlers[child_type] = _before_non_expression
        elif not child.is_potentially_variable():
            handlers[child_type] = _before_npv
        elif id(child) in self.subexpression_cache:
            handlers[child_type] = _before_named_expression
        else:
            handlers[child_type] = _before_general_expression
