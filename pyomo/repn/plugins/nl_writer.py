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
from collections import Counter, deque
from operator import itemgetter, attrgetter

from pyomo.common.backports import nullcontext
from pyomo.common.config import ConfigBlock, ConfigValue, InEnum
from pyomo.common.errors import DeveloperError

from pyomo.core.expr.current import (
    NegationExpression, ProductExpression, DivisionExpression,
    PowExpression, AbsExpression, UnaryFunctionExpression,
    MonomialTermExpression, LinearExpression, SumExpressionBase,
    EqualityExpression, InequalityExpression, RangedExpression,
    Expr_ifExpression, ExternalFunctionExpression,
    native_types, value,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.base import (
    Block, Objective, Constraint, Var, Param, Expression, ExternalFunction,
    SymbolMap, NameLabeler, SortComponents, minimize,
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
    WriterFactory.register('nl', doc)(WriterFactory.get_class('nl_v%s' % n))

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
        assert filename.lower().endswith('.nl')
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
        })
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the NL writer does not know how to process:\n\t%s" %
                (model.name, "\n\t".join("%s:\n\t\t%s" % (
                    k, "\n\t\t".join(map(attrgetter('name'), v)))
                    for k, v in unknown.items())))

        _impl = _NLWriter_impl(ostream, rowstream, colstream, config)
        return _impl.write(model)


class _NLWriter_impl(object):
    RANGE_TYPE = {
        # has_lb, has_ub, lb == ub
        (True, True, False): 0,   # L <= c <= U
        (False, True, False): 1,  #      c <= U
        (True, False, False): 2,  # L <= c
        (False, False, True): 3,  # -inf <= c <= inf
        (True, True, True): 4,    # L == c == U
        # complementarity: 5,
    }

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
        self.var_map = _deterministic_dict()
        self.visitor = AMPLRepnVisitor(
            self.template,
            self.subexpression_cache,
            self.subexpression_order,
            self.external_functions,
            self.var_map,
        )
        self.next_V_line_id = 0

    def write(self, model):
        sorter = SortComponents.unsorted
        if self.config.file_determinism >= FileDeterminism.SORT_INDICES:
            sorter = sorter | SortComponents.indices
            if self.config.file_determinism  >= FileDeterminism.SORT_SYMBOLS:
                sorter = sorter | SortComponents.alphabetical

        symbolic_solver_labels = self.symbolic_solver_labels
        visitor = self.visitor
        ostream = self.ostream

        if self.config.file_determinism > FileDeterminism.NONE:
            # We will pre-gather the variables so that their order
            # matches the file_determinism flag.
            #
            # This is a little cumbersome, but is implemented this way
            # for consistency with the original NL writer.  Note that
            # Vars that appear twice (e.g., through a Reference) will be
            # sorted with the LAST occurance.
            self.var_map = {
                id(var): var
                for var in model.component_data_objects(
                        Var, descend_into=True, sort=sorter)
            }

        #
        # Tabulate the model expressions
        #

        objectives = [
            (obj,
             visitor.walk_expression((obj.expr, obj, 1)),
         ) for obj in model.component_data_objects(
             Objective, active=True, descend_into=True, sort=sorter)]

        constraints = [(
            con,
            visitor.walk_expression((con.body, con, 0)),
            self.RANGE_TYPE[con.has_lb(), con.has_ub(), con.lb == con.ub],
        ) for con in model.component_data_objects(
            Constraint, active=True, descend_into=True, sort=sorter
        ) if con.has_lb() or con.has_ub() ]

        #
        # Collect constraints and objectives into the groupings
        # necessary for AMPL
        #
        # For efficiency, we will do everything with ids (and not the
        # var objects themselves)
        #

        # Reorder the constraints, moving all nonlinear constraints to
        # the beginning
        nonlinear_cons = [con for con in constraints if con[1].nonlinear]
        linear_cons = [con for con in constraints if not con[1].nonlinear]
        constraints = nonlinear_cons + linear_cons

        n_ranges = sum(1 for con in constraints if con[2] == 0)
        n_equality = sum(1 for con in constraints if con[2] == 4)

        # nonzeros by (constraint, objective) component.  Keys are
        # component id(), Values are tuples with three sets:
        # (linear_vars, nonlinear_vars, nonzeros) [where nonzeros is the
        # union of linear andd nonlinear].  Note that variables can
        # appear in both linear and nonlinear sets.
        nz_by_comp = {}

        # We need to categorize the named subexpressions first so that
        # we know their linear / nonlinear vars when we encounter them
        # in constraints / objectives
        subexpressions = map(self.subexpression_cache.__getitem__,
                             self.subexpression_order)
        self._categorize_vars(subexpressions, nz_by_comp)
        n_subexpressions = self._count_subexpression_occurances()

        n_objs = len(objectives)
        n_nonlinear_objs = sum(1 for obj in objectives if obj[1].nonlinear)
        obj_vars_linear, obj_vars_nonlinear, obj_nnz_by_var \
            = self._categorize_vars(objectives, nz_by_comp)

        n_cons = len(constraints)
        n_nonlinear_cons = len(nonlinear_cons)
        con_vars_linear, con_vars_nonlinear, con_nnz_by_var \
            = self._categorize_vars(constraints, nz_by_comp)

        n_lcons = 0 # We do not yet support logical constraints

        obj_vars = obj_vars_linear.union(obj_vars_nonlinear)
        con_vars = con_vars_linear.union(con_vars_nonlinear)
        all_vars = con_vars.union(obj_vars)
        n_vars = len(all_vars)

        binary_vars = set(
            _id for _id in all_vars if self.var_map[_id].is_binary()
        )
        integer_vars = set(
            _id for _id in all_vars - binary_vars
            if self.var_map[_id].is_integer()
        )
        discrete_vars = binary_vars.union(integer_vars)
        continuous_vars = all_vars - discrete_vars

        nonlinear_vars = con_vars_nonlinear.union(obj_vars_nonlinear)
        linear_only_vars = con_vars_linear.union(obj_vars_linear) \
                           - nonlinear_vars

        column_order = {_id: i for i, _id in enumerate(self.var_map)}
        variables = []
        #
        both_vars_nonlinear = con_vars_nonlinear.intersection(
            obj_vars_nonlinear)
        variables.extend(sorted(
            both_vars_nonlinear.intersection(continuous_vars),
            key=column_order.__getitem__))
        variables.extend(sorted(
            both_vars_nonlinear.intersection(discrete_vars),
            key=column_order.__getitem__))
        #
        con_only_nonlinear_vars = con_vars_nonlinear - both_vars_nonlinear
        variables.extend(sorted(
            con_only_nonlinear_vars.intersection(continuous_vars),
            key=column_order.__getitem__))
        variables.extend(sorted(
            con_only_nonlinear_vars.intersection(discrete_vars),
            key=column_order.__getitem__))
        #
        obj_only_nonlinear_vars = obj_vars_nonlinear - both_vars_nonlinear
        variables.extend(sorted(
            obj_only_nonlinear_vars.intersection(continuous_vars),
            key=column_order.__getitem__))
        variables.extend(sorted(
            obj_only_nonlinear_vars.intersection(discrete_vars),
            key=column_order.__getitem__))
        #
        variables.extend(sorted(
            linear_only_vars - discrete_vars,
            key=column_order.__getitem__))
        variables.extend(sorted(
            linear_only_vars.intersection(binary_vars),
            key=column_order.__getitem__))
        variables.extend(sorted(
            linear_only_vars.intersection(integer_vars),
            key=column_order.__getitem__))
        assert len(variables) == n_vars
        # Compute variable id to position (given the new ordering)
        self.var_idx = {_id: idx for idx, _id in enumerate(variables)}
        # Fill in the variable list
        for idx, _id in enumerate(variables):
            v = self.var_map[_id]
            variables[idx] = (v, _id, _RANGE_TYPE(*v.bounds))

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

        if symbolic_solver_labels:
            labeler = NameLabeler()
            row_labels = [labeler(info[0]) for info in constraints] \
                         + [labeler(info[0]) for info in objectives]
            row_comments = ['\t#%s' % lbl for lbl in row_labels]
            col_labels = [labeler(info[0]) for info in variables]
            col_comments = ['\t#%s' % lbl for lbl in col_labels]
            self.var_id_to_nl = {
                info[1]: '%d%s' % (var_idx, col_comments[var_idx])
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
                info[1]: str(var_idx) for var_idx, info in enumerate(variables)
            }

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
        for i, _id in enumerate(self.subexpression_order):
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
        lbl = ''
        for row_idx, info in enumerate(constraints):
            for _id in single_use_subexpressions.get(id(info[0]), ()):
                self._write_v_line(_id, row_idx)
            ostream.write('C%d%s\n' % (row_idx, row_comments[row_idx]))
            self._write_nl_expression(info[1], 0)

        #
        # "O" lines (objectives: nonlinear expression)
        #
        lbl = ''
        for obj_idx, info in enumerate(objectives):
            for _id in single_use_subexpressions.get(id(info[0]), ()):
                self._write_v_line(_id, n_cons + n_lcons + obj_idx)
            if symbolic_solver_labels:
                lbl = '\t#%s' % info[0].name
            sense = 0 if info[0].sense == minimize else 1
            ostream.write('O%d %d%s\n' % (obj_idx, sense, lbl))
            self._write_nl_expression(info[1])

        #
        # "d" lines (dual initialization)
        #

        #
        # "x" lines (variable initialization)
        #
        _init_lines = [
            '%d %r%s\n' % (var_idx, info[0].value, col_comments[var_idx])
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
        _include_lb = {0, 2, 4}
        _include_ub = {0, 1}
        ostream.writelines(
            str(info[2])
            + (' %r' % (info[0].lb - info[1].const)
               if info[2] in _include_lb else '')
            + (' %r' % (info[0].ub - info[1].const)
               if info[2] in _include_ub else '')
            + row_comments[row_idx]
            + '\n'
            for row_idx, info in enumerate(constraints)
        )

        #
        # "b" lines (variable bounds)
        #
        ostream.write('b%s\n' % (
            "\t#%d bounds (on variables)" % len(variables)
            if symbolic_solver_labels else '',
        ))
        ostream.writelines(
            str(info[2])
            + (' %r' % info[0].lb if info[2] in _include_lb else '')
            + (' %r' % info[0].ub if info[2] in _include_ub else '')
            + col_comments[var_idx]
            + '\n'
            for var_idx, info in enumerate(variables)
        )

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
            ktot += con_nnz_by_var[info[1]]
            ostream.write("%d\n" % ktot)

        #
        # "J" lines (non-empty terms in the Jacobian)
        #
        lbl = ''
        for row_idx, info in enumerate(constraints):
            nz = nz_by_comp[id(info[0])][2]
            ostream.write('J%d %d%s\n'
                          % (row_idx, len(nz), row_comments[row_idx]))
            linear = info[1].linear or {}
            for entry in sorted((self.var_idx[_id], linear.get(_id, 0))
                                for _id in nz):
                ostream.write('%d %r\n' % entry)

        #
        # "G" lines (non-empty terms in the Objective)
        #
        lbl = ''
        for obj_idx, info in enumerate(objectives):
            if symbolic_solver_labels:
                lbl = '\t#%s' % info[0].name
            nz = nz_by_comp[id(info[0])][2]
            linear = info[1].linear or {}
            ostream.write('G%d %d%s\n' % (obj_idx, len(nz), lbl))
            for entry in sorted((self.var_idx[_id], linear.get(_id, 0))
                                for _id in nz):
                ostream.write('%d %r\n' % entry)

        return symbol_map, sorted(amplfunc_libraries)


    def _categorize_vars(self, comp_list, nz_by_comp):
        """Categorize compiled expression vars into linear and nonlinear

        This routine takes an iterable of compiled component expression
        infos and returns the sets of variables appearing lineary and
        nonlinearly in those components.

        This routine has a number of side effects:

          - the ``nnz_by_var`` counter is updated with the count of
            components that each var appears in.

          - the ``nz_by_comp`` dict is updates to contain a tuple of
            three sets, containing the id() of variables appearing
            linearly, nonlinearly, and the union of those two sets for
            each component in the ``comp_list``

          - the expr_info (the second element in each tuple in
            ``comp_list``) is "compiled": the ``linear`` attribute is
            converted from a list of coef, var_id terms (potentially with
            duplicate entries) into a dict that maps var id to
            coefficients

        """
        all_linear_vars = set()
        all_nonlinear_vars = set()
        nnz_by_var = Counter()

        for comp_info in comp_list:
            expr_info = comp_info[1]
            nz_by_comp[id(comp_info[0])] \
                = linear_vars, nonlinear_vars, nz = set(), set(), set()
            if expr_info.linear:
                if expr_info.linear.__class__ is not dict:
                    coefs = {}
                    for c, v in expr_info.linear:
                        if v in coefs:
                            coefs[v] += c
                        elif c:
                            coefs[v] = c
                    expr_info.linear = coefs
                linear_vars.update(expr_info.linear)
            if expr_info.nonlinear:
                for _id in expr_info.nonlinear[1]:
                    if _id in nz:
                       continue
                    nz.add(_id)
                    if _id in nz_by_comp:
                        # This is a defined variable (Expression node)
                        #
                        # ... as this subexpression appears in the
                        # "nonlinear" expression tree, all variables in
                        # it are nonlinear in the context of this
                        # expression
                        nonlinear_vars.update(nz_by_comp[_id][0]) # linear
                        nonlinear_vars.update(nz_by_comp[_id][1]) # nonlinear
                    else:
                        # Regular variable
                        nonlinear_vars.add(_id)
            nz.clear()
            nz.update(linear_vars)
            nz.update(nonlinear_vars)
            nnz_by_var.update(nz)
            all_linear_vars.update(linear_vars)
            all_nonlinear_vars.update(nonlinear_vars)
        return (
            all_linear_vars - all_nonlinear_vars,
            all_nonlinear_vars,
            nnz_by_var,
        )

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

    def _write_nl_expression(self, repn, const_=None):
        if const_ is None:
            const_ = repn.const
        if repn.nonlinear and repn.nonlinear[0] != 'n0\n':
            nl, args = repn.nonlinear
            if const_:
                # Add the constant to the NL expression.  AMPL adds the
                # constant as the second argument, so we will too.
                nl = self.template.binary_sum + nl + (
                    self.template.const % const_)
            self.ostream.write(
                nl % tuple(map(self.var_id_to_nl.__getitem__, args))
            )
        else:
            self.ostream.write(self.template.const % const_)

    def _write_v_line(self, expr_id, k):
        ostream = self.ostream
        info = self.subexpression_cache[expr_id]
        if self.symbolic_solver_labels and info[0].__class__ is not AMPLRepn:
            lbl = '\t#%s' % info[0].name
        else:
            lbl = ''
        self.var_id_to_nl[expr_id] = "%d%s" % (self.next_V_line_id, lbl)
        linear = info[1].linear or {}
        ostream.write('V%d %d %d%s\n' %
                      (self.next_V_line_id, len(linear), k, lbl))
        for entry in sorted(map(
                lambda _id: (self.var_idx[_id], linear.get(_id, 0)),
                linear)):
            ostream.write('%d %r\n' % entry)
        self._write_nl_expression(info[1])
        self.next_V_line_id += 1


class AMPLRepn(object):
    __slots__ = ('nl', 'const', 'linear', 'nonlinear')

    def __init__(self, linear, nonlinear):
        self.nl = None
        self.const = 0
        self.linear = linear
        self.nonlinear = nonlinear

    def to_nl_node(self, visitor):
        if self.nl is not None:
            return self.nl
        nl = ''
        args = []
        nterms = 0
        if self.const:
            nterms += 1
            nl += visitor.template.const % self.const
        if self.linear:
            for c, v in self.linear:
                if c != 1:
                    nl += visitor.template.product + (
                        visitor.template.const % c)
                nl += visitor.template.var
            args.extend(map(itemgetter(1), self.linear))
            nterms += len(self.linear)
        if self.nonlinear:
            if self.nonlinear.__class__ is list:
                nterms += len(self.nonlinear)
                tmp_nl, tmp_args = zip(*self.nonlinear)
                nl += ''.join(tmp_nl)
                deque(map(args.extend, tmp_args), maxlen=0)
            else:
                nterms += 1
                nl += self.nonlinear[0]
                args.extend(self.nonlinear[1])

        if nterms > 2:
            self.nl = (visitor.template.nary_sum % nterms) + nl, tuple(args)
        elif nterms == 2:
            self.nl = visitor.template.binary_sum + nl, tuple(args)
        elif nterms == 1:
            self.nl = nl, tuple(args)
        else: # nterms == 0
            self.nl = visitor.template.const % 0, ()
        return self.nl

    def accumulate(self, other):
        self.const += other.const
        if other.linear:
            self.linear.extend(other.linear)
        if other.nonlinear:
            if other.nonlinear.__class__ is list:
                self.nonlinear.extend(other.nonlinear)
            else:
                self.nonlinear.append(other.nonlinear)

    def distribute_multiplicand(self, visitor, mult):
        if self.nl:
            if not mult:
                return AMPLRepn(None, None)
            elif mult == 1:
                return AMPLRepn(None, self.nl)
            return AMPLRepn(
                None,
                ( visitor.template.product \
                  + (visitor.template.const % mult) + self.nl[0],
                  self.nl[1] )
            )
        ans = AMPLRepn(None, None)
        if self.nonlinear:
            if type(self.nonlinear) is tuple:
                nl, args = self.nonlinear
            else:
                nl, args = AMPLRepn(None, self.nonlinear).to_nl_node(visitor)
            if mult == -1:
                ans.nonlinear = (
                    visitor.template.negation + nl,
                    args
                )
            else:
                ans.nonlinear = (
                    visitor.template.product \
                    + (visitor.template.const % mult) + nl,
                    args,
                )
        ans.const = mult * self.const
        if self.linear is not None:
            ans.linear = [(mult * c, v) for c, v in self.linear]
        return ans

    def distribute_divisor(self, visitor, div):
        ans = AMPLRepn(None, None)
        if self.nonlinear:
            if type(self.nonlinear) is tuple:
                nl, args = self.nonlinear
            else:
                nl, args = AMPLRepn(None, self.nonlinear).to_nl_node(visitor)
            ans.nonlinear = (
                visitor.template.division + nl + (visitor.template.const % div),
                args,
            )
        ans.const = self.const / div
        if self.linear is not None:
            ans.linear = [(c / div, v) for c, v in self.linear]
        return ans


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
        'ceil':   'o13\t#ceil\n',
        'floor':  'o14\t#floor\n',
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

    _create_strict_inequality_map(vars())

def _strip_template_comments(vars_, base_):
    vars_['unary'] = {k: v[:v.find('\t#')]+'\n'
             for k, v in base_.unary.items()}
    for k, v in base_.__dict__.items():
        if type(v) is str and '\t#' in v:
            vars_[k] = '\n'.join(l[:l.find('\t#')] for l in v.split('\n'))

# The "standard" text mode template is the debugging template with the
# comments removed
class text_nl_template(text_nl_debug_template):
    _strip_template_comments(vars(), text_nl_debug_template)
    _create_strict_inequality_map(vars())


def node_result_to_amplrepn(visitor, data):
    if data[0] is _GENERAL:
        ans = data[1]
        if ans.nonlinear.__class__ is list:
            if ans.nonlinear:
                ans.nonlinear = AMPLRepn(
                    None, ans.nonlinear).to_nl_node(visitor)
            else:
                ans.nonlinear = None
    elif data[0] is _MONOMIAL:
        ans = AMPLRepn([], None)
        if data[1]:
            ans.linear.append(data[1:])
    elif data[0] is _CONSTANT:
        ans = AMPLRepn(None, None)
        ans.const = data[1]
    else:
        raise DeveloperError("unknown result type")
    return ans

def handle_negation_node(visitor, node, arg1):
    if arg1[0] is _MONOMIAL:
        return (_MONOMIAL, -1*arg1[1], arg1[2])
    elif arg1[0] is _GENERAL:
        return (_GENERAL, arg1[1].distribute_multiplicand(visitor, -1))
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
            return (_MONOMIAL, mult*arg2[1], arg2[2])
        elif arg2[0] is _GENERAL:
            return (_GENERAL, arg2[1].distribute_multiplicand(visitor, mult))
        elif arg2[0] is _CONSTANT:
            return (_CONSTANT, mult*arg2[1])
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    nl2 = node_result_to_amplrepn(visitor, arg2).to_nl_node(visitor)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.product + nl1[0] + nl2[0], nl1[1] + nl2[1],
    )))

def handle_division_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        div = arg2[1]
        if div == 1:
            return arg1
        if arg1[0] is _MONOMIAL:
            return (_MONOMIAL, arg1[1]/div, arg1[2])
        elif arg1[0] is _GENERAL:
            return (_GENERAL, arg1[1].distribute_divisor(visitor, div))
        elif arg1[0] is _CONSTANT:
            return (_CONSTANT, arg1[1]/div)
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    nl2 = node_result_to_amplrepn(visitor, arg2).to_nl_node(visitor)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.division + nl1[0] + nl2[0], nl1[1] + nl2[1],
    )))

def handle_pow_node(visitor, node, arg1, arg2):
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    nl2 = node_result_to_amplrepn(visitor, arg2).to_nl_node(visitor)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.pow + nl1[0] + nl2[0], nl1[1] + nl2[1],
    )))

def handle_abs_node(visitor, node, arg1):
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.abs + nl1[0], nl1[1],
    )))

def handle_unary_node(visitor, node, arg1):
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.unary[node.name] + nl1[0],
        nl1[1],
    )))

def handle_exprif_node(visitor, node, arg1, arg2, arg3):
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    nl2 = node_result_to_amplrepn(visitor, arg2).to_nl_node(visitor)
    nl3 = node_result_to_amplrepn(visitor, arg3).to_nl_node(visitor)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.exprif + nl1[0] + nl2[0] + nl3[0],
        nl1[1] + nl2[1] + nl3[1],
    )))

def handle_equality_node(visitor, node, arg1, arg2):
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    nl2 = node_result_to_amplrepn(visitor, arg2).to_nl_node(visitor)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.equality + nl1[0] + nl2[0],
        nl1[1] + nl2[1],
    )))

def handle_inequality_node(visitor, node, arg1, arg2):
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    nl2 = node_result_to_amplrepn(visitor, arg2).to_nl_node(visitor)
    op = visitor.template.strict_inequality_map[node.strict]
    return (_GENERAL, AMPLRepn(None, (
        op + nl1[0] + nl2[0],
        nl1[1] + nl2[1],
    )))

def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    nl1 = node_result_to_amplrepn(visitor, arg1).to_nl_node(visitor)
    nl2 = node_result_to_amplrepn(visitor, arg2).to_nl_node(visitor)
    nl3 = node_result_to_amplrepn(visitor, arg3).to_nl_node(visitor)
    op = visitor.template.strict_inequality_map[node.strict]
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.and_expr
        + op[0] + nl1[0] + nl2[0]
        + op[1] + nl2[0] + nl3[0],
        nl1[1] + nl2[1] + nl2[1] + nl3[1],
    )))

def handle_expression_node(visitor, node, arg1):
    _id = id(node)
    # Note that while named subexpressions ('defined variables' in the
    # ASL NL file vernacular) look like variables, they are not allowed
    # to appear in the 'linear' portion of a constraint / objective
    # definition.  We will return this as a "var" template, but
    # wrapped in the nonlinear portion of the expression tree.
    repn = node_result_to_amplrepn(visitor, arg1)
    # When converting this shared subexpression to a (nonlinear) node,
    # we want to just reference this subexpression:
    repn.nl = (visitor.template.var, (_id,))

    # A local copy of the expression source list.  This will be updated
    # later if the same Expression node is encountered in another
    # expression tree.
    expression_source = list(visitor.active_expression_source)

    if repn.linear:
        if repn.nonlinear:
            # If this expession has both linear and nonlinear
            # components, we will follow the ASL convention and break
            # the named subexpression into two named subexpressions: one
            # that is only the nonlinear component and one that has the
            # const/linear component (and references the first).  This
            # will allow us to propagate linear coefficients up from
            # named subexpressions when appropriate.
            subid = id(repn)
            sub_repn = AMPLRepn(None, repn.nonlinear)
            sub_repn.nl = (visitor.template.var, (subid,))
            # See below for the meaning of this tuple
            visitor.subexpression_cache[subid] = (
                repn, sub_repn, list(expression_source),
            )
            repn.nonlinear = sub_repn.nl
            # It is important that the NL subexpression comes before the
            # main named expression:
            visitor.subexpression_order.append(subid)
        elif (not repn.const and len(repn.linear) == 1
              and repn.linear[0][0] == 1):
            # This Expression holds only a variable (multiplied by 1).
            # Do not emit this as a named variable and instead just
            # inject the variable where this expression is used.
            repn.nl = None
            expression_source[2] = True
    elif not repn.nonlinear:
        # This Expression holds only a constant.  Do not emit this as a
        # named variable and instead just inject the constant where this
        # expression is used.
        repn.nl = None
        expression_source[2] = True

    visitor.subexpression_cache[_id] = (
        # 0: the "component" that generated this expression ID
        node,
        # 1: the common subexpression
        repn,
        # 2: the (single) component that uses this subexpression.  This
        # is a 3-tuple [con_id, obj_id, substitute_expression].  If the
        # expression is used by 1 constraint / objective, then the id is
        # set to 0.  If it is not used by any, then it is None.
        # substitue_expression is a bool indicating id this named
        # subexpression tree should be directly substituted into any
        # expression tree that references this node (i.e., do NOT emit
        # the V line).
        expression_source,
    )
    visitor.subexpression_order.append(_id)
    return (_GENERAL, visitor.subexpression_cache[_id][1])

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
    _amplrepn = lambda arg: \
                node_result_to_amplrepn(visitor, arg).to_nl_node(visitor)
    nl, arg_tuples = zip(*map(_amplrepn, args))
    all_args = []
    deque(map(all_args.extend, arg_tuples), maxlen=0)
    return (_GENERAL, AMPLRepn(None, (
        (visitor.template.external_fcn % (
            visitor.external_functions[func][0], len(args))) + ''.join(nl),
        tuple(all_args)
    )))


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
    _GeneralExpressionData: handle_expression_node,
    ScalarExpression: handle_expression_node,
    kernel.expression.expression: handle_expression_node,
    ExternalFunctionExpression: handle_external_function_node,
    # These are handled explicitly in beforeChild():
    # LinearExpression: handle_linear_expression,
    # SumExpression: handle_sum_expression,
    # MonomialTermExpression: handle_monomial_term,
}


class AMPLRepnVisitor(StreamBasedExpressionVisitor):

    def __init__(self, template, subexpression_cache, subexpression_order,
                 external_functions, var_map):
        super().__init__()
        self.template = template
        self.subexpression_cache = subexpression_cache
        self.subexpression_order = subexpression_order
        self.external_functions = external_functions
        self.active_expression_source = None
        self.var_map = var_map
        self.value_cache = {}

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
        child_type = child.__class__
        if child_type in native_types:
            return False, (_CONSTANT, child)
        if not child.is_expression_type():
            if child.is_fixed():
                return False, (_CONSTANT, child())
            else:
                _id = id(child)
                if _id not in self.var_map:
                    self.var_map[_id] = child
                return False, (_MONOMIAL, 1, _id)
        if not child.is_potentially_variable():
            _id = id(child)
            if _id in self.value_cache:
                child = self.value_cache[_id]
            else:
                child = self.value_cache[_id] = child()
            return False, (_CONSTANT, child)

        #
        # The following are performance optimizations for common
        # situations (Monomial terms and Linear expressions)
        #

        if child_type is MonomialTermExpression:
            arg1, arg2 = child._args_
            if arg1.__class__ not in native_types:
                _id = id(arg1)
                if _id in self.value_cache:
                    arg1 = self.value_cache[_id]
                else:
                    arg1 = self.value_cache[_id] = value(arg1)
            if arg2.is_fixed():
                return False, (_CONSTANT, arg1 * arg2())
            else:
                _id = id(arg2)
                if _id not in self.var_map:
                    self.var_map[_id] = arg2
                return False, (_MONOMIAL, arg1, _id)

        if child_type is LinearExpression:
            # Because we are going to modify the LinearExpression in this
            # walker, we need to make a copy of the LinearExpression from
            # the original expression tree.
            data = AMPLRepn(list(zip(map(value, child.linear_coefs),
                                     map(id, child.linear_vars))),
                            None)
            data.const = child.constant
            return False, (_GENERAL, data)

        _id = id(child)
        if _id in self.subexpression_cache:
            cache = self.subexpression_cache[_id]
            cache[2][self.active_expression_source_idx] = 0
            return False, (_GENERAL, cache[1])

        return True, None

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if isinstance(node, SumExpressionBase):
            return node.args, AMPLRepn([], [])
        else:
            return node.args, []

    def acceptChildResult(self, node, data, child_result, child_idx):
        if data.__class__ is list:
            # General expression... cache the child result until the end
            data.append(child_result)
        else:
            # Sum Expression
            child_type = child_result[0]
            if child_type is _MONOMIAL:
                data.linear.append(child_result[1:])
            elif child_type is _CONSTANT:
                data.const += child_result[1]
            elif child_type is _GENERAL:
                data.accumulate(child_result[1])
        return data

    def exitNode(self, node, data):
        if data.__class__ is AMPLRepn:
            return (_GENERAL, data)
        #
        # General expressions...
        #
        if all(_[0] is _CONSTANT for _ in data):
            return (
                _CONSTANT, node._apply_operation(tuple(map(
                    itemgetter(1), data)))
            )
        return _operator_handles[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        ans = node_result_to_amplrepn(self, result)
        self.active_expression_source = None
        return ans
