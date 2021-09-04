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

from pyomo.common.config import ConfigBlock, ConfigValue, InEnum

from pyomo.core.expr.current import (
    NegationExpression, ProductExpression, DivisionExpression,
    PowExpression, AbsExpression, UnaryFunctionExpression,
    MonomialTermExpression, LinearExpression, SumExpressionBase,
    native_types,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.base import (
    Block, Objective, Constraint, Var, Param, Expression,
    SymbolMap, NameLabeler, SortComponents, minimize,
)
from pyomo.core.base.expression import Expression, _GeneralExpressionData
from pyomo.opt import WriterFactory

from pyomo.common.formatting import tostr

class _CONSTANT(object): pass
class _MONOMIAL(object): pass
class _GENERAL(object): pass

class FileDeterminism(enum.IntEnum):
    NONE = 0
    SORT_INDICES = 1
    SORT_SYMBOLS = 2

def identify_unrecognized_components(model, active=True, valid={}):
    pass


@WriterFactory.register('new_nl', 'Generate the corresponding AMPL NL file.')
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
        default=FileDeterminism.SORT_INDICES,
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
        self.config = NLWriter.CONFIG()

    def __call__(self, model, filename, solver_capability, io_options):
        if filename is None:
            filename = model.name + ".nl"
        with open(filename, 'w') as FILE:
            symbol_map = self.write(model, FILE, **io_options)
        return filename, symbol_map

    def write(self, model, ostream, **options):
        config = self.config(options)

        unknown = identify_unrecognized_components(model, active=True, valid={
            Block, Objective, Constraint, Var, Param, Expression,
            # TODO: Suffix, Piecewise, SOSConstraint, Complementarity
        })
        if unknown:
            raise ValueError(
                "The model (%s) contains the following active components "
                "that the NL writer does not know how to process:\n\t" %
                (model.name, "\n\t".join("%s:\n\t\t%s" % (
                    k, "\n\t\t".join(v)) for k, v in unknown.items())))

        sorter = SortComponents.unsorted
        if config.file_determinism >= FileDeterminism.SORT_INDICES:
            sorter = sorter | SortComponents.indices
            if config.file_determinism  >= FileDeterminism.SORT_SYMBOLS:
                sorter = sorter | SortComponents.alphabetical

        symbolic_solver_labels = config.symbolic_solver_labels

        generate_amplrepn = AMPLRepnVisitor(
            text_nl_debug_template if symbolic_solver_labels else
            text_nl_template
        )

        range_type = {
            # has_lb, has_ub, lb == ub
            (True, True, False): 0,   # L <= c <= U
            (False, True, False): 1,  #      c <= U
            (True, False, False): 2,  # L <= c     
            (False, False, True): 3,  # -inf <= c <= inf
            (True, True, True): 4,    # L == c == U
            # complementarity: 5,
        }

        var_map = {}
        next_var_num = 0

        if config.file_determinism > FileDeterminism.NONE:
            # We will pre-gather the variables so that their order
            # matches the file_determinism flag.
            #
            # This is a little cumbersome, but is imlemented this wat
            # for consistency with the original NL writer.  Note that
            # Vars that appear twice (e.g., through a Reference) will be
            # sorted with the LAST occurance.
            var_map = {id(var): (var, i) for i, var in enumerate(
                model.component_data_objects(
                    Var, descend_into=True, sort=sorter))}
            next_var_num = max(info[1] for info in var_map.values()) + 1

        #
        # Tabulate the model expressions
        #

        objectives = [
            (obj,
             generate_amplrepn.walk_expression(obj.expr),
         ) for obj in model.component_data_objects(
             Objective, active=True, descend_into=True, sort=sorter)]

        constraints = [(
            con,
            generate_amplrepn.walk_expression(con.body),
            range_type[con.has_lb(), con.has_ub(), con.lb == con.ub],
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
        # component id(), Values are sets of var id()
        nz_by_comp = {}

        n_objs = len(objectives)
        n_nonlinear_objs = sum(1 for obj in objectives if obj[1].nonlinear)

        obj_vars_linear = set()
        obj_vars_nonlinear = set()
        obj_nnz_by_var = {}
        next_var_num = self._categorize_vars(
            var_map, next_var_num, objectives, obj_vars_linear,
            obj_vars_nonlinear, obj_nnz_by_var, nz_by_comp)

        n_cons = len(constraints)
        n_nonlinear_cons = len(nonlinear_cons)

        con_vars_linear = set()
        con_vars_nonlinear = set()
        con_nnz_by_var = {}
        next_var_num = self._categorize_vars(
            var_map, next_var_num, constraints, con_vars_linear,
            con_vars_nonlinear, con_nnz_by_var, nz_by_comp)

        obj_vars = obj_vars_linear.union(obj_vars_nonlinear)
        con_vars = con_vars_linear.union(con_vars_nonlinear)
        all_vars = con_vars.union(obj_vars)

        binary_vars = set(
            _id for _id in all_vars if var_map[_id][0].is_binary()
        )
        integer_vars = set(
            _id for _id in all_vars - binary_vars
            if var_map[_id][0].is_integer()
        )
        discrete_vars = binary_vars.union(integer_vars)
        continuous_vars = all_vars - discrete_vars

        nonlinear_vars = con_vars_nonlinear.union(obj_vars_nonlinear)
        linear_only_vars = con_vars_linear.union(obj_vars_linear) \
                           - nonlinear_vars

        column_order = lambda _id: var_map[_id][1]
        variables = []
        #
        both_vars_nonlinear = con_vars_nonlinear.intersection(
            obj_vars_nonlinear)
        variables.extend(sorted(
            both_vars_nonlinear.intersection(continuous_vars),
            key=column_order))
        variables.extend(sorted(
            both_vars_nonlinear.intersection(discrete_vars),
            key=column_order))
        #
        con_only_nonlinear_vars = con_vars_nonlinear - both_vars_nonlinear
        variables.extend(sorted(
            con_only_nonlinear_vars.intersection(continuous_vars),
            key=column_order))
        variables.extend(sorted(
            con_only_nonlinear_vars.intersection(discrete_vars),
            key=column_order))
        #
        obj_only_nonlinear_vars = obj_vars_nonlinear - both_vars_nonlinear
        variables.extend(sorted(
            obj_only_nonlinear_vars.intersection(continuous_vars),
            key=column_order))
        variables.extend(sorted(
            obj_only_nonlinear_vars.intersection(discrete_vars),
            key=column_order))
        #
        variables.extend(sorted(
            linear_only_vars - discrete_vars,
            key=column_order))
        variables.extend(sorted(
            linear_only_vars.intersection(binary_vars),
            key=column_order))
        variables.extend(sorted(
            linear_only_vars.intersection(integer_vars),
            key=column_order))
        assert len(variables) == len(all_vars)
        for var_idx, _id in enumerate(variables):
            v = var_map[_id][0]
            bnds = v.bounds
            variables[var_idx] = (v, _id, range_type[
                bnds[0] is not None,
                bnds[1] is not None,
                bnds[0] == bnds[1]])
        # Update the variable ID to reflect the new ordering
        var_map = {info[1]: (v, var_idx)
                   for var_idx, info in enumerate(variables)}

        # Now that the row/column ordering is resolved, create the labels
        if symbolic_solver_labels:
            labeler = NameLabeler()
            row_labels = [labeler(info[0]) for info in constraints]
            row_comments = ['\t#%s' % lbl for lbl in row_labels]
            col_labels = [labeler(info[0]) for info in variables]
            col_comments = ['\t#%s' % lbl for lbl in col_labels]
            var_id_to_nl = {
                info[1]: '%d%s' % (var_idx, col_comments[var_idx])
                for var_idx, info in enumerate(variables)
            }
        else:
            row_labels = row_comments = [''] * n_cons
            col_labels = col_comments = [''] * len(variables)
            var_id_to_nl = {
                info[1]: str(var_idx) for var_idx, info in enumerate(variables)
            }
            

        #
        # Print Header
        #
        # LINE 1
        #
        ostream.write("g3 1 1 0\t# problem %s\n".format(model.name))
        #
        # LINE 2
        #
        ostream.write(
            " %d %d %d %d %d \t"
            "# vars, constraints, objectives, ranges, eqns\n"
            % ( len(all_vars),
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
            % ( 0, # len(self.external_byFcn),
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
            % ( max(len(lbl) for lbl in row_labels),
                max(len(lbl) for lbl in col_labels),
            ))
        #
        # LINE 10
        #
        ostream.write(" 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\n")

        #
        # "F" lines (external function definitions)
        #

        #
        # "S" lines (suffixes)
        #

        #
        # "C" lines (constraints: nonlinear expression)
        #
        lbl = ''
        for row_idx, info in enumerate(constraints):
            ostream.write('C%d%s\n' % (row_idx, row_comments[row_idx]))
            if info[1].nonlinear:
                args = tuple(var_id_to_nl[_id] for _id in info[1].nonlinear[1])
                ostream.write(info[1].nonlinear[0] % args)
            else:
                ostream.write('n0\n')

        #
        # "O" lines (objectives: nonlinear expression)
        #
        lbl = ''
        for obj_idx, info in enumerate(objectives):
            if symbolic_solver_labels:
                lbl = '\t#%s' % info[0].name
            sense = 0 if info[0].sense == minimize else 1
            ostream.write('O%d %d%s\n' % (obj_idx, sense, lbl))
            if info[1].nonlinear:
                args = tuple(var_id_to_nl[_id] for _id in info[1].nonlinear[1])
                ostream.write(info[1].nonlinear[0] % args)
            else:
                ostream.write('n0\n')

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
            nz = nz_by_comp[id(info[0])]
            ostream.write('J%d %d%s\n'
                          % (row_idx, len(nz), row_comments[row_idx]))
            linear = info[1].linear or {}
            for entry in sorted((var_map[_id][1], linear.get(_id, 0))
                                for _id in nz):
                ostream.write('%d %r\n' % entry)

        #
        # "G" lines (non-empty terms in the Objective)
        #
        lbl = ''
        for obj_idx, info in enumerate(objectives):
            if symbolic_solver_labels:
                lbl = '\t#%s' % info[0].name
            nz = nz_by_comp[id(info[0])]
            linear = info[1].linear or {}
            ostream.write('G%d %d%s\n' % (obj_idx, len(nz), lbl))
            for entry in sorted((var_map[_id][1], linear.get(_id, 0))
                                for _id in nz):
                ostream.write('%d %r\n' % entry)
        

    def _categorize_vars(self, var_map, next_var_num, comp_list,
                         comp_linear_vars, comp_nonlinear_vars,
                         nnz_by_var, nz_by_comp):
        for comp_info in comp_list:
            expr_info = comp_info[1]
            nz_by_comp[id(comp_info[0])] = comp_nz = set()
            if expr_info.linear:
                coefs, vars_ = expr_info.collect_linear()
                for _id, v in vars_:
                    if _id not in var_map:
                        var_map[_id] = (v, next_var_num)
                        next_var_num += 1
                    expr_info.linear = coefs
                    comp_linear_vars.add(_id)
                    comp_nz.add(_id)
            if expr_info.nonlinear:
                args = expr_info.nonlinear[1]
                ids = tuple(id(v) for v in args)
                for v, _id in zip(args, ids):
                    if _id not in var_map:
                        var_map[_id] = (v, next_var_num)
                        next_var_num += 1
                    comp_nonlinear_vars.add(_id)
                    comp_nz.add(_id)
                expr_info.nonlinear = (expr_info.nonlinear[0], ids)
            for _id in comp_nz:
                if _id in nnz_by_var:
                    nnz_by_var[_id] += 1
                else:
                    nnz_by_var[_id] = 1
        return next_var_num


class AMPLRepn(object):
    __slots__ = ('const', 'linear', 'nonlinear')

    def __init__(self, linear, nonlinear):
        self.const = 0
        self.linear = linear
        self.nonlinear = nonlinear

    def to_nl_node(self, visitor):
        nl = ''
        args = []
        nterms = 0
        if self.const:
            nterms += 1
            nl += visitor.template.const % self.const
        if self.linear:
            coefs, vars_ = self.collect_linear()
            for _id, v in vars_:
                c = coefs[_id]
                if not c:
                    continue
                nterms += 1
                if c != 1:
                    nl += visitor.template.product
                    nl += visitor.template.const % c
                nl += visitor.template.var
            args.append(tuple(v[1] for v in vars_))
        if self.nonlinear:
            if self.nonlinear.__class__ is list:
                nterms += len(self.nonlinear)
                tmp_nl, tmp_args = zip(*self.nonlinear)
                nl += ''.join(tmp_nl)
                args.extend(tmp_args)
            else:
                nterms += 1
                nl += self.nonlinear[0]
                args.append(self.nonlinear[1])
        if nterms > 2:
            return (visitor.template.nary_sum % nterms) + nl, sum(args, ())
        elif nterms == 2:
            return visitor.template.binary_sum + nl, sum(args, ())
        elif nterms == 1:
            return nl, sum(args, ())
        else: # nterms == 0
            return visitor.template.const % 0, ()

    def accumulate(self, other):
        self.const += other.const
        if other.linear:
            self.linear.extend(other.linear)
        if other.nonlinear:
            if other.nonlinear.__class__ is list:
                self.nonlinear.extend(other.nonlinear)
            else:
                self.nonlinear.append(other.nonlinear)

    def collect_linear(self):
        coef = {}
        vars_ = []
        for c, v in self.linear:
            _id = id(v)
            if _id in coef:
                coef[_id] += c
            else:
                coef[_id] = c
                vars_.append((_id, v))
        return coef, vars_

    def distribute_multiplicand(self, visitor, mult):
        if mult == 1:
            return self
        ans = AMPLRepn(None, self.nonlinear)
        if self.nonlinear:
            nl, args = ans.to_nl_node(visitor)
            ans.nonlinear = (
                (visitor.template.product + (visitor.template.const % mult)
                 if mult != -1 else visitor.template.negation) + nl,
                args,
            )
        ans.const = mult * self.const
        if self.linear is not None:
            ans.linear = [(mult * c, v) for c, v in self.linear]
        return ans

    def distribute_divisor(self, visitor, div):
        if div == 1:
            return self
        ans = AMPLRepn(None, self.nonlinear)
        if self.nonlinear:
            nl, args = ans.to_nl_node(visitor)
            ans.nonlinear = (
                visitor.template.division + nl + (visitor.template.const % div),
                args,
            )
        ans.const = self.const / div
        if self.linear is not None:
            ans.linear = [(c / div, v) for c, v in self.linear]
        return ans


class text_nl_debug_template(object):
    unary = {
        'log':    'o43 # log\n',
        'log10':  'o42 # log10\n',
        'sin':    'o41 # sin\n',
        'cos':    'o46 # cos\n',
        'tan':    'o38 # tan\n',
        'sinh':   'o40 # sinh\n',
        'cosh':   'o45 # cosh\n',
        'tanh':   'o37 # tanh\n',
        'asin':   'o51 # asin\n',
        'acos':   'o53 # acos\n',
        'atan':   'o49 # atan\n',
        'exp':    'o44 # exp\n',
        'sqrt':   'o39 # sqrt\n',
        'asinh':  'o50 # asinh\n',
        'acosh':  'o52 # acosh\n',
        'atanh':  'o47 # atanh\n',
        'ceil':   'o13 # ceil\n',
        'floor':  'o14 # floor\n',
    }

    binary_sum = 'o0 # +\n'
    product = 'o2 # *\n'
    division = 'o3 # /\n'
    pow = 'o5 # pow\n'
    abs = 'o15 # abs\n'
    negation = 'o16 # -\n'
    nary_sum = 'o54 # sumlist\n%d # (n)\n'
    var = 'v%s\n'
    const = 'n%r\n'


# The "standard" text mode template is the debugging template with the
# comments removed
class text_nl_template(text_nl_debug_template):
    unary = {k: v[:v.find(' #')]+'\n'
             for k, v in text_nl_debug_template.unary.items()}
    for k, v in text_nl_debug_template.__dict__.items():
        if type(v) is str and ' #' in v:
            vars()[k] = '\n'.join(l[:l.find(' #')] for l in v.split('\n'))+'\n'


def to_nl_node(visitor, data):
    if data[0] is _GENERAL:
        ans = data[1]
    elif data[0] is _CONSTANT:
        ans = AMPLRepn(None, None)
        ans.const = data[1]
    else: # _MONOMIAL
        ans = AMPLRepn([], None)
        if data[1]:
            ans.linear.append(data[1:])
    return ans.to_nl_node(visitor)

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
        if arg2[0] is _MONOMIAL:
            return (_MONOMIAL, arg1[1]*arg2[1], arg2[2])
        elif arg2[0] is _GENERAL:
            return (_GENERAL, arg2[1].distribute_multiplicand(visitor, arg1[1]))
        elif arg2[0] is _CONSTANT:
            return (_CONSTANT, arg1[1]*arg2[1])
    nl1 = to_nl_node(visitor, arg1)
    nl2 = to_nl_node(visitor, arg2)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.product + nl1[0] + nl2[0], nl1[1] + nl2[1],
    )))

def handle_division_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        div = arg2[1]
        if arg1[0] is _MONOMIAL:
            return (_MONOMIAL, (arg1[1]/div, arg1[2])
            )
        elif arg1[0] is _GENERAL:
            return (_GENERAL, arg1[1].distribute_divisor(visitor, arg2[1]))
        elif arg1[0] is _CONSTANT:
            return (_CONSTANT, arg1[1]/arg2[1])
    nl1 = to_nl_node(visitor, arg1)
    nl2 = to_nl_node(visitor, arg2)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.division + nl1[0] + nl2[0], nl1[1] + nl2[1],
    )))

def handle_pow_node(visitor, node, arg1, arg2):
    nl1 = to_nl_node(visitor, arg1)
    nl2 = to_nl_node(visitor, arg2)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.pow + nl1[0] + nl2[0], nl1[1] + nl2[1],
    )))
    
def handle_abs_node(visitor, node, arg1):
    nl1 = to_nl_node(visitor, arg1)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.abs + nl1[0], nl1[1],
    )))

def handle_unary_node(visitor, node, arg1):
    nl1 = to_nl_node(visitor, arg1)
    return (_GENERAL, AMPLRepn(None, (
        visitor.template.unary[node.name] + nl1[0], nl1[1],
    )))
    
def handle_expression_node(visitor, node, arg1):
    visitor.expression_cache[id(node)] = arg1
    return arg1
    
_operator_handles = dict()
_operator_handles[NegationExpression] = handle_negation_node
_operator_handles[ProductExpression] = handle_product_node
_operator_handles[DivisionExpression] = handle_division_node
_operator_handles[PowExpression] = handle_pow_node
_operator_handles[AbsExpression] = handle_abs_node
_operator_handles[UnaryFunctionExpression] = handle_unary_node
_operator_handles[_GeneralExpressionData] = handle_expression_node
_operator_handles[Expression] = handle_expression_node
# TODO
#handler[ExternalFunctionExpression] = handle_external_function_expression
#handler[Expr_ifExpression] = handle_exprif_node
# These are handled explicitly in beforeChild():
#handler[LinearExpression] = handle_linear_expression
#handler[SumExpression] = handle_expression
#handler[MonomialTermExpression] = handle_expression


class AMPLRepnVisitor(StreamBasedExpressionVisitor):

    def __init__(self, template):
        super().__init__()
        self.template = template
        self.expression_cache = {}

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, None

    def beforeChild(self, node, child, child_idx):
        child_type = child.__class__
        if child_type in native_types:
            return False, (_CONSTANT, child)
        if not child.is_expression_type():
            if child.is_fixed():
                return False, (_CONSTANT, child())
            else:
                return False, (_MONOMIAL, 1, child)
        if not child.is_potentially_variable():
            return False, (_CONSTANT, child())

        #
        # The following are performance optimizations for common
        # situations (Monomial terms and Linear expressions)
        #

        if child_type is MonomialTermExpression:
            arg1, arg2 = child._args_
            if arg1.__class__ not in native_types:
                arg1 = value(arg1)
            if arg2.is_fixed():
                return False, (_CONSTANT, arg1 * arg2())
            else:
                return False, (_MONOMIAL, arg1, arg2)

        if child_type is LinearExpression:
            # Because we are going to modify the LinearExpression in this
            # walker, we need to make a copy of the LinearExpression from
            # the original expression tree.
            data = AMPLRepn([], [])
            data.fromLinearExpr(child)
            return False, (_GENERAL, data)

        if id(child) in self.expression_cache:
            return False, self.expression_cache[id(child)]

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
                _CONSTANT, node._apply_operation(tuple(arg[1] for arg in data))
            )
        return _operator_handles[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        result_type = result[0]
        if result_type is _GENERAL:
            ans = result[1]
        elif result_type is _MONOMIAL:
            ans = AMPLRepn([], None)
            if result[1]:
                ans.linear.append(result[1:])
        elif result_type is _CONSTANT:
            ans = AMPLRepn(None, None)
            ans.const = result[1]
        else:
            raise DeveloperError("unknown result type")
        if ans.nonlinear.__class__ is list:
            ans.nonlinear = AMPLRepn(None, ans.nonlinear).to_nl_node(self)
        return ans
