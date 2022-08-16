#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
import logging
import os
import sys
from collections import deque
from operator import itemgetter, attrgetter, setitem

from pyomo.common.backports import nullcontext
from pyomo.common.config import (
    ConfigBlock, ConfigValue, InEnum, add_docstring_list,
)
from pyomo.common.errors import DeveloperError
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer

from pyomo.core.expr.current import (
    NegationExpression, ProductExpression, DivisionExpression,
    PowExpression, AbsExpression, UnaryFunctionExpression,
    MonomialTermExpression, LinearExpression, SumExpression,
    EqualityExpression, InequalityExpression, RangedExpression,
    Expr_ifExpression, ExternalFunctionExpression,
    native_types, native_numeric_types, value,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.base import (
    Block, Objective, Constraint, Var, Param, Expression, ExternalFunction,
    Suffix, SOSConstraint, SymbolMap, NameLabeler, SortComponents, minimize,
)
from pyomo.core.base.block import SortComponents
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
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

logger=logging.getLogger(__name__)

# Feasibility tolerance for trivial (fixed) constraints
TOL = 1e-8
inf = float('inf')

class _CONSTANT(object): pass
class _MONOMIAL(object): pass
class _GENERAL(object): pass

# TODO: make a proper base class
class NLWriterInfo(object):
    """Return type for NLWriter.write()

    Attributes
    ----------
    variables: List[_VarData]

        The list of (unfixed) Pyomo model variables in the order written
        to the NL file

    constraints: List[_ConstraintData]

        The list of (active) Pyomo model constraints in the order written
        to the NL file

    objectives: List[_ObjectiveData]

        The list of (active) Pyomo model objectives in the order written
        to the NL file

    external_function_libraries: List[str]

        The list of paths to external function libraries referenced by
        the constraints / objectives written to the NL file

    row_labels: List[str]

        The list of string names for the constraints / objectives
        written to the NL file in the same order as
        :py:attr:`constraints` + :\p:attr:`objectives` and the generated
        .row file.

    column_labels: List[str]

        The list of string names for the variables written to the NL
        file in the same order as the :py:attr:`variables` and generated
        .col file.

    """
    def __init__(self, var, con, obj, extlib, row_lbl, col_lbl):
        self.variables = var
        self.constraints = con
        self.objectives = obj
        self.external_function_libraries = extlib
        self.row_labels = row_lbl
        self.column_labels = col_lbl


class FileDeterminism(enum.IntEnum):
    NONE = 0
    ORDERED = 1
    SORT_INDICES = 2
    SORT_SYMBOLS = 3


def _activate_nl_writer_version(n):
    """DEBUGGING TOOL to switch the "default" NL writer"""
    doc = WriterFactory.doc('nl')
    WriterFactory.unregister('nl')
    WriterFactory.register('nl', doc)(WriterFactory.get_class(f'nl_v{n}'))

def categorize_valid_components(
        model, active=True, sort=None, valid=set(), targets=set()):
    assert active in (True, None)
    unrecognized = {}
    component_map = {k: [] for k in targets}
    for block in model.block_data_objects(active=active,
                                          descend_into=True,
                                          sort=sort):
        local_ctypes = block.collect_ctypes(active=None, descend_into=False)
        for ctype in local_ctypes:
            if ctype in kernel.base._kernel_ctype_backmap:
                ctype = kernel.base._kernel_ctype_backmap[ctype]
            if ctype in targets:
                component_map[ctype].append(block)
                continue
            if ctype in valid:
                continue
            # TODO: we should rethink the definition of "active" for
            # Components that are not subclasses of ActiveComponent
            if not issubclass(ctype, ActiveComponent) and \
               not issubclass(ctype, kernel.base.ICategorizedObject):
                continue
            if ctype not in unrecognized:
                unrecognized[ctype] = []
            unrecognized[ctype].extend(
                block.component_data_objects(
                    ctype=ctype,
                    active=active,
                    descend_into=False,
                    sort=SortComponents.unsorted))
    return component_map, {k:v for k,v in unrecognized.items() if v}

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
        default=FileDeterminism.ORDERED,
        domain=InEnum(FileDeterminism),
        description='How much effort to ensure file is deterministic',
        doc="""
        How much effort do we want to put into ensuring the
        NL file is written deterministically for a Pyomo model:
            NONE (0) : None
            ORDERED (1): rely on underlying component ordering (default)
            SORT_INDICES (2) : sort keys of indexed components
            SORT_SYMBOLS (3) : sort keys AND sort names (over declaration order)
        """
    ))
    CONFIG.declare('symbolic_solver_labels', ConfigValue(
        default=False,
        domain=bool,
        description='Write the corresponding .row and .col files',
    ))
    CONFIG.declare('export_nonlinear_variables', ConfigValue(
        default=None,
        domain=list,
        description='Extra variables to include in NL file',
        doc="""
        List of variables to ensure are in the NL file (even if they
        don't appear in any constraints)."""
    ))
    CONFIG.declare('row_order', ConfigValue(
        default=None,
        description='Preferred constraint ordering',
        doc="""
        List of constraints in the order that they should appear in the
        NL file.  Note that this is only a suggestion, as the NL writer
        will move all nonlinear constraints before linear ones
        (preserving row_order within each group)."""
    ))
    CONFIG.declare('column_order', ConfigValue(
        default=None,
        description='Preferred variable ordering',
        doc="""
        List of variables in the order that they should appear in the NL
        file.  Note that this is only a suggestion, as the NL writer
        will move all nonlinear variables before linear ones, and within
        nonlinear variables, variables appearing in both objectives and
        constraints before variables appearing only in constraints,
        which appear before variables appearing only in objectives.
        Within each group, continuous variables appear before discrete
        variables.  In all cases, column_order is preserved within each
        group."""
    ))

    def __init__(self):
        self.config = self.CONFIG()

    def __call__(self, model, filename, solver_capability, io_options):
        if filename is None:
            filename = model.name + ".nl"
        filename_base = os.path.splitext(filename)[0]
        row_fname = filename_base + '.row'
        col_fname = filename_base + '.col'

        config = self.config(io_options)
        if config.symbolic_solver_labels:
            _open = lambda fname: open(fname, 'w')
        else:
            _open = nullcontext
        with open(filename, 'w') as FILE, \
             _open(row_fname) as ROWFILE, \
             _open(col_fname) as COLFILE:
            info = self.write(
                model, FILE, ROWFILE, COLFILE, config=config)
        # Historically, the NL writer communicated the external function
        # libraries back to the ASL interface through the PYOMO_AMPLFUNC
        # environment variable.
        set_pyomo_amplfunc_env(info.external_function_libraries)
        # Generate the symbol map expected by the old readers
        symbol_map = self._generate_symbol_map(info)
        # The ProblemWriter callable interface returns the filename that
        # was generated and the symbol_map
        return filename, symbol_map

    def write(self, model, ostream, rowstream=None, colstream=None, **options):
        """Write a model in NL format.

        Returns
        -------
        NLWriterInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: io.TextIOBase
            The text output stream where the NL "file" will be written.
            Could be an opened file or a io.StringIO.

        rowstream: io.TextIOBase
            A text output stream to write the ASL "row file" (list of
            constraint / objective names).  Ignored unless
            `symbolic_solver_labels` is True.

        colstream: io.TextIOBase
            A text output stream to write the ASL "col file" (list of
            variable names).  Ignored unless `symbolic_solver_labels` is True.

        """
        config = options.pop('config', self.config)(options)

        # Pause the GC, as the walker that generates the compiled NL
        # representation generates (and disposes of) a large number of
        # small objects.
        with _NLWriter_impl(ostream, rowstream, colstream, config) as impl:
            return impl.write(model)

    write.__doc__ = add_docstring_list(write.__doc__, CONFIG)

    def _generate_symbol_map(self, info):
        # Now that the row/column ordering is resolved, create the labels
        symbol_map = SymbolMap()
        symbol_map.addSymbols(
            (info[0], f"v{idx}") for idx, info in enumerate(info.variables)
        )
        symbol_map.addSymbols(
            (info[0], f"c{idx}") for idx, info in enumerate(info.constraints)
        )
        symbol_map.addSymbols(
            (info[0], f"o{idx}") for idx, info in enumerate(info.objectives)
        )
        return symbol_map

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


class _SuffixData(object):
    def __init__(self, name, column_order, row_order, obj_order, model_id):
        self._name = name
        self._column_order = column_order
        self._row_order = row_order
        self._obj_order = obj_order
        self._model_id = model_id
        self.obj = {}
        self.con = {}
        self.var = {}
        self.prob = {}
        self.datatype = set()

    def update(self, suffix):
        self.datatype.add(suffix.datatype)
        for item in suffix.items():
            self.store(*item)

    def store(self, obj, val):
        _id = id(obj)
        if _id in self._column_order:
            self.var[self._column_order[_id]] = val
        elif _id in self._row_order:
            self.con[self._row_order[_id]] = val
        elif _id in self._obj_order:
            self.obj[self._obj_order[_id]] = val
        elif _id == self._model_id:
            self.prob[0] = val
        elif isinstance(obj, PyomoObject):
            if obj.is_indexed():
                for o in obj.values():
                    self.store(o, val)
            else:
                logger.warning(
                    f"model contained export suffix {self._name} with "
                    f"{obj.ctype.__name__} key '{obj.name}', but that "
                    "object is not exported as part of the NL file.  "
                    "Skipping.")
        else:
            logger.warning(
                f"model contained export suffix {self._name} with "
                f"{obj.__class__.__name__} key '{obj}' that is not "
                "a Var, Constrtaint, Objective, or the model.  Skipping.")


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
            config.symbolic_solver_labels,
        )
        self.next_V_line_id = 0
        self.pause_gc = None

    def __enter__(self):
        assert AMPLRepn.ActiveVisitor is None
        AMPLRepn.ActiveVisitor = self.visitor
        self.pause_gc = PauseGC()
        self.pause_gc.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.pause_gc.__exit__(exc_type, exc_value, tb)
        assert AMPLRepn.ActiveVisitor is self.visitor
        AMPLRepn.ActiveVisitor = None

    def write(self, model):
        timer = TicTocTimer(
            logger=logging.getLogger('pyomo.common.timing.writer')
        )

        sorter = SortComponents.unsorted
        if self.config.file_determinism >= FileDeterminism.SORT_INDICES:
            sorter = sorter | SortComponents.indices
            if self.config.file_determinism  >= FileDeterminism.SORT_SYMBOLS:
                sorter = sorter | SortComponents.alphabetical

        component_map, unknown = categorize_valid_components(
            model,
            active=True,
            sort=sorter,
            valid={
                Block, Objective, Constraint, Var, Param, Expression,
                ExternalFunction, Suffix, SOSConstraint,
                # FIXME: Non-active components should not report as Active
                Set, RangeSet, Port,
                # TODO: Piecewise, Complementarity
            },
            targets={
                Objective, Constraint, Suffix, SOSConstraint,
            }
        )
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the NL writer does not know how to process:\n\t%s" %
                (model.name, "\n\t".join("%s:\n\t\t%s" % (
                    k, "\n\t\t".join(map(attrgetter('name'), v)))
                    for k, v in unknown.items())))

        # Caching some frequently-used objects into the locals()
        symbolic_solver_labels = self.symbolic_solver_labels
        visitor = self.visitor
        ostream = self.ostream
        var_map = self.var_map

        if self.config.column_order == True:
            self.config.column_order = list(model.component_data_objects(
                Var, descend_into=True, sort=sorter))
        elif self.config.file_determinism > FileDeterminism.ORDERED:
            # We will pre-gather the variables so that their order
            # matches the file_determinism flag.  This is a little
            # cumbersome, but is implemented this way for consistency
            # with the original NL writer.
            if self.config.column_order is None:
                self.config.column_order = []
            self.config.column_order.extend(model.component_data_objects(
                Var, descend_into=True, sort=sorter))
        if self.config.column_order is not None:
            # Note that Vars that appear twice (e.g., through a
            # Reference) will be sorted with the FIRST occurrence.
            for var in self.config.column_order:
                if var.is_indexed():
                    for _v in var.values():
                        if not _v.fixed:
                            var_map[id(_v)] = _v
                elif not var.fixed:
                    var_map[id(var)] = var

        #
        # Tabulate the model expressions
        #
        objectives = []
        linear_objs = []
        for block in component_map[Objective]:
            for obj_comp in block.component_objects(
                    Objective, active=True, descend_into=False, sort=sorter):
                try:
                    obj_vals = obj_comp.values()
                except AttributeError:
                    # kernel does not define values() for scalar
                    # objectives or list/tuple containers
                    try:
                        # This could be a list/tuple container.  Try to
                        # iterate over it, and if that fails assume it
                        # is a scalar
                        obj_vals = iter(obj_comp)
                    except:
                        obj_vals = (obj_comp,)
                for obj in obj_vals:
                    if not obj.active:
                        continue
                    expr = visitor.walk_expression((obj.expr, obj, 1))
                    if expr.nonlinear:
                        objectives.append((obj, expr))
                    else:
                        linear_objs.append((obj, expr))
                timer.toc('Objective %s', obj_comp, level=logging.DEBUG)

        # Order the objectives, moving all nonlinear objectives to
        # the beginning
        n_nonlinear_objs = len(objectives)
        objectives.extend(linear_objs)
        n_objs = len(objectives)

        constraints = []
        linear_cons = []
        n_ranges = 0
        n_equality = 0
        for block in component_map[Constraint]:
            for con_comp in block.component_objects(
                    Constraint, active=True, descend_into=False, sort=sorter):
                try:
                    con_vals = con_comp.values()
                except AttributeError:
                    # kernel does not define values() for scalar
                    # constraints or list/tuple containers
                    try:
                        # This could be a list/tuple container.  Try to
                        # iterate over it, and if that fails assume it
                        # is a scalar
                        con_vals = iter(con_comp)
                    except:
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
                    elif _type == 3: #and self.config.skip_trivial_constraints:
                        # FIXME: historically the NL writer was
                        # hard-coded to skip all unbounded constraints
                        continue
                    if expr.nonlinear:
                        constraints.append((con, expr, _type, lb, ub))
                    elif expr.linear:
                        linear_cons.append((con, expr, _type, lb, ub))
                    elif not self.config.skip_trivial_constraints:
                        linear_cons.append((con, expr, _type, lb, ub))
                    else: # constant constraint and skip_trivial_constraints
                        #
                        # TODO: skip_trivial_constraints should be an
                        # enum that also accepts "Exception" so that
                        # solvers can be (easily) notified of infeasible
                        # trivial constraints.
                        if (lb is not None and float(lb) > TOL) or (
                                ub is not None and float(ub) < -TOL):
                            logger.warning(
                                "model contains a trivially infeasible "
                                f"constraint {con.name}, but "
                                "skip_trivial_constraints==True and the "
                                "constraint is being omitted from the NL "
                                "file.  Solving the model may incorrectly "
                                "report a feasible solution.")
                timer.toc('Constraint %s', con_comp, level=logging.DEBUG)

        if self.config.row_order:
            # Note: this relies on two things: 1) dict are ordered, and
            # 2) updating an entry in a dict does not change its
            # ordering.
            row_order = {}
            for con in self.config.row_order:
                if con.is_indexed():
                    for c in con.values():
                        row_order[id(c)] = c
                else:
                    row_order[id(con)] = con
            for c in constraints:
                row_order[id(c)] = c
            for c in linear_cons:
                row_order[id(c)] = c
            # map the implicit dict ordering to an explicit 0..n ordering
            row_order = {_id: i for i, _id in enumerate(row_order.keys())}
            constraints.sort(key=itemgetter(row_order))
            linear_cons.sort(key=itemgetter(row_order))
        else:
            row_order = {}

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
        obj_vars_linear, obj_vars_nonlinear, obj_nnz_by_var \
            = self._categorize_vars(objectives, linear_by_comp)
        con_vars_linear, con_vars_nonlinear, con_nnz_by_var \
            = self._categorize_vars(constraints, linear_by_comp)

        if self.config.export_nonlinear_variables:
            for v in self.config.export_nonlinear_variables:
                if v.is_indexed():
                    _iter = v.values()
                else:
                    _iter = (v,)
                for _v in _iter:
                    _id = id(_v)
                    if _id not in var_map:
                        var_map[_id] = _v
                    con_vars_nonlinear.add(_id)

        con_nnz = sum(con_nnz_by_var.values())
        timer.toc('Categorized model variables: %s nnz', con_nnz,
                  level=logging.DEBUG)

        n_lcons = 0 # We do not yet support logical constraints

        # We need to check the SOS constraints before finalizing the
        # variable order because the SOS constraint *could* reference a
        # variable not yet seen in the model.
        for block in component_map[SOSConstraint]:
            for sos in block.component_objects(
                    SOSConstraint, active=True, descend_into=False,
                    sort=sorter):
                for v in sos.variables:
                    if id(v) not in var_map:
                        _id = id(v)
                        var_map[_id] = v
                        con_vars_linear.add(_id)

        obj_vars = obj_vars_linear | obj_vars_nonlinear
        con_vars = con_vars_linear | con_vars_nonlinear
        all_vars = con_vars | obj_vars
        n_vars = len(all_vars)
        if n_vars < 1:
            # TODO: Remove this.  This exception is included for
            # compatibility with the original NL writer v1.
            raise ValueError(
                "No variables appear in the Pyomo model constraints or"
                " objective. This is not supported by the NL file interface"
            )

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
            linear_binary_vars = linear_only_vars & binary_vars
            variables.extend(sorted(
                linear_binary_vars,
                key=column_order.__getitem__))
            linear_integer_vars = linear_only_vars & integer_vars
            variables.extend(sorted(
                linear_integer_vars,
                key=column_order.__getitem__))
        else:
            linear_binary_vars = linear_integer_vars = set()
        assert len(variables) == n_vars
        timer.toc(
            'Set row / column ordering: %s variables [%s, %s, %s R/B/Z], '
            '%s constraints [%s, %s L/NL]',
            n_vars, len(continuous_vars), len(binary_vars), len(integer_vars),
            len(constraints), n_cons-n_nonlinear_cons, n_nonlinear_cons,
            level=logging.DEBUG)

        # Fill in the variable list and update the new column order.
        #
        # Note that as we allow var_map to contain "known" variables
        # that are not needed in the NL file (and column_order was
        # originally generated from var_map), we will rebuild the
        # column_order to *just* contain the variables that we are
        # sending to the NL.
        self.column_order = column_order = {
            _id: i for i, _id in enumerate(variables)
        }
        for idx, _id in enumerate(variables):
            v = var_map[_id]
            lb, ub = v.bounds
            if lb is not None:
                lb = repr(lb)
            if ub is not None:
                ub = repr(ub)
            variables[idx] = (v, _id, _RANGE_TYPE(lb, ub), lb, ub)
        timer.toc("Computed variable bounds", level=logging.DEBUG)

        # Collect all defined EXPORT suffixes on the model
        suffix_data = {}
        if component_map[Suffix]:
            if not row_order:
                row_order = {id(con[0]): i for i, con in enumerate(constraints)}
            obj_order = {id(obj[0]): i for i, obj in enumerate(objectives)}
            model_id = id(model)
            # Note: reverse the block list so that higher-level Suffix
            # components override lower level ones.
            for block in reversed(component_map[Suffix]):
                for suffix in block.component_objects(
                        Suffix, active=True, descend_into=False, sort=sorter):
                    if not (suffix.direction & Suffix.EXPORT):
                        continue
                    name = suffix.local_name
                    if name not in suffix_data:
                        suffix_data[name] = _SuffixData(
                            name, column_order, row_order, obj_order, model_id)
                    suffix_data[name].update(suffix)
            timer.toc("Collected suffixes", level=logging.DEBUG)

        # Collect all defined SOSConstraints on the model
        if component_map[SOSConstraint]:
            if not row_order:
                row_order = {id(con[0]): i for i, con in enumerate(constraints)}
            if not component_map[Suffix]:
                obj_order = {id(obj[0]): i for i, obj in enumerate(objectives)}
                model_id = id(model)
            for name in ('sosno', 'ref'):
                # I am choosing not to allow a user to mix the use of the Pyomo
                # SOSConstraint component and manual sosno declarations within
                # a single model. I initially tried to allow this but the
                # var section of the code below blows up for two reason. (1)
                # we have to make sure that the sosno suffix is not defined
                # twice for the same variable (2) We have to make sure that
                # the automatically chosen sosno used by the SOSConstraint
                # translation does not already match one a user has manually
                # implemented (this would modify the members in an sos set).
                # Since this suffix is exclusively used for defining sos sets,
                # there is no reason a user can not just stick to one method.
                if name in suffix_data:
                    raise RuntimeError(
                        "The Pyomo NL file writer does not allow both "
                        f"manually declared '{name}' suffixes as well "
                        "as SOSConstraint components to exist on a single "
                        "model. To avoid this error please use only one of "
                        "these methods to define special ordered sets.")
                suffix_data[name] = _SuffixData(
                    name, column_order, row_order, obj_order, model_id)
                suffix_data[name].datatype.add(Suffix.INT)
            sos_id = 0
            sosno = suffix_data['sosno']
            ref = suffix_data['ref']
            for block in reversed(component_map[SOSConstraint]):
                for sos in block.component_data_objects(
                        SOSConstraint, active=True, descend_into=False,
                        sort=sorter):
                    sos_id += 1
                    if sos.level == 1:
                        tag = sos_id
                    elif sos.level == 2:
                        tag = -sos_id
                    else:
                        raise ValueError(
                            f"SOSContraint '{sos.name}' has sos "
                            f"type='{sos.level}', which is not supported "
                            "by the NL file interface")
                    try:
                        _items = sos.get_items()
                    except AttributeError:
                        # kernel doesn't provide the get_items API
                        _items = sos.items()
                    for v, r in _items:
                        sosno.store(v, tag)
                        ref.store(v, r)

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
        timer.toc("Generated row/col labels & comments", level=logging.DEBUG)

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
            " %d %d %d %d %d %d\t"
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
        # Note: con_vars_nonlinear & obj_vars_nonlinear == both_vars_nonlinear
        _n_both_vars = len(both_vars_nonlinear)
        _n_con_vars = len(con_vars_nonlinear)
        # Subtract _n_both_vars to avoid double-counting the overlapping
        # variables
        #
        # This is used to allocate arrays, so the _n_obj_vars needs to
        # include the constraint vars (because they appear between the
        # shared and objective-only vars in the standard variable
        # ordering).  If there are no objective-only variables, then the
        # vector only needs to hold the shared variables.
        _n_obj_vars = _n_con_vars + len(obj_vars_nonlinear) - _n_both_vars
        if _n_obj_vars == _n_con_vars:
            _n_obj_vars = _n_both_vars
        ostream.write(
            " %d %d %d \t"
            "# nonlinear vars in constraints, objectives, both\n"
            % (_n_con_vars, _n_obj_vars, _n_both_vars))

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
            % ( len(linear_binary_vars),
                len(linear_integer_vars),
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
                sum(obj_nnz_by_var.values()),
            ))
        #
        # LINE 9
        #
        ostream.write(
            " %d %d\t# max name lengths: constraints, variables\n"
            % ( max(map(len, row_labels), default=0),
                max(map(len, col_labels), default=0),
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
        for name, data in suffix_data.items():
            if name == 'dual':
                continue
            if len(data.datatype) > 1:
                raise ValueError(
                    "The NL file writer found multiple active export "
                    "suffix components with name '{name}' and different "
                    "datatypes. A single datatype must be declared.")
            _type = next(iter(data.datatype))
            if _type == Suffix.FLOAT:
                _float = 4
            elif _type == Suffix.INT:
                _float = 0
            else:
                raise ValueError(
                    "The NL file writer only supports export suffixes "
                    "declared with a numeric datatype.  Suffix "
                    f"component '{name}' declares type '{_type}'")
            for _field, _vals in zip(
                    range(4),
                    (data.var, data.con, data.obj, data.prob)):
                if not _vals:
                    continue
                ostream.write(f"S{_field|_float} {len(_vals)} {name}\n")
                ostream.write(''.join(f"{_id} {_vals[_id]!r}\n"
                                      for _id in sorted(_vals)))

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
            cache_info = self.subexpression_cache[_id][2]
            if cache_info[2]:
                # substitute expression directly into expression trees
                # and do NOT emit the V line
                continue
            elif 0 in cache_info[:2] or None not in cache_info[:2]:
                self._write_v_line(_id, 0)
            else:
                target_expr = tuple(filter(None, cache_info))[0]
                if target_expr not in single_use_subexpressions:
                    single_use_subexpressions[target_expr] = []
                single_use_subexpressions[target_expr].append(_id)

        #
        # "C" lines (constraints: nonlinear expression)
        #
        for row_idx, info in enumerate(constraints):
            if info[1].nonlinear is None:
                # Because we have moved the nonlinear constraints to the
                # beginning, we can very quickly write all the linear
                # constraints at the end (as their nonlinear expressions
                # are the constant 0).
                _expr = self.template.const % 0
                ostream.write(
                    _expr.join(f'C{i}{row_comments[i]}\n'
                               for i in range(row_idx, len(constraints))))
                # We know that there is at least one linear expression
                # (row_idx), so we can unconditionally emit the last "0
                # expression":
                ostream.write(_expr)
                break
            if single_use_subexpressions:
                for _id in single_use_subexpressions.get(id(info[0]), ()):
                    self._write_v_line(_id, row_idx + 1)
            ostream.write(f'C{row_idx}{row_comments[row_idx]}\n')
            self._write_nl_expression(info[1], False)

        #
        # "O" lines (objectives: nonlinear expression)
        #
        for obj_idx, info in enumerate(objectives):
            if single_use_subexpressions:
                for _id in single_use_subexpressions.get(id(info[0]), ()):
                    # Note that "Writing .nl files" (2005) is incorrectly
                    # missing the "+ 1" in the description of V lines
                    # appearing in only Objectives (bottom of page 9).
                    self._write_v_line(_id, n_cons + n_lcons + obj_idx + 1)
            lbl = row_comments[n_cons + obj_idx]
            sense = 0 if info[0].sense == minimize else 1
            ostream.write(f'O{obj_idx} {sense}{lbl}\n')
            self._write_nl_expression(info[1], True)

        #
        # "d" lines (dual initialization)
        #
        if 'dual' in suffix_data:
            _data = suffix_data['dual']
            if _data.var:
                logger.warning("ignoring 'dual' suffix for Var types")
            if _data.obj:
                logger.warning("ignoring 'dual' suffix for Objective types")
            if _data.prob:
                logger.warning("ignoring 'dual' suffix for Model")
            if _data.con:
                ostream.write(f"d{len(_data.con)}\n")
                ostream.write(''.join(f"{_id} {_data.con[_id]!r}\n"
                                      for _id in sorted(_data.con)))

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
        for row_idx, info in enumerate(constraints):
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
            # ASL will fail on "J<N> 0", so if there are no coefficients
            # (i.e., a constant objective), then skip this entry
            if not linear:
                continue
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
            # ASL will fail on "G<N> 0", so if there are no coefficients
            # (i.e., a constant objective), then skip this entry
            if not linear:
                continue
            ostream.write(
                f'G{obj_idx} {len(linear)}{row_comments[obj_idx + n_cons]}\n')
            for _id in sorted(linear.keys(), key=column_order.__getitem__):
                ostream.write(
                    f'{column_order[_id]} {linear[_id]!r}\n'
                )

        # Generate the return information
        info = NLWriterInfo(
            variables, constraints, objectives, sorted(amplfunc_libraries),
            row_labels, col_labels)
        timer.toc("Wrote NL stream", level=logging.DEBUG)
        timer.toc("Generated NL representation", delta=False)
        return info

    def _categorize_vars(self, comp_list, linear_by_comp):
        """Categorize compiled expression vars into linear and nonlinear

        This routine takes an iterable of compiled component expression
        infos and returns the sets of variables appearing linearly and
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
            #
            # For efficiency, we will omit the obvious assertion:
            #   assert expr_info.mult == 1
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
            #     # guarded by 'if expr_info.linear', so it is OK to
            #     # leave the symbol undefined here:
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
                        if target[idx] is None:
                            target[idx] = src_id
                            queue.append(subid)
                        elif target[idx] and target[idx] != src_id:
                            target[idx] = 0
                            queue.append(subid)
        # Now we can reliably know where nested subexpressions are used.
        # Group them into:
        #   [ used in both objectives and constraints,
        #     used by more than one constraint (but no objectives),
        #     used by more than one objective (but no constraints),
        #     used by one constraint,
        #     used by one objective ]
        n_subexpressions = [0]*5
        for info in map(itemgetter(2),
                        map(self.subexpression_cache.__getitem__,
                            filter(self.used_named_expressions.__contains__,
                                   self.subexpression_order))):
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
        # Note that repn.mult should always be 1 (the AMPLRepn was
        # compiled before this point).  Omitting the assertion for
        # efficiency.
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
        # Do NOT write out 0 coefficients here: doing so fouls up the
        # ASL's logic for calculating derivatives, leading to 'nan' in
        # the Hessian results.
        linear = dict(item for item in info[1].linear.items() if item[1])
        #
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
            if self.mult == -1:
                prefix += template.negation
            else:
                prefix += template.multiplier % self.mult
        if self.nl is not None:
            nl, nl_args = self.nl
            visitor._mark_named_expression_as_used(nl_args)
            if prefix:
                nl = prefix + nl
            if args is not None and args is not nl_args:
                args.extend(nl_args)
            else:
                args = list(nl_args)
            return nl, args

        if args is None:
            args = []
        if self.linear:
            nterms = len(self.linear)
            _v_template = template.var
            _m_template = template.monomial
            nl_sum = ''.join(_v_template if c == 1 else _m_template % c
                             for c in map(itemgetter(1), self.linear))
            args.extend(map(itemgetter(0), self.linear))
        else:
            nterms = 0
            nl_sum = ''
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
        if self.const:
            nterms += 1
            nl_sum += template.const % self.const

        if nterms > 2:
            return prefix + (template.nary_sum % nterms) + nl_sum, args
        elif nterms == 2:
            return prefix + template.binary_sum + nl_sum, args
        elif nterms == 1:
            return prefix + nl_sum, args
        else: # nterms == 0
            return prefix + (template.const % 0), []

    def compile_nonlinear_fragment(self, visitor):
        args = []
        nterms = len(self.nonlinear)
        nl_sum = ''.join(map(itemgetter(0), self.nonlinear))
        deque(map(args.extend, map(itemgetter(1), self.nonlinear)),
              maxlen=0)

        if nterms > 2:
            self.nonlinear = (visitor.template.nary_sum % nterms) + nl_sum, args
        elif nterms == 2:
            self.nonlinear = visitor.template.binary_sum + nl_sum, args
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
        # Note that self.mult will always be 1 (we only call append()
        # within a sum, so there is no opportunity for self.mult to
        # change). Omitting the assertion for efficiency.
        #assert self.mult == 1
        _type = other[0]
        if _type is _MONOMIAL:
            self.linear.append(other[1:])
        elif _type is _GENERAL:
            other = other[1]
            if other.nl is not None and other.nonlinear:
                if other.linear:
                    # This is a named expression with both a linear and
                    # nonlinear component.  We want to merge it with
                    # this AMPLRepn, preserving the named expression for
                    # only the nonlinear component (merging the linear
                    # component with this AMPLRepn).  We need to make
                    # sure that we have marked that we are using the
                    # named expression for the nonlinear component.
                    self.ActiveVisitor._mark_named_expression_as_used(
                        other.nonlinear[1])
                else:
                    # This is a nonlinear-only named expression,
                    # possibly with a multiplier that is not 1.  Compile
                    # it and append it (this both resolves the
                    # multiplier, and marks the named expression as
                    # having been used)
                    self.nonlinear.append(
                        other.compile_repn(self.ActiveVisitor))
                    return
            if other.mult != 1:
                mult = other.mult
                self.const += mult * other.const
                if other.linear:
                    self.linear.extend((v, c*mult) for v, c in other.linear)
                if other.nonlinear:
                    if other.nonlinear.__class__ is list:
                        other.compile_nonlinear_fragment(self.ActiveVisitor)
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

    binary_sum = 'o0\t#+\n'
    product = 'o2\t#*\n'
    division = 'o3\t# /\n'
    pow = 'o5\t#^\n'
    abs = 'o15\t# abs\n'
    negation = 'o16\t#-\n'
    nary_sum = 'o54\t# sumlist\n%d\t# (n)\n'
    exprif = 'o35\t# if\n'
    and_expr = 'o21\t# and\n'
    less_than = 'o22\t# lt\n'
    less_equal = 'o23\t# le\n'
    equality = 'o24\t# eq\n'
    external_fcn = 'f%d %d%s\n'
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
            return AMPLRepn(0, [data[1:]], None)
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
    #
    # This is a 3-tuple [con_id, obj_id, substitute_expression].  If the
    # expression is used by more than 1 constraint / objective, then the
    # id is set to 0.  If it is not used by any, then it is None.
    # substitue_expression is a bool indicating if this named
    # subexpression tree should be directly substituted into any
    # expression tree that references this node (i.e., do NOT emit the V
    # line).
    expression_source = [None, None, False]

    if repn.nonlinear:
        # As we will eventually need the compiled form of any nonlinear
        # expression, we will go ahead and compile it here.  We do not
        # do the same for the linear component as we will only need the
        # linear component compiled to a dict if we are emitting the
        # original (linear + nonlinear) V line (which will not happen if
        # the V line is part of a larger linear operator).
        if repn.nonlinear.__class__ is list:
            repn.compile_nonlinear_fragment(visitor)

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
            # See above for the meaning of this source information
            nl_info = list(expression_source)
            visitor.subexpression_cache[sub_id] = (
                sub_node, sub_repn, nl_info,
            )
            repn.nonlinear = sub_repn.nl
            # It is important that the NL subexpression comes before the
            # main named expression:
            visitor.subexpression_order.append(sub_id)
            # The nonlinear identifier is *always* used
            visitor.used_named_expressions.add(sub_id)
        else:
            nl_info = expression_source
        # The nonlinear component of this named expression is
        # guaranteed to be used by this expression
        setitem(nl_info, *visitor.active_expression_source)
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
        # 2: the source usage information for this subexpression:
        #    [(con_id, obj_id, substitute); see above]
        expression_source,
    )
    visitor.subexpression_order.append(_id)
    ans = AMPLRepn(
        repn.const,
        list(repn.linear) if repn.linear is not None else repn.linear,
        repn.nonlinear
    )
    ans.nl = repn.nl
    return (_GENERAL, ans)

def handle_external_function_node(visitor, node, *args):
    func = node._fcn._function
    # There is a special case for external functions: these are the only
    # expressions that can accept string arguments. As we currently pass
    # these as 'precompiled' general NL fragments, the normal trap for
    # constant subexpressions will miss constant external function calls
    # that contain strings.  We will catch that case here.
    if all(arg[0] is _CONSTANT or
           (arg[0] is _GENERAL and arg[1].nl and not arg[1].nl[1])
           for arg in args):
        arg_list = [arg[1] if arg[0] is _CONSTANT else arg[1].const
                    for arg in args]
        return (_CONSTANT, node._apply_operation(arg_list))
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
    comment = f'\t#{node.local_name}' if visitor.symbolic_solver_labels else ''
    nonlin = node_result_to_amplrepn(args[0]).compile_repn(
        visitor, visitor.template.external_fcn % (
            visitor.external_functions[func][0], len(args), comment))
    for arg in args[1:]:
        nonlin = node_result_to_amplrepn(arg).compile_repn(visitor, *nonlin)
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
    kernel.expression.noclone: handle_named_expression_node,
    # Note: objectives are special named expressions
    _GeneralObjectiveData: handle_named_expression_node,
    ScalarObjective: handle_named_expression_node,
    kernel.objective.objective: handle_named_expression_node,
    ExternalFunctionExpression: handle_external_function_node,
    # These are handled explicitly in beforeChild():
    # LinearExpression: handle_linear_expression,
    # SumExpression: handle_sum_expression,
    # MonomialTermExpression: handle_monomial_term,
}


def _before_native(visitor, child):
    return False, (_CONSTANT, child)

def _before_string(visitor, child):
    ans = AMPLRepn(None, None, None)
    ans.nl = (visitor.template.string % (len(child), child), ())
    return False, (_GENERAL, ans)

def _before_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        if child.fixed:
            return False, (_CONSTANT, child())
        visitor.var_map[_id] = child
    return False, (_MONOMIAL, _id, 1)

def _before_npv(visitor, child):
    # TBD: It might be more efficient to cache the value of NPV
    # expressions to avoid duplicate evaluations.  However, current
    # examples do not benefit from this cache.
    #
    # _id = id(child)
    # if _id in visitor.value_cache:
    #     child = visitor.value_cache[_id]
    # else:
    #     child = visitor.value_cache[_id] = child()
    # return False, (_CONSTANT, child)
    return False, (_CONSTANT, child())

def _before_monomial(visitor, child):
    #
    # The following are performance optimizations for common
    # situations (Monomial terms and Linear expressions)
    #
    arg1, arg2 = child._args_
    if arg1.__class__ not in native_types:
        # TBD: It might be more efficient to cache the value of NPV
        # expressions to avoid duplicate evaluations.  However, current
        # examples do not benefit from this cache.
        #
        # _id = id(arg1)
        # if _id in visitor.value_cache:
        #     arg1 = visitor.value_cache[_id]
        # else:
        #     arg1 = visitor.value_cache[_id] = arg1()
        arg1 = arg1()
    # Trap multiplication by 0
    if not arg1:
        return False, (_CONSTANT, 0)
    _id = id(arg2)
    if _id not in visitor.var_map:
        if arg2.fixed:
            return False, (_CONSTANT, arg1 * arg2())
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
        if not c:
            continue
        elif v.fixed:
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
        ans = AMPLRepn(
            repn.const,
            list(repn.linear) if repn.linear is not None else repn.linear,
            repn.nonlinear
        )
        ans.nl = repn.nl
        return False, (_GENERAL, ans)
    else:
        return True, None

def _before_general_expression(visitor, child):
    return True, None


# Register an initial set of known expression types with the "before
# child" expression handler lookup table.
_before_child_handlers = {
    _type: _before_native for _type in native_numeric_types
}
for _type in native_types:
    if issubclass(_type, str):
        _before_child_handlers[_type] = _before_string
# general operators
for _type in _operator_handles:
    _before_child_handlers[_type] = _before_general_expression
# named subexpressions
for _type in (
        _GeneralExpressionData, ScalarExpression,
        kernel.expression.expression, kernel.expression.noclone,
        _GeneralObjectiveData, ScalarObjective, kernel.objective.objective):

    _before_child_handlers[_type] = _before_named_expression
# Special linear / summation expressions
_before_child_handlers[MonomialTermExpression] = _before_monomial
_before_child_handlers[LinearExpression] = _before_linear
_before_child_handlers[SumExpression] = _before_general_expression

class AMPLRepnVisitor(StreamBasedExpressionVisitor):

    def __init__(self, template, subexpression_cache, subexpression_order,
                 external_functions, var_map, used_named_expressions,
                 symbolic_solver_labels):
        super().__init__()
        self.template = template
        self.subexpression_cache = subexpression_cache
        self.subexpression_order = subexpression_order
        self.external_functions = external_functions
        self.active_expression_source = None
        self.var_map = var_map
        self.used_named_expressions = used_named_expressions
        self.symbolic_solver_labels = symbolic_solver_labels
        #self.value_cache = {}
        self._before_child_handlers = _before_child_handlers
        self._operator_handles = _operator_handles

    def initializeWalker(self, expr):
        expr, src, src_idx = expr
        self.active_expression_source = (src_idx, id(src))
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        try:
            return self._before_child_handlers[child.__class__](self, child)
        except KeyError:
            self._register_new_before_child_processor(child)
        return self._before_child_handlers[child.__class__](self, child)

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if node.__class__ is SumExpression:
            return node.args, AMPLRepn(0, [], [])
        else:
            return node.args, []

    def exitNode(self, node, data):
        if data.__class__ is AMPLRepn:
            # If the summation resulted in a constant, return the constant
            if data.linear or data.nonlinear or data.nl:
                return (_GENERAL, data)
            else:
                return (_CONSTANT, data.const)
        #
        # General expressions...
        #
        if all(arg[0] is _CONSTANT for arg in data):
            return (
                _CONSTANT, node._apply_operation(list(map(
                    itemgetter(1), data)))
            )
        return self._operator_handles[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        ans = node_result_to_amplrepn(result)
        # If this was a nonlinear named expression, and that expression
        # has no linear portion, then we will directly use this as a
        # named expression.  We need to mark that the expression was
        # used and return it as a simple nonlinear expression pointing
        # to this named expression.  In all other cases, we will return
        # the processed representation (which will reference the
        # nonlinear-only named subexpression - if it exists - but not
        # this outer named expression).  This prevents accidentally
        # recharacterizing variables that only appear linearly as
        # nonlinear variables.
        if ans.nl and ans.nonlinear:
            if ans.linear:
                self._mark_named_expression_as_used(ans.nonlinear[1])
            else:
                self._mark_named_expression_as_used(ans.nl[1])
                ans.nonlinear = ans.nl

        ans.nl = None

        if ans.nonlinear.__class__ is list:
            if ans.nonlinear:
                ans.compile_nonlinear_fragment(self)
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
        if child_type in native_numeric_types:
            handlers[child_type] = _before_native
        elif issubclass(child_type, str):
            handlers[child_type] = _before_string
        elif child_type in native_types:
            handlers[child_type] = _before_native
        elif not child.is_expression_type():
            if child.is_potentially_variable():
                handlers[child_type] = _before_var
            else:
                handlers[child_type] = _before_npv
        elif not child.is_potentially_variable():
            handlers[child_type] = _before_npv
        elif id(child) in self.subexpression_cache:
            handlers[child_type] = _before_named_expression
        else:
            handlers[child_type] = _before_general_expression

    def _mark_named_expression_as_used(self, ref):
        assert len(ref) == 1
        _named_expr_id = ref[0]
        self.used_named_expressions.add(_named_expr_id)
        # Record that this named expression was used
        info = self.subexpression_cache[_named_expr_id][2]
        _idx = self.active_expression_source[0]
        if info[_idx] is None:
            info[_idx] = self.active_expression_source[1]
        elif info[_idx] != self.active_expression_source[1]:
            info[_idx] = 0
