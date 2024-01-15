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

import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    InEnum,
    document_kwargs_from_configdict,
)
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
    native_complex_types,
    native_numeric_types,
    native_types,
    value,
)
from pyomo.common.timing import TicTocTimer

from pyomo.core.expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    MonomialTermExpression,
    LinearExpression,
    SumExpression,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
    Expr_ifExpression,
    ExternalFunctionExpression,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
    Block,
    Objective,
    Constraint,
    Var,
    Param,
    Expression,
    ExternalFunction,
    Suffix,
    SOSConstraint,
    SymbolMap,
    NameLabeler,
    SortComponents,
    minimize,
)
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
    ScalarObjective,
    _GeneralObjectiveData,
    _ObjectiveData,
)
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory

from pyomo.repn.util import (
    BeforeChildDispatcher,
    ExitNodeDispatcher,
    ExprType,
    FileDeterminism,
    FileDeterminism_to_SortComponents,
    InvalidNumber,
    apply_node_operation,
    categorize_valid_components,
    complex_number_error,
    initialize_var_map_from_column_order,
    int_float,
    ordered_active_constraints,
    nan,
    sum_like_expression_types,
)
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port

###

logger = logging.getLogger(__name__)

# Feasibility tolerance for trivial (fixed) constraints
TOL = 1e-8
inf = float('inf')
minus_inf = -inf

_CONSTANT = ExprType.CONSTANT
_MONOMIAL = ExprType.MONOMIAL
_GENERAL = ExprType.GENERAL

ScalingFactors = namedtuple(
    'ScalingFactors', ['variables', 'constraints', 'objectives']
)


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
        :py:attr:`constraints` + :py:attr:`objectives` and the generated
        .row file.

    column_labels: List[str]

        The list of string names for the variables written to the NL
        file in the same order as the :py:attr:`variables` and generated
        .col file.

    eliminated_vars: List[Tuple[_VarData, NumericExpression]]

        The list of variables in the model that were eliminated by the
        presolve.  Each entry is a 2-tuple of (:py:class:`_VarData`,
        :py:class`NumericExpression`|`float`).  The list is in the
        necessary order for correct evaluation (i.e., all variables
        appearing in the expression must either have been sent to the
        solver, or appear *earlier* in this list.

    scaling: ScalingFactors or None

        namedtuple holding 3 lists of (variables, constraints, objectives)
        scaling factors in the same order (and size) as the `variables`,
        `constraints`, and `objectives` attributes above.

    """

    def __init__(
        self,
        var,
        con,
        obj,
        external_libs,
        row_labels,
        col_labels,
        eliminated_vars,
        scaling,
    ):
        self.variables = var
        self.constraints = con
        self.objectives = obj
        self.external_function_libraries = external_libs
        self.row_labels = row_labels
        self.column_labels = col_labels
        self.eliminated_vars = eliminated_vars
        self.scaling = scaling


@WriterFactory.register('nl_v2', 'Generate the corresponding AMPL NL file (version 2).')
class NLWriter(object):
    CONFIG = ConfigBlock('nlwriter')
    CONFIG.declare(
        'show_section_timing',
        ConfigValue(
            default=False,
            domain=bool,
            description='Print timing after writing each section of the NL file',
        ),
    )
    CONFIG.declare(
        'skip_trivial_constraints',
        ConfigValue(
            default=False,
            domain=bool,
            description='Skip writing constraints whose body is constant',
        ),
    )
    CONFIG.declare(
        'file_determinism',
        ConfigValue(
            default=FileDeterminism.ORDERED,
            domain=InEnum(FileDeterminism),
            description='How much effort to ensure file is deterministic',
            doc="""
        How much effort do we want to put into ensuring the
        NL file is written deterministically for a Pyomo model:
            NONE (0) : None
            ORDERED (10): rely on underlying component ordering (default)
            SORT_INDICES (20) : sort keys of indexed components
            SORT_SYMBOLS (30) : sort keys AND sort names (not declaration order)
        """,
        ),
    )
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description='Write the corresponding .row and .col files',
        ),
    )
    CONFIG.declare(
        'scale_model',
        ConfigValue(
            default=True,
            domain=bool,
            description="Write variables and constraints in scaled space",
            doc="""
            If True, then the writer will output the model constraints and
            variables in 'scaled space' using the scaling from the
            'scaling_factor' Suffix, if provided.""",
        ),
    )
    CONFIG.declare(
        'export_nonlinear_variables',
        ConfigValue(
            default=None,
            domain=list,
            description='Extra variables to include in NL file',
            doc="""
        List of variables to ensure are in the NL file (even if they
        don't appear in any constraints).""",
        ),
    )
    CONFIG.declare(
        'row_order',
        ConfigValue(
            default=None,
            description='Preferred constraint ordering',
            doc="""
        List of constraints in the order that they should appear in the
        NL file.  Note that this is only a suggestion, as the NL writer
        will move all nonlinear constraints before linear ones
        (preserving row_order within each group).""",
        ),
    )
    CONFIG.declare(
        'column_order',
        ConfigValue(
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
        group.""",
        ),
    )
    CONFIG.declare(
        'export_defined_variables',
        ConfigValue(
            default=True,
            domain=bool,
            description='Preferred variable ordering',
            doc="""
        If True, export Expression objects to the NL file as 'defined
        variables'.""",
        ),
    )
    CONFIG.declare(
        'linear_presolve',
        ConfigValue(
            default=True,
            domain=bool,
            description='Perform linear presolve',
            doc="""
        If True, we will perform a basic linear presolve by performing
        variable elimination (without fill-in).""",
        ),
    )

    def __init__(self):
        self.config = self.CONFIG()

    def __call__(self, model, filename, solver_capability, io_options):
        if filename is None:
            filename = model.name + ".nl"
        filename_base = os.path.splitext(filename)[0]
        row_fname = filename_base + '.row'
        col_fname = filename_base + '.col'

        config = self.config(io_options)

        # There is no (convenient) way to pass the scaling factors or
        # information about presolved variables back to the solver
        # through the old "call" interface (since solvers that used that
        # interface predated scaling / presolve).  We will play it safe
        # and disable scaling / presolve when called through this API
        config.scale_model = False
        config.linear_presolve = False

        if config.symbolic_solver_labels:
            _open = lambda fname: open(fname, 'w')
        else:
            _open = nullcontext
        with open(filename, 'w', newline='') as FILE, _open(
            row_fname
        ) as ROWFILE, _open(col_fname) as COLFILE:
            info = self.write(model, FILE, ROWFILE, COLFILE, config=config)
        # Historically, the NL writer communicated the external function
        # libraries back to the ASL interface through the PYOMO_AMPLFUNC
        # environment variable.
        set_pyomo_amplfunc_env(info.external_function_libraries)
        # Generate the symbol map expected by the old readers
        symbol_map = self._generate_symbol_map(info)
        # The ProblemWriter callable interface returns the filename that
        # was generated and the symbol_map
        return filename, symbol_map

    @document_kwargs_from_configdict(CONFIG)
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

    def _generate_symbol_map(self, info):
        # Now that the row/column ordering is resolved, create the labels
        symbol_map = SymbolMap()
        symbol_map.addSymbols(
            (info, f"v{idx}") for idx, info in enumerate(info.variables)
        )
        symbol_map.addSymbols(
            (info, f"c{idx}") for idx, info in enumerate(info.constraints)
        )
        symbol_map.addSymbols(
            (info, f"o{idx}") for idx, info in enumerate(info.objectives)
        )
        return symbol_map


class _SuffixData(object):
    def __init__(self, name):
        self.name = name
        self.obj = {}
        self.con = {}
        self.var = {}
        self.prob = {}
        self.datatype = set()
        self.values = ComponentMap()

    def update(self, suffix):
        self.datatype.add(suffix.datatype)
        self.values.update(suffix)

    def store(self, obj, val):
        self.values[obj] = val

    def compile(self, column_order, row_order, obj_order, model_id):
        missing_component_data = ComponentSet()
        unknown_data = ComponentSet()
        queue = [self.values.items()]
        while queue:
            for obj, val in queue.pop(0):
                if val.__class__ not in int_float:
                    # [JDS] I am not entirely sure why, but we have
                    # historically supported suffix values that hold
                    # dictionaries that map arbitrary component data
                    # objects to values.  We will preserve that behavior
                    # here.  This behavior is exercised by a
                    # ExternalGreyBox test.
                    if isinstance(val, dict):
                        queue.append(val.items())
                        continue
                    val = float(val)
                _id = id(obj)
                if _id in column_order:
                    self.var[column_order[_id]] = val
                elif _id in row_order:
                    self.con[row_order[_id]] = val
                elif _id in obj_order:
                    self.obj[obj_order[_id]] = val
                elif _id == model_id:
                    self.prob[0] = val
                elif isinstance(obj, (_VarData, _ConstraintData, _ObjectiveData)):
                    missing_component_data.add(obj)
                elif isinstance(obj, (Var, Constraint, Objective)):
                    # Expand this indexed component to store the
                    # individual ComponentDatas, but ONLY if the
                    # component data is not in the original dictionary
                    # of values that we extracted from the Suffixes
                    queue.append(
                        product(
                            filterfalse(self.values.__contains__, obj.values()), (val,)
                        )
                    )
                else:
                    unknown_data.add(obj)
        if missing_component_data:
            logger.warning(
                f"model contains export suffix '{self.name}' that "
                f"contains {len(missing_component_data)} component keys that are "
                "not exported as part of the NL file.  "
                "Skipping."
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Skipped component keys:\n\t"
                    + "\n\t".join(sorted(map(str, missing_component_data)))
                )
        if unknown_data:
            logger.warning(
                f"model contains export suffix '{self.name}' that "
                f"contains {len(unknown_data)} keys that are not "
                "Var, Constraint, Objective, or the model.  Skipping."
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Skipped component keys:\n\t"
                    + "\n\t".join(sorted(map(str, unknown_data)))
                )


class CachingNumericSuffixFinder(SuffixFinder):
    scale = True

    def __init__(self, name, default=None):
        super().__init__(name, default)
        self.suffix_cache = {}

    def __call__(self, obj):
        _id = id(obj)
        if _id in self.suffix_cache:
            return self.suffix_cache[_id]
        ans = self.find(obj)
        if ans.__class__ not in int_float:
            ans = float(ans)
        self.suffix_cache[_id] = ans
        return ans


class _NoScalingFactor(object):
    scale = False

    def __call__(self, obj):
        return 1


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
        self.var_map = {}
        self.sorter = FileDeterminism_to_SortComponents(config.file_determinism)
        self.visitor = AMPLRepnVisitor(
            self.template,
            self.subexpression_cache,
            self.subexpression_order,
            self.external_functions,
            self.var_map,
            self.used_named_expressions,
            self.symbolic_solver_labels,
            self.config.export_defined_variables,
            self.sorter,
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
        timing_logger = logging.getLogger('pyomo.common.timing.writer')
        timer = TicTocTimer(logger=timing_logger)
        with_debug_timing = (
            timing_logger.isEnabledFor(logging.DEBUG) and timing_logger.hasHandlers()
        )

        sorter = FileDeterminism_to_SortComponents(self.config.file_determinism)
        component_map, unknown = categorize_valid_components(
            model,
            active=True,
            sort=sorter,
            valid={
                Block,
                Objective,
                Constraint,
                Var,
                Param,
                Expression,
                # FIXME: Non-active components should not report as Active
                ExternalFunction,
                Set,
                RangeSet,
                Port,
                # TODO: Piecewise, Complementarity
            },
            targets={Suffix, SOSConstraint},
        )
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the NL writer does not know how to process:\n\t%s"
                % (
                    model.name,
                    "\n\t".join(
                        "%s:\n\t\t%s" % (k, "\n\t\t".join(map(attrgetter('name'), v)))
                        for k, v in unknown.items()
                    ),
                )
            )

        # Caching some frequently-used objects into the locals()
        symbolic_solver_labels = self.symbolic_solver_labels
        visitor = self.visitor
        ostream = self.ostream
        linear_presolve = self.config.linear_presolve

        var_map = self.var_map
        initialize_var_map_from_column_order(model, self.config, var_map)
        timer.toc('Initialized column order', level=logging.DEBUG)

        # Collect all defined EXPORT suffixes on the model
        suffix_data = {}
        if component_map[Suffix]:
            # Note: reverse the block list so that higher-level Suffix
            # components override lower level ones.
            for block in reversed(component_map[Suffix]):
                for suffix in block.component_objects(
                    Suffix, active=True, descend_into=False, sort=sorter
                ):
                    if not suffix.export_enabled() or not suffix:
                        continue
                    name = suffix.local_name
                    if name not in suffix_data:
                        suffix_data[name] = _SuffixData(name)
                    suffix_data[name].update(suffix)
        #
        # Data structures to support variable/constraint scaling
        #
        if self.config.scale_model and 'scaling_factor' in suffix_data:
            scaling_factor = CachingNumericSuffixFinder('scaling_factor', 1)
            scaling_cache = scaling_factor.suffix_cache
            del suffix_data['scaling_factor']
        else:
            scaling_factor = _NoScalingFactor()
        scale_model = scaling_factor.scale

        timer.toc("Collected suffixes", level=logging.DEBUG)

        #
        # Data structures to support presolve
        #
        # lcon_by_linear_nnz stores all linear constraints grouped by the NNZ
        # in the linear portion of the expression.  The value is another
        # dict mapping id(con) to constraint info
        lcon_by_linear_nnz = defaultdict(dict)
        # comp_by_linear_var maps id(var) to lists of constraint /
        # object infos that have that var in the linear portion of the
        # expression
        comp_by_linear_var = defaultdict(list)

        #
        # Tabulate the model expressions
        #
        objectives = []
        linear_objs = []
        last_parent = None
        for obj in model.component_data_objects(Objective, active=True, sort=sorter):
            if with_debug_timing and obj.parent_component() is not last_parent:
                if last_parent is None:
                    timer.toc(None)
                else:
                    timer.toc('Objective %s', last_parent, level=logging.DEBUG)
                last_parent = obj.parent_component()
            expr_info = visitor.walk_expression((obj.expr, obj, 1, scaling_factor(obj)))
            if expr_info.named_exprs:
                self._record_named_expression_usage(expr_info.named_exprs, obj, 1)
            if expr_info.nonlinear:
                objectives.append((obj, expr_info))
            else:
                linear_objs.append((obj, expr_info))
            if linear_presolve:
                obj_id = id(obj)
                for _id in expr_info.linear:
                    comp_by_linear_var[_id].append((obj_id, expr_info))
        if with_debug_timing:
            # report the last objective
            timer.toc('Objective %s', last_parent, level=logging.DEBUG)
        else:
            timer.toc('Processed %s objectives', len(objectives))

        # Order the objectives, moving all nonlinear objectives to
        # the beginning
        n_nonlinear_objs = len(objectives)
        objectives.extend(linear_objs)
        n_objs = len(objectives)

        constraints = []
        linear_cons = []
        n_ranges = 0
        n_equality = 0
        n_complementarity_nonlin = 0
        n_complementarity_lin = 0
        # TODO: update the writer to tabulate and report the range and
        # nzlb values.  Low priority, as they do not appear to be
        # required for solvers like PATH.
        n_complementarity_range = 0
        n_complementarity_nz_var_lb = 0
        #
        last_parent = None
        for con in ordered_active_constraints(model, self.config):
            if with_debug_timing and con.parent_component() is not last_parent:
                if last_parent is None:
                    timer.toc(None)
                else:
                    timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
                last_parent = con.parent_component()
            scale = scaling_factor(con)
            expr_info = visitor.walk_expression((con.body, con, 0, scale))
            if expr_info.named_exprs:
                self._record_named_expression_usage(expr_info.named_exprs, con, 0)

            # Note: Constraint.lb/ub guarantee a return value that is
            # either a (finite) native_numeric_type, or None
            lb = con.lb
            ub = con.ub
            if lb is None and ub is None:  # and self.config.skip_trivial_constraints:
                continue
            if scale != 1:
                if lb is not None:
                    lb = lb * scale
                if ub is not None:
                    ub = ub * scale
                if scale < 0:
                    lb, ub = ub, lb
            if expr_info.nonlinear:
                constraints.append((con, expr_info, lb, ub))
            elif expr_info.linear:
                linear_cons.append((con, expr_info, lb, ub))
            elif not self.config.skip_trivial_constraints:
                linear_cons.append((con, expr_info, lb, ub))
            else:  # constant constraint and skip_trivial_constraints
                c = expr_info.const
                if (lb is not None and lb - c > TOL) or (
                    ub is not None and ub - c < -TOL
                ):
                    raise InfeasibleConstraintException(
                        "model contains a trivially infeasible "
                        f"constraint '{con.name}' (fixed body value "
                        f"{c} outside bounds [{lb}, {ub}])."
                    )
            if linear_presolve:
                con_id = id(con)
                if not expr_info.nonlinear and lb == ub and lb is not None:
                    lcon_by_linear_nnz[len(expr_info.linear)][con_id] = expr_info, lb
                for _id in expr_info.linear:
                    comp_by_linear_var[_id].append((con_id, expr_info))
        if with_debug_timing:
            # report the last constraint
            timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
        else:
            timer.toc('Processed %s constraints', len(constraints))

        # This may fetch more bounds than needed, but only in the cases
        # where variables were completely eliminated while walking the
        # expressions, or when users provide superfluous variables in
        # the column ordering.
        var_bounds = {_id: v.bounds for _id, v in var_map.items()}

        eliminated_cons, eliminated_vars = self._linear_presolve(
            comp_by_linear_var, lcon_by_linear_nnz, var_bounds
        )
        del comp_by_linear_var
        del lcon_by_linear_nnz

        # Order the constraints, moving all nonlinear constraints to
        # the beginning
        n_nonlinear_cons = len(constraints)
        if eliminated_cons:
            _removed = eliminated_cons.__contains__
            constraints.extend(filterfalse(lambda c: _removed(id(c[0])), linear_cons))
        else:
            constraints.extend(linear_cons)
        n_cons = len(constraints)

        #
        # Collect variables from constraints and objectives into the
        # groupings necessary for AMPL
        #
        # For efficiency, we will do everything with ids (and not the
        # var objects themselves)
        #

        # Filter out any unused named expressions
        self.subexpression_order = list(
            filter(self.used_named_expressions.__contains__, self.subexpression_order)
        )

        # linear contribution by (constraint, objective, variable) component.
        # Keys are component id(), Values are dicts mapping variable
        # id() to linear coefficient.  All nonzeros in the component
        # (variables appearing in the linear and/or nonlinear
        # subexpressions) will appear in the dict.
        #
        # We initialize the map with any variables eliminated from
        # (presolved out of) the model (necessary so that
        # _categorize_vars will map eliminated vars to the current
        # vars).  Note that at the moment, we only consider linear
        # equality constraints in the presolve.  If that ever changes
        # (e.g., to support eliminating variables appearing linearly in
        # nonlinear equality constraints), then this logic will need to
        # be revisited.
        linear_by_comp = {_id: info.linear for _id, info in eliminated_vars.items()}

        # We need to categorize the named subexpressions first so that
        # we know their linear / nonlinear vars when we encounter them
        # in constraints / objectives
        self._categorize_vars(
            map(self.subexpression_cache.__getitem__, self.subexpression_order),
            linear_by_comp,
        )
        n_subexpressions = self._count_subexpression_occurrences()
        obj_vars_linear, obj_vars_nonlinear, obj_nnz_by_var = self._categorize_vars(
            objectives, linear_by_comp
        )
        con_vars_linear, con_vars_nonlinear, con_nnz_by_var = self._categorize_vars(
            constraints, linear_by_comp
        )

        if self.config.export_nonlinear_variables:
            for v in self.config.export_nonlinear_variables:
                # Note that because we have already walked all the
                # expressions, we have already "seen" all the variables
                # we will see, so we don't need to fill in any VarData
                # from IndexedVar containers here.
                if v.is_indexed():
                    _iter = v.values(sorter)
                else:
                    _iter = (v,)
                for _v in _iter:
                    _id = id(_v)
                    if _id not in var_map:
                        var_map[_id] = _v
                        var_bounds[_id] = _v.bounds
                    con_vars_nonlinear.add(_id)

        con_nnz = sum(con_nnz_by_var.values())
        timer.toc('Categorized model variables: %s nnz', con_nnz, level=logging.DEBUG)

        n_lcons = 0  # We do not yet support logical constraints

        # We need to check the SOS constraints before finalizing the
        # variable order because the SOS constraint *could* reference a
        # variable not yet seen in the model.
        for block in component_map[SOSConstraint]:
            for sos in block.component_data_objects(
                SOSConstraint, active=True, descend_into=False, sort=sorter
            ):
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
                    f"Integer, or Binary: Cannot write a legal NL file."
                )
        discrete_vars = binary_vars | integer_vars

        nonlinear_vars = con_vars_nonlinear | obj_vars_nonlinear
        linear_only_vars = (con_vars_linear | obj_vars_linear) - nonlinear_vars

        self.column_order = column_order = {_id: i for i, _id in enumerate(var_map)}
        variables = []
        #
        both_vars_nonlinear = con_vars_nonlinear & obj_vars_nonlinear
        if both_vars_nonlinear:
            variables.extend(
                sorted(
                    both_vars_nonlinear & continuous_vars, key=column_order.__getitem__
                )
            )
            variables.extend(
                sorted(
                    both_vars_nonlinear & discrete_vars, key=column_order.__getitem__
                )
            )
        #
        con_only_nonlinear_vars = con_vars_nonlinear - both_vars_nonlinear
        if con_only_nonlinear_vars:
            variables.extend(
                sorted(
                    con_only_nonlinear_vars & continuous_vars,
                    key=column_order.__getitem__,
                )
            )
            variables.extend(
                sorted(
                    con_only_nonlinear_vars & discrete_vars,
                    key=column_order.__getitem__,
                )
            )
        #
        obj_only_nonlinear_vars = obj_vars_nonlinear - both_vars_nonlinear
        if obj_vars_nonlinear:
            variables.extend(
                sorted(
                    obj_only_nonlinear_vars & continuous_vars,
                    key=column_order.__getitem__,
                )
            )
            variables.extend(
                sorted(
                    obj_only_nonlinear_vars & discrete_vars,
                    key=column_order.__getitem__,
                )
            )
        #
        if linear_only_vars:
            variables.extend(
                sorted(linear_only_vars - discrete_vars, key=column_order.__getitem__)
            )
            linear_binary_vars = linear_only_vars & binary_vars
            variables.extend(sorted(linear_binary_vars, key=column_order.__getitem__))
            linear_integer_vars = linear_only_vars & integer_vars
            variables.extend(sorted(linear_integer_vars, key=column_order.__getitem__))
        else:
            linear_binary_vars = linear_integer_vars = set()
        assert len(variables) == n_vars
        timer.toc(
            'Set row / column ordering: %s var [%s, %s, %s R/B/Z], '
            '%s con [%s, %s L/NL]',
            n_vars,
            len(continuous_vars),
            len(binary_vars),
            len(integer_vars),
            len(constraints),
            n_cons - n_nonlinear_cons,
            n_nonlinear_cons,
            level=logging.DEBUG,
        )

        # Update the column order (based on our reordering of the variables above).
        #
        # Note that as we allow var_map to contain "known" variables
        # that are not needed in the NL file (and column_order was
        # originally generated from var_map), we will rebuild the
        # column_order to *just* contain the variables that we are
        # sending to the NL.
        self.column_order = column_order = {_id: i for i, _id in enumerate(variables)}

        # Collect all defined SOSConstraints on the model
        if component_map[SOSConstraint]:
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
                        "these methods to define special ordered sets."
                    )
                suffix_data[name] = _SuffixData(name)
                suffix_data[name].datatype.add(Suffix.INT)
            sos_id = 0
            sosno = suffix_data['sosno']
            ref = suffix_data['ref']
            for block in reversed(component_map[SOSConstraint]):
                for sos in block.component_data_objects(
                    SOSConstraint, active=True, descend_into=False, sort=sorter
                ):
                    sos_id += 1
                    if sos.level == 1:
                        tag = sos_id
                    elif sos.level == 2:
                        tag = -sos_id
                    else:
                        raise ValueError(
                            f"SOSConstraint '{sos.name}' has sos "
                            f"type='{sos.level}', which is not supported "
                            "by the NL file interface"
                        )
                    try:
                        _items = sos.get_items()
                    except AttributeError:
                        # kernel doesn't provide the get_items API
                        _items = sos.items()
                    for v, r in _items:
                        sosno.store(v, tag)
                        ref.store(v, r)

        if suffix_data:
            row_order = {id(con[0]): i for i, con in enumerate(constraints)}
            obj_order = {id(obj[0]): i for i, obj in enumerate(objectives)}
            model_id = id(model)

        if symbolic_solver_labels:
            labeler = NameLabeler()
            row_labels = [labeler(info[0]) for info in constraints] + [
                labeler(info[0]) for info in objectives
            ]
            row_comments = [f'\t#{lbl}' for lbl in row_labels]
            col_labels = [labeler(var_map[_id]) for _id in variables]
            col_comments = [f'\t#{lbl}' for lbl in col_labels]
            self.var_id_to_nl = {
                _id: f'v{var_idx}{col_comments[var_idx]}'
                for var_idx, _id in enumerate(variables)
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
                _id: f"v{var_idx}" for var_idx, _id in enumerate(variables)
            }

        _vmap = self.var_id_to_nl
        if scale_model:
            template = self.template
            objective_scaling = [scaling_cache[id(info[0])] for info in objectives]
            constraint_scaling = [scaling_cache[id(info[0])] for info in constraints]
            variable_scaling = [scaling_factor(var_map[_id]) for _id in variables]
            for _id, scale in zip(variables, variable_scaling):
                if scale == 1:
                    continue
                # Update var_bounds to be scaled bounds
                if scale < 0:
                    # Note: reverse bounds for negative scaling factors
                    ub, lb = var_bounds[_id]
                else:
                    lb, ub = var_bounds[_id]
                if lb is not None:
                    lb *= scale
                if ub is not None:
                    ub *= scale
                var_bounds[_id] = lb, ub
                # Update _vmap to output scaled variables in NL expressions
                _vmap[_id] = (
                    template.division + _vmap[_id] + '\n' + template.const % scale
                ).rstrip()

        # Update any eliminated variables to point to the (potentially
        # scaled) substituted variables
        for _id, expr_info in eliminated_vars.items():
            nl, args, _ = expr_info.compile_repn(visitor)
            _vmap[_id] = nl.rstrip() % tuple(_vmap[_id] for _id in args)

        r_lines = [None] * n_cons
        for idx, (con, expr_info, lb, ub) in enumerate(constraints):
            if lb == ub:  # TBD: should this be within tolerance?
                if lb is None:
                    # type = 3  # -inf <= c <= inf
                    r_lines[idx] = "3"
                else:
                    # _type = 4  # L == c == U
                    r_lines[idx] = f"4 {lb - expr_info.const!r}"
                    n_equality += 1
            elif lb is None:
                # _type = 1  # c <= U
                r_lines[idx] = f"1 {ub - expr_info.const!r}"
            elif ub is None:
                # _type = 2  # L <= c
                r_lines[idx] = f"2 {lb - expr_info.const!r}"
            else:
                # _type = 0  # L <= c <= U
                r_lines[idx] = f"0 {lb - expr_info.const!r} {ub - expr_info.const!r}"
                n_ranges += 1
            expr_info.const = 0
            # FIXME: this is a HACK to be compatible with the NLv1
            # writer.  In the future, this writer should be expanded to
            # look for and process Complementarity components (assuming
            # that they are in an acceptable form).
            if hasattr(con, '_complementarity'):
                # _type = 5
                r_lines[idx] = f"5 {con._complementarity} {1+column_order[con._vid]}"
                if expr_info.nonlinear:
                    n_complementarity_nonlin += 1
                else:
                    n_complementarity_lin += 1
        if symbolic_solver_labels:
            for idx in range(len(constraints)):
                r_lines[idx] += row_comments[idx]

        timer.toc("Generated row/col labels & comments", level=logging.DEBUG)

        #
        # Print Header
        #
        # LINE 1
        #
        if visitor.encountered_string_arguments and 'b' not in getattr(
            ostream, 'mode', ''
        ):
            # Not all streams support tell()
            try:
                _written_bytes = ostream.tell()
            except IOError:
                _written_bytes = None

        line_1_txt = f"g3 1 1 0\t# problem {model.name}\n"
        ostream.write(line_1_txt)

        # If there were any string arguments, then we need to ensure
        # that ostream is not converting newlines to something other
        # than '\n'.  Binary files do not perform newline mapping (of
        # course, we will also need to map all the str to bytes for
        # binary-mode I/O).
        if visitor.encountered_string_arguments and 'b' not in getattr(
            ostream, 'mode', ''
        ):
            if _written_bytes is None:
                _written_bytes = 0
            else:
                _written_bytes = ostream.tell() - _written_bytes
            if not _written_bytes:
                if os.linesep != '\n':
                    logger.warning(
                        "Writing NL file containing string arguments to a "
                        "text output stream that does not support tell() on "
                        "a platform with default line endings other than "
                        "'\\n'. Current versions of the ASL "
                        "(through at least 20190605) require UNIX-style "
                        "newlines as terminators for string arguments: "
                        "it is possible that the ASL may refuse to read "
                        "the NL file."
                    )
            else:
                if ostream.encoding:
                    line_1_txt = line_1_txt.encode(ostream.encoding)
                if len(line_1_txt) != _written_bytes:
                    logger.error(
                        "Writing NL file containing string arguments to a "
                        "text output stream with line endings other than '\\n' "
                        "Current versions of the ASL "
                        "(through at least 20190605) require UNIX-style "
                        "newlines as terminators for string arguments."
                    )

        #
        # LINE 2
        #
        ostream.write(
            " %d %d %d %d %d \t"
            "# vars, constraints, objectives, ranges, eqns\n"
            % (n_vars, n_cons, n_objs, n_ranges, n_equality)
        )
        #
        # LINE 3
        #
        ostream.write(
            " %d %d %d %d %d %d\t"
            "# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n"
            % (
                n_nonlinear_cons,
                n_nonlinear_objs,
                # num linear complementarity constraints
                n_complementarity_lin,
                # num nonlinear complementarity constraints
                n_complementarity_nonlin,
                # num complementarities involving double inequalities
                n_complementarity_range,
                # num complemented variables with either a nonzero lower
                # bound or any upper bound (excluding ranges)
                n_complementarity_nz_var_lb,
            )
        )
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
            % (_n_con_vars, _n_obj_vars, _n_both_vars)
        )

        #
        # LINE 6
        #
        ostream.write(
            " 0 %d 0 1\t"
            "# linear network variables; functions; arith, flags\n"
            % (len(self.external_functions),)
        )
        #
        # LINE 7
        #
        ostream.write(
            " %d %d %d %d %d \t"
            "# discrete variables: binary, integer, nonlinear (b,c,o)\n"
            % (
                len(linear_binary_vars),
                len(linear_integer_vars),
                len(both_vars_nonlinear.intersection(discrete_vars)),
                len(con_vars_nonlinear.intersection(discrete_vars)),
                len(obj_vars_nonlinear.intersection(discrete_vars)),
            )
        )
        #
        # LINE 8
        #
        # objective info computed above
        ostream.write(
            " %d %d \t# nonzeros in Jacobian, obj. gradient\n"
            % (sum(con_nnz_by_var.values()), sum(obj_nnz_by_var.values()))
        )
        #
        # LINE 9
        #
        ostream.write(
            " %d %d\t# max name lengths: constraints, variables\n"
            % (
                max(map(len, row_labels), default=0),
                max(map(len, col_labels), default=0),
            )
        )
        #
        # LINE 10
        #
        ostream.write(
            " %d %d %d %d %d\t# common exprs: b,c,o,c1,o1\n" % tuple(n_subexpressions)
        )

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
            data.compile(column_order, row_order, obj_order, model_id)
            if len(data.datatype) > 1:
                raise ValueError(
                    "The NL file writer found multiple active export "
                    "suffix components with name '{name}' and different "
                    "datatypes. A single datatype must be declared."
                )
            _type = next(iter(data.datatype))
            if _type == Suffix.FLOAT:
                _float = 4
            elif _type == Suffix.INT:
                _float = 0
            else:
                raise ValueError(
                    "The NL file writer only supports export suffixes "
                    "declared with a numeric datatype.  Suffix "
                    f"component '{name}' declares type '{_type}'"
                )
            for _field, _vals in zip(
                range(4), (data.var, data.con, data.obj, data.prob)
            ):
                if not _vals:
                    continue
                ostream.write(f"S{_field|_float} {len(_vals)} {name}\n")
                # Note: _SuffixData.compile() guarantees the value is int/float
                ostream.write(
                    ''.join(f"{_id} {_vals[_id]!r}\n" for _id in sorted(_vals))
                )

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
            _con_id, _obj_id, _sub = self.subexpression_cache[_id][2]
            if _sub:
                # substitute expression directly into expression trees
                # and do NOT emit the V line
                continue
            target_expr = 0
            if _obj_id is None:
                target_expr = _con_id
            elif _con_id is None:
                target_expr = _obj_id
            if target_expr == 0:
                # Note: checking target_expr == 0 is equivalent to
                # testing "(_con_id is not None and _obj_id is not None)
                # or _con_id == 0 or _obj_id == 0"
                self._write_v_line(_id, 0)
            else:
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
                if symbolic_solver_labels:
                    ostream.write(
                        _expr.join(
                            f'C{i}{row_comments[i]}\n'
                            for i in range(row_idx, len(constraints))
                        )
                    )
                else:
                    ostream.write(
                        _expr.join(f'C{i}\n' for i in range(row_idx, len(constraints)))
                    )

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
            data = suffix_data['dual']
            data.compile(column_order, row_order, obj_order, model_id)
            if scale_model:
                if objectives:
                    if len(objectives) > 1:
                        logger.warning(
                            "Scaling model with dual suffixes and multiple "
                            "objectives.  Assuming that the duals are computed "
                            "against the first objective."
                        )
                    _obj_scale = objective_scaling[0]
                else:
                    _obj_scale = 1
                for i in data.con:
                    data.con[i] *= _obj_scale / constraint_scaling[i]
            if data.var:
                logger.warning("ignoring 'dual' suffix for Var types")
            if data.obj:
                logger.warning("ignoring 'dual' suffix for Objective types")
            if data.prob:
                logger.warning("ignoring 'dual' suffix for Model")
            if data.con:
                ostream.write(f"d{len(data.con)}\n")
                # Note: _SuffixData.compile() guarantees the value is int/float
                ostream.write(
                    ''.join(f"{_id} {data.con[_id]!r}\n" for _id in sorted(data.con))
                )

        #
        # "x" lines (variable initialization)
        #
        _init_lines = [
            (var_idx, val if val.__class__ in int_float else float(val))
            for var_idx, val in enumerate(var_map[_id].value for _id in variables)
            if val is not None
        ]
        if scale_model:
            _init_lines = [
                (var_idx, val * variable_scaling[var_idx])
                for var_idx, val in _init_lines
            ]
        ostream.write(
            'x%d%s\n'
            % (len(_init_lines), "\t# initial guess" if symbolic_solver_labels else '')
        )
        ostream.write(
            ''.join(
                f'{var_idx} {val!r}{col_comments[var_idx]}\n'
                for var_idx, val in _init_lines
            )
        )

        #
        # "r" lines (constraint bounds)
        #
        ostream.write(
            'r%s\n'
            % (
                "\t#%d ranges (rhs's)" % len(constraints)
                if symbolic_solver_labels
                else '',
            )
        )
        ostream.write("\n".join(r_lines))
        if r_lines:
            ostream.write("\n")

        #
        # "b" lines (variable bounds)
        #
        ostream.write(
            'b%s\n'
            % (
                "\t#%d bounds (on variables)" % len(variables)
                if symbolic_solver_labels
                else '',
            )
        )
        for var_idx, _id in enumerate(variables):
            lb, ub = var_bounds[_id]
            if lb == ub:
                if lb is None:  # unbounded
                    ostream.write(f"3{col_comments[var_idx]}\n")
                else:  # ==
                    ostream.write(f"4 {lb!r}{col_comments[var_idx]}\n")
            elif lb is None:  # var <= ub
                ostream.write(f"1 {ub!r}{col_comments[var_idx]}\n")
            elif ub is None:  # lb <= body
                ostream.write(f"2 {lb!r}{col_comments[var_idx]}\n")
            else:  # lb <= body <= ub
                ostream.write(f"0 {lb!r} {ub!r}{col_comments[var_idx]}\n")

        #
        # "k" lines (column offsets in Jacobian NNZ)
        #
        ostream.write(
            'k%d%s\n'
            % (
                len(variables) - 1,
                "\t#intermediate Jacobian column lengths"
                if symbolic_solver_labels
                else '',
            )
        )
        ktot = 0
        for var_idx, _id in enumerate(variables[:-1]):
            ktot += con_nnz_by_var.get(_id, 0)
            ostream.write(f"{ktot}\n")

        #
        # "J" lines (non-empty terms in the Jacobian)
        #
        for row_idx, info in enumerate(constraints):
            linear = info[1].linear
            # ASL will fail on "J<N> 0", so if there are no coefficients
            # (e.g., a nonlinear-only constraint), then skip this entry
            if not linear:
                continue
            if scale_model:
                for _id, val in linear.items():
                    linear[_id] /= scaling_cache[_id]
            ostream.write(f'J{row_idx} {len(linear)}{row_comments[row_idx]}\n')
            for _id in sorted(linear, key=column_order.__getitem__):
                ostream.write(f'{column_order[_id]} {linear[_id]!r}\n')

        #
        # "G" lines (non-empty terms in the Objective)
        #
        for obj_idx, info in enumerate(objectives):
            linear = info[1].linear
            # ASL will fail on "G<N> 0", so if there are no coefficients
            # (e.g., a constant objective), then skip this entry
            if not linear:
                continue
            if scale_model:
                for _id, val in linear.items():
                    linear[_id] /= scaling_cache[_id]
            ostream.write(f'G{obj_idx} {len(linear)}{row_comments[obj_idx + n_cons]}\n')
            for _id in sorted(linear, key=column_order.__getitem__):
                ostream.write(f'{column_order[_id]} {linear[_id]!r}\n')

        # Generate the return information
        eliminated_vars = [
            (var_map[_id], expr_info) for _id, expr_info in eliminated_vars.items()
        ]
        eliminated_vars.reverse()
        if scale_model:
            scaling = ScalingFactors(
                variables=variable_scaling,
                constraints=constraint_scaling,
                objectives=objective_scaling,
            )
        else:
            scaling = None
        info = NLWriterInfo(
            var=[var_map[_id] for _id in variables],
            con=[info[0] for info in constraints],
            obj=[info[0] for info in objectives],
            external_libs=sorted(amplfunc_libraries),
            row_labels=row_labels,
            col_labels=col_labels,
            eliminated_vars=eliminated_vars,
            scaling=scaling,
        )
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
                    if _id in nonlinear_vars:
                        continue
                    if _id in linear_by_comp:
                        nonlinear_vars.update(linear_by_comp[_id])
                    else:
                        nonlinear_vars.add(_id)
                # Recreate nz if this component has both linear and
                # nonlinear components.
                if expr_info.linear:
                    # Ensure any variables that only appear nonlinearly
                    # in the expression have 0's in the linear dict
                    for i in filterfalse(linear_vars.__contains__, nonlinear_vars):
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

    def _count_subexpression_occurrences(self):
        """Categorize named subexpressions based on where they are used.

        This iterates through the `subexpression_order` and categorizes
        each _id based on where it is used (1 constraint, many
        constraints, 1 objective, many objectives, constraints and
        objectives).

        """
        # Group them into:
        #   [ used in both objectives and constraints,
        #     used by more than one constraint (but no objectives),
        #     used by more than one objective (but no constraints),
        #     used by one constraint,
        #     used by one objective ]
        n_subexpressions = [0] * 5
        for info in map(
            itemgetter(2),
            map(self.subexpression_cache.__getitem__, self.subexpression_order),
        ):
            if info[2]:
                pass
            elif info[1] is None:
                # assert info[0] is not None:
                n_subexpressions[3 if info[0] else 1] += 1
            elif info[0] is None:
                n_subexpressions[4 if info[1] else 2] += 1
            else:
                n_subexpressions[0] += 1
        return n_subexpressions

    def _linear_presolve(self, comp_by_linear_var, lcon_by_linear_nnz, var_bounds):
        eliminated_vars = {}
        eliminated_cons = set()
        if not self.config.linear_presolve:
            return eliminated_cons, eliminated_vars

        # We need to record all named expressions with linear components
        # so that any eliminated variables are removed from them.
        for expr, info, _ in self.subexpression_cache.values():
            if not info.linear:
                continue
            expr_id = id(expr)
            for _id in info.linear:
                comp_by_linear_var[_id].append((expr_id, info))

        fixed_vars = [
            _id for _id, (lb, ub) in var_bounds.items() if lb == ub and lb is not None
        ]
        var_map = self.var_map
        substitutions_by_linear_var = defaultdict(set)
        template = self.template
        one_var = lcon_by_linear_nnz[1]
        two_var = lcon_by_linear_nnz[2]
        while 1:
            if fixed_vars:
                _id = fixed_vars.pop()
                a = x = None
                b, _ = var_bounds[_id]
                logger.debug("NL presolve: bounds fixed %s := %s", var_map[_id], b)
                eliminated_vars[_id] = AMPLRepn(b, {}, None)
            elif one_var:
                con_id, info = one_var.popitem()
                expr_info, lb = info
                _id, coef = expr_info.linear.popitem()
                # substituting _id with a*x + b
                a = x = None
                b = expr_info.const = (lb - expr_info.const) / coef
                logger.debug("NL presolve: substituting %s := %s", var_map[_id], b)
                eliminated_vars[_id] = expr_info
                lb, ub = var_bounds[_id]
                if (lb is not None and lb - b > TOL) or (
                    ub is not None and ub - b < -TOL
                ):
                    raise InfeasibleConstraintException(
                        "model contains a trivially infeasible variable "
                        f"'{var_map[_id].name}' (presolved to a value of "
                        f"{b} outside bounds [{lb}, {ub}])."
                    )
                eliminated_cons.add(con_id)
            elif two_var:
                con_id, info = two_var.popitem()
                expr_info, lb = info
                _id, coef = expr_info.linear.popitem()
                id2, coef2 = expr_info.linear.popitem()
                #
                id2_isdiscrete = var_map[id2].domain.isdiscrete()
                if var_map[_id].domain.isdiscrete() ^ id2_isdiscrete:
                    # if only one variable is discrete, then we need to
                    # substiitute out the other
                    if id2_isdiscrete:
                        _id, id2 = id2, _id
                        coef, coef2 = coef2, coef
                else:
                    # In an attempt to improve numerical stability, we will
                    # solve for (and substitute out) the variable with the
                    # coefficient closer to +/-1)
                    log_coef = _log10(abs(coef))
                    log_coef2 = _log10(abs(coef2))
                    if abs(log_coef2) < abs(log_coef) or (
                        log_coef2 == -log_coef and log_coef2 < log_coef
                    ):
                        _id, id2 = id2, _id
                        coef, coef2 = coef2, coef
                # substituting _id with a*x + b
                a = -coef2 / coef
                x = id2
                b = expr_info.const = (lb - expr_info.const) / coef
                expr_info.linear[x] = a
                substitutions_by_linear_var[x].add(_id)
                eliminated_vars[_id] = expr_info
                logger.debug(
                    "NL presolve: substituting %s := %s*%s + %s",
                    var_map[_id],
                    a,
                    var_map[x],
                    b,
                )
                # Tighten variable bounds
                x_lb, x_ub = var_bounds[x]
                lb, ub = var_bounds[_id]
                if lb is not None:
                    lb = (lb - b) / a
                if ub is not None:
                    ub = (ub - b) / a
                if a < 0:
                    lb, ub = ub, lb
                if x_lb is None or (lb is not None and lb > x_lb):
                    x_lb = lb
                if x_ub is None or (ub is not None and ub < x_ub):
                    x_ub = ub
                var_bounds[x] = x_lb, x_ub
                if x_lb == x_ub and x_lb is not None:
                    fixed_vars.append(x)
                eliminated_cons.add(con_id)
            else:
                return eliminated_cons, eliminated_vars
            for con_id, expr_info in comp_by_linear_var[_id]:
                # Note that if we were aggregating (i.e., _id was
                # from two_var), then one of these info's will be
                # for the constraint we just eliminated.  In this
                # case, _id will no longer be in expr_info.linear - so c
                # will be 0 - thereby preventing us from re-updating
                # the expression.  We still want it to persist so
                # that if later substitutions replace x with
                # something else, then the expr_info gets updated
                # appropriately (that expr_info is persisting in the
                # eliminated_vars dict - and we will use that to
                # update other linear expressions later.)
                c = expr_info.linear.pop(_id, 0)
                expr_info.const += c * b
                if x in expr_info.linear:
                    expr_info.linear[x] += c * a
                elif a:
                    expr_info.linear[x] = c * a
                    # replacing _id with x... NNZ is not changing,
                    # but we need to remember that x is now part of
                    # this constraint
                    comp_by_linear_var[x].append((con_id, expr_info))
                    continue
                # NNZ has been reduced by 1
                nnz = len(expr_info.linear)
                _old = lcon_by_linear_nnz[nnz + 1]
                if con_id in _old:
                    lcon_by_linear_nnz[nnz][con_id] = _old.pop(con_id)
            # If variables were replaced by the variable that
            # we are currently eliminating, then we need to update
            # the representation of those variables
            for resubst in substitutions_by_linear_var.pop(_id, ()):
                expr_info = eliminated_vars[resubst]
                c = expr_info.linear.pop(_id, 0)
                expr_info.const += c * b
                if x in expr_info.linear:
                    expr_info.linear[x] += c * a
                elif a:
                    expr_info.linear[x] = c * a

    def _record_named_expression_usage(self, named_exprs, src, comp_type):
        self.used_named_expressions.update(named_exprs)
        src = id(src)
        for _id in named_exprs:
            info = self.subexpression_cache[_id][2]
            if info[comp_type] is None:
                info[comp_type] = src
            elif info[comp_type] != src:
                info[comp_type] = 0

    def _write_nl_expression(self, repn, include_const):
        # Note that repn.mult should always be 1 (the AMPLRepn was
        # compiled before this point).  Omitting the assertion for
        # efficiency.
        # assert repn.mult == 1
        #
        # Note that repn.const should always be a int/float (it has
        # already been compiled)
        if repn.nonlinear:
            nl, args = repn.nonlinear
            if include_const and repn.const:
                # Add the constant to the NL expression.  AMPL adds the
                # constant as the second argument, so we will too.
                nl = self.template.binary_sum + nl + self.template.const % repn.const
            self.ostream.write(nl % tuple(map(self.var_id_to_nl.__getitem__, args)))
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
        self.var_id_to_nl[expr_id] = f"v{self.next_V_line_id}{lbl}"
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
    __slots__ = ('nl', 'mult', 'const', 'linear', 'nonlinear', 'named_exprs')

    ActiveVisitor = None

    def __init__(self, const, linear, nonlinear):
        self.nl = None
        self.mult = 1
        self.const = const
        self.linear = linear
        if nonlinear is None:
            self.nonlinear = self.named_exprs = None
        else:
            nl, nl_args, self.named_exprs = nonlinear
            self.nonlinear = nl, nl_args

    def __str__(self):
        return (
            f'AMPLRepn(mult={self.mult}, const={self.const}, '
            f'linear={self.linear}, nonlinear={self.nonlinear}, '
            f'nl={self.nl}, named_exprs={self.named_exprs})'
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return other.__class__ is AMPLRepn and (
            self.nl == other.nl
            and self.mult == other.mult
            and self.const == other.const
            and self.linear == other.linear
            and self.nonlinear == other.nonlinear
            and self.named_exprs == other.named_exprs
        )

    def __hash__(self):
        # Approximation of the Python default object hash
        # (4 LSB are rolled to the MSB to reduce hash collisions)
        return id(self) // 16 + (
            (id(self) & 15) << 8 * ctypes.sizeof(ctypes.c_void_p) - 4
        )

    def duplicate(self):
        ans = self.__class__.__new__(self.__class__)
        ans.nl = self.nl
        ans.mult = self.mult
        ans.const = self.const
        ans.linear = None if self.linear is None else dict(self.linear)
        ans.nonlinear = self.nonlinear
        ans.named_exprs = self.named_exprs
        return ans

    def compile_repn(self, visitor, prefix='', args=None, named_exprs=None):
        template = visitor.template
        if self.mult != 1:
            if self.mult == -1:
                prefix += template.negation
            else:
                prefix += template.multiplier % self.mult
            self.mult = 1
        if self.named_exprs is not None:
            if named_exprs is None:
                named_exprs = set(self.named_exprs)
            else:
                named_exprs.update(self.named_exprs)
        if self.nl is not None:
            # This handles both named subexpressions and embedded
            # non-numeric (e.g., string) arguments.
            nl, nl_args = self.nl
            if prefix:
                nl = prefix + nl
            if args is not None:
                assert args is not nl_args
                args.extend(nl_args)
            else:
                args = list(nl_args)
            if nl_args:
                # For string arguments, nl_args is an empty tuple and
                # self.named_exprs is None.  For named subexpressions,
                # we are guaranteed that named_exprs is NOT None.  We
                # need to ensure that the named subexpression that we
                # are returning is added to the named_exprs set.
                named_exprs.update(nl_args)
            return nl, args, named_exprs

        if args is None:
            args = []
        if self.linear:
            nterms = -len(args)
            _v_template = template.var
            _m_template = template.monomial
            # Because we are compiling this expression (into a NL
            # expression), we will go ahead and filter the 0*x terms
            # from the expression.  Note that the args are accumulated
            # by side-effect, which prevents iterating over the linear
            # terms twice.
            nl_sum = ''.join(
                args.append(v) or (_v_template if c == 1 else _m_template % c)
                for v, c in self.linear.items()
                if c
            )
            nterms += len(args)
        else:
            nterms = 0
            nl_sum = ''
        if self.nonlinear:
            if self.nonlinear.__class__ is list:
                nterms += len(self.nonlinear)
                nl_sum += ''.join(map(itemgetter(0), self.nonlinear))
                deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)
            else:
                nterms += 1
                nl_sum += self.nonlinear[0]
                args.extend(self.nonlinear[1])
        if self.const:
            nterms += 1
            nl_sum += template.const % self.const

        if nterms > 2:
            return (prefix + (template.nary_sum % nterms) + nl_sum, args, named_exprs)
        elif nterms == 2:
            return prefix + template.binary_sum + nl_sum, args, named_exprs
        elif nterms == 1:
            return prefix + nl_sum, args, named_exprs
        else:  # nterms == 0
            return prefix + (template.const % 0), args, named_exprs

    def compile_nonlinear_fragment(self, visitor):
        if not self.nonlinear:
            self.nonlinear = None
            return
        args = []
        nterms = len(self.nonlinear)
        nl_sum = ''.join(map(itemgetter(0), self.nonlinear))
        deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)

        if nterms > 2:
            self.nonlinear = (visitor.template.nary_sum % nterms) + nl_sum, args
        elif nterms == 2:
            self.nonlinear = visitor.template.binary_sum + nl_sum, args
        else:  # nterms == 1:
            self.nonlinear = nl_sum, args

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
        # assert self.mult == 1
        _type = other[0]
        if _type is _MONOMIAL:
            _, v, c = other
            if v in self.linear:
                self.linear[v] += c
            else:
                self.linear[v] = c
        elif _type is _GENERAL:
            _, other = other
            if other.nl is not None and other.nl[1]:
                if other.linear:
                    # This is a named expression with both a linear and
                    # nonlinear component.  We want to merge it with
                    # this AMPLRepn, preserving the named expression for
                    # only the nonlinear component (merging the linear
                    # component with this AMPLRepn).
                    pass
                else:
                    # This is a nonlinear-only named expression,
                    # possibly with a multiplier that is not 1.  Compile
                    # it and append it (this both resolves the
                    # multiplier, and marks the named expression as
                    # having been used)
                    other = other.compile_repn(
                        self.ActiveVisitor, '', None, self.named_exprs
                    )
                    nl, nl_args, self.named_exprs = other
                    self.nonlinear.append((nl, nl_args))
                    return
            if other.named_exprs is not None:
                if self.named_exprs is None:
                    self.named_exprs = set(other.named_exprs)
                else:
                    self.named_exprs.update(other.named_exprs)
            if other.mult != 1:
                mult = other.mult
                self.const += mult * other.const
                if other.linear:
                    linear = self.linear
                    for v, c in other.linear.items():
                        if v in linear:
                            linear[v] += c * mult
                        else:
                            linear[v] = c * mult
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
                    linear = self.linear
                    for v, c in other.linear.items():
                        if v in linear:
                            linear[v] += c
                        else:
                            linear[v] = c
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
        'log': 'o43\t#log\n',
        'log10': 'o42\t#log10\n',
        'sin': 'o41\t#sin\n',
        'cos': 'o46\t#cos\n',
        'tan': 'o38\t#tan\n',
        'sinh': 'o40\t#sinh\n',
        'cosh': 'o45\t#cosh\n',
        'tanh': 'o37\t#tanh\n',
        'asin': 'o51\t#asin\n',
        'acos': 'o53\t#acos\n',
        'atan': 'o49\t#atan\n',
        'exp': 'o44\t#exp\n',
        'sqrt': 'o39\t#sqrt\n',
        'asinh': 'o50\t#asinh\n',
        'acosh': 'o52\t#acosh\n',
        'atanh': 'o47\t#atanh\n',
        'ceil': 'o14\t#ceil\n',
        'floor': 'o13\t#floor\n',
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
    var = '%s\n'  # NOTE: to support scaling, we do NOT include the 'v' here
    const = 'n%r\n'
    string = 'h%d:%s\n'
    monomial = product + const + var.replace('%', '%%')
    multiplier = product + const

    _create_strict_inequality_map(vars())


def _strip_template_comments(vars_, base_):
    vars_['unary'] = {k: v[: v.find('\t#')] + '\n' for k, v in base_.unary.items()}
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
        _, v, c = data
        if c:
            return AMPLRepn(0, {v: c}, None)
        else:
            return AMPLRepn(0, None, None)
    elif data[0] is _CONSTANT:
        return AMPLRepn(data[1], None, None)
    else:
        raise DeveloperError("unknown result type")


def handle_negation_node(visitor, node, arg1):
    if arg1[0] is _MONOMIAL:
        return (_MONOMIAL, arg1[1], -1 * arg1[2])
    elif arg1[0] is _GENERAL:
        arg1[1].mult *= -1
        return arg1
    elif arg1[0] is _CONSTANT:
        return (_CONSTANT, -1 * arg1[1])
    else:
        raise RuntimeError("%s: %s" % (type(arg1[0]), arg1))


def handle_product_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        arg2, arg1 = arg1, arg2
    if arg1[0] is _CONSTANT:
        mult = arg1[1]
        if not mult:
            # simplify multiplication by 0 (if arg2 is zero, the
            # simplification happens when we evaluate the constant
            # below).  Note that this is not IEEE-754 compliant, and
            # will map 0*inf and 0*nan to 0 (and not to nan).  We are
            # including this for backwards compatibility with the NLv1
            # writer, but arguably we should deprecate/remove this
            # "feature" in the future.
            if arg2[0] is _CONSTANT:
                _prod = mult * arg2[1]
                if _prod:
                    deprecation_warning(
                        f"Encountered {mult}*{str(arg2[1])} in expression tree.  "
                        "Mapping the NaN result to 0 for compatibility "
                        "with the nl_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.4.3',
                    )
                    _prod = 0
                return (_CONSTANT, _prod)
            return arg1
        if mult == 1:
            return arg2
        elif arg2[0] is _MONOMIAL:
            if mult != mult:
                # This catches mult (i.e., arg1) == nan
                return arg1
            return (_MONOMIAL, arg2[1], mult * arg2[2])
        elif arg2[0] is _GENERAL:
            if mult != mult:
                # This catches mult (i.e., arg1) == nan
                return arg1
            arg2[1].mult *= mult
            return arg2
        elif arg2[0] is _CONSTANT:
            if not arg2[1]:
                # Simplify multiplication by 0; see note above about
                # IEEE-754 incompatibility.
                _prod = mult * arg2[1]
                if _prod:
                    deprecation_warning(
                        f"Encountered {str(mult)}*{arg2[1]} in expression tree.  "
                        "Mapping the NaN result to 0 for compatibility "
                        "with the nl_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.4.3',
                    )
                    _prod = 0
                return (_CONSTANT, _prod)
            return (_CONSTANT, mult * arg2[1])
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.product
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_division_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        div = arg2[1]
        if div == 1:
            return arg1
        if arg1[0] is _MONOMIAL:
            tmp = apply_node_operation(node, (arg1[2], div))
            if tmp != tmp:
                # This catches if the coefficient division results in nan
                return _CONSTANT, tmp
            return (_MONOMIAL, arg1[1], tmp)
        elif arg1[0] is _GENERAL:
            tmp = apply_node_operation(node, (arg1[1].mult, div))
            if tmp != tmp:
                # This catches if the multiplier division results in nan
                return _CONSTANT, tmp
            arg1[1].mult = tmp
            return arg1
        elif arg1[0] is _CONSTANT:
            return _CONSTANT, apply_node_operation(node, (arg1[1], div))
    elif arg1[0] is _CONSTANT and not arg1[1]:
        return _CONSTANT, 0
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.division
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_pow_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        if arg1[0] is _CONSTANT:
            ans = apply_node_operation(node, (arg1[1], arg2[1]))
            if ans.__class__ in native_complex_types:
                ans = complex_number_error(ans, visitor, node)
            return _CONSTANT, ans
        elif not arg2[1]:
            return _CONSTANT, 1
        elif arg2[1] == 1:
            return arg1
    nonlin = node_result_to_amplrepn(arg1).compile_repn(visitor, visitor.template.pow)
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_abs_node(visitor, node, arg1):
    if arg1[0] is _CONSTANT:
        return (_CONSTANT, abs(arg1[1]))
    nonlin = node_result_to_amplrepn(arg1).compile_repn(visitor, visitor.template.abs)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_unary_node(visitor, node, arg1):
    if arg1[0] is _CONSTANT:
        return _CONSTANT, apply_node_operation(node, (arg1[1],))
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.unary[node.name]
    )
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_exprif_node(visitor, node, arg1, arg2, arg3):
    if arg1[0] is _CONSTANT:
        if arg1[1]:
            return arg2
        else:
            return arg3
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.exprif
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    nonlin = node_result_to_amplrepn(arg3).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_equality_node(visitor, node, arg1, arg2):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT:
        return (_CONSTANT, arg1[1] == arg2[1])
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.equality
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_inequality_node(visitor, node, arg1, arg2):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT:
        return (_CONSTANT, node._apply_operation((arg1[1], arg2[1])))
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.strict_inequality_map[node.strict]
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT and arg3[0] is _CONSTANT:
        return (_CONSTANT, node._apply_operation((arg1[1], arg2[1], arg3[1])))
    op = visitor.template.strict_inequality_map[node.strict]
    nl, args, named = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.and_expr + op[0]
    )
    nl2, args2, named = node_result_to_amplrepn(arg2).compile_repn(
        visitor, '', None, named
    )
    nl += nl2 + op[1] + nl2
    args.extend(args2)
    args.extend(args2)
    nonlin = node_result_to_amplrepn(arg3).compile_repn(visitor, nl, args, named)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_named_expression_node(visitor, node, arg1):
    _id = id(node)
    # Note that while named subexpressions ('defined variables' in the
    # ASL NL file vernacular) look like variables, they are not allowed
    # to appear in the 'linear' portion of a constraint / objective
    # definition.  We will return this as a "var" template, but
    # wrapped in the nonlinear portion of the expression tree.
    repn = node_result_to_amplrepn(arg1)

    # A local copy of the expression source list.  This will be updated
    # later if the same Expression node is encountered in another
    # expression tree.
    #
    # This is a 3-tuple [con_id, obj_id, substitute_expression].  If the
    # expression is used by more than 1 constraint / objective, then the
    # id is set to 0.  If it is not used by any, then it is None.
    # substitute_expression is a bool indicating if this named
    # subexpression tree should be directly substituted into any
    # expression tree that references this node (i.e., do NOT emit the V
    # line).
    expression_source = [None, None, False]
    # Record this common expression
    visitor.subexpression_cache[_id] = (
        # 0: the "component" that generated this expression ID
        node,
        # 1: the common subexpression (to be written out)
        repn,
        # 2: the source usage information for this subexpression:
        #    [(con_id, obj_id, substitute); see above]
        expression_source,
    )

    if not visitor.use_named_exprs:
        return _GENERAL, repn.duplicate()

    mult, repn.mult = repn.mult, 1
    if repn.named_exprs is None:
        repn.named_exprs = set()

    # When converting this shared subexpression to a (nonlinear)
    # node, we want to just reference this subexpression:
    repn.nl = (visitor.template.var, (_id,))

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
            # If this expression has both linear and nonlinear
            # components, we will follow the ASL convention and break
            # the named subexpression into two named subexpressions: one
            # that is only the nonlinear component and one that has the
            # const/linear component (and references the first).  This
            # will allow us to propagate linear coefficients up from
            # named subexpressions when appropriate.
            sub_node = NLFragment(repn, node)
            sub_id = id(sub_node)
            sub_repn = AMPLRepn(0, None, None)
            sub_repn.nonlinear = repn.nonlinear
            sub_repn.nl = (visitor.template.var, (sub_id,))
            sub_repn.named_exprs = set(repn.named_exprs)

            repn.named_exprs.add(sub_id)
            repn.nonlinear = sub_repn.nl

            # See above for the meaning of this source information
            nl_info = list(expression_source)
            visitor.subexpression_cache[sub_id] = (sub_node, sub_repn, nl_info)
            # It is important that the NL subexpression comes before the
            # main named expression:
            visitor.subexpression_order.append(sub_id)
        else:
            nl_info = expression_source
    else:
        repn.nonlinear = None
        if repn.linear:
            if (
                not repn.const
                and len(repn.linear) == 1
                and next(iter(repn.linear.values())) == 1
            ):
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

    if mult != 1:
        repn.const *= mult
        if repn.linear:
            _lin = repn.linear
            for v in repn.linear:
                _lin[v] *= mult
        if repn.nonlinear:
            if mult == -1:
                prefix = visitor.template.negation
            else:
                prefix = visitor.template.multiplier % mult
            repn.nonlinear = prefix + repn.nonlinear[0], repn.nonlinear[1]

    if expression_source[2]:
        if repn.linear:
            return (_MONOMIAL, next(iter(repn.linear)), 1)
        else:
            return (_CONSTANT, repn.const)

    # Defer recording this _id until after we know that this repn will
    # not be directly substituted (and to ensure that the NL fragment is
    # added to the order first).
    visitor.subexpression_order.append(_id)

    return (_GENERAL, repn.duplicate())


def handle_external_function_node(visitor, node, *args):
    func = node._fcn._function
    # There is a special case for external functions: these are the only
    # expressions that can accept string arguments. As we currently pass
    # these as 'precompiled' general NL fragments, the normal trap for
    # constant subexpressions will miss constant external function calls
    # that contain strings.  We will catch that case here.
    if all(
        arg[0] is _CONSTANT or (arg[0] is _GENERAL and arg[1].nl and not arg[1].nl[1])
        for arg in args
    ):
        arg_list = [arg[1] if arg[0] is _CONSTANT else arg[1].const for arg in args]
        return _CONSTANT, apply_node_operation(node, arg_list)
    if func in visitor.external_functions:
        if node._fcn._library != visitor.external_functions[func][1]._library:
            raise RuntimeError(
                "The same external function name (%s) is associated "
                "with two different libraries (%s through %s, and %s "
                "through %s).  The ASL solver will fail to link "
                "correctly."
                % (
                    func,
                    visitor.external_byFcn[func]._library,
                    visitor.external_byFcn[func]._library.name,
                    node._fcn._library,
                    node._fcn.name,
                )
            )
    else:
        visitor.external_functions[func] = (len(visitor.external_functions), node._fcn)
    comment = f'\t#{node.local_name}' if visitor.symbolic_solver_labels else ''
    nonlin = node_result_to_amplrepn(args[0]).compile_repn(
        visitor,
        visitor.template.external_fcn
        % (visitor.external_functions[func][0], len(args), comment),
    )
    for arg in args[1:]:
        nonlin = node_result_to_amplrepn(arg).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


_operator_handles = ExitNodeDispatcher(
    {
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
        Expression: handle_named_expression_node,
        ExternalFunctionExpression: handle_external_function_node,
        # These are handled explicitly in beforeChild():
        # LinearExpression: handle_linear_expression,
        # SumExpression: handle_sum_expression,
        #
        # Note: MonomialTermExpression is only hit when processing NPV
        # subexpressions that raise errors (e.g., log(0) * m.x), so no
        # special processing is needed [it is just a product expression]
        MonomialTermExpression: handle_product_node,
    }
)


class AMPLBeforeChildDispatcher(BeforeChildDispatcher):
    __slots__ = ()

    def __init__(self):
        # Special linear / summation expressions
        self[MonomialTermExpression] = self._before_monomial
        self[LinearExpression] = self._before_linear
        self[SumExpression] = self._before_general_expression

    @staticmethod
    def _record_var(visitor, var):
        # We always add all indices to the var_map at once so that
        # we can honor deterministic ordering of unordered sets
        # (because the user could have iterated over an unordered
        # set when constructing an expression, thereby altering the
        # order in which we would see the variables)
        vm = visitor.var_map
        try:
            _iter = var.parent_component().values(visitor.sorter)
        except AttributeError:
            # Note that this only works for the AML, as kernel does not
            # provide a parent_component()
            _iter = (var,)
        for v in _iter:
            if v.fixed:
                continue
            vm[id(v)] = v

    @staticmethod
    def _before_string(visitor, child):
        visitor.encountered_string_arguments = True
        ans = AMPLRepn(child, None, None)
        ans.nl = (visitor.template.string % (len(child), child), ())
        return False, (_GENERAL, ans)

    @staticmethod
    def _before_var(visitor, child):
        _id = id(child)
        if _id not in visitor.var_map:
            if child.fixed:
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(_id, child)
                return False, (_CONSTANT, visitor.fixed_vars[_id])
            _before_child_handlers._record_var(visitor, child)
        return False, (_MONOMIAL, _id, 1)

    @staticmethod
    def _before_monomial(visitor, child):
        #
        # The following are performance optimizations for common
        # situations (Monomial terms and Linear expressions)
        #
        arg1, arg2 = child._args_
        if arg1.__class__ not in native_types:
            try:
                arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
            except (ValueError, ArithmeticError):
                return True, None

        # Trap multiplication by 0 and nan.
        if not arg1:
            if arg2.fixed:
                _id = id(arg2)
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(id(arg2), arg2)
                arg2 = visitor.fixed_vars[_id]
                if arg2 != arg2:
                    deprecation_warning(
                        f"Encountered {arg1}*{arg2} in expression tree.  "
                        "Mapping the NaN result to 0 for compatibility "
                        "with the nl_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.4.3',
                    )
            return False, (_CONSTANT, arg1)

        _id = id(arg2)
        if _id not in visitor.var_map:
            if arg2.fixed:
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(_id, arg2)
                return False, (_CONSTANT, arg1 * visitor.fixed_vars[_id])
            _before_child_handlers._record_var(visitor, arg2)
        return False, (_MONOMIAL, _id, arg1)

    @staticmethod
    def _before_linear(visitor, child):
        # Because we are going to modify the LinearExpression in this
        # walker, we need to make a copy of the arg list from the original
        # expression tree.
        var_map = visitor.var_map
        const = 0
        linear = {}
        for arg in child.args:
            if arg.__class__ is MonomialTermExpression:
                arg1, arg2 = arg._args_
                if arg1.__class__ not in native_types:
                    try:
                        arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
                    except (ValueError, ArithmeticError):
                        return True, None

                # Trap multiplication by 0 and nan.
                if not arg1:
                    if arg2.fixed:
                        arg2 = visitor.check_constant(arg2.value, arg2)
                        if arg2 != arg2:
                            deprecation_warning(
                                f"Encountered {arg1}*{str(arg2.value)} in expression "
                                "tree.  Mapping the NaN result to 0 for compatibility "
                                "with the nl_v1 writer.  In the future, this NaN "
                                "will be preserved/emitted to comply with IEEE-754.",
                                version='6.4.3',
                            )
                    continue

                _id = id(arg2)
                if _id not in var_map:
                    if arg2.fixed:
                        if _id not in visitor.fixed_vars:
                            visitor.cache_fixed_var(_id, arg2)
                        const += arg1 * visitor.fixed_vars[_id]
                        continue
                    _before_child_handlers._record_var(visitor, arg2)
                    linear[_id] = arg1
                elif _id in linear:
                    linear[_id] += arg1
                else:
                    linear[_id] = arg1
            elif arg.__class__ in native_types:
                const += arg
            else:
                try:
                    const += visitor.check_constant(visitor.evaluate(arg), arg)
                except (ValueError, ArithmeticError):
                    return True, None

        if linear:
            return False, (_GENERAL, AMPLRepn(const, linear, None))
        else:
            return False, (_CONSTANT, const)

    @staticmethod
    def _before_named_expression(visitor, child):
        _id = id(child)
        if _id in visitor.subexpression_cache:
            obj, repn, info = visitor.subexpression_cache[_id]
            if info[2]:
                if repn.linear:
                    return False, (_MONOMIAL, next(iter(repn.linear)), 1)
                else:
                    return False, (_CONSTANT, repn.const)
            return False, (_GENERAL, repn.duplicate())
        else:
            return True, None


_before_child_handlers = AMPLBeforeChildDispatcher()


class AMPLRepnVisitor(StreamBasedExpressionVisitor):
    def __init__(
        self,
        template,
        subexpression_cache,
        subexpression_order,
        external_functions,
        var_map,
        used_named_expressions,
        symbolic_solver_labels,
        use_named_exprs,
        sorter,
    ):
        super().__init__()
        self.template = template
        self.subexpression_cache = subexpression_cache
        self.subexpression_order = subexpression_order
        self.external_functions = external_functions
        self.active_expression_source = None
        self.var_map = var_map
        self.used_named_expressions = used_named_expressions
        self.symbolic_solver_labels = symbolic_solver_labels
        self.use_named_exprs = use_named_exprs
        self.encountered_string_arguments = False
        self.fixed_vars = {}
        self._eval_expr_visitor = _EvaluationVisitor(True)
        self.evaluate = self._eval_expr_visitor.dfs_postorder_stack
        self.sorter = sorter

    def check_constant(self, ans, obj):
        if ans.__class__ not in native_numeric_types:
            # None can be returned from uninitialized Var/Param objects
            if ans is None:
                return InvalidNumber(
                    None, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
                )
            if ans.__class__ is InvalidNumber:
                return ans
            elif ans.__class__ in native_complex_types:
                return complex_number_error(ans, self, obj)
            else:
                # It is possible to get other non-numeric types.  Most
                # common are bool and 1-element numpy.array().  We will
                # attempt to convert the value to a float before
                # proceeding.
                #
                # TODO: we should check bool and warn/error (while bool is
                # convertible to float in Python, they have very
                # different semantic meanings in Pyomo).
                try:
                    ans = float(ans)
                except:
                    return InvalidNumber(
                        ans, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
                    )
        if ans != ans:
            return InvalidNumber(
                nan, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
            )
        return ans

    def cache_fixed_var(self, _id, child):
        val = self.check_constant(child.value, child)
        lb, ub = child.bounds
        if (lb is not None and lb - val > TOL) or (ub is not None and ub - val < -TOL):
            raise InfeasibleConstraintException(
                "model contains a trivially infeasible "
                f"variable '{child.name}' (fixed value "
                f"{val} outside bounds [{lb}, {ub}])."
            )
        self.fixed_vars[_id] = self.check_constant(child.value, child)

    def initializeWalker(self, expr):
        expr, src, src_idx, self.expression_scaling_factor = expr
        self.active_expression_source = (src_idx, id(src))
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        return _before_child_handlers[child.__class__](self, child)

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if node.__class__ in sum_like_expression_types:
            data = AMPLRepn(0, {}, None)
            data.nonlinear = []
            return node.args, data
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
        return _operator_handles[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        ans = node_result_to_amplrepn(result)

        # Multiply the expression by the scaling factor provided by the caller
        ans.mult *= self.expression_scaling_factor

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
        if ans.nl is not None:
            if not ans.nl[1]:
                raise ValueError("Numeric expression resolved to a string constant")
            # This *is* a named subexpression.  If there is no linear
            # component, then replace this expression with the named
            # expression.  The mult will be handled later.  We know that
            # the const is built into the nonlinear expression, because
            # it cannot be changed "in place" (only through addition,
            # which would have "cleared" the nl attribute)
            if not ans.linear:
                ans.named_exprs.update(ans.nl[1])
                ans.nonlinear = ans.nl
                ans.const = 0
            else:
                # This named expression has both a linear and a
                # nonlinear component, and possibly a multiplier and
                # constant.  We will not include this named expression
                # and instead will expose the components so that linear
                # variables are not accidentally re-characterized as
                # nonlinear.
                pass
            ans.nl = None

        if ans.nonlinear.__class__ is list:
            ans.compile_nonlinear_fragment(self)

        if not ans.linear:
            ans.linear = {}
        if ans.mult != 1:
            linear = ans.linear
            mult, ans.mult = ans.mult, 1
            ans.const *= mult
            if linear:
                for k in linear:
                    linear[k] *= mult
            if ans.nonlinear:
                if mult == -1:
                    prefix = self.template.negation
                else:
                    prefix = self.template.multiplier % mult
                ans.nonlinear = prefix + ans.nonlinear[0], ans.nonlinear[1]
        #
        self.active_expression_source = None
        return ans
