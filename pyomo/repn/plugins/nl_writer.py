#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import os
from collections import defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue, InEnum, document_class_CONFIG
from pyomo.common.deprecation import relocated_module_attribute
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer

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
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.expression import ScalarExpression, ExpressionData
from pyomo.core.base.objective import ScalarObjective, ObjectiveData
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory

from pyomo.repn.ampl import AMPLRepnVisitor, evaluate_ampl_nl_expression, TOL
from pyomo.repn.util import (
    FileDeterminism,
    FileDeterminism_to_SortComponents,
    categorize_valid_components,
    initialize_var_map_from_column_order,
    int_float,
    ordered_active_constraints,
)
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port

###

logger = logging.getLogger(__name__)

relocated_module_attribute('AMPLRepn', 'pyomo.repn.ampl.AMPLRepn', version='6.8.0')

inf = float('inf')
minus_inf = -inf
allowable_binary_var_bounds = {(0, 0), (0, 1), (1, 1)}

ScalingFactors = namedtuple(
    'ScalingFactors', ['variables', 'constraints', 'objectives']
)


# TODO: make a proper base class
class NLWriterInfo(object):
    """Return type for NLWriter.write()

    Attributes
    ----------
    variables: List[VarData]

        The list of (unfixed) Pyomo model variables in the order written
        to the NL file

    constraints: List[ConstraintData]

        The list of (active) Pyomo model constraints in the order written
        to the NL file

    objectives: List[ObjectiveData]

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

    eliminated_vars: List[Tuple[VarData, NumericExpression]]

        The list of variables in the model that were eliminated by the
        presolve.  Each entry is a 2-tuple of (:py:class:`VarData`,
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
@document_class_CONFIG(methods=['write'])
class NLWriter(object):
    #: Global class configuration;
    #: see :ref:`pyomo.repn.plugins.nl_writer.NLWriter::CONFIG`.
    CONFIG = ConfigDict('nlwriter')
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
            default=True,
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

           - NONE (0) : None
           - ORDERED (10): rely on underlying component ordering (default)
           - SORT_INDICES (20) : sort keys of indexed components
           - SORT_SYMBOLS (30) : sort keys AND sort names (not declaration order)

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
        #: Instance configuration;
        #: see :ref:`pyomo.repn.plugins.nl_writer.NLWriter::CONFIG`.
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

        # just for backwards compatibility
        config.skip_trivial_constraints = False

        if config.symbolic_solver_labels:
            _open = lambda fname: open(fname, 'w')
        else:
            _open = nullcontext
        with (
            open(filename, 'w', newline='') as FILE,
            _open(row_fname) as ROWFILE,
            _open(col_fname) as COLFILE,
        ):
            info = self.write(model, FILE, ROWFILE, COLFILE, config=config)
        if not info.variables:
            # This exception is included for compatibility with the
            # original NL writer v1.
            os.remove(filename)
            if config.symbolic_solver_labels:
                os.remove(row_fname)
                os.remove(col_fname)
            raise ValueError(
                "No variables appear in the Pyomo model constraints or"
                " objective. This is not supported by the NL file interface"
            )

        # Historically, the NL writer communicated the external function
        # libraries back to the ASL interface through the PYOMO_AMPLFUNC
        # environment variable.
        set_pyomo_amplfunc_env(info.external_function_libraries)
        # Generate the symbol map expected by the old readers
        symbol_map = self._generate_symbol_map(info)
        # The ProblemWriter callable interface returns the filename that
        # was generated and the symbol_map
        return filename, symbol_map

    def write(
        self, model, ostream, rowstream=None, colstream=None, **options
    ) -> NLWriterInfo:
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
        var_con_obj = {Var, Constraint, Objective}
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
                elif getattr(obj, 'ctype', None) in var_con_obj:
                    if obj.is_indexed():
                        # Expand this indexed component to store the
                        # individual ComponentDatas, but ONLY if the
                        # component data is not in the original dictionary
                        # of values that we extracted from the Suffixes
                        queue.append(
                            product(
                                filterfalse(self.values.__contains__, obj.values()),
                                (val,),
                            )
                        )
                    else:
                        missing_component_data.add(obj)
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

    def __init__(self, name, default=None, context=None):
        super().__init__(name, default, context)
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
        self.subexpression_cache = {}
        self.subexpression_order = None  # set to [] later
        self.external_functions = {}
        self.used_named_expressions = set()
        self.var_map = {}
        self.var_id_to_nl_map = {}
        self.sorter = FileDeterminism_to_SortComponents(config.file_determinism)
        self.visitor = AMPLRepnVisitor(
            self.subexpression_cache,
            self.external_functions,
            self.var_map,
            self.used_named_expressions,
            self.symbolic_solver_labels,
            self.config.export_defined_variables,
            self.sorter,
        )
        self.next_V_line_id = 0
        self.pause_gc = None
        self.template = self.visitor.Result.template

    def __enter__(self):
        self.pause_gc = PauseGC()
        self.pause_gc.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.pause_gc.__exit__(exc_type, exc_value, tb)

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

        nl_map = self.var_id_to_nl_map
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
            scaling_factor = CachingNumericSuffixFinder('scaling_factor', 1, model)
            scaling_cache = scaling_factor.suffix_cache
            del suffix_data['scaling_factor']
        else:
            scaling_factor = _NoScalingFactor()
            scaling_cache = None
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

        all_constraints = []
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
            # Note: Constraint.to_bounded_expression(evaluate_bounds=True)
            # guarantee a return value that is either a (finite)
            # native_numeric_type, or None
            lb, body, ub = con.to_bounded_expression(True)
            expr_info = visitor.walk_expression((body, con, 0, scale))
            if expr_info.named_exprs:
                self._record_named_expression_usage(expr_info.named_exprs, con, 0)

            if lb is None and ub is None:  # and self.config.skip_trivial_constraints:
                continue
            if scale != 1:
                if lb is not None:
                    lb = lb * scale
                if ub is not None:
                    ub = ub * scale
                if scale < 0:
                    lb, ub = ub, lb
            all_constraints.append((con, expr_info, lb, ub))
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
            timer.toc('Processed %s constraints', len(all_constraints))

        # We have identified all the external functions (resolving them
        # by name).  Now we may need to resolve the function by the
        # (local) FID, which we know is indexed by integers starting at
        # 0.  We will convert the dict to a list for efficient lookup.
        self.external_functions = list(self.external_functions.values())

        # This may fetch more bounds than needed, but only in the cases
        # where variables were completely eliminated while walking the
        # expressions, or when users provide superfluous variables in
        # the column ordering.
        var_bounds = {_id: v.bounds for _id, v in var_map.items()}
        var_values = {_id: v.value for _id, v in var_map.items()}

        eliminated_cons, eliminated_vars = self._linear_presolve(
            comp_by_linear_var, lcon_by_linear_nnz, var_bounds, var_values
        )
        del comp_by_linear_var
        del lcon_by_linear_nnz

        # Note: defer categorizing constraints until after presolve, as
        # the presolver could result in nonlinear constraints becoming
        # linear (or trivial)
        constraints = []
        linear_cons = []
        if eliminated_cons:
            _removed = eliminated_cons.__contains__
            _constraints = filterfalse(lambda c: _removed(id(c[0])), all_constraints)
        else:
            _constraints = all_constraints
        for info in _constraints:
            expr_info = info[1]
            if expr_info.nonlinear:
                nl, args = expr_info.nonlinear
                if any(vid not in nl_map for vid in args):
                    constraints.append(info)
                    continue
                expr_info.const += evaluate_ampl_nl_expression(
                    nl % tuple(nl_map[i] for i in args), self.external_functions
                )
                expr_info.nonlinear = None
            if expr_info.linear:
                linear_cons.append(info)
            elif not self.config.skip_trivial_constraints:
                linear_cons.append(info)
            else:  # constant constraint and skip_trivial_constraints
                c = expr_info.const
                con, expr_info, lb, ub = info
                if (lb is not None and lb - c > TOL) or (
                    ub is not None and ub - c < -TOL
                ):
                    raise InfeasibleConstraintException(
                        "model contains a trivially infeasible "
                        f"constraint '{con.name}' (fixed body value "
                        f"{c} outside bounds [{lb}, {ub}])."
                    )

        # Order the constraints, moving all nonlinear constraints to
        # the beginning
        n_nonlinear_cons = len(constraints)
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
            filter(self.used_named_expressions.__contains__, self.subexpression_cache)
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
        self._categorize_vars(self.subexpression_cache.values(), linear_by_comp)
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
                        var_values[_id] = _v.value
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
                # Note: integer variables whose bounds are in {0, 1}
                # should be classified as binary
                if var_bounds[_id] in allowable_binary_var_bounds:
                    binary_vars.add(_id)
                else:
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
            id2nl = {
                _id: f'v{var_idx}{col_comments[var_idx]}\n'
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
            id2nl = {_id: f"v{var_idx}\n" for var_idx, _id in enumerate(variables)}

        if nl_map:
            nl_map.update(id2nl)
        else:
            self.var_id_to_nl_map = nl_map = id2nl
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
                # Update nl_map to output scaled variables in NL expressions
                nl_map[_id] = template.division + nl_map[_id] + template.const % scale

        # Update any eliminated variables to point to the (potentially
        # scaled) substituted variables
        for _id, expr_info in list(eliminated_vars.items()):
            nl, args, _ = expr_info.compile_repn()
            for _i in args:
                # It is possible that the eliminated variable could
                # reference another variable that is no longer part of
                # the model and therefore does not have a nl_map entry.
                # This can happen when there is an underdetermined
                # independent linear subsystem and the presolve removed
                # all the constraints from the subsystem.  Because the
                # free variables in the subsystem are not referenced
                # anywhere else in the model, they are not part of the
                # `variables` list.  Implicitly "fix" it to an arbitrary
                # valid value from the presolved domain (see #3192).
                if _i not in nl_map:
                    lb, ub = var_bounds[_i]
                    if lb is None:
                        lb = -inf
                    if ub is None:
                        ub = inf
                    if lb <= 0 <= ub:
                        val = 0
                    else:
                        val = lb if abs(lb) < abs(ub) else ub
                    eliminated_vars[_i] = visitor.Result(val, {}, None)
                    nl_map[_i] = expr_info.compile_repn()[0]
                    logger.warning(
                        "presolve identified an underdetermined independent "
                        "linear subsystem that was removed from the model.  "
                        f"Setting '{var_map[_i]}' == {val}"
                    )
            nl_map[_id] = nl % tuple(nl_map[_i] for _i in args)

        r_lines = [None] * n_cons
        for idx, (con, expr_info, lb, ub) in enumerate(constraints):
            if lb == ub:  # TBD: should this be within tolerance?
                if lb is None:
                    # type = 3  # -inf <= c <= inf
                    r_lines[idx] = "3"
                else:
                    # _type = 4  # L == c == U
                    r_lines[idx] = f"4 {lb - expr_info.const!s}"
                    n_equality += 1
            elif lb is None:
                # _type = 1  # c <= U
                r_lines[idx] = f"1 {ub - expr_info.const!s}"
            elif ub is None:
                # _type = 2  # L <= c
                r_lines[idx] = f"2 {lb - expr_info.const!s}"
            else:
                # _type = 0  # L <= c <= U
                r_lines[idx] = f"0 {lb - expr_info.const!s} {ub - expr_info.const!s}"
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
                len(con_only_nonlinear_vars.intersection(discrete_vars)),
                len(obj_only_nonlinear_vars.intersection(discrete_vars)),
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
        for fid, fcn in self.external_functions:
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
                    ''.join(f"{_id} {_vals[_id]!s}\n" for _id in sorted(_vals))
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
                self._write_v_line(_id, 0, scale_model, scaling_cache)
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
                    self._write_v_line(_id, row_idx + 1, scale_model, scaling_cache)
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
                    self._write_v_line(
                        _id, n_cons + n_lcons + obj_idx + 1, scale_model, scaling_cache
                    )
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
                    ''.join(f"{_id} {data.con[_id]!s}\n" for _id in sorted(data.con))
                )

        #
        # "x" lines (variable initialization)
        #
        _init_lines = [
            (var_idx, val if val.__class__ in int_float else float(val))
            for var_idx, val in enumerate(map(var_values.__getitem__, variables))
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
                f'{var_idx} {val!s}{col_comments[var_idx]}\n'
                for var_idx, val in _init_lines
            )
        )

        #
        # "r" lines (constraint bounds)
        #
        ostream.write(
            'r%s\n'
            % (
                (
                    "\t#%d ranges (rhs's)" % len(constraints)
                    if symbolic_solver_labels
                    else ''
                ),
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
                (
                    "\t#%d bounds (on variables)" % len(variables)
                    if symbolic_solver_labels
                    else ''
                ),
            )
        )
        for var_idx, _id in enumerate(variables):
            lb, ub = var_bounds[_id]
            if lb == ub:
                if lb is None:  # unbounded
                    ostream.write(f"3{col_comments[var_idx]}\n")
                else:  # ==
                    ostream.write(f"4 {lb!s}{col_comments[var_idx]}\n")
            elif lb is None:  # var <= ub
                ostream.write(f"1 {ub!s}{col_comments[var_idx]}\n")
            elif ub is None:  # lb <= body
                ostream.write(f"2 {lb!s}{col_comments[var_idx]}\n")
            else:  # lb <= body <= ub
                ostream.write(f"0 {lb!s} {ub!s}{col_comments[var_idx]}\n")

        #
        # "k" lines (column offsets in Jacobian NNZ)
        #
        ostream.write(
            'k%d%s\n'
            % (
                len(variables) - 1,
                (
                    "\t#intermediate Jacobian column lengths"
                    if symbolic_solver_labels
                    else ''
                ),
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
                ostream.write(f'{column_order[_id]} {linear[_id]!s}\n')

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
                ostream.write(f'{column_order[_id]} {linear[_id]!s}\n')

        # Generate the return information
        eliminated_vars = [
            (var_map[_id], expr_info.to_expr(var_map))
            for _id, expr_info in eliminated_vars.items()
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

            if expr_info.linear:
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

    def _linear_presolve(
        self, comp_by_linear_var, lcon_by_linear_nnz, var_bounds, var_values
    ):
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
        nl_map = self.var_id_to_nl_map
        one_var = lcon_by_linear_nnz[1]
        two_var = lcon_by_linear_nnz[2]
        while 1:
            if fixed_vars:
                _id = fixed_vars.pop()
                a = x = None
                b, _ = var_bounds[_id]
                logger.debug("NL presolve: bounds fixed %s := %s", var_map[_id], b)
                eliminated_vars[_id] = self.visitor.Result(b, {}, None)
                nl_map[_id] = template.const % b
            elif one_var:
                con_id, info = one_var.popitem()
                expr_info, lb = info
                _id, coef = expr_info.linear.popitem()
                # substituting _id with a*x + b
                a = x = None
                b = expr_info.const = (lb - expr_info.const) / coef
                logger.debug("NL presolve: substituting %s := %s", var_map[_id], b)
                eliminated_vars[_id] = expr_info
                nl_map[_id] = template.const % b
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
                    # substitute out the other
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
                # eliminating _id and replacing it with a*x + b
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
                # Given that we are eliminating a variable, we want to
                # attempt to sanely resolve the initial variable values.
                y_init = var_values[_id]
                if y_init is not None:
                    # Y has a value
                    x_init = var_values[x]
                    if x_init is None:
                        # X does not; just use the one calculated from Y
                        x_init = (y_init - b) / a
                    else:
                        # X does too, use the average of the two values
                        x_init = (x_init + (y_init - b) / a) / 2.0
                    # Ensure that the initial value respects the
                    # tightened bounds
                    if x_ub is not None and x_init > x_ub:
                        x_init = x_ub
                    if x_lb is not None and x_init < x_lb:
                        x_init = x_lb
                    var_values[x] = x_init
                eliminated_cons.add(con_id)
            else:
                break
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
                old_nnz = len(expr_info.linear)
                c = expr_info.linear.pop(_id, 0)
                nnz = old_nnz - 1
                expr_info.const += c * b
                if x in expr_info.linear:
                    expr_info.linear[x] += c * a
                    if expr_info.linear[x] == 0:
                        nnz -= 1
                        coef = expr_info.linear.pop(x)
                elif a:
                    expr_info.linear[x] = c * a
                    # replacing _id with x... NNZ is not changing,
                    # but we need to remember that x is now part of
                    # this constraint
                    comp_by_linear_var[x].append((con_id, expr_info))
                    continue
                _old = lcon_by_linear_nnz[old_nnz]
                if con_id in _old:
                    if not nnz:
                        if abs(expr_info.const) > TOL:
                            # constraint is trivially infeasible
                            raise InfeasibleConstraintException(
                                "model contains a trivially infeasible constraint "
                                f"{expr_info.const} == {coef}*{var_map[x]}"
                            )
                        # constraint is trivially feasible
                        eliminated_cons.add(con_id)
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
                elif not expr_info.linear:
                    nl_map[resubst] = template.const % expr_info.const

        # Note: the ASL will (silently) produce incorrect answers if the
        # nonlinear portion of a defined variable is a constant
        # expression.  This may now be the case if all the variables in
        # the original nonlinear expression have been fixed.
        for _id, (expr, info, sub) in self.subexpression_cache.items():
            if info.nonlinear:
                nl, args = info.nonlinear
                # Note: 'not args' skips string arguments
                # Note: 'vid in nl_map' skips eliminated
                #   variables and defined variables reduced to constants
                if not args or any(vid not in nl_map for vid in args):
                    continue
                # Ideally, we would just evaluate the named expression.
                # However, there might be a linear portion of the named
                # expression that still has free variables, and there is no
                # guarantee that the user actually initialized the
                # variables.  So, we will fall back on parsing the (now
                # constant) nonlinear fragment and evaluating it.
                info.nonlinear = None
                info.const += evaluate_ampl_nl_expression(
                    nl % tuple(nl_map[i] for i in args), self.external_functions
                )
            if not info.linear:
                # This has resolved to a constant: the ASL will fail for
                # defined variables containing ONLY a constant.  We
                # need to substitute the constant directly into the
                # original constraint/objective expression(s)
                info.linear = {}
                self.used_named_expressions.discard(_id)
                nl_map[_id] = template.const % info.const
                self.subexpression_cache[_id] = (expr, info, [None, None, True])

        return eliminated_cons, eliminated_vars

    def _record_named_expression_usage(self, named_exprs, src, comp_type):
        self.used_named_expressions.update(named_exprs)
        src = id(src)
        for _id in named_exprs:
            info = self.subexpression_cache[_id][2]
            if info[comp_type] is None:
                info[comp_type] = src
            elif info[comp_type] != src:
                info[comp_type] = 0

    def _resolve_subexpression_args(self, nl, args):
        final_args = []
        for arg in args:
            if arg in self.var_id_to_nl_map:
                final_args.append(self.var_id_to_nl_map[arg])
            else:
                _nl, _ids, _ = self.subexpression_cache[arg][1].compile_repn()
                final_args.append(self._resolve_subexpression_args(_nl, _ids))
        return nl % tuple(final_args)

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
            try:
                self.ostream.write(
                    nl % tuple(map(self.var_id_to_nl_map.__getitem__, args))
                )
            except KeyError:
                self.ostream.write(self._resolve_subexpression_args(nl, args))

        elif include_const:
            self.ostream.write(self.template.const % repn.const)
        else:
            self.ostream.write(self.template.const % 0)

    def _write_v_line(self, expr_id, k, scale_model, scaling_cache):
        ostream = self.ostream
        column_order = self.column_order
        info = self.subexpression_cache[expr_id]
        if self.symbolic_solver_labels:
            lbl = '\t#%s' % info[0].name
        else:
            lbl = ''
        self.var_id_to_nl_map[expr_id] = f"v{self.next_V_line_id}{lbl}\n"
        # Do NOT write out 0 coefficients here: doing so fouls up the
        # ASL's logic for calculating derivatives, leading to 'nan' in
        # the Hessian results.
        linear = info[1].linear
        linear_ids = list(_id for _id, coef in linear.items() if coef)
        if scale_model:
            for _id in linear_ids:
                linear[_id] /= scaling_cache[_id]
        #
        ostream.write(f'V{self.next_V_line_id} {len(linear_ids)} {k}{lbl}\n')
        for _id in sorted(linear_ids, key=column_order.__getitem__):
            ostream.write(f'{column_order[_id]} {linear[_id]!s}\n')
        self._write_nl_expression(info[1], True)
        self.next_V_line_id += 1
