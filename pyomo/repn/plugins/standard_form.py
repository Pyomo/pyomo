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

import collections
import itertools
import operator
import logging

from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    InEnum,
    document_kwargs_from_configdict,
)
from pyomo.common.dependencies import scipy, numpy as np
from pyomo.common.enums import ObjectiveSense
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import native_types, value
from pyomo.common.timing import TicTocTimer

from pyomo.core.base import (
    Block,
    Objective,
    Constraint,
    Var,
    Param,
    Expression,
    SortComponents,
    Suffix,
    SymbolMap,
)
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.linear_template import LinearTemplateRepnVisitor
from pyomo.repn.util import (
    FileDeterminism,
    FileDeterminism_to_SortComponents,
    TemplateVarRecorder,
    categorize_valid_components,
    initialize_var_map_from_column_order,
    ordered_active_constraints,
)

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port

logger = logging.getLogger(__name__)

RowEntry = collections.namedtuple('RowEntry', ['constraint', 'bound_type'])


# TODO: make a proper base class
class LinearStandardFormInfo(object):
    """Return type for LinearStandardFormCompiler.write()

    Attributes
    ----------
    c : scipy.sparse.csc_array

        The objective coefficients.  Note that this is a sparse array
        and may contain multiple rows (for multiobjective problems).  The
        objectives may be calculated by ``c @ x``

    c_offset : numpy.ndarray

        The list of objective constant offsets

    A : scipy.sparse.csc_array

        The constraint coefficients.  The constraint bodies may be
        calculated by ``A @ x``

    rhs : numpy.ndarray

        The constraint right-hand sides.

    rows : List[Tuple[ConstraintData, int]]

        The list of Pyomo constraint objects corresponding to the rows
        in `A`.  Each element in the list is a 2-tuple of
        (ConstraintData, row_multiplier).  The `row_multiplier` will be
        +/- 1 indicating if the row was multiplied by -1 (corresponding
        to a constraint lower bound) or +1 (upper bound).

    columns : List[VarData]

        The list of Pyomo variable objects corresponding to columns in
        the `A` and `c` matrices.

    objectives : List[ObjectiveData]

        The list of Pyomo objective objects corresponding to the active objectives

    eliminated_vars: List[Tuple[VarData, NumericExpression]]

        The list of variables from the original model that do not appear
        in the standard form (usually because they were replaced by
        nonnegative variables).  Each entry is a 2-tuple of
        (:py:class:`VarData`, :py:class`NumericExpression`|`float`).
        The list is in the necessary order for correct evaluation (i.e.,
        all variables appearing in the expression must either have
        appeared in the standard form, or appear *earlier* in this list.

    """

    def __init__(self, c, c_offset, A, rhs, rows, columns, objectives, eliminated_vars):
        self.c = c
        self.c_offset = c_offset
        self.A = A
        self.rhs = rhs
        self.rows = rows
        self.columns = columns
        self.objectives = objectives
        self.eliminated_vars = eliminated_vars

    @property
    def x(self):
        "Alias for :attr:`columns`"
        return self.columns

    @property
    def b(self):
        "Alias for :attr:`rhs`"
        return self.rhs


@WriterFactory.register(
    'compile_standard_form',
    r'Compile an LP to standard form (:math:`\min c^Tx s.t. Ax \le b)`',
)
class LinearStandardFormCompiler(object):
    r"""Compiler to convert an LP to the matrix representation of the
    standard form:

    .. math::

        \min\ & c^Tx \\
        s.t.\ & Ax \le b

    and return the compiled representation as NumPy arrays and SciPy
    sparse matrices.

    """

    CONFIG = ConfigBlock('compile_standard_form')

    CONFIG.declare(
        'nonnegative_vars',
        ConfigValue(
            default=False,
            domain=bool,
            description='Convert all variables to be nonnegative variables',
        ),
    )
    CONFIG.declare(
        'slack_form',
        ConfigValue(
            default=False,
            domain=bool,
            description='Add slack variables and return '
            r':math:`\min c^Tx; s.t. Ax = b`',
        ),
    )
    CONFIG.declare(
        'mixed_form',
        ConfigValue(
            default=False,
            domain=bool,
            description='Return A in mixed form (the comparison operator is a '
            'mix of <=, ==, and >=)',
        ),
    )
    CONFIG.declare(
        'set_sense',
        ConfigValue(
            default=ObjectiveSense.minimize,
            domain=InEnum(ObjectiveSense),
            description='If not None, map all objectives to the specified sense.',
        ),
    )
    CONFIG.declare(
        'show_section_timing',
        ConfigValue(
            default=False,
            domain=bool,
            description='Print timing after each stage of the compilation process',
        ),
    )
    CONFIG.declare(
        'file_determinism',
        ConfigValue(
            default=FileDeterminism.ORDERED,
            domain=InEnum(FileDeterminism),
            description='How much effort to ensure result is deterministic',
            doc="""
            How much effort do we want to put into ensuring the
            resulting matrices are produced deterministically:

               - ``NONE`` (0): None
               - ``ORDERED`` (10): rely on underlying component ordering (default)
               - ``SORT_INDICES`` (20) : sort keys of indexed components
               - ``SORT_SYMBOLS`` (30) : sort keys AND sort names (not
                 declaration order)

            """,
        ),
    )
    CONFIG.declare(
        'row_order',
        ConfigValue(
            default=None,
            description='Preferred constraint ordering',
            doc="""
            List of constraints in the order that they should appear in
            the resulting ``A`` matrix.  Unspecified constraints will
            appear at the end.""",
        ),
    )
    CONFIG.declare(
        'column_order',
        ConfigValue(
            default=None,
            description='Preferred variable ordering',
            doc="""
            List of variables in the order that they should appear in
            the compiled representation.  Unspecified variables will be
            appended to the end of this list.""",
        ),
    )

    def __init__(self):
        self.config = self.CONFIG()

    @document_kwargs_from_configdict(CONFIG)
    def write(self, model, ostream=None, **options):
        """Convert a model to standard form

        Returns
        -------
        LinearStandardFormInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream:
            This is provided for API compatibility with other writers
            and is ignored here.

        """
        config = self.config(options)

        # Pause the GC, as the walker that generates the compiled LP
        # representation generates (and disposes of) a large number of
        # small objects.
        with PauseGC():
            return _LinearStandardFormCompiler_impl(config).write(model)


class _LinearStandardFormCompiler_impl(object):
    # Making these methods class attributes so that others can change the hooks
    _get_visitor = LinearRepnVisitor
    _to_vector = None
    _csc_matrix = None
    _csr_matrix = None

    def __init__(self, config):
        self.config = config
        # We defer the first instantiation of these attributes so we do
        # not trigger the numpy / scipy imports when the module is
        # imported
        if _LinearStandardFormCompiler_impl._to_vector is None:
            _LinearStandardFormCompiler_impl._to_vector = np.fromiter
            _LinearStandardFormCompiler_impl._csc_matrix = scipy.sparse.csc_array
            _LinearStandardFormCompiler_impl._csr_matrix = scipy.sparse.csr_array

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
            targets={Suffix, Objective},
        )
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the Linear Standard Form compiler does not know how to "
                "process:\n\t%s"
                % (
                    model.name,
                    "\n\t".join(
                        "%s:\n\t\t%s"
                        % (k, "\n\t\t".join(map(operator.attrgetter('name'), v)))
                        for k, v in unknown.items()
                    ),
                )
            )

        self.var_map = var_map = {}
        initialize_var_map_from_column_order(model, self.config, var_map)

        var_recorder = TemplateVarRecorder(var_map, sorter)
        visitor = self._get_visitor({}, var_recorder=var_recorder)
        template_visitor = LinearTemplateRepnVisitor({}, var_recorder=var_recorder)

        timer.toc('Initialized column order', level=logging.DEBUG)

        # We don't export any suffix information to the Standard Form
        #
        if component_map[Suffix]:
            suffixesByName = {}
            for block in component_map[Suffix]:
                for suffix in block.component_objects(
                    Suffix, active=True, descend_into=False, sort=sorter
                ):
                    if not suffix.export_enabled() or not suffix:
                        continue
                    name = suffix.local_name
                    if name in suffixesByName:
                        suffixesByName[name].append(suffix)
                    else:
                        suffixesByName[name] = [suffix]
            for name, suffixes in suffixesByName.items():
                n = len(suffixes)
                plural = 's' if n > 1 else ''
                logger.warning(
                    f"EXPORT Suffix '{name}' found on {n} block{plural}:\n    "
                    + "\n    ".join(s.name for s in suffixes)
                    + "\nStandard Form compiler ignores export suffixes.  Skipping."
                )

        #
        # Process objective
        #
        set_sense = self.config.set_sense
        objectives = []
        for blk in component_map[Objective]:
            objectives.extend(
                blk.component_data_objects(
                    Objective, active=True, descend_into=False, sort=sorter
                )
            )
        obj_nnz = 0
        obj_offset = []
        obj_data = []
        obj_index = []
        obj_index_ptr = [0]
        for obj in objectives:
            if hasattr(obj, 'template_expr'):
                offset, linear_index, linear_data, _, _ = (
                    template_visitor.expand_expression(obj, obj.template_expr())
                )
                N = len(linear_index)
                obj_index.append(linear_index)
                obj_data.append(linear_data)
                obj_offset.append(offset)
            else:
                repn = visitor.walk_expression(obj.expr)
                N = len(repn.linear)
                obj_index.append(map(var_recorder.var_order.__getitem__, repn.linear))
                obj_data.append(repn.linear.values())
                obj_offset.append(repn.constant)

                if repn.nonlinear is not None:
                    raise ValueError(
                        f"Model objective ({obj.name}) contains nonlinear terms that "
                        "cannot be compiled to standard (linear) form."
                    )

            obj_nnz += N
            if set_sense is not None and set_sense != obj.sense:
                obj_data[-1] = -self._to_vector(obj_data[-1], float, N)
                obj_offset[-1] *= -1
            obj_index_ptr.append(obj_index_ptr[-1] + N)
            if with_debug_timing:
                timer.toc('Objective %s', obj, level=logging.DEBUG)

        #
        # Tabulate constraints
        #
        slack_form = self.config.slack_form
        mixed_form = self.config.mixed_form
        if slack_form and mixed_form:
            raise ValueError("cannot specify both slack_form and mixed_form")
        rows = []
        rhs = []
        con_nnz = 0
        con_data = []
        con_index = []
        con_index_ptr = [0]
        last_parent = None
        for con in ordered_active_constraints(model, self.config):
            if with_debug_timing and con._component is not last_parent:
                if last_parent is not None:
                    timer.toc('Constraint %s', last_parent(), level=logging.DEBUG)
                last_parent = con._component

            if hasattr(con, 'template_expr'):
                offset, linear_index, linear_data, lb, ub = (
                    template_visitor.expand_expression(con, con.template_expr())
                )
                N = len(linear_data)
            else:
                # Note: lb and ub could be a number, expression, or None
                lb, body, ub = con.to_bounded_expression()
                if lb.__class__ not in native_types:
                    lb = value(lb)
                if ub.__class__ not in native_types:
                    ub = value(ub)
                repn = visitor.walk_expression(body)
                if repn.nonlinear is not None:
                    raise ValueError(
                        f"Model constraint ({con.name}) contains nonlinear terms that "
                        "cannot be compiled to standard (linear) form."
                    )

                N = len(repn.linear)
                # Pull out the constant: we will move it to the bounds
                offset = repn.constant
                linear_index = map(var_recorder.var_order.__getitem__, repn.linear)
                linear_data = repn.linear.values()

            if lb is None and ub is None:
                # Note: you *cannot* output trivial (unbounded)
                # constraints in matrix format.  I suppose we could add a
                # slack variable, but that seems rather silly.
                continue

            if not N and offset.__class__ in native_types:
                # This is a constant constraint
                # TODO: add a (configurable) feasibility tolerance
                if (lb is None or lb <= offset) and (ub is None or ub >= offset):
                    continue
                raise InfeasibleError(
                    f"model contains a trivially infeasible constraint, '{con.name}'"
                )

            if mixed_form:
                if lb == ub:
                    con_nnz += N
                    rows.append(RowEntry(con, 0))
                    rhs.append(ub - offset)
                    con_data.append(linear_data)
                    con_index.append(linear_index)
                    con_index_ptr.append(con_nnz)
                else:
                    if ub is not None:
                        if lb is not None:
                            linear_index = list(linear_index)
                        con_nnz += N
                        rows.append(RowEntry(con, 1))
                        rhs.append(ub - offset)
                        con_data.append(linear_data)
                        con_index.append(linear_index)
                        con_index_ptr.append(con_nnz)
                    if lb is not None:
                        con_nnz += N
                        rows.append(RowEntry(con, -1))
                        rhs.append(lb - offset)
                        con_data.append(linear_data)
                        con_index.append(linear_index)
                        con_index_ptr.append(con_nnz)
            elif slack_form:
                if lb == ub:  # TODO: add tolerance?
                    rhs.append(ub - offset)
                else:
                    # add slack variable
                    con_nnz += 1
                    v = Var(name=f'_slack_{len(rhs)}', bounds=(None, None))
                    v.construct()
                    if lb is None:
                        rhs.append(ub - offset)
                        v.lb = 0
                    else:
                        rhs.append(lb - offset)
                        v.ub = 0
                        if ub is not None:
                            v.lb = lb - ub
                    var_map[id(v)] = v
                    if var_recorder.var_order is not None:
                        var_recorder.var_order[id(v)] = slack_col = len(
                            var_recorder.var_order
                        )
                    linear_data = list(linear_data)
                    linear_data.append(1)
                    linear_index = list(linear_index)
                    linear_index.append(slack_col)
                con_nnz += N
                rows.append(RowEntry(con, 1))
                con_data.append(linear_data)
                con_index.append(linear_index)
                con_index_ptr.append(con_nnz)
            else:
                if ub is not None:
                    if lb is not None:
                        linear_index = list(linear_index)
                    con_nnz += N
                    rows.append(RowEntry(con, 1))
                    rhs.append(ub - offset)
                    con_data.append(linear_data)
                    con_index.append(linear_index)
                    con_index_ptr.append(con_nnz)
                if lb is not None:
                    con_nnz += N
                    rows.append(RowEntry(con, -1))
                    rhs.append(offset - lb)
                    con_data.append(-np.array(list(linear_data)))
                    con_index.append(linear_index)
                    con_index_ptr.append(con_nnz)

        if with_debug_timing:
            # report the last constraint
            timer.toc('Constraint %s', last_parent(), level=logging.DEBUG)

        # Get the variable list
        columns = list(var_map.values())
        n_cols = len(columns)

        # Convert the compiled data to scipy sparse matrices
        c = self._create_csc(obj_data, obj_index, obj_index_ptr, obj_nnz, n_cols)
        A = self._create_csc(con_data, con_index, con_index_ptr, con_nnz, n_cols)

        if with_debug_timing:
            timer.toc('Formed matrices', level=logging.DEBUG)

        # Some variables in the var_map may not actually appear in the
        # objective or constraints (e.g., added from col_order, or
        # multiplied by 0 in the expressions).  The easiest way to check
        # for empty columns is to convert from CSR to CSC and then look
        # at the index pointer list (an O(num_var) operation).
        c_ip = c.indptr
        A_ip = A.indptr
        active_var_mask = (A_ip[1:] > A_ip[:-1]) | (c_ip[1:] > c_ip[:-1])

        obj_offset = np.array(obj_offset)

        # Note: since we are doing matrix multiplication and then
        # removing the fixed columns, we only want to go through that
        # hassle for columns with actual nonzero entries.
        fixed_vars = (
            np.array([v.fixed for v in columns], dtype=np.bool_) & active_var_mask
        )
        if any(fixed_vars):
            x = np.array([columns[i].value for i, f in enumerate(fixed_vars) if f])
            obj_offset = obj_offset + c[:, fixed_vars] @ x
            rhs = np.array(rhs) - A[:, fixed_vars] @ x
            # Update the active variable mask to exclude the fixed_vars
            # (that we just moved to the RHS)
            active_var_mask &= ~fixed_vars
            c = c[:, active_var_mask]
            A = A[:, active_var_mask]
            columns = [v for k, v in zip(active_var_mask, columns) if k]
            n_cols = len(columns)
        else:
            # Masks on NumPy arrays are very fast.  Build the reduced A
            # indptr and then check if we actually have to manipulate the
            # columns
            augmented_mask = np.concatenate((active_var_mask, [True]))
            reduced_A_indptr = A.indptr[augmented_mask]
            n_cols -= len(reduced_A_indptr) - 1
            if n_cols > 0:
                columns = [v for k, v in zip(active_var_mask, columns) if k]
                c = self._csc_matrix(
                    (c.data, c.indices, c.indptr[augmented_mask]),
                    [c.shape[0], len(columns)],
                )
                # active_var_idx[-1] = len(columns)
                A = self._csc_matrix(
                    (A.data, A.indices, reduced_A_indptr), [A.shape[0], len(columns)]
                )

        if with_debug_timing:
            timer.toc('Eliminated %s unused columns', n_cols, level=logging.DEBUG)

        if self.config.nonnegative_vars:
            c, A, columns, eliminated_vars = self._csc_to_nonnegative_vars(
                c, A, columns
            )
        else:
            eliminated_vars = []

        info = LinearStandardFormInfo(
            c, obj_offset, A, rhs, rows, columns, objectives, eliminated_vars
        )
        timer.toc("Generated linear standard form representation", delta=False)
        return info

    def _create_csc(self, data, index, index_ptr, nnz, n_cols):
        data = self._to_vector(itertools.chain.from_iterable(data), np.float64, nnz)
        index = self._to_vector(itertools.chain.from_iterable(index), np.int32, nnz)
        index_ptr = np.array(index_ptr, dtype=np.int32)

        if not nnz:
            # The empty CSC has no (or few) rows and a large number of
            # columns and no nonzeros: it is faster / easier to create
            # the empty CSR on the python side and convert it to CSC on
            # the C (numpy) side, as opposed to creating the large [0] *
            # (n_cols + 1) array on the Python side and transfer it to C
            # (numpy)
            return self._csr_matrix(
                (data, index, index_ptr), [len(index_ptr) - 1, n_cols]
            ).tocsc()

        A = self._csr_matrix((data, index, index_ptr), [len(index_ptr) - 1, n_cols])
        A = A.tocsc()
        A.sum_duplicates()
        A.eliminate_zeros()
        return A

    def _csc_to_nonnegative_vars(self, c, A, columns):
        eliminated_vars = []
        new_columns = []
        new_c_data = []
        new_c_indices = []
        new_c_indptr = [0]
        new_A_data = []
        new_A_indices = []
        new_A_indptr = [0]
        for i, v in enumerate(columns):
            lb, ub = v.bounds
            if lb is None or lb < 0:
                name = v.name
                new_columns.append(
                    Var(
                        name=f'_neg_{i}',
                        domain=v.domain,
                        bounds=(0, None if lb is None else -lb),
                    )
                )
                new_columns[-1].construct()
                s, e = A.indptr[i : i + 2]
                new_A_data.append(-A.data[s:e])
                new_A_indices.append(A.indices[s:e])
                new_A_indptr.append(new_A_indptr[-1] + e - s)
                s, e = c.indptr[i : i + 2]
                new_c_data.append(-c.data[s:e])
                new_c_indices.append(c.indices[s:e])
                new_c_indptr.append(new_c_indptr[-1] + e - s)
                if ub is None or ub > 0:
                    # Crosses 0; split into 2 vars
                    new_columns.append(
                        Var(name=f'_pos_{i}', domain=v.domain, bounds=(0, ub))
                    )
                    new_columns[-1].construct()
                    s, e = A.indptr[i : i + 2]
                    new_A_data.append(A.data[s:e])
                    new_A_indices.append(A.indices[s:e])
                    new_A_indptr.append(new_A_indptr[-1] + e - s)
                    s, e = c.indptr[i : i + 2]
                    new_c_data.append(c.data[s:e])
                    new_c_indices.append(c.indices[s:e])
                    new_c_indptr.append(new_c_indptr[-1] + e - s)
                    eliminated_vars.append((v, new_columns[-1] - new_columns[-2]))
                else:
                    new_columns[-1].lb = -ub
                    eliminated_vars.append((v, -new_columns[-1]))
            else:  # lb >= 0
                new_columns.append(v)
                s, e = A.indptr[i : i + 2]
                new_A_data.append(A.data[s:e])
                new_A_indices.append(A.indices[s:e])
                new_A_indptr.append(new_A_indptr[-1] + e - s)
                s, e = c.indptr[i : i + 2]
                new_c_data.append(c.data[s:e])
                new_c_indices.append(c.indices[s:e])
                new_c_indptr.append(new_c_indptr[-1] + e - s)

        n_cols = len(new_columns)
        c = self._csc_matrix(
            (np.concatenate(new_c_data), np.concatenate(new_c_indices), new_c_indptr),
            [c.shape[0], n_cols],
        )
        A = self._csc_matrix(
            (np.concatenate(new_A_data), np.concatenate(new_A_indices), new_A_indptr),
            [A.shape[0], n_cols],
        )
        return c, A, new_columns, eliminated_vars
