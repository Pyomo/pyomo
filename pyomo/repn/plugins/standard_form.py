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

import collections
import logging
from operator import attrgetter

from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    InEnum,
    document_kwargs_from_configdict,
)
from pyomo.common.dependencies import scipy, numpy as np
from pyomo.common.gc_manager import PauseGC
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
    maximize,
)
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import (
    FileDeterminism,
    FileDeterminism_to_SortComponents,
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
    c : scipy.sparse.csr_array

        The objective coefficients.  Note that this is a sparse array
        and may contain multiple rows (for multiobjective problems).  The
        objectives may be calculated by "c @ x"

    A : scipy.sparse.csc_array

        The constraint coefficients.  The constraint bodies may be
        calculated by "A @ x"

    rhs : numpy.ndarray

        The constraint right-hand sides.

    rows : List[Tuple[_ConstraintData, int]]

        The list of Pyomo constraint objects corresponding to the rows
        in `A`.  Each element in the list is a 2-tuple of
        (_ConstraintData, row_multiplier).  The `row_multiplier` will be
        +/- 1 indicating if the row was multiplied by -1 (corresponding
        to a constraint lower bound) or +1 (upper bound).

    columns : List[_VarData]

        The list of Pyomo variable objects corresponding to columns in
        the `A` and `c` matrices.

    eliminated_vars: List[Tuple[_VarData, NumericExpression]]

        The list of variables from the original model that do not appear
        in the standard form (usually because they were replaced by
        nonnegative variables).  Each entry is a 2-tuple of
        (:py:class:`_VarData`, :py:class`NumericExpression`|`float`).
        The list is in the necessary order for correct evaluation (i.e.,
        all variables appearing in the expression must either have
        appeared in the standard form, or appear *earlier* in this list.

    """

    def __init__(self, c, A, rhs, rows, columns, eliminated_vars):
        self.c = c
        self.A = A
        self.rhs = rhs
        self.rows = rows
        self.columns = columns
        self.eliminated_vars = eliminated_vars

    @property
    def x(self):
        return self.columns

    @property
    def b(self):
        return self.rhs


@WriterFactory.register(
    'compile_standard_form', 'Compile an LP to standard form (`min cTx s.t. Ax <= b`)'
)
class LinearStandardFormCompiler(object):
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
            description='Add slack variables and return `min cTx s.t. Ax == b`',
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
                NONE (0) : None
                ORDERED (10): rely on underlying component ordering (default)
                SORT_INDICES (20) : sort keys of indexed components
                SORT_SYMBOLS (30) : sort keys AND sort names (not declaration order)
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
            the resulting `A` matrix.  Unspecified constraints will
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
        """Convert a model to standard form (`min cTx s.t. Ax <= b`)

        Returns
        -------
        LinearStandardFormInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: None
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
    def __init__(self, config):
        self.config = config

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
                        "%s:\n\t\t%s" % (k, "\n\t\t".join(map(attrgetter('name'), v)))
                        for k, v in unknown.items()
                    ),
                )
            )

        self.var_map = var_map = {}
        initialize_var_map_from_column_order(model, self.config, var_map)
        var_order = {_id: i for i, _id in enumerate(var_map)}

        visitor = LinearRepnVisitor({}, var_map, var_order, sorter)

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
        if not component_map[Objective]:
            objectives = [Objective(expr=1)]
            objectives[0].construct()
        else:
            objectives = []
            for blk in component_map[Objective]:
                objectives.extend(
                    blk.component_data_objects(
                        Objective, active=True, descend_into=False, sort=sorter
                    )
                )
        obj_data = []
        obj_index = []
        obj_index_ptr = [0]
        for i, obj in enumerate(objectives):
            repn = visitor.walk_expression(obj.expr)
            if repn.nonlinear is not None:
                raise ValueError(
                    f"Model objective ({obj.name}) contains nonlinear terms that "
                    "cannot be compiled to standard (linear) form."
                )
            N = len(repn.linear)
            obj_data.append(np.fromiter(repn.linear.values(), float, N))
            if obj.sense == maximize:
                obj_data[-1] *= -1
            obj_index.append(
                np.fromiter(map(var_order.__getitem__, repn.linear), float, N)
            )
            obj_index_ptr.append(obj_index_ptr[-1] + N)
            if with_debug_timing:
                timer.toc('Objective %s', obj, level=logging.DEBUG)

        #
        # Tabulate constraints
        #
        slack_form = self.config.slack_form
        rows = []
        rhs = []
        con_data = []
        con_index = []
        con_index_ptr = [0]
        last_parent = None
        for con in ordered_active_constraints(model, self.config):
            if with_debug_timing and con.parent_component() is not last_parent:
                if last_parent is not None:
                    timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
                last_parent = con.parent_component()
            # Note: Constraint.lb/ub guarantee a return value that is
            # either a (finite) native_numeric_type, or None
            lb = con.lb
            ub = con.ub

            repn = visitor.walk_expression(con.body)

            if lb is None and ub is None:
                # Note: you *cannot* output trivial (unbounded)
                # constraints in matrix format.  I suppose we could add a
                # slack variable, but that seems rather silly.
                continue
            if repn.nonlinear is not None:
                raise ValueError(
                    f"Model constraint ({con.name}) contains nonlinear terms that "
                    "cannot be compiled to standard (linear) form."
                )

            # Pull out the constant: we will move it to the bounds
            offset = repn.constant
            repn.constant = 0

            if not repn.linear:
                if (lb is None or lb <= offset) and (ub is None or ub >= offset):
                    continue
                raise InfeasibleError(
                    f"model contains a trivially infeasible constraint, '{con.name}'"
                )

            if slack_form:
                _data = list(repn.linear.values())
                _index = list(map(var_order.__getitem__, repn.linear))
                if lb == ub:  # TODO: add tolerance?
                    rhs.append(ub - offset)
                else:
                    # add slack variable
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
                    var_order[id(v)] = slack_col = len(var_order)
                    _data.append(1)
                    _index.append(slack_col)
                rows.append(RowEntry(con, 1))
                con_data.append(np.array(_data))
                con_index.append(np.array(_index))
                con_index_ptr.append(con_index_ptr[-1] + len(_index))
            else:
                N = len(repn.linear)
                _data = np.fromiter(repn.linear.values(), float, N)
                _index = np.fromiter(map(var_order.__getitem__, repn.linear), float, N)
                if ub is not None:
                    rows.append(RowEntry(con, 1))
                    rhs.append(ub - offset)
                    con_data.append(_data)
                    con_index.append(_index)
                    con_index_ptr.append(con_index_ptr[-1] + N)
                if lb is not None:
                    rows.append(RowEntry(con, -1))
                    rhs.append(offset - lb)
                    con_data.append(-_data)
                    con_index.append(_index)
                    con_index_ptr.append(con_index_ptr[-1] + N)

        if with_debug_timing:
            # report the last constraint
            timer.toc('Constraint %s', last_parent, level=logging.DEBUG)

        # Get the variable list
        columns = list(var_map.values())
        # Convert the compiled data to scipy sparse matrices
        c = scipy.sparse.csr_array(
            (np.concatenate(obj_data), np.concatenate(obj_index), obj_index_ptr),
            [len(obj_index_ptr) - 1, len(columns)],
        ).tocsc()
        A = scipy.sparse.csr_array(
            (np.concatenate(con_data), np.concatenate(con_index), con_index_ptr),
            [len(rows), len(columns)],
        ).tocsc()

        # Some variables in the var_map may not actually appear in the
        # objective or constraints (e.g., added from col_order, or
        # multiplied by 0 in the expressions).  The easiest way to check
        # for empty columns is to convert from CSR to CSC and then look
        # at the index pointer list (an O(num_var) operation).
        c_ip = c.indptr
        A_ip = A.indptr
        active_var_idx = list(
            filter(
                lambda i: A_ip[i] != A_ip[i + 1] or c_ip[i] != c_ip[i + 1],
                range(len(columns)),
            )
        )
        nCol = len(active_var_idx)
        if nCol != len(columns):
            # Note that the indptr can't just use range() because a var
            # may only appear in the objectives or the constraints.
            columns = list(map(columns.__getitem__, active_var_idx))
            active_var_idx.append(c.indptr[-1])
            c = scipy.sparse.csc_array(
                (c.data, c.indices, c.indptr.take(active_var_idx)), [c.shape[0], nCol]
            )
            active_var_idx[-1] = A.indptr[-1]
            A = scipy.sparse.csc_array(
                (A.data, A.indices, A.indptr.take(active_var_idx)), [A.shape[0], nCol]
            )

        if self.config.nonnegative_vars:
            c, A, columns, eliminated_vars = _csc_to_nonnegative_vars(c, A, columns)
        else:
            eliminated_vars = []

        info = LinearStandardFormInfo(c, A, rhs, rows, columns, eliminated_vars)
        timer.toc("Generated linear standard form representation", delta=False)
        return info


def _csc_to_nonnegative_vars(c, A, columns):
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

    nCol = len(new_columns)
    c = scipy.sparse.csc_array(
        (np.concatenate(new_c_data), np.concatenate(new_c_indices), new_c_indptr),
        [c.shape[0], nCol],
    )
    A = scipy.sparse.csc_array(
        (np.concatenate(new_A_data), np.concatenate(new_A_indices), new_A_indptr),
        [A.shape[0], nCol],
    )
    return c, A, new_columns, eliminated_vars
