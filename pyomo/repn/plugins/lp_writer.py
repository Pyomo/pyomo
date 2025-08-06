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
from io import StringIO
from operator import itemgetter, attrgetter

from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    InEnum,
    document_kwargs_from_configdict,
)
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer

from pyomo.core.base import (
    Block,
    Objective,
    Constraint,
    Var,
    Param,
    Expression,
    SOSConstraint,
    Suffix,
    SymbolMap,
    minimize,
)
from pyomo.core.base.label import LPFileLabeler, NumericLabeler
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import (
    FileDeterminism,
    FileDeterminism_to_SortComponents,
    OrderedVarRecorder,
    categorize_valid_components,
    initialize_var_map_from_column_order,
    int_float,
    ordered_active_constraints,
    row_order2row_map,
)

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port

logger = logging.getLogger(__name__)
inf = float('inf')
neg_inf = float('-inf')


# TODO: make a proper base class
class LPWriterInfo(object):
    """Return type for LPWriter.write()

    Attributes
    ----------
    symbol_map: SymbolMap

        The :py:class:`SymbolMap` bimap between row/column labels and
        Pyomo components.

    """

    def __init__(self, symbol_map):
        self.symbol_map = symbol_map


@WriterFactory.register(
    'cpxlp_v2', 'Generate the corresponding CPLEX LP file (version 2).'
)
@WriterFactory.register('lp_v2', 'Generate the corresponding LP file (version 2).')
class LPWriter(object):
    CONFIG = ConfigBlock('lpwriter')
    CONFIG.declare(
        'show_section_timing',
        ConfigValue(
            default=False,
            domain=bool,
            description='Print timing after writing each section of the LP file',
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
            LP file is written deterministically for a Pyomo model:

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
            description='Write variables/constraints using model names',
            doc="""
            Export variables and constraints to the LP file using human-readable
            text names derived from the corresponding Pyomo component names.
            """,
        ),
    )
    CONFIG.declare(
        'row_order',
        ConfigValue(
            default=None,
            description='Preferred constraint ordering',
            doc="""
            List of constraints in the order that they should appear in the
            LP file.  Unspecified constraints will appear at the end.""",
        ),
    )
    CONFIG.declare(
        'column_order',
        ConfigValue(
            default=None,
            description='Preferred variable ordering',
            doc="""
            List of variables in the order that they should appear in
            the LP file.  Note that this is only a suggestion, as the LP
            file format is row-major and the columns are inferred from
            the order in which variables appear in the objective
            followed by each constraint.""",
        ),
    )
    CONFIG.declare(
        'labeler',
        ConfigValue(
            default=None,
            description='Callable to use to generate symbol names in LP file',
            doc="""
            Export variables and constraints to the LP file using human-readable
            text names derived from the corresponding Pyomo component names.
            """,
        ),
    )
    CONFIG.declare(
        'output_fixed_variable_bounds',
        ConfigValue(
            default=False,
            domain=bool,
            description='DEPRECATED option from LPv1 that has no effect in the LPv2',
        ),
    )
    CONFIG.declare(
        'allow_quadratic_objective',
        ConfigValue(
            default=True,
            domain=bool,
            description='If True, allow quadratic terms in the model objective',
        ),
    )
    CONFIG.declare(
        'allow_quadratic_constraint',
        ConfigValue(
            default=True,
            domain=bool,
            description='If True, allow quadratic terms in the model constraints',
        ),
    )

    def __init__(self):
        self.config = self.CONFIG()

    def __call__(self, model, filename, solver_capability, io_options):
        if filename is None:
            filename = model.name + ".lp"

        # Duplicate io_options to avoid side-effects
        io_options = dict(io_options)
        # Map old solver capabilities to new writer options
        qp = solver_capability('quadratic_objective')
        if 'allow_quadratic_objective' not in io_options:
            io_options['allow_quadratic_objective'] = qp
        qc = solver_capability('quadratic_constraint')
        if 'allow_quadratic_constraint' not in io_options:
            io_options['allow_quadratic_constraint'] = qc

        with open(filename, 'w', newline='') as FILE:
            info = self.write(model, FILE, **io_options)
        return filename, info.symbol_map

    @document_kwargs_from_configdict(CONFIG)
    def write(self, model, ostream, **options):
        """Write a model in LP format.

        Returns
        -------
        LPWriterInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: io.TextIOBase
            The text output stream where the LP "file" will be written.
            Could be an opened file or a io.StringIO.

        """
        config = self.config(options)

        if config.output_fixed_variable_bounds:
            deprecation_warning(
                "The 'output_fixed_variable_bounds' option to the LP "
                "writer is deprecated and is ignored by the lp_v2 writer."
            )

        # Pause the GC, as the walker that generates the compiled LP
        # representation generates (and disposes of) a large number of
        # small objects.
        with PauseGC():
            return _LPWriter_impl(ostream, config).write(model)


class _LPWriter_impl(object):
    def __init__(self, ostream, config):
        self.ostream = ostream
        self.config = config
        self.symbol_map = None

    def write(self, model):
        timing_logger = logging.getLogger('pyomo.common.timing.writer')
        timer = TicTocTimer(logger=timing_logger)
        with_debug_timing = (
            timing_logger.isEnabledFor(logging.DEBUG) and timing_logger.hasHandlers()
        )

        ostream = self.ostream

        labeler = self.config.labeler
        if labeler is None:
            if self.config.symbolic_solver_labels:
                labeler = LPFileLabeler()
            else:
                labeler = NumericLabeler('x')
        self.symbol_map = SymbolMap(labeler)
        addSymbol = self.symbol_map.addSymbol
        aliasSymbol = self.symbol_map.alias
        getSymbol = self.symbol_map.getSymbol

        self.sorter = sorter = FileDeterminism_to_SortComponents(
            self.config.file_determinism
        )
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
            targets={Suffix, SOSConstraint, Objective},
        )
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the LP writer does not know how to process:\n\t%s"
                % (
                    model.name,
                    "\n\t".join(
                        "%s:\n\t\t%s" % (k, "\n\t\t".join(map(attrgetter('name'), v)))
                        for k, v in unknown.items()
                    ),
                )
            )

        ONE_VAR_CONSTANT = Var(name='ONE_VAR_CONSTANT', bounds=(1, 1))
        ONE_VAR_CONSTANT.construct()

        self.var_map = {id(ONE_VAR_CONSTANT): ONE_VAR_CONSTANT}
        initialize_var_map_from_column_order(model, self.config, self.var_map)
        self.var_order = {_id: i for i, _id in enumerate(self.var_map)}
        self.var_recorder = OrderedVarRecorder(self.var_map, self.var_order, sorter)

        _qp = self.config.allow_quadratic_objective
        _qc = self.config.allow_quadratic_constraint
        objective_visitor = (QuadraticRepnVisitor if _qp else LinearRepnVisitor)(
            {}, var_recorder=self.var_recorder
        )
        constraint_visitor = (QuadraticRepnVisitor if _qc else LinearRepnVisitor)(
            objective_visitor.subexpression_cache if _qp == _qc else {},
            var_recorder=self.var_recorder,
        )

        timer.toc('Initialized column order', level=logging.DEBUG)

        # We don't export any suffix information to the LP file
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
                    + "\nLP writer cannot export suffixes to LP files.  Skipping."
                )

        ostream.write(f"\\* Source Pyomo model name={model.name} *\\\n\n")

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
        if len(objectives) > 1:
            raise ValueError(
                "More than one active objective defined for input model '%s'; "
                "Cannot write legal LP file\nObjectives: %s"
                % (model.name, ' '.join(obj.name for obj in objectives))
            )

        obj = objectives[0]
        ostream.write(
            ("min \n%s:\n" if obj.sense == minimize else "max \n%s:\n")
            % (getSymbol(obj, labeler),)
        )
        repn = objective_visitor.walk_expression(obj.expr)
        if repn.nonlinear is not None:
            raise ValueError(
                f"Model objective ({obj.name}) contains nonlinear terms that "
                "cannot be written to LP format"
            )
        if repn.constant or not (repn.linear or getattr(repn, 'quadratic', None)):
            # Older versions of CPLEX (including 12.6) and all versions
            # of GLPK (through 5.0) do not support constants in the
            # objective in LP format.  To avoid painful bookkeeping, we
            # introduce the following "variable", constrained to the
            # value 1.
            #
            # In addition, most solvers do no tolerate an empty
            # objective, this will ensure we at least write out
            # 0*ONE_VAR_CONSTANT.
            repn.linear[id(ONE_VAR_CONSTANT)] = repn.constant
            repn.constant = 0
        self.write_expression(ostream, repn, True)
        aliasSymbol(obj, '__default_objective__')
        if with_debug_timing:
            timer.toc('Objective %s', obj, level=logging.DEBUG)

        ostream.write("\ns.t.\n")

        #
        # Tabulate constraints
        #
        skip_trivial_constraints = self.config.skip_trivial_constraints
        have_nontrivial = False
        last_parent = None
        for con in ordered_active_constraints(model, self.config):
            if with_debug_timing and con.parent_component() is not last_parent:
                timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
                last_parent = con.parent_component()
            # Note: Constraint.to_bounded_expression(evaluate_bounds=True)
            # guarantee a return value that is either a (finite)
            # native_numeric_type, or None
            lb, body, ub = con.to_bounded_expression(True)

            if lb is None and ub is None:
                # Note: you *cannot* output trivial (unbounded)
                # constraints in LP format.  I suppose we could add a
                # slack variable if skip_trivial_constraints is False,
                # but that seems rather silly.
                continue
            repn = constraint_visitor.walk_expression(body)
            if repn.nonlinear is not None:
                raise ValueError(
                    f"Model constraint ({con.name}) contains nonlinear terms that "
                    "cannot be written to LP format"
                )

            # Pull out the constant: we will move it to the bounds
            offset = repn.constant
            repn.constant = 0

            if repn.linear or getattr(repn, 'quadratic', None):
                have_nontrivial = True
            else:
                if (
                    skip_trivial_constraints
                    and (lb is None or lb <= offset)
                    and (ub is None or ub >= offset)
                ):
                    continue
                # This is a trivially infeasible model.  We could raise
                # an exception, or we could allow the solver to return
                # infeasible.  There are fewer logic paths (in
                # particular related to mapping solver result status) if
                # we just defer to the solver.
                #
                # Add a dummy (fixed) variable to the constraint,
                # because some solvers (including versions of GLPK)
                # cannot parse an LP file without a variable on the left
                # hand side.
                repn.linear[id(ONE_VAR_CONSTANT)] = 0

            symbol = labeler(con)
            if lb is not None:
                if ub is None:
                    label = f'c_l_{symbol}_'
                    addSymbol(con, label)
                    ostream.write(f'\n{label}:\n')
                    self.write_expression(ostream, repn, False)
                    ostream.write(f'>= {(lb - offset)!s}\n')
                elif lb == ub:
                    label = f'c_e_{symbol}_'
                    addSymbol(con, label)
                    ostream.write(f'\n{label}:\n')
                    self.write_expression(ostream, repn, False)
                    ostream.write(f'= {(lb - offset)!s}\n')
                else:
                    # We will need the constraint body twice.  Generate
                    # in a buffer so we only have to do that once.
                    buf = StringIO()
                    self.write_expression(buf, repn, False)
                    buf = buf.getvalue()
                    #
                    label = f'r_l_{symbol}_'
                    addSymbol(con, label)
                    ostream.write(f'\n{label}:\n')
                    ostream.write(buf)
                    ostream.write(f'>= {(lb - offset)!s}\n')
                    label = f'r_u_{symbol}_'
                    aliasSymbol(con, label)
                    ostream.write(f'\n{label}:\n')
                    ostream.write(buf)
                    ostream.write(f'<= {(ub - offset)!s}\n')
            elif ub is not None:
                label = f'c_u_{symbol}_'
                addSymbol(con, label)
                ostream.write(f'\n{label}:\n')
                self.write_expression(ostream, repn, False)
                ostream.write(f'<= {(ub - offset)!s}\n')

        if with_debug_timing:
            # report the last constraint
            timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
        if not have_nontrivial:
            # Some solvers (notably CBC through at least 2.10.4) will
            # return a nonzero return code when the model has no
            # constraints.  To work around the original Pyomo solver
            # hierarchy (where the return code was processed in the base
            # class), we will add a dummy constraint here.
            repn = constraint_visitor.Result()  # walk_expression(ONE_VAR_CONSTANT)
            repn.linear[id(ONE_VAR_CONSTANT)] = 1
            ostream.write(f'\nc_e_ONE_VAR_CONSTANT:\n')
            self.write_expression(ostream, repn, False)
            ostream.write(f'= 1\n')

        ostream.write("\nbounds")

        # Track the number of integer and binary variables, so you can
        # output their status later.
        integer_vars = []
        binary_vars = []
        getSymbolByObjectID = self.symbol_map.byObject.get
        for vid, v in self.var_map.items():
            # Some variables in the var_map may not actually have been
            # written out to the LP file (e.g., added from col_order, or
            # multiplied by 0 in the expressions).  Check to see that
            # the variable is in the symbol_map before outputting.
            v_symbol = getSymbolByObjectID(vid, None)
            if not v_symbol:
                continue
            if v.is_binary():
                binary_vars.append(v_symbol)
            elif v.is_integer():
                integer_vars.append(v_symbol)

            # Note: Var.bounds guarantees the values are either (finite)
            # native_numeric_types or None
            lb, ub = v.bounds
            lb = '-inf' if lb is None else str(lb)
            ub = '+inf' if ub is None else str(ub)
            ostream.write(f"\n   {lb} <= {v_symbol} <= {ub}")

        if integer_vars:
            ostream.write("\ngeneral\n  ")
            ostream.write("\n  ".join(integer_vars))

        if binary_vars:
            ostream.write("\nbinary\n  ")
            ostream.write("\n  ".join(binary_vars))

        timer.toc("Wrote variable bounds and domains", level=logging.DEBUG)

        #
        # Tabulate SOS constraints
        #
        if component_map[SOSConstraint]:
            sos = []
            for blk in component_map[SOSConstraint]:
                sos.extend(
                    blk.component_data_objects(
                        SOSConstraint, active=True, descend_into=False, sort=sorter
                    )
                )
            if self.config.row_order:
                row_map = row_order2row_map(self.config)
                _n = len(row_map)
                sos.sort(key=lambda x: row_map.get(id(x), _n))

            ostream.write("\nSOS\n")
            for soscon in sos:
                ostream.write(f'\n{getSymbol(soscon)}: S{soscon.level}::\n')
                for v, w in getattr(soscon, 'get_items', soscon.items)():
                    if w.__class__ not in int_float:
                        w = float(w)
                    ostream.write(f"  {getSymbol(v)}:{w!s}\n")

        ostream.write("\nend\n")

        info = LPWriterInfo(self.symbol_map)
        timer.toc("Generated LP representation", delta=False)
        return info

    def write_expression(self, ostream, expr, is_objective):
        assert not expr.constant
        getSymbol = self.symbol_map.getSymbol
        getVarOrder = self.var_order.__getitem__
        getVar = self.var_map.__getitem__

        if expr.linear:
            for vid, coef in sorted(
                expr.linear.items(), key=lambda x: getVarOrder(x[0])
            ):
                if coef < 0:
                    ostream.write(f'{coef!s} {getSymbol(getVar(vid))}\n')
                else:
                    ostream.write(f'+{coef!s} {getSymbol(getVar(vid))}\n')

        quadratic = getattr(expr, 'quadratic', None)
        if quadratic:

            def _normalize_constraint(data):
                (vid1, vid2), coef = data
                c1 = getVarOrder(vid1)
                c2 = getVarOrder(vid2)
                if c2 < c1:
                    col = c2, c1
                    sym = f' {getSymbol(getVar(vid2))} * {getSymbol(getVar(vid1))}\n'
                elif c1 == c2:
                    col = c1, c1
                    sym = f' {getSymbol(getVar(vid2))} ^ 2\n'
                else:
                    col = c1, c2
                    sym = f' {getSymbol(getVar(vid1))} * {getSymbol(getVar(vid2))}\n'
                if coef < 0:
                    return col, str(coef) + sym
                else:
                    return col, f'+{coef!s}{sym}'

            if is_objective:
                #
                # Times 2 because LP format requires /2 for all the
                # quadratic terms /of the objective only/.  Discovered
                # the last bit through trial and error.
                # Ref: ILog CPlex 8.0 User's Manual, p197.
                #
                def _normalize_objective(data):
                    vids, coef = data
                    return _normalize_constraint((vids, 2 * coef))

                _normalize = _normalize_objective
            else:
                _normalize = _normalize_constraint

            ostream.write('+ [\n')
            quadratic = sorted(map(_normalize, quadratic.items()), key=itemgetter(0))
            ostream.write(''.join(map(itemgetter(1), quadratic)))
            if is_objective:
                ostream.write("] / 2\n")
            else:
                ostream.write("]\n")
