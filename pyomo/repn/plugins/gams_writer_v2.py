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
    SortComponents,
    Suffix,
    SymbolMap,
    minimize,
    ShortNameLabeler,
)
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.label import NumericLabeler
from pyomo.opt import WriterFactory
from pyomo.repn.util import ftoa

# from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import (
    FileDeterminism,
    FileDeterminism_to_SortComponents,
    OrderedVarRecorder,
    categorize_valid_components,
    initialize_var_map_from_column_order,
    int_float,
    ordered_active_constraints,
)

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port

logger = logging.getLogger(__name__)
inf = float('inf')
neg_inf = float('-inf')


class GAMSWriterInfo(object):
    """Return type for GAMSWriter.write()

    Attributes
    ----------
    symbol_map: SymbolMap

        The :py:class:`SymbolMap` bimap between row/column labels and
        Pyomo components.
    """

    def __init__(self, var_symbol_map, con_symbol_map):
        self.var_symbol_map = var_symbol_map
        self.con_symbol_map = con_symbol_map


@WriterFactory.register(
    'gams_writer_v2', 'Generate the corresponding gms file (version 2).'
)
class GAMSWriter(object):
    CONFIG = ConfigBlock('gamswriter')

    """
    Write a model in the GAMS modeling language format.

    Keyword Arguments
    -----------------
    output_filename: str
        Name of file to write GAMS model to. Optionally pass a file-like
        stream and the model will be written to that instead.
    io_options: str
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
                - NONE (0) : None
                - ORDERED (10): rely on underlying component ordering (default)
                - SORT_INDICES (20) : sort keys of indexed components
                - SORT_SYMBOLS (30) : sort keys AND sort names (not declaration order)
        - put_results='results'
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
    # old GAMS config
    CONFIG.declare(
        'warmstart',
        ConfigValue(
            default=True,
            domain=bool,
            description="Warmstart by initializing model's variables to their values.",
        ),
    )
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description='Write variables/constraints using model names',
            doc="""
            Export variables and constraints to the gms file using human-readable
            text names derived from the corresponding Pyomo component names.
            """,
        ),
    )
    CONFIG.declare(
        'labeler',
        ConfigValue(
            default=None,
            description='Callable to use to generate symbol names in gms file',
        ),
    )
    CONFIG.declare(
        'solver',
        ConfigValue(
            default=None,
            description='If None, GAMS will use default solver for model type.',
        ),
    )
    CONFIG.declare(
        'mtype',
        ConfigValue(
            default=None,
            description='Model type. If None, will chose from lp, mip. nlp and minlp will be implemented in the future.',
        ),
    )
    CONFIG.declare(
        'add_options',
        ConfigValue(
            default=None,
            doc="""
            List of additional lines to write directly
            into model file before the solve statement.
            For model attributes, <model name> is GAMS_MODEL.
            """,
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
        'output_fixed_variables',
        ConfigValue(
            default=False,
            domain=bool,
            description='If True, output fixed variables as variables; otherwise,output numeric value',
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
            GAMS file is written deterministically for a Pyomo model:

               - NONE (0) : None
               - ORDERED (10): rely on underlying component ordering (default)
               - SORT_INDICES (20) : sort keys of indexed components
               - SORT_SYMBOLS (30) : sort keys AND sort names (not declaration order)

            """,
        ),
    )
    CONFIG.declare(
        'put_results',
        ConfigValue(
            default='results',
            domain=str,
            doc="""
            Filename for optionally writing solution values and
            marginals.  If put_results_format is 'gdx', then GAMS
            will write solution values and marginals to
            GAMS_MODEL_p.gdx and solver statuses to
            {put_results}_s.gdx.  If put_results_format is 'dat',
            then solution values and marginals are written to
            (put_results).dat, and solver statuses to (put_results +
            'stat').dat.
            """,
        ),
    )
    CONFIG.declare(
        'put_results_format',
        ConfigValue(
            default='gdx',
            description="Format used for put_results, one of 'gdx', 'dat'",
        ),
    )
    # NOTE: Taken from the lp_writer
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
    CONFIG.declare(
        'row_order',
        ConfigValue(
            default=None,
            description='Preferred constraint ordering',
            doc="""
            To use with ordered_active_constraints function.""",
        ),
    )

    def __init__(self):
        self.config = self.CONFIG()

    def __call__(self, model, filename, solver_capability, io_options):
        if filename is None:
            filename = model.name + ".gms"

        config = self.config(io_options)

        with open(filename, 'w', newline='') as FILE:
            info = self.write(model, FILE, config=config)

        return filename, info.symbol_map

    @document_kwargs_from_configdict(CONFIG)
    def write(self, model, ostream, **options) -> GAMSWriterInfo:
        """Write a model in GMS format.

        Returns
        -------
        GAMSWriterInfo

        Parameters
        -------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: io.TextIOBase
            The text output stream where the GMS "file" will be written.
            Could be an opened file or a io.StringIO.
        """
        config = self.config(options)

        # Pause the GC, as the walker that generates the compiled GMS
        # representation generates (and disposes of) a large number of
        # small objects.

        # NOTE: First pass write the model but needs variables/equations definition first
        with PauseGC():
            return _GMSWriter_impl(ostream, config).write(model)


class _GMSWriter_impl(object):
    def __init__(self, ostream, config):
        # taken from lp_writer.py
        self.ostream = ostream
        self.config = config
        self.symbol_map = None

        # Taken from nl_writer.py
        self.symbolic_solver_labels = config.symbolic_solver_labels
        self.add_options = config.add_options
        self.subexpression_cache = {}
        self.subexpression_order = None  # set to [] later
        self.external_functions = {}
        self.used_named_expressions = set()
        self.var_map = {}
        self.var_id_to_nl_map = {}
        self.next_V_line_id = 0
        self.pause_gc = None

    def write(self, model):
        timing_logger = logging.getLogger('pyomo.common.timing.writer')
        timer = TicTocTimer(logger=timing_logger)
        with_debug_timing = (
            timing_logger.isEnabledFor(logging.DEBUG) and timing_logger.hasHandlers()
        )

        # Caching some frequently-used objects into the locals()
        symbolic_solver_labels = self.symbolic_solver_labels
        add_options = self.add_options
        ostream = self.ostream
        config = self.config
        labeler = config.labeler
        var_labeler, con_labeler = None, None
        warmstart = config.warmstart

        sorter = FileDeterminism_to_SortComponents(config.file_determinism)

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

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError(
                "GAMS writer: Using both the "
                "'symbolic_solver_labels' and 'labeler' "
                "I/O options is forbidden"
            )

        if symbolic_solver_labels:
            # Note that the Var and Constraint labelers must use the
            # same labeler, so that we can correctly detect name
            # collisions (which can arise when we truncate the labels to
            # the max allowable length.  GAMS requires all identifiers
            # to start with a letter.  We will (randomly) choose "s_"
            # (for 'shortened')
            var_labeler = con_labeler = ShortNameLabeler(
                60,
                prefix='s_',
                suffix='_',
                caseInsensitive=True,
                legalRegex='^[a-zA-Z]',
            )
        elif labeler is None:
            var_labeler = NumericLabeler('x')
            con_labeler = NumericLabeler('c')
        else:
            var_labeler = con_labeler = labeler

        self.var_symbol_map = SymbolMap(var_labeler)
        self.con_symbol_map = SymbolMap(con_labeler)
        self.var_order = {_id: i for i, _id in enumerate(self.var_map)}
        self.var_recorder = OrderedVarRecorder(self.var_map, self.var_order, sorter)

        visitor = LinearRepnVisitor(
            self.subexpression_cache, var_recorder=self.var_recorder
        )

        #
        # Tabulate constraints
        #
        skip_trivial_constraints = self.config.skip_trivial_constraints
        last_parent = None
        con_list = (
            {}
        )  # NOTE: Save the constraint representation and write it after variables/equations declare
        for con in ordered_active_constraints(model, self.config):
            if with_debug_timing and con.parent_component() is not last_parent:
                timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
                last_parent = con.parent_component()
            # Note: Constraint.to_bounded_expression(evaluate_bounds=True)
            # guarantee a return value that is either a (finite)
            # native_numeric_type, or None
            lb, body, ub = con.to_bounded_expression(True)

            if lb is None and ub is None:
                # WIP: handling unbounded variable
                continue
            repn = visitor.walk_expression(body)
            if repn.nonlinear is not None:
                raise ValueError(
                    f"Model constraint ({con.name}) contains nonlinear terms that is currently not supported in the new gams_writer"
                )

            # Pull out the constant: we will move it to the bounds
            offset = repn.constant
            repn.constant = 0

            if repn.linear or getattr(repn, 'quadratic', None):
                pass
            else:
                if (
                    skip_trivial_constraints
                    and (lb is None or lb <= offset)
                    and (ub is None or ub >= offset)
                ):
                    continue

            con_symbol = con_labeler(con)
            declaration, definition, bounds = None, None, None

            if lb is not None:
                if ub is None:
                    label = f'{con_symbol}_lo'
                    self.con_symbol_map.addSymbol(con, label)
                    self.var_symbol_map.addSymbol(con, label)
                    declaration = f'\n{label}.. '
                    definition = self.write_expression(ostream, repn)
                    bounds = f' =G= {(lb - offset)!s};'
                    con_list[label] = declaration + definition + bounds
                elif lb == ub:
                    label = f'{con_symbol}'
                    self.con_symbol_map.addSymbol(con, label)
                    self.var_symbol_map.addSymbol(con, label)
                    declaration = f'\n{label}.. '
                    definition = self.write_expression(ostream, repn)
                    bounds = f' =E= {(lb - offset)!s};'
                    con_list[label] = declaration + definition + bounds
                else:
                    raise NotImplementedError(
                        "Bounded constraints within the same expression is not supported"
                    )

            elif ub is not None:
                label = f'{con_symbol}_hi'
                self.con_symbol_map.addSymbol(con, label)
                declaration = f'\n{label}.. '
                definition = self.write_expression(ostream, repn)
                bounds = f' =L= {(lb - offset)!s};'
                con_list[label] = declaration + definition + bounds

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
        repn = visitor.walk_expression(obj.expr)
        if repn.nonlinear is not None:
            raise ValueError(
                f"Model objective ({obj.name}) contains nonlinear terms that "
                "is currently not supported in this new GAMSWriter"
            )

        label = self.con_symbol_map.getSymbol(obj, con_labeler)
        declaration = f'\n{label}.. -GAMS_OBJECTIVE '
        definition = self.write_expression(ostream, repn, True)
        bounds = f' =E= {(-repn.constant)!s};\n\n'
        con_list[label] = declaration + definition + bounds
        self.var_symbol_map.addSymbol(obj, label)

        #
        # Write out variable declaration
        #
        integer_vars = []
        binary_vars = []
        var_bounds = {}
        getSymbolByObjectID = self.var_symbol_map.byObject.get

        ostream.write("VARIABLES \n")
        for vid, v in self.var_map.items():
            v_symbol = getSymbolByObjectID(vid, None)
            if not v_symbol:
                continue
            if v.is_continuous():
                ostream.write(f"\t{v_symbol} \n")
                lb, ub = v.bounds
                var_bounds[v_symbol] = (lb, ub)
            elif v.is_binary():
                binary_vars.append(v_symbol)
            elif v.is_integer():
                lb, ub = v.bounds
                var_bounds[v_symbol] = (lb, ub)
                integer_vars.append(v_symbol)

        ostream.write(f"\tGAMS_OBJECTIVE;\n\n")

        if integer_vars:
            ostream.write("\nINTEGER VARIABLES\n\t")
            ostream.write("\n\t".join(integer_vars) + ';\n\n')

        if binary_vars:
            ostream.write("\nBINARY VARIABLES\n\t")
            ostream.write("\n\t".join(binary_vars) + ';\n\n')

        #
        # Writing out the equations/constraints
        #
        ostream.write("EQUATIONS \n")
        for id, cid in enumerate(self.con_symbol_map.byObject.keys()):
            c = self.con_symbol_map.byObject[cid]
            if id != len(self.con_symbol_map.byObject.keys()) - 1:
                ostream.write(f"\t{c}\n")
            else:
                ostream.write(f"\t{c};\n\n")

        for con_label, con in con_list.items():
            ostream.write(con)

        #
        # Handling variable bounds
        #
        for v, (lb, ub) in var_bounds.items():
            pyomo_v = self.var_symbol_map.bySymbol[v]
            if lb is not None:
                ostream.write(f'{v}.lo = {lb};\n')
            if ub is not None:
                ostream.write(f'{v}.up = {ub};\n')
            if warmstart and pyomo_v.value is not None:
                ostream.write("%s.l = %s;\n" % (v, ftoa(pyomo_v.value, False)))
        ostream.write(f'\nModel {model.name} / all /;\n')
        ostream.write(f'{model.name}.limrow = 0;\n')
        ostream.write(f'{model.name}.limcol = 0;\n')

        # CHECK FOR mtype flag based on variable domains - reals, integer
        if config.mtype is None:
            if binary_vars or integer_vars:
                config.mtype = 'mip'  # expand this to nlp, minlp
            else:
                config.mtype = 'lp'

        if config.put_results_format == 'gdx':
            ostream.write("option savepoint=1;\n")

        if add_options is not None:
            ostream.write("\n* START USER ADDITIONAL OPTIONS\n")
            for line in add_options:
                ostream.write('option ' + line + '\n')
            ostream.write("\n\n* END USER ADDITIONAL OPTIONS\n\n")

        ostream.write(
            "SOLVE %s USING %s %simizing GAMS_OBJECTIVE;\n"
            % (model.name, config.mtype, 'min' if obj.sense == minimize else 'max')
        )
        # Set variables to store certain statuses and attributes
        stat_vars = [
            'MODELSTAT',
            'SOLVESTAT',
            'OBJEST',
            'OBJVAL',
            'NUMVAR',
            'NUMEQU',
            'NUMDVAR',
            'NUMNZ',
            'ETSOLVE',
        ]
        ostream.write("\nScalars MODELSTAT 'model status', SOLVESTAT 'solve status';\n")
        ostream.write("MODELSTAT = %s.modelstat;\n" % model.name)
        ostream.write("SOLVESTAT = %s.solvestat;\n\n" % model.name)

        ostream.write("Scalar OBJEST 'best objective', OBJVAL 'objective value';\n")
        ostream.write("OBJEST = %s.objest;\n" % model.name)
        ostream.write("OBJVAL = %s.objval;\n\n" % model.name)

        ostream.write("Scalar NUMVAR 'number of variables';\n")
        ostream.write("NUMVAR = %s.numvar\n\n" % model.name)

        ostream.write("Scalar NUMEQU 'number of equations';\n")
        ostream.write("NUMEQU = %s.numequ\n\n" % model.name)

        ostream.write("Scalar NUMDVAR 'number of discrete variables';\n")
        ostream.write("NUMDVAR = %s.numdvar\n\n" % model.name)

        ostream.write("Scalar NUMNZ 'number of nonzeros';\n")
        ostream.write("NUMNZ = %s.numnz\n\n" % model.name)

        ostream.write("Scalar ETSOLVE 'time to execute solve statement';\n")
        ostream.write("ETSOLVE = %s.etsolve\n\n" % model.name)

        if config.put_results is not None:
            if config.put_results_format == 'gdx':
                ostream.write("\nexecute_unload '%s_s.gdx'" % config.put_results)
                for stat in stat_vars:
                    ostream.write(", %s" % stat)
                ostream.write(";\n")
            else:
                results = config.put_results + '.dat'
                ostream.write("\nfile results /'%s'/;" % results)
                ostream.write("\nresults.nd=15;")
                ostream.write("\nresults.nw=21;")
                ostream.write("\nput results;")
                ostream.write("\nput 'SYMBOL  :  LEVEL  :  MARGINAL' /;")
                for sym, var in self.var_symbol_map.bySymbol.items():
                    if var.parent_component().ctype is Var:
                        ostream.write("\nput %s ' ' %s.l ' ' %s.m /;" % (sym, sym, sym))
                for con in self.con_symbol_map.bySymbol.keys():
                    ostream.write("\nput %s ' ' %s.l ' ' %s.m /;" % (con, con, con))
                ostream.write(
                    "\nput GAMS_OBJECTIVE ' ' GAMS_OBJECTIVE.l "
                    "' ' GAMS_OBJECTIVE.m;\n"
                )

                statresults = config.put_results + 'stat.dat'
                ostream.write("\nfile statresults /'%s'/;" % statresults)
                ostream.write("\nstatresults.nd=15;")
                ostream.write("\nstatresults.nw=21;")
                ostream.write("\nput statresults;")
                ostream.write("\nput 'SYMBOL   :   VALUE' /;")
                for stat in stat_vars:
                    ostream.write("\nput '%s' ' ' %s /;\n" % (stat, stat))

        timer.toc("Finished writing .gsm file", level=logging.DEBUG)

        info = GAMSWriterInfo(self.var_symbol_map, self.con_symbol_map)
        return info

    def write_expression(self, ostream, expr, is_objective=False):
        if not is_objective:
            assert not expr.constant
        getSymbol = self.var_symbol_map.getSymbol
        getVarOrder = self.var_order.__getitem__
        getVar = self.var_map.__getitem__
        expr_str = ''
        if expr.linear:
            for vid, coef in sorted(
                expr.linear.items(), key=lambda x: getVarOrder(x[0])
            ):
                if coef < 0:
                    # ostream.write(f'{coef!s}*{getSymbol(getVar(vid))}')
                    expr_str += f'{coef!s}*{getSymbol(getVar(vid))} \n'
                else:
                    # ostream.write(f'+{coef!s}*{getSymbol(getVar(vid))}')
                    expr_str += f'+ {coef!s} * {getSymbol(getVar(vid))} \n'

        return expr_str
