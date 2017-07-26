#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Problem Writer for GAMS Format Files
#

from six import StringIO, string_types

from pyutilib.misc import PauseGC

from pyomo.core.base import (
    SymbolMap, AlphaNumericTextLabeler, NumericLabeler, 
    Block, Constraint, Expression, Objective, Var, Set, RangeSet, Param,
    value, minimize, Suffix, SortComponents)

from pyomo.core.base.component import ComponentData
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter
import pyomo.util.plugin

from pyomo.core.kernel import (block, constraint, expression, objective,
    variable, parameter, suffix, linear_constraint)

from pyomo.core.kernel.component_interface import ICategorizedObject

import logging

logger = logging.getLogger('pyomo.core')


class ProblemWriter_gams(AbstractProblemWriter):
    pyomo.util.plugin.alias('gams', 'Generate the corresponding GAMS file')

    def __init__(self):
        AbstractProblemWriter.__init__(self, ProblemFormat.gams)

    def __call__(self,
                 model,
                 output_filename,
                 solver_capability,
                 io_options):
        """
        output_filename:
            Name of file to write GAMS model to. Optionally pass a file-like
            stream and the model will be written to that instead.
        io_options:
            symbolic_solver_labels=False:
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
            labeler=None:
                Custom labeler option. Incompatible with symbolic_solver_labels.
            solver=None:
                If None, GAMS will use default solver for model type.
            mtype=None:
                Model type. If None, will chose from lp, nlp, mip, and minlp.
            add_options:
                List of additional lines to write directly
                into model file before the solve statement.
                For model attributes, <model name> = GAMS_MODEL
            put_results:
                Filename for optionally writing solution values and marginals.
        """

        # Make sure not to modify the user's dictionary,
        # they may be reusing it outside of this call
        io_options = dict(io_options)

        # Use full Pyomo component names rather than
        # shortened symbols (slower, but useful for debugging).
        symbolic_solver_labels = io_options.pop("symbolic_solver_labels", False)

        # Custom labeler option. Incompatible with symbolic_solver_labels.
        labeler = io_options.pop("labeler", None)

        # If None, GAMS will use default solver for model type.
        solver = io_options.pop("solver", None)

        # If None, will chose from lp, nlp, mip, and minlp.
        mtype = io_options.pop("mtype", None)

        # Lines to add before solve statement.
        add_options = io_options.pop("add_options", None)

        # Skip writing constraints whose body section is
        # fixed (i.e., no variables)
        skip_trivial_constraints = \
            io_options.pop("skip_trivial_constraints", True)

        # How much effort do we want to put into ensuring the
        # LP file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)
        sorter_map = {0:SortComponents.unsorted,
                      1:SortComponents.deterministic,
                      2:SortComponents.sortBoth}
        sort = sorter_map[file_determinism]

        # Filename for optionally writing solution values and marginals
        # Set to True by GAMSSolver
        put_results = io_options.pop("put_results", None)

        if len(io_options):
            raise ValueError(
                "ProblemWriter_gams passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s"
                            % (k,v) for k,v in io_options.iteritems()))

        if solver is not None:
            if solver.upper() not in valid_solvers:
                raise ValueError("ProblemWriter_gams passed unrecognized "
                                 "solver: %s" % solver)

        if mtype is not None:
            valid_mtypes = set([
                'lp', 'qcp', 'nlp', 'dnlp', 'rmip', 'mip', 'rmiqcp', 'rminlp',
                'miqcp', 'minlp', 'rmpec', 'mpec', 'mcp', 'cns', 'emp'])
            if mtype.lower() not in valid_mtypes:
                raise ValueError("ProblemWriter_gams passed unrecognized "
                                 "model type: %s" % mtype)
            if (solver is not None and
                mtype.upper() not in valid_solvers[solver.upper()]):
                raise ValueError("ProblemWriter_gams passed solver (%s) "
                                 "unsuitable for given model type (%s)"
                                 % (solver, mtype))

        if output_filename is None:
            output_filename = model.name + ".gms"

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError("ProblemWriter_gams: Using both the "
                             "'symbolic_solver_labels' and 'labeler' "
                             "I/O options is forbidden")

        if symbolic_solver_labels:
            var_labeler = con_labeler = AlphaNumericTextLabeler()
        elif labeler is None:
            var_labeler = NumericLabeler('x')
            con_labeler = NumericLabeler('c')
        else:
            var_labeler = con_labeler = labeler

        var_list = []
        symbolMap = SymbolMap()

        def var_recorder(obj):
            ans = var_labeler(obj)
            var_list.append(ans)
            return ans

        def var_label(obj):
            if obj.is_fixed():
                return str(value(obj))
            return symbolMap.getSymbol(obj, var_recorder)

        # when sorting, there are a non-trivial number of
        # temporary objects created. these all yield
        # non-circular references, so disable GC - the
        # overhead is non-trivial, and because references
        # are non-circular, everything will be collected
        # immediately anyway.
        with PauseGC() as pgc:
            try:
                if isinstance(output_filename, string_types):
                    output_file = open(output_filename, "w")
                else:
                    # Support passing of stream such as a StringIO
                    # on which to write the model file
                    output_file = output_filename
                if type(model) is block:
                    # Kernel
                    ICategorizedObject.labeler.append(var_label)
                else:
                    ComponentData.labeler.append(var_label)
                self._write_model(
                    model=model,
                    output_file=output_file,
                    solver_capability=solver_capability,
                    var_list=var_list,
                    symbolMap=symbolMap,
                    con_labeler=con_labeler,
                    sort=sort,
                    skip_trivial_constraints=skip_trivial_constraints,
                    solver=solver,
                    mtype=mtype,
                    add_options=add_options,
                    put_results=put_results
                )
            finally:
                if isinstance(output_filename, string_types):
                    output_file.close()
                if type(model) is block:
                    # Kernel
                    ICategorizedObject.labeler.pop()
                else:
                    ComponentData.labeler.pop()

        return output_filename, symbolMap

    def _write_model(self,
                     model,
                     output_file,
                     solver_capability,
                     var_list,
                     symbolMap,
                     con_labeler,
                     sort,
                     skip_trivial_constraints,
                     solver,
                     mtype,
                     add_options,
                     put_results):
        constraint_names = []
        ConstraintIO = StringIO()
        linear = True
        linear_degree = set([0,1])

        # Sanity check: all active components better be things we know
        # how to deal with, plus Suffix if solving
        valid_ctypes = set([
            Block, Constraint, Expression, Objective, Param, 
            Set, RangeSet, Var, Suffix ])
        valid_kernel_ctypes = set([
            block, constraint, expression, objective,
            variable, parameter, suffix, linear_constraint])
        for c in model.component_objects(active=True):
            if type(model) is block:
                # Kernel
                if type(c) not in valid_kernel_ctypes:
                    raise RuntimeError(
                        "Unallowable component %s.\nThe GAMS writer cannot "
                        "export models with this component type" %
                        ( type(c).__name__, ))
            elif c.type() not in valid_ctypes:
                raise RuntimeError(
                    "Unallowable component %s.\nThe GAMS writer cannot export"
                    " models with this component type" % ( c.type().__name__, ))

        # Walk through the model and generate the constraint definition
        # for all active constraints.  Any Vars / Expressions that are
        # encountered will be added to the var_list due to the labeler
        # defined above.
        for con in model.component_data_objects(Constraint,
                                                active=True,
                                                sort=sort):
            if skip_trivial_constraints and con.body.is_fixed():
                continue
            if linear:
                if con.body.polynomial_degree() not in linear_degree:
                    linear = False
            body = str(con.body)
            cName = symbolMap.getSymbol(con, con_labeler)
            if con.equality:
                constraint_names.append('%s' % cName)
                ConstraintIO.write('%s.. %s =e= %s;\n' % (
                    constraint_names[-1],
                    body,
                    value(con.upper)
                ))
            else:
                if con.lower is not None:
                    constraint_names.append('%s_lo' % cName)
                    ConstraintIO.write('%s.. %s =l= %s;\n' % (
                        constraint_names[-1],
                        value(con.lower),
                        body
                    ))
                if con.upper is not None:
                    constraint_names.append('%s_hi' % cName)
                    ConstraintIO.write('%s.. %s =l= %s;\n' % (
                        constraint_names[-1],
                        body,
                        value(con.upper)
                    ))

        obj = list(model.component_data_objects(Objective,
                                                active=True,
                                                sort=sort))
        if len(obj) != 1:
            raise RuntimeError(
                "GAMS writer requires exactly one active objective (found %s)"
                % (len(obj)))
        obj = obj[0]
        if linear:
            if obj.expr.polynomial_degree() not in linear_degree:
                linear = False
        oName = symbolMap.getSymbol(obj, con_labeler)
        body = str(obj.expr)
        constraint_names.append(oName)
        ConstraintIO.write('%s.. GAMS_OBJECTIVE =e= %s;\n' % (
            oName,
            body
        ))

        # Categorize the variables that we found
        binary = []
        posInts = []
        otherInts = []
        positive = []
        reals = []
        i = 0
        numVar = len(var_list)
        while i < numVar:
            var = var_list[i]
            i += 1
            v = symbolMap.getObject(var)
            if not v.is_expression():
                if v.is_binary():
                    binary.append(var)
                elif v.is_integer():
                    if v.bounds == (0,1):
                        binary.append(var)
                    elif v.lb is not None and v.lb >= 0:
                        posInts.append(var)
                    else:
                        otherInts.append(var)
                elif v.lb == 0:
                    positive.append(var)
                else:
                    reals.append(var)
            else:
                body = str(v.expr)
                if linear:
                    if v.expr.polynomial_degree() not in linear_degree:
                        linear = False
                constraint_names.append('%s_expr' % var)
                ConstraintIO.write('%s.. %s =e= %s;\n' % (
                    constraint_names[-1], 
                    var,
                    body
                ))
                reals.append(var)
                # The Expression could have hit new variables (or other
                # expressions) -- so update the length of the var_list
                # so that we process / categorize these new symbols
                numVar = len(var_list)

        # Write the GAMS model
        output_file.write("EQUATIONS\n\t")
        output_file.write("\n\t".join(constraint_names))
        if binary:
            output_file.write(";\n\nBINARY VARIABLES\n\t")
            output_file.write("\n\t".join(binary))
        if posInts or otherInts:
            output_file.write(";\n\nINTEGER VARIABLES")
            if posInts:
                output_file.write("\n\t")
                output_file.write("\n\t".join(posInts))
            if otherInts:
                output_file.write("\n\t")
                output_file.write("\n\t".join(otherInts))
        if positive:
            output_file.write(";\n\nPOSITIVE VARIABLES\n\t")
            output_file.write("\n\t".join(positive))
        output_file.write(";\n\nVARIABLES\n\tGAMS_OBJECTIVE\n\t")
        output_file.write("\n\t".join(reals))
        output_file.write(";\n\n")
        output_file.write(ConstraintIO.getvalue())
        output_file.write("\n")
        for varName in var_list:
            var = symbolMap.getObject(varName)
            if var.is_expression():
                continue
            if varName in positive:
                if var.ub is not None:
                    output_file.write("%s.up = %s;\n" % (varName, var.ub))
            elif varName in posInts:
                if var.lb != 0:
                    output_file.write("%s.lo = %s;\n" % (varName, var.lb))
                if var.ub is not None:
                    output_file.write("%s.up = %s;\n" % (varName, var.ub))
            elif varName in otherInts:
                if var.lb is None:
                    # GAMS doesn't allow -INF lower bound for ints
                    # Set bound to lowest possible bound in GAMS
                    logger.warning("Lower bound for integer variable %s "
                                   "set to lowest possible in GAMS: -1.0E+10"
                                   % var.name)
                    output_file.write("%s.lo = -1.0E+10;\n" % (varName))
                if var.ub is not None:
                    output_file.write("%s.up = %s;\n" % (varName, var.ub))
            elif varName in reals:
                if var.lb is not None:
                    output_file.write("%s.lo = %s;\n" % (varName, var.lb))
                if var.ub is not None:
                    output_file.write("%s.up = %s;\n" % (varName, var.ub))
            if var.value is not None:
                output_file.write("%s.l = %s;\n" % (varName, var.value))
            if var.is_fixed():
                assert var.value is not None, "Cannot fix variable at None"
                output_file.write("%s.fx = %s;\n" % (varName, var.value))

        model_name = "GAMS_MODEL"
        output_file.write("\nMODEL %s /all/ ;\n" % model_name)

        if mtype is None:
            mtype =  ('lp','nlp','mip','minlp')[
                (0 if linear else 1) +
                (2 if (binary or posInts or otherInts) else 0)]

        if solver is not None:
            if mtype.upper() not in valid_solvers[solver.upper()]:
                raise ValueError("ProblemWriter_gams passed solver (%s) "
                                 "unsuitable for model type (%s)"
                                 % (solver, mtype))
            output_file.write("option %s=%s;\n" % (mtype, solver))

        if add_options is not None:
            output_file.write("\n* START USER ADDITIONAL OPTIONS\n")
            for line in add_options:
                output_file.write('\n' + line)
            output_file.write("\n\n* END USER ADDITIONAL OPTIONS\n\n")

        output_file.write(
            "SOLVE %s USING %s %simizing GAMS_OBJECTIVE;\n"
            % ( model_name, 
                mtype,
                'min' if obj.sense == minimize else 'max'))

        if put_results is not None:
            output_file.write("\nfile results /'%s'/;" % put_results)
            output_file.write("\nresults.nd=15;")
            output_file.write("\nresults.nw=21;")
            output_file.write("\nput results;")
            output_file.write("\nput 'SYMBOL  :  LEVEL  :  MARGINAL' /;")
            for var in var_list:
                output_file.write("\nput %s %s.l %s.m /;" % (var, var, var))
            for con in constraint_names:
                output_file.write("\nput %s %s.l %s.m /;" % (con, con, con))
            output_file.write("\nput GAMS_OBJECTIVE GAMS_OBJECTIVE.l "
                              "GAMS_OBJECTIVE.m;\n")


valid_solvers = {
'ALPHAECP': ['MINLP','MIQCP'],
'AMPL': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'],
'ANTIGONE': ['NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'BARON': ['LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'BDMLP': ['LP','MIP','RMIP'],
'BDMLPD': ['LP','RMIP'],
'BENCH': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'BONMIN': ['MINLP','MIQCP'],
'BONMINH': ['MINLP','MIQCP'],
'CBC': ['LP','MIP','RMIP'],
'COINBONMIN': ['MINLP','MIQCP'],
'COINCBC': ['LP','MIP','RMIP'],
'COINCOUENNE': ['NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'COINIPOPT': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'COINOS': ['LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'COINSCIP': ['MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'CONOPT': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'CONOPT3': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'CONOPT4': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'CONOPTD': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'CONVERT': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'CONVERTD': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'],
'COUENNE': ['NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'CPLEX': ['LP','MIP','RMIP','QCP','MIQCP','RMIQCP'],
'CPLEXD': ['LP','MIP','RMIP','QCP','MIQCP','RMIQCP'],
'CPOPTIMIZER': ['MIP','MINLP','MIQCP'],
'DE': ['EMP'],
'DECIS': ['EMP'],
'DECISC': ['LP'],
'DECISM': ['LP'],
'DICOPT': ['MINLP','MIQCP'],
'DICOPTD': ['MINLP','MIQCP'],
'EXAMINER': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'EXAMINER2': ['LP','MIP','RMIP','NLP','MCP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'GAMSCHK': ['LP','MIP','RMIP','NLP','MCP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'GLOMIQO': ['QCP','MIQCP','RMIQCP'],
'GUROBI': ['LP','MIP','RMIP','QCP','MIQCP','RMIQCP'],
'IPOPT': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'IPOPTH': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'JAMS': ['EMP'],
'KESTREL': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'],
'KNITRO': ['LP','RMIP','NLP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'LGO': ['LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'],
'LGOD': ['LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'],
'LINDO': ['LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP','EMP'],
'LINDOGLOBAL': ['LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'LINGO': ['LP','MIP','RMIP','NLP','DNLP','RMINLP','MINLP'],
'LOCALSOLVER': ['MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'LOGMIP': ['EMP'],
'LS': ['LP','RMIP'],
'MILES': ['MCP'],
'MILESE': ['MCP'],
'MINOS': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'MINOS5': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'MINOS55': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'MOSEK': ['LP','MIP','RMIP','NLP','DNLP','RMINLP','QCP','MIQCP','RMIQCP'],
'MPECDUMP': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'],
'MPSGE': [],
'MSNLP': ['NLP','DNLP','RMINLP','QCP','RMIQCP'],
'NLPEC': ['MCP','MPEC','RMPEC'],
'OS': ['LP','MIP','RMIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'OSICPLEX': ['LP','MIP','RMIP'],
'OSIGUROBI': ['LP','MIP','RMIP'],
'OSIMOSEK': ['LP','MIP','RMIP'],
'OSISOPLEX': ['LP','RMIP'],
'OSIXPRESS': ['LP','MIP','RMIP'],
'PATH': ['MCP','CNS'],
'PATHC': ['MCP','CNS'],
'PATHNLP': ['LP','RMIP','NLP','DNLP','RMINLP','QCP','RMIQCP'],
'PYOMO': ['LP','MIP','RMIP','NLP','MCP','MPEC','RMPEC','CNS','DNLP','RMINLP','MINLP'],
'QUADMINOS': ['LP'],
'SBB': ['MINLP','MIQCP'],
'SCENSOLVER': ['LP','MIP','RMIP','NLP','MCP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'SCIP': ['MIP','NLP','CNS','DNLP','RMINLP','MINLP','QCP','MIQCP','RMIQCP'],
'SNOPT': ['LP','RMIP','NLP','CNS','DNLP','RMINLP','QCP','RMIQCP'],
'SOPLEX': ['LP','RMIP'],
'XA': ['LP','MIP','RMIP'],
'XPRESS': ['LP','MIP','RMIP','QCP','MIQCP','RMIQCP']
}
