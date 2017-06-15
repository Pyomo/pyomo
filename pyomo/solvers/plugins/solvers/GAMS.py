#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import print_function
from six import StringIO
from pyutilib.misc import PauseGC
from pyomo.core.base import (
    SymbolMap, AlphaNumericTextLabeler, NumericLabeler,
    Block, Constraint, Expression, Objective, Var, Set, RangeSet, Param,
    value, minimize, Suffix)
from pyomo.core.base.component import ComponentData
from gams import GamsWorkspace, DebugLevel
import os, sys

import pyomo.util.plugin
from pyomo.opt.base import IOptSolver
import pyutilib.services

from pyutilib.misc import Options


pyutilib.services.register_executable(name="gams")
 
class GAMSSolver(pyomo.util.plugin.Plugin):
    """A generic interface to GAMS solvers"""
    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('gams', doc='The GAMS modeling language')

    options = Options()

    def available(self, exception_flag=True):
        return pyutilib.services.registered_executable("gams") is not None
 
    def solve(self, model):
        """
        Solver for pyomo models using GAMS. Converts pyomo model
        to GAMS input file, solves in GAMS, then returns solution
        values back into pyomo model.

        Uses GAMS Python API. For installation help visit:
        https://www.gams.com/latest/docs/apis/examples_python/index.html

        self.options:
            output_filename=None:
                GAMS model file. If None, will use model name. Kept if given.
            symbolic_solver_labels=False:
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
            labeler=None:
                Custom labeler option. Incompatible with symbolic_solver_labels.
            solve=True:
                If False, GAMSSolver will only create and keep GAMS model file.
            solver=None:
                If None, GAMS will use default solver for model type.
            mtype=None:
                Model type. If None, will chose from lp, nlp, mip, and minlp.
            load_model=True:
                Load results back into pyomo model.
            print_result=False:
                Print summary of solution to stdout.
            show_log=False:
                Print GAMS log to stdout
            keep_files=False:
                Keep temporary files in current directory.
                Equivalent of DebugLevel.KeepFiles.
                If False, GAMS model file will be deleted as well.
                Summary of temp files can be found in _gams_py_gjo0.pf
        """

        # Make sure not to modify options dictionary
        opts = dict(self.options)

        # GAMS model file. If None, will use model name.
        output_filename = opts.pop("output_filename", None)

        # Use full Pyomo component names rather than
        # shortened symbols (slower, but useful for debugging).
        symbolic_solver_labels = opts.pop("symbolic_solver_labels", False)

        # Custom labeler option. Incompatible with symbolic_solver_labels.
        labeler = opts.pop("labeler", None)

        # If False, GAMSSolver will only create and keep GAMS model file.
        solve = opts.pop("solve", True)

        # If None, GAMS will use default solver for model type.
        solver = opts.pop("solver", None)

        # If None, will chose from lp, nlp, mip, and minlp.
        mtype = opts.pop("mtype", None)

        # After solving, store level and marginal
        # values in original model's variables
        load_model = opts.pop("load_model", True)

        # Print the results to sys.stdout
        print_result = opts.pop("print_result", False)

        # Print the GAMS log to sys.stdout
        show_log = opts.pop("show_log", False)

        # Keep tmp files: gms, lst, gdx, pf
        keep_files = opts.pop("keep_files", False)

        if len(opts):
            raise ValueError(
                "GAMSSolver passed unrecognized solve options:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(opts)))

        if solver is not None:
            if solver.upper() not in valid_solvers:
                raise ValueError(
                    "GAMSSolver passed unrecognized solver: %s" % solver)

        if mtype is not None:
            valid_mtypes = set([
                'lp', 'qcp', 'nlp', 'dnlp', 'rmip', 'mip', 'rmiqcp', 'rminlp',
                'miqcp', 'minlp', 'rmpec', 'mpec', 'mcp', 'cns', 'emp'])
            if mtype.lower() not in valid_mtypes:
                raise ValueError(
                    "GAMSSolver passed unrecognized model type: %s" % mtype)
            if (solver is not None and
                mtype.upper() not in valid_solvers[solver.upper()]):
                raise ValueError("GAMSSolver passed solver (%s) unsuitable for"
                                 " given model type (%s)" % (solver, mtype))

        # Flag to keep GAMS model file, implied to keep if output_filename given
        keep_output = True
        if output_filename is None:
            output_filename = AlphaNumericTextLabeler()(model) + ".gms"
            keep_output = False

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError("GAMSSolver: Using both the "
                             "'symbolic_solver_labels' and 'labeler' "
                             "options is forbidden")

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
            return symbolMap.getSymbol(obj, var_recorder)

        with PauseGC() as pgc:
            with open(output_filename, "w") as output_file:
                try:
                    ComponentData.labeler.append(var_label)
                    self._write_model(
                        model=model,
                        output_file=output_file,
                        var_list=var_list,
                        symbolMap=symbolMap,
                        con_labeler=con_labeler,
                        solver=solver,
                        mtype=mtype
                    )
                finally:
                    ComponentData.labeler.pop()
        if solve:
            self._solve_model(
                model=model,
                output_filename=output_filename,
                var_list=var_list,
                symbolMap=symbolMap,
                load_model=load_model,
                print_result=print_result,
                show_log=show_log,
                keep_files=keep_files,
                keep_output=keep_output
            )

        return None # should this return a results object?

    def _write_model(self,
                    model,
                    output_file,
                    var_list,
                    symbolMap,
                    con_labeler,
                    solver,
                    mtype):
        constraint_names = []
        ConstraintIO = StringIO()
        linear = True
        linear_degree = set([0,1])

        # Sanity check: all active components better be things we know
        # how to deal with, plus Suffix for later
        valid_ctypes = set([
            Block, Constraint, Expression, Objective, Param, 
            Set, RangeSet, Var, Suffix ])
        for c in model.component_objects(active=True):
            if c.type() not in valid_ctypes:
                raise RuntimeError(
                    "Unallowable component %s.\nThe GAMS writer cannot export"
                    " models with this component type" % ( c.type().__name__, ))

        # Temporarily initialize uninitialized variables in order to call
        # value() on each constraint to check domain violations
        uninit_vars = list()
        for var in model.component_data_objects(Var, active=True):
            if var.value is None:
                uninit_vars.append(var)
                var.value = 0

        # Walk through the model and generate the constraint definition
        # for all active constraints.  Any Vars / Expressions that are
        # encountered will be added to the var_list due to the labeler
        # defined above.
        for con in model.component_data_objects(Constraint, active=True):
            if con.body.is_fixed():
                continue

            # Ensure GAMS will not encounter domain violations in presolver
            # operations at current values, which are None (0) by default
            # Used to handle log and log10 violations, for example
            try:
                value(con.body)
            except:
                raise ValueError("GAMSSolver encountered an error while"
                                 " attemtping to evaluate\n            %s"
                                 " at initial variable values.\n            "
                                 "Ensure set variable values do not violate any"
                                 " domains (are you using log or log10?)"
                                 % con.name)

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

        obj = list(model.component_data_objects(Objective, active=True))
        if len(obj) != 1:
            raise RuntimeError(
                "GAMS writer requires exactly one active objective (found %s)"
                % (len(obj)))
        obj = obj[0]

        # Same domain violation check as above
        try:
            value(con.body)
        except:
            raise ValueError("GAMSSolver encountered an error while"
                             " attemtping to evaluate\n            %s"
                             " at initial variable values.\n            "
                             "Ensure set variable values do not violate any"
                             " domains (are you using log or log10?)"
                             % con.name)

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

        # Return uninitialized variables to None
        for var in uninit_vars:
            var.value = None

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
            else: # ADD LOG EXCEPTION HANDLING!!!!!!!!!!!!!!!!!!!
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
            output_file.write(";\n\nINTEGER VARIABLES\n\t")
            output_file.write("\n\t".join(posInts))
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
                    # Set bound to lowest possible bound in Gams
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

        model_name = symbolMap.getSymbol(model, con_labeler)
        output_file.write("\nMODEL %s /all/ ;\n" % model_name)

        if mtype is None:
            mtype =  ('lp','nlp','mip','minlp')[
                (0 if linear else 1) +
                (2 if (binary or posInts or otherInts) else 0)]

        if solver is not None:
            if mtype.upper() not in valid_solvers[solver.upper()]:
                raise ValueError("GAMSSolver passed solver (%s) unsuitable for"
                                 " model type (%s)" % (solver, mtype))
            output_file.write("option %s=%s;\n" % (mtype, solver))

        output_file.write(
            "SOLVE %s USING %s %simizing GAMS_OBJECTIVE;\n"
            % ( model_name, 
                mtype,
                'min' if obj.sense == minimize else 'max'))

    def _solve_model(self,
                    model,
                    output_filename,
                    var_list,
                    symbolMap,
                    load_model,
                    print_result,
                    show_log,
                    keep_files,
                    keep_output):

        def abt_eq(x, y, eps=1E-8):
            """Return if x and y within epsilon, used for values close to 0"""
            return abs(x - y) < eps

        # If keeping files, set dir to current dir
        # All tmp files will be created and kept there
        # Otherwise dir is in /tmp/ folder and deleted after
        if keep_files:
            ws = GamsWorkspace(working_directory=os.getcwd(),
                               debug=DebugLevel.KeepFiles)
        else:
            ws = GamsWorkspace()

        t1 = ws.add_job_from_file(os.path.join(os.getcwd(), output_filename))

        if show_log:
            t1.run(output=sys.stdout)
        else:
            t1.run()

        if not (keep_files or keep_output):
            os.remove(output_filename)

        has_dual = has_rc = False
        for suf in model.component_data_objects(Suffix, active=True):
            if (suf.name == 'dual' and suf.import_enabled()):
                has_dual = True
            elif (suf.name == 'rc' and suf.import_enabled()):
                has_rc = True

        if load_model or print_result:
            if print_result:
                print("\n==================================================="
                      "\n                 GAMSSolver Results                "
                      "\n===================================================\n")
            for var in var_list:
                v = symbolMap.getObject(var)
                rec = t1.out_db[var].first_record()
                if load_model:
                    if not v.is_expression():
                        if v.is_binary() or v.is_integer():
                            v.set_value(int(rec.level))
                        else:
                            v.set_value(rec.level)
                    if has_rc:
                        model.rc.set_value(v, rec.marginal)
                if print_result:
                    print(v.name + ": level=" + str(rec.level)
                          + " marginal=" + str(rec.marginal))

            obj = list(model.component_data_objects(Objective, active=True))
            obj = obj[0]
            oName = "GAMS_OBJECTIVE"
            rec = t1.out_db[oName].first_record()
            if load_model:
                obj.set_value(rec.level)
                if has_rc:
                    model.rc.set_value(obj, rec.marginal)
            if print_result:
                print(obj.name + ": level=" + str(rec.level)
                      + " marginal=" + str(rec.marginal))

            for c in model.component_data_objects(Constraint, active=True):
                if c.body.is_fixed():
                    continue
                con = symbolMap.getSymbol(c)
                if c.equality:
                    rec = t1.out_db[con].first_record()
                    if load_model and has_dual:
                        model.dual.set_value(c, rec.marginal)
                    if print_result:
                        print(c.name + ": level=" + str(rec.level)
                              + " marginal=" + str(rec.marginal))
                else:
                    # Inequality, assume if 2-sided that only
                    # one side's marginal is nonzero
                    marg = 0
                    if c.lower is not None:
                        rec_lo = t1.out_db[con + '_lo'].first_record()
                        if load_model and has_dual:
                            marg += rec_lo.marginal
                        if print_result:
                            print(c.name + "(lo): level=" + str(rec_lo.level)
                                  + " marginal=" + str(rec_lo.marginal))
                    if c.upper is not None:
                        rec_hi = t1.out_db[con + '_hi'].first_record()
                        if load_model and has_dual:
                            marg += rec_hi.marginal
                        if print_result:
                            print(c.name + "(hi): level=" + str(rec_hi.level)
                                  + " marginal=" + str(rec_hi.marginal))
                    if load_model and has_dual:
                        model.dual.set_value(c, marg)
                    # DEBUG only 1 side should be nonzero
                    if c.lower is not None and c.upper is not None:
                        rec_lo = t1.out_db[con + '_lo'].first_record()
                        rec_hi = t1.out_db[con + '_hi'].first_record()
                        if not abt_eq(rec_lo.marginal, 0):
                            assert abt_eq(rec_hi.marginal, 0), (
                                "2-sided constraint %s has 2 nonzero marginals"
                                % c.name)
            if print_result:
                print("\n===================================================\n")


solver_chart = """\
ALPHAECP    MINLP MIQCP
AMPL        LP MIP RMIP NLP MCP MPEC RMPEC CNS DNLP RMINLP MINLP
ANTIGONE    NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
BARON       LP MIP RMIP NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
BDMLP       LP MIP RMIP
BDMLPD      LP RMIP
BENCH       LP MIP RMIP NLP MCP MPEC RMPEC CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
BONMIN      MINLP MIQCP
BONMINH     MINLP MIQCP
CBC         LP MIP RMIP
COINBONMIN  MINLP MIQCP
COINCBC     LP MIP RMIP
COINCOUENNE NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
COINIPOPT   LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
COINOS      LP MIP RMIP NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
COINSCIP    MIP NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
CONOPT      LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
CONOPT3     LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
CONOPT4     LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
CONOPTD     LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
CONVERT     LP MIP RMIP NLP MCP MPEC RMPEC CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
CONVERTD    LP MIP RMIP NLP MCP MPEC RMPEC CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP EMP
COUENNE     NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
CPLEX       LP MIP RMIP QCP MIQCP RMIQCP
CPLEXD      LP MIP RMIP QCP MIQCP RMIQCP
CPOPTIMIZER MIP MINLP MIQCP
DE          EMP
DECIS       EMP
DECISC      LP
DECISM      LP
DICOPT      MINLP MIQCP
DICOPTD     MINLP MIQCP
EXAMINER    LP MIP RMIP NLP MCP MPEC RMPEC DNLP RMINLP MINLP QCP MIQCP RMIQCP
EXAMINER2   LP MIP RMIP NLP MCP DNLP RMINLP MINLP QCP MIQCP RMIQCP
GAMSCHK     LP MIP RMIP NLP MCP DNLP RMINLP MINLP QCP MIQCP RMIQCP
GLOMIQO     QCP MIQCP RMIQCP
GUROBI      LP MIP RMIP QCP MIQCP RMIQCP
IPOPT       LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
IPOPTH      LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
JAMS        EMP
KESTREL     LP MIP RMIP NLP MCP MPEC RMPEC CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP EMP
KNITRO      LP RMIP NLP MPEC RMPEC CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
LGO         LP RMIP NLP DNLP RMINLP QCP RMIQCP
LGOD        LP RMIP NLP DNLP RMINLP QCP RMIQCP
LINDO       LP MIP RMIP NLP DNLP RMINLP MINLP QCP MIQCP RMIQCP EMP
LINDOGLOBAL LP MIP RMIP NLP DNLP RMINLP MINLP QCP MIQCP RMIQCP
LINGO       LP MIP RMIP NLP DNLP RMINLP MINLP
LOCALSOLVER MIP NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
LOGMIP      EMP
LS          LP RMIP
MILES       MCP
MILESE      MCP
MINOS       LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
MINOS5      LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
MINOS55     LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
MOSEK       LP MIP RMIP NLP DNLP RMINLP QCP MIQCP RMIQCP
MPECDUMP    LP MIP RMIP NLP MCP MPEC RMPEC CNS DNLP RMINLP MINLP
MPSGE      
MSNLP       NLP DNLP RMINLP QCP RMIQCP
NLPEC       MCP MPEC RMPEC
OS          LP MIP RMIP NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
OSICPLEX    LP MIP RMIP
OSIGUROBI   LP MIP RMIP
OSIMOSEK    LP MIP RMIP
OSISOPLEX   LP RMIP
OSIXPRESS   LP MIP RMIP
PATH        MCP CNS
PATHC       MCP CNS
PATHNLP     LP RMIP NLP DNLP RMINLP QCP RMIQCP
PYOMO       LP MIP RMIP NLP MCP MPEC RMPEC CNS DNLP RMINLP MINLP
QUADMINOS   LP
SBB         MINLP MIQCP
SCENSOLVER  LP MIP RMIP NLP MCP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
SCIP        MIP NLP CNS DNLP RMINLP MINLP QCP MIQCP RMIQCP
SNOPT       LP RMIP NLP CNS DNLP RMINLP QCP RMIQCP
SOPLEX      LP RMIP
XA          LP MIP RMIP
XPRESS      LP MIP RMIP QCP MIQCP RMIQCP\
"""

valid_solvers = dict()
for line in solver_chart.splitlines():
    items = line.split()
    valid_solvers[items[0]] = items[1:]