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

from six import StringIO

from pyutilib.misc import PauseGC

from pyomo.core.base import (
    SymbolMap, AlphaNumericTextLabeler, NumericLabeler, 
    Block, Constraint, Expression, Objective, Var, Set, RangeSet, Param,
    value, minimize
)
from pyomo.core.base.component import ComponentData
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter
import pyomo.util.plugin


class ProblemWriter_gams(AbstractProblemWriter):
    pyomo.util.plugin.alias('gams', 'Generate the corresponding GAMS file')

    def __init__(self):
        AbstractProblemWriter.__init__(self, ProblemFormat.gams)

    def __call__(self,
                 model,
                 output_filename,
                 solver_capability,
                 io_options):

        # Make sure not to modify the user's dictionary,
        # they may be reusing it outside of this call
        io_options = dict(io_options)

        # Skip writing constraints whose body section is
        # fixed (i.e., no variables)
        skip_trivial_constraints = \
            io_options.pop("skip_trivial_constraints", True)

        # Use full Pyomo component names in the LP file rather
        # than shortened symbols (slower, but useful for debugging).
        symbolic_solver_labels = \
            io_options.pop("symbolic_solver_labels", False)

        labeler = io_options.pop("labeler", None)

        # How much effort do we want to put into ensuring the
        # LP file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)

        if len(io_options):
            raise ValueError(
                "ProblemWriter_gams passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(io_options)))

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
            return symbolMap.getSymbol(obj, var_recorder)

        # when sorting, there are a non-trivial number of
        # temporary objects created. these all yield
        # non-circular references, so disable GC - the
        # overhead is non-trivial, and because references
        # are non-circular, everything will be collected
        # immediately anyway.
        with PauseGC() as pgc:
            with open(output_filename, "w") as output_file:
                try:
                    ComponentData.labeler.append(var_label)
                    self._print_model(
                        model=model,
                        output_file=output_file,
                        solver_capability=solver_capability,
                        var_list=var_list,
                        symbolMap=symbolMap,
                        labeler=con_labeler,
                        file_determinism=file_determinism,
                        skip_trivial_constraints=skip_trivial_constraints,
                    )
                finally:
                    ComponentData.labeler.pop()

        return output_filename, symbolMap

    def _print_model( self, model, output_file, solver_capability, var_list, 
                      symbolMap, labeler, file_determinism, 
                      skip_trivial_constraints ):
        constraint_names = []
        ConstraintIO = StringIO()
        linear = True
        linear_degree = set([0,1])

        # Sanity check: all active components better be things we know
        # how to deal with
        valid_ctypes = set([
            Block, Constraint, Expression, Objective, Param, 
            Set, RangeSet, Var, ])
        for c in model.component_objects(active=True):
            if c.type() not in valid_ctypes:
                raise RuntimeError(
                    "Unallowable component %s.\nThe GAMS writer cannot export"
                    " models with this Component type" % ( c.type().__name__, ))

        # Walk through the model and generate the constraint definition
        # for all active constraints.  Any Vars / Expressions that are
        # encountered will be added to the var_list due to the labeler
        # defined above.
        for con in model.component_data_objects(Constraint, active=True):
            if con.body.is_fixed():
                continue
            if linear:
                if con.body.polynomial_degree() not in linear_degree:
                    linear = False
            body = StringIO()
            con.body.to_string(body)
            cName=symbolMap.getSymbol(con, labeler)
            if con.equality:
                constraint_names.append('%s' % cName)
                ConstraintIO.write('%s.. %s =e= %s;\n' % (
                    constraint_names[-1], 
                    body.getvalue(),
                    value(con.upper),
                ))
            else:
                if con.lower is not None:
                    constraint_names.append('%s_lo' % cName)
                    ConstraintIO.write('%s.. %s =l= %s;\n' % (
                        constraint_names[-1], 
                        value(con.lower),
                        body.getvalue(),
                    ))
                if con.upper is not None:
                    constraint_names.append('%s_hi' % cName)
                    ConstraintIO.write('%s.. %s =l= %s;\n' % (
                        constraint_names[-1], 
                        body.getvalue(),
                        value(con.upper),
                    ))

        obj = list( model.component_data_objects(Objective, active=True) )
        if len(obj) != 1:
            raise RuntimeError(
                "GAMS writer requires exactly one active objective (found %s)"
                % (len(obj),))
        obj = obj[0]
        if linear:
            if obj.expr.polynomial_degree() not in linear_degree:
                linear = False
        oName = symbolMap.getSymbol(obj, labeler)
        body = StringIO()
        obj.expr.to_string(body)
        constraint_names.append('%s' % oName)
        ConstraintIO.write('%s.. GAMS_OBJECTIVE =e= %s;\n' % (
            oName,
            body.getvalue(),
        ))

        # Categorize the variables that we found
        binary = []
        ints = []
        positive = []
        reals = []
        i = 0
        numVar = len(var_list)
        while i < numVar:
            var = var_list[i]
            i += 1
            v = symbolMap.getObject(var)
            try:
                if v.is_binary():
                    binary.append(var)
                elif v.is_integer():
                    if v.bounds == (0,1):
                        binary.append(var)
                    else:
                        ints.append(var)
                elif v.lb == 0:
                    positive.append(var)
                else:
                    reals.append(var)
            except AttributeError:
                # This is an Expression object
                body = StringIO()
                v.to_string(body)
                if linear:
                    if v.expr.polynomial_degree() not in linear_degree:
                        linear = False
                constraint_names.append('%s_expr' % var)
                ConstraintIO.write('%s.. %s =e= %s;\n' % (
                    constraint_names[-1], 
                    var,
                    body.getvalue(),
                ))
                reals.append(var)
                # The Expression could have hit new variables (or other
                # expressions) -- so update the length of the var_list
                # so that we process / categorize these new symbols
                numVar = len(var_list)

        # Write the GAMS model
        output_file.write('EQUATIONS\n\t')
        output_file.write("\n\t".join(constraint_names))
        if binary:
            output_file.write(";\n\nBINARY VARIABLES\n\t")
            output_file.write("\n\t".join(binary))
        if ints:
            output_file.write(";\n\nINTEGER VARIABLES\n\t")
            output_file.write("\n\t".join(inst))
        if positive:
            output_file.write(";\n\nPOSITIVE VARIABLES\n\t")
            output_file.write("\n\t".join(positive))
        output_file.write(";\n\nVARIABLES\n\tGAMS_OBJECTIVE\n\t")
        output_file.write("\n\t".join(reals))
        output_file.write(";\n\n")
        output_file.write(ConstraintIO.getvalue())
        output_file.write("\n")
        for varName in positive:
            var = symbolMap.getObject(varName)
            if var.ub is not None:
                output_file.write("%s.up = %s;\n" % (varName, var.ub))
        for varName in ints + reals:
            var = symbolMap.getObject(varName)
            if var.lb is not None:
                output_file.write("%s.lo = %s;\n" % (varName, var.lb))
            if var.ub is not None:
                output_file.write("%s.up = %s;\n" % (varName, var.ub))

        model_name = symbolMap.getSymbol(model, labeler)
        output_file.write("\nMODEL %s /all/ ;\n"
                          % model_name )
        output_file.write(
            "SOLVE %s USING %s %simizing GAMS_OBJECTIVE;\n"
            % ( model_name, 
                ('lp','nlp','mip','minlp')[
                    (0 if linear else 1) + (2 if (binary or ints) else 0)],
                'min' if obj.sense == minimize else 'max'))
