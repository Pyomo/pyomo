#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['pyomo2lp', 'pyomo2nl', 'pyomo2osil', 'pyomo2dakota']

import os
import sys
import argparse

from pyutilib.misc import Options, Container

from pyomo.misc import pyomo_command
from pyomo.opt import ProblemFormat
import pyomo.core.scripting.util
from pyomo.core.base import Objective, Var, Constraint, active_components_data


_format = None

def convert(options=Options(), parser=None, model_format=None):
    global _format
    if not model_format is None:
        _format = model_format
    #
    # Import plugins
    #
    import pyomo.environ
    #
    if options.save_model is None:
        if _format == ProblemFormat.cpxlp:
            options.save_model = 'unknown.lp'
        else:
            options.save_model = 'unknown.'+str(_format)
    options.format = _format
    #
    data = Options(options=options)
    #
    if options.help_components:
        pyomo.core.scripting.util.print_components(data)
        return Container()
    #
    pyomo.core.scripting.util.setup_environment(data)
    #
    pyomo.core.scripting.util.apply_preprocessing(data, parser=parser)
    if data.error:
        return Container()
    #
    model_data = pyomo.core.scripting.util.create_model(data)
    #
    pyomo.core.scripting.util.finalize(data, model=model_data.model)
    #
    model_data.options = options
    return model_data


def convert_dakota(options=Options(), parser=None):
    #
    # Import plugins
    #
    import pyomo.environ

    model_file = os.path.basename(options.model_file)
    model_file_no_ext = os.path.splitext(model_file)[0]

    #
    # Set options for writing the .nl and related files
    #

    # By default replace .py with .nl
    if options.save_model is None:
       options.save_model = model_file_no_ext + '.nl'
    options.format = ProblemFormat.nl
    # Dakota requires .row/.col files
    options.symbolic_solver_labels = True

    #
    # Call the core converter
    #
    model_data = convert(options, parser)

    #
    # Generate Dakota input file fragments for the Vars, Objectives, Constraints
    #

    # TODO: the converted model doesn't expose the right symbol_map
    #       for only the vars active in the .nl

    model = model_data.instance

    # Easy way
    #print "VARIABLE:"
    #lines = open(options.save_model.replace('.nl','.col'),'r').readlines()
    #for varName in lines:
    #    varName = varName.strip()
    #    var = model_data.symbol_map.getObject(varName)
    #    print "'%s': %s" % (varName, var)
    #    #print var.pprint()

    # Hard way
    variables = 0
    var_descriptors = []
    var_lb = []
    var_ub = []
    var_initial = []
    tmpDict = model_data.symbol_map.getByObjectDictionary()
    for var in active_components_data(model, Var):
        if id(var) in tmpDict:
            variables += 1
            var_descriptors.append(var.cname(True))

            # apply user bound, domain bound, or infinite
            _lb, _ub = var.bounds
            if _lb is not None:
                var_lb.append(str(_lb))
            else:
                var_lb.append("-inf")

            if _ub is not None:
                var_ub.append(str(_ub))
            else:
                var_ub.append("inf")

            try:
                val = value(var)
            except:
                val = None
            var_initial.append(str(val))

    objectives = 0
    obj_descriptors = []
    for obj in active_components_data(model, Objective):
        objectives += 1
        obj_descriptors.append(obj.cname(True))

    constraints = 0
    cons_descriptors = []
    cons_lb = []
    cons_ub = []
    for con in active_components_data(model, Constraint):
        constraints += 1
        cons_descriptors.append(con.cname(True))
        if con.lower is not None:
            cons_lb.append(str(con.lower))
        else:
            cons_lb.append("-inf")
        if con.upper is not None:
            cons_ub.append(str(con.upper))
        else:
            cons_ub.append("inf")

    # Write the Dakota input file fragments

    dakfrag = open(model_file_no_ext + ".dak", 'w')

    dakfrag.write("#--- Dakota variables block ---#\n")
    dakfrag.write("variables\n")
    dakfrag.write("  continuous_design " + str(variables) + '\n')
    dakfrag.write("    descriptors\n")
    for vd in var_descriptors:
        dakfrag.write("      '%s'\n" % vd)
    dakfrag.write("    lower_bounds " + " ".join(var_lb) + '\n')
    dakfrag.write("    upper_bounds " + " ".join(var_ub) + '\n')
    dakfrag.write("    initial_point " + " ".join(var_initial) + '\n')

    dakfrag.write("#--- Dakota interface block ---#\n")
    dakfrag.write("interface\n")
    dakfrag.write("  algebraic_mappings = '" + options.save_model  + "'\n")

    dakfrag.write("#--- Dakota responses block ---#\n")
    dakfrag.write("responses\n")
    dakfrag.write("  objective_functions " + str(objectives) + '\n')

    if (constraints > 0):
        dakfrag.write("  nonlinear_inequality_constraints " + str(constraints) + '\n')
        dakfrag.write("    lower_bounds " + " ".join(cons_lb) + '\n')
        dakfrag.write("    upper_bounds " + " ".join(cons_ub) + '\n')

    dakfrag.write("    descriptors\n")
    for od in obj_descriptors:
        dakfrag.write("      '%s'\n" % od)
    if (constraints > 0):
        for cd in cons_descriptors:
            dakfrag.write("      '%s'\n" % cd)

    # TODO: detect whether gradient information available in model
    dakfrag.write("  analytic_gradients\n")
    dakfrag.write("  no_hessians\n")

    dakfrag.close()

    sys.stdout.write( "Dakota input fragment written to file '%s'\n" 
                      % (model_file_no_ext + ".dak",) )
    return model_data


def pyomo2lp(args=None):
    global _format
    import pyomo.core.plugins.drivers.convert
    parser = pyomo.core.plugins.drivers.convert.create_parser(cmd='pyomo2lp')
    _format = ProblemFormat.cpxlp
    return pyomo.core.scripting.util.run_command(command=convert, parser=parser, args=args, name='pyomo2lp')

@pyomo_command('pyomo2lp', "Convert a Pyomo model to a LP file")
def pyomo2lp_main(args=None):
    sys.exit(pyomo2lp(args).errorcode)

def pyomo2nl(args=None):
    global _format
    import pyomo.core.plugins.drivers.convert
    parser = pyomo.core.plugins.drivers.convert.create_parser(cmd='pyomo2nl')
    _format = ProblemFormat.nl
    return pyomo.core.scripting.util.run_command(command=convert, parser=parser, args=args, name='pyomo2nl')

@pyomo_command('pyomo2nl', "Convert a Pyomo model to a NL file")
def pyomo2nl_main(args=None):
    sys.exit(pyomo2nl(args).errorcode)

def pyomo2osil(args=None):
    global _format
    import pyomo.core.plugins.drivers.convert
    parser = pyomo.core.plugins.drivers.convert.create_parser(cmd='pyomo2osil')
    _format = ProblemFormat.osil
    return pyomo.core.scripting.util.run_command(command=convert, parser=parser, args=args, name='pyomo2osil')

@pyomo_command('pyomo2osil', "Convert a Pyomo model to a OSiL file")
def pyomo2osil_main(args=None):
    sys.exit(pyomo2osil(args).errorcode)

def pyomo2dakota(args=None):
    global _format
    import pyomo.core.plugins.drivers.convert
    parser = pyomo.core.plugins.drivers.convert.create_parser(cmd='pyomo2dakota')
    return pyomo.core.scripting.util.run_command(command=convert_dakota, parser=parser, args=args, name='pyomo2dakota')

@pyomo_command('pyomo2dakota', "Convert a Pyomo model to a Dakota file")
def pyomo2dakota_main(args=None):
    sys.exit(pyomo2dakota(args).errorcode)

