#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['pyomo2lp', 'pyomo2nl', 'pyomo2dakota']

import os
import sys

from pyutilib.misc import Options, Container

from pyomo.util import pyomo_command
from pyomo.opt import ProblemFormat
from pyomo.core.base import (Objective,
                             Var,
                             Constraint,
                             value,
                             ConcreteModel)
import pyomo.scripting.util

_format = None


def convert(options=Options(), parser=None, model_format=None):
    global _format
    if not model_format is None:
        _format = model_format
    #
    # Import plugins
    #
    import pyomo.environ

    if options.model.save_file is None:
        if _format == ProblemFormat.cpxlp:
            options.model.save_file = 'unknown.lp'
        else:
            options.model.save_file = 'unknown.'+str(_format)
    options.model.save_format = _format

    data = Options(options=options)

    model_data = None
    try:
        pyomo.scripting.util.setup_environment(data)

        pyomo.scripting.util.apply_preprocessing(data, parser=parser)

        if data.error:
            return Container()

        model_data = pyomo.scripting.util.create_model(data)

        model_data.options = options
    except:

        # TBD: I should be able to call this function in the case of
        #      an exception to perform cleanup. However, as it stands
        #      calling finalize with its default keyword value for
        #      model(=None) results in an a different error related to
        #      task port values.  Not sure how to interpret that.
        pyomo.scripting.util.finalize(data,
                                      model=ConcreteModel(),
                                      instance=None,
                                      results=None)
        raise

    else:

        pyomo.scripting.util.finalize(data, model=model_data.model)

    return model_data

def convert_dakota(options=Options(), parser=None):
    #
    # Import plugins
    #
    import pyomo.environ

    model_file = os.path.basename(options.model.save_file)
    model_file_no_ext = os.path.splitext(model_file)[0]

    #
    # Set options for writing the .nl and related files
    #

    # By default replace .py with .nl
    if options.model.save_file is None:
       options.model.save_file = model_file_no_ext + '.nl'
    options.model.save_format = ProblemFormat.nl
    # Dakota requires .row/.col files
    options.model.symbolic_solver_labels = True

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
    for var in model.component_data_objects(Var, active=True):
        if id(var) in tmpDict:
            variables += 1
            var_descriptors.append(var.name)

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
    for obj in model.component_data_objects(Objective, active=True):
        objectives += 1
        obj_descriptors.append(obj.name)

    constraints = 0
    cons_descriptors = []
    cons_lb = []
    cons_ub = []
    for con in model.component_data_objects(Constraint, active=True):
        constraints += 1
        cons_descriptors.append(con.name)
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
    dakfrag.write("  algebraic_mappings = '" + options.model.save_file  + "'\n")

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
    from pyomo.scripting.pyomo_main import main
    if args is None:
        return main()
    else:
        return main(['convert', '--format=lp']+args, get_return=True)

def pyomo2nl(args=None):
    from pyomo.scripting.pyomo_main import main
    if args is None:
        return main()
    else:
        return main(['convert', '--format=nl']+args, get_return=True)

def pyomo2bar(args=None):
    from pyomo.scripting.pyomo_main import main
    if args is None:
        return main()
    else:
        return main(['convert', '--format=bar']+args, get_return=True)

def pyomo2dakota(args=None):
    from pyomo.scripting.pyomo_main import main
    if args is None:
        return main()
    else:
        return main(['convert','--format=dakota']+args, get_return=True)
