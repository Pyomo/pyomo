#! /usr/bin/env python
#
# pyro_mip_server: A script that sets up a Pyro server for solving MIPs in
#           a distributed manner.
#
#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import os.path
import sys
import traceback
import datetime
import pyutilib.services
import pyutilib.pyro
from pyomo.util import pyomo_command

try:
    import cPickle as pickle
except:
    import pickle

class PyomoMIPWorker(pyutilib.pyro.TaskWorker):

    def process(self, data):
        import pyomo.opt

        data = pyutilib.misc.Bunch(**data)

        pyutilib.services.TempfileManager.push()

        # construct the solver on this end, based on the input type stored in "data.opt".
        # this is slightly more complicated for asl-based solvers, whose real executable
        # name is stored in data.solver_options["solver"].
        if data.opt == "asl":
           solver_name = data.solver_options["solver"]
           opt = pyomo.opt.SolverFactory(solver_name)
        else:
           opt = pyomo.opt.SolverFactory(data.opt)
        if opt is None:
            raise ValueError("Problem constructing solver `"+data.opt+"'")

        opt.suffixes = data.suffixes

        # here is where we should set any options required by the solver, available
        # as specific attributes of the input data object.
        solver_options = data.solver_options
        del data.solver_options
        for key,value in solver_options.items():
            setattr(opt.options,key,value)

        problem_filename_suffix = os.path.split(data.filename)[1]
        temp_problem_filename = pyutilib.services.TempfileManager.create_tempfile(suffix="."+problem_filename_suffix)
        OUTPUT=open(temp_problem_filename,'w')
        OUTPUT.write(data.file)
        OUTPUT.close()

        if data.warmstart_file is not None:
            warmstart_filename_suffix = os.path.split(data.warmstart_filename)[1]
            temp_warmstart_filename = pyutilib.services.TempfileManager.create_tempfile(suffix="."+warmstart_filename_suffix)
            OUTPUT=open(temp_warmstart_filename,'w')
            OUTPUT.write(str(data.warmstart_file)+'\n')
            OUTPUT.close()
            opt.warm_start_solve = True
            opt.warm_start_file_name = temp_warmstart_filename

        now = datetime.datetime.now()
        print(str(now) + ": Applying solver="+data.opt+" to solve problem="+temp_problem_filename)
        sys.stdout.flush()
        results = opt.solve(temp_problem_filename, **data.kwds)

        # IMPT: The results object will *not* have a symbol map, as the symbol
        #       map is not pickle'able. The responsibility for translation will
        #       will have to be done on the client end.

        pyutilib.services.TempfileManager.pop()

        now = datetime.datetime.now()
        print(str(now) + ": Solve completed - number of solutions="+str(len(results.solution)))
        sys.stdout.flush()
#        results.write()
#        sys.stdout.flush()
        return pickle.dumps(results)


@pyomo_command('pyro_mip_server', "Launch a Pyro server for Pyomo MIP solvers")
def main():
    #
    # Handle error when pyro is not installed
    #
    if pyutilib.pyro.Pyro is None:
        raise ImportError("Pyro or Pyro4 is not available")

    #
    # Import plugins
    #
    import pyomo.environ
    #
    exception_trapped = False
    try:
        pyutilib.pyro.TaskWorkerServer(PyomoMIPWorker, argv=sys.argv)
    except IOError:
        msg = sys.exc_info()[1]
        print("IO ERROR:")
        print(msg)
        exception_trapped = True
    except pyutilib.common.ApplicationError:
        msg = sys.exc_info()[1]
        print("APPLICATION ERROR:")
        print(str(msg))
        exception_trapped = True
    except RuntimeError:
        msg = sys.exc_info()[1]
        print("RUN-TIME ERROR:")
        print(str(msg))
        exception_trapped = True
    # pyutilib.pyro tends to throw SystemExit exceptions if things
    # cannot be found or hooked up in the appropriate fashion. the
    # name is a bit odd, but we have other issues to worry about. we
    # are dumping the trace in case this does happen, so we can figure
    # out precisely who is at fault.
    except SystemExit:
        msg = sys.exc_info()[1]
        print("PH solver server encountered system error")
        print("Error: "+str(msg))
        print("Stack trace:")
        traceback.print_exc()
        exception_trapped = True
    except:
        print("Encountered unhandled exception")
        traceback.print_exc()
        exception_trapped = True

    # if an exception occurred, then we probably want to shut down all
    # Pyro components.  otherwise, the client may have forever while
    # waiting for results that will never arrive. there are better
    # ways to handle this at the client level, but until those are
    # implemented, this will suffice for cleanup.  NOTE: this should
    # perhaps be command-line driven, so it can be disabled if
    # desired.
    if exception_trapped == True:
        print("Pyro solver server aborted")
