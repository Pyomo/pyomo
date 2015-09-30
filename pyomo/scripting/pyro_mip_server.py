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
import time
import sys
import traceback
import datetime
import base64
from optparse import OptionParser
try:
    import cPickle as pickle
except:
    import pickle

import pyutilib.services
import pyutilib.pyro
from pyutilib.pyro import using_pyro4
import pyutilib.common
from pyomo.util import pyomo_command
from pyomo.opt.base import SolverFactory, ConverterError

import six

class PyomoMIPWorker(pyutilib.pyro.TaskWorker):

    def __init__(self, *args, **kwds):
        super(PyomoMIPWorker, self).__init__(*args, **kwds)

    def process(self, data):
        self._worker_task_return_queue = self._current_task_client
        data = pyutilib.misc.Bunch(**data)

        if hasattr(data, 'action') and \
           data.action == 'Pyomo_pyro_mip_server_shutdown':
            print("Received shutdown request")
            self._worker_shutdown = True
            return

        time_start = time.time()
        with pyutilib.services.TempfileManager.push():
            #
            # Construct the solver on this end, based on the input
            # type stored in "data.opt".  This is slightly more
            # complicated for asl-based solvers, whose real executable
            # name is stored in data.solver_options["solver"].
            #
            with SolverFactory(data.opt) as opt:

                if opt is None:
                    self._worker_error = True
                    return TaskProcessingError("Problem constructing solver `"
                                               +data.opt+"'")

                # here is where we should set any options required by
                # the solver, available as specific attributes of the
                # input data object.
                solver_options = data.solver_options
                del data.solver_options
                for key,value in solver_options.items():
                    setattr(opt.options,key,value)

                problem_filename_suffix = os.path.split(data.filename)[1]
                temp_problem_filename = \
                    pyutilib.services.TempfileManager.\
                    create_tempfile(suffix="."+problem_filename_suffix)

                with open(temp_problem_filename, 'w') as f:
                    f.write(data.file)

                if data.warmstart_filename is not None:
                    warmstart_filename_suffix = \
                        os.path.split(data.warmstart_filename)[1]
                    temp_warmstart_filename = \
                        pyutilib.services.TempfileManager.\
                        create_tempfile(suffix="."+warmstart_filename_suffix)
                    with open(temp_warmstart_filename, 'w') as f:
                        f.write(data.warmstart_file)
                    assert opt.warm_start_capable()
                    assert (('warmstart' in data.kwds) and \
                            data.kwds['warmstart'])
                    data.kwds['warmstart_file'] = temp_warmstart_filename

                now = datetime.datetime.now()
                if self._verbose:
                    print(str(now) + ": Applying solver="+data.opt
                          +" to solve problem="+temp_problem_filename)
                    sys.stdout.flush()
                results = opt.solve(temp_problem_filename,
                                    **data.kwds)
                assert results._smap_id is None
                # NOTE: This results object contains solutions,
                # because no model is provided (just a model file).
                # Also, the results._smap_id value is None.

        results.pyomo_solve_time = time.time()-time_start

        now = datetime.datetime.now()
        if self._verbose:
            print(str(now) + ": Solve completed - number of solutions="
                  +str(len(results.solution)))
            sys.stdout.flush()

        # PYTHON3 / PYRO4 Fix
        # The default serializer in Pyro4 is not pickle and does not
        # support user defined types (e.g., the results object).
        # Therefore, we pickle the results object before sending it
        # over the wire so the user does not need to change the Pyro
        # serializer.
        results = pickle.dumps(results, protocol=pickle.HIGHEST_PROTOCOL)

        if using_pyro4:
            #
            # The standard bytes object returned by pickle.dumps must be
            # converted to base64 to avoid errors sending over the
            # wire with Pyro4. Also, the base64 bytes must be wrapped
            # in a str object to avoid a different set of Pyro4 errors
            # related to its default serializer (Serpent)
            if six.PY3:
                results = str(base64.encodebytes(results))
            else:
                results = base64.encodestring(results)

        return results

@pyomo_command('pyro_mip_server', "Launch a Pyro server for Pyomo MIP solvers")
def main():
    #
    # Handle error when pyro is not installed
    #
    if pyutilib.pyro.Pyro is None:
        raise ImportError("Pyro or Pyro4 is not available")

    parser = OptionParser()
    parser.add_option(
        "--verbose", dest="verbose",
        help="Activate verbose output.",
        action="store_true", default=False)
    parser.add_option(
        "--pyro-host", dest="pyro_host",
        help="Hostname that the nameserver is bound on",
        default=None)
    parser.add_option(
        "--pyro-port", dest="pyro_port",
        help="Port that the nameserver is bound on",
        type="int",
        default=None)
    parser.add_option(
        "--request-timeout",
        dest="request_timeout",
        help=("The timeout to use when requesting tasks from "
              "the dispatcher. Default is None, implying the "
              "call will block indefinitely until a task is "
              "received."),
        default=None)
    parser.add_option(
        "--traceback",
        dest="traceback",
        help=("When an exception is thrown, show the entire "
              "call stack."),
        action="store_true",
        default=False)

    options, args = parser.parse_args()
    # Handle the old syntax which was purly argument driven
    # e.g., <host>
    verbose = False
    if len(args) == 1:
        host=sys.argv[1]
        if host == "None":
            host=None
        print("DEPRECATION WARNING: pyro_mip_server is now option "
              "driven (see pyro_mip_server --help)")
    else:
        host = options.pyro_host

    kwds = {}
    kwds['host'] = host
    kwds['port'] = options.pyro_port
    kwds['verbose'] = options.verbose
    kwds['timeout'] = options.request_timeout
    kwds['block'] = True

    #
    # Import plugins
    #
    import pyomo.environ
    #

    if options.traceback:
        pyutilib.pyro.TaskWorkerServer(PyomoMIPWorker,
                                       **kwds)
    else:
        try:
            try:
                pyutilib.pyro.TaskWorkerServer(PyomoMIPWorker,
                                               **kwds)
            except ValueError:
                sys.stderr.write("VALUE ERROR:\n")
                sys.stderr.write(str(sys.exc_info()[1])+"\n")
                raise
            except KeyError:
                sys.stderr.write("KEY ERROR:\n")
                sys.stderr.write(str(sys.exc_info()[1])+"\n")
                raise
            except TypeError:
                sys.stderr.write("TYPE ERROR:\n")
                sys.stderr.write(str(sys.exc_info()[1])+"\n")
                raise
            except NameError:
                sys.stderr.write("NAME ERROR:\n")
                sys.stderr.write(str(sys.exc_info()[1])+"\n")
                raise
            except IOError:
                sys.stderr.write("IO ERROR:\n")
                sys.stderr.write(str(sys.exc_info()[1])+"\n")
                raise
            except ConverterError:
                sys.stderr.write("CONVERTER ERROR:\n")
                sys.stderr.write(str(sys.exc_info()[1])+"\n")
                raise
            except pyutilib.common.ApplicationError:
                sys.stderr.write("APPLICATION ERROR:\n")
                sys.stderr.write(str(sys.exc_info()[1])+"\n")
                raise
            except RuntimeError:
                sys.stderr.write("RUN-TIME ERROR:\n")
                sys.stderr.write(str(sys.exc_info()[1])+"\n")
                raise
            except:
                sys.stderr.write("Encountered unhandled exception:\n")
                if len(sys.exc_info()) > 1:
                    sys.stderr.write(str(sys.exc_info()[1])+"\n")
                else:
                    traceback.print_exc(file=sys.stderr)
                raise
        except:
            print("Pyro solver server aborted")
            raise
