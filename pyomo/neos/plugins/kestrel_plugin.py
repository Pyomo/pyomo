#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os

import pyomo.util.plugin
from pyomo.opt.parallel.manager import *
from pyomo.opt.parallel.async_solver import *
from pyomo.opt.base import SolverFactory, OptSolver
from pyomo.core.base import Block
import pyomo.neos.kestrel

import six


class SolverManager_NEOS(AsynchronousSolverManager):

    pyomo.util.plugin.alias('neos', doc="Asynchronously execute solvers on the NEOS server")

    def clear(self):
        """
        Clear manager state
        """
        AsynchronousSolverManager.clear(self)
        self.kestrel = pyomo.neos.kestrel.kestrelAMPL()
        self._opt = None
        self._ah = {} # maps NEOS job numbers to their corresponding action handle.
        self._args = {}
        self._opt_data = {}

        # to grab streamed output from NEOS, need to keep
        # map of action handle to the to-date string of
        # extracted output.
        # TBD: The following entries aren't currently cleaned up, but
        #      we're still trying to get the basics down.
        # store pairs of NEOS message offset and NEOS message string.
        # index into the map is the NEOS job number
        self._neos_log = {} 
        self._solvers = {}

    def _perform_queue(self, ah, *args, **kwds):
        """
        Perform the queue operation.  This method returns the ActionHandle,
        and the ActionHandle status indicates whether the queue was successful.
        """
        if 'opt' in kwds:
            solver = kwds['opt']
            del kwds['opt']
        elif 'solver' in kwds:
            solver = kwds['solver']
            del kwds['solver']
        else:                           #pragma:nocover
            raise ActionManagerError("Undefined solver")
        if not isinstance(solver, six.string_types):
            solver = solver.name

        #
        # Handle ephemeral solvers options here. These
        # will override whatever is currently in the options
        # dictionary, but we will reset these options to
        # their original value at the end of this method.
        #
        ephemeral_solver_options = {}
        ephemeral_solver_options.update(kwds.pop('options', {}))
        ephemeral_solver_options.update(
            OptSolver._options_string_to_dict(kwds.pop('options_string', '')))

        self._opt = SolverFactory('_neos')
        self._opt._presolve(*args, **kwds)
        #
        # Map NEOS name, using lowercase convention in Pyomo
        #
        if len(self._solvers) == 0:
            for name in self.kestrel.solvers():
                if name.endswith('AMPL'):
                    self._solvers[ name[:-5].lower() ] = name[:-5]
        if not solver in self._solvers:
            raise ActionManagerError("Solver '%s' is not recognized by NEOS" % solver)
        #
        # Apply kestrel
        #
        os.environ['kestrel_options'] = 'solver=%s' % self._solvers[solver]
        solver_options = {}
        for key in self._opt.options:
            solver_options[key]=self._opt.options[key]
        solver_options.update(ephemeral_solver_options)

        options = self._opt._get_options_string(solver_options)
        if not options == "":
            os.environ[self._solvers[solver].lower()+'_options'] = self._opt._get_options_string()
        xml = self.kestrel.formXML(self._opt._problem_files[0])
        (jobNumber, password) = self.kestrel.submit(xml)
        ah.job = jobNumber
        ah.password = password
        #
        # Store action handle, and return
        #
        self._ah[jobNumber] = ah
        self._neos_log[jobNumber] = (0, "")
        self._opt_data[jobNumber] = self._opt._smap_id
        self._args[jobNumber] = args
        return ah

    def _perform_wait_any(self):
        """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method again.
        Note that an ActionHandle can be returned with a dummy value,
        to indicate an error.
        """
        for jobNumber in self._ah:

            status = self.kestrel.neos.getJobStatus(jobNumber,
                                                    self._ah[jobNumber].password)

            if not status in ("Running", "Waiting"):

                # the job is done.
                ah = self._ah[jobNumber]                
                del self._ah[jobNumber]
                ah.status = ActionStatus.done
                
                smap_id = self._opt_data[jobNumber]
                del self._opt_data[jobNumber]

                args = self._args[jobNumber]
                del self._args[jobNumber]

                # retrieve the final results, which are in message/log format.
                results = self.kestrel.neos.getFinalResults(jobNumber, ah.password)

                (current_offset, current_message) = self._neos_log[jobNumber]
                OUTPUT=open(self._opt._log_file, 'w')
                six.print_(current_message, file=OUTPUT)
                OUTPUT.close()
                OUTPUT=open(self._opt._soln_file, 'w')
                six.print_(results.data, file=OUTPUT)
                OUTPUT.close()

                rc = None
                solver_results = self._opt.process_output(rc)
                #solver_results._symbol_map = self._opt._symbol_map
                solver_results._smap_id = smap_id
                self.results[ah.id] = solver_results

                if isinstance(args[0], Block):
                    _model = args[0]
                    _model.solutions.load_from(solver_results)
                    solver_results._smap_id = None
                    solver_results.solution.clear()

                return ah
            else:
                # Grab the partial messages from NEOS as you go, in case you want
                # to output on-the-fly. We don't currently do this, but the infrastructure
                # is in place.
                (current_offset, current_message) = self._neos_log[jobNumber]
                # TBD: blocking isn't the way to go, but non-blocking was triggering some exception in kestrel.
                (message_fragment, new_offset) = \
                    self.kestrel.neos.getIntermediateResults(jobNumber,
                                                             self._ah[jobNumber].password,
                                                             current_offset)
                six.print_(message_fragment, end="")
                self._neos_log[jobNumber] = (new_offset, str(current_message) + str(message_fragment.data))

        return None

