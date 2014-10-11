
import os
import six

import pyomo.util.plugin
from pyomo.opt.parallel.manager import *
from pyomo.opt.parallel.async_solver import *
from pyomo.opt.results import SolverResults
from pyomo.opt.base import SolverFactory

import pyomo.neos.kestrel


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
        else:                           #pragma:nocover
            raise ActionManagerError("Undefined solver")
        if not isinstance(solver, six.string_types):
            solver = solver.name
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
        os.environ[self._solvers[solver].lower()+'_options'] = self._opt.solver_options
        xml = self.kestrel.formXML(self._opt._problem_files[0])
        (jobNumber, password) = self.kestrel.submit(xml)
        ah.job = jobNumber
        ah.password = password
        #
        # Store action handle, and return
        #
        self._ah[jobNumber] = ah
        self._neos_log[jobNumber] = (0, "")
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

            status = self.kestrel.neos.getJobStatus(jobNumber,self._ah[jobNumber].password)

            if not status in ("Running", "Waiting"):

                ah = self._ah[jobNumber]                
                
                # the job is done.
                self._ah[jobNumber] = None
                ah.status = ActionStatus.done
                
                # retrieve the final results, which are in message/log format.
                results = self.kestrel.neos.getFinalResults(jobNumber, ah.password)

                (current_offset, current_message) = self._neos_log[jobNumber]
                OUTPUT=open(self._opt.log_file,'w')
                six.print_(current_message, file=OUTPUT)
                OUTPUT.close()

                #print("HERE")
                #print(current_message)
                #print(results.data)
                #print(self._opt.soln_file)
                #print("HERE")
                OUTPUT=open(self._opt.soln_file,'w')
                six.print_(results.data, file=OUTPUT)
                OUTPUT.close()

                rc = None
                solver_results = self._opt.process_output(rc)
                solver_results._symbol_map = self._opt._symbol_map
                self.results[ah.id] = solver_results
                return ah
            else:
                # grab the partial messages from NEOS as you go, in case you want
                # to output on-the-fly. we don't currently do this, but the infrastructure
                # is in place.
                (current_offset, current_message) = self._neos_log[jobNumber]
                # TBD: blocking isn't the way to go, but non-blocking was triggering some exception in kestrel.
                (message_fragment, new_offset) = self.kestrel.neos.getIntermediateResults(jobNumber, self._ah[jobNumber].password, current_offset)
                six.print_(message_fragment, end="")
                self._neos_log[jobNumber] = (new_offset, str(current_message) + str(message_fragment.data))

        return None

