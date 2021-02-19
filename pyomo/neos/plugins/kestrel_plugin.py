#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import os
import re
import six
import sys

from pyomo.common.dependencies import attempt_import
from pyomo.opt import SolverFactory, SolverManagerFactory, OptSolver
from pyomo.opt.parallel.manager import ActionManagerError, ActionStatus
from pyomo.opt.parallel.async_solver import (
    AsynchronousSolverManager
)
from pyomo.core.base import Block
import pyomo.neos.kestrel

xmlrpc_client = attempt_import('six.moves.xmlrpc_client')[0]

logger = logging.getLogger('pyomo.neos')


def _neos_error(msg, results, current_message):
    error_re = re.compile('error', flags=re.I)
    warn_re = re.compile('warn', flags=re.I)

    logger.error("%s  NEOS log:\n%s" % ( msg, current_message, ),
                 exc_info=sys.exc_info())
    soln_data = results.data
    if six.PY3:
        soln_data = soln_data.decode('utf-8')
    for line in soln_data.splitlines():
        if error_re.search(line):
            logger.error(line)
        elif warn_re.search(line):
            logger.warn(line)


@SolverManagerFactory.register(
    'neos', doc="Asynchronously execute solvers on the NEOS server")
class SolverManager_NEOS(AsynchronousSolverManager):

    def clear(self):
        """
        Clear manager state
        """
        AsynchronousSolverManager.clear(self)
        self.kestrel = pyomo.neos.kestrel.kestrelAMPL()
        self._ah = {} # maps NEOS job numbers to their corresponding
                      # action handle.
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
        solver = kwds.pop('solver', kwds.pop('opt', None))
        if solver is None:
            raise ActionManagerError(
                "No solver passed to %s, use keyword option 'solver'"
                % (type(self).__name__) )
        if not isinstance(solver, six.string_types):
            solver_name = solver.name
            if solver_name == 'asl':
                solver_name = \
                    os.path.basename(solver.executable())
        else:
            solver_name = solver
            solver = None

        #
        # Handle ephemeral solvers options here. These
        # will override whatever is currently in the options
        # dictionary, but we will reset these options to
        # their original value at the end of this method.
        #
        user_solver_options = {}
        # make sure to transfer the options dict on the
        # solver plugin if the user does not use a string
        # to identify the neos solver. The ephemeral
        # options must also go after these.
        if solver is not None:
            user_solver_options.update(solver.options)
        _options = kwds.pop('options', {})
        if isinstance(_options, six.string_types):
            _options = OptSolver._options_string_to_dict(_options)
        user_solver_options.update(_options)
        user_solver_options.update(
            OptSolver._options_string_to_dict(kwds.pop('options_string', '')))

        # JDS: [5/13/17] The following is a HACK.  This timeout flag is
        # set by pyomo/scripting/util.py:apply_optimizer.  If we do not
        # remove it, it will get passed to the NEOS solver.  For solvers
        # like CPLEX 12.7.0, this will cause a fatal error as it is not
        # a known option.
        if user_solver_options.get('timelimit',0) is None:
            del user_solver_options['timelimit']

        opt = SolverFactory('_neos')
        opt._presolve(*args, **kwds)
        #
        # Map NEOS name, using lowercase convention in Pyomo
        #
        if len(self._solvers) == 0:
            for name in self.kestrel.solvers():
                if name.endswith('AMPL'):
                    self._solvers[ name[:-5].lower() ] = name[:-5]
        if solver_name not in self._solvers:
            raise ActionManagerError(
                "Solver '%s' is not recognized by NEOS. "
                "Solver names recognized:\n%s"
                % (solver_name, str(sorted(self._solvers.keys()))))
        #
        # Apply kestrel
        #
        # Set the kestrel_options environment
        #
        neos_sname = self._solvers[solver_name].lower()
        os.environ['kestrel_options'] = 'solver=%s' % self._solvers[solver_name]
        #
        # Set the <solver>_options environment
        #
        solver_options = {}
        for key in opt.options:
            solver_options[key]=opt.options[key]
        solver_options.update(user_solver_options)
        options = opt._get_options_string(solver_options)
        if not options == "":
            os.environ[neos_sname+'_options'] = options
        #
        # Generate an XML string using these two environment variables
        #
        xml = self.kestrel.formXML(opt._problem_files[0])
        (jobNumber, password) = self.kestrel.submit(xml)
        ah.job = jobNumber
        ah.password = password
        #
        # Cleanup
        #
        del os.environ['kestrel_options']
        try:
            del os.environ[neos_sname+"_options"]
        except:
            pass
        #
        # Store action handle, and return
        #
        self._ah[jobNumber] = ah
        self._neos_log[jobNumber] = (0, "")
        self._opt_data[jobNumber] = (opt,
                                     opt._smap_id,
                                     opt._load_solutions,
                                     opt._select_index,
                                     opt._default_variable_value)
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

            if status not in ("Running", "Waiting"):
                # the job is done.
                ah = self._ah[jobNumber]
                del self._ah[jobNumber]
                ah.status = ActionStatus.done

                (opt,
                 smap_id,
                 load_solutions,
                 select_index,
                 default_variable_value) = self._opt_data[jobNumber]
                del self._opt_data[jobNumber]

                args = self._args[jobNumber]
                del self._args[jobNumber]

                # retrieve the final results, which are in message/log format.
                results = self.kestrel.neos.getFinalResults(jobNumber, ah.password)

                (current_offset, current_message) = self._neos_log[jobNumber]
                with open(opt._log_file, 'w') as OUTPUT:
                    OUTPUT.write(current_message)
                with open(opt._soln_file, 'w') as OUTPUT:
                    if six.PY2:
                        OUTPUT.write(results.data)
                    else:
                        OUTPUT.write(results.data.decode('utf-8'))

                rc = None
                try:
                    solver_results = opt.process_output(rc)
                except:
                    _neos_error( "Error parsing NEOS solution file",
                                 results, current_message )
                    return ah

                solver_results._smap_id = smap_id
                self.results[ah.id] = solver_results

                if isinstance(args[0], Block):
                    _model = args[0]
                    if load_solutions:
                        try:
                            _model.solutions.load_from(
                                solver_results,
                                select=select_index,
                                default_variable_value=default_variable_value)
                        except:
                            _neos_error(
                                "Error loading NEOS solution into model",
                                results, current_message )
                        solver_results._smap_id = None
                        solver_results.solution.clear()
                    else:
                        solver_results._smap = _model.solutions.symbol_map[smap_id]
                        _model.solutions.delete_symbol_map(smap_id)

                return ah
            else:
                # The job is still running...
                #
                # Grab the partial messages from NEOS as you go, in case
                # you want to output on-the-fly. You will only get data
                # if the job was routed to the "short" priority queue.
                (current_offset, current_message) = self._neos_log[jobNumber]
                # TBD: blocking isn't the way to go, but non-blocking
                # was triggering some exception in kestrel.
                #
                # [5/13/17]: The blocking fetch will timeout in 2
                # minutes.  If NEOS doesn't produce intermediate results
                # by then we will need to catch (and eat) the exception
                try:
                    (message_fragment, new_offset) \
                        = self.kestrel.neos.getIntermediateResults(
                            jobNumber,
                            self._ah[jobNumber].password,
                            current_offset )
                    logger.info(message_fragment)
                    self._neos_log[jobNumber] = (
                        new_offset,
                        current_message + (
                            message_fragment.data if six.PY2
                            else (message_fragment.data).decode('utf-8') ) )
                except xmlrpc_client.ProtocolError:
                    # The command probably timed out
                    pass

        return None

