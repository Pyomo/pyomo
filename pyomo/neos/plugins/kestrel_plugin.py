#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import six

import pyomo.util.plugin
from pyomo.opt.parallel.manager import *
from pyomo.opt.parallel.async_solver import *
from pyomo.opt.base import SolverFactory, OptSolver
from pyomo.core.base import Block
import pyomo.neos.kestrel



class SolverManager_NEOS(AsynchronousSolverManager):

    pyomo.util.plugin.alias('neos', doc="Asynchronously execute solvers on the NEOS server")

    def clear(self):
        """
        Clear manager state
        """
        AsynchronousSolverManager.clear(self)
        self.kestrel = pyomo.neos.kestrel.kestrelAMPL()
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
        solver = kwds.pop('solver', kwds.pop('opt', None))
        if solver is None:
            raise ActionManagerError(
                "No solver passed to %s, use keyword option 'solver'"
                % (type(self).__name__) )
        if not isinstance(solver, six.string_types):
            solver_name = solver.name
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
        user_solver_options.update(
            kwds.pop('options', {}))
        user_solver_options.update(
            OptSolver._options_string_to_dict(kwds.pop('options_string', '')))

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
        os.environ['kestrel_options'] = 'solver=%s' % self._solvers[solver_name]
        solver_options = {}
        for key in opt.options:
            solver_options[key]=opt.options[key]
        solver_options.update(user_solver_options)

        options = opt._get_options_string(solver_options)
        # GH: Should we really be modifying the environment
        #     for this manager (knowing that we are not
        #     executing locally)
        if not options == "":
            os.environ[self._solvers[solver_name].lower()+'_options'] = \
                opt._get_options_string()
        xml = self.kestrel.formXML(opt._problem_files[0])
        (jobNumber, password) = self.kestrel.submit(xml)
        ah.job = jobNumber
        ah.password = password
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

            if not status in ("Running", "Waiting"):

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
                    six.print_(current_message, file=OUTPUT)
                with open(opt._soln_file, 'w') as OUTPUT:
                    if six.PY2:
                        six.print_(results.data, file=OUTPUT)
                    else:
                        six.print_((results.data).decode('utf-8'), file=OUTPUT)

                rc = None
                solver_results = opt.process_output(rc)
                solver_results._smap_id = smap_id
                self.results[ah.id] = solver_results
                opt.deactivate()

                if isinstance(args[0], Block):
                    _model = args[0]
                    if load_solutions:
                        _model.solutions.load_from(
                            solver_results,
                            select=select_index,
                            default_variable_value=default_variable_value)
                        solver_results._smap_id = None
                        solver_results.solution.clear()
                    else:
                        solver_results._smap = _model.solutions.symbol_map[smap_id]
                        _model.solutions.delete_symbol_map(smap_id)

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

