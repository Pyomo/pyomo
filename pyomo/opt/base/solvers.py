#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('OptSolver',
           'SolverFactory',
           'UnknownSolver',
           'check_available_solvers')

import re
import os
import sys
import time
import logging

from pyutilib.misc.config import ConfigBlock, ConfigList, ConfigValue
from pyomo.common import Factory
import pyutilib.common
import pyutilib.misc
import pyutilib.services

from pyomo.opt.base.problem import ProblemConfigFactory
from pyomo.opt.base.convert import convert_problem
from pyomo.opt.base.formats import ResultsFormat, ProblemFormat
import pyomo.opt.base.results

import six
from six import iteritems
from six.moves import xrange

logger = logging.getLogger('pyomo.opt')

# The version string is first searched for trunk/Trunk, and if
# found a tuple of infinities is returned. Otherwise, the first
# match of number[.number] where [.number] can repeat 1-3 times
# is used, which is translated into a tuple of size matching
# the keyword length (appending 0's when necessary). If no match
# is found None is returned (although one could argue a tuple of
# 0's might be appropriated).
def _extract_version(x, length=4):
    """
    Attempts to extract solver version information from a string.
    """
    assert (1 <= length) and (length <= 4)
    m = re.search('[t,T]runk',x)
    if m is not None:
        # Since most version checks are comparing if the current
        # version is greater/less than some other version, it makes
        # since that a solver advertising trunk should always be greater
        # than a version check, hence returning a tuple of infinities
        return tuple(float('inf') for i in xrange(length))
    m = re.search('[0-9]+(\.[0-9]+){1,3}',x)
    if not m is None:
        version = tuple(int(i) for i in m.group(0).split('.')[:length])
        while(len(version) < length):
            version += (0,)
        return version
    return None #(0,0,0,0)[:length]


class UnknownSolver(object):

    def __init__(self, *args, **kwds):
        #super(UnknownSolver,self).__init__(**kwds)

        #
        # The 'type' is the class type of the solver instance
        #
        if "type" in kwds:
            self.type = kwds["type"]
        else:  #pragma:nocover
            raise ValueError(
                "Expected option 'type' for UnknownSolver constructor")

        self.options = {}
        self._args = args
        self._kwds = kwds

    #
    # Support "with" statements. Forgetting to call deactivate
    # on Plugins is a common source of memory leaks
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Determine if this optimizer is available."""
        if exception_flag:
            from pyutilib.common import ApplicationError
            raise pyutilib.common.ApplicationError("Solver (%s) not available" % str(self.name))
        return False

    def warm_start_capable(self):
        """ True is the solver can accept a warm-start solution."""
        return False

    def solve(self, *args, **kwds):
        """Perform optimization and return an SolverResults object."""
        self._solver_error('solve')

    def reset(self):
        """Reset the state of an optimizer"""
        self._solver_error('reset')

    def set_options(self, istr):
        """Set the options in the optimizer from a string."""
        self._solver_error('set_options')

    def __bool__(self):
        return self.available()

    def __getattr__(self, attr):
        self._solver_error(attr)

    def _solver_error(self, method_name):
        raise RuntimeError("""Attempting to use an unavailable solver.

The SolverFactory was unable to create the solver "%s"
and returned an UnknownSolver object.  This error is raised at the point
where the UnknownSolver object was used as if it were valid (by calling
method "%s").

The original solver was created with the following parameters:
\t""" % ( self.type, method_name )
+ "\n\t".join("%s: %s" % i for i in sorted(self._kwds.items()))
+ "\n\t_args: %s" % ( self._args, )
+ "\n\toptions: %s" % ( self.options, ) )


class SolverFactoryClass(Factory):

    def __call__(self, _name=None, **kwds):
        if _name is None:
            return self
        _name=str(_name)
        if ':' in _name:
            _name, subsolver = _name.split(':',1)
            kwds['solver'] = subsolver
        elif 'solver' in kwds:
            subsolver = kwds['solver']
        else:
            subsolver = None
        opt = None
        try:
            if _name in self._cls:
                opt = self._cls[_name](**kwds)
            else:
                mode = kwds.get('solver_io', 'nl')
                if mode is None:
                    mode = 'nl'
                _implicit_solvers = {'nl': 'asl' }
                if "executable" not in kwds:
                    kwds["executable"] = _name
                if mode in _implicit_solvers:
                    if _implicit_solvers[mode] not in self._cls:
                        raise RuntimeError(
                            "  The solver plugin was not registered.\n"
                            "  Please confirm that the 'pyomo.environ' package has been imported.")
                    opt = self._cls[_implicit_solvers[mode]](**kwds)
                    if opt is not None:
                        opt.set_options('solver='+_name)
        except:
            err = sys.exc_info()[1]
            logger.warning("Failed to create solver with name '%s':\n%s"
                         % (_name, err))
            opt = None
        if opt is not None and _name != "py" and subsolver is not None:
            # py just creates instance of its subsolver, no need for this option
            opt.set_options('solver='+subsolver)
        if opt is None:
            opt = UnknownSolver( type=_name, **kwds )
            opt.name = _name
        return opt

SolverFactory = SolverFactoryClass('solver type')

#
# TODO: It is impossible to load CBC with NL file-io using this function,
#       i.e., SolverFactory("cbc", solver_io='nl'),
#       this is NOT asl:cbc (same with PICO)
# WEH:  Why is there a distinction between SolverFactory('asl:cbc') and SolverFactory('cbc', solver_io='nl')???   This is bad.
#
def check_available_solvers(*args):
    from pyomo.solvers.plugins.solvers.GUROBI import GUROBISHELL
    from pyomo.solvers.plugins.solvers.BARON import BARONSHELL

    logging.disable(logging.WARNING)

    ans = []
    for arg in args:
        if not isinstance(arg,tuple):
            name = arg
            arg = (arg,)
        else:
            name = arg[0]
        opt = SolverFactory(*arg)
        if opt is None or isinstance(opt, UnknownSolver):
            available = False
        elif (arg[0] == "gurobi") and \
           (not GUROBISHELL.license_is_valid()):
            available = False
        elif (arg[0] == "baron") and \
           (not BARONSHELL.license_is_valid()):
            available = False
        else:
            available = \
                (opt.available(exception_flag=False)) and \
                ((not hasattr(opt,'executable')) or \
                (opt.executable() is not None))
        if available:
            ans.append(name)

    logging.disable(logging.NOTSET)

    return ans

def _raise_ephemeral_error(name, keyword=""):
    raise AttributeError(
        "The property '%s' can no longer be set directly on "
        "the solver object. It should instead be passed as a "
        "keyword into the solve method%s. It will automatically "
        "be reset to its default value after each invocation of "
        "solve." % (name, keyword))


class OptSolver(object):
    """A generic optimization solver"""

    #
    # Support "with" statements. Forgetting to call deactivate
    # on Plugins is a common source of memory leaks
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    #
    # Adding to help track down invalid code after making
    # the following attributes private
    #
    @property
    def tee(self):
        _raise_ephemeral_error('tee')
    @tee.setter
    def tee(self, val):
        _raise_ephemeral_error('tee')

    @property
    def suffixes(self):
        _raise_ephemeral_error('suffixes')
    @suffixes.setter
    def suffixes(self, val):
        _raise_ephemeral_error('suffixes')

    @property
    def keepfiles(self):
        _raise_ephemeral_error('keepfiles')
    @keepfiles.setter
    def keepfiles(self, val):
        _raise_ephemeral_error('keepfiles')

    @property
    def soln_file(self):
        _raise_ephemeral_error('soln_file')
    @soln_file.setter
    def soln_file(self, val):
        _raise_ephemeral_error('soln_file')

    @property
    def log_file(self):
        _raise_ephemeral_error('log_file')
    @log_file.setter
    def log_file(self, val):
        _raise_ephemeral_error('log_file')

    @property
    def symbolic_solver_labels(self):
        _raise_ephemeral_error('symbolic_solver_labels')
    @symbolic_solver_labels.setter
    def symbolic_solver_labels(self, val):
        _raise_ephemeral_error('symbolic_solver_labels')

    @property
    def warm_start_solve(self):
        _raise_ephemeral_error('warm_start_solve', keyword=" (warmstart)")
    @warm_start_solve.setter
    def warm_start_solve(self, val):
        _raise_ephemeral_error('warm_start_solve', keyword=" (warmstart)")

    @property
    def warm_start_file_name(self):
        _raise_ephemeral_error('warm_start_file_name', keyword=" (warmstart_file)")
    @warm_start_file_name.setter
    def warm_start_file_name(self, val):
        _raise_ephemeral_error('warm_start_file_name', keyword=" (warmstart_file)")

    def __init__(self, **kwds):
        """ Constructor """
        #
        # The 'type' is the class type of the solver instance
        #
        if "type" in kwds:
            self.type = kwds["type"]
        else:                           #pragma:nocover
            raise ValueError("Expected option 'type' for OptSolver constructor")

        #
        # The 'name' is either the class type of the solver instance, or a
        # assigned name.
        #
        if "name" in kwds:
            self.name = kwds["name"]
        else:
            self.name = self.type

        if "doc" in kwds:
            self._doc = kwds["doc"]
        else:
            if self.type is None:           # pragma:nocover
                self._doc = ""
            elif self.name == self.type:
                self._doc = "%s OptSolver" % self.name
            else:
                self._doc = "%s OptSolver (type %s)" % (self.name,self.type)
        #
        # Options are persistent, meaning users must modify the
        # options dict directly rather than pass them into _presolve
        # through the solve command. Everything else is reset inside
        # presolve
        #
        self.options = pyutilib.misc.Options()
        if 'options' in kwds and not kwds['options'] is None:
            for key in kwds['options']:
                setattr(self.options, key, kwds['options'][key])

        # the symbol map is an attribute of the solver plugin only
        # because it is generated in presolve and used to tag results
        # so they are interpretable - basically, it persists across
        # multiple methods.
        self._smap_id = None

        # These are ephimeral options that can be set by the user during
        # the call to solve, but will be reset to defaults if not given
        self._load_solutions = True
        self._select_index = 0
        self._report_timing = False
        self._suffixes = []
        self._log_file = None
        self._soln_file = None

        # overridden by a solver plugin when it returns sparse results
        self._default_variable_value = None
        # overridden by a solver plugin when it is always available
        self._assert_available = False
        # overridden by a solver plugin to indicate its input file format
        self._problem_format = None
        self._valid_problem_formats = []
        # overridden by a solver plugin to indicate its results file format
        self._results_format = None
        self._valid_result_formats = {}

        self._results_reader = None
        self._problem = None
        self._problem_files = None

        #
        # Used to document meta solvers
        #
        self._metasolver = False

        self._version = None
        #
        # Data for solver callbacks
        #
        self._allow_callbacks = False
        self._callback = {}

        # We define no capabilities for the generic solver; base
        # classes must override this
        self._capabilities = pyutilib.misc.Options()

    @staticmethod
    def _options_string_to_dict(istr):
        ans = {}
        istr = istr.strip()
        if not istr:
            return ans
        if istr[0] == "'" or istr[0] == '"':
            istr = eval(istr)
        tokens = pyutilib.misc.quote_split('[ ]+',istr)
        for token in tokens:
            index = token.find('=')
            if index is -1:
                raise ValueError(
                    "Solver options must have the form option=value: '%s'" % istr)
            try:
                val = eval(token[(index+1):])
            except:
                val = token[(index+1):]
            ans[token[:index]] = val
        return ans

    def default_variable_value(self):
        return self._default_variable_value

    def __bool__(self):
        return self.available()

    def version(self):
        """
        Returns a 4-tuple describing the solver executable version.
        """
        if self._version is None:
            self._version = self._get_version()
        return self._version

    def _get_version(self):
        return None

    def problem_format(self):
        """
        Returns the current problem format.
        """
        return self._problem_format

    def set_problem_format(self, format):
        """
        Set the current problem format (if it's valid) and update
        the results format to something valid for this problem format.
        """
        if format in self._valid_problem_formats:
            self._problem_format = format
        else:
            raise ValueError("%s is not a valid problem format for solver plugin %s"
                             % (format, self))
        self._results_format = self._default_results_format(self._problem_format)

    def results_format(self):
        """
        Returns the current results format.
        """
        return self._results_format

    def set_results_format(self,format):
        """
        Set the current results format (if it's valid for the current
        problem format).
        """
        if (self._problem_format in self._valid_results_formats) and \
           (format in self._valid_results_formats[self._problem_format]):
            self._results_format = format
        else:
            raise ValueError("%s is not a valid results format for "
                             "problem format %s with solver plugin %s"
                             % (format, self._problem_format, self))

    def has_capability(self, cap):
        """
        Returns a boolean value representing whether a solver supports
        a specific feature. Defaults to 'False' if the solver is unaware
        of an option. Expects a string.

        Example:
        # prints True if solver supports sos1 constraints, and False otherwise
        print(solver.has_capability('sos1')

        # prints True is solver supports 'feature', and False otherwise
        print(solver.has_capability('feature')

        Parameters
        ----------
        cap: str
            The feature

        Returns
        -------
        val: bool
            Whether or not the solver has the specified capability.
        """
        if not isinstance(cap, str):
            raise TypeError("Expected argument to be of type '%s', not " + \
                  "'%s'." % (str(type(str())), str(type(cap))))
        else:
            val = self._capabilities[str(cap)]
            if val is None:
                return False
            else:
                return val

    def available(self, exception_flag=True):
        """ True if the solver is available """
        return True

    def warm_start_capable(self):
        """ True is the solver can accept a warm-start solution """
        return False

    def solve(self, *args, **kwds):
        """ Solve the problem """

        self.available(exception_flag=True)
        #
        # If the inputs are models, then validate that they have been
        # constructed! Collect suffix names to try and import from solution.
        #
        from pyomo.core.base.block import _BlockData
        import pyomo.core.base.suffix
        from pyomo.core.kernel.block import IBlock
        import pyomo.core.kernel.suffix
        _model = None
        for arg in args:
            if isinstance(arg, (_BlockData, IBlock)):
                if isinstance(arg, _BlockData):
                    if not arg.is_constructed():
                        raise RuntimeError(
                            "Attempting to solve model=%s with unconstructed "
                            "component(s)" % (arg.name,) )

                _model = arg
                # import suffixes must be on the top-level model
                if isinstance(arg, _BlockData):
                    model_suffixes = list(name for (name,comp) \
                                          in pyomo.core.base.suffix.\
                                          active_import_suffix_generator(arg))
                else:
                    assert isinstance(arg, IBlock)
                    model_suffixes = list(comp.storage_key for comp
                                          in pyomo.core.kernel.suffix.\
                                          import_suffix_generator(arg,
                                                                  active=True,
                                                                  descend_into=False))

                if len(model_suffixes) > 0:
                    kwds_suffixes = kwds.setdefault('suffixes',[])
                    for name in model_suffixes:
                        if name not in kwds_suffixes:
                            kwds_suffixes.append(name)

        #
        # Handle ephemeral solvers options here. These
        # will override whatever is currently in the options
        # dictionary, but we will reset these options to
        # their original value at the end of this method.
        #

        orig_options = self.options

        self.options = pyutilib.misc.Options()
        self.options.update(orig_options)
        self.options.update(kwds.pop('options', {}))
        self.options.update(
            self._options_string_to_dict(kwds.pop('options_string', '')))
        try:

            # we're good to go.
            initial_time = time.time()

            self._presolve(*args, **kwds)

            presolve_completion_time = time.time()
            if self._report_timing:
                print("      %6.2f seconds required for presolve" % (presolve_completion_time - initial_time))

            if not _model is None:
                self._initialize_callbacks(_model)

            _status = self._apply_solver()
            if hasattr(self, '_transformation_data'):
                del self._transformation_data
            if not hasattr(_status, 'rc'):
                logger.warning(
                    "Solver (%s) did not return a solver status code.\n"
                    "This is indicative of an internal solver plugin error.\n"
                    "Please report this to the Pyomo developers." )
            elif _status.rc:
                logger.error(
                    "Solver (%s) returned non-zero return code (%s)"
                    % (self.name, _status.rc,))
                if self._tee:
                    logger.error(
                        "See the solver log above for diagnostic information." )
                elif hasattr(_status, 'log') and _status.log:
                    logger.error("Solver log:\n" + str(_status.log))
                raise pyutilib.common.ApplicationError(
                    "Solver (%s) did not exit normally" % self.name)
            solve_completion_time = time.time()
            if self._report_timing:
                print("      %6.2f seconds required for solver" % (solve_completion_time - presolve_completion_time))

            result = self._postsolve()
            result._smap_id = self._smap_id
            result._smap = None
            if _model:
                if isinstance(_model, IBlock):
                    if len(result.solution) == 1:
                        result.solution(0).symbol_map = \
                            getattr(_model, "._symbol_maps")[result._smap_id]
                        result.solution(0).default_variable_value = \
                            self._default_variable_value
                        if self._load_solutions:
                            _model.load_solution(result.solution(0))
                    else:
                        assert len(result.solution) == 0
                    # see the hack in the write method
                    # we don't want this to stick around on the model
                    # after the solve
                    assert len(getattr(_model, "._symbol_maps")) == 1
                    delattr(_model, "._symbol_maps")
                    del result._smap_id
                    if self._load_solutions and \
                       (len(result.solution) == 0):
                        logger.error("No solution is available")
                else:
                    if self._load_solutions:
                        _model.solutions.load_from(
                            result,
                            select=self._select_index,
                            default_variable_value=self._default_variable_value)
                        result._smap_id = None
                        result.solution.clear()
                    else:
                        result._smap = _model.solutions.symbol_map[self._smap_id]
                        _model.solutions.delete_symbol_map(self._smap_id)
            postsolve_completion_time = time.time()

            if self._report_timing:
                print("      %6.2f seconds required for postsolve"
                      % (postsolve_completion_time - solve_completion_time))

        finally:
            #
            # Reset the options dict
            #
            self.options = orig_options

        return result

    def _presolve(self, *args, **kwds):

        self._log_file                = kwds.pop("logfile", None)
        self._soln_file               = kwds.pop("solnfile", None)
        self._select_index            = kwds.pop("select", 0)
        self._load_solutions          = kwds.pop("load_solutions", True)
        self._timelimit               = kwds.pop("timelimit", None)
        self._report_timing           = kwds.pop("report_timing", False)
        self._tee                     = kwds.pop("tee", False)
        self._assert_available        = kwds.pop("available", True)
        self._suffixes                = kwds.pop("suffixes", [])

        self.available()

        if self._problem_format:
            write_start_time = time.time()
            (self._problem_files, self._problem_format, self._smap_id) = \
                self._convert_problem(args,
                                      self._problem_format,
                                      self._valid_problem_formats,
                                      **kwds)
            total_time = time.time() - write_start_time
            if self._report_timing:
                print("      %6.2f seconds required to write file" % total_time)
        else:
            if len(kwds):
                raise ValueError(
                    "Solver="+self.type+" passed unrecognized keywords: \n\t"
                    +("\n\t".join("%s = %s" % (k,v) for k,v in iteritems(kwds))))

        if six.PY3:
            compare_type = str
        else:
            compare_type = basestring

        if (type(self._problem_files) in (list,tuple)) and \
           (not isinstance(self._problem_files[0], compare_type)):
            self._problem_files = self._problem_files[0]._problem_files()
        if self._results_format is None:
            self._results_format = self._default_results_format(self._problem_format)

        #
        # Disabling this check for now.  A solver doesn't have just
        # _one_ results format.
        #
        #if self._results_format not in \
        #   self._valid_result_formats[self._problem_format]:
        #    raise ValueError("Results format '"+str(self._results_format)+"' "
        #                     "cannot be used with problem format '"
        #                     +str(self._problem_format)+"' in solver "+self.name)
        if self._results_format == ResultsFormat.soln:
            self._results_reader = None
        else:
            self._results_reader = \
                pyomo.opt.base.results.ReaderFactory(self._results_format)

    def _initialize_callbacks(self, model):
        """Initialize call-back functions"""
        pass

    def _apply_solver(self):
        """The routine that performs the solve"""
        raise NotImplementedError       #pragma:nocover

    def _postsolve(self):
        """The routine that does solve post-processing"""
        return self.results

    def _convert_problem(self,
                         args,
                         problem_format,
                         valid_problem_formats,
                         **kwds):
        #
        # If the problem is not None, then we assume that it has
        # already been appropriately defined.  Either it's a string
        # name of the problem we want to solve, or its a functor
        # object that we can evaluate directly.
        #
        if self._problem is not None:
            return (self._problem,
                    ProblemFormat.colin_optproblem,
                    None)

        #
        # Otherwise, we try to convert the object explicitly.
        #
        return convert_problem(args,
                               problem_format,
                               valid_problem_formats,
                               self.has_capability,
                               **kwds)

    def _default_results_format(self, prob_format):
        """Returns the default results format for different problem
            formats.
        """
        return ResultsFormat.results

    def reset(self):
        """
        Reset the state of the solver
        """
        pass

    def _get_options_string(self, options=None):
        if options is None:
            options = self.options
        ans = []
        for key in options:
            val = options[key]
            if isinstance(val, six.string_types) and ' ' in val:
                ans.append("%s=\"%s\"" % (str(key), str(val)))
            else:
                ans.append("%s=%s" % (str(key), str(val)))
        return ' '.join(ans)

    def set_options(self, istr):
        if isinstance(istr, six.string_types):
            istr = self._options_string_to_dict(istr)
        for key in istr:
            if not istr[key] is None:
                setattr(self.options, key, istr[key])

    def set_callback(self, name, callback_fn=None):
        """
        Set the callback function for a named callback.

        A call-back function has the form:

            def fn(solver, model):
                pass

        where 'solver' is the native solver interface object and 'model' is
        a Pyomo model instance object.
        """
        if not self._allow_callbacks:
            raise pyutilib.common.ApplicationError(
                "Callbacks disabled for solver %s" % self.name)
        if callback_fn is None:
            if name in self._callback:
                del self._callback[name]
        else:
            self._callback[name] = callback_fn

    def config_block(self, init=False):
        config, blocks = default_config_block(self, init=init)
        return config


def default_config_block(solver, init=False):
    config, blocks = ProblemConfigFactory('default').config_block(init)

    #
    # Solver
    #
    solver = ConfigBlock()
    solver.declare('solver name', ConfigValue(
                'glpk',
                str,
                'Solver name',
                None) )
    solver.declare('solver executable', ConfigValue(
        default=None,
        domain=str,
        description="The solver executable used by the solver interface.",
        doc=("The solver executable used by the solver interface. "
             "This option is only valid for those solver interfaces that "
             "interact with a local executable through the shell. If unset, "
             "the solver interface will attempt to find an executable within "
             "the search path of the shell's environment that matches a name "
             "commonly associated with the solver interface.")))
    solver.declare('io format', ConfigValue(
                None,
                str,
                'The type of IO used to execute the solver. Different solvers support different types of IO, but the following are common options: lp - generate LP files, nl - generate NL files, python - direct Python interface, os - generate OSiL XML files.',
                None) )
    solver.declare('manager', ConfigValue(
                'serial',
                str,
                'The technique that is used to manage solver executions.',
                None) )
    solver.declare('pyro host', ConfigValue(
                None,
                str,
                "The hostname to bind on when searching for a Pyro nameserver.",
                None) )
    solver.declare('pyro port', ConfigValue(
                None,
                int,
                "The port to bind on when searching for a Pyro nameserver.",
                None) )
    solver.declare('options', ConfigBlock(
                implicit=True,
                implicit_domain=ConfigValue(
                    None,
                    str,
                    'Solver option',
                    None),
                description="Options passed into the solver") )
    solver.declare('options string', ConfigValue(
                None,
                str,
                'String describing solver options',
                None) )
    solver.declare('suffixes', ConfigList(
                [],
                ConfigValue(None, str, 'Suffix', None),
                'Solution suffixes that will be extracted by the solver (e.g., rc, dual, or slack). The use of this option is not required when a suffix has been declared on the model using Pyomo\'s Suffix component.',
                None) )
    blocks['solver'] = solver
    #
    solver_list = config.declare('solvers', ConfigList(
                [],
                solver, #ConfigValue(None, str, 'Solver', None),
                'List of solvers.  The first solver in this list is the master solver.',
                None) )
    #
    # Make sure that there is one solver in the list.
    #
    # This will be the solver into which we dump command line options.
    # Note that we CANNOT declare the argparse options on the base block
    # definition above, as we use that definition as the DOMAIN TYPE for
    # the list of solvers.  As that information is NOT copied to
    # derivative blocks, the initial solver entry we are creating would
    # be missing all argparse information. Plus, if we were to have more
    # than one solver defined, we wouldn't want command line options
    # going to both.
    solver_list.append()
    solver_list[0].get('solver name').\
        declare_as_argument('--solver', dest='solver')
    solver_list[0].get('solver executable').\
        declare_as_argument('--solver-executable',
                            dest="solver_executable", metavar="FILE")
    solver_list[0].get('io format').\
        declare_as_argument('--solver-io', dest='io_format', metavar="FORMAT")
    solver_list[0].get('manager').\
        declare_as_argument('--solver-manager', dest="smanager_type",
                            metavar="TYPE")
    solver_list[0].get('pyro host').\
        declare_as_argument('--pyro-host', dest="pyro_host")
    solver_list[0].get('pyro port').\
        declare_as_argument('--pyro-port', dest="pyro_port")
    solver_list[0].get('options string').\
        declare_as_argument('--solver-options', dest='options_string',
                            metavar="STRING")
    solver_list[0].get('suffixes').\
        declare_as_argument('--solver-suffix', dest="solver_suffixes")

    #
    # Postprocess
    #
    config.declare('postprocess', ConfigList(
                [],
                ConfigValue(None, str, 'Module', None),
                'Specify a Python module that gets executed after optimization.',
                None) ).declare_as_argument(dest='postprocess')

    #
    # Postsolve
    #
    postsolve = config.declare('postsolve', ConfigBlock())
    postsolve.declare('print logfile', ConfigValue(
                False,
                bool,
                'Print the solver logfile after performing optimization.',
                None) ).declare_as_argument('-l', '--log', dest="log")
    postsolve.declare('save results', ConfigValue(
                None,
                str,
                'Specify the filename to which the results are saved.',
                None) ).declare_as_argument('--save-results', dest="save_results", metavar="FILE")
    postsolve.declare('show results', ConfigValue(
                False,
                bool,
                'Print the results object after optimization.',
                None) ).declare_as_argument(dest="show_results")
    postsolve.declare('results format', ConfigValue(
                None,
                str,
                'Specify the results format:  json or yaml.',
                None) ).declare_as_argument('--results-format', dest="results_format", metavar="FORMAT").declare_as_argument('--json', dest="results_format", action="store_const", const="json", help="Store results in JSON format")
    postsolve.declare('summary', ConfigValue(
                False,
                bool,
                'Summarize the final solution after performing optimization.',
                None) ).declare_as_argument(dest="summary")
    blocks['postsolve'] = postsolve

    #
    # Runtime
    #
    runtime = blocks['runtime']
    runtime.declare('only instance', ConfigValue(
                False,
                bool,
                "Generate a model instance, and then exit",
                None) ).declare_as_argument('--instance-only', dest='only_instance')
    runtime.declare('stream output', ConfigValue(
                False,
                bool,
                "Stream the solver output to provide information about the solver's progress.",
                None) ).declare_as_argument('--stream-output', '--stream-solver', dest="tee")
    #
    return config, blocks

