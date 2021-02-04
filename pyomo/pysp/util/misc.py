#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("launch_command",
           "load_external_module",
           "parse_command_line")

import logging
import time
import six
import sys
import subprocess
import inspect
import argparse

from pyutilib.misc import import_file
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.errors import ApplicationError
from pyomo.common.gc_manager import PauseGC
from pyomo.opt.base import ConverterError
from pyomo.common.dependencies import attempt_import
from pyomo.common.plugin import (ExtensionPoint,
                                 SingletonPlugin)
from pyomo.pysp.util.config import PySPConfigBlock
from pyomo.pysp.util.configured_object import PySPConfiguredObject

pyu_pyro = attempt_import('pyutilib.pyro', alt_names=['pyu_pyro'])[0]


logger = logging.getLogger('pyomo.pysp')

def _generate_unique_module_name():
    import uuid
    name = str(uuid.uuid4())
    while name in sys.modules:
        name = str(uuid.uuid4())
    return name

def load_external_module(module_name,
                         unique=False,
                         clear_cache=False,
                         verbose=False):
    try:
        # make sure "." is in the PATH.
        original_path = list(sys.path)
        sys.path.insert(0,'.')

        sys_modules_key = None
        module_to_find = None
        #
        # Getting around CPython implementation detail:
        #   sys.modules contains dummy entries set to None.
        #   It is related to relative imports. Long story short,
        #   we must check that both module_name is in sys.modules
        #   AND its entry is not None.
        #
        if (module_name in sys.modules) and \
           (sys.modules[module_name] is not None):
            sys_modules_key = module_name
            if clear_cache:
                if unique:
                    sys_modules_key = _generate_unique_module_name()
                    if verbose:
                        print("Module="+module_name+" is already imported - "
                              "forcing re-import using unique module id="
                              +str(sys_modules_key))
                    module_to_find = import_file(module_name, name=sys_modules_key)
                    if verbose:
                        print("Module successfully loaded")
                else:
                    if verbose:
                        print("Module="+module_name+" is already imported - "
                              "forcing re-import")
                    module_to_find = import_file(module_name, clear_cache=True)
                    if verbose:
                        print("Module successfully loaded")
            else:
                if verbose:
                    print("Module="+module_name+" is already imported - skipping")
                module_to_find = sys.modules[module_name]
        else:
            if unique:
                sys_modules_key = _generate_unique_module_name()
                if verbose:
                    print("Importing module="+module_name+" using "
                          "unique module id="+str(sys_modules_key))
                module_to_find = import_file(module_name, name=sys_modules_key)
                if verbose:
                    print("Module successfully loaded")
            else:
                if verbose:
                    print("Importing module="+module_name)
                _context = {}
                module_to_find = import_file(module_name, context=_context, clear_cache=clear_cache)
                assert len(_context) == 1
                sys_modules_key = list(_context.keys())[0]
                if verbose:
                    print("Module successfully loaded")

    finally:
        # restore to what it was
        sys.path[:] = original_path

    return module_to_find, sys_modules_key

def sort_extensions_by_precedence(extensions):
    import pyomo.pysp.util.configured_object
    return tuple(sorted(
        extensions,
        key=lambda ext:
        (ext.get_option('extension_precedence') if \
         isinstance(ext, pyomo.pysp.util.configured_object.\
                    PySPConfiguredExtension) else \
         float('-inf'))))

def load_extensions(names, ep_type):
    import pyomo.environ

    plugins = ExtensionPoint(ep_type)

    active_plugins = []
    for this_extension in names:
        module, _ = load_external_module(this_extension)
        assert module is not None
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # the second condition gets around goofyness related
            # to issubclass returning True when the obj is the
            # same as the test class.
            if issubclass(obj, SingletonPlugin) and \
               (name != "SingletonPlugin"):
                for plugin in plugins(all=True):
                    if isinstance(plugin, obj):
                        active_plugins.append(plugin)

    return tuple(active_plugins)

#
# A utility function for generating an argparse object and parsing the
# command line from a callback that registers options onto a
# PySPConfigBlock.  Optionally, a list of extension point types can be
# supplied, which causes reparsing to occur when any extensions are
# specified on the command-line that might register additional
# options.
#
# with_extensions: should be a dictionary mapping registered
#                  option name to the ExtensionPoint service
#

def parse_command_line(args,
                       register_options_callback,
                       with_extensions=None,
                       **kwds):
    import pyomo.pysp.plugins
    pyomo.pysp.plugins.load()
    from pyomo.pysp.util.config import _domain_tuple_of_str

    registered_extensions = {}
    if with_extensions is not None:
        for name in with_extensions:
            plugins = ExtensionPoint(with_extensions[name])
            for plugin in plugins(all=True):
                registered_extensions.setdefault(name,[]).\
                    append(plugin.__class__.__module__)

    def _get_argument_parser(options):
        # if we modify this and don't copy it,
        # the this output will appear twice the second
        # time this function gets called
        _kwds = dict(kwds)
        if len(registered_extensions) > 0:
            assert with_extensions is not None
            epilog = _kwds.pop('epilog',"")
            if epilog != "":
                epilog += "\n\n"
            epilog += "Registered Extensions:\n"
            for name in registered_extensions:
                epilog += " - "+str(with_extensions[name].__name__)+": "
                epilog += str(registered_extensions[name])+"\n"
            _kwds['epilog'] = epilog
        ap = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **_kwds)
        options.initialize_argparse(ap)
        ap.add_argument("-h", "--help", dest="show_help",
                        action="store_true", default=False,
                        help="show this help message and exit")

        return ap

    #
    # Register options
    #
    options = PySPConfigBlock()
    register_options_callback(options)

    if with_extensions is not None:
        for name in with_extensions:
            configval = options.get(name, None)
            assert configval is not None
            assert configval._domain is _domain_tuple_of_str

    ap = _get_argument_parser(options)
    # First parse known args, then import any extension plugins
    # specified by the user, regenerate the options block and
    # reparse to pick up plugin specific registered options
    opts, _ = ap.parse_known_args(args=args)
    options.import_argparse(opts)
    extensions = {}
    if with_extensions is None:
        if opts.show_help:
            pass
    else:
        if all(len(options.get(name).value()) == 0
               for name in with_extensions) and \
               opts.show_help:
            ap.print_help()
            sys.exit(0)
        for name in with_extensions:
            extensions[name] = load_extensions(
                options.get(name).value(),
                with_extensions[name])

    # regenerate the options
    options = PySPConfigBlock()
    register_options_callback(options)
    for name in extensions:
        for plugin in extensions[name]:
            if isinstance(plugin, PySPConfiguredObject):
                plugin.register_options(options)
        # do a dummy access to option to prevent
        # a warning about it not being used
        options.get(name).value()

    ap = _get_argument_parser(options)
    opts = ap.parse_args(args=args)
    options.import_argparse(opts)
    for name in extensions:
        for plugin in extensions[name]:
            if isinstance(plugin, PySPConfiguredObject):
                plugin.set_options(options)
    if opts.show_help:
        ap.print_help()
        sys.exit(0)

    if with_extensions:
        for name in extensions:
            extensions[name] = sort_extensions_by_precedence(extensions[name])
        return options, extensions
    else:
        return options

#
# When we create official command-line applications
# there is a long list of processing related to
# traceback and profile handling that should not need
# to be copy-pasted everywhere
#
def launch_command(command,
                   options,
                   cmd_args=None,
                   cmd_kwds=None,
                   error_label="",
                   disable_gc=False,
                   profile_count=0,
                   log_level=logging.INFO,
                   traceback=False):
    # This is not the effective level, but the
    # level on the current logger. We want to
    # return the logger to its original state
    # before this function exits
    prev_log_level = logger.level
    logger.setLevel(log_level)

    if cmd_args is None:
        cmd_args = ()
    if cmd_kwds is None:
        cmd_kwds = {}

    #
    # Control the garbage collector - more critical than I would like
    # at the moment.
    #
    with PauseGC(disable_gc) as pgc:

        #
        # Run command - precise invocation depends on whether we want
        # profiling output, traceback, etc.
        #

        rc = 0

        if profile_count > 0:
            # Defer import of profiling packages until we know that they
            # are needed
            try:
                try:
                    import cProfile as profile
                except ImportError:
                    import profile
                import pstats
            except ImportError:
                raise ValueError(
                    "Cannot use the 'profile' option: the Python "
                    "'profile' or 'pstats' package cannot be imported!")
            #
            # Call the main routine with profiling.
            #
            try:
                tfile = TempfileManager.create_tempfile(suffix=".profile")
                tmp = profile.runctx('command(options, *cmd_args, **cmd_kwds)',
                                     globals(),
                                     locals(),
                                     tfile)
                p = pstats.Stats(tfile).strip_dirs()
                p.sort_stats('time', 'cumulative')
                p = p.print_stats(profile_count)
                p.print_callers(profile_count)
                p.print_callees(profile_count)
                p = p.sort_stats('cumulative','calls')
                p.print_stats(profile_count)
                p.print_callers(profile_count)
                p.print_callees(profile_count)
                p = p.sort_stats('calls')
                p.print_stats(profile_count)
                p.print_callers(profile_count)
                p.print_callees(profile_count)
                TempfileManager.clear_tempfiles()
                rc = tmp
            finally:
                logger.setLevel(prev_log_level)
        else:

            #
            # Call the main PH routine without profiling.
            #
            if traceback:
                try:
                    rc = command(options, *cmd_args, **cmd_kwds)
                finally:
                    logger.setLevel(prev_log_level)
            else:
                try:
                    try:
                        rc = command(options, *cmd_args, **cmd_kwds)
                    except ValueError:
                        sys.stderr.write(error_label+"VALUE ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except KeyError:
                        sys.stderr.write(error_label+"KEY ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except TypeError:
                        sys.stderr.write(error_label+"TYPE ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except NameError:
                        sys.stderr.write(error_label+"NAME ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except IOError:
                        sys.stderr.write(error_label+"IO ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except ConverterError:
                        sys.stderr.write(error_label+"CONVERTER ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except ApplicationError:
                        sys.stderr.write(error_label+"APPLICATION ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except RuntimeError:
                        sys.stderr.write(error_label+"RUN-TIME ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except:
                        sys.stderr.write(error_label+
                                         "Encountered unhandled exception:\n")
                        if len(sys.exc_info()) > 1:
                            sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        else:
                            traceback.print_exc(file=sys.stderr)
                        raise
                except:
                    sys.stderr.write("\n")
                    sys.stderr.write(
                        "To obtain further information regarding the "
                        "source of the exception, use the "
                        "--traceback option\n")
                    rc = 1

    #
    # TODO: Once we incorporate options registration into
    #       all of the PySP commands we will assume the
    #       options object is always a PySPConfigBlock
    #
    if isinstance(options, PySPConfigBlock):
        options.check_usage(error=False)

    logger.setLevel(prev_log_level)

    return rc

def _poll(proc):
    if proc is None:
        return
    proc.poll()
    if proc.returncode:
        raise OSError

def _kill(proc):
    if proc is None:
        return
    if proc.stdout is not None:
        proc.stdout.close()
    if proc.stderr is not None:
        proc.stderr.close()
    while proc.returncode is None:
        try:
            proc.terminate()
        except:
            pass
        if six.PY3:
            proc.wait(timeout=1)
            if proc.returncode is None:
                try:
                    proc.kill()
                except:
                    pass
            proc.wait(timeout=1)
        else:
            proc.poll()
            if proc.returncode is None:
                time.sleep(0.5)
                try:
                    proc.kill()
                except:
                    pass
            proc.poll()

def _get_test_nameserver(ns_host="127.0.0.1", num_tries=20):
    if not (pyu_pyro.using_pyro3 or pyu_pyro.using_pyro4):
        return None, None
    ns_options = None
    if pyu_pyro.using_pyro3:
        ns_options = ["-r","-k","-n "+ns_host]
    elif pyu_pyro.using_pyro4:
        ns_options = ["--host="+ns_host]
    # don't start the broadcast server
    ns_options += ["-x"]
    ns_port = None
    ns_process = None
    for i in range(num_tries):
        try:
            ns_port = pyu_pyro.util.find_unused_port()
            print("Trying nameserver with port: "
                  +str(ns_port))
            cmd = ["pyomo_ns"] + ns_options
            if pyu_pyro.using_pyro3:
                cmd += ["-p "+str(ns_port)]
            elif pyu_pyro.using_pyro4:
                cmd += ["--port="+str(ns_port)]
            print(' '.join(cmd))
            ns_process = \
                subprocess.Popen(cmd, stdout=subprocess.PIPE)
            time.sleep(5)
            _poll(ns_process)
            break
        except OSError:
            print(sys.exc_info())
            print("Failed to find open port - trying again in 20 seconds")
            time.sleep(20)
            _kill(ns_process)
            ns_port = None
            ns_process = None
    return ns_process, ns_port

def _get_test_dispatcher(ns_host=None,
                         ns_port=None,
                         dispatcher_host="127.0.0.1",
                         num_tries=20):
    if not (pyu_pyro.using_pyro3 or pyu_pyro.using_pyro4):
        return None, None
    dispatcher_port = None
    dispatcher_process = None
    for i in range(num_tries):
        try:
            dispatcher_port = pyu_pyro.util.find_unused_port()
            print("Trying dispatcher with port: "
                  +str(dispatcher_port))
            cmd = ["dispatch_srvr",
                   "--host="+str(ns_host),
                   "--port="+str(ns_port),
                   "--daemon-host="+str(dispatcher_host),
                   "--daemon-port="+str(dispatcher_port)]
            print(' '.join(cmd))
            dispatcher_process = \
                subprocess.Popen(cmd, stdout=subprocess.PIPE)
            time.sleep(5)
            _poll(dispatcher_process)
            break
        except OSError as e:
            print(sys.exc_info())
            print("Failed to find open port - trying again in 20 seconds")
            time.sleep(20)
            _kill(dispatcher_process)
            dispatcher_port = None
            dispatcher_process = None
    return dispatcher_process, dispatcher_port
