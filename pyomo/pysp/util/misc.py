#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("launch_command", "load_external_module")

import sys
import traceback
# for profiling
try:
    import cProfile as profile
except ImportError:
    import profile
try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

from pyutilib.misc import PauseGC, import_file
from pyutilib.misc.config import ConfigBlock
from pyutilib.services import TempfileManager
import pyutilib.common
from pyomo.opt.base import ConverterError

#
# When we create official command-line applications
# there is a long list of processing related to
# traceback and profile handling that should not need
# to be copy-pasted everywhere
#
def launch_command(command,
                   options,
                   error_label="",
                   disable_gc=False,
                   profile_count=0,
                   traceback=False):

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

        if pstats_available and (profile_count > 0):
            #
            # Call the main PH routine with profiling.
            #
            tfile = TempfileManager.create_tempfile(suffix=".profile")
            tmp = profile.runctx('command(options)',
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
        else:
            
            #
            # Call the main PH routine without profiling.
            #
            if traceback:
                rc = command(options)
            else:
                try:
                    try:
                        rc = command(options)
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
                    except pyutilib.common.ApplicationError:
                        sys.stderr.write(error_label+"APPLICATION ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except RuntimeError:
                        sys.stderr.write(error_label+"RUN-TIME ERROR:\n")
                        sys.stderr.write(str(sys.exc_info()[1])+"\n")
                        raise
                    except:
                        sys.stderr.write(error_label+"Encountered unhandled exception:\n")
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
    #       options object is always a ConfigBlock
    #
    if isinstance(options, ConfigBlock):
        ignored_options = dict((_c._name, _c.value(False))
                              for _c in options.unused_user_values())
        if len(ignored_options):
            print("")
            print("*** WARNING: The following options were "
                  "set but never checked by this command:")
            for name in ignored_options:
                print(" - %s: %s" % (name, ignored_options[name]))
            print("*** If you believe this is a bug, please report it "
                  "to the PySP developers.")
            print("")

    return rc

def _generate_unique_module_name():
    import uuid
    name = str(uuid.uuid4())
    while name in sys.modules:
        name = str(uuid.uuid4())
    return name

def load_external_module(module_name):
    sys_modules_key = None
    module_to_find = None
    if module_name in sys.modules:
        print("Module="+module_name+" already imported - skipping")
        module_to_find = sys.modules[module_name]
        sys_modules_key = module_name
    else:
        # If the module_name is not an imported module then import it using
        # a unique module id.
        print("Trying to import module="+module_name)
        sys_modules_key = _generate_unique_module_name()
        module_to_find = import_file(module_name, name=sys_modules_key)
        print("Module successfully loaded")

    return sys_modules_key, module_to_find
