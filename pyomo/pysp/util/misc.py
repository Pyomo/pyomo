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

from pyutilib.misc import PauseGC
from pyutilib.services import TempfileManager
import pyutilib.common
from pyomo.opt.base import ConverterError

#
# When we create official command-line applications
# there is a long list of processing related to
# traceback and profile handling that should not need
# to be copy-pasted everywhere
#
def launch_command(command_string,
                   global_context, # (e.g., globals())
                   local_context,  # (e.g., locals())
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
            tmp = profile.runctx(command_string,
                                 global_context,
                                 local_context,
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
                rc = eval(command_string,
                          global_context,
                          local_context)
            else:
                try:
                    try:
                        rc = eval(command_string,
                                  global_context,
                                  local_context)
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

    return rc
