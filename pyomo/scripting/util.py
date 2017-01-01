#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import argparse
import gc
import logging
import os
import sys
import traceback
import types
import time
import json
from six import itervalues, iterkeys, iteritems
from six.moves import xrange
from pyomo.util import pyomo_api

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False
try:
    import cProfile as profile
except ImportError:
    import profile
try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

try:
    import IPython
    IPython_available=True
    from IPython.Shell import IPShellEmbed
except:
    IPython_available=False
else:
    ipshell = IPShellEmbed([''],
                banner = '\n# Dropping into Python interpreter',
                exit_msg = '\n# Leaving Interpreter, back to Pyomo\n')

from pyutilib.misc import Options
try:
    from pympler import muppy
    from pympler import summary
    from pympler.asizeof import *
    pympler_available = True
except:
    pympler_available = False
memory_data = Options()

import pyutilib.misc
from pyomo.util.plugin import ExtensionPoint, Plugin, implements
from pyutilib.misc import Container
from pyutilib.services import TempfileManager

from pyomo.opt import ProblemFormat
from pyomo.opt.base import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.core import *
from pyomo.core.base import TextLabeler
import pyomo.core.base


filter_excepthook=False
modelapi = {    'pyomo_create_model':IPyomoScriptCreateModel,
                'pyomo_create_dataportal':IPyomoScriptCreateDataPortal,
                'pyomo_print_model':IPyomoScriptPrintModel,
                'pyomo_modify_instance':IPyomoScriptModifyInstance,
                'pyomo_print_instance':IPyomoScriptPrintInstance,
                'pyomo_save_instance':IPyomoScriptSaveInstance,
                'pyomo_print_results':IPyomoScriptPrintResults,
                'pyomo_save_results':IPyomoScriptSaveResults,
                'pyomo_postprocess':IPyomoScriptPostprocess}


logger = logging.getLogger('pyomo.core')
start_time = 0.0


@pyomo_api(namespace='pyomo.script')
def setup_environment(data):
    """
    Setup Pyomo execution environment
    """
    #
    postsolve = getattr(data.options, 'postsolve', None)
    if postsolve:
        if not yaml_available and data.options.postsolve.results_format == 'yaml':
            raise ValueError("Configuration specifies a yaml file, but pyyaml is not installed!")
        if data.options.postsolve.results_format is None:
            if yaml_available:
                data.options.postsolve.results_format = 'yaml'
            else:
                data.options.postsolve.results_format = 'json'
    #
    global start_time
    start_time = time.time()
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Setting up Pyomo environment\n' % 0.0)
        sys.stdout.flush()

    #
    # Disable garbage collection
    #
    if data.options.runtime.disable_gc:
        gc.disable()
    #
    # Setup management for temporary files
    #
    if not data.options.runtime.tempdir is None:
        if not os.path.exists(data.options.runtime.tempdir):
            msg =  'Directory for temporary files does not exist: %s'
            raise ValueError(msg % data.options.runtime.tempdir)
        TempfileManager.tempdir = data.options.runtime.tempdir

    #
    # Configure exception management
    #
    def pyomo_excepthook(etype,value,tb):
        """
        This exception hook gets called when debugging is on. Otherwise,
        run_command in this module is called.
        """
        global filter_excepthook
        if len(data.options.model.filename) > 0:
            name = "model " + data.options.model.filename
        else:
            name = "model"


        if filter_excepthook:
            action = "loading"
        else:
            action = "running"

        msg = "Unexpected exception (%s) while %s %s:\n" % (etype.__name__, action, name)

        #
        # This handles the case where the error is propagated by a KeyError.
        # KeyError likes to pass raw strings that don't handle newlines
        # (they translate "\n" to "\\n"), as well as tacking on single
        # quotes at either end of the error message. This undoes all that.
        #
        valueStr = str(value)
        if etype == KeyError:
            valueStr = valueStr.replace("\\n","\n")
            if valueStr[0] == valueStr[-1] and valueStr[0] in "\"'":
                valueStr = valueStr[1:-1]

        logger.error(msg+valueStr)

        tb_list = traceback.extract_tb(tb,None)
        i = 0
        if not logger.isEnabledFor(logging.DEBUG) and filter_excepthook:
            while i < len(tb_list):
                if data.options.model.filename in tb_list[i][0]:
                    break
                i += 1
            if i == len(tb_list):
                i = 0
        print("\nTraceback (most recent call last):")
        for item in tb_list[i:]:
            print("  File \""+item[0]+"\", line "+str(item[1])+", in "+item[2])
            if item[3] is not None:
                print("    "+item[3])
        sys.exit(1)
    sys.excepthook = pyomo_excepthook


@pyomo_api(namespace='pyomo.script')
def apply_preprocessing(data, parser=None):
    """
    Execute preprocessing files

    Required:
        parser: Command line parser object

    Returned:
        error: This is true if an error has occurred.
    """
    data.local = pyutilib.misc.Options()
    #
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Applying Pyomo preprocessing actions\n' % (time.time()-start_time))
        sys.stdout.flush()
    #
    global filter_excepthook
    #
    #
    # Setup solver and model
    #
    #
    if len(data.options.model.filename) == 0:
        parser.print_help()
        data.error = True
        return data
    #
    if not data.options.preprocess is None:
        for config_value in data.options.preprocess:
            preprocess = pyutilib.misc.import_file(config_value, clear_cache=True)
    #
    for ep in ExtensionPoint(IPyomoScriptPreprocess):
        ep.apply( options=data.options )
    #
    # Verify that files exist
    #
    for file in [data.options.model.filename]+data.options.data.files.value():
        if not os.path.exists(file):
            raise IOError("File "+file+" does not exist!")
    #
    filter_excepthook=True
    data.local.usermodel = pyutilib.misc.import_file(data.options.model.filename, clear_cache=True)
    filter_excepthook=False

    usermodel_dir = dir(data.local.usermodel)
    data.local._usermodel_plugins = []
    for key in modelapi:
        if key in usermodel_dir:
            class TMP(Plugin):
                implements(modelapi[key], service=True)
                def __init__(self):
                    self.fn = getattr(data.local.usermodel, key)
                def apply(self,**kwds):
                    return self.fn(**kwds)
            tmp = TMP()
            data.local._usermodel_plugins.append( tmp )
            #print "HERE", modelapi[key], pyomo.util.plugin.interface_services[modelapi[key]]

    #print "HERE", data.options._usermodel_plugins

    if 'pyomo_preprocess' in usermodel_dir:
        if data.options.model.object_name in usermodel_dir:
            msg = "Preprocessing function 'pyomo_preprocess' defined in file" \
                  " '%s', but model is already constructed!"
            raise SystemExit(msg % data.options.model.filename)
        getattr(data.local.usermodel, 'pyomo_preprocess')( options=data.options )
    #
    return data

@pyomo_api(namespace='pyomo.script')
def create_model(data):
    """
    Create instance of Pyomo model.

    Return:
        model:      Model object.
        instance:   Problem instance.
        symbol_map: Symbol map created when writing model to a file.
        filename:    Filename that a model instance was written to.
    """
    #
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Creating model\n' % (time.time()-start_time))
        sys.stdout.flush()
    #
    if (pympler_available is True) and (data.options.runtime.profile_memory >= 1):
        global memory_data
        mem_used = muppy.get_size(muppy.get_objects())
        data.local.max_memory = mem_used
        print("   Total memory = %d bytes prior to model construction" % mem_used)
    #
    # Find the Model objects
    #
    _models = {}
    for _name, _obj in iteritems(data.local.usermodel.__dict__):
        if isinstance(_obj, Model):
            _models[_name] = _obj
    model_name = data.options.model.object_name
    if len(_models) == 1:
        _name  = list(_models.keys())[0]
        if model_name is None:
            model_name  = _name
        elif model_name != _name:
            msg = "Model '%s' is not defined in file '%s'!"
            raise SystemExit(msg % (model_name, data.options.model.filename))
    elif len(_models) > 1:
        if model_name is None:
            msg = "Multiple models defined in file '%s'!"
            raise SystemExit(msg % data.options.model.filename)
        elif not model_name in _models:
            msg = "Unknown model '%s' in file '%s'!"
            raise SystemExit(msg % (model_name, data.options.model.filename))

    ep = ExtensionPoint(IPyomoScriptCreateModel)

    if model_name is None:
        if len(ep) == 0:
            msg = "A model is not defined and the 'pyomo_create_model' is not "\
                  "provided in module %s"
            raise SystemExit(msg % data.options.model.filename)
        elif len(ep) > 1:
            msg = 'Multiple model construction plugins have been registered in module %s!'
            raise SystemExit(msg % data.options.model.filename)
        else:
            model_options = data.options.model.options.value()
            model = ep.service().apply( options = pyutilib.misc.Container(*data.options), model_options=pyutilib.misc.Container(*model_options) )
    else:
        if model_name not in _models:
            msg = "Model '%s' is not defined in file '%s'!"
            raise SystemExit(msg % (model_name, data.options.model.filename))
        model = _models[model_name]
        if model is None:
            msg = "'%s' object is 'None' in module %s"
            raise SystemExit(msg % (model_name, data.options.model.filename))
        elif len(ep) > 0:
            msg = "Model construction function 'create_model' defined in "    \
                  "file '%s', but model is already constructed!"
            raise SystemExit(msg % data.options.model.filename)

    #
    # Print model
    #
    for ep in ExtensionPoint(IPyomoScriptPrintModel):
        ep.apply( options=data.options, model=model )

    #
    # Create Problem Instance
    #
    ep = ExtensionPoint(IPyomoScriptCreateDataPortal)
    if len(ep) > 1:
        msg = 'Multiple model data construction plugins have been registered!'
        raise SystemExit(msg)

    if len(ep) == 1:
        modeldata = ep.service().apply( options=data.options, model=model )
    else:
        modeldata = DataPortal()


    if model._constructed:
        #
        # TODO: use a better test for ConcreteModel
        #
        instance = model

    elif len(data.options.data.files) > 1:
        #
        # Load a list of *.dat files
        #
        for file in data.options.data.files:
            suffix = (file).split(".")[-1]
            if suffix != "dat":
                msg = 'When specifiying multiple data files, they must all '  \
                      'be *.dat files.  File specified: %s'
                raise SystemExit(msg % str( file ))

            modeldata.load(filename=file, model=model)

        instance = model.create_instance(modeldata,
                                         namespaces=data.options.data.namespaces,
                                         profile_memory=data.options.runtime.profile_memory,
                                         report_timing=data.options.runtime.report_timing)

    elif len(data.options.data.files) == 1:
        #
        # Load a *.dat file or process a *.py data file
        #
        suffix = (data.options.data.files[0]).split(".")[-1].lower()
        if suffix == "dat":
            instance = model.create_instance(data.options.data.files[0],
                                             namespaces=data.options.data.namespaces,
                                             profile_memory=data.options.runtime.profile_memory,
                                             report_timing=data.options.runtime.report_timing)
        elif suffix == "py":
            userdata = pyutilib.misc.import_file(data.options.data.files[0], clear_cache=True)
            if "modeldata" in dir(userdata):
                if len(ep) == 1:
                    msg = "Cannot apply 'pyomo_create_modeldata' and use the" \
                          " 'modeldata' object that is provided in the model"
                    raise SystemExit(msg)

                if userdata.modeldata is None:
                    msg = "'modeldata' object is 'None' in module %s"
                    raise SystemExit(msg % str( data.options.data.files[0] ))

                modeldata=userdata.modeldata

            else:
                if len(ep) == 0:
                    msg = "Neither 'modeldata' nor 'pyomo_create_dataportal' "  \
                          'is defined in module %s'
                    raise SystemExit(msg % str( data.options.data.files[0] ))

            modeldata.read(model)
            instance = model.create_instance(modeldata,
                                             namespaces=data.options.data.namespaces,
                                             profile_memory=data.options.runtime.profile_memory,
                                             report_timing=data.options.runtime.report_timing)
        elif suffix == "yml" or suffix == 'yaml':
            try:
                import yaml
            except:
                msg = "Cannot apply load data from a YAML file: PyYaml is not installed"
                raise SystemExit(msg)

            modeldata = yaml.load(open(data.options.data.files[0]))
            instance = model.create_instance(modeldata,
                                             namespaces=data.options.data.namespaces,
                                             profile_memory=data.options.runtime.profile_memory,
                                             report_timing=data.options.runtime.report_timing)
        else:
            raise ValueError("Unknown data file type: "+data.options.data.files[0])
    else:
        instance = model.create_instance(modeldata,
                                         namespaces=data.options.data.namespaces,
                                         profile_memory=data.options.runtime.profile_memory,
                                         report_timing=data.options.runtime.report_timing)

    #
    modify_start_time = time.time()
    for ep in ExtensionPoint(IPyomoScriptModifyInstance):
        if data.options.runtime.report_timing is True:
            tick = time.time()
        ep.apply( options=data.options, model=model, instance=instance )
        if data.options.runtime.report_timing is True:
            print("      %6.2f seconds to apply %s" % (time.time() - tick, type(ep)))
            tick = time.time()
    #
    for transformation in data.options.transform:
        with TransformationFactory(transformation) as xfrm:
            instance = xfrm.create_using(instance)
            if instance is None:
                raise SystemExit("Unexpected error while applying "
                                 "transformation '%s'" % transformation)
    #
    if data.options.runtime.report_timing is True:
        total_time = time.time() - modify_start_time
        print("      %6.2f seconds required for problem transformations" % total_time)

    if logger.isEnabledFor(logging.DEBUG):
        print("MODEL INSTANCE")
        instance.pprint()
        print("")

    for ep in ExtensionPoint(IPyomoScriptPrintInstance):
        ep.apply( options=data.options, instance=instance )

    fname=None
    smap_id=None
    if not data.options.model.save_file is None:

        if data.options.runtime.report_timing is True:
            write_start_time = time.time()

        if data.options.model.save_file == True:
            if data.local.model_format in (ProblemFormat.cpxlp, ProblemFormat.lpxlp):
                fname = (data.options.data.files[0])[:-3]+'lp'
            else:
                fname = (data.options.data.files[0])[:-3]+str(data.local.model_format)
            format=data.local.model_format
        else:
            fname = data.options.model.save_file
            format= data.options.model.save_format

        io_options = {}
        if data.options.model.symbolic_solver_labels:
            io_options['symbolic_solver_labels'] = True
        if data.options.model.file_determinism != 1:
            io_options['file_determinism'] = data.options.model.file_determinism
        (fname, smap_id) = instance.write(filename=fname,
                                          format=format,
                                          io_options=io_options)

        if not data.options.runtime.logging == 'quiet':
            if not os.path.exists(fname):
                print("ERROR: file "+fname+" has not been created!")
            else:
                print("Model written to file '"+str(fname)+"'")

        if data.options.runtime.report_timing is True:
            total_time = time.time() - write_start_time
            print("      %6.2f seconds required to write file" % total_time)

        if (pympler_available is True) and (data.options.runtime.profile_memory >= 2):
            print("")
            print("      Summary of objects following file output")
            post_file_output_summary = summary.summarize(muppy.get_objects())
            summary.print_(post_file_output_summary, limit=100)

            print("")

    for ep in ExtensionPoint(IPyomoScriptSaveInstance):
        ep.apply( options=data.options, instance=instance )

    if (pympler_available is True) and (data.options.runtime.profile_memory >= 1):
        mem_used = muppy.get_size(muppy.get_objects())
        if mem_used > data.local.max_memory:
            data.local.max_memory = mem_used
        print("   Total memory = %d bytes following Pyomo instance creation" % mem_used)

    return pyutilib.misc.Options(
                    model=model, instance=instance,
                    smap_id=smap_id, filename=fname, local=data.local )

@pyomo_api(namespace='pyomo.script')
def apply_optimizer(data, instance=None):
    """
    Perform optimization with a concrete instance

    Required:
        instance:   Problem instance.

    Returned:
        results:    Optimization results.
        opt:        Optimizer object.
    """
    #
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Applying solver\n' % (time.time()-start_time))
        sys.stdout.flush()
    #
    #
    # Create Solver and Perform Optimization
    #
    solver = data.options.solvers[0].solver_name
    if solver is None:
        raise ValueError("Problem constructing solver:  no solver specified")

    if len(data.options.solvers[0].suffixes) > 0:
        for suffix_name in data.options.solvers[0].suffixes:
            if suffix_name[0] in ['"',"'"]:
                suffix_name = suffix[1:-1]
            # Don't redeclare the suffix if it already exists
            suffix = getattr(instance, suffix_name, None)
            if suffix is None:
                setattr(instance, suffix_name, Suffix(direction=Suffix.IMPORT))
            else:
                raise ValueError("Problem declaring solver suffix %s. A component "\
                                  "with that name already exists on model %s."
                                 % (suffix_name, instance.name))

    if getattr(data.options.solvers[0].options, 'timelimit', 0) == 0:
        data.options.solvers[0].options.timelimit = None
    #
    # Default results
    #
    results = None
    #
    # Figure out the type of solver manager
    #
    solver_mngr_name = None
    if data.options.solvers[0].manager is None:
        solver_mngr_name = 'serial'
    elif not data.options.solvers[0].manager in SolverManagerFactory.services():
        raise ValueError("Unknown solver manager %s"
                         % data.options.solvers[0].manager)
    else:
        solver_mngr_name = data.options.solvers[0].manager
    #
    # Create the solver manager
    #
    solver_mngr_kwds = {}
    if data.options.solvers[0].pyro_host is not None:
        solver_mngr_kwds['host'] = data.options.solvers[0].pyro_host
    if data.options.solvers[0].pyro_port is not None:
        solver_mngr_kwds['port'] = data.options.solvers[0].pyro_port
    with SolverManagerFactory(solver_mngr_name, **solver_mngr_kwds) as solver_mngr:
        if solver_mngr is None:
            msg = "Problem constructing solver manager '%s'"
            raise ValueError(msg % str(data.options.solvers[0].manager))
        #
        # Setup keywords for the solve
        #
        keywords = {}
        if (data.options.runtime.keep_files or \
            data.options.postsolve.print_logfile):
            keywords['keepfiles'] = True
        if data.options.model.symbolic_solver_labels:
            keywords['symbolic_solver_labels'] = True
        if data.options.model.file_determinism != 1:
            keywords['file_determinism'] = data.options.model.file_determinism
        keywords['tee'] = data.options.runtime.stream_output
        keywords['timelimit'] = getattr(data.options.solvers[0].options, 'timelimit', 0)
        #
        # Call the solver
        #
        if solver_mngr_name == 'serial':
            #
            # If we're running locally, then we create the optimizer and pass it into the 
            # solver manager.
            #
            with SolverFactory(solver, solver_io=data.options.solvers[0].io_format) as opt:
                if opt is None:
                    raise ValueError("Problem constructing solver `%s`" % str(solver))

                from pyomo.core.base.plugin import registered_callback
                for name in registered_callback:
                    opt.set_callback(name, registered_callback[name])

                if len(data.options.solvers[0].options) > 0:
                    opt.set_options(data.options.solvers[0].options)
                    #opt.set_options(" ".join("%s=%s" % (key, value)
                    #                         for key, value in data.options.solvers[0].options.iteritems()
                    #                         if not key == 'timelimit'))
                if not data.options.solvers[0].options_string is None:
                    opt.set_options(data.options.solvers[0].options_string)
                #
                # Use the solver manager to call the optimizer
                #
                results = solver_mngr.solve(instance, opt=opt, **keywords)
        else:
            #
            # Get the solver option arguments
            #
            if len(data.options.solvers[0].options) > 0 and not data.options.solvers[0].options_string is None:
                # If both 'options' and 'options_string' were specified, then create a
                # single options string that is passed to the solver.
                ostring = " ".join("%s=%s" % (key, value)
                                             for key, value in data.options.solvers[0].options.iteritems()
                                             if not value is None)
                keywords['options'] = ostring + ' ' + data.options.solvers[0].options_string
            elif len(data.options.solvers[0].options) > 0:
                keywords['options'] = data.options.solvers[0].options
            else:
                keywords['options'] = data.options.solvers[0].options_string
            #
            # If we're running remotely, then we pass the optimizer name to the solver
            # manager.
            #
            results = solver_mngr.solve(instance, opt=solver, **keywords)

    if (pympler_available is True) and \
       (data.options.runtime.profile_memory >= 1):
        global memory_data
        mem_used = muppy.get_size(muppy.get_objects())
        if mem_used > data.local.max_memory:
            data.local.max_memory = mem_used
        print("   Total memory = %d bytes following optimization" % mem_used)

    return pyutilib.misc.Options(results=results, opt=opt, local=data.local)


@pyomo_api(namespace='pyomo.script')
def process_results(data, instance=None, results=None, opt=None):
    """
    Process optimization results.

    Required:
        instance:   Problem instance.
        results:    Optimization results object.
        opt:        Optimizer object.
    """
    #
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Processing results\n' % (time.time()-start_time))
        sys.stdout.flush()
    #
    if data.options.postsolve.print_logfile:
        print("")
        print("==========================================================")
        print("Solver Logfile: "+str(opt._log_file))
        print("==========================================================")
        print("")
        with open(opt._log_file, "r") as INPUT:
            for line in INPUT:
                sys.stdout.write(line)
        print("==========================================================")
        print("Solver Logfile - END")
        print("==========================================================")
    #
    try:
        # transform the results object into human-readable names.
        instance.solutions.store_to(results)
    except Exception:
        print("Problem updating solver results")
        raise
    #
    if not data.options.postsolve.show_results:
        if data.options.postsolve.save_results:
            results_file = data.options.postsolve.save_results
        elif data.options.postsolve.results_format == 'yaml':
            results_file = 'results.yml'
        else:
            results_file = 'results.json'
        results.write(filename=results_file,
                      format=data.options.postsolve.results_format)
        if not data.options.runtime.logging == 'quiet':
            print("    Number of solutions: "+str(len(results.solution)))
            if len(results.solution) > 0:
                print("    Solution Information")
                print("      Gap: "+str(results.solution[0].gap))
                print("      Status: "+str(results.solution[0].status))
                if len(results.solution[0].objective) == 1:
                    key = list(results.solution[0].objective.keys())[0]
                    print("      Function Value: "+str(results.solution[0].objective[key]['Value']))
            print("    Solver results file: "+results_file)
    #
    #ep = ExtensionPoint(IPyomoScriptPrintResults)
    if data.options.postsolve.show_results:
        print("")
        results.write(num=1, format=data.options.postsolve.results_format)
        print("")
    #
    if data.options.postsolve.summary:
        print("")
        print("==========================================================")
        print("Solution Summary")
        print("==========================================================")
        if len(results.solution(0).variable) > 0:
            print("")
            display(instance)
            print("")
        else:
            print("No solutions reported by solver.")
    #
    for ep in ExtensionPoint(IPyomoScriptPrintResults):
        ep.apply( options=data.options, instance=instance, results=results )
    #
    for ep in ExtensionPoint(IPyomoScriptSaveResults):
        ep.apply( options=data.options, instance=instance, results=results )
    #
    if (pympler_available is True) and (data.options.runtime.profile_memory >= 1):
        global memory_data
        mem_used = muppy.get_size(muppy.get_objects())
        if mem_used > data.local.max_memory:
            data.local.max_memory = mem_used
        print("   Total memory = %d bytes following results processing" % mem_used)

@pyomo_api(namespace='pyomo.script')
def apply_postprocessing(data, instance=None, results=None):
    """
    Apply post-processing steps.

    Required:
        instance:   Problem instance.
        results:    Optimization results object.
    """
    #
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Applying Pyomo postprocessing actions\n' % (time.time()-start_time))
        sys.stdout.flush()

    # options are of type ConfigValue, not raw strings / atomics.
    for config_value in data.options.postprocess:
        postprocess = pyutilib.misc.import_file(config_value, clear_cache=True)
        if "pyomo_postprocess" in dir(postprocess):
            postprocess.pyomo_postprocess(data.options, instance,results)

    for ep in ExtensionPoint(IPyomoScriptPostprocess):
        ep.apply( options=data.options, instance=instance, results=results )

    if (pympler_available is True) and (data.options.runtime.profile_memory >= 1):
        mem_used = muppy.get_size(muppy.get_objects())
        if mem_used > data.local.max_memory:
            data.local.max_memory = mem_used
        print("   Total memory = %d bytes upon termination" % mem_used)

@pyomo_api(namespace='pyomo.script')
def finalize(data, model=None, instance=None, results=None):
    """
    Perform final actions to finish the execution of the pyomo script.

    This function prints statistics related to the execution of the pyomo script.
    Additionally, this function will drop into the python interpreter if the `interactive`
    option is `True`.

    Required:
        model:      A pyomo model object.

    Optional:
        instance:   A problem instance derived from the model object.
        results:    Optimization results object.
    """
    #
    # Deactivate and delete plugins
    #
    ##import gc
    ##print "HERE - usermodel_plugins"
    ##_tmp = data.options._usermodel_plugins[0]
    cleanup()
    # NOTE: This function gets called for cleanup during exceptions
    #       to prevent memory leaks. Don't reconfigure the loggers
    #       here or we will lose the exception information.
    #configure_loggers(reset=True)
    data.local._usermodel_plugins = []
    ##gc.collect()
    ##print gc.get_referrers(_tmp)
    ##import pyomo.core.base.plugin
    ##print pyomo.util.plugin.interface_services[pyomo.core.base.plugin.IPyomoScriptSaveResults]
    ##print "HERE - usermodel_plugins"
    ##
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Pyomo Finished\n' % (time.time()-start_time))
        if (pympler_available is True) and (data.options.runtime.profile_memory >= 1):
            sys.stdout.write('Maximum memory used = %d bytes\n' % data.local.max_memory)
        sys.stdout.flush()
    #
    model=model
    instance=instance
    results=results
    #
    if data.options.runtime.interactive:
        if IPython_available:
            ipshell()
        else:
            import code
            shell = code.InteractiveConsole(locals())
            print('\n# Dropping into Python interpreter')
            shell.interact()
            print('\n# Leaving Interpreter, back to Pyomo\n')


def configure_loggers(options=None, reset=False):
    if reset:
        options = Options()
        options.runtime = Options()
        options.runtime.logging = 'quiet'
        logging.getLogger('pyomo.core').handlers = []
        logging.getLogger('pyomo').handlers = []
        logging.getLogger('pyutilib').handlers = []
    #
    # Configure the logger
    #
    if options.runtime is None:
        options.runtime = Options()
    if options.runtime.logging == 'quiet':
        logging.getLogger('pyomo.opt').setLevel(logging.ERROR)
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        logging.getLogger('pyomo').setLevel(logging.ERROR)
        logging.getLogger('pyutilib').setLevel(logging.ERROR)
    elif options.runtime.logging == 'warning':
        logging.getLogger('pyomo.opt').setLevel(logging.WARNING)
        logging.getLogger('pyomo.core').setLevel(logging.WARNING)
        logging.getLogger('pyomo').setLevel(logging.WARNING)
        logging.getLogger('pyutilib').setLevel(logging.WARNING)
    elif options.runtime.logging == 'info':
        logging.getLogger('pyomo.opt').setLevel(logging.INFO)
        logging.getLogger('pyomo.core').setLevel(logging.INFO)
        logging.getLogger('pyomo').setLevel(logging.INFO)
        logging.getLogger('pyutilib').setLevel(logging.INFO)
    elif options.runtime.logging == 'verbose':
        logger.setLevel(logging.DEBUG)
        logging.getLogger('pyomo').setLevel(logging.DEBUG)
        logging.getLogger('pyutilib').setLevel(logging.DEBUG)
    elif options.runtime.logging == 'debug':
        logging.getLogger('pyomo.opt').setLevel(logging.DEBUG)
        logging.getLogger('pyomo.core').setLevel(logging.DEBUG)
        logging.getLogger('pyomo').setLevel(logging.DEBUG)
        logging.getLogger('pyutilib').setLevel(logging.DEBUG)
    if options.runtime.logfile:
        logging.getLogger('pyomo.opt').handlers = []
        logging.getLogger('pyomo.core').handlers = []
        logging.getLogger('pyomo').handlers = []
        logging.getLogger('pyutilib').handlers = []
        logging.getLogger('pyomo.core').addHandler( logging.FileHandler(options.runtime.logfile, 'w'))
        logging.getLogger('pyomo').addHandler( logging.FileHandler(options.runtime.logfile, 'w'))
        logging.getLogger('pyutilib').addHandler( logging.FileHandler(options.runtime.logfile, 'w'))

@pyomo_api(namespace='pyomo.script')
def run_command(command=None, parser=None, args=None, name='unknown', data=None, options=None):
    """
    Execute a function that processes command-line arguments and
    then calls a command-line driver.

    This function provides a generic facility for executing a command
    function is rather generic.  This function is segregated from
    the driver to enable profiling of the command-line execution.

    Required:
        command:    The name of a function that will be executed to perform process the command-line
                    options with a parser object.
        parser:     The parser object that is used by the command-line function.

    Optional:
        options:    If this is not None, then ignore the args option and use
                    this to specify command options.
        args:       Command-line arguments that are parsed.  If this value is `None`, then the
                    arguments in `sys.argv` are used to parse the command-line.
        name:       Specifying the name of the command-line (for error messages).
        data:       A container of labeled data.

    Returned:
        retval:     Return values from the command-line execution.
        errorcode:  0 if Pyomo ran successfully
    """
    #
    #
    # Parse command-line options
    #
    #
    retval = None
    errorcode = 0
    if options is None:
        try:
            if type(args) is argparse.Namespace:
                _options = args
            else:
                _options = parser.parse_args(args=args)
            # Replace the parser options object with a pyutilib.misc.Options object
            options = pyutilib.misc.Options()
            for key in dir(_options):
                if key[0] != '_':
                    val = getattr(_options, key)
                    if not isinstance(val, types.MethodType):
                        options[key] = val
        except SystemExit:
            # the parser throws a system exit if "-h" is specified - catch
            # it to exit gracefully.
            return Container(retval=retval, errorcode=errorcode)
    #
    # Configure loggers
    #
    configure_loggers(options=options)
    #
    # Setup I/O redirect to a file
    #
    logfile = options.runtime.logfile
    if not logfile is None:
        pyutilib.misc.setup_redirect(logfile)
    #
    # Call the main Pyomo runner with profiling
    #
    TempfileManager.push()
    pcount = options.runtime.profile_count
    if pcount > 0:
        if not pstats_available:
            if not logfile is None:
                pyutilib.misc.reset_redirect()
            msg = "Cannot use the 'profile' option.  The Python 'pstats' "    \
                  'package cannot be imported!'
            raise ValueError(msg)
        tfile = TempfileManager.create_tempfile(suffix=".profile")
        tmp = profile.runctx(
          command.__name__ + '(options=options,parser=parser)', command.__globals__, locals(), tfile
        )
        p = pstats.Stats(tfile).strip_dirs()
        p.sort_stats('time', 'cumulative')
        p = p.print_stats(pcount)
        p.print_callers(pcount)
        p.print_callees(pcount)
        p = p.sort_stats('cumulative','calls')
        p.print_stats(pcount)
        p.print_callers(pcount)
        p.print_callees(pcount)
        p = p.sort_stats('calls')
        p.print_stats(pcount)
        p.print_callers(pcount)
        p.print_callees(pcount)
        retval = tmp
    else:
        #
        # Call the main Pyomo runner without profiling
        #
        TempfileManager.push()
        try:
            retval = command(options=options, parser=parser)
        except SystemExit:
            err = sys.exc_info()[1]
            #
            # If debugging is enabled or the 'catch' option is specified, then
            # exit.  Otherwise, print an "Exiting..." message.
            #
            if __debug__ and (options.runtime.logging == 'debug' or options.runtime.catch_errors):
                sys.exit(0)
            print('Exiting %s: %s' % (name, str(err)))
            errorcode = err.code
        except Exception:
            err = sys.exc_info()[1]
            #
            # If debugging is enabled or the 'catch' option is specified, then
            # pass the exception up the chain (to pyomo_excepthook)
            #
            if __debug__ and (options.runtime.logging == 'debug' or options.runtime.catch_errors):
                if not logfile is None:
                    pyutilib.misc.reset_redirect()
                TempfileManager.pop(remove=not options.runtime.keep_files)
                raise

            if not options.model is None and not options.model.save_file is None:
                model = "model " + options.model.save_file
            else:
                model = "model"

            global filter_excepthook
            if filter_excepthook:
                action = "loading"
            else:
                action = "running"

            msg = "Unexpected exception while %s %s:\n" % (action, model)
            #
            # This handles the case where the error is propagated by a KeyError.
            # KeyError likes to pass raw strings that don't handle newlines
            # (they translate "\n" to "\\n"), as well as tacking on single
            # quotes at either end of the error message. This undoes all that.
            #
            errStr = str(err)
            if type(err) == KeyError and errStr != "None":
                errStr = str(err).replace(r"\n","\n")[1:-1]

            logging.getLogger('pyomo.core').error(msg+errStr)
            errorcode = 1

    if not logfile is None:
        pyutilib.misc.reset_redirect()

    if options.runtime.disable_gc:
        gc.enable()
    TempfileManager.pop(remove=not options.runtime.keep_files)
    return Container(retval=retval, errorcode=errorcode)

def cleanup():
    for key in modelapi:
        for ep in ExtensionPoint(modelapi[key]):
            ep.deactivate()

def get_config_values(filename):
    if filename.endswith('.yml') or filename.endswith('.yaml'):
        if not yaml_available:
            raise ValueError("ERROR: yaml configuration file specified, but pyyaml is not installed!")
        INPUT = open(filename, 'r')
        val = yaml.load(INPUT)
        INPUT.close()
        return val
    elif filename.endswith('.jsn') or filename.endswith('.json'):
        INPUT = open(filename, 'r')
        val = json.load(INPUT)
        INPUT.close()
        return val
    raise IOError("ERROR: Unexpected configuration file '%s'" % filename)


