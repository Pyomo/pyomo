#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.config import ConfigBlock, ConfigList, ConfigValue
from pyomo.opt.base.problem import ProblemConfigFactory, BaseProblemConfig


@ProblemConfigFactory.register('default')
class Default_Config(BaseProblemConfig):

    def config_block(self, init=False):
        config, blocks = minlp_config_block(init=init)
        return config, blocks


def minlp_config_block(init=False):
    config = ConfigBlock("Configuration for a canonical model construction and optimization sequence")
    blocks={}
       
    #
    # Data
    #
    data = config.declare('data', ConfigBlock())
    data.declare('files', ConfigList(
                [],
                ConfigValue(None, str, 'Filename', None),
                'Model data files',
                None) )
    data.declare('namespaces', ConfigList(
                [],
                ConfigValue(None, str, 'Namespace', None),
                'A namespace that is used to select data in Pyomo data files.',
                None) ).declare_as_argument('--namespace', dest='namespaces', action='append')
    blocks['data'] = data

    #
    # Model
    #
    model = config.declare('model', ConfigBlock())
    model.declare('filename', ConfigValue(
                None,
                str,
                'The Python module that specifies the model',
                None ) )
    model.declare('object name', ConfigValue(
                None, 
                str,
                'The name of the model object that is created in the specified Pyomo module',
                None ) ).declare_as_argument('--model-name', dest='model_name') 
    model.declare('type', ConfigValue(
                None,
                str,
                'The problem type',
                None ) )
    model.declare('options', ConfigBlock(
                implicit=True,
                description='Options used to construct the model') )
    model.declare('linearize expressions', ConfigValue(
                False,
                bool,
                'An option intended for use on linear or mixed-integer models in which expression trees in a model (constraints or objectives) are compacted into a more memory-efficient and concise form.',
                None) )
    model.declare('save file', ConfigValue(
                None,
                str,
                "The filename to which the model is saved. The suffix of this filename specifies the file format.",
                None) )
    model.declare('save format', ConfigValue(
                None,
                str,
                "The format that the model is saved. When specified, this overrides the format implied by the 'save file' option.",
                None) )
    model.declare('symbolic solver labels', ConfigValue(
                False,
                bool,
                'When interfacing with the solver, use symbol names derived from the model. For example, \"my_special_variable[1_2_3]\" instead of \"v1\". Useful for debugging. When using the ASL interface (--solver-io=nl), generates corresponding .row (constraints) and .col (variables) files. The ordering in these files provides a mapping from ASL index to symbolic model names.',
                None) ).declare_as_argument(dest='symbolic_solver_labels')
    model.declare('file determinism', ConfigValue(
                1,
                int,
                'When interfacing with a solver using file based I/O, set the effort level for ensuring the file creation process is determistic. The default (1) sorts the index of components when transforming the model. Anything less than 1 disables index sorting. Anything greater than 1 additionaly sorts by component name to override declartion order.',
                None) ).declare_as_argument(dest='file_determinism')
    blocks['model'] = model

    #
    # Transform
    #
    transform = ConfigBlock()
    transform.declare('name', ConfigValue(
                None, 
                str,
                'Name of the model transformation',
                None ) )
    transform.declare('options', ConfigBlock(
                implicit=True,
                description='Transformation options'
                ) )
    blocks['transform'] = transform
    #
    transform_list = config.declare('transform', ConfigList(
                [],
                ConfigValue(None, str, 'Transformation', None),
                'List of model transformations',
                None) ).declare_as_argument( dest='transformations',
                                             action='append' )
    if init:
        transform_list.append()

    #
    # Preprocess
    #
    config.declare('preprocess', ConfigList(
                [],
                ConfigValue(None, str, 'Module', None),
                'Specify a Python module that gets immediately executed (before the optimization model is setup).',
                None) ).declare_as_argument(dest='preprocess')

    #
    # Runtime
    #
    runtime = config.declare('runtime', ConfigBlock())
    runtime.declare('logging', ConfigValue(
                None, 
                str,
                'Logging level:  quiet, warning, info, verbose, debug',
                None) ).declare_as_argument(dest="logging", metavar="LEVEL")
    runtime.declare('logfile', ConfigValue(
                None, 
                str,
                'Redirect output to the specified file.',
                None) ).declare_as_argument(dest="output", metavar="FILE")
    runtime.declare('catch errors', ConfigValue(
                False, 
                bool,
                'Trigger failures for exceptions to print the program stack.',
                None) ).declare_as_argument('-c', '--catch-errors', dest="catch")
    runtime.declare('disable gc', ConfigValue(
                False, 
                bool,
                'Disable the garbage collecter.',
                None) ).declare_as_argument('--disable-gc', dest='disable_gc')
    runtime.declare('interactive', ConfigValue(
                False, 
                bool,
                'After executing Pyomo, launch an interactive Python shell. If IPython is installed, this shell is an IPython shell.',
                None) )
    runtime.declare('keep files', ConfigValue(
                False, 
                bool,
                'Keep temporary files',
                None) ).declare_as_argument('-k', '--keepfiles', dest='keepfiles')
    runtime.declare('paths', ConfigList(
                [], 
                ConfigValue(None, str, 'Path', None),
                'Give a path that is used to find the Pyomo python files.',
                None) ).declare_as_argument('--path', dest='path')
    runtime.declare('profile count', ConfigValue(
                0, 
                int,
                'Enable profiling of Python code. The value of this option is the number of functions that are summarized.',
                None) ).declare_as_argument(dest='profile_count', metavar='COUNT')
    runtime.declare('profile memory', ConfigValue(
                0, 
                int,
                "Report memory usage statistics for the generated instance and any associated processing steps. A value of 0 indicates disabled. A value of 1 forces the print of the total memory after major stages of the pyomo script. A value of 2 forces summary memory statistics after major stages of the pyomo script. A value of 3 forces detailed memory statistics during instance creation and various steps of preprocessing. Values equal to 4 and higher currently provide no additional information. Higher values automatically enable all functionality associated with lower values, e.g., 3 turns on detailed and summary statistics.",
                None) )
    runtime.declare('report timing', ConfigValue(
                False, 
                bool,
                'Report various timing statistics during model construction.',
                None) ).declare_as_argument(dest='report_timing')
    runtime.declare('tempdir', ConfigValue(
                None, 
                str,
                'Specify the directory where temporary files are generated.',
                None) ).declare_as_argument(dest='tempdir')
    blocks['runtime'] = runtime
    #
    return config, blocks

