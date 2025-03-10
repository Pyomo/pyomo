#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.config import ConfigBlock, ConfigList, ConfigValue


class Default_Config(object):
    def config_block(self, init=False):
        config, blocks = minlp_config_block(init=init)
        return config, blocks


def minlp_config_block(init=False):
    config = ConfigBlock(
        "Configuration for a canonical model construction and optimization sequence"
    )
    blocks = {}

    #
    # Data
    #
    data = config.declare('data', ConfigBlock())
    blocks['data'] = data

    data.declare(
        'files',
        ConfigList(
            [], ConfigValue(None, str, 'Filename', None), 'Model data files', None
        ),
    )
    data.declare(
        'namespaces',
        ConfigList(
            [],
            ConfigValue(None, str, 'Namespace', None),
            'A namespace that is used to select data in Pyomo data files.',
            None,
        ),
    ).declare_as_argument('--namespace', dest='namespaces', action='append')

    #
    # Model
    #
    model = config.declare('model', ConfigBlock())
    blocks['model'] = model

    model.declare(
        'filename',
        ConfigValue(None, str, 'The Python module that specifies the model', None),
    )
    model.declare(
        'object name',
        ConfigValue(
            None,
            str,
            'The name of the model object that is created in the '
            'specified Pyomo module',
            None,
        ),
    ).declare_as_argument('--model-name', dest='model_name')
    model.declare('type', ConfigValue(None, str, 'The problem type', None))
    model.declare(
        'options',
        ConfigBlock(implicit=True, description='Options used to construct the model'),
    )
    model.declare(
        'linearize expressions',
        ConfigValue(
            False,
            bool,
            'An option intended for use on linear or mixed-integer models '
            'in which expression trees in a model (constraints or objectives) '
            'are compacted into a more memory-efficient and concise form.',
            None,
        ),
    )
    model.declare(
        'save file',
        ConfigValue(
            None,
            str,
            'The filename to which the model is saved. The suffix of this '
            'filename specifies the file format.',
            None,
        ),
    )
    model.declare(
        'save format',
        ConfigValue(
            None,
            str,
            "The format that the model is saved. When specified, this "
            "overrides the format implied by the 'save file' option.",
            None,
        ),
    )
    model.declare(
        'symbolic solver labels',
        ConfigValue(
            False,
            bool,
            'When interfacing with the solver, use symbol names derived '
            'from the model. For example, "my_special_variable[1_2_3]" '
            'instead of "v1". Useful for debugging. When using the ASL '
            'interface (--solver-io=nl), generates corresponding .row '
            '(constraints) and .col (variables) files. The ordering in '
            'these files provides a mapping from ASL index to symbolic '
            'model names.',
            None,
        ),
    ).declare_as_argument(dest='symbolic_solver_labels')
    model.declare(
        'file determinism',
        ConfigValue(
            None,
            int,
            'When interfacing with a solver using file based I/O, set '
            'the effort level for ensuring the file creation process is '
            'determistic. See the individual solver interfaces for '
            'valid values and default level of file determinism.',
            None,
        ),
    ).declare_as_argument(dest='file_determinism')

    #
    # Transform
    #
    transform = ConfigBlock()
    blocks['transform'] = transform

    transform.declare(
        'name', ConfigValue(None, str, 'Name of the model transformation', None)
    )
    transform.declare(
        'options', ConfigBlock(implicit=True, description='Transformation options')
    )
    #
    transform_list = config.declare(
        'transform',
        ConfigList(
            [],
            ConfigValue(None, str, 'Transformation', None),
            'List of model transformations',
            None,
        ),
    ).declare_as_argument(dest='transformations', action='append')
    if init:
        transform_list.append()

    #
    # Preprocess
    #
    config.declare(
        'preprocess',
        ConfigList(
            [],
            ConfigValue(None, str, 'Module', None),
            'Specify a Python module that gets immediately executed '
            '(before the optimization model is setup).',
            None,
        ),
    ).declare_as_argument(dest='preprocess')

    #
    # Runtime
    #
    runtime = config.declare('runtime', ConfigBlock())
    blocks['runtime'] = runtime

    runtime.declare(
        'logging',
        ConfigValue(
            None, str, 'Logging level:  quiet, warning, info, verbose, debug', None
        ),
    ).declare_as_argument(dest="logging", metavar="LEVEL")
    runtime.declare(
        'logfile',
        ConfigValue(None, str, 'Redirect output to the specified file.', None),
    ).declare_as_argument(dest="output", metavar="FILE")
    runtime.declare(
        'catch errors',
        ConfigValue(
            False,
            bool,
            'Trigger failures for exceptions to print the program stack.',
            None,
        ),
    ).declare_as_argument('-c', '--catch-errors', dest="catch")
    runtime.declare(
        'disable gc', ConfigValue(False, bool, 'Disable the garbage collector.', None)
    ).declare_as_argument('--disable-gc', dest='disable_gc')
    runtime.declare(
        'interactive',
        ConfigValue(
            False,
            bool,
            'After executing Pyomo, launch an interactive Python shell. '
            'If IPython is installed, this shell is an IPython shell.',
            None,
        ),
    )
    runtime.declare(
        'keep files', ConfigValue(False, bool, 'Keep temporary files', None)
    ).declare_as_argument('-k', '--keepfiles', dest='keepfiles')
    runtime.declare(
        'paths',
        ConfigList(
            [],
            ConfigValue(None, str, 'Path', None),
            'Give a path that is used to find the Pyomo python files.',
            None,
        ),
    ).declare_as_argument('--path', dest='path')
    runtime.declare(
        'profile count',
        ConfigValue(
            0,
            int,
            'Enable profiling of Python code. The value of this option '
            'is the number of functions that are summarized.',
            None,
        ),
    ).declare_as_argument(dest='profile_count', metavar='COUNT')
    runtime.declare(
        'profile memory',
        ConfigValue(
            0,
            int,
            "Report memory usage statistics for the generated instance "
            "and any associated processing steps. A value of 0 indicates "
            "disabled. A value of 1 forces the print of the total memory "
            "after major stages of the pyomo script. A value of 2 forces "
            "summary memory statistics after major stages of the pyomo "
            "script. A value of 3 forces detailed memory statistics "
            "during instance creation and various steps of preprocessing. "
            "Values equal to 4 and higher currently provide no additional "
            "information. Higher values automatically enable all "
            "functionality associated with lower values, e.g., 3 turns "
            "on detailed and summary statistics.",
            None,
        ),
    )
    runtime.declare(
        'report timing',
        ConfigValue(
            False,
            bool,
            'Report various timing statistics during model construction.',
            None,
        ),
    ).declare_as_argument(dest='report_timing')
    runtime.declare(
        'tempdir',
        ConfigValue(
            None,
            str,
            'Specify the directory where temporary files are generated.',
            None,
        ),
    ).declare_as_argument(dest='tempdir')

    return config, blocks


def default_config_block(solver, init=False):
    config, blocks = Default_Config().config_block(init)

    #
    # Solver
    #
    solver = ConfigBlock()
    solver.declare('solver name', ConfigValue('glpk', str, 'Solver name', None))
    solver.declare(
        'solver executable',
        ConfigValue(
            default=None,
            domain=str,
            description="The solver executable used by the solver interface.",
            doc=(
                "The solver executable used by the solver interface. "
                "This option is only valid for those solver interfaces that "
                "interact with a local executable through the shell. If unset, "
                "the solver interface will attempt to find an executable within "
                "the search path of the shell's environment that matches a name "
                "commonly associated with the solver interface."
            ),
        ),
    )
    solver.declare(
        'io format',
        ConfigValue(
            None,
            str,
            'The type of IO used to execute the solver. Different solvers '
            'support different types of IO, but the following are common '
            'options: lp - generate LP files, nl - generate NL files, '
            'python - direct Python interface, os - generate OSiL XML files.',
            None,
        ),
    )
    solver.declare(
        'manager',
        ConfigValue(
            'serial',
            str,
            'The technique that is used to manage solver executions.',
            None,
        ),
    )
    solver.declare(
        'options',
        ConfigBlock(
            implicit=True,
            implicit_domain=ConfigValue(None, str, 'Solver option', None),
            description="Options passed into the solver",
        ),
    )
    solver.declare(
        'options string',
        ConfigValue(None, str, 'String describing solver options', None),
    )
    solver.declare(
        'suffixes',
        ConfigList(
            [],
            ConfigValue(None, str, 'Suffix', None),
            'Solution suffixes that will be extracted by the solver '
            '(e.g., rc, dual, or slack). The use of this option is not '
            'required when a suffix has been declared on the model '
            'using Pyomo\'s Suffix component.',
            None,
        ),
    )
    blocks['solver'] = solver
    #
    solver_list = config.declare(
        'solvers',
        ConfigList(
            [],
            solver,  # ConfigValue(None, str, 'Solver', None),
            'List of solvers.  The first solver in this list is the main solver.',
            None,
        ),
    )
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
    # solver_list[0].get('solver name').\
    #    declare_as_argument('--solver', dest='solver')
    solver_list[0].get('solver executable').declare_as_argument(
        '--solver-executable', dest="solver_executable", metavar="FILE"
    )
    solver_list[0].get('io format').declare_as_argument(
        '--solver-io', dest='io_format', metavar="FORMAT"
    )
    solver_list[0].get('manager').declare_as_argument(
        '--solver-manager', dest="smanager_type", metavar="TYPE"
    )
    solver_list[0].get('options string').declare_as_argument(
        '--solver-options', dest='options_string', metavar="STRING"
    )
    solver_list[0].get('suffixes').declare_as_argument(
        '--solver-suffix', dest="solver_suffixes"
    )

    #
    # Postprocess
    #
    config.declare(
        'postprocess',
        ConfigList(
            [],
            ConfigValue(None, str, 'Module', None),
            'Specify a Python module that gets executed after optimization.',
            None,
        ),
    ).declare_as_argument(dest='postprocess')

    #
    # Postsolve
    #
    postsolve = config.declare('postsolve', ConfigBlock())
    blocks['postsolve'] = postsolve

    postsolve.declare(
        'print logfile',
        ConfigValue(
            False, bool, 'Print the solver logfile after performing optimization.', None
        ),
    ).declare_as_argument('-l', '--log', dest="log")
    postsolve.declare(
        'save results',
        ConfigValue(
            None, str, 'Specify the filename to which the results are saved.', None
        ),
    ).declare_as_argument('--save-results', dest="save_results", metavar="FILE")
    postsolve.declare(
        'show results',
        ConfigValue(False, bool, 'Print the results object after optimization.', None),
    ).declare_as_argument(dest="show_results")
    postsolve.declare(
        'results format',
        ConfigValue(None, str, 'Specify the results format:  json or yaml.', None),
    ).declare_as_argument(
        '--results-format', dest="results_format", metavar="FORMAT"
    ).declare_as_argument(
        '--json',
        dest="results_format",
        action="store_const",
        const="json",
        help="Store results in JSON format",
    )
    postsolve.declare(
        'summary',
        ConfigValue(
            False,
            bool,
            'Summarize the final solution after performing optimization.',
            None,
        ),
    ).declare_as_argument(dest="summary")

    #
    # Runtime
    #
    runtime = blocks['runtime']
    runtime.declare(
        'only instance',
        ConfigValue(False, bool, "Generate a model instance, and then exit", None),
    ).declare_as_argument('--instance-only', dest='only_instance')
    runtime.declare(
        'stream output',
        ConfigValue(
            False,
            bool,
            "Stream the solver output to provide information about the "
            "solver's progress.",
            None,
        ),
    ).declare_as_argument('--stream-output', '--stream-solver', dest="tee")
    #
    return config, blocks
