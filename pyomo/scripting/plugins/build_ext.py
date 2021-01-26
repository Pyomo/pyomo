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
import sys

from pyomo.common.extensions import ExtensionBuilderFactory
from pyomo.scripting.pyomo_parser import add_subparser

class ExtensionBuilder(object):
    def create_parser(self, parser):
        return parser

    def call(self, args, unparsed):
        logger = logging.getLogger('pyomo.common')
        original_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            return self._call_impl(args, unparsed, logger)
        finally:
            logger.setLevel(original_level)

    def _call_impl(self, args, unparsed, logger):
        results = []
        result_fmt = "[%s]  %s"
        returncode = 0
        for target in ExtensionBuilderFactory:
            try:
                ext = ExtensionBuilderFactory(target)
                if hasattr(ext, 'skip') and ext.skip():
                    result = 'SKIP'
                elif hasattr(ext, '__call__'):
                    ext(parallel=args.parallel)
                    result = ' OK '
                else:
                    # Extension was a simple function and already ran
                    result = ' OK '
            except SystemExit:
                _info = sys.exc_info()
                _cls = str(_info[0].__name__ if _info[0] is not None
                           else "NoneType") + ": "
                logger.error(_cls + str(_info[1]))
                result = 'FAIL'
                returncode |= 2
            except:
                _info = sys.exc_info()
                _cls = str(_info[0].__name__ if _info[0] is not None
                           else "NoneType") + ": "
                logger.error(_cls + str(_info[1]))
                result = 'FAIL'
                returncode |= 1
            results.append(result_fmt % (result, target))
        logger.info("Finished building Pyomo extensions.")
        logger.info(
            "The following extensions were built:\n    " +
            "\n    ".join(results))
        return returncode

#
# Add a subparser for the download-extensions command
#
_extension_builder = ExtensionBuilder()
_parser = _extension_builder.create_parser(
    add_subparser(
        'build-extensions',
        func=_extension_builder.call,
        help='Build compiled extension modules',
        add_help=False,
        description='This builds all registered (compileable) extension modules'
    ))

_parser.add_argument(
    '-j', '--parallel',
    action='store',
    type=int,
    dest='parallel',
    default=None,
    help="Build with this many processes/cores",
    )
