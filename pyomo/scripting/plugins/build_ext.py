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
from six import iteritems

from pyomo.common.extensions import ExtensionBuilderFactory
from pyomo.scripting.pyomo_parser import add_subparser

class ExtensionBuilder(object):
    def create_parser(self, parser):
        return parser

    def call(self, args, unparsed):
        logger = logging.getLogger('pyomo.common')
        logger.setLevel(logging.INFO)
        results = {}
        returncode = 0
        for target in ExtensionBuilderFactory:
            try:
                ExtensionBuilderFactory(target)
                results[target] = ' OK '
            except SystemExit:
                results[target] = 'FAIL'
                returncode = 1
            except:
                results[target] = 'FAIL'
                returncode = 1
        logger.info("Finished building Pyomo extensions.")
        logger.info(
            "The following extensions were built:\n    " +
            "\n    ".join(["[%s]  %s" % (v,k) for k,v in iteritems(results)]))
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

