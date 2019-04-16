#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.extensions import ExtensionBuilderFactory
from pyomo.scripting.pyomo_parser import add_subparser

class ExtensionBuilder(object):
    def create_parser(self, parser):
        return parser

    def call(self, args, unparsed):
        for target in ExtensionBuilderFactory:
            ExtensionBuilderFactory(target)


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

