#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import argparse
import pyomo.scripting.pyomo_parser
import os.path

class EnableDisableAction(argparse.Action):

    def add_package(self, namespace, package):
        if namespace.checkers.get(package, None) is None:
            namespace.checkers[package] = []
        for c in pyomo.checker.runner.ModelCheckRunner._checkers(all=True):
            if c._checkerPackage() == package:
                namespace.checkers[package].append(c._checkerName())

    def remove_package(self, namespace, package):
        if package in namespace.checkers:
            del namespace.checkers[package]

    def add_checker(self, namespace, checker):
        for c in pyomo.checker.runner.ModelCheckRunner._checkers(all=True):
            if c._checkerName() == checker:
                if namespace.checkers.get(c._checkerPackage(), None) is None:
                    namespace.checkers[c._checkerPackage()] = []
                if c._checkerName() not in namespace.checkers[c._checkerPackage()]:
                    namespace.checkers[c._checkerPackage()].append(c._checkerName())

    def remove_checker(self, namespace, checker):
        for c in pyomo.core.check.ModelCheckRunner._checkers(all=True):
            if c._checkerName() == checker:
                if namespace.checkers.get(c._checkerPackage(), None) is not None:
                    for i in range(namespace.checkers[c._checkerPackage()].count(c._checkerName())):
                        namespace.checkers[c._checkerPackage()].remove(c._checkerName())

    def add_default_checkers(self, namespace):
        self.add_package(namespace, 'model')
        self.add_package(namespace, 'py3k')

    def __call__(self, parser, namespace, values, option_string=None):
        if 'checkers' not in dir(namespace):
            setattr(namespace, 'checkers', {})
            self.add_default_checkers(namespace)
        
        if option_string == '-c':
            self.add_checker(namespace, values)
        elif option_string == '-C':
            self.add_package(namespace, values)
        elif option_string == '-x':
            self.remove_checker(namespace, values)
        elif option_string == '-X':
            self.remove_package(namespace, values)

def setup_parser(parser):
    parser.add_argument("script", metavar="SCRIPT", default=None,
                        help="A Pyomo script that is checked")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Enable additional output messages")
    parser.add_argument("-c", "--enable-checker", metavar="CHECKER", action=EnableDisableAction, 
                        help="Activate a specific checker")
    parser.add_argument("-C", "--enable-package", metavar="PACKAGE", action=EnableDisableAction,
                        help="Activate an entire checker package")
    parser.add_argument("-x", "--disable-checker", metavar="CHECKER", action=EnableDisableAction,
                        help="Disable a specific checker")
    parser.add_argument("-X", "--disable-package", metavar="PACKAGE", action=EnableDisableAction,
                        help="Disable an entire checker package")


def main_exec(options):
    import pyomo.checker.runner as check

    if options.script is None:
        raise IOError("Must specify a model script!")
    if not os.path.exists(options.script):
        raise IOError("Model script '%s' does not exist!" % options.script)

    # force default checkers
    if getattr(options, 'checkers', None) is None:
        EnableDisableAction(None, None)(None, options, None, None)

    runner = check.ModelCheckRunner()
    runner.run(**vars(options))

#
# Add a subparser for the check command
#
setup_parser(
    pyomo.scripting.pyomo_parser.add_subparser('check',
        func=main_exec, 
        help='Check a model for errors.',
        description='This pyomo subcommand is used to check a model script for errors.',
        epilog="""
The default behavior of this command is to assume that the model
script is a simple Pyomo model.  Eventually, this script will support
options that allow other Pyomo models to be checked.
"""
        ))
