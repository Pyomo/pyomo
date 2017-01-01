#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import glob
import pyutilib.dev.runtests
import sys
import os.path
import optparse
import pyomo.util

@pyomo.util.pyomo_command('test.pyomo', "Execute Pyomo tests")
def runPyomoTests():
    parser = optparse.OptionParser(usage='test.pyomo [options] <dirs>')

    parser.add_option('-d','--dir',
        action='store',
        dest='dir',
        default=None,
        help='Top-level source directory where the tests are applied.')
    parser.add_option('-e','--exclude',
        action='append',
        dest='exclude',
        default=[],
        help='Top-level source directories that are excluded.')
    parser.add_option('--cat','--category',
        action='store',
        dest='cat',
        default='smoke',
        help='Specify test category.')
    parser.add_option('--cov','--coverage',
        action='store_true',
        dest='coverage',
        default=False,
        help='Indicate that coverage information is collected')
    parser.add_option('-v','--verbose',
        action='store_true',
        dest='verbose',
        default=False,
        help='Verbose output')
    parser.add_option('-o','--output',
        action='store',
        dest='output',
        default=None,
        help='Redirect output to a file')

    _options, args = parser.parse_args(sys.argv)

    if _options.output:
        outfile = os.path.abspath(_options.output)
    else:
        outfile = None
    if _options.dir is None:
        # the /src directory (for development installations)
        os.chdir( os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) )
    else:
        if os.path.exists(_options.dir):
            os.chdir( _options.dir )

    print("Running tests in directory %s" % os.getcwd())
    _options.cat = os.environ.get('PYUTILIB_UNITTEST_CATEGORY', _options.cat)
    if _options.cat == 'all':
        if 'PYUTILIB_UNITTEST_CATEGORY' in os.environ:
            del os.environ['PYUTILIB_UNITTEST_CATEGORY']
    elif _options.cat:
        os.environ['PYUTILIB_UNITTEST_CATEGORY'] = _options.cat
        print(" ... for test category: %s" % os.environ['PYUTILIB_UNITTEST_CATEGORY'])

    options=[]
    if _options.coverage:
        options.append('--coverage')
    if _options.verbose:
        options.append('-v')
    if outfile:
        options.append('-o')
        options.append(outfile)

    if len(args) > 1:
        mydirs = args[1:]
    else:
        mydirs = [os.path.join('pyomo','pyomo'), 'pyomo-model-libraries']
    #
    dirs=[]
    for dir in mydirs:
        if dir in _options.exclude:
            continue
        if dir.startswith('-'):
            options.append(dir)
        if dir.startswith('pyomo'):
            if os.path.exists(dir):
                dirs.append(dir)
            elif '.' in dir:
                dirs.append(os.path.join('pyomo','pyomo',dir.split('.')[1]))
        else:
            if os.path.exists('pyomo.'+dir):
                dirs.append('pyomo.'+dir)
            else:
                dirs.append(os.path.join('pyomo','pyomo',dir))
    #
    excluding = set()
    for e in _options.exclude:
        excluding.add(os.path.join('pyomo','pyomo',e))
    testdirs = []
    for topdir in dirs:
        for root, subdirs, files in os.walk(topdir):
            if not '__init__.py' in files:
                # Skip directories that do not contain a __init__.py file.
                continue
            for f in files:
                if f.startswith("test"):
                    skip=False
                    for e in excluding:
                        if root.startswith(e):
                            skip=True
                    if not skip:
                        testdirs.append(root)
                    break
    #
    return pyutilib.dev.runtests.run('pyomo', ['runtests']+options+['-p','pyomo']+testdirs)
