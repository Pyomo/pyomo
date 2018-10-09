#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import glob
import sys
import os
import subprocess
from subprocess import check_output as _run_cmd
import driver

config = sys.argv[1]
hname = os.uname()[1]
hname = hname.split('.')[0]

print("\nStarting jenkins.py")
print("Configuration=%s" % config)

if os.environ.get('WORKSPACE', None) is None:
    sys.stdout.write(
        "\n(INFO) WORKSPACE environment vatiable not found."
        "\n       Assuming WORKSPACE==%s\n\n" % (os.getcwd(),) )
    os.environ['WORKSPACE'] = os.getcwd()

os.environ['CONFIGFILE'] = os.environ['WORKSPACE']+'/src/pyomo/admin/config.ini'
#
# Is the following even needed?
#
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] += os.environ['WORKSPACE'] + os.pathsep + os.environ['PYTHONPATH']
else:
    os.environ['PYTHONPATH'] = os.environ['WORKSPACE']
sys.path.append(os.environ['WORKSPACE'])

sys.argv = ['dummy', '--trunk', '--source', 'src', '-a', 'pyyaml']

#
# Machine-specific configurations
#
#if hname == "snotra":
#    ### snotra configuration is now handled through local module files
#


if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = ""

print("\nPython version: %s" % sys.version)
print("\nSystem PATH:\n\t%s" % os.environ['PATH'])
print("\nPython path:\n\t%s" % sys.path)

coverage_omit=','.join([
    os.sep.join([os.environ['WORKSPACE'], 'src', 'pyomo', 'pyomo', '*', 'tests']),
    'pyomo.*.tests',
    os.sep.join([os.environ['WORKSPACE'], 'src', 'pyutilib.*']),
    'pyutilib.*',
])

pyomo_packages = [
    'pyomo.%s' % os.path.basename(x) for x in
    glob.glob(os.path.join(
            os.environ['WORKSPACE'], 'src', 'pyomo', 'pyomo', '*' ))
    if os.path.isdir(x) ]

if config == "notests":
    driver.perform_install('pyomo', config='pyomo_all.ini')

elif config == "default":
    driver.perform_build('pyomo', coverage=True, omit=coverage_omit, config='pyomo_all.ini')

elif config == "core":
    # Install
    print("-" * 60)
    print("Installing Pyomo")
    print("-" * 60)
    driver.perform_install('pyomo', config='pyomo_all.ini')
    print("-" * 60)
    print("Running 'pyomo install-extras' ...")
    print("-" * 60)
    if _run_cmd is subprocess.check_call:
        _run_cmd("python/bin/pyomo install-extras", shell=True)
    elif _run_cmd is subprocess.check_output:
        output = _run_cmd("python/bin/pyomo install-extras", shell=True)
        if hasattr(output, 'encode'):
            output = output.encode('utf-8','replace')
        print(output.decode('utf-8'))
    else:
        assert False
    # Test
    os.environ['TEST_PACKAGES'] = ' '.join([
            'pyomo.checker','pyomo.common','pyomo.core','pyomo.environ',
            'pyomo.opt','pyomo.repn','pyomo.scripting','pyomo.solvers',
            'pyomo.util','pyomo.version'])
    print("-" * 60)
    print("Performing tests")
    print("-" * 60)
    driver.perform_tests('pyomo', coverage=True, omit=coverage_omit)

elif config == "nonpysp":
    os.environ['TEST_PACKAGES'] = ' '.join(
        x for x in pyomo_packages if x != 'pyomo.pysp' )
    driver.perform_build('pyomo', coverage=True, omit=coverage_omit, config='pyomo_all.ini')

elif config == "parallel":
    os.environ['NOSE_PROCESS_TIMEOUT'] = '1800' # 30 minutes
    driver.perform_build('pyomo', cat='parallel', coverage=True, omit=coverage_omit, config='pyomo_all.ini')

elif config == "expensive":
    driver.perform_build('pyomo',
        cat='expensive', coverage=True, omit=coverage_omit,
        virtualenv_args=sys.argv[1:])

elif config == "booktests" or config == "book":
    # Install
    driver.perform_install('pyomo', config='pyomo_all.ini')
    print("Running 'pyomo install-extras' ...")
    if _run_cmd is subprocess.check_call:
        output = _run_cmd("python/bin/python src/pyomo/scripts/get_pyomo_extras.py -v", shell=True)
    elif _run_cmd is subprocess.check_output:
        output = _run_cmd("python/bin/python src/pyomo/scripts/get_pyomo_extras.py -v", shell=True)
        if hasattr(output, 'encode'):
            output = output.encode('utf-8','replace')
        print(output.decode('utf-8'))
    else:
        assert False
    # Test
    os.environ['NOSE_PROCESS_TIMEOUT'] = '1800'
    driver.perform_tests('pyomo', cat='book')

elif config == "perf":
    os.environ['NOSE_PROCESS_TIMEOUT'] = '1800'
    driver.perform_build('pyomo', cat='performance')
else:
    raise RuntimeError("Unknown Jenkins configuration: '%s'" % (config,))
