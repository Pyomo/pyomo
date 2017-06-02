#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import os
import subprocess
try:
    from subprocess import check_output as _run_cmd
except:
    # python 2.6
    from subprocess import check_call as _run_cmd
import driver

config = sys.argv[1]
hname = os.uname()[1]
hname = hname.split('.')[0]

print("\nStarting jenkins.py")
print("Configuration=%s" % config)

os.environ['CONFIGFILE'] = os.environ['WORKSPACE']+'/src/pyomo/admin/config.ini'
sys.path.append(os.getcwd())

sys.argv = ['dummy', '--trunk', '--source', 'src', '-a', 'pyyaml']

if hname == "carr":
    os.environ['PATH'] = ':'.join(['/collab/common/bin',
                              '/collab/common/acro/bin',
                              '/collab/gurobi/gurobi501/linux64/bin',
                              '/usr/lib64/openmpi/bin',
                              os.environ['PATH']]
                              )

    if 'LD_LIBRARY_PATH' in os.environ:
        tmp_ = "%s:" % os.environ['LD_LIBRARY_PATH']
    else:
        tmp_ = ""
    os.environ['LD_LIBRARY_PATH'] = tmp_ + '/collab/gurobi/gurobi501/linux64/lib'
    os.environ['GUROBI_HOME'] = '/collab/gurobi/gurobi501/linux64'
    os.environ['GRB_LICENSE_FILE']='/collab/gurobi/gurobi.lic'

    if sys.version_info < (3,):
        sys.argv.append('-a')
        sys.argv.append('/collab/packages/ibm/CPLEX_Studio124/cplex/python/x86-64_sles10_4.1/')
        sys.argv.append('-a')
        sys.argv.append('/collab/gurobi/gurobi501/linux64')

elif hname == "sleipnir":
    os.environ['PATH'] = ':'.join(['/collab/common/bin',
                                '/collab/common/acro/bin',
                                os.environ['PATH']]
                                )

    if sys.version_info < (3,):
        sys.argv.append('-a')
        sys.argv.append('/usr/ilog/cplex124/cplex/python/x86-64_sles10_4.1/')
elif hname == "snotra":
    if sys.version_info < (3,) and sys.version_info[1] >= 7:
        sys.argv.append('-a')
        sys.argv.append('/usr/gurobi/gurobi600/linux64')
        sys.argv.append('-a')
        sys.argv.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux')

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
        print(output.decode('ascii'))
    else:
        assert False
    # Test
    os.environ['TEST_PACKAGES'] = 'checker core environ opt repn scripting solvers util version'
    print("-" * 60)
    print("Performing tests")
    print("-" * 60)
    driver.perform_tests('pyomo', coverage=True, omit=coverage_omit)

elif config == "nonpysp":
    os.environ['TEST_PACKAGES'] = '-e pysp'
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
        print(output.decode('ascii'))
    else:
        assert False
    # Test
    os.environ['NOSE_PROCESS_TIMEOUT'] = '1800'
    driver.perform_tests('pyomo', cat='book')

elif config == "perf":
    os.environ['NOSE_PROCESS_TIMEOUT'] = '1800'
    driver.perform_build('pyomo', cat='performance')

