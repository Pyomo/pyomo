#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import os
import subprocess

config = sys.argv[1]
hname = os.uname()[1]
hname = hname.split('.')[0]

print("\nStarting jenkins.py")
print("Configuration=%s" % config)

os.environ['CONFIGFILE'] = os.environ['WORKSPACE']+'/hudson/pyomo-vpy/test_tpls.ini'
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

if config == "default":
    import hudson.pyomo_cov

elif config == "core":
    import hudson.driver
    # Install
    hudson.driver.perform_install('pyomo', config='pyomo_all.ini')
    print("Running 'pyomo install-extras' ...")
    print( subprocess.check_output(["python/bin/pyomo", "install-extras"], shell=True) )
    # Test
    os.environ['TEST_PACKAGES'] = 'checker core environ opt repn scripting solvers util version'
    pyutilib=os.sep.join([os.environ['WORKSPACE'], 'src', 'pyutilib.*'])+',pyutilib.*'
    hudson.driver.perform_build('pyomo', coverage=True, omit=pyutilib)

elif config == "nonpysp":
    os.environ['TEST_PACKAGES'] = '-e pysp'
    import hudson.pyomo_cov

elif config == "parallel":
    import hudson.pyomo_parallel

elif config == "expensive":
    pyutilib=os.sep.join([os.environ['WORKSPACE'], 'src', 'pyutilib.*'])+',pyutilib.*'

    from hudson.driver import perform_build
    perform_build('pyomo', 
        cat='all', coverage=True, omit=pyutilib,
        virtualenv_args=sys.argv[1:])

elif config == "booktests":
    import hudson.driver
    # Install
    hudson.driver.perform_install('pyomo', config='pyomo_all.ini')
    print("Running 'pyomo install-extras' ...")
    print( subprocess.check_output(["python/bin/pyomo", "install-extras"], shell=True) )
    # Test
    os.environ['NOSE_PROCESS_TIMEOUT'] = '1800'
    pyutilib=os.sep.join([os.environ['WORKSPACE'], 'src', 'pyutilib.*'])+',pyutilib.*'
    hudson.driver.perform_build('pyomo', cat='book')

elif config == "perf":
    os.environ['NOSE_PROCESS_TIMEOUT'] = '1800'
    import hudson.pyomo_perf

