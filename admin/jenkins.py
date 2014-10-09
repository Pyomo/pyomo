import sys
import os

config = sys.argv[1]
hname = os.uname()[1]
hname = hname.split('.')[0]

os.environ['CONFIGFILE'] = os.environ['WORKSPACE']+'/hudson/coopr-vpy/test_tpls.ini'
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


if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = ""

print("\nPython version: %s" % sys.version)
print("\nSystem PATH:\n\t%s" % os.environ['PATH'])
print("\nPython path:\n\t%s" % sys.path)


if config == "default":
    import hudson.coopr_cov

elif config == "core":
    os.environ['TEST_PACKAGES'] = 'coopr.opt coopr.pyomo coopr.plugins'
    import hudson.coopr_cov

elif config == "parallel":
    import hudson.coopr_parallel

elif config == "expensive":
    pyutilib=os.sep.join([os.environ['WORKSPACE'], 'src', 'pyutilib.*'])+',pyutilib.*'

    from hudson.driver import perform_build
    perform_build('coopr', 
        cat='all', coverage=True, omit=pyutilib,
        virtualenv_args=sys.argv[1:])

elif config == "booktests":
    import hudson.coopr_book

elif config == "perf":
    os.environ['NOSE_PROCESS_TIMEOUT'] = '1800'
    import hudson.coopr_perf

