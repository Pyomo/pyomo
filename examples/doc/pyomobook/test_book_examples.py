# Imports
import pyutilib.th as unittest
import glob
import os
import os.path
import sys
import pyomo.environ

try:
    import yaml
    yaml_available=True
except:
    yaml_available=False

# Find all *.txt files, and use them to define baseline tests
currdir = os.path.dirname(os.path.abspath(__file__))
datadir = currdir
testdirs = [currdir, ]

solver_dependencies =   {
    'Test_nonlinear_ch': {
        'test_rosen_pyomo_rosen': 'ipopt',
        'test_react_design_run_pyomo_reactor_table': 'ipopt',
        'test_react_design_run_pyomo_reactor': 'ipopt',
        'test_multimodal_pyomo_multimodal_init1': 'ipopt',
        'test_multimodal_pyomo_multimodal_init2': 'ipopt',
        'test_disease_est_run_disease_summary': 'ipopt',
        'test_disease_est_run_disease_callback': 'ipopt',
        'test_deer_run_deer': 'ipopt',
    },
    'Test_mpec_ch': {
        'test_mpec_ch_path1': 'path',
    },
    'Test_dae_ch': {
        'test_run_path_constraint_tester': 'ipopt',
    },
}
package_dependencies =  {
    'Test_data_abstract_ch': {
        'test_data_abstract_ch_ABCD9': ['pyodbc',],
        'test_data_abstract_ch_ABCD8': ['pyodbc',],
        'test_data_abstract_ch_ABCD7': ['win32com',],
    },
    'Test_overview_ch': {
        'test_overview_ch_pyomo_wl_excel': ['numpy','pandas','xlrd',],
    },
    'Test_scripts_ch': {
        'test_scripts_ch_warehouse_function_cuts': ['numpy','matplotlib',],
    },
}
solver_available = {}
package_available = {}

only_book_tests = set(['Test_nonlinear_ch', 'Test_scripts_ch'])

def _check_available(name):
    from pyomo.opt.base import (UnknownSolver, SolverFactory)
    try:
        opt = SolverFactory(name)
    except:
        return False
    if opt is None or isinstance(opt, UnknownSolver):
        return False
    elif (name == "gurobi") and \
       (not GUROBISHELL.license_is_valid()):
        return False
    elif (name == "baron") and \
       (not BARONSHELL.license_is_valid()):
        return False
    else:
        return (opt.available(exception_flag=False)) and \
            ((not hasattr(opt,'executable')) or \
             (opt.executable() is not None))

def check_skip(tfname_, name):
    #
    # Skip if YAML isn't installed
    #
    if not yaml_available:
        return "YAML is not available"
    #
    # Initialize the availability data
    #
    if len(solver_available) == 0:
        for tf_ in solver_dependencies:
            for n_ in solver_dependencies[tf_]:
                solver_ = solver_dependencies[tf_][n_]
                if not solver_ in solver_available:
                    solver_available[solver_] = _check_available(solver_)
        for tf_ in package_dependencies:
            for n_ in package_dependencies[tf_]:
                packages_ = package_dependencies[tf_][n_]
                for package_ in packages_:
                    if not package_ in package_available:
                        try:
                            __import__(package_)
                            package_available[package_] = True
                        except:
                            package_available[package_] = False
    #
    # Return a boolean if the test should be skipped
    #
    if tfname_ in solver_dependencies:
        if name in solver_dependencies[tfname_] and \
           not solver_available[solver_dependencies[tfname_][name]]:
            # Skip the test because a solver is not available
            # print('Skipping %s because of missing solver' %(name)) 
            return 'Solver "%s" is not available' % (
                solver_dependencies[tfname_][name], )
    if tfname_ in package_dependencies:
        if name in package_dependencies[tfname_]:
            packages_ = package_dependencies[tfname_][name]
            if not all([package_available[i] for i in packages_]):
                # Skip the test because a package is not available
                # print('Skipping %s because of missing package' %(name))
                _missing = []
                for i in packages_:
                    if not package_available[i]:
                        _missing.append(i)
                return "Package%s %s %s not available" % (
                    's' if len(_missing) > 1 else '',
                    ", ".join(_missing),
                    'are' if len(_missing) > 1 else 'is',)
    return False

_DEPRECATION_MESSAGES = """
WARNING: DEPRECATED: Chained inequalities are deprecated. Use the inequality()
    function to express ranged inequality expressions.
WARNING: DEPRECATED: Use of the pyomo.bilevel package is deprecated. There are
    known bugs in pyomo.bilevel, and we do not recommend the use of this code.
    Development of bilevel optimization capabilities has been shifted to the
    Pyomo Adversarial Optimization (PAO) library. Please contact William Hart
    for further details (wehart@sandia.gov). (deprecated in 5.6.2)
WARNING: DEPRECATED: Use of the pyomo.duality package is deprecated. There are
    known bugs in pyomo.duality, and we do not recommend the use of this code.
    Development of dualization capabilities has been shifted to the Pyomo
    Adversarial Optimization (PAO) library. Please contact William Hart for
    further details (wehart@sandia.gov).  (deprecated in 5.6.2)
""".strip()

def filter(line):
    # Ignore certain text when comparing output with baseline

    # Ipopt 3.12.4 puts BACKSPACE (chr(8) / ^H) into the output.
    line = line.strip(" \n\t"+chr(8))

    if not line:
        return True
    for field in ( '[',
                   'password:',
                   'http:',
                   'Job ',
                   'Importing module',
                   'Function',
                   'File', ):
        if line.startswith(field):
            return True
    for field in ( 'Total CPU',
                   'Ipopt',
                   'Status: optimal',
                   'Status: feasible',
                   'time:',
                   'Time:',
                   'with format cpxlp',
                   'usermodel = <module',
                   'execution time=',
                   'Solver results file:' ):
        if field in line:
            return True
    for field in _DEPRECATION_MESSAGES.splitlines():
        strip_field = field.strip()
        if strip_field and strip_field in line:
            return True
    return False

for tdir in testdirs:

  for testdir in glob.glob(os.path.join(tdir,'*')):
    if not os.path.isdir(testdir):
        continue
    # Only test files in directories ending in -ch. These directories
    # contain the updated python and scripting files corresponding to
    # each chapter in the book.
    if '-ch' not in testdir:
        continue

    # print("Testing ",testdir)

    #
    # JDS: This is crazy fragile.  If testdirs is ever anything BUT
    # "pyomobook" you will be creating invalid class names
    #
    #testdir_ = testdir.replace('-','_')
    #testClassName = 'Test_'+testdir_.split("pyomobook"+os.sep)[1]
    testClassName = 'Test_'+os.path.basename(testdir).replace('-','_')
    assert '.' not in testClassName
    Test = globals()[testClassName] = type(
        testClassName, (unittest.TestCase,), {})
    if testClassName in only_book_tests:
        Test = unittest.category("book")(Test)
    else:
        Test = unittest.category("book","smoke","nightly")(Test)
    
    # Find all .py files in the test directory
    for file in list(glob.glob(os.path.join(testdir,'*.py'))) \
        + list(glob.glob(os.path.join(testdir,'*','*.py'))):
    
        test_file = os.path.abspath(file)
        bname = os.path.basename(test_file)
        dir_ = os.path.dirname(test_file)
        name=os.path.splitext(bname)[0]
        tname = os.path.basename(dir_)+'_'+name
    
        suffix = None
        # Look for txt and yml file names matching py file names. Add
        # a test for any found
        for suffix_ in ['.txt', '.yml']:
            if os.path.exists(os.path.join(dir_,name+suffix_)):
                suffix = suffix_
                break
            elif os.path.exists(os.path.join(dir_,name+'.py2'+suffix_)) \
                 and sys.version_info[0] == 2:
                suffix = '.py2'+suffix_
                break
            elif os.path.exists(os.path.join(dir_, name+'.py3'+suffix_)) \
                 and sys.version_info[0] == 3:
                suffix = '.py3'+suffix_
                break
        if not suffix is None:
            cwd = os.getcwd()
            tname = tname.replace('-','_')
            tname = tname.replace('.','_')
            # print(tname)
            forceskip = check_skip(testClassName, 'test_'+tname)
            Test.add_baseline_test(
                cmd=(sys.executable, test_file),
                cwd = dir_,
                baseline=os.path.join(dir_,name+suffix),
                name=tname,
                filter=filter,
                tolerance=1e-3,
                forceskip=forceskip)
            os.chdir(cwd)

    # Find all .sh files in the test directory
    for file in list(glob.glob(os.path.join(testdir,'*.sh'))) \
            + list(glob.glob(os.path.join(testdir,'*','*.sh'))):
        bname = os.path.basename(file)
        dir_ = os.path.dirname(os.path.abspath(file))+os.sep
        name='.'.join(bname.split('.')[:-1])
        tname = os.path.basename(os.path.dirname(dir_))+'_'+name
        suffix = None
        # Look for txt and yml file names matching sh file names. Add
        # a test for any found
        for suffix_ in ['.txt', '.yml']:
            if os.path.exists(dir_+name+suffix_):
                suffix = suffix_
                break
            elif os.path.exists(dir_+name+'.py2'+suffix_) and sys.version_info[0] == 2:
                suffix = '.py2'+suffix_
                break
            elif os.path.exists(dir_+name+'.py3'+suffix_) and sys.version_info[0] == 3:
                suffix = '.py3'+suffix_
                break
        if not suffix is None:
            cwd = os.getcwd()
            os.chdir(dir_)
            tname = tname.replace('-','_')
            tname = tname.replace('.','_')
            # For now, skip all shell tests on Windows.
            if os.name == 'nt':
                forceskip = "Shell tests are not runnable on Windows"
            else:
                forceskip = check_skip(testClassName, 'test_'+tname)
            Test.add_baseline_test(cmd='cd %s; %s' % (dir_,
                                                      os.path.abspath(bname)), baseline=dir_+name+suffix,
                                   name=tname, filter=filter, tolerance=1e-3,
                                   forceskip=forceskip)
            os.chdir(cwd)
    Test = None

# Execute the tests
if __name__ == '__main__':
    unittest.main()
