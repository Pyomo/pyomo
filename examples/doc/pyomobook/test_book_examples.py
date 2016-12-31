# Imports
import pyutilib.th as unittest
import glob
import os
import os.path
import sys
import pyomo.environ

# Find all *.txt files, and use them to define baseline tests
currdir = os.path.dirname(os.path.abspath(__file__))+os.sep
datadir = currdir
testdirs = [currdir, ]

solver_dependencies =   {
                        'Test_nonlinear_ch': 
                            {'test_rosen_pyomo_rosen': 'ipopt',
                            'test_react_design_run_pyomo_reactor_table': 'ipopt',
                            'test_react_design_run_pyomo_reactor': 'ipopt',
                            'test_multimodal_pyomo_multimodal_init1': 'ipopt',
                            'test_multimodal_pyomo_multimodal_init2': 'ipopt',
                            'test_disease_est_run_disease_summary': 'ipopt',
                            'test_disease_est_run_disease_callback': 'ipopt',
                            'test_deer_run_deer': 'ipopt'}
                        }
package_dependencies =  {
                        'Test_data_abstract_ch':
                            {'test_data_abstract_ch_ABCD9': ['pyodbc',],
                            'test_data_abstract_ch_ABCD8': ['pyodbc',],
                             'test_data_abstract_ch_ABCD7': ['win32com',]},
                        'Test_overview_ch':
                            {'test_overview_ch_pyomo_wl_excel': ['numpy','pandas','xlrd',]},
                        'Test_scripts_ch':
                            {'test_scripts_ch_warehouse_function_cuts': ['numpy','matplotlib',]},
                        }
solver_available = {}
package_available = {}

def check_skip(tfname_, name):
    #
    # Initialize the availability data
    #
    if len(solver_available) == 0:
        for tf_ in solver_dependencies:
            for n_ in solver_dependencies[tf_]:
                solver_ = solver_dependencies[tf_][n_]
                if not solver_ in solver_available:
                    opt = pyomo.environ.SolverFactory(solver_)
                    solver_available[solver_] = opt.available()
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
            return True
    if tfname_ in package_dependencies:
        if name in package_dependencies[tfname_]:
            packages_ = package_dependencies[tfname_][name]
            if not all([package_available[i] for i in packages_]):
                # Skip the test because a package is not available
                # print('Skipping %s because of missing package' %(name))
                return True
    return False


def filter(line):
    # Ignore certain text when comparing output with baseline
    line.strip()
    if line.startswith('password:') or line.startswith('http:') or line.startswith('Job '):
        return True
    if 'Total CPU' in line:
        return True
    if 'Ipopt' in line:
        return True
    if line.startswith('Importing module'):
        return True
    if line.startswith('Function'):
        return True
    if 'Status: optimal' in line or 'Status: feasible' in line:
        return True
    status = 'time:' in line or 'Time:' in line or \
             line.startswith('[') or 'with format cpxlp' in line or \
             'usermodel = <module' in line or line.startswith('File') or \
             'execution time=' in line or 'Solver results file:' in line
    return status

for tdir in testdirs:

  for fname in glob.glob(os.path.join(tdir,'*')):
    if not os.path.isdir(fname):
        continue
    # Only test files in directories ending in -ch. These directories
    # contain the updated python and scripting files corresponding to
    # each chapter in the book.
    if '-ch' not in fname:
        continue

    # print("Testing ",fname)

    # Declare an empty TestCase class
    fname_ = fname.replace('-','_')
    tfname_ = 'Test_'+fname_.split("pyomobook"+os.sep)[1]
    Test = globals()[tfname_] = type(tfname_, (unittest.TestCase,), {})
    Test = unittest.category("book","smoke")(Test)
    
    # Find all .py files in the test directory
    for file in list(glob.glob(fname+'/*.py')) + list(glob.glob(fname+'/*/*.py')):
    
        bname = os.path.basename(file)
        dir_ = os.path.dirname(os.path.abspath(file))+os.sep
        name='.'.join(bname.split('.')[:-1])
        tname = os.path.basename(os.path.dirname(dir_))+'_'+name
    
        suffix = None
        # Look for txt and yml file names matching py file names. Add
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
            # print(tname)
            forceskip = check_skip(tfname_, 'test_'+tname)
            Test.add_baseline_test(cmd='cd %s; %s %s' % (dir_,
                                                         sys.executable, os.path.abspath(bname)),
                                   baseline=dir_+name+suffix, name=tname, filter=filter,
                                   tolerance=1e-3, forceskip=forceskip)
            os.chdir(cwd)

    # Find all .sh files in the test directory
    for file in list(glob.glob(fname+'/*.sh')) + list(glob.glob(fname+'/*/*.sh')):
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
            # print(tname)
            forceskip = check_skip(tfname_, 'test_'+tname)
            Test.add_baseline_test(cmd='cd %s; %s' % (dir_,
                                                      os.path.abspath(bname)), baseline=dir_+name+suffix,
                                   name=tname, filter=filter, tolerance=1e-3,
                                   forceskip=forceskip)
            os.chdir(cwd)
    Test = None

# Execute the tests
if __name__ == '__main__':
    unittest.main()
