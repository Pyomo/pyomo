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

solver_dependencies =   {
                        'Test_nonlinear': 
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
                        'Test_data':
                            {'test_data_ABCD9': 'pyodbc',
                            'test_data_ABCD8': 'pyodbc',
                            'test_data_ABCD7': 'win32com'}
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
                package_ = package_dependencies[tf_][n_]
                if not package_ in package_available:
                    try:
                        __import__()
                        package_available[package_] = True
                    except:
                        package_available[package_] = False
        #print solver_available
        #print package_available
    #
    # Return a boolean if the test should be skipped
    #
    if tfname_ in solver_dependencies:
        if name in solver_dependencies[tfname_] and \
           not solver_available[solver_dependencies[tfname_][name]]:
            # Skip the test because a solver is not available
            return True
    if tfname_ in package_dependencies:
        if name in package_dependencies[tfname_] and \
           not package_available[package_dependencies[tfname_][name]]:
            # Skip the test because a package is not available
            return True
    return False


def filter(line):
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
    status = 'Time:' in line or line.startswith('[') or 'with format cpxlp' in line or 'usermodel = <module' in line or line.startswith('File') or 'execution time=' in line
    return status

for fname in glob.glob(os.path.join(currdir,'*')):
    if not os.path.isdir(fname):
        continue

    # Declare an empty TestCase class
    fname_ = fname.replace('-','_')
    if 'pyomobook' in fname_:
        tfname_ = 'Test_'+fname_.split("pyomobook"+os.sep)[1]
        tfname2_ = 'Test2_'+fname_.split("pyomobook"+os.sep)[1]
    else:
        tfname_ = 'Test_'+fname_.split("examples"+os.sep)[1]
        tfname2_ = 'Test2_'+fname_.split("examples"+os.sep)[1]
    Test = globals()[tfname_] = type(tfname_, (unittest.TestCase,), {})
    Test = unittest.category("book")(Test)
    Test2 = globals()[tfname2_] = type(tfname2_, (unittest.TestCase,), {})
    Test2 = unittest.category("book2")(Test2)
    
    #
    for file in list(glob.glob(fname+'/*.py')) + list(glob.glob(fname+'/*/*.py')):
        bname = os.path.basename(file)
        dir_ = os.path.dirname(os.path.abspath(file))+os.sep
        name='.'.join(bname.split('.')[:-1])
        tname = os.path.basename(os.path.dirname(dir_))+'_'+name
        #
        suffix = None
        for suffix_ in ['.txt', '.yml', '.txt2', '.yml2']:
            if os.path.exists(dir_+name+suffix_):
                suffix = suffix_
                break
        #
        if not suffix is None:
            os.chdir(dir_)
            if suffix_ in ['.txt2', '.yml2']:
                forceskip = check_skip(tfname2_, 'test_'+tname.replace('.','_'))
                Test2.add_baseline_test(cmd='cd %s; %s %s' % (dir_, sys.executable, os.path.abspath(bname)),  baseline=dir_+name+suffix, name=tname, filter=filter, tolerance=1e-7, forceskip=forceskip)
            else:
                forceskip = check_skip(tfname_, 'test_'+tname.replace('.','_'))
                Test.add_baseline_test(cmd='cd %s; %s %s' % (dir_, sys.executable, os.path.abspath(bname)),  baseline=dir_+name+suffix, name=tname, filter=filter, tolerance=1e-7, forceskip=forceskip)
            os.chdir(currdir)

    #
    for file in list(glob.glob(fname+'/*.sh')) + list(glob.glob(fname+'/*/*.sh')):
        bname = os.path.basename(file)
        dir_ = os.path.dirname(os.path.abspath(file))+os.sep
        name='.'.join(bname.split('.')[:-1])
        tname = os.path.basename(os.path.dirname(dir_))+'_'+name
        #
        #
        suffix = None
        for suffix_ in ['.txt', '.yml', '.txt2', '.yml2']:
            if os.path.exists(dir_+name+suffix_):
                suffix = suffix_
                break
        #
        if not suffix is None:
            os.chdir(dir_)
            if suffix_ in ['.txt2', '.yml2']:
                forceskip = check_skip(tfname2_, 'test_'+tname.replace('.','_'))
                Test2.add_baseline_test(cmd='cd %s; %s' % (dir_, os.path.abspath(bname)),  baseline=dir_+name+suffix, name=tname, filter=filter, tolerance=1e-7, forceskip=forceskip)
            else:
                forceskip = check_skip(tfname_, 'test_'+tname.replace('.','_'))
                Test.add_baseline_test(cmd='cd %s; %s' % (dir_, os.path.abspath(bname)),  baseline=dir_+name+suffix, name=tname, filter=filter, tolerance=1e-7, forceskip=forceskip)
            os.chdir(currdir)
    #
    Test = None

# Execute the tests
if __name__ == '__main__':
    unittest.main()
