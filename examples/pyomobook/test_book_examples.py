#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import filecmp
import glob
import os
import os.path
import subprocess
import sys
from itertools import zip_longest
from pyomo.opt import check_available_solvers
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output

parameterized, param_available = attempt_import('parameterized')
if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')

# Find all *.txt files, and use them to define baseline tests
currdir = this_file_dir()
datadir = currdir

solver_dependencies =   {
    # abstract_ch
    'test_abstract_ch_wl_abstract_script': ['glpk'],
    'test_abstract_ch_pyomo_wl_abstract': ['glpk'],
    'test_abstract_ch_pyomo_solve1': ['glpk'],
    'test_abstract_ch_pyomo_solve2': ['glpk'],
    'test_abstract_ch_pyomo_solve3': ['glpk'],
    'test_abstract_ch_pyomo_solve4': ['glpk'],
    'test_abstract_ch_pyomo_solve5': ['glpk'],
    'test_abstract_ch_pyomo_diet1': ['glpk'],
    'test_abstract_ch_pyomo_buildactions_works': ['glpk'],
    'test_abstract_ch_pyomo_abstract5_ns1': ['glpk'],
    'test_abstract_ch_pyomo_abstract5_ns2': ['glpk'],
    'test_abstract_ch_pyomo_abstract5_ns3': ['glpk'],
    'test_abstract_ch_pyomo_abstract6': ['glpk'],
    'test_abstract_ch_pyomo_abstract7': ['glpk'],
    'test_abstract_ch_pyomo_AbstractH': ['ipopt'],
    'test_abstract_ch_AbstHLinScript': ['glpk'],
    'test_abstract_ch_pyomo_AbstractHLinear': ['glpk'],

    # blocks_ch
    'test_blocks_ch_lotsizing': ['glpk'],
    'test_blocks_ch_blocks_lotsizing': ['glpk'],

    # dae_ch
    'test_dae_ch_run_path_constraint_tester': ['ipopt'],

    # gdp_ch
    'test_gdp_ch_pyomo_scont': ['glpk'],
    'test_gdp_ch_pyomo_scont2': ['glpk'],
    'test_gdp_ch_scont_script': ['glpk'],

    # intro_ch'
    'test_intro_ch_pyomo_concrete1_generic': ['glpk'],
    'test_intro_ch_pyomo_concrete1': ['glpk'],
    'test_intro_ch_pyomo_coloring_concrete': ['glpk'],
    'test_intro_ch_pyomo_abstract5': ['glpk'],

    # mpec_ch
    'test_mpec_ch_path1': ['path'],
    'test_mpec_ch_nlp_ex1b': ['ipopt'],
    'test_mpec_ch_nlp_ex1c': ['ipopt'],
    'test_mpec_ch_nlp_ex1d': ['ipopt'],
    'test_mpec_ch_nlp_ex1e': ['ipopt'],
    'test_mpec_ch_nlp_ex2': ['ipopt'],
    'test_mpec_ch_nlp1': ['ipopt'],
    'test_mpec_ch_nlp2': ['ipopt'],
    'test_mpec_ch_nlp3': ['ipopt'],
    'test_mpec_ch_mip1': ['glpk'],

    # nonlinear_ch
    'test_rosen_rosenbrock': ['ipopt'],
    'test_react_design_ReactorDesign': ['ipopt'],
    'test_react_design_ReactorDesignTable': ['ipopt'],
    'test_multimodal_multimodal_init1': ['ipopt'],
    'test_multimodal_multimodal_init2': ['ipopt'],
    'test_disease_est_disease_estimation': ['ipopt'],
    'test_deer_DeerProblem': ['ipopt'],

    # scripts_ch
    'test_sudoku_sudoku_run': ['glpk'],
    'test_scripts_ch_warehouse_script': ['glpk'],
    'test_scripts_ch_warehouse_print': ['glpk'],
    'test_scripts_ch_warehouse_cuts': ['glpk'],
    'test_scripts_ch_prob_mod_ex': ['glpk'],
    'test_scripts_ch_attributes': ['glpk'],

    # optimization_ch
    'test_optimization_ch_ConcHLinScript': ['glpk'],

    # overview_ch
    'test_overview_ch_wl_mutable_excel': ['glpk'],
    'test_overview_ch_wl_excel': ['glpk'],
    'test_overview_ch_wl_concrete_script': ['glpk'],
    'test_overview_ch_wl_abstract_script': ['glpk'],
    'test_overview_ch_pyomo_wl_abstract': ['glpk'],

    # performance_ch
    'test_performance_ch_wl': ['gurobi', 'gurobi_persistent'],
    'test_performance_ch_persistent': ['gurobi_persistent'],
}
package_dependencies =  {
    # abstract_ch'
    'test_abstract_ch_pyomo_solve4': ['yaml'],
    'test_abstract_ch_pyomo_solve5': ['yaml'],

    # gdp_ch
    'test_gdp_ch_pyomo_scont': ['yaml'],
    'test_gdp_ch_pyomo_scont2': ['yaml'],
    'test_gdp_ch_pyomo_gdp_uc': ['sympy'],

    # overview_ch'
    'test_overview_ch_wl_excel': ['pandas', 'xlrd'],
    'test_overview_ch_wl_mutable_excel': ['pandas', 'xlrd'],

    # scripts_ch'
    'test_scripts_ch_warehouse_cuts': ['matplotlib'],

    # performance_ch'
    'test_performance_ch_wl': ['numpy','matplotlib'],
}


#
# Initialize the availability data
#
solvers_used = set(sum(list(solver_dependencies.values()), []))
available_solvers = check_available_solvers(*solvers_used)
solver_available = {solver_:solver_ in available_solvers for solver_ in solvers_used}

package_available = {}
package_modules = {}
packages_used = set(sum(list(package_dependencies.values()), []))
for package_ in packages_used:
    pack, pack_avail = attempt_import(package_)
    package_available[package_] = pack_avail
    package_modules[package_] = pack


def check_skip(name):
    """
    Return a boolean if the test should be skipped
    """

    if name in solver_dependencies:
        solvers_ = solver_dependencies[name]
        if not all([solver_available[i] for i in solvers_]):
            # Skip the test because a solver is not available
            _missing = []
            for i in solvers_:
                if not solver_available[i]:
                    _missing.append(i)
            return "Solver%s %s %s not available" % (
                's' if len(_missing) > 1 else '',
                ", ".join(_missing),
                'are' if len(_missing) > 1 else 'is',)

    if name in package_dependencies:
        packages_ = package_dependencies[name]
        if not all([package_available[i] for i in packages_]):
            # Skip the test because a package is not available
            _missing = []
            for i in packages_:
                if not package_available[i]:
                    _missing.append(i)
            return "Package%s %s %s not available" % (
                's' if len(_missing) > 1 else '',
                ", ".join(_missing),
                'are' if len(_missing) > 1 else 'is',)

        # This is a hack, xlrd dropped support for .xlsx files in 2.0.1 which
        # causes problems with older versions of Pandas<=1.1.5 so skipping
        # tests requiring both these packages when incompatible versions are found
        if 'pandas' in package_dependencies[name] and 'xlrd' in package_dependencies[name]:
            if check_min_version(package_modules['xlrd'], '2.0.1') and \
               not check_min_version(package_modules['pandas'], '1.1.6'):
                return "Incompatible versions of xlrd and pandas"

    return False

def filter(line):
    """
    Ignore certain text when comparing output with baseline
    """
    for field in ( '[',
                   'password:',
                   'http:',
                   'Job ',
                   'Importing module',
                   'Function',
                   'File',
                   'Matplotlib',
                   '    ^'):
        if line.startswith(field):
            return True
    for field in ( 'Total CPU',
                   'Ipopt',
                   'license',
                   'Status: optimal',
                   'Status: feasible',
                   'time:',
                   'Time:',
                   'with format cpxlp',
                   'usermodel = <module',
                   'execution time=',
                   'Solver results file:',
                   'TokenServer',
                   'function calls',
                   'List reduced',
                   '.py:',
                   'built-in method',
                   '{method'):
        if field in line:
            return True
    return False


def filter_file_contents(lines):
    filtered = []
    for line in lines:
        if not line or filter(line):
            continue

        # Strip off beginning of lines giving time in seconds
        # Needed for the performance chapter tests
        if "seconds" in line:
            s = line.find("seconds")+7
            line = line[s:]

        items = line.strip().split()
        for i in items:
            if not i:
                continue
            if i.startswith('/') or i.startswith(":\\", 1):
                continue

            # A few substitutions to get tests passing on pypy3
            if ".inf" in i:
                i = i.replace(".inf", "inf")
            if "null" in i:
                i = i.replace("null", "None")

            try:
                filtered.append(float(i))
            except:
                filtered.append(i)

    return filtered


py_test_tuples=[]
sh_test_tuples=[]

for testdir in glob.glob(os.path.join(currdir,'*')):
    if not os.path.isdir(testdir):
        continue
    # Only test files in directories ending in -ch. These directories
    # contain the updated python and scripting files corresponding to
    # each chapter in the book.
    if '-ch' not in testdir:
        continue

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
        if not suffix is None:
            tname = tname.replace('-','_')
            tname = tname.replace('.','_')

            # Create list of tuples with (test_name, test_file, baseline_file)
            py_test_tuples.append((tname, test_file, os.path.join(dir_,name+suffix)))

    # Find all .sh files in the test directory
    for file in list(glob.glob(os.path.join(testdir,'*.sh'))) \
            + list(glob.glob(os.path.join(testdir,'*','*.sh'))):
        test_file = os.path.abspath(file)
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
        if not suffix is None:
            tname = tname.replace('-','_')
            tname = tname.replace('.','_')

            # Create list of tuples with (test_name, test_file, baseline_file)
            sh_test_tuples.append((tname, test_file, os.path.join(dir_,name+suffix)))


def custom_name_func(test_func, test_num, test_params):
    func_name = test_func.__name__
    return "test_%s_%s" %(test_params.args[0], func_name[-2:])


def compare_files(out_file, base_file, abstol, reltol,
                  exception, formatter):
    try:
        if filecmp.cmp(out_file, base_file):
            return True
    except:
        pass

    with open(out_file, 'r') as f1, open(base_file, 'r') as f2:
        out_file_contents = f1.read()
        base_file_contents = f2.read()

    # Filter files independently and then compare filtered contents
    out_filtered = filter_file_contents(
        out_file_contents.strip().split('\n'))
    base_filtered = filter_file_contents(
        base_file_contents.strip().split('\n'))

    if len(out_filtered) != len(base_filtered):
        # it is likely that a solver returned a (slightly) nonzero
        # value for a variable that is normally 0.  Try to look for
        # sequences like "['varname:', 'Value:', 1e-9]" that appear
        # in one result but not the other and remove them.
        i = 0
        while i < len(base_filtered):
            try:
                unittest.assertStructuredAlmostEqual(
                    out_filtered[i], base_filtered[i],
                    abstol=abstol, reltol=reltol,
                    allow_second_superset=False,
                    exception=exception)
                i += 1
                continue
            except exception:
                pass

            try:
                index_of_out_i_in_base = base_filtered.index(
                    out_filtered[i], i)
            except ValueError:
                index_of_out_i_in_base = float('inf')
            try:
                index_of_base_i_in_out = out_filtered.index(
                    base_filtered[i], i)
            except ValueError:
                index_of_base_i_in_out = float('inf')
            if index_of_out_i_in_base < index_of_base_i_in_out:
                extra = base_filtered
                n = index_of_out_i_in_base
            else:
                extra = out_filtered
                n = index_of_base_i_in_out
            extra_terms = extra[i:n]
            try:
                assert len(extra_terms) % 3 == 0
                assert all(str(_)[-1] == ":" for _ in extra_terms[0::3])
                assert all(str(_) == "Value:" for _ in extra_terms[1::3])
                assert all(abs(_) < abstol for _ in extra_terms[2::3])
            except:
                # This does not match the pattern we are looking
                # for: quit processing, and let the next
                # assertStructuredAlmostEqual raise the appropriate
                # failureException
                break
            extra[i:n] = []

    try:
        unittest.assertStructuredAlmostEqual(out_filtered, base_filtered,
                                             abstol=abstol, reltol=reltol,
                                             allow_second_superset=False,
                                             exception=exception,
                                             formatter=formatter)
    except exception:
        # Print helpful information when file comparison fails
        print('---------------------------------')
        print('BASELINE FILE')
        print('---------------------------------')
        print(base_file_contents)
        print('=================================')
        print('---------------------------------')
        print('TEST OUTPUT FILE')
        print('---------------------------------')
        print(out_file_contents)
        raise
    return True


class TestBookExamples(unittest.TestCase):

    def compare_files(self, out_file, base_file, abstol=1e-6):
        return compare_files(
            out_file,
            base_file,
            abstol=abstol,
            reltol=None,
            exception=self.failureException,
            formatter=self._formatMessage,
        )

    @parameterized.parameterized.expand(py_test_tuples, name_func=custom_name_func)
    def test_book_py(self, tname, test_file, base_file):
        bname = os.path.basename(test_file)
        dir_ = os.path.dirname(test_file)

        skip_msg = check_skip('test_'+tname)
        if skip_msg:
            raise unittest.SkipTest(skip_msg)

        cwd = os.getcwd()
        os.chdir(dir_)
        out_file = os.path.splitext(test_file)[0]+'.out'
        with open(out_file, 'w') as f:
            subprocess.run([sys.executable, bname], stdout=f, stderr=f, cwd=dir_)
        os.chdir(cwd)

        self.compare_files(out_file, base_file)
        os.remove(out_file)

    @parameterized.parameterized.expand(sh_test_tuples, name_func=custom_name_func)
    def test_book_sh(self, tname, test_file, base_file):
        bname = os.path.basename(test_file)
        dir_ = os.path.dirname(test_file)

        skip_msg = check_skip('test_'+tname)
        if skip_msg:
            raise unittest.SkipTest(skip_msg)

        # Skip all shell tests on Windows.
        if os.name == 'nt':
           raise unittest.SkipTest("Shell tests are not runnable on Windows")

        cwd = os.getcwd()
        os.chdir(dir_)
        out_file = os.path.splitext(test_file)[0]+'.out'
        with open(out_file, 'w') as f:
            _env = os.environ.copy()
            _env['PATH'] = os.pathsep.join([os.path.dirname(sys.executable), _env['PATH']])
            subprocess.run(['bash', bname], stdout=f, stderr=f, cwd=dir_, env=_env)
        os.chdir(cwd)

        self.compare_files(out_file, base_file)
        os.remove(out_file)

    def test_test_functions(self):
        with capture_output() as OUT:
            self.assertTrue(self.compare_files(
                os.path.join(currdir,'tests','ref1.txt'),
                os.path.join(currdir,'tests','ref2.txt'),
            ))
            self.assertTrue(self.compare_files(
                os.path.join(currdir,'tests','ref2.txt'),
                os.path.join(currdir,'tests','ref1.txt'),
            ))
        self.assertEqual(OUT.getvalue(), "")

        with self.assertRaises(self.failureException):
            with capture_output() as OUT:
                self.compare_files(
                    os.path.join(currdir,'tests','ref1.txt'),
                    os.path.join(currdir,'tests','ref2.txt'),
                    abstol=1e-10,
                )
        self.assertIn('BASELINE FILE', OUT.getvalue())
        self.assertIn('TEST OUTPUT FILE', OUT.getvalue())

if __name__ == "__main__":
    unittest.main()
