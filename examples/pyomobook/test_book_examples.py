#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import glob
import os
from pyomo.common.dependencies import attempt_import, matplotlib_available
from pyomo.common.fileutils import this_file_dir
import pyomo.environ as pyo


currdir = this_file_dir()

parameterized, param_available = attempt_import('parameterized')
if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')

# Needed for testing (triggers matplotlib import and switches its backend):
bool(matplotlib_available)


class TestBookExamples(unittest.BaselineTestDriver, unittest.TestCase):
    # Only test files in directories ending in -ch. These directories
    # contain the updated python and scripting files corresponding to
    # each chapter in the book.
    py_tests, sh_tests = unittest.BaselineTestDriver.gather_tests(
        list(filter(os.path.isdir, glob.glob(os.path.join(currdir, '*-ch'))))
    )

    solver_dependencies = {
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
        'test_gdp_ch_pyomo_gdp_uc': ['glpk'],
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
        'test_performance_ch_wl': ['gurobi', 'gurobi_persistent', 'gurobi_license'],
        'test_performance_ch_persistent': ['gurobi_persistent'],
    }
    package_dependencies = {
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
        'test_performance_ch_wl': ['numpy', 'matplotlib'],
    }

    def initialize_dependencies(self):
        super().initialize_dependencies()
        self._check_gurobi_fully_licensed()

    def _check_gurobi_fully_licensed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(list(range(2001)), within=pyo.NonNegativeReals)
        m.o = pyo.Objective(expr=sum(m.x.values()))
        try:
            results = pyo.SolverFactory('gurobi').solve(m, tee=False)
            pyo.assert_optimal_termination(results)
            self.__class__.solver_available['gurobi_license'] = True
        except:
            self.__class__.solver_available['gurobi_license'] = False

    @parameterized.parameterized.expand(
        sh_tests, name_func=unittest.BaselineTestDriver.custom_name_func
    )
    def test_book_sh(self, tname, test_file, base_file):
        self.shell_test_driver(tname, test_file, base_file)

    @parameterized.parameterized.expand(
        py_tests, name_func=unittest.BaselineTestDriver.custom_name_func
    )
    def test_book_py(self, tname, test_file, base_file):
        self.python_test_driver(tname, test_file, base_file)


if __name__ == "__main__":
    unittest.main()
