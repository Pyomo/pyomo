# Provide some test for daps.
# As of March 2017, these are mostly smoke tests.
# Author: David L. Woodruff (circa March 2017)

import unittest
import pkg_resources
import tempfile
import sys
import os
import shutil
import basicclasses as bc
import stoch_solver as st
import distr2pysp as dp
import pyomo.pysp.daps

__author__ = 'David L. Woodruff <DLWoodruff@UCDavis.edu>'
__date__ = 'March 15, 2017'

class TestDAPS(unittest.TestCase):
    """ Test the daps code."""

    def do_the_deal(self, fromdir):
        """ local to move files from fromdir to the tmp dir"""
        for filename in os.listdir(fromdir):
            if filename.endswith(".json") \
               or filename.endswith(".dat") \
               or filename.endswith(".py"):
                src = str(fromdir + os.sep + filename)
                shutil.copy(src, self.tdir)

    def setUp(self):
        """ Get ready for tests and assign file names.
            Put all literals in this routine."""

        self.seed = 7734  # random number seed

        p = str(pyomo.pysp.daps.__path__)
        ## _NamespacePath(['/home/david/software/pyomo/pyomo/pyomo/pysp/daps'])
        l = p.find("'")
        r = p.find("'", l+1)
        dapspath = p[l+1:r]

        self.farmer_json_dir = dapspath + os.sep + 'concrete_farmer'
        self.farmer_AMPL_dir = dapspath + os.sep + 'farmer'

        self.dptest_dir = self.farmer_json_dir + os.sep + 'dptest'

        self.farmer_concrete_model = 'ReferenceModel.py'
        self.farmer_json_tree_file = 'TreeTemplateFile.json'
        self.farmer_AMPL_tree_file = "TreeTemplateFile.dat"
        self.farmer_AMPL_scen_template_file = "ScenTemplate.dat"

        self.dptest_data_file_dict = 'datafiledict.json'
        self.dptest_distr_dict_file = 'distrdict.json'
        self.indep_norms_n = 4
        self.scipy_norms_n = 3
        
        ######## make a temp dir to which files can be copied  #####
        self.tdir = tempfile.mkdtemp()    #TemporaryDirectory().name
        sys.path.append(self.tdir)

        """ During debugging, local files might get in the way
        of finding the file in the temp dir, so we cd there."""
        os.chdir(self.tdir)

    def tearDown(self):
        pass
    
    def test_2stage_json(self):
        """ smoke for concrete two-stage json"""

        self.do_the_deal(self.farmer_json_dir)
        tree_model = bc.Tree_2Stage_json_dir(self.tdir, \
                                             self.farmer_json_tree_file)
        assert(tree_model is not None)
        solver = st.StochSolver(self.farmer_concrete_model, \
                                tree_model)
        assert(solver is not None)

    def test_2stage_AMPL(self):
        """ smoke for 2 stage AMPL"""
        self.do_the_deal(self.farmer_AMPL_dir)
        bc.do_2Stage_AMPL_dir(self.tdir, \
                              self.farmer_AMPL_tree_file, \
                              self.farmer_AMPL_scen_template_file)

    def test_scipy_2stage(self):
        """ smoke for named scipy dists """

        self.do_the_deal(self.farmer_json_dir)
        self.do_the_deal(self.dptest_dir)
        dp.json_scipy_2stage(self.dptest_distr_dict_file,
                             self.farmer_AMPL_tree_file,
                             self.scipy_norms_n,
                             self.tdir)


        ### the indep norms code needs a better way to specify file locations as of March 2017
    ###def test_indep_norms_from_data(self):
        """ smoke for indep norms from data files"""
        ###self.do_the_deal(self.farmer_dir)
        ###self.do_the_deal(self.dptest_dir)
        ###pass
        """
        dp.indep_norms_from_data_2stage(self.dptest_data_file_dict,\
                                'concrete_farmer/TreeTemplateFile.dat',
                                4,
                                'concrete_farmer/dptest',
                                Seed = 7734)
        """
        
if __name__ == '__main__':
    unittest.main()
