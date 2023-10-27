import coramin
import unittest
import os
from pyomo.common.fileutils import this_file_dir
from urllib.request import urlopen
from socket import timeout


try:
    urlopen('http://www.minlplib.org', timeout=3)
except timeout:
    raise unittest.SkipTest('an internet connection is required to test minlplib_tools')


class TestMINLPLibTools(unittest.TestCase):
    def test_get_minlplib_instancedata(self):
        current_dir = os.getcwd()
        coramin.third_party.get_minlplib_instancedata()
        self.assertTrue(os.path.exists(os.path.join(current_dir, 'minlplib', 'instancedata.csv')))
        cases = coramin.third_party.filter_minlplib_instances()
        self.assertEqual(len(cases), 1595)
        os.remove(os.path.join(current_dir, 'minlplib', 'instancedata.csv'))
        os.rmdir(os.path.join(current_dir, 'minlplib'))

    def test_filter_minlplib_instances(self):
        current_dir = this_file_dir()
        coramin.third_party.get_minlplib_instancedata(target_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'))

        total_cases = 1595
        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              acceptable_formats='osil',
                                                              acceptable_probtype='QCQP',
                                                              min_njacobiannz=1000,
                                                              max_njacobiannz=10000)
        self.assertEqual(len(cases), 6)  # regression

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              acceptable_formats=['osil', 'gms'])
        self.assertEqual(len(cases), total_cases)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'))
        self.assertEqual(len(cases), total_cases)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              acceptable_probtype=['QCQP', 'MIQCQP', 'MBQCQP'])
        self.assertEqual(len(cases), 56)  # regression

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              acceptable_objtype='linear',
                                                              acceptable_objcurvature='linear',
                                                              acceptable_conscurvature='convex',
                                                              acceptable_convex=True)
        self.assertEqual(len(cases), 280)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              acceptable_convex=[True])
        self.assertEqual(len(cases), 377)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              min_nvars=2, max_nvars=200000)
        self.assertEqual(len(cases), total_cases-16-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nbinvars=31000)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nintvars=1999)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_ncons=164000)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nsemi=13)
        self.assertEqual(len(cases), total_cases)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nsos1=0, max_nsos2=0)
        self.assertEqual(len(cases), total_cases-6)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nnlvars=199998)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nnlbinvars=23867)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nnlintvars=1999)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nobjnz=99997)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nobjnlnz=99997)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nlincons=164319)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nquadcons=139999)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_npolynomcons=13975)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nsignomcons=801)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_ngennlcons=13975)
        self.assertEqual(len(cases), total_cases-2)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_njacobiannlnz=1623023)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nlaghessiannz=1825419)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              max_nlaghessiandiagnz=100000)
        self.assertEqual(len(cases), total_cases-1)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              min_nnlsemi=1)
        self.assertEqual(len(cases), 0)  # unit

        cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=os.path.join(current_dir, 'minlplib', 'instancedata.csv'),
                                                              acceptable_objcurvature=['linear', 'convex'])
        self.assertEqual(len(cases), 1220)  # unit
        
        os.remove(os.path.join(current_dir, 'minlplib', 'instancedata.csv'))
        os.rmdir(os.path.join(current_dir, 'minlplib'))

    def test_get_minlplib(self):
        current_dir = this_file_dir()
        coramin.third_party.get_minlplib(download_dir=os.path.join(current_dir, 'minlplib', 'osil'))
        files = os.listdir(os.path.join(current_dir, 'minlplib', 'osil'))
        self.assertEqual(len(files), 1594)
        for i in files:
            self.assertTrue(i.endswith('.osil'))
        for i in os.listdir(os.path.join(current_dir, 'minlplib', 'osil')):
            os.remove(os.path.join(current_dir, 'minlplib', 'osil', i))
        os.rmdir(os.path.join(current_dir, 'minlplib', 'osil'))
        os.rmdir(os.path.join(current_dir, 'minlplib'))

    def test_get_minlplib_problem(self):
        current_dir = os.getcwd()
        coramin.third_party.get_minlplib(format='gms', problem_name='ex4_1_1')
        files = os.listdir(os.path.join(current_dir, 'minlplib', 'gms'))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], 'ex4_1_1.gms')
        os.remove(os.path.join(current_dir, 'minlplib', 'gms', files[0]))
        os.rmdir(os.path.join(current_dir, 'minlplib', 'gms'))
        os.rmdir(os.path.join(current_dir, 'minlplib'))

        
class TestExceptions(unittest.TestCase):
    def test_exceptions1(self):
        current_dir = this_file_dir()
        filename = os.path.join(current_dir, 'instancedata.csv')
        f = open(filename, 'w')
        f.write('blah')
        f.close()

        with self.assertRaises(ValueError):
            coramin.third_party.get_minlplib_instancedata(target_filename=filename)

        f = open(filename, 'r')
        self.assertEqual(f.read(), 'blah')
        os.remove(filename)

    def test_exceptions2(self):
        current_dir = this_file_dir()
        filename = os.path.join(current_dir, 'minlplib', 'instancedata.csv')
        coramin.third_party.get_minlplib_instancedata(target_filename=filename)

        with self.assertRaises(ValueError):
            cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=filename, acceptable_probtype='foo')

        with self.assertRaises(ValueError):
            cases = coramin.third_party.filter_minlplib_instances(instancedata_filename=filename, acceptable_probtype=['QCQP', 'foo'])

        os.remove(filename)
        os.rmdir(os.path.dirname(filename))

    def test_exceptions3(self):
        current_dir = this_file_dir()
        os.makedirs(os.path.join(current_dir, 'minlplib', 'osil'))
        with self.assertRaises(ValueError):
            coramin.third_party.get_minlplib(download_dir=os.path.join(current_dir, 'minlplib', 'osil'))
        files = os.listdir(os.path.join(current_dir, 'minlplib', 'osil'))
        self.assertEqual(len(files), 0)
        os.rmdir(os.path.join(current_dir, 'minlplib', 'osil'))
        os.rmdir(os.path.join(current_dir, 'minlplib'))
