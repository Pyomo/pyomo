#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import platform
import shutil
import tempfile

import pyutilib.th as unittest

from pyomo.common.config import PYOMO_CONFIG_DIR
from pyomo.common.fileutils import thisFile
from pyomo.common.download import FileDownloader

class Test_FileDownloader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = None
        self.basedir = os.path.abspath(os.path.curdir)

    def tearDown(self):
        if self.tmpdir:
            shutil.rmtree(self.tmpdir)

    def test_init(self):
        f = FileDownloader()
        self.assertFalse(f.insecure)
        self.assertIsNone(f.cacert)
        self.assertIsNone(f.fname)

        f = FileDownloader(True)
        self.assertTrue(f.insecure)
        self.assertIsNone(f.cacert)
        self.assertIsNone(f.fname)

        f = FileDownloader(True, thisFile())
        self.assertTrue(f.insecure)
        self.assertEqual(f.cacert, thisFile())
        self.assertIsNone(f.fname)

        with self.assertRaisesRegexp(
                RuntimeError, "cacert='nonexistant_file_name' does not "
                "refer to a valid file."):
            FileDownloader(True, 'nonexistant_file_name')

    def test_parse(self):
        f = FileDownloader()
        f.parse_args([])
        self.assertFalse(f.insecure)
        self.assertIsNone(f.cacert)
        self.assertIsNone(f.fname)

        f = FileDownloader()
        f.parse_args(['--insecure'])
        self.assertTrue(f.insecure)
        self.assertIsNone(f.cacert)
        self.assertIsNone(f.fname)

        f = FileDownloader()
        f.parse_args(['--insecure', '--cacert', thisFile()])
        self.assertTrue(f.insecure)
        self.assertEqual(f.cacert, thisFile())
        self.assertIsNone(f.fname)

        f = FileDownloader()
        f.parse_args(['--insecure', 'bar', '--cacert', thisFile()])
        self.assertTrue(f.insecure)
        self.assertEqual(f.cacert, thisFile())
        self.assertEqual(f.fname, 'bar')

        f = FileDownloader()
        with self.assertRaisesRegexp(
                RuntimeError, '--cacert argument must be followed by the path '
                'to the PEM certificate'):
            f.parse_args(['--cacert'])

        f = FileDownloader()
        with self.assertRaisesRegexp(
                RuntimeError, '--cacert argument must be followed by the path '
                'to the PEM certificate'):
            f.parse_args(['--cacert', '--insecure'])

        f = FileDownloader()
        with self.assertRaisesRegexp(
                RuntimeError, '--cacert argument must be followed by the path '
                'to the PEM certificate'):
            f.parse_args(['--cacert', 'nonexistant_file_name'])

        f = FileDownloader()
        with self.assertRaisesRegexp(
                RuntimeError, "Unrecognized arguments: \['--foo'\]"):
            f.parse_args(['--foo'])

    def test_set_destination_filename(self):
        self.tmpdir = os.path.abspath(tempfile.mkdtemp())

        f = FileDownloader()
        self.assertIsNone(f.fname)
        f.set_destination_filename('foo')
        self.assertEqual(f.fname, os.path.join(PYOMO_CONFIG_DIR, 'foo'))
        # By this point, the CONFIG_DIR is guaranteed to have been created
        self.assertTrue(os.path.isdir(PYOMO_CONFIG_DIR))

        f.fname = self.tmpdir
        f.set_destination_filename('foo')
        target = os.path.join(self.tmpdir, 'foo')
        self.assertEqual(f.fname, target)
        self.assertFalse(os.path.exists(target))

        f.fname = self.tmpdir
        f.set_destination_filename(os.path.join('foo','bar'))
        target = os.path.join(self.tmpdir, 'foo', 'bar')
        self.assertEqual(f.fname, target)
        self.assertFalse(os.path.exists(target))
        target_dir = os.path.join(self.tmpdir, 'foo',)
        self.assertTrue(os.path.isdir(target_dir))

    def test_get_sysinfo(self):
        f = FileDownloader()
        ans = f.get_sysinfo()
        self.assertIs(type(ans), tuple)
        self.assertEqual(len(ans), 2)
        self.assertTrue(len(ans[0]) > 0)
        self.assertTrue(platform.system().lower().startswith(ans[0]))
        self.assertFalse(any(c in ans[0] for c in '.-_'))
        self.assertIn(ans[1], (32,64))

    def test_get_url(self):
        f = FileDownloader()
        urlmap = {'bogus_sys': 'bogus'}
        with self.assertRaisesRegexp(
                RuntimeError, "cannot infer the correct url for platform '.*'"):
            f.get_url(urlmap)

        urlmap[f.get_sysinfo()[0]] = 'correct'
        self.assertEqual(f.get_url(urlmap), 'correct')

