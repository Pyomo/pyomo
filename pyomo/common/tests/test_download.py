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
import re
import six
import shutil
import tempfile

import pyutilib.th as unittest
from pyutilib.misc import capture_output
from pyutilib.subprocess import run

from pyomo.common import DeveloperError
from pyomo.common.config import PYOMO_CONFIG_DIR
from pyomo.common.fileutils import this_file
from pyomo.common.download import FileDownloader, distro_available

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
        self.assertIsNone(f._fname)

        f = FileDownloader(True)
        self.assertTrue(f.insecure)
        self.assertIsNone(f.cacert)
        self.assertIsNone(f._fname)

        f = FileDownloader(True, this_file())
        self.assertTrue(f.insecure)
        self.assertEqual(f.cacert, this_file())
        self.assertIsNone(f._fname)

        with self.assertRaisesRegexp(
                RuntimeError, "cacert='nonexistant_file_name' does not "
                "refer to a valid file."):
            FileDownloader(True, 'nonexistant_file_name')

    def test_parse(self):
        f = FileDownloader()
        f.parse_args([])
        self.assertFalse(f.insecure)
        self.assertIsNone(f.cacert)
        self.assertIsNone(f.target)

        f = FileDownloader()
        f.parse_args(['--insecure'])
        self.assertTrue(f.insecure)
        self.assertIsNone(f.cacert)
        self.assertIsNone(f.target)

        f = FileDownloader()
        f.parse_args(['--insecure', '--cacert', this_file()])
        self.assertTrue(f.insecure)
        self.assertEqual(f.cacert, this_file())
        self.assertIsNone(f.target)

        f = FileDownloader()
        f.parse_args(['--insecure', 'bar', '--cacert', this_file()])
        self.assertTrue(f.insecure)
        self.assertEqual(f.cacert, this_file())
        self.assertEqual(f.target, 'bar')

        f = FileDownloader()
        with capture_output() as io:
            with self.assertRaises(SystemExit):
                f.parse_args(['--cacert'])
            self.assertIn('argument --cacert: expected one argument',
                          io.getvalue())

        f = FileDownloader()
        with capture_output() as io:
            with self.assertRaises(SystemExit):
                f.parse_args(['--cacert', '--insecure'])
            self.assertIn('argument --cacert: expected one argument',
                          io.getvalue())

        f = FileDownloader()
        with self.assertRaisesRegexp(
                RuntimeError, "--cacert='nonexistant_file_name' does "
                "not refer to a valid file"):
            f.parse_args(['--cacert', 'nonexistant_file_name'])

        f = FileDownloader()
        with capture_output() as io:
            with self.assertRaises(SystemExit):
                f.parse_args(['--foo'])
            self.assertIn('error: unrecognized arguments: --foo',
                          io.getvalue())

    def test_set_destination_filename(self):
        self.tmpdir = os.path.abspath(tempfile.mkdtemp())

        f = FileDownloader()
        self.assertIsNone(f._fname)
        f.set_destination_filename('foo')
        self.assertEqual(f._fname, os.path.join(PYOMO_CONFIG_DIR, 'foo'))
        # By this point, the CONFIG_DIR is guaranteed to have been created
        self.assertTrue(os.path.isdir(PYOMO_CONFIG_DIR))

        f.target = self.tmpdir
        f.set_destination_filename('foo')
        target = os.path.join(self.tmpdir, 'foo')
        self.assertEqual(f._fname, target)
        self.assertFalse(os.path.exists(target))

        f.target = self.tmpdir
        f.set_destination_filename(os.path.join('foo','bar'))
        target = os.path.join(self.tmpdir, 'foo', 'bar')
        self.assertEqual(f._fname, target)
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

    def test_get_os_version(self):
        f = FileDownloader()
        _os, _ver = f.get_os_version(normalize=False)
        _norm = f.get_os_version(normalize=True)
        #print(_os,_ver,_norm)
        _sys = f.get_sysinfo()[0]
        if _sys == 'linux':
            dist, dist_ver = re.match('^([^0-9]+)(.*)', _norm).groups()
            self.assertNotIn('.', dist_ver)
            self.assertGreater(int(dist_ver), 0)
            if dist == 'ubuntu':
                self.assertEqual(dist_ver, ''.join(_ver.split('.')[:2]))
            else:
                self.assertEqual(dist_ver, _ver.split('.')[0])

            if distro_available:
                d, v = f._get_distver_from_distro()
                #print(d,v)
                self.assertEqual(_os, d)
                self.assertEqual(_ver, v)
                self.assertTrue(v.replace('.','').startswith(dist_ver))

            if os.path.exists('/etc/redhat-release'):
                d, v = f._get_distver_from_redhat_release()
                #print(d,v)
                self.assertEqual(_os, d)
                self.assertEqual(_ver, v)
                self.assertTrue(v.replace('.','').startswith(dist_ver))

            if run(['lsb_release'])[0] == 0:
                d, v = f._get_distver_from_lsb_release()
                #print(d,v)
                self.assertEqual(_os, d)
                self.assertEqual(_ver, v)
                self.assertTrue(v.replace('.','').startswith(dist_ver))

            if os.path.exists('/etc/os-release'):
                d, v = f._get_distver_from_os_release()
                #print(d,v)
                self.assertEqual(_os, d)
                # Note that (at least on centos), os_release is an
                # imprecise version string
                self.assertTrue(_ver.startswith(v))
                self.assertTrue(v.replace('.','').startswith(dist_ver))

        elif _sys == 'darwin':
            dist, dist_ver = re.match('^([^0-9]+)(.*)', _norm).groups()
            self.assertEqual(_os, 'macos')
            self.assertEqual(dist, 'macos')
            self.assertNotIn('.', dist_ver)
            self.assertGreater(int(dist_ver), 0)
            self.assertEqual(_norm, _os+''.join(_ver.split('.')[:2]))
        elif _sys == 'windows':
            self.assertEqual(_os, 'win')
            self.assertEqual(_norm, _os+''.join(_ver.split('.')[:2]))
        else:
            self.assertEqual(ans, '')

        self.assertEqual((_os, _ver), FileDownloader._os_version)
        # Exercise the fetch from CACHE
        try:
            FileDownloader._os_version, tmp \
                = ("test", '2'), FileDownloader._os_version
            self.assertEqual(f.get_os_version(False), ("test","2"))
            self.assertEqual(f.get_os_version(), "test2")
        finally:
            FileDownloader._os_version = tmp


    def test_get_platform_url(self):
        f = FileDownloader()
        urlmap = {'bogus_sys': 'bogus'}
        with self.assertRaisesRegexp(
                RuntimeError, "cannot infer the correct url for platform '.*'"):
            f.get_platform_url(urlmap)

        urlmap[f.get_sysinfo()[0]] = 'correct'
        self.assertEqual(f.get_platform_url(urlmap), 'correct')


    def test_get_files_requires_set_destination(self):
        f = FileDownloader()
        with self.assertRaisesRegexp(
                DeveloperError, 'target file name has not been initialized'):
            f.get_binary_file('bogus')

        with self.assertRaisesRegexp(
                DeveloperError, 'target file name has not been initialized'):
            f.get_binary_file_from_zip_archive('bogus', 'bogus')

        with self.assertRaisesRegexp(
                DeveloperError, 'target file name has not been initialized'):
            f.get_gzipped_binary_file('bogus')

    def test_get_test_binary_file(self):
        tmpdir = tempfile.mkdtemp()
        try:
            f = FileDownloader()

            # Mock retrieve_url so network connections are not necessary
            if six.PY3:
                f.retrieve_url = lambda url: bytes("\n", encoding='utf-8')
            else:
                f.retrieve_url = lambda url: str("\n")

            # Binary files will preserve line endings
            target = os.path.join(tmpdir, 'bin.txt')
            f.set_destination_filename(target)
            f.get_binary_file(None)
            self.assertEqual(os.path.getsize(target), 1)

            # Text files will convert line endings to the local platform
            target = os.path.join(tmpdir, 'txt.txt')
            f.set_destination_filename(target)
            f.get_text_file(None)
            self.assertEqual(os.path.getsize(target), len(os.linesep))
        finally:
            shutil.rmtree(tmpdir)
