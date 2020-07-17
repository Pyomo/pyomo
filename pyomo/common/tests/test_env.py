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
import pyutilib.th as unittest

from pyomo.common.env import TemporaryEnv

class TestTemporaryEnv(unittest.TestCase):
    def test_temp_env_str(self):
        orig_env = TemporaryEnv()
        orig_env_has_1 = 'TEST_ENV_1' in orig_env
        orig_env_has_2 = 'TEST_ENV_2' in orig_env

        # Ensure that TEST_ENV_1 is not in the environment
        if 'TEST_ENV_1' in orig_env:
            del orig_env['TEST_ENV_1']
        orig_env['TEST_ENV_2'] = "test value: 2"

        self.assertIsNone(os.environ.get('TEST_ENV_1', None))
        self.assertEqual(os.environ['TEST_ENV_2'], "test value: 2")
        if orig_env.msvcrt.available():
            self.assertIsNone(orig_env.msvcrt.wgetenv(u'TEST_ENV_1'))
            self.assertEqual(
                orig_env.msvcrt.wgetenv(u'TEST_ENV_2'), "test value: 2")
            
        with TemporaryEnv(TEST_ENV_1="test value: 1") as env:
            self.assertEqual(os.environ['TEST_ENV_1'], "test value: 1")
            self.assertEqual(os.environ['TEST_ENV_2'], "test value: 2")
            if env.msvcrt.available():
                self.assertEqual(
                    env.msvcrt.wgetenv(u'TEST_ENV_1'), "test value: 1")
                self.assertEqual(
                    env.msvcrt.wgetenv(u'TEST_ENV_2'), "test value: 2")

            del env['TEST_ENV_2']
            self.assertIsNone(os.environ.get('TEST_ENV_2', None))
            if env.msvcrt.available():
                self.assertIsNone(env.msvcrt.wgetenv(u'TEST_ENV_2'))

        self.assertIsNone(os.environ.get('TEST_ENV_1', None))
        self.assertEqual(os.environ['TEST_ENV_2'], "test value: 2")
        if orig_env.msvcrt.available():
            self.assertIsNone(orig_env.msvcrt.wgetenv(u'TEST_ENV_1'))
            self.assertEqual(
                orig_env.msvcrt.wgetenv(u'TEST_ENV_2'), "test value: 2")

        orig_env.restore()
        self.assertEqual(orig_env_has_1, 'TEST_ENV_1' in os.environ)
        self.assertEqual(orig_env_has_2, 'TEST_ENV_2' in os.environ)

        

    def test_temp_env_unicode(self):
        orig_env = TemporaryEnv()
        orig_env_has_1 = u'TEST_ENV_1' in orig_env
        orig_env_has_2 = u'TEST_ENV_2' in orig_env

        # Ensure that TEST_ENV_1 is not in the environment
        if u'TEST_ENV_1' in orig_env:
            del orig_env[u'TEST_ENV_1']
        orig_env[u'TEST_ENV_2'] = "test value: 2"

        self.assertIsNone(os.environ.get(u'TEST_ENV_1', None))
        self.assertEqual(os.environ[u'TEST_ENV_2'], "test value: 2")
        if orig_env.msvcrt.available():
            self.assertIsNone(orig_env.msvcrt.wgetenv(u'TEST_ENV_1'))
            self.assertEqual(
                orig_env.msvcrt.wgetenv(u'TEST_ENV_2'), "test value: 2")
            
        with TemporaryEnv(TEST_ENV_1="test value: 1") as env:
            self.assertEqual(os.environ[u'TEST_ENV_1'], "test value: 1")
            self.assertEqual(os.environ[u'TEST_ENV_2'], "test value: 2")
            if env.msvcrt.available():
                self.assertEqual(
                    env.msvcrt.wgetenv(u'TEST_ENV_1'), "test value: 1")
                self.assertEqual(
                    env.msvcrt.wgetenv(u'TEST_ENV_2'), "test value: 2")

            del env[u'TEST_ENV_2']
            self.assertIsNone(os.environ.get(u'TEST_ENV_2', None))
            if env.msvcrt.available():
                self.assertIsNone(env.msvcrt.wgetenv(u'TEST_ENV_2'))

        self.assertIsNone(os.environ.get(u'TEST_ENV_1', None))
        self.assertEqual(os.environ[u'TEST_ENV_2'], "test value: 2")
        if orig_env.msvcrt.available():
            self.assertIsNone(orig_env.msvcrt.wgetenv(u'TEST_ENV_1'))
            self.assertEqual(
                orig_env.msvcrt.wgetenv(u'TEST_ENV_2'), "test value: 2")

        orig_env.restore()
        self.assertEqual(orig_env_has_1, u'TEST_ENV_1' in os.environ)
        self.assertEqual(orig_env_has_2, u'TEST_ENV_2' in os.environ)
