#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import sys
from os.path import abspath, dirname, join
currdir = dirname(abspath(__file__))+os.sep
import shutil
import subprocess
import traceback

import pyutilib.th as unittest

pyomo_install = abspath(join(dirname(dirname(dirname(dirname(currdir)))), 'scripts', 'pyomo_install'))
vpy_install = abspath(join(dirname(sys.executable),'vpy_install'))
test_zipfile = os.environ.get("PYOMO_INSTALLER_ZIPFILE", None)

def call_subprocess(cmd, stdout=False, exception=False):
    env = os.environ.copy()
    cwd = os.getcwd()
    print("Testing with subcommand: "+' '.join(cmd))
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdin=None, stdout=subprocess.PIPE, cwd=cwd, env=env)
    except Exception:
        e = sys.exc_info()[1]
        print("Error %s while executing command '%s'" % (e, cmd))
        raise
    _out, _err = proc.communicate()
    if exception and proc.returncode:
        raise RuntimeError("Error running command: %s\nOutput:\n%s" %
                           ( ' '.join(cmd), _out ) )
    if not stdout:
        return _out
    else:
        print(_out)
        return None

if __name__ == "__main__":
    include_in_all = True
else:
    include_in_all = False


@unittest.category('installer', include_in_all=include_in_all)
class Tests(unittest.TestCase):

    def validate(self, venv):
        if not os.path.exists(venv):
            raise RuntimeError("Missing installation venv: "+abspath(venv))
        #
        pythonexe = join(venv, 'bin', 'python')
        if not os.path.exists(pythonexe):
            raise RuntimeError("Missing python command: "+abspath(pythonexe))
        #
        venv_output = call_subprocess([pythonexe, '--version'])
        #if sys.version_info[:2] < (3,0):
        #    output = '\n'.join([str(line) for line in venv_output])
        #else:
        #    output = '\n'.join([str(line, encoding='utf-8') for line in venv_output])
        output = venv_output.strip()
        venv_version = output.split(' ')[1]
        venv_version = str(venv_version.rstrip())
        sys_version = ".".join(str(x) for x in sys.version_info[:3])
        sys_version = str(sys_version.rstrip())
        if venv_version != sys_version:
            #print("HERE '%s' '%s'" % (venv_version, sys_version))
            raise RuntimeError("Missing test python version != venv python version: %s != %s" % (sys_version, venv_version))
        #    
        check_output = call_subprocess([pythonexe, join(currdir,'check.py')], stdout=False)
        #if sys.version_info[:2] < (3,0):
        #    output = "\n".join([line for line in check_output])
        #else:
        #    output = "\n".join([str(line, encoding='utf-8') for line in check_output])
        output = check_output.strip()
        if output != 'OK':
            raise RuntimeError("Installation configuration error: "+output)

    def run_installer(self, name, pypi=False, zipfile=False, trunk=False, system=False, user=False, venv=False, error=False, offline=False):
        testdir=abspath(join(currdir,name))
        if os.path.exists(testdir):
            shutil.rmtree(testdir)
        os.mkdir(testdir)
        os.chdir(testdir)
        os.environ['PYTHONUSERBASE'] = testdir
        os.environ['PYTHON_EGG_CACHE'] = testdir
        if zipfile and not test_zipfile:
            shutil.rmtree(testdir)
            self.skipTest("Cannot test zipfile installation: zipfile not specified through the PYOMO_INSTALLER_ZIPFILE environment variable")
        #
        try:
            proxy = {}
            if offline:
                for name_ in ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
                    if name_ in os.environ:
                        proxy[name_] = os.environ[name_]
                    else:
                        proxy[name_] = '_del_'
                    os.environ[name_] = 'http://www.bad.proxy.org:80'
            if venv:
                cmd = [pyomo_install, '--venv='+name, '-p', sys.executable]
                if zipfile:
                    cmd.append('--zip='+test_zipfile)
                elif trunk:
                    cmd.append('--trunk')
                call_subprocess(cmd, stdout=True, exception=True)
            else:  # system or user
                call_subprocess([vpy_install, name, '-p', sys.executable], stdout=True, exception=True)
                os.chdir(name)
                pythonexe = join(os.getcwd(), 'bin', 'python')
                cmd = [pythonexe, pyomo_install]
                if user:
                    cmd.append('--user')
                if zipfile:
                    cmd.append('--zip='+test_zipfile)
                elif trunk:
                    cmd.append('--trunk')
                call_subprocess(cmd, stdout=True, exception=True)
                os.chdir(testdir)
            self.validate(name)
        except Exception:
            if not error:
                e, tb = sys.exc_info()[1:3]
                self.fail("Unexpected exception: '%s'\nTraceback:\n%s" % 
                          ( str(e), ''.join(traceback.format_tb(tb)) ))
        else:
            if error:
                self.fail("Expected the installation to fail, but no exception was raised")
        finally:
            for name_ in proxy:
                if proxy[name_] == '_del_':
                    del os.environ[name_]
                else:
                    os.environ[name_] = proxy[name_]
            proxy = {}
            os.chdir(currdir)
            if os.path.exists(testdir):
                shutil.rmtree(testdir)
            
    # Install to System

    def test_1a_fromPyPI_toSystem(self):
        self.run_installer('1a', pypi=True, system=True)

    def test_2a_fromZip_toSystem_offline(self):
        self.run_installer('2a', zipfile=True, system=True, offline=True)

    # Install to User - OBSOLETE

    def Xtest_1b_fromPyPI_toUser(self):
        self.run_installer('1b', pypi=True, user=True)

    def Xtest_2b_fromZip_toUser_offline(self):
        self.run_installer('2b', zipfile=True, user=True, offline=True)

    # Install to Venv

    def test_1c_fromPyPI_toVEnv(self):
        self.run_installer('1c', pypi=True, venv=True)

    def test_2c_fromZip_toVEnv_offline(self):
        self.run_installer('2c', zipfile=True, venv=True, offline=True)

    # TODO
    def Xtest_3c_fromSrc_toVEnv_offline(self):
        self.run_installer('3c', srcdir=True, venv=True, offline=True)

    def test_4c_fromTrunk_toVEnv(self):
        self.run_installer('4c', trunk=True, venv=True)


    # ERROR TESTS

    # System

    def test_1a_fromPyPI_toSystem_offline_expectError(self):
        self.run_installer('1a_offline', pypi=True, system=True, offline=True, error=True)

    def test_3a_fromTrunk_toSystem_expectError(self):
        self.run_installer('3a', trunk=True, system=True, error=True)

    # User - OBSOLETE

    def Xtest_1b_fromPyPI_toUser_offline_expectError(self):
        self.run_installer('1b_offline', pypi=True, user=True, offline=True, error=True)

    def Xtest_3b_fromTrunk_toUser_expectError(self):
        self.run_installer('3b', trunk=True, user=True, error=True)

    # Venv

    def test_1c_fromPyPI_toVEnv_offline_expectError(self):
        self.run_installer('1c_offline', pypi=True, venv=True, offline=True, error=True)

    def test_3c_fromTrunk_toVEnv_offline_expectError(self):
        self.run_installer('3c_offline', trunk=True, venv=True, offline=True, error=True)


if __name__ == "__main__":
    unittest.main()

