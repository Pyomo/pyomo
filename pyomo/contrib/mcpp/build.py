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
import shutil
import tempfile

import distutils.core
from distutils.command.build_ext import build_ext
from setuptools.extension import Extension

from pyomo.common.config import PYOMO_CONFIG_DIR
from pyomo.common.fileutils import this_file_dir, find_dir

def _generate_configuration():
    # Try and find MC++.  Defer to the MCPP_ROOT if it is set;
    # otherwise, look in common locations for a mcpp directory.
    pathlist=[
        os.path.join(PYOMO_CONFIG_DIR, 'src'),
        this_file_dir(),
    ]
    if 'MCPP_ROOT' in os.environ:
        mcpp = os.environ['MCPP_ROOT']
    else:
        mcpp = find_dir('mcpp', cwd=True, pathlist=pathlist)
    if mcpp:
        print("Found MC++ at %s" % ( mcpp, ))
    else:
        raise RuntimeError(
            "Cannot identify the location of the MCPP source distribution")

    #
    # Configuration for this extension
    #
    project_dir = this_file_dir()
    sources = [
        os.path.join(project_dir, 'mcppInterface.cpp'),
    ]
    include_dirs = [
        os.path.join(mcpp, 'src', 'mc'),
        os.path.join(mcpp, 'src', '3rdparty', 'fadbad++'),
    ]

    mcpp_ext = Extension(
        "mcppInterface",
        sources=sources,
        language="c++",
        extra_compile_args=[],
        include_dirs=include_dirs,
        library_dirs=[],
        libraries=[],
    )
    
    package_config = {
        'name': 'mcpp',
        'packages': [],
        'ext_modules': [mcpp_ext],
    }

    return package_config


class _BuildWithoutPlatformInfo(build_ext, object):
    # Python3.x puts platform information into the generated SO file
    # name, which is usually fine for python extensions, but since this
    # is not a "real" extension, we will hijack things to remove the
    # platform information from the filename so that Pyomo can more
    # easily locate it.  Note that build_ext is not a new-style class in
    # Python 2.7, so we will add an explicit inheritance from object so
    # that super() works.
    def get_ext_filename(self, ext_name):
        filename = super(_BuildWithoutPlatformInfo, self).get_ext_filename(
            ext_name).split('.')
        filename = '.'.join([filename[0],filename[-1]])
        return filename

def build_mcpp():
    package_config = _generate_configuration()
    package_config['cmdclass'] = {'build_ext': _BuildWithoutPlatformInfo}
    dist = distutils.core.Distribution(package_config)
    install_dir = os.path.join(PYOMO_CONFIG_DIR, 'lib')
    dist.get_command_obj('install_lib').install_dir = install_dir
    print("**** Building library ****")
    try:
        basedir = os.path.abspath(os.path.curdir)
        tmpdir = os.path.abspath(tempfile.mkdtemp())
        print("   tmpdir = %s" % ( tmpdir, ))
        os.chdir(tmpdir)
        dist.run_command('install_lib')
        print("Installed mcppInterface to %s" % ( install_dir, ))
    finally:
        os.chdir(basedir)
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    build_mcpp()

