#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

"""
Script to generate the installer for pyomo.
"""

import sys
import glob
import os

def _find_packages(path):
    """
    Generate a list of nested packages
    """
    pkg_list=[]
    if not os.path.exists(path):
        return []
    if not os.path.exists(path+os.sep+"__init__.py"):
        return []
    else:
        pkg_list.append(path)
    for root, dirs, files in os.walk(path, topdown=True):
        if root in pkg_list and "__init__.py" in files:
            for name in dirs:
                if os.path.exists(root+os.sep+name+os.sep+"__init__.py"):
                    pkg_list.append(root+os.sep+name)
    return [pkg for pkg in map(lambda x:x.replace(os.sep,"."), pkg_list)]

def read(*rnames):
    return open(os.path.join(os.path.dirname(__file__), *rnames)).read()

requires=[
            'PyUtilib>=4.7.3340',
            'ply',
            'nose',
            'six>=1.6.1'
            ]
if sys.version_info < (2,7):
        requires.append('argparse')
        requires.append('unittest2')
        requires.append('ordereddict')

from setuptools import setup
packages = _find_packages('pyomo')

setup(name='Pyomo',
      # Note: trunk should have *next* major.minor
      #       VOTD + Final releases will have major.minor.revnum
      # When cutting a release, ALSO update _major/_minor/_micro in 
      #   pyomo/pyomo/__init__.py
      #   pyomo/RELEASE.txt
      version='3.6',
      maintainer='William E. Hart',
      maintainer_email='wehart@sandia.gov',
      url = 'https://software.sandia.gov/pyomo',
      license = 'BSD',
      platforms = ["any"],
      description = 'Pyomo: Python Optimization Modeling Objects',
      long_description = read('README.txt'),
      classifiers = [
            'Development Status :: 4 - Beta',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Programming Language :: Unix Shell',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development :: Libraries :: Python Modules'
        ],
      packages=packages,
      keywords=['optimization'],
      namespace_packages=['pyomo'],
      install_requires=requires
      )
