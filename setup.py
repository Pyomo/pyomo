#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

"""
Script to generate the installer for coopr.
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
            'coopr.age>=1.1.4',
            'coopr.bilevel>=1.0',
            'coopr.core>=2.0.4',
            'coopr.dae>=1.2',
            'coopr.environ>=1.0.1',
            'coopr.gdp>=1.2',
            'coopr.misc>=2.8.2',
            'coopr.mpec>=1.0',
            'coopr.neos>=1.1.2',
            'coopr.openopt>=1.1.3',
            'coopr.opt>=2.12.2',
            'coopr.os>=1.0.4',
            'coopr.pyomo>=3.6.4',
            'coopr.pysp>=3.5.5',
            'coopr.solvers>=3.2.1',
            'ply',
            'nose',
            'six>=1.6.1'
            ]
if sys.version_info < (2,7):
        requires.append('argparse')
        requires.append('unittest2')
        requires.append('ordereddict')

from setuptools import setup
packages = _find_packages('coopr')

setup(name='Coopr',
      # Note: trunk should have *next* major.minor
      #       VOTD + Final releases will have major.minor.revnum
      # When cutting a release, ALSO update _major/_minor/_micro in 
      #   coopr/coopr/__init__.py
      #   coopr/RELEASE.txt
      version='3.6',
      maintainer='William E. Hart',
      maintainer_email='wehart@sandia.gov',
      url = 'https://software.sandia.gov/coopr',
      license = 'BSD',
      platforms = ["any"],
      description = 'Coopr: a COmmon Optimization Python Repository',
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
      namespace_packages=['coopr'],
      install_requires=requires
      )
