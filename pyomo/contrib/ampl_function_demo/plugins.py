#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.extensions import ExtensionBuilderFactory
from pyomo.contrib.ampl_function_demo.build import AMPLFunctionDemoBuilder


def load():
    ExtensionBuilderFactory.register('ampl_function_demo')(AMPLFunctionDemoBuilder)
