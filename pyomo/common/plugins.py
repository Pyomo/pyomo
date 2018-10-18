#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

import pyomo.common.download
import pyomo.common.getGSL

def load():
    pass

pyomo.common.download.DownloadFactory.register(
    'gsl')(pyomo.common.getGSL.main)
