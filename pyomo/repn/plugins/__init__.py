#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

def load():
    import pyomo.repn.plugins.cpxlp
    import pyomo.repn.plugins.ampl
    import pyomo.repn.plugins.baron_writer
    import pyomo.repn.plugins.mps
    import pyomo.repn.plugins.gams_writer
    import pyomo.repn.plugins.nl_writer

    from pyomo.opt import WriterFactory
    WriterFactory.register('nl', 'Generate the corresponding AMPL NL file.')(
        WriterFactory.get_class('nl_v1'))
