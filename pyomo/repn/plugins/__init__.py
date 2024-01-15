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
    import pyomo.repn.plugins.lp_writer
    import pyomo.repn.plugins.nl_writer
    import pyomo.repn.plugins.standard_form

    from pyomo.opt import WriterFactory

    # Register the "default" versions of writers that have more than one
    # implementation
    WriterFactory.register('nl', 'Generate the corresponding AMPL NL file.')(
        WriterFactory.get_class('nl_v2')
    )
    WriterFactory.register('lp', 'Generate the corresponding CPLEX LP file.')(
        WriterFactory.get_class('lp_v2')
    )
    WriterFactory.register('cpxlp', 'Generate the corresponding CPLEX LP file.')(
        WriterFactory.get_class('cpxlp_v2')
    )


def activate_writer_version(name, ver):
    """DEBUGGING TOOL to switch the "default" writer implementation"""
    doc = WriterFactory.doc(name)
    WriterFactory.unregister(name)
    WriterFactory.register(name, doc)(WriterFactory.get_class(f'{name}_v{ver}'))
