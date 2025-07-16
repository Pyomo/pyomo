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


def load():
    from pyomo.repn.plugins import (
        cpxlp,
        ampl,
        baron_writer,
        mps,
        gams_writer,
        lp_writer,
        nl_writer,
        standard_form,
        parameterized_standard_form,
    )
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
    from pyomo.opt import WriterFactory

    doc = WriterFactory.doc(name)
    WriterFactory.unregister(name)
    WriterFactory.register(name, doc)(WriterFactory.get_class(f'{name}_v{ver}'))


def active_writer_version(name):
    """DEBUGGING TOOL to switch the "default" writer implementation"""
    from pyomo.opt import WriterFactory

    ref = WriterFactory.get_class(name)
    ver = 1
    try:
        while 1:
            if WriterFactory.get_class(f'{name}_v{ver}') is ref:
                return ver
            ver += 1
    except KeyError:
        return None
