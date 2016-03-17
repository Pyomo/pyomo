#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from six import iteritems
from pyomo.opt import SolverFactory
from pyomo.opt.base.solvers import UnknownSolver
import pyomo.environ

class SolverTestCase(object):

    def __init__(self,name=None,io=None,**kwds):
        assert (name is not None) and (type(name) is str)
        assert (io is not None) and (type(io) is str)
        self.name = name
        self.io = io
        self.capabilities = kwds.pop('capabilities',[])
        self.export_suffixes = kwds.pop('export_suffixes',[])
        self.import_suffixes = kwds.pop('import_suffixes',[])
        self.options = kwds.pop('options',{})
        self.io_options = kwds.pop('io_options',{})
        assert type(self.capabilities) in [list,tuple]
        assert type(self.export_suffixes) in [list,tuple]
        assert type(self.import_suffixes) in [list,tuple]
        assert type(self.options) is dict
        for tag in self.capabilities:
            assert type(tag) is str
        for tag in self.export_suffixes:
            assert type(tag) is str
        for tag in self.import_suffixes:
            assert type(tag) is str
        self.solver = None
        self.initialize()

    def initialize(self):
        if self.solver is not None:
            self.solver.deactivate()
        self.solver = None
        opt = None
        try:
            opt = SolverFactory(self.name,solver_io=self.io)
        except:
            pass
        if isinstance(opt, UnknownSolver):
            opt = None
        if opt is not None:
            for key,value in iteritems(self.options):
                opt.options[key] = value
        self.solver = opt
        self.available = (self.solver is not None) and \
                         (self.solver.available(exception_flag=False)) and \
                         ((not hasattr(self.solver,'executable')) or \
                          (self.solver.executable() is not None))
        return self.solver, self.io_options

    def has_capability(self,tag):
        if tag in self.capabilities:
            return True
        return False

    def __str__(self):
        tmp  = "SolverTestCase:\n"
        tmp += "\tname = "+self.name+"\n"
        tmp += "\tio = "+self.io+"\n"
        tmp += "\tavailable = "+str(self.available)+"\n"
        if self.solver is not None:
            tmp += "\tversion = "+str(self.solver.version())+"\n"
        else:
            tmp += "\tversion = unknown\n"
        tmp += "\tcapabilities: "
        if len(self.capabilities):
            tmp += "\n"
            for tag in self.capabilities:
                tmp += "\t   - "+tag+"\n"
        else:
            tmp += "None\n"
        tmp += "\tsuffixes: \n"
        tmp += "\t  export: "
        if len(self.export_suffixes):
            tmp += "\n"
            for tag in self.export_suffixes:
                tmp += "\t   - "+tag+"\n"
        else:
            tmp += "None\n"
        tmp += "\t  import: "
        if len(self.import_suffixes):
            tmp += "\n"
            for tag in self.import_suffixes:
                tmp += "\t   - "+tag+"\n"
        else:
            tmp += "None\n"
        tmp += "\toptions: "+str(self.options)+"\n"
        tmp += "\tio_options: "+str(self.io_options)+"\n"
        return tmp

# The capabilities listed below should be what is
# advertised by the solver and not the Pyomo plugin
# But we are testing that they match
testCases = []

#
#    ADD CPLEX TEST CASES
#
cplex_capabilities = ['linear',
                      'integer',
                      'quadratic_objective',
                      'quadratic_constraint',
                      'sos1',
                      'sos2']
testCases.append(SolverTestCase(name='cplex',
                                io='lp',
                                capabilities=cplex_capabilities,
                                import_suffixes=['slack','dual','rc']))
testCases.append(SolverTestCase(name='cplex',
                                io='mps',
                                capabilities=cplex_capabilities,
                                import_suffixes=['slack','dual','rc']))
testCases.append(SolverTestCase(name='cplex',
                                io='nl',
                                capabilities=cplex_capabilities,
                                import_suffixes=['dual']))
testCases.append(SolverTestCase(name='cplex',
                                io='python',
                                capabilities=cplex_capabilities,
                                import_suffixes=['slack','dual','rc']))
testCases.append(SolverTestCase(name='_cplex_persistent',
                                io='python',
                                capabilities=cplex_capabilities,
                                import_suffixes=['slack','dual','rc']))

#
#    ADD GUROBI TEST CASES
#
gurobi_capabilities = ['linear',
                       'integer',
                       'quadratic_objective',
                       'quadratic_constraint',
                       'sos1',
                       'sos2']
# **NOTE: Gurobi does not handle quadratic constraints before
#         Major Version 5
testCases.append(SolverTestCase(name='gurobi',
                                io='lp',
                                capabilities=gurobi_capabilities,
                                import_suffixes=['rc','dual','slack']))
testCases.append(SolverTestCase(name='gurobi',
                                io='mps',
                                capabilities=gurobi_capabilities,
                                import_suffixes=['rc','dual','slack']))
testCases.append(SolverTestCase(name='gurobi',
                                io='nl',
                                capabilities=gurobi_capabilities,
                                import_suffixes=['dual'],
                                options={'qcpdual':1,'simplex':1}))
testCases.append(SolverTestCase(name='gurobi',
                                io='python',
                                capabilities=gurobi_capabilities,
                                import_suffixes=['rc','dual','slack']))

#
#    ADD GLPK TEST CASES
#
glpk_capabilities = ['linear',
                     'integer']

if 'GLPKSHELL_old' in str(pyomo.solvers.plugins.solvers.GLPK.GLPK().__class__):
    glpk_import_suffixes = ['dual']
else:
    glpk_import_suffixes = ['rc','dual']
testCases.append(SolverTestCase(name='glpk',
                                io='lp',
                                capabilities=glpk_capabilities,
                                import_suffixes=glpk_import_suffixes))
testCases.append(SolverTestCase(name='glpk',
                                io='mps',
                                capabilities=glpk_capabilities,
                                import_suffixes=glpk_import_suffixes,
                                io_options={"skip_objective_sense": True}))
testCases.append(SolverTestCase(name='glpk',
                                io='python',
                                capabilities=glpk_capabilities,
                                import_suffixes=[]))


#
#    ADD CBC TEST CASES
#
cbc_lp_capabilities = ['linear',
                       'integer']
testCases.append(SolverTestCase(name='cbc',
                                io='lp',
                                capabilities=cbc_lp_capabilities,
                                import_suffixes=['rc','dual']))
#cbc_mps_capabilities = ['linear',
#                        'integer',
#                        'sos1',
#                        'sos2']
#testCases.append(SolverTestCase(name='cbc',
#                                io='mps',
#                                capabilities=cbc_mps_capabilities,
#                                import_suffixes=['rc','dual']))
cbc_nl_capabilities = ['linear',
                       'integer',
                       'sos1',
                       'sos2']
testCases.append(SolverTestCase(name='cbc',
                                io='nl',
                                capabilities=cbc_nl_capabilities,
                                import_suffixes=['dual']))

#
#    ADD PICO TEST CASES
#
pico_capabilities = ['linear',
                     'integer']
testCases.append(SolverTestCase(name='pico',
                                io='lp',
                                capabilities=pico_capabilities,
                                import_suffixes=['dual']))
#testCases.append(SolverTestCase(name='pico',
#                                io='mps',
#                                capabilities=pico_capabilities,
#                                import_suffixes=['dual']))
testCases.append(SolverTestCase(name='pico',
                                io='nl',
                                capabilities=pico_capabilities,
                                import_suffixes=['dual']))
#
#    ADD XPRESS TEST CASES
#
xpress_capabilities = ['linear',
                     'integer',
                     'quadratic_objective',
                     'quadratic_constraint',
                     'sos1',
                     'sos2']
testCases.append(SolverTestCase(name='xpress',
                                io='lp',
                                capabilities=xpress_capabilities,
                                import_suffixes=['rc','dual','slack']))
testCases.append(SolverTestCase(name='xpress',
                                io='mps',
                                capabilities=xpress_capabilities,
                                import_suffixes=['rc','dual','slack']))
testCases.append(SolverTestCase(name='xpress',
                                io='nl',
                                capabilities=xpress_capabilities,
                                import_suffixes=['dual']))

#
#    ADD IPOPT TEST CASES
#
ipopt_capabilities = ['linear',
                      'quadratic_objective',
                      'quadratic_constraint']
testCases.append(SolverTestCase(name='ipopt',
                                io='nl',
                                capabilities=ipopt_capabilities,
                                import_suffixes=['dual']))
#
#    ADD SCIP TEST CASES
#
scip_capabilities = ['linear',
                     'integer',
                     'quadratic_objective',
                     'quadratic_constraint',
                     'sos1',
                     'sos2']
testCases.append(SolverTestCase(name='scip',
                                io='nl',
                                capabilities=scip_capabilities,
                                import_suffixes=[]))

#
#    ADD BARON TEST CASES
#
baron_capabilities = ['linear',
                     'integer',
                     'quadratic_objective',
                     'quadratic_constraint']
testCases.append(SolverTestCase(name='baron',
                                io='bar',
                                capabilities=baron_capabilities,
                                import_suffixes=['rc','dual']))

#
#    ADD KNITROAMPL TEST CASES
#
knitroampl_capabilities = ['linear',
                           'integer',
                           'quadratic_objective',
                           'quadratic_constraint']
testCases.append(SolverTestCase(name='knitroampl',
                                io='nl',
                                capabilities=knitroampl_capabilities,
                                import_suffixes=['dual']))

if __name__ == "__main__":

    for case in testCases:
        case.initialize()
        print(case)
