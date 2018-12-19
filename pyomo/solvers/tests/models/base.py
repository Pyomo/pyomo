#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from os.path import join, dirname, abspath
import json
import six

import pyutilib.th as unittest

import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core import Suffix, Var, Constraint, Objective
from pyomo.opt import ProblemFormat, SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver

thisDir = dirname(abspath( __file__ ))

_test_models = {}


@unittest.nottest
def test_models(arg=None):
    if arg is None:
        return _test_models
    else:
        return _test_models[arg]


def register_model(cls):
    """ Decorator for test model classes """
    global _test_models
    assert cls.__name__ not in _test_models
    _test_models[cls.__name__] = cls
    return cls


class _BaseTestModel(object):
    """
    This is a base class for test models
    """

    description = "unknown"
    level = ('smoke', 'nightly', 'expensive')
    capabilities = set([])
    test_pickling = True

    def __init__(self):
        self.model = None
        self.results_file = None
        self.disable_suffix_tests = False
        self.test_suffixes = []
        self.diff_tol = 1e-4
        self.solve_should_fail = False

    def add_results(self, filename):
        """ Add results file """
        self.results_file = join(thisDir, filename)

    def generate_model(self, import_suffixes=[]):
        """ Generate the model """
        self._generate_model()
        # Add suffixes
        self.test_suffixes = [] if self.disable_suffix_tests else \
                        import_suffixes
        if isinstance(self.model, IBlock):
            for suffix in self.test_suffixes:
                setattr(self.model, suffix, pmo.suffix(direction=pmo.suffix.IMPORT))
        else:
            for suffix in self.test_suffixes:
                setattr(self.model, suffix, Suffix(direction=Suffix.IMPORT))

    def solve(self,
              solver,
              io,
              io_options,
              solver_options,
              symbolic_labels,
              load_solutions):
        """ Optimize the model """
        assert self.model is not None

        opt = SolverFactory(solver, solver_io=io)
        opt.options.update(solver_options)

        if io == 'nl':
            assert opt.problem_format() == ProblemFormat.nl
        elif io == 'lp':
            assert opt.problem_format() == ProblemFormat.cpxlp
        elif io == 'mps':
            assert opt.problem_format() == ProblemFormat.mps
        #elif io == 'python':
        #    print opt.problem_format()
        #    assert opt.problem_format() is None

        try:
            if isinstance(opt, PersistentSolver):
                opt.set_instance(self.model, symbolic_solver_labels=symbolic_labels)
                if opt.warm_start_capable():
                    results = opt.solve(warmstart=True,
                                        load_solutions=load_solutions,
                                        **io_options)
                else:
                    results = opt.solve(load_solutions=load_solutions,
                                        **io_options)
            else:
                if opt.warm_start_capable():
                    results = opt.solve(
                        self.model,
                        symbolic_solver_labels=symbolic_labels,
                        warmstart=True,
                        load_solutions=load_solutions,
                        **io_options)
                else:
                    results = opt.solve(
                        self.model,
                        symbolic_solver_labels=symbolic_labels,
                        load_solutions=load_solutions,
                        **io_options)

            return opt, results
        finally:
            pass
            #opt.deactivate()
        del opt
        return None, None

    def save_current_solution(self, filename, **kwds):
        """ Save the solution in a specified file name """
        assert self.model is not None
        model = self.model
        suffixes = dict((suffix, getattr(model,suffix))
                        for suffix in kwds.pop('suffixes',[]))
        for suf in suffixes.values():
            if isinstance(self.model, IBlock):
                assert isinstance(suf,pmo.suffix)
                assert suf.import_enabled
            else:
                assert isinstance(suf,Suffix)
                assert suf.import_enabled()

        with open(filename,'w') as f:
            #
            # Collect Block, Variable, Constraint, Objective and Suffix data
            #
            soln = {}
            for block in model.block_data_objects():
                soln[block.name] = {}
                for suffix_name, suffix in suffixes.items():
                    if suffix.get(block) is not None:
                        soln[block.name][suffix_name] = suffix.get(block)
            for var in model.component_data_objects(Var):
                soln[var.name] = {}
                soln[var.name]['value'] = var.value
                soln[var.name]['stale'] = var.stale
                for suffix_name, suffix in suffixes.items():
                    if suffix.get(var) is not None:
                        soln[var.name][suffix_name] = suffix.get(var)
            for con in model.component_data_objects(Constraint):
                soln[con.name] = {}
                con_value = con(exception=False)
                soln[con.name]['value'] = con_value
                for suffix_name, suffix in suffixes.items():
                    if suffix.get(con) is not None:
                        soln[con.name][suffix_name] = suffix.get(con)
            for obj in model.component_data_objects(Objective):
                soln[obj.name] = {}
                obj_value = obj(exception=False)
                soln[obj.name]['value'] = obj_value
                for suffix_name, suffix in suffixes.items():
                    if suffix.get(obj) is not None:
                        soln[obj.name][suffix_name] = suffix.get(obj)
            #
            # Write the results
            #
            json.dump(soln, f, indent=2, sort_keys=True)

    def validate_current_solution(self, **kwds):
        """
        Validate the solution
        """
        assert self.model is not None
        assert self.results_file is not None
        model = self.model
        suffixes = dict((suffix, getattr(model,suffix))
                        for suffix in kwds.pop('suffixes',[]))
        for suf in suffixes.values():
            if isinstance(self.model, IBlock):
                assert isinstance(suf,pmo.suffix)
                assert suf.import_enabled
            else:
                assert isinstance(suf,Suffix)
                assert suf.import_enabled()
        solution = None
        error_str = ("Difference in solution for {0}.{1}:\n\tBaseline "
                     "- {2}\n\tCurrent - {3}")

        with open(self.results_file,'r') as f:
            try:
                solution = json.load(f)
            except:
                return (False,"Problem reading file "+self.results_file)

        for var in model.component_data_objects(Var):
            var_value_sol = solution[var.name]['value']
            var_value = var.value
            if not ((var_value is None) and (var_value_sol is None)):
                if ((var_value is None) ^ (var_value_sol is None)) or \
                   (abs(var_value_sol - var_value) > self.diff_tol):
                    return (False,
                            error_str.format(var.name,
                                             'value',
                                             var_value_sol,
                                             var_value))
            if not (solution[var.name]['stale'] is var.stale):
                return (False,
                        error_str.format(var.name,
                                         'stale',
                                         solution[var.name]['stale'],
                                         var.stale))
            for suffix_name, suffix in suffixes.items():
                if suffix_name in solution[var.name]:
                    if suffix.get(var) is None:
                        if not(solution[var.name][suffix_name] in \
                               solution["suffix defaults"][suffix_name]):
                            return (False,
                                    error_str.format(
                                        var.name,
                                        suffix,
                                        solution[var.name][suffix_name],
                                        "none defined"))
                    elif not abs(solution[var.name][suffix_name] - \
                                 suffix.get(var)) < self.diff_tol:
                        return (False,
                                error_str.format(
                                    var.name,
                                    suffix,
                                    solution[var.name][suffix_name],
                                    suffix.get(var)))

        for con in model.component_data_objects(Constraint):
            con_value_sol = solution[con.name]['value']
            con_value = con(exception=False)
            if not ((con_value is None) and (con_value_sol is None)):
                if ((con_value is None) ^ (con_value_sol is None)) or \
                   (abs(con_value_sol - con_value) > self.diff_tol):
                    return (False,
                            error_str.format(con.name,
                                             'value',
                                             con_value_sol,
                                             con_value))
            for suffix_name, suffix in suffixes.items():
                if suffix_name in solution[con.name]:
                    if suffix.get(con) is None:
                        if not (solution[con.name][suffix_name] in \
                                solution["suffix defaults"][suffix_name]):
                            return (False,
                                    error_str.format(
                                        con.name,
                                        suffix,
                                        solution[con.name][suffix_name],
                                        "none defined"))
                    elif not abs(solution[con.name][suffix_name] - \
                                 suffix.get(con)) < self.diff_tol:
                        return (False,
                                error_str.format(
                                    con.name,
                                    suffix,
                                    solution[con.name][suffix_name],
                                    suffix.get(con)))

        for obj in model.component_data_objects(Objective):
            obj_value_sol = solution[obj.name]['value']
            obj_value = obj(exception=False)
            if not ((obj_value is None) and (obj_value_sol is None)):
                if ((obj_value is None) ^ (obj_value_sol is None)) or \
                   (abs(obj_value_sol - obj_value) > self.diff_tol):
                    return (False,
                            error_str.format(obj.name,
                                             'value',
                                             obj_value_sol,
                                             obj_value))
            for suffix_name, suffix in suffixes.items():
                if suffix_name in solution[obj.name]:
                    if suffix.get(obj) is None:
                        if not(solution[obj.name][suffix_name] in \
                               solution["suffix defaults"][suffix_name]):
                            return (False,
                                    error_str.format(
                                        obj.name,
                                        suffix,
                                        solution[obj.name][suffix_name],
                                        "none defined"))
                    elif not abs(solution[obj.name][suffix_name] - \
                                 suffix.get(obj)) < self.diff_tol:
                        return (False,
                                error_str.format(
                                    obj.name,
                                    suffix,
                                    solution[obj.name][suffix_name],
                                    suffix.get(obj)))

        first=True
        for block in model.block_data_objects():
            if first:
                first=False
                continue
            for suffix_name, suffix in suffixes.items():
                if (solution[block.name] is not None) and \
                   (suffix_name in solution[block.name]):
                    if suffix.get(block) is None:
                        if not(solution[block.name][suffix_name] in \
                               solution["suffix defaults"][suffix_name]):
                            return (False,
                                    error_str.format(
                                        block.name,
                                        suffix,
                                        solution[block.name][suffix_name],
                                        "none defined"))
                    elif not abs(solution[block.name][suffix_name] - \
                                 suffix.get(block)) < sefl.diff_tol:
                        return (False,
                                error_str.format(
                                    block.name,
                                    suffix,
                                    solution[block.name][suffix_name],
                                    suffix.get(block)))
        return (True,"")

    def validate_capabilities(self, opt):
        """ Validate the capabilites of the optimizer """
        if (self.linear is True) and \
           (not opt.has_capability('linear') is True):
            return False
        if (self.integer is True) and \
           (not opt.has_capability('integer') is True):
            return False
        if (self.quadratic_objective is True) and \
           (not opt.has_capability('quadratic_objective') is True):
            return False
        if (self.quadratic_constraint is True) and \
           (not opt.has_capability('quadratic_constraint') is True):
            return False
        if (self.sos1 is True) and \
           (not opt.has_capability('sos1') is True):
            return False
        if (self.sos2 is True) and \
           (not opt.has_capability('sos2') is True):
            return False
        return True

    def post_solve_test_validation(self, tester, results):
        """ Perform post-solve validation tests """
        if tester is None:
            assert results['Solver'][0]['termination condition'] == TerminationCondition.optimal
        else:
            tester.assertEqual(results['Solver'][0]['termination condition'], TerminationCondition.optimal)

    def warmstart_model(self):
        """ Initialize model parameters """
        pass


if __name__ == "__main__":
    import pyomo.solvers.tests.models
    for key, value in six.iteritems(_test_models):
        print(key)
        obj = value()
        obj.generate_model()
        obj.warmstart_model()

