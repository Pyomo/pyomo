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

import sys
from io import StringIO
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.repn.tests.lp_diff import lp_diff

_baseline = """\\* Source Pyomo model name=unknown *\\

min
x2:
+ [
+2 x1 ^ 2
] / 2

s.t.

c_l_x3_:
+1 x1
>= 1

c_e_ONE_VAR_CONSTANT:
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= x1 <= +inf
end
"""


def _check_log_and_out(LOG, OUT, offset, msg=None):
    sys.stdout.flush()
    sys.stderr.flush()
    msg = str(msg) + ': ' if msg else ''
    if LOG.getvalue():
        raise RuntimeError(
            "FAIL: %sMessage logged to the Logger:\n>>>\n%s<<<" % (msg, LOG.getvalue())
        )

    if OUT.getvalue():
        raise RuntimeError(
            "FAIL: %sMessage sent to stdout/stderr:\n>>>\n%s<<<" % (msg, OUT.getvalue())
        )


def import_pyomo_environ():
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        import pyomo.environ as pyo

        globals()['pyo'] = pyo
    _check_log_and_out(LOG, OUT, 0)


def run_writer_test():
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        # Enumerate the writers...
        from pyomo.opt import WriterFactory

        info = []
        for writer in sorted(WriterFactory):
            info.append("  %s: %s" % (writer, WriterFactory.doc(writer)))
            _check_log_and_out(LOG, OUT, 10, writer)

    print("Pyomo Problem Writers")
    print("---------------------")
    print('\n'.join(info))

    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        # Test a writer
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.o = pyo.Objective(expr=m.x**2)

        from pyomo.common.tempfiles import TempfileManager

        with TempfileManager:
            fname = TempfileManager.create_tempfile(suffix='pyomo.lp_v1')
            m.write(fname, format='lp_v1')
            with open(fname, 'r') as FILE:
                data = FILE.read()

    base, test = lp_diff(_baseline, data)
    if base != test:
        print(
            "Result did not match baseline.\nRESULT:\n%s\nBASELINE:\n%s"
            % (data, _baseline)
        )
        print(data.strip().splitlines())
        print(_baseline.strip().splitlines())
        sys.exit(2)

    _check_log_and_out(LOG, OUT, 10)


def run_solverfactory_test():
    skip_solvers = {'py', 'xpress', '_xpress_shell', '_mock_xpress'}

    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        info = []
        for solver in sorted(pyo.SolverFactory):
            _doc = pyo.SolverFactory.doc(solver)
            if _doc is not None and 'DEPRECATED' in _doc:
                _avail = 'DEPR'
            elif solver in skip_solvers:
                _avail = 'SKIP'
            else:
                _avail = str(pyo.SolverFactory(solver).available(False))
            info.append("   %s(%s): %s" % (solver, _avail, _doc))
            # _check_log_and_out(LOG, OUT, 20, solver)

        glpk = pyo.SolverFactory('glpk')

    print("")
    print("Pyomo Solvers")
    print("-------------")
    print("\n".join(info))

    if type(glpk.available(False)) != bool:
        print("Solver glpk.available() did not return bool")
        sys.exit(3)

    _check_log_and_out(LOG, OUT, 20)


def run_transformationfactory_test():
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        info = []
        for t in sorted(pyo.TransformationFactory):
            _doc = pyo.TransformationFactory.doc(t)
            info.append("   %s: %s" % (t, _doc))
            if 'DEPRECATED' not in _doc:
                pyo.TransformationFactory(t)
            _check_log_and_out(LOG, OUT, 30, t)

        bigm = pyo.TransformationFactory('gdp.bigm')

    print("")
    print("Pyomo Transformations")
    print("---------------------")
    print('\n'.join(info))

    if not isinstance(bigm, pyo.Transformation):
        print("TransformationFactory(gdp.bigm) did not return a transformation")
        sys.exit(4)

    _check_log_and_out(LOG, OUT, 30)


if __name__ == '__main__':
    # Run some basic tests: map all errors / warnings / exceptions to a
    # non-zero process return code, so that these tests can be run
    # outside of any test harness
    try:
        import_pyomo_environ()
        run_writer_test()
        run_solverfactory_test()
        run_transformationfactory_test()
    except:
        et, e, tb = sys.exc_info()
        print(str(e))
        sys.exit(1)
    sys.exit(0)
