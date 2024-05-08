#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pytest

_implicit_markers = {'default'}
_extended_implicit_markers = _implicit_markers.union({'solver'})


def pytest_collection_modifyitems(items):
    """
    This method will mark any unmarked tests with the implicit marker ('default')

    """
    for item in items:
        try:
            next(item.iter_markers())
        except StopIteration:
            for marker in _implicit_markers:
                item.add_marker(getattr(pytest.mark, marker))


def pytest_runtest_setup(item):
    """
    This method overrides pytest's default behavior for marked tests.

    The logic below follows this flow:
        1) Did the user ask for a specific solver using the '--solver' flag?
            If so: Add skip statements to any test NOT labeled with the
            requested solver category.
        2) Did the user ask for a specific marker using the '-m' flag?
            If so: Return to pytest's default behavior.
        3) If the user requested no specific solver or marker, look at each
           test for the following:
            a) If unmarked, run the test
            b) If marked with implicit_markers, run the test
            c) If marked "solver" and NOT any explicit marker, run the test
            OTHERWISE: Skip the test.
    In other words - we want to run unmarked, implicit, and solver tests as
    the default mode; but if solver tests are also marked with an explicit
    category (e.g., "expensive"), we will skip them.
    """
    solvernames = [mark.args[0] for mark in item.iter_markers(name="solver")]
    solveroption = item.config.getoption("--solver")
    markeroption = item.config.getoption("-m")
    item_markers = set(mark.name for mark in item.iter_markers())
    if solveroption:
        if solveroption not in solvernames:
            pytest.skip("SKIPPED: Test not marked {!r}".format(solveroption))
            return
    elif markeroption:
        return
    elif item_markers:
        if not _implicit_markers.issubset(item_markers) and not item_markers.issubset(
            _extended_implicit_markers
        ):
            pytest.skip('SKIPPED: Only running default, solver, and unmarked tests.')


def pytest_addoption(parser):
    """
    Add another parser option to specify suite of solver tests to run
    """
    parser.addoption(
        "--solver",
        action="store",
        metavar="SOLVER",
        help="Run tests matching the requested SOLVER.",
    )


def pytest_configure(config):
    """
    Register additional solver marker, as applicable.
    This stops pytest from printing a warning about unregistered solver options.
    """
    config.addinivalue_line(
        "markers", "solver(name): mark test to run the named solver"
    )
