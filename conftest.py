# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pytest

_implicit_markers = {'default'}
_category_markers = {'solver', 'writer'}
_extended_implicit_markers = _implicit_markers.union(_category_markers)


def pytest_configure(config):
    # If the user specified "--solver" or "--writer", then add that
    # logic to the marker expression
    markexpr = config.option.markexpr
    for cat in _category_markers:
        opt = config.getoption('--' + cat)
        if opt:
            if markexpr:
                markexpr = f"({markexpr}) and "
            markexpr += f"{cat}(id='{opt}')"
    config.option.markexpr = markexpr


def pytest_itemcollected(item):
    markers = list(item.iter_markers())
    if not markers:
        # No markers; add the implicit (default) markers
        for marker in _implicit_markers:
            item.add_marker(getattr(pytest.mark, marker))
        return
    # We have historically supported:
    #
    #     @pytest.mark.solve("highs")
    #
    # Unfortunately, pytest doesn't allow for filtering tests with "-m"
    # based on the marker.args.  We will map the positional argument
    # (for pytest.mark.solver and pytest.mark.writer) to the keyword
    # argument "id".
    #
    # We will take this opportunity to also set a keyword argument for
    # the solver/writer "vendor" (defined as the id up to the first
    # underscore).  This will allow running "all Gurobi tests"
    # (including, e.g., lp, direct, and persistent) with
    #
    #     "-m solver(vendor='gurobi')"
    #
    for mark in markers:
        if mark.name not in _category_markers:
            continue
        if mark.args:
            (_id,) = mark.args
            mark.kwargs['id'] = _id
        if 'vendor' not in mark.kwargs:
            mark.kwargs['vendor'] = mark.kwargs['id'].split("_")[0]


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
    if item.config.option.markexpr:
        # PyTest has already filtered tests by the marker.  There is
        # nothing more to check here
        return
    item_markers = set(mark.name for mark in item.iter_markers())
    if item_markers:
        if not _implicit_markers.issubset(item_markers) and not item_markers.issubset(
            _extended_implicit_markers
        ):
            pytest.skip('SKIPPED: Only running default, solver, and unmarked tests.')


def pytest_addoption(parser):
    """
    Add parser options as shorthand for running tests marked by specific
    solvers or writers.
    """
    parser.addoption(
        "--solver",
        action="store",
        metavar="SOLVER",
        help="Run tests matching the requested SOLVER.",
    )
    parser.addoption(
        "--writer",
        action="store",
        metavar="WRITER",
        help="Run tests matching the requested WRITER.",
    )
