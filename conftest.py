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
    # If the user didn't specify a marker expression, then we will
    # select all "default" tests.
    if not markexpr:
        markexpr = 'default'
    config.option.markexpr = markexpr


def pytest_itemcollected(item):
    """Standardize all Pyomo test markers.

    This callback ensures that all unmarked tests, along with all tests
    that are only marked by category markers (e.g., "solver" or
    "writer"), are also marked with the default (implicit) markers
    (currently just "default").

    About category markers
    ----------------------

    We have historically supported "category markers"::

        @pytest.mark.solver("highs")

    Unfortunately, pytest doesn't allow for building marker
    expressions (e.g., for "-m") based on the marker.args.  We will
    map the positional argument (for pytest.mark.solver and
    pytest.mark.writer) to the keyword argument "id".  This will allow
    querying against specific solver interfaces in marker expressions
    with::

        solver(id='highs')

    We will take this opportunity to also set a keyword argument for
    the solver/writer "vendor" (defined as the id up to the first
    underscore).  This will allow running "all Gurobi tests"
    (including, e.g., lp, direct, and persistent) with::

        -m solver(vendor='gurobi')

    As with all pytest markers, these can be combined into more complex
    "marker expressions" using ``and``, ``or``, ``not``, and ``()``.

    """
    markers = list(item.iter_markers())
    if not markers:
        # No markers; add the implicit (default) markers
        for marker in _implicit_markers:
            item.add_marker(getattr(pytest.mark, marker))
        return

    marker_set = {mark.name for mark in markers}
    # If the item is only marked by extended implicit markers (e.g.,
    # solver and/or writer), then make sure it is also marked by all
    # implicit markers (i.e., "default")
    if marker_set.issubset(_extended_implicit_markers):
        for marker in _implicit_markers - marker_set:
            item.add_marker(getattr(pytest.mark, marker))

    # Map any "category" markers (solver or writer) positional arguments
    # to the id keyword, and ensure the 'vendor' keyword is populated
    for mark in markers:
        if mark.name not in _category_markers:
            continue
        if mark.args:
            (_id,) = mark.args
            mark.kwargs['id'] = _id
        if 'vendor' not in mark.kwargs:
            mark.kwargs['vendor'] = mark.kwargs['id'].split("_")[0]


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
