#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
from .interval_data import (
    assert_disjoint_intervals,
    load_inputs_into_model,
    interval_data_from_time_series,
    time_series_from_interval_data,
)
from .find_nearest_index import (
    find_nearest_index,
)

__doc__ = (
    """A module of data structures for storing values associated with """
    """time-indexed variables."""
)
