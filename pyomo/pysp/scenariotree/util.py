#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import hashlib
import uuid

pysp_namespace_hash = hashlib.md5('pysp'.encode())
def _compute_namespace(node_name):
    node_namespace_hash = pysp_namespace_hash.copy()
    node_namespace_hash.update(node_name.encode())
    return uuid.UUID(bytes=node_namespace_hash.digest())

#
# 32-bit family
#

_max_int32 = 2**31 - 1
_max_uint32 = 2**32 - 1

def _convert_range_zero_to_max_int32(x):
    assert x >= 0
    # use int for py2 to eliminate long
    # types when possible
    return int(x % (_max_int32+1))
def _convert_range_one_to_max_int32(x):
    v = _convert_range_zero_to_max_int32(x)
    # a collision happens for 0 and 1
    if v == 0:
        v += 1
    return v
def _convert_range_zero_to_max_uint32(x):
    assert x >= 0
    # use int for py2 to eliminate long
    # types when possible
    return int(x % (_max_uint32+1))
def _convert_range_one_to_max_uint32(x):
    v = _convert_range_zero_to_max_uint32(x)
    # a collision happens for 0 and 1
    if v == 0:
        v += 1
    return v

def scenario_tree_id_to_pint32(node_name, vid):
    """
    Maps a PySP scenario tree variable id paired with a node
    name to a positive integer that can be stored in a
    32-bit signed integer type.

    The mapping is unique with high probability for a set of
    scenario tree ids that is not too large.
    (see: birthday problem)
    """
    return _convert_range_one_to_max_int32(
        uuid.uuid5(_compute_namespace(node_name), vid).int)

def scenario_tree_id_to_nzint32(node_name, vid):
    """
    Maps a PySP scenario tree variable id paired with a node
    name to a nonnegative integer that can be stored in a
    32-bit signed integer type.

    The mapping is unique with high probability for a set of
    scenario tree ids that is not too large.
    (see: birthday problem)
    """
    return _convert_range_zero_to_max_int32(
        uuid.uuid5(_compute_namespace(node_name), vid).int)

def scenario_tree_id_to_puint32(node_name, vid):
    """
    Maps a PySP scenario tree variable id paired with a node
    name to a positive integer that can be stored in a
    32-bit signed integer type.

    The mapping is unique with high probability for a set of
    scenario tree ids that is not too large.
    (see: birthday problem)
    """
    return _convert_range_one_to_max_uint32(
        uuid.uuid5(_compute_namespace(node_name), vid).int)

def scenario_tree_id_to_nzuint32(node_name, vid):
    """
    Maps a PySP scenario tree variable id paired with a node
    name to a nonnegative integer that can be stored in a
    32-bit signed integer type.

    The mapping is unique with high probability for a set of
    scenario tree ids that is not too large.
    (see: birthday problem)
    """
    return _convert_range_zero_to_max_uint32(
        uuid.uuid5(_compute_namespace(node_name), vid).int)

#
# 64-bit family
#

_max_int64 = 2**63 - 1
_max_uint64 = 2**64 - 1

def _convert_range_zero_to_max_int64(x):
    assert x >= 0
    # use int for py2 to eliminate long
    # types when possible
    return int(x % (_max_int64+1))
def _convert_range_one_to_max_int64(x):
    v = _convert_range_zero_to_max_int64(x)
    # a collision happens for 0 and 1
    if v == 0:
        v += 1
    return v
def _convert_range_zero_to_max_uint64(x):
    assert x >= 0
    # use int for py2 to eliminate long
    # types when possible
    return int(x % (_max_uint64+1))
def _convert_range_one_to_max_uint64(x):
    v = _convert_range_zero_to_max_uint64(x)
    # a collision happens for 0 and 1
    if v == 0:
        v += 1
    return v

def scenario_tree_id_to_pint64(node_name, vid):
    """
    Maps a PySP scenario tree variable id paired with a node
    name to a positive integer that can be stored in a
    64-bit signed integer type.

    The mapping is unique with high probability for a set of
    scenario tree ids that is not too large.
    (see: birthday problem)
    """
    return _convert_range_one_to_max_int64(
        uuid.uuid5(_compute_namespace(node_name), vid).int)

def scenario_tree_id_to_nzint64(node_name, vid):
    """
    Maps a PySP scenario tree variable id paired with a node
    name to a nonnegative integer that can be stored in a
    64-bit signed integer type.

    The mapping is unique with high probability for a set of
    scenario tree ids that is not too large.
    (see: birthday problem)
    """
    return _convert_range_zero_to_max_int64(
        uuid.uuid5(_compute_namespace(node_name), vid).int)

def scenario_tree_id_to_puint64(node_name, vid):
    """
    Maps a PySP scenario tree variable id paired with a node
    name to a positive integer that can be stored in a
    64-bit signed integer type.

    The mapping is unique with high probability for a set of
    scenario tree ids that is not too large.
    (see: birthday problem)
    """
    return _convert_range_one_to_max_uint64(
        uuid.uuid5(_compute_namespace(node_name), vid).int)

def scenario_tree_id_to_nzuint64(node_name, vid):
    """
    Maps a PySP scenario tree variable id paired with a node
    name to a nonnegative integer that can be stored in a
    64-bit signed integer type.

    The mapping is unique with high probability for a set of
    scenario tree ids that is not too large.
    (see: birthday problem)
    """
    return _convert_range_zero_to_max_uint64(
        uuid.uuid5(_compute_namespace(node_name), vid).int)
