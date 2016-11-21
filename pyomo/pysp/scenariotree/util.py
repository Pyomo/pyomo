import hashlib
import uuid

_namespace_uuid = uuid.UUID(bytes=hashlib.md5('pysp'.encode()).digest())

#
# 32-bit family
#

_max_int32 = 2**31 - 1
_max_uint32 = 2**32 - 1

def _convert_range_zero_to_max_int32(x):
    assert x >= 0
    return x % (_max_int32+1)
def _convert_range_one_to_max_int32(x):
    v = _convert_range_zero_to_max_int32(x)
    # the collision happens for 0 and 1
    if v == 0:
        v += 1
    return v
def _convert_range_zero_to_max_uint32(x):
    assert x >= 0
    return x % (_max_uint32+1)
def _convert_range_one_to_max_uint32(x):
    v = _convert_range_zero_to_max_uint32(x)
    # the collision happens for 0 and 1
    if v == 0:
        v += 1
    return v

def scenario_tree_id_to_pint32(vid):
    """
    Maps a PySP scenario tree variable id to a positive integer
    that can be stored in a 32-bit signed integer type.

    The mapping is unique with high probability for a set of scenario
    tree ids that is not too large (see birthday problem).
    """
    return _convert_range_one_to_max_int32(
        uuid.uuid5(_namespace_uuid, vid).int)

def scenario_tree_id_to_nzint32(vid):
    """
    Maps a PySP scenario tree variable id to a nonnegative integer
    that can be stored in a 32-bit signed integer type.

    The mapping is unique with high probability for a set of scenario
    tree ids that is not too large (see birthday problem).
    """
    return _convert_range_zero_to_max_int32(
        uuid.uuid5(_namespace_uuid, vid).int)

def scenario_tree_id_to_puint32(vid):
    """
    Maps a PySP scenario tree variable id to a positive integer
    that can be stored in a 32-bit signed integer type.

    The mapping is unique with high probability for a set of scenario
    tree ids that is not too large (see birthday problem).
    """
    return _convert_range_one_to_max_uint32(
        uuid.uuid5(_namespace_uuid, vid).int)

def scenario_tree_id_to_nzuint32(vid):
    """
    Maps a PySP scenario tree variable id to a nonnegative integer
    that can be stored in a 32-bit signed integer type

    The mapping is unique with high probability for a set of scenario
    tree ids that is not too large (see birthday problem).
    """
    return _convert_range_zero_to_max_uint32(
        uuid.uuid5(_namespace_uuid, vid).int)

#
# 64-bit family
#

_max_int64 = 2**63 - 1
_max_uint64 = 2**64 - 1

def _convert_range_zero_to_max_int64(x):
    assert x >= 0
    return x % (_max_int64+1)
def _convert_range_one_to_max_int64(x):
    v = _convert_range_zero_to_max_int64(x)
    # the collision happens for 0 and 1
    if v == 0:
        v += 1
    return v
def _convert_range_zero_to_max_uint64(x):
    assert x >= 0
    return x % (_max_uint64+1)
def _convert_range_one_to_max_uint64(x):
    v = _convert_range_zero_to_max_uint64(x)
    # the collision happens for 0 and 1
    if v == 0:
        v += 1
    return v

def scenario_tree_id_to_pint64(vid):
    """
    Maps a PySP scenario tree variable id to a positive integer
    that can be stored in a 64-bit signed integer type.

    The mapping is unique with high probability for a set of scenario
    tree ids that is not too large (see birthday problem).
    """
    return _convert_range_one_to_max_int64(
        uuid.uuid5(_namespace_uuid, vid).int)

def scenario_tree_id_to_nzint64(vid):
    """
    Maps a PySP scenario tree variable id to a nonnegative integer
    that can be stored in a 64-bit signed integer type.

    The mapping is unique with high probability for a set of scenario
    tree ids that is not too large (see birthday problem).
    """
    return _convert_range_zero_to_max_int64(
        uuid.uuid5(_namespace_uuid, vid).int)

def scenario_tree_id_to_puint64(vid):
    """
    Maps a PySP scenario tree variable id to a positive integer
    that can be stored in a 64-bit signed integer type.

    The mapping is unique with high probability for a set of scenario
    tree ids that is not too large (see birthday problem).
    """
    return _convert_range_one_to_max_uint64(
        uuid.uuid5(_namespace_uuid, vid).int)

def scenario_tree_id_to_nzuint64(vid):
    """
    Maps a PySP scenario tree variable id to a nonnegative integer
    that can be stored in a 64-bit signed integer type

    The mapping is unique with high probability for a set of scenario
    tree ids that is not too large (see birthday problem).
    """
    return _convert_range_zero_to_max_uint64(
        uuid.uuid5(_namespace_uuid, vid).int)
