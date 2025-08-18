import pytest
import pprint

from pyomo.contrib.alternative_solutions import (
    PoolManager,
    Solution,
    Variable,
    Objective,
)


def soln(value, objective):
    return Solution(
        variables=[Variable(value=value)], objectives=[Objective(value=objective)]
    )


def test_pool_active_name():
    pm = PoolManager()
    assert pm.get_active_pool_name() == None, "Should only have the None pool"
    pm.add_pool("pool_1", policy="keep_all")
    assert pm.get_active_pool_name() == "pool_1", "Should only have 'pool_1'"


def test_get_pool_names():
    pm = PoolManager()
    assert pm.get_pool_names() == [None], "Should only be [None]"
    pm.add_pool("pool_1", policy="keep_all")
    assert pm.get_pool_names() == ["pool_1"], "Should only be ['pool_1']"
    pm.add_pool("pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.get_pool_names() == ["pool_1", "pool_2"], "Should be ['pool_1', 'pool_2']"


def test_get_active_pool_policy():
    pm = PoolManager()
    assert pm.get_active_pool_policy() == "keep_best", "Should only be 'keep_best'"
    pm.add_pool("pool_1", policy="keep_all")
    assert pm.get_active_pool_policy() == "keep_all", "Should only be 'keep_best'"
    pm.add_pool("pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.get_active_pool_policy() == "keep_latest", "Should only be 'keep_latest'"


def test_get_pool_policies():
    pm = PoolManager()
    assert pm.get_pool_policies() == {
        None: "keep_best"
    }, "Should only be {None : 'keep_best'}"
    pm.add_pool("pool_1", policy="keep_all")
    assert pm.get_pool_policies() == {
        "pool_1": "keep_all"
    }, "Should only be {'pool_1' : 'keep_best'}"
    pm.add_pool("pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.get_pool_policies() == {
        "pool_1": "keep_all",
        "pool_2": "keep_latest",
    }, "Should only be {'pool_1' : 'keep_best', 'pool_2' : 'keep_latest'}"


def test_get_max_pool_size():
    pm = PoolManager()
    assert pm.get_max_pool_size() == None, "Should only be None"
    pm.add_pool("pool_1", policy="keep_all")
    assert pm.get_max_pool_size() == None, "Should only be None"
    pm.add_pool("pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.get_max_pool_size() == 1, "Should only be 1"


def test_get_max_pool_sizes():
    pm = PoolManager()
    assert pm.get_max_pool_sizes() == {None: None}, "Should only be {None: None}"
    pm.add_pool("pool_1", policy="keep_all")
    assert pm.get_max_pool_sizes() == {
        "pool_1": None
    }, "Should only be {'pool_1': None}"
    pm.add_pool("pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.get_max_pool_sizes() == {
        "pool_1": None,
        "pool_2": 1,
    }, "Should only be {'pool_1': None, 'pool_2': 1}"


def test_get_pool_sizes():
    pm = PoolManager()
    pm.add_pool("pool_1", policy="keep_all")

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 3

    pm.add_pool("pool_2", policy="keep_latest", max_pool_size=1)
    retval = pm.add(soln(0, 0))
    assert len(pm) == 1
    retval = pm.add(soln(0, 1))

    assert pm.get_pool_sizes() == {
        "pool_1": 3,
        "pool_2": 1,
    }, "Should be {'pool_1' :3, 'pool_2' : 1}"


def test_multiple_pools():
    pm = PoolManager()
    pm.add_pool("pool_1", policy="keep_all")

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 3

    assert pm.to_dict() == {
        "pool_1": {
            "metadata": {"context_name": "pool_1"},
            "pool_config": {"policy": "keep_all"},
            "solutions": {
                0: {
                    "id": 0,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 0}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 0,
                        }
                    ],
                },
                1: {
                    "id": 1,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 0,
                        }
                    ],
                },
                2: {
                    "id": 2,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 1,
                        }
                    ],
                },
            },
        }
    }
    pm.add_pool("pool_2", policy="keep_latest", max_pool_size=1)
    retval = pm.add(soln(0, 0))
    assert len(pm) == 1
    retval = pm.add(soln(0, 1))
    assert pm.to_dict() == {
        "pool_1": {
            "metadata": {"context_name": "pool_1"},
            "solutions": {
                0: {
                    "id": 0,
                    "variables": [
                        {
                            "value": 0,
                            "fixed": False,
                            "name": None,
                            "index": None,
                            "discrete": False,
                            "suffix": {},
                        }
                    ],
                    "objectives": [
                        {"value": 0, "name": None, "index": None, "suffix": {}}
                    ],
                    "suffix": {},
                },
                1: {
                    "id": 1,
                    "variables": [
                        {
                            "value": 0,
                            "fixed": False,
                            "name": None,
                            "index": None,
                            "discrete": False,
                            "suffix": {},
                        }
                    ],
                    "objectives": [
                        {"value": 1, "name": None, "index": None, "suffix": {}}
                    ],
                    "suffix": {},
                },
                2: {
                    "id": 2,
                    "variables": [
                        {
                            "value": 1,
                            "fixed": False,
                            "name": None,
                            "index": None,
                            "discrete": False,
                            "suffix": {},
                        }
                    ],
                    "objectives": [
                        {"value": 1, "name": None, "index": None, "suffix": {}}
                    ],
                    "suffix": {},
                },
            },
            "pool_config": {"policy": "keep_all"},
        },
        "pool_2": {
            "metadata": {"context_name": "pool_2"},
            "solutions": {
                4: {
                    "id": 4,
                    "variables": [
                        {
                            "value": 0,
                            "fixed": False,
                            "name": None,
                            "index": None,
                            "discrete": False,
                            "suffix": {},
                        }
                    ],
                    "objectives": [
                        {"value": 1, "name": None, "index": None, "suffix": {}}
                    ],
                    "suffix": {},
                }
            },
            "pool_config": {"policy": "keep_latest", "max_pool_size": 1},
        },
    }
    assert len(pm) == 1


def test_keepall_add():
    pm = PoolManager()
    pm.add_pool("pool", policy="keep_all")

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 3

    assert pm.to_dict() == {
        "pool": {
            "metadata": {"context_name": "pool"},
            "pool_config": {"policy": "keep_all"},
            "solutions": {
                0: {
                    "id": 0,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 0}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 0,
                        }
                    ],
                },
                1: {
                    "id": 1,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 0,
                        }
                    ],
                },
                2: {
                    "id": 2,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 1,
                        }
                    ],
                },
            },
        }
    }


def test_invalid_policy_1():
    pm = PoolManager()
    try:
        pm.add_pool("pool", policy="invalid_policy")
    except ValueError as e:
        pass


def test_invalid_policy_2():
    pm = PoolManager()
    try:
        pm.add_pool("pool", policy="invalid_policy", max_pool_size=-2)
    except ValueError as e:
        pass


def test_keeplatest_bad_max_pool_size():
    pm = PoolManager()
    try:
        pm.add_pool("pool", policy="keep_latest", max_pool_size=-2)
    except AssertionError as e:
        pass


def test_keeplatest_add():
    pm = PoolManager()
    pm.add_pool("pool", policy="keep_latest", max_pool_size=2)

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 2

    assert pm.to_dict() == {
        "pool": {
            "metadata": {"context_name": "pool"},
            "pool_config": {"max_pool_size": 2, "policy": "keep_latest"},
            "solutions": {
                1: {
                    "id": 1,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 0,
                        }
                    ],
                },
                2: {
                    "id": 2,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 1,
                        }
                    ],
                },
            },
        }
    }


def test_keeplatestunique_bad_max_pool_size():
    pm = PoolManager()
    try:
        pm.add_pool("pool", policy="keep_latest_unique", max_pool_size=-2)
    except AssertionError as e:
        pass


def test_keeplatestunique_add():
    pm = PoolManager()
    pm.add_pool("pool", policy="keep_latest_unique", max_pool_size=2)

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is None
    assert len(pm) == 1

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 2

    assert pm.to_dict() == {
        "pool": {
            "metadata": {"context_name": "pool"},
            "pool_config": {"max_pool_size": 2, "policy": "keep_latest_unique"},
            "solutions": {
                0: {
                    "id": 0,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 0}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 0,
                        }
                    ],
                },
                1: {
                    "id": 1,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 1,
                        }
                    ],
                },
            },
        }
    }


def test_keepbest_bad_max_pool_size():
    pm = PoolManager()
    try:
        pm.add_pool("pool", policy="keep_best", max_pool_size=-2)
    except AssertionError as e:
        pass


def test_keepbest_add1():
    pm = PoolManager()
    pm.add_pool("pool", policy="keep_best", abs_tolerance=1)

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))  # not unique
    assert retval is None
    assert len(pm) == 1

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 2

    assert pm.to_dict() == {
        "pool": {
            "metadata": {"context_name": "pool"},
            "pool_config": {
                "abs_tolerance": 1,
                "max_pool_size": None,
                "objective": 0,
                "policy": "keep_best",
                "rel_tolerance": None,
            },
            "solutions": {
                0: {
                    "id": 0,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 0}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 0,
                        }
                    ],
                },
                1: {
                    "id": 1,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 1,
                        }
                    ],
                },
            },
        }
    }


def test_keepbest_add2():
    pm = PoolManager()
    pm.add_pool("pool", policy="keep_best", abs_tolerance=1)

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))  # not unique
    assert retval is None
    assert len(pm) == 1

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(2, -1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(3, -0.5))
    assert retval is not None
    assert len(pm) == 3

    assert pm.to_dict() == {
        "pool": {
            "metadata": {"context_name": "pool"},
            "pool_config": {
                "abs_tolerance": 1,
                "max_pool_size": None,
                "objective": 0,
                "policy": "keep_best",
                "rel_tolerance": None,
            },
            "solutions": {
                0: {
                    "id": 0,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": 0}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 0,
                        }
                    ],
                },
                2: {
                    "id": 2,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 2,
                        }
                    ],
                },
                3: {
                    "id": 3,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -0.5}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 3,
                        }
                    ],
                },
            },
        }
    }

    retval = pm.add(soln(4, -1.5))
    assert retval is not None
    assert len(pm) == 3

    assert pm.to_dict() == {
        "pool": {
            "metadata": {"context_name": "pool"},
            "pool_config": {
                "abs_tolerance": 1,
                "max_pool_size": None,
                "objective": 0,
                "policy": "keep_best",
                "rel_tolerance": None,
            },
            "solutions": {
                2: {
                    "id": 2,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 2,
                        }
                    ],
                },
                3: {
                    "id": 3,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -0.5}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 3,
                        }
                    ],
                },
                4: {
                    "id": 4,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -1.5}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 4,
                        }
                    ],
                },
            },
        }
    }


def test_keepbest_add3():
    pm = PoolManager()
    pm.add_pool("pool", policy="keep_best", abs_tolerance=1, max_pool_size=2)

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))  # not unique
    assert retval is None
    assert len(pm) == 1

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(2, -1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(3, -0.5))
    assert retval is not None
    assert len(pm) == 2

    assert pm.to_dict() == {
        "pool": {
            "metadata": {"context_name": "pool"},
            "pool_config": {
                "abs_tolerance": 1,
                "max_pool_size": 2,
                "objective": 0,
                "policy": "keep_best",
                "rel_tolerance": None,
            },
            "solutions": {
                2: {
                    "id": 2,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 2,
                        }
                    ],
                },
                3: {
                    "id": 3,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -0.5}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 3,
                        }
                    ],
                },
            },
        }
    }

    retval = pm.add(soln(4, -1.5))
    assert retval is not None
    assert len(pm) == 2

    assert pm.to_dict() == {
        "pool": {
            "metadata": {"context_name": "pool"},
            "pool_config": {
                "abs_tolerance": 1,
                "max_pool_size": 2,
                "objective": 0,
                "policy": "keep_best",
                "rel_tolerance": None,
            },
            "solutions": {
                2: {
                    "id": 2,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -1}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 2,
                        }
                    ],
                },
                4: {
                    "id": 4,
                    "objectives": [
                        {"index": None, "name": None, "suffix": {}, "value": -1.5}
                    ],
                    "suffix": {},
                    "variables": [
                        {
                            "discrete": False,
                            "fixed": False,
                            "index": None,
                            "name": None,
                            "suffix": {},
                            "value": 4,
                        }
                    ],
                },
            },
        }
    }
