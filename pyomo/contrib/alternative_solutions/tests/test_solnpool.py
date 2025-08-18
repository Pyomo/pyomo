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
