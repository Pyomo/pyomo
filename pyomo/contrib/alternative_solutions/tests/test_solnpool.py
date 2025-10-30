#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pytest
import pprint

from pyomo.contrib.alternative_solutions import (
    PoolManager,
    Solution,
    VariableInfo,
    ObjectiveInfo,
)

# from pyomo.contrib.alternative_solutions.aos_utils import MyMunch


def soln(value, objective):
    return Solution(
        variables=[VariableInfo(value=value)],
        objectives=[ObjectiveInfo(value=objective)],
    )


def test_pool_active_name():
    pm = PoolManager()
    assert pm.name == None, "Should only have the None pool"
    pm.add_pool(name="pool_1", policy="keep_all")
    assert pm.name == "pool_1", "Should only have 'pool_1'"


def test_get_pool_names():
    pm = PoolManager()
    assert pm.get_pool_names() == [None], "Should only be [None]"
    pm.add_pool(name="pool_1", policy="keep_all")
    assert pm.get_pool_names() == ["pool_1"], "Should only be ['pool_1']"
    pm.add_pool(name="pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.get_pool_names() == ["pool_1", "pool_2"], "Should be ['pool_1', 'pool_2']"


def test_get_active_pool_policy():
    pm = PoolManager()
    assert pm.policy == "keep_best", "Should only be 'keep_best'"
    pm.add_pool(name="pool_1", policy="keep_all")
    assert pm.policy == "keep_all", "Should only be 'keep_best'"
    pm.add_pool(name="pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.policy == "keep_latest", "Should only be 'keep_latest'"


def test_get_pool_policies():
    pm = PoolManager()
    assert pm.get_pool_policies() == {
        None: "keep_best"
    }, "Should only be {None : 'keep_best'}"
    pm.add_pool(name="pool_1", policy="keep_all")
    assert pm.get_pool_policies() == {
        "pool_1": "keep_all"
    }, "Should only be {'pool_1' : 'keep_best'}"
    pm.add_pool(name="pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.get_pool_policies() == {
        "pool_1": "keep_all",
        "pool_2": "keep_latest",
    }, "Should only be {'pool_1' : 'keep_best', 'pool_2' : 'keep_latest'}"


def test_get_max_pool_size():
    pm = PoolManager()
    assert pm.max_pool_size == None, "Should only be None"
    pm.add_pool(name="pool_1", policy="keep_all")
    assert pm.max_pool_size == None, "Should only be None"
    pm.add_pool(name="pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.max_pool_size == 1, "Should only be 1"


def test_get_max_pool_sizes():
    pm = PoolManager()
    assert pm.get_max_pool_sizes() == {None: None}, "Should only be {None: None}"
    pm.add_pool(name="pool_1", policy="keep_all")
    assert pm.get_max_pool_sizes() == {
        "pool_1": None
    }, "Should only be {'pool_1': None}"
    pm.add_pool(name="pool_2", policy="keep_latest", max_pool_size=1)
    assert pm.get_max_pool_sizes() == {
        "pool_1": None,
        "pool_2": 1,
    }, "Should only be {'pool_1': None, 'pool_2': 1}"


def test_get_pool_sizes():
    pm = PoolManager()
    pm.add_pool(name="pool_1", policy="keep_all")

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 3

    pm.add_pool(name="pool_2", policy="keep_latest", max_pool_size=1)
    retval = pm.add(soln(0, 0))
    assert len(pm) == 1
    retval = pm.add(soln(0, 1))

    assert pm.get_pool_sizes() == {
        "pool_1": 3,
        "pool_2": 1,
    }, "Should be {'pool_1' :3, 'pool_2' : 1}"


def test_multiple_pools():
    pm = PoolManager()
    pm.add_pool(name="pool_1", policy="keep_all")

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 3

    assert pm.get_pool_dicts() == {
        "pool_1": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool_1",
                "policy": "keep_all",
            },
            "pool_config": {},
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
    pm.add_pool(name="pool_2", policy="keep_latest", max_pool_size=1)
    retval = pm.add(soln(0, 0))
    assert len(pm) == 1
    retval = pm.add(soln(0, 1))
    assert pm.get_pool_dicts() == {
        "pool_1": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool_1",
                "policy": "keep_all",
            },
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
            "pool_config": {},
        },
        "pool_2": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool_2",
                "policy": "keep_latest",
            },
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
            "pool_config": {"max_pool_size": 1},
        },
    }
    assert len(pm) == 1


def test_keepall_add():
    pm = PoolManager()
    pm.add_pool(name="pool", policy="keep_all")

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 3

    assert pm.get_pool_dicts() == {
        "pool": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool",
                "policy": "keep_all",
            },
            "pool_config": {},
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
        pm.add_pool(name="pool", policy="invalid_policy")
    except ValueError as e:
        pass


def test_invalid_policy_2():
    pm = PoolManager()
    try:
        pm.add_pool(name="pool", policy="invalid_policy", max_pool_size=-2)
    except ValueError as e:
        pass


def test_keeplatest_bad_max_pool_size():
    pm = PoolManager()
    try:
        pm.add_pool(name="pool", policy="keep_latest", max_pool_size=-2)
    except AssertionError as e:
        pass


def test_keeplatest_add():
    pm = PoolManager()
    pm.add_pool(name="pool", policy="keep_latest", max_pool_size=2)

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is not None
    assert len(pm) == 2

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 2

    assert pm.get_pool_dicts() == {
        "pool": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool",
                "policy": "keep_latest",
            },
            "pool_config": {"max_pool_size": 2},
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
        pm.add_pool(name="pool", policy="keep_latest_unique", max_pool_size=-2)
    except AssertionError as e:
        pass


def test_keeplatestunique_add():
    pm = PoolManager()
    pm.add_pool(name="pool", policy="keep_latest_unique", max_pool_size=2)

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))
    assert retval is None
    assert len(pm) == 1

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 2

    assert pm.get_pool_dicts() == {
        "pool": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool",
                "policy": "keep_latest_unique",
            },
            "pool_config": {"max_pool_size": 2},
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
        pm.add_pool(name="pool", policy="keep_best", max_pool_size=-2)
    except AssertionError as e:
        pass


def test_pool_manager_to_dict_passthrough():
    pm = PoolManager()
    pm = PoolManager()
    pm.add_pool(name="pool", policy="keep_best", abs_tolerance=1)

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
        "metadata": {
            "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
            "context_name": "pool",
            "policy": "keep_best",
        },
        "pool_config": {
            'abs_tolerance': 1,
            'best_value': 0,
            'max_pool_size': None,
            'objective': 0,
            'rel_tolerance': None,
            'sense_is_min': True,
        },
        "solutions": {
            0: {
                "id": 0,
                "objectives": [{"index": None, "name": None, "suffix": {}, "value": 0}],
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
                "objectives": [{"index": None, "name": None, "suffix": {}, "value": 1}],
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


def test_keepbest_add1():
    pm = PoolManager()
    pm.add_pool(name="pool", policy="keep_best", abs_tolerance=1)

    retval = pm.add(soln(0, 0))
    assert retval is not None
    assert len(pm) == 1

    retval = pm.add(soln(0, 1))  # not unique
    assert retval is None
    assert len(pm) == 1

    retval = pm.add(soln(1, 1))
    assert retval is not None
    assert len(pm) == 2

    assert pm.get_pool_dicts() == {
        "pool": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool",
                "policy": "keep_best",
            },
            "pool_config": {
                'abs_tolerance': 1,
                'best_value': 0,
                'max_pool_size': None,
                'objective': 0,
                'rel_tolerance': None,
                'sense_is_min': True,
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
    pm.add_pool(name="pool", policy="keep_best", abs_tolerance=1)

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

    assert pm.get_pool_dicts() == {
        "pool": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool",
                "policy": "keep_best",
            },
            "pool_config": {
                'abs_tolerance': 1,
                'best_value': -1,
                'max_pool_size': None,
                'objective': 0,
                'rel_tolerance': None,
                'sense_is_min': True,
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

    assert pm.get_pool_dicts() == {
        "pool": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool",
                "policy": "keep_best",
            },
            "pool_config": {
                "abs_tolerance": 1,
                'best_value': -1.5,
                "max_pool_size": None,
                "objective": 0,
                "rel_tolerance": None,
                'sense_is_min': True,
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
    pm.add_pool(name="pool", policy="keep_best", abs_tolerance=1, max_pool_size=2)

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

    assert pm.get_pool_dicts() == {
        "pool": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool",
                "policy": "keep_best",
            },
            "pool_config": {
                "abs_tolerance": 1,
                'best_value': -1,
                "max_pool_size": 2,
                "objective": 0,
                "rel_tolerance": None,
                'sense_is_min': True,
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

    assert pm.get_pool_dicts() == {
        "pool": {
            "metadata": {
                "as_solution_source": "pyomo.contrib.alternative_solutions.solnpool._as_solution",
                "context_name": "pool",
                "policy": "keep_best",
            },
            "pool_config": {
                "abs_tolerance": 1,
                'best_value': -1.5,
                "max_pool_size": 2,
                "objective": 0,
                "rel_tolerance": None,
                "sense_is_min": True,
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
