"""
Test objects for construction of PyROS ConfigDict.
"""

import logging
import unittest

from pyomo.core.base import ConcreteModel, Var, VarData
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import ApplicationError
from pyomo.core.base.param import Param, ParamData
from pyomo.contrib.pyros.config import (
    InputDataStandardizer,
    mutable_param_validator,
    logger_domain,
    SolverNotResolvable,
    positive_int_or_minus_one,
    pyros_config,
    SolverIterable,
    SolverResolvable,
)
from pyomo.contrib.pyros.util import ObjectiveType
from pyomo.opt import SolverFactory, SolverResults
from pyomo.contrib.pyros.uncertainty_sets import BoxSet
from pyomo.common.dependencies import numpy_available


class TestInputDataStandardizer(unittest.TestCase):
    """
    Test standardizer method for Pyomo component-type inputs.
    """

    def test_single_component_data(self):
        """
        Test standardizer works for single component
        data-type entry.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])

        standardizer_func = InputDataStandardizer(Var, VarData)

        standardizer_input = mdl.v[0]
        standardizer_output = standardizer_func(standardizer_input)

        self.assertIsInstance(
            standardizer_output,
            list,
            msg=(
                "Standardized output should be of type list, "
                f"but is of type {standardizer_output.__class__.__name__}."
            ),
        )
        self.assertEqual(
            len(standardizer_output),
            1,
            msg="Length of standardizer output is not as expected.",
        )
        self.assertIs(
            standardizer_output[0],
            mdl.v[0],
            msg=(
                f"Entry {standardizer_output[0]} (id {id(standardizer_output[0])}) "
                "is not identical to "
                f"input component data object {mdl.v[0]} "
                f"(id {id(mdl.v[0])})"
            ),
        )

    def test_standardizer_indexed_component(self):
        """
        Test component standardizer works on indexed component.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])

        standardizer_func = InputDataStandardizer(Var, VarData)

        standardizer_input = mdl.v
        standardizer_output = standardizer_func(standardizer_input)

        self.assertIsInstance(
            standardizer_output,
            list,
            msg=(
                "Standardized output should be of type list, "
                f"but is of type {standardizer_output.__class__.__name__}."
            ),
        )
        self.assertEqual(
            len(standardizer_output),
            2,
            msg="Length of standardizer output is not as expected.",
        )
        enum_zip = enumerate(zip(standardizer_input.values(), standardizer_output))
        for idx, (input, output) in enum_zip:
            self.assertIs(
                input,
                output,
                msg=(
                    f"Entry {input} (id {id(input)}) "
                    "is not identical to "
                    f"input component data object {output} "
                    f"(id {id(output)})"
                ),
            )

    def test_standardizer_multiple_components(self):
        """
        Test standardizer works on sequence of components.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])
        mdl.x = Var(["a", "b"])

        standardizer_func = InputDataStandardizer(Var, VarData)

        standardizer_input = [mdl.v[0], mdl.x]
        standardizer_output = standardizer_func(standardizer_input)
        expected_standardizer_output = [mdl.v[0], mdl.x["a"], mdl.x["b"]]

        self.assertIsInstance(
            standardizer_output,
            list,
            msg=(
                "Standardized output should be of type list, "
                f"but is of type {standardizer_output.__class__.__name__}."
            ),
        )
        self.assertEqual(
            len(standardizer_output),
            len(expected_standardizer_output),
            msg="Length of standardizer output is not as expected.",
        )
        enum_zip = enumerate(zip(expected_standardizer_output, standardizer_output))
        for idx, (input, output) in enum_zip:
            self.assertIs(
                input,
                output,
                msg=(
                    f"Entry {input} (id {id(input)}) "
                    "is not identical to "
                    f"input component data object {output} "
                    f"(id {id(output)})"
                ),
            )

    def test_standardizer_invalid_duplicates(self):
        """
        Test standardizer raises exception if input contains duplicates
        and duplicates are not allowed.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])
        mdl.x = Var(["a", "b"])

        standardizer_func = InputDataStandardizer(Var, VarData, allow_repeats=False)

        exc_str = r"Standardized.*list.*contains duplicate entries\."
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func([mdl.x, mdl.v, mdl.x])

    def test_standardizer_invalid_type(self):
        """
        Test standardizer raises exception as expected
        when input is of invalid type.
        """
        standardizer_func = InputDataStandardizer(Var, VarData)

        exc_str = r"Input object .*is not of valid component type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            standardizer_func(2)

    def test_standardizer_iterable_with_invalid_type(self):
        """
        Test standardizer raises exception as expected
        when input is an iterable with entries of invalid type.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])
        standardizer_func = InputDataStandardizer(Var, VarData)

        exc_str = r"Input object .*entry of iterable.*is not of valid component type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            standardizer_func([mdl.v, 2])

    def test_standardizer_invalid_str_passed(self):
        """
        Test standardizer raises exception as expected
        when input is of invalid type str.
        """
        standardizer_func = InputDataStandardizer(Var, VarData)

        exc_str = r"Input object .*is not of valid component type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            standardizer_func("abcd")

    def test_standardizer_invalid_uninitialized_params(self):
        """
        Test standardizer raises exception when Param with
        uninitialized entries passed.
        """
        standardizer_func = InputDataStandardizer(
            ctype=Param, cdatatype=ParamData, ctype_validator=mutable_param_validator
        )

        mdl = ConcreteModel()
        mdl.p = Param([0, 1])

        exc_str = r"Length of .*does not match that of.*index set"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(mdl.p)

    def test_standardizer_invalid_immutable_params(self):
        """
        Test standardizer raises exception when immutable
        Param object(s) passed.
        """
        standardizer_func = InputDataStandardizer(
            ctype=Param, cdatatype=ParamData, ctype_validator=mutable_param_validator
        )

        mdl = ConcreteModel()
        mdl.p = Param([0, 1], initialize=1)

        exc_str = r"Param object with name .*immutable"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(mdl.p)

    def test_standardizer_valid_mutable_params(self):
        """
        Test Param-like standardizer works as expected for sequence
        of valid mutable Param objects.
        """
        mdl = ConcreteModel()
        mdl.p1 = Param([0, 1], initialize=0, mutable=True)
        mdl.p2 = Param(["a", "b"], initialize=1, mutable=True)

        standardizer_func = InputDataStandardizer(
            ctype=Param, cdatatype=ParamData, ctype_validator=mutable_param_validator
        )

        standardizer_input = [mdl.p1[0], mdl.p2]
        standardizer_output = standardizer_func(standardizer_input)
        expected_standardizer_output = [mdl.p1[0], mdl.p2["a"], mdl.p2["b"]]

        self.assertIsInstance(
            standardizer_output,
            list,
            msg=(
                "Standardized output should be of type list, "
                f"but is of type {standardizer_output.__class__.__name__}."
            ),
        )
        self.assertEqual(
            len(standardizer_output),
            len(expected_standardizer_output),
            msg="Length of standardizer output is not as expected.",
        )
        enum_zip = enumerate(zip(expected_standardizer_output, standardizer_output))
        for idx, (input, output) in enum_zip:
            self.assertIs(
                input,
                output,
                msg=(
                    f"Entry {input} (id {id(input)}) "
                    "is not identical to "
                    f"input component data object {output} "
                    f"(id {id(output)})"
                ),
            )


AVAILABLE_SOLVER_TYPE_NAME = "available_pyros_test_solver"


class AvailableSolver:
    """
    Perennially available placeholder solver.
    """

    def available(self, exception_flag=False):
        """
        Check solver available.
        """
        return True

    def solve(self, model, **kwds):
        """
        Return SolverResults object with 'unknown' termination
        condition. Model remains unchanged.
        """
        return SolverResults()


class UnavailableSolver:
    def available(self, exception_flag=True):
        if exception_flag:
            raise ApplicationError(f"Solver {self.__class__} not available")
        return False

    def solve(self, model, *args, **kwargs):
        return SolverResults()


class TestSolverResolvable(unittest.TestCase):
    """
    Test PyROS standardizer for solver-type objects.
    """

    def setUp(self):
        SolverFactory.register(AVAILABLE_SOLVER_TYPE_NAME)(AvailableSolver)

    def tearDown(self):
        SolverFactory.unregister(AVAILABLE_SOLVER_TYPE_NAME)

    def test_solver_resolvable_valid_str(self):
        """
        Test solver resolvable class is valid for string
        type.
        """
        solver_str = AVAILABLE_SOLVER_TYPE_NAME
        standardizer_func = SolverResolvable()
        solver = standardizer_func(solver_str)
        expected_solver_type = type(SolverFactory(solver_str))

        self.assertIsInstance(
            solver,
            type(SolverFactory(solver_str)),
            msg=(
                "SolverResolvable object should be of type "
                f"{expected_solver_type.__name__}, "
                f"but got object of type {solver.__class__.__name__}."
            ),
        )

    def test_solver_resolvable_valid_solver_type(self):
        """
        Test solver resolvable class is valid for string
        type.
        """
        solver = SolverFactory(AVAILABLE_SOLVER_TYPE_NAME)
        standardizer_func = SolverResolvable()
        standardized_solver = standardizer_func(solver)

        self.assertIs(
            solver,
            standardized_solver,
            msg=(
                f"Test solver {solver} and standardized solver "
                f"{standardized_solver} are not identical."
            ),
        )

    def test_solver_resolvable_invalid_type(self):
        """
        Test solver resolvable object raises expected
        exception when invalid entry is provided.
        """
        invalid_object = 2
        standardizer_func = SolverResolvable(solver_desc="local solver")

        exc_str = (
            r"Cannot cast object `2` to a Pyomo optimizer.*"
            r"local solver.*got type int.*"
        )
        with self.assertRaisesRegex(SolverNotResolvable, exc_str):
            standardizer_func(invalid_object)

    def test_solver_resolvable_unavailable_solver(self):
        """
        Test solver standardizer fails in event solver is
        unavailable.
        """
        unavailable_solver = UnavailableSolver()
        standardizer_func = SolverResolvable(
            solver_desc="local solver", require_available=True
        )

        exc_str = r"Solver.*UnavailableSolver.*not available"
        with self.assertRaisesRegex(ApplicationError, exc_str):
            with LoggingIntercept(level=logging.ERROR) as LOG:
                standardizer_func(unavailable_solver)

        error_msgs = LOG.getvalue()[:-1]
        self.assertRegex(
            error_msgs, r"Output of `available\(\)` method.*local solver.*"
        )


class TestSolverIterable(unittest.TestCase):
    """
    Test standardizer method for iterable of solvers,
    used to validate `backup_local_solvers` and `backup_global_solvers`
    arguments.
    """

    def setUp(self):
        SolverFactory.register(AVAILABLE_SOLVER_TYPE_NAME)(AvailableSolver)

    def tearDown(self):
        SolverFactory.unregister(AVAILABLE_SOLVER_TYPE_NAME)

    def test_solver_iterable_valid_list(self):
        """
        Test solver type standardizer works for list of valid
        objects castable to solver.
        """
        solver_list = [
            AVAILABLE_SOLVER_TYPE_NAME,
            SolverFactory(AVAILABLE_SOLVER_TYPE_NAME),
        ]
        expected_solver_types = [AvailableSolver] * 2
        standardizer_func = SolverIterable()

        standardized_solver_list = standardizer_func(solver_list)

        # check list of solver types returned
        for idx, standardized_solver in enumerate(standardized_solver_list):
            self.assertIsInstance(
                standardized_solver,
                expected_solver_types[idx],
                msg=(
                    f"Standardized solver {standardized_solver} "
                    f"(index {idx}) expected to be of type "
                    f"{expected_solver_types[idx].__name__}, "
                    f"but is of type {standardized_solver.__class__.__name__}"
                ),
            )

        # second entry of standardized solver list should be the same
        # object as that of input list, since the input solver is a Pyomo
        # solver type
        self.assertIs(
            standardized_solver_list[1],
            solver_list[1],
            msg=(
                f"Test solver {solver_list[1]} and standardized solver "
                f"{standardized_solver_list[1]} should be identical."
            ),
        )

    def test_solver_iterable_valid_str(self):
        """
        Test SolverIterable raises exception when str passed.
        """
        solver_str = AVAILABLE_SOLVER_TYPE_NAME
        standardizer_func = SolverIterable()

        solver_list = standardizer_func(solver_str)
        self.assertEqual(
            len(solver_list), 1, "Standardized solver list is not of expected length"
        )

    def test_solver_iterable_unavailable_solver(self):
        """
        Test SolverIterable addresses unavailable solvers appropriately.
        """
        solvers = (AvailableSolver(), UnavailableSolver())

        standardizer_func = SolverIterable(
            require_available=True,
            filter_by_availability=True,
            solver_desc="example solver list",
        )
        exc_str = r"Solver.*UnavailableSolver.* not available"
        with self.assertRaisesRegex(ApplicationError, exc_str):
            standardizer_func(solvers)
        with self.assertRaisesRegex(ApplicationError, exc_str):
            standardizer_func(solvers, filter_by_availability=False)

        standardized_solver_list = standardizer_func(
            solvers, filter_by_availability=True, require_available=False
        )
        self.assertEqual(
            len(standardized_solver_list),
            1,
            msg=("Length of filtered standardized solver list not as " "expected."),
        )
        self.assertIs(
            standardized_solver_list[0],
            solvers[0],
            msg="Entry of filtered standardized solver list not as expected.",
        )

        standardized_solver_list = standardizer_func(
            solvers, filter_by_availability=False, require_available=False
        )
        self.assertEqual(
            len(standardized_solver_list),
            2,
            msg=("Length of filtered standardized solver list not as " "expected."),
        )
        self.assertEqual(
            standardized_solver_list,
            list(solvers),
            msg="Entry of filtered standardized solver list not as expected.",
        )

    def test_solver_iterable_invalid_list(self):
        """
        Test SolverIterable raises exception if iterable contains
        at least one invalid object.
        """
        invalid_object = [AVAILABLE_SOLVER_TYPE_NAME, 2]
        standardizer_func = SolverIterable(solver_desc="backup solver")

        exc_str = (
            r"Cannot cast object `2` to a Pyomo optimizer.*"
            r"backup solver.*index 1.*got type int.*"
        )
        with self.assertRaisesRegex(SolverNotResolvable, exc_str):
            standardizer_func(invalid_object)


class TestPyROSConfig(unittest.TestCase):
    """
    Test PyROS ConfigDict behaves as expected.
    """

    CONFIG = pyros_config()

    def test_config_objective_focus(self):
        """
        Test config parses objective focus as expected.
        """
        config = self.CONFIG()

        for obj_focus_name in ["nominal", "worst_case"]:
            config.objective_focus = obj_focus_name
            self.assertEqual(
                config.objective_focus,
                ObjectiveType[obj_focus_name],
                msg="Objective focus not set as expected.",
            )

        for obj_focus in ObjectiveType:
            config.objective_focus = obj_focus
            self.assertEqual(
                config.objective_focus,
                obj_focus,
                msg="Objective focus not set as expected.",
            )

        invalid_focus = "test_example"
        exc_str = f".*{invalid_focus!r} is not a valid ObjectiveType"
        with self.assertRaisesRegex(ValueError, exc_str):
            config.objective_focus = invalid_focus


class TestPositiveIntOrMinusOne(unittest.TestCase):
    """
    Test validator for -1 or positive int works as expected.
    """

    def test_positive_int_or_minus_one(self):
        """
        Test positive int or -1 validator works as expected.
        """
        standardizer_func = positive_int_or_minus_one
        ans = standardizer_func(1.0)
        self.assertEqual(
            ans,
            1,
            msg=f"{positive_int_or_minus_one.__name__} output value not as expected.",
        )
        self.assertIs(
            type(ans),
            int,
            msg=f"{positive_int_or_minus_one.__name__} output type not as expected.",
        )

        ans = standardizer_func(-1.0)
        self.assertEqual(
            ans,
            -1,
            msg=f"{positive_int_or_minus_one.__name__} output value not as expected.",
        )
        self.assertIs(
            type(ans),
            int,
            msg=f"{positive_int_or_minus_one.__name__} output type not as expected.",
        )

        exc_str = r"Expected positive int or -1, but received value.*"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(1.5)
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(0)


class TestLoggerDomain(unittest.TestCase):
    """
    Test logger type domain validator.
    """

    def test_logger_type(self):
        """
        Test logger type validator.
        """
        standardizer_func = logger_domain
        mylogger = logging.getLogger("example")
        self.assertIs(
            standardizer_func(mylogger),
            mylogger,
            msg=f"{standardizer_func.__name__} output not as expected",
        )
        self.assertIs(
            standardizer_func(mylogger.name),
            mylogger,
            msg=f"{standardizer_func.__name__} output not as expected",
        )

        exc_str = r"A logger name must be a string"
        with self.assertRaisesRegex(Exception, exc_str):
            standardizer_func(2)


if __name__ == "__main__":
    unittest.main()
