"""
Test objects for construction of PyROS ConfigDict.
"""

import logging
import os
import unittest

from pyomo.core.base import ConcreteModel, Var, _VarData
from pyomo.common.config import Path
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import ApplicationError
from pyomo.core.base.param import Param, _ParamData
from pyomo.contrib.pyros.config import (
    InputDataStandardizer,
    mutable_param_validator,
    LoggerType,
    SolverNotResolvable,
    PathLikeOrNone,
    PositiveIntOrMinusOne,
    pyros_config,
    resolve_keyword_arguments,
    SolverIterable,
    SolverResolvable,
    UncertaintySetDomain,
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

        standardizer_func = InputDataStandardizer(Var, _VarData)

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

        standardizer_func = InputDataStandardizer(Var, _VarData)

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

        standardizer_func = InputDataStandardizer(Var, _VarData)

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

        standardizer_func = InputDataStandardizer(Var, _VarData, allow_repeats=False)

        exc_str = r"Standardized.*list.*contains duplicate entries\."
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func([mdl.x, mdl.v, mdl.x])

    def test_standardizer_invalid_type(self):
        """
        Test standardizer raises exception as expected
        when input is of invalid type.
        """
        standardizer_func = InputDataStandardizer(Var, _VarData)

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
        standardizer_func = InputDataStandardizer(Var, _VarData)

        exc_str = r"Input object .*entry of iterable.*is not of valid component type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            standardizer_func([mdl.v, 2])

    def test_standardizer_invalid_str_passed(self):
        """
        Test standardizer raises exception as expected
        when input is of invalid type str.
        """
        standardizer_func = InputDataStandardizer(Var, _VarData)

        exc_str = r"Input object .*is not of valid component type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            standardizer_func("abcd")

    def test_standardizer_invalid_uninitialized_params(self):
        """
        Test standardizer raises exception when Param with
        uninitialized entries passed.
        """
        standardizer_func = InputDataStandardizer(
            ctype=Param, cdatatype=_ParamData, ctype_validator=mutable_param_validator
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
            ctype=Param, cdatatype=_ParamData, ctype_validator=mutable_param_validator
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
            ctype=Param, cdatatype=_ParamData, ctype_validator=mutable_param_validator
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


class TestUncertaintySetDomain(unittest.TestCase):
    """
    Test domain validator for uncertainty set arguments.
    """

    @unittest.skipUnless(numpy_available, "Numpy is not available.")
    def test_uncertainty_set_domain_valid_set(self):
        """
        Test validator works for valid argument.
        """
        standardizer_func = UncertaintySetDomain()
        bset = BoxSet([[0, 1]])
        self.assertIs(
            bset,
            standardizer_func(bset),
            msg="Output of uncertainty set domain not as expected.",
        )

    def test_uncertainty_set_domain_invalid_type(self):
        """
        Test validator works for valid argument.
        """
        standardizer_func = UncertaintySetDomain()
        exc_str = "Expected an .*UncertaintySet object.*received object 2"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(2)


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


class TestPathLikeOrNone(unittest.TestCase):
    """
    Test interface for validating path-like arguments.
    """

    def test_none_valid(self):
        """
        Test `None` is valid.
        """
        standardizer_func = PathLikeOrNone()

        self.assertIs(
            standardizer_func(None),
            None,
            msg="Output of `PathLikeOrNone` standardizer not as expected.",
        )

    def test_str_bytes_path_like_valid(self):
        """
        Check path-like validator handles str, bytes, and path-like
        inputs correctly.
        """

        class ExamplePathLike(os.PathLike):
            """
            Path-like class for testing. Key feature: __fspath__
            and __str__ return different outputs.
            """

            def __init__(self, path_str_or_bytes):
                self.path = path_str_or_bytes

            def __fspath__(self):
                return self.path

            def __str__(self):
                path_str = os.fsdecode(self.path)
                return f"{type(self).__name__}({path_str})"

        path_standardization_func = PathLikeOrNone()

        # construct path arguments of different type
        path_as_str = "example_output_dir/"
        path_as_bytes = os.fsencode(path_as_str)
        path_like_from_str = ExamplePathLike(path_as_str)
        path_like_from_bytes = ExamplePathLike(path_as_bytes)

        # for all possible arguments, output should be
        # the str returned by ``common.config.Path`` when
        # string representation of the path is input.
        expected_output = Path()(path_as_str)

        # check output is as expected in all cases
        self.assertEqual(
            path_standardization_func(path_as_str),
            expected_output,
            msg=(
                "Path-like validator output from str input "
                "does not match expected value."
            ),
        )
        self.assertEqual(
            path_standardization_func(path_as_bytes),
            expected_output,
            msg=(
                "Path-like validator output from bytes input "
                "does not match expected value."
            ),
        )
        self.assertEqual(
            path_standardization_func(path_like_from_str),
            expected_output,
            msg=(
                "Path-like validator output from path-like input "
                "derived from str does not match expected value."
            ),
        )
        self.assertEqual(
            path_standardization_func(path_like_from_bytes),
            expected_output,
            msg=(
                "Path-like validator output from path-like input "
                "derived from bytes does not match expected value."
            ),
        )


class TestPositiveIntOrMinusOne(unittest.TestCase):
    """
    Test validator for -1 or positive int works as expected.
    """

    def test_positive_int_or_minus_one(self):
        """
        Test positive int or -1 validator works as expected.
        """
        standardizer_func = PositiveIntOrMinusOne()
        self.assertIs(
            standardizer_func(1.0),
            1,
            msg=(f"{PositiveIntOrMinusOne.__name__} does not standardize as expected."),
        )
        self.assertEqual(
            standardizer_func(-1.00),
            -1,
            msg=(f"{PositiveIntOrMinusOne.__name__} does not standardize as expected."),
        )

        exc_str = r"Expected positive int or -1, but received value.*"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(1.5)
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(0)


class TestLoggerType(unittest.TestCase):
    """
    Test logger type validator.
    """

    def test_logger_type(self):
        """
        Test logger type validator.
        """
        standardizer_func = LoggerType()
        mylogger = logging.getLogger("example")
        self.assertIs(
            standardizer_func(mylogger),
            mylogger,
            msg=f"{LoggerType.__name__} output not as expected",
        )
        self.assertIs(
            standardizer_func(mylogger.name),
            mylogger,
            msg=f"{LoggerType.__name__} output not as expected",
        )

        exc_str = r"A logger name must be a string"
        with self.assertRaisesRegex(Exception, exc_str):
            standardizer_func(2)


class TestResolveKeywordArguments(unittest.TestCase):
    """
    Test keyword argument resolution function works as expected.
    """

    def test_resolve_kwargs_simple_dict(self):
        """
        Test resolve kwargs works, simple example
        where there is overlap.
        """
        explicit_kwargs = dict(arg1=1)
        implicit_kwargs_1 = dict(arg1=2, arg2=3)
        implicit_kwargs_2 = dict(arg1=4, arg2=4, arg3=5)

        # expected answer
        expected_resolved_kwargs = dict(arg1=1, arg2=3, arg3=5)

        # attempt kwargs resolve
        with LoggingIntercept(level=logging.WARNING) as LOG:
            resolved_kwargs = resolve_keyword_arguments(
                prioritized_kwargs_dicts={
                    "explicitly": explicit_kwargs,
                    "implicitly through set 1": implicit_kwargs_1,
                    "implicitly through set 2": implicit_kwargs_2,
                }
            )

        # check kwargs resolved as expected
        self.assertEqual(
            resolved_kwargs,
            expected_resolved_kwargs,
            msg="Resolved kwargs do not match expected value.",
        )

        # extract logger warning messages
        warning_msgs = LOG.getvalue().split("\n")[:-1]

        self.assertEqual(
            len(warning_msgs), 3, msg="Number of warning messages is not as expected."
        )

        # check contents of warning msgs
        self.assertRegex(
            warning_msgs[0],
            expected_regex=(
                r"Arguments \['arg1'\] passed implicitly through set 1 "
                r"already passed explicitly.*"
            ),
        )
        self.assertRegex(
            warning_msgs[1],
            expected_regex=(
                r"Arguments \['arg1'\] passed implicitly through set 2 "
                r"already passed explicitly.*"
            ),
        )
        self.assertRegex(
            warning_msgs[2],
            expected_regex=(
                r"Arguments \['arg2'\] passed implicitly through set 2 "
                r"already passed implicitly through set 1.*"
            ),
        )


if __name__ == "__main__":
    unittest.main()
