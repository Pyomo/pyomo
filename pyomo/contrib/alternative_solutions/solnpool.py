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

import copy
from enum import Enum
import heapq
import collections
import dataclasses
import json
import weakref

from pyomo.contrib.alternative_solutions.aos_utils import MyMunch, to_dict
from pyomo.contrib.alternative_solutions.solution import Solution, PyomoSolution

nan = float("nan")


def default_as_solution(*args, **kwargs):
    """
    A default function that creates a solution from the args and kwargs that
    are passed-in to the add() method in a solution pool.

    This passes arguments to the Solution() class constructor, so the API for this method is
    the same as that method.
    """
    return Solution(*args, **kwargs)


def _as_pyomo_solution(*args, **kwargs):
    """
    A pyomo-specific function that creates a solution from the args and kwargs that
    are passed-in to the add() method in a solution pool.

    This passes arguments to the PyomoSolution() class constructor, so the API for this method is
    the same as that method.
    """
    return PyomoSolution(*args, **kwargs)


class PoolCounter:
    """
    A class to wrap the counter element for solution pools.
    It contains just the solution_counter element.
    """

    solution_counter = 0


class PoolPolicy(Enum):
    unspecified = 'unspecified'
    keep_all = 'keep_all'
    keep_best = 'keep_best'
    keep_latest = 'keep_latest'
    keep_latest_unique = 'keep_latest_unique'

    def __str__(self):
        return f"{self.value}"


class SolutionPoolBase:
    """
    A class to manage groups of solutions as a pool.
    This is the general base pool class.

    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of SolutionPool objects can be handled with the PoolManager class.

    Parameters
    ----------
    name : str
        String name to describe the pool.
    as_solution : Callable[..., Solution][..., Solution] or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default_as_solution function being used.
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing.
        A value of None will result in a new PoolCounter object being created and used.
    policy : PoolPolicy
        Enum value to describe the pool construction and management policy.
    """

    def __init__(self, name, as_solution, counter, policy=PoolPolicy.unspecified):
        self._solutions = {}
        if as_solution is None:
            self._as_solution = default_as_solution
        else:
            self._as_solution = as_solution
        if counter is None:
            self.counter = PoolCounter()
        else:
            self.counter = counter
        # TODO: consider renaming context_name to name
        self._metadata = MyMunch(
            context_name=name,
            policy=policy,
            as_solution_source=f"{self._as_solution.__module__}.{self._as_solution.__qualname__}",
        )

    @property
    def metadata(self):
        """
        Property to return SolutionPool metadata that all SolutionPool subclasses have.
        """
        return self._metadata

    @property
    def solutions(self):
        """
        Property to return values of the dictionary of solutions.
        """
        return self._solutions.values()

    @property
    def last_solution(self):
        """
        Property to return last (successfully) added solution.
        """
        index = next(reversed(self._solutions.keys()))
        return self._solutions[index]

    @property
    def policy(self):
        """
        Property to return pool construction policy.
        """
        return self.metadata['policy']

    @property
    def as_solution(self):
        """
        Property to return solution conversion method.
        """
        return self._as_solution

    @property
    def pool_config(self):
        """
        Property to return SolutionPool class specific configuration data.
        """
        return dict()

    def __iter__(self):
        for soln in self._solutions.values():
            yield soln

    def __len__(self):
        return len(self._solutions)

    def __getitem__(self, soln_id):
        return self._solutions[soln_id]

    def _next_solution_counter(self):
        tmp = self.counter.solution_counter
        self.counter.solution_counter += 1
        return tmp

    def to_dict(self):
        """
        Converts SolutionPool to a dictionary object.

        Returns
        ----------
        dict
            Dictionary with three keys: 'metadata', 'solutions', 'pool_config'
            'metadata' contains a dictionary of information about SolutionPools that is always present.
            'solutions' contains a dictionary of the pool's solutions.
            'pool_config' contains a dictionary of details conditional to the SolutionPool type.
        """
        md = copy.copy(self.metadata)
        md.policy = str(md.policy)
        return dict(
            metadata=to_dict(md),
            solutions=to_dict(self._solutions),
            pool_config=to_dict(self.pool_config),
        )


class SolutionPool_KeepAll(SolutionPoolBase):
    """
    A SolutionPool subclass to keep all added solutions.

    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of solution pools can be handled with the PoolManager class.

    Parameters
    ----------
    name : str
        String name to describe the pool.
    as_solution : Callable[..., Solution] or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default_as_solution function being used.
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing.
        A value of None will result in a new PoolCounter object being created and used.
    """

    def __init__(self, name=None, as_solution=None, counter=None):
        super().__init__(name, as_solution, counter, policy=PoolPolicy.keep_all)

    def add(self, *args, **kwargs):
        """
        Add inputted solution to SolutionPool.
        Relies on the instance as_solution conversion method to convert inputs to Solution Object.
        Adds the converted Solution object to the pool dictionary.
        ID value for the solution generated as next increment of instance PoolCounter.

        Parameters
        ----------
        Input needs to match as_solution format from pool initialization.

        Returns
        ----------
        int
            The ID value to match the added solution from the solution pool's PoolCounter.
            The ID value is also the pool dictionary key for this solution.
        """
        if len(args) == 1 and not kwargs and isinstance(args[0], Solution):
            soln = args[0]
        else:
            soln = self._as_solution(*args, **kwargs)
        #
        soln.id = self._next_solution_counter()
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self._solutions[soln.id] = soln
        return soln.id


class SolutionPool_KeepLatest(SolutionPoolBase):
    """
    A subclass of SolutionPool with the policy of keep the latest k solutions.
    Added solutions are not checked for uniqueness.

    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of solution pools can be handled with the PoolManager class.

    Parameters
    ----------
    name : str
        String name to describe the pool.
    as_solution : Callable[..., Solution] or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default_as_solution function being used.
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing.
        A value of None will result in a new PoolCounter object being created and used.
    max_pool_size : int
        The max_pool_size is the K value for keeping the latest K solutions.
        Must be a positive integer.
    """

    def __init__(self, name=None, as_solution=None, counter=None, *, max_pool_size=1):
        assert max_pool_size >= 1, "max_pool_size must be positive integer"
        super().__init__(name, as_solution, counter, policy=PoolPolicy.keep_latest)
        self.max_pool_size = max_pool_size
        self._int_deque = collections.deque()

    @property
    def pool_config(self):
        return dict(max_pool_size=self.max_pool_size)

    def add(self, *args, **kwargs):
        """
        Add inputed solution to SolutionPool.

        This method relies on the instance as_solution conversion function
        to convert the inputs to a Solution object.  This solution is
        added to the pool dictionary.  The ID value for the solution
        generated is the next increment of instance PoolCounter.

        When pool size < max_pool_size, new solution is added without deleting old solutions.
        When pool size == max_pool_size, new solution is added and oldest solution deleted.

        Parameters
        ----------
        Input needs to match as_solution format from pool initialization.

        Returns
        ----------
        int
            The ID value to match the added solution from the solution pool's PoolCounter.
            The ID value is also the pool dictionary key for this solution.
        """
        if len(args) == 1 and not kwargs and isinstance(args[0], Solution):
            soln = args[0]
        else:
            soln = self._as_solution(*args, **kwargs)
        #
        soln.id = self._next_solution_counter()
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self._solutions[soln.id] = soln
        self._int_deque.append(soln.id)
        if len(self._int_deque) > self.max_pool_size:
            index = self._int_deque.popleft()
            del self._solutions[index]
        #
        return soln.id


class SolutionPool_KeepLatestUnique(SolutionPoolBase):
    """
    A subclass of SolutionPool with the policy of keep the latest k unique solutions.
    Added solutions are checked for uniqueness.

    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of solution pools can be handled with the PoolManager class.

    Parameters
    ----------
    name : str
        String name to describe the pool.
    as_solution : Callable[..., Solution] or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default_as_solution function being used.
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing.
        A value of None will result in a new PoolCounter object being created and used.
    max_pool_size : int
        The max_pool_size is the K value for keeping the latest K solutions.
        Must be a positive integer.
    """

    def __init__(self, name=None, as_solution=None, counter=None, *, max_pool_size=1):
        assert max_pool_size >= 1, "max_pool_size must be positive integer"
        super().__init__(
            name, as_solution, counter, policy=PoolPolicy.keep_latest_unique
        )
        self.max_pool_size = max_pool_size
        self._int_deque = collections.deque()
        self._unique_solutions = set()

    @property
    def pool_config(self):
        return dict(max_pool_size=self.max_pool_size)

    def add(self, *args, **kwargs):
        """
        Add inputted solution to SolutionPool.
        Relies on the instance as_solution conversion method to convert inputs to Solution Object.
        If solution already present, new solution is not added.
        If input solution is new, the converted Solution object to the pool dictionary.
        ID value for the solution generated as next increment of instance PoolCounter.
        When pool size < max_pool_size, new solution is added without deleting old solutions.
        When pool size == max_pool_size, new solution is added and oldest solution deleted.

        Parameters
        ----------
        Input needs to match as_solution format from pool initialization.

        Returns
        ----------
        None or int
            None value corresponds to solution was already present and is ignored.
            When not present, the ID value to match the added solution from the solution pool's PoolCounter.
            The ID value is also the pool dictionary key for this solution.
        """
        if len(args) == 1 and not kwargs and isinstance(args[0], Solution):
            soln = args[0]
        else:
            soln = self._as_solution(*args, **kwargs)
        #
        # Return None if the solution has already been added to the pool
        #
        tuple_repn = soln._tuple_repn()
        if tuple_repn in self._unique_solutions:
            return None
        self._unique_solutions.add(tuple_repn)
        #
        soln.id = self._next_solution_counter()
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self._int_deque.append(soln.id)
        if len(self._int_deque) > self.max_pool_size:
            index = self._int_deque.popleft()
            del self._solutions[index]
        #
        self._solutions[soln.id] = soln
        return soln.id


@dataclasses.dataclass(order=True)
class HeapItem:
    value: float
    id: int = dataclasses.field(compare=False)


class SolutionPool_KeepBest(SolutionPoolBase):
    """
    A subclass of SolutionPool with the policy of keep the best k unique solutions based on objective.
    Added solutions are checked for uniqueness.
    Both the relative and absolute tolerance must be passed to add a solution.


    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of solution pools can be handled with the PoolManager class.

    Parameters
    ----------
    name : str
        String name to describe the pool.
    as_solution : Callable[..., Solution] or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default_as_solution function being used.
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing.
        A value of None will result in a new PoolCounter object being created and used.
    max_pool_size : None or int
        Value of None results in no max pool limit based on number of solutions.
        If not None, the value must be a positive integer.
        The max_pool_size is the K value for keeping the latest K solutions.
    objective : int
        The index of the objective function that is used to compare solutions.
    abs_tolerance : None or int
        absolute tolerance from best solution based on objective beyond which to reject a solution.
        None results in absolute tolerance test passing new solution.
    rel_tolernace : None or int
        relative tolerance from best solution based on objective beyond which to reject a solution.
        None results in relative tolerance test passing new solution.
    sense_is_min : Boolean
        Sense information to encode either minimization or maximization.
        True means minimization problem. False means maximization problem.
    best_value : float
        Optional information to provide a starting best-discovered value for tolerance comparisons.
        Defaults to a 'nan' value that the first added solution's value will replace.
    """

    def __init__(
        self,
        name=None,
        as_solution=None,
        counter=None,
        *,
        max_pool_size=None,
        objective=0,
        abs_tolerance=0.0,
        rel_tolerance=None,
        sense_is_min=True,
        best_value=nan,
    ):
        super().__init__(name, as_solution, counter, policy=PoolPolicy.keep_best)
        assert (max_pool_size is None) or (
            max_pool_size >= 1
        ), "max_pool_size must be None or positive integer"
        self.max_pool_size = max_pool_size
        self.objective = objective
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance
        self.sense_is_min = sense_is_min
        self.best_value = best_value
        self._heap = []
        self._unique_solutions = set()

    @property
    def pool_config(self):
        return dict(
            max_pool_size=self.max_pool_size,
            objective=self.objective,
            abs_tolerance=self.abs_tolerance,
            rel_tolerance=self.rel_tolerance,
            sense_is_min=self.sense_is_min,
            best_value=self.best_value,
        )

    def add(self, *args, **kwargs):
        """
        Add inputted solution to SolutionPool.
        Relies on the instance as_solution conversion method to convert inputs to Solution Object.
        If solution already present or outside tolerance of the best objective value, new solution is not added.
        If input solution is new and within tolerance of the best objective value, the converted Solution object to the pool dictionary.
        ID value for the solution generated as next increment of instance PoolCounter.
        When pool size < max_pool_size, new solution is added without deleting old solutions.
        When pool size == max_pool_size, new solution is added and oldest solution deleted.

        Parameters
        ----------
        Input needs to match as_solution format from pool initialization.

        Returns
        ----------
        None or int
            None value corresponds to solution was already present and is ignored.
            When not present, the ID value to match the added solution from the solution pool's PoolCounter.
            The ID value is also the pool dictionary key for this solution.
        """
        if len(args) == 1 and not kwargs and isinstance(args[0], Solution):
            soln = args[0]
        else:
            soln = self._as_solution(*args, **kwargs)
        #
        # Return None if the solution has already been added to the pool
        #
        tuple_repn = soln._tuple_repn()
        if tuple_repn in self._unique_solutions:
            return None
        self._unique_solutions.add(tuple_repn)
        #
        value = soln.objective(self.objective).value
        keep = False
        new_best_value = False
        if self.best_value is nan:
            self.best_value = value
            keep = True
        else:
            diff = (
                value - self.best_value
                if self.sense_is_min
                else self.best_value - value
            )
            if diff < 0.0:
                # Keep if this is a new best value
                self.best_value = value
                keep = True
                new_best_value = True
            elif ((self.abs_tolerance is None) or (diff <= self.abs_tolerance)) and (
                (self.rel_tolerance is None)
                or (
                    diff / min(math.fabs(value), math.fabs(self.best_value))
                    <= self.rel_tolerance
                )
            ):
                # Keep if the absolute or relative difference with the best value is small enough
                keep = True

        if not keep:
            return None

        soln.id = self._next_solution_counter()
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self._solutions[soln.id] = soln
        #
        item = HeapItem(value=-value if self.sense_is_min else value, id=soln.id)
        if self.max_pool_size is None or len(self._heap) < self.max_pool_size:
            # There is room in the pool, so we just add it
            heapq.heappush(self._heap, item)
        else:
            # We add the item to the pool and pop the worst item in the pool
            item = heapq.heappushpop(self._heap, item)
            del self._solutions[item.id]

        if new_best_value:
            # We have a new best value, so we need to check that all existing solutions are close enough and re-heapify
            tmp = []
            for item in self._heap:
                value = -item.value if self.sense_is_min else item.value
                diff = (
                    value - self.best_value
                    if self.sense_is_min
                    else self.best_value - value
                )
                if ((self.abs_tolerance is None) or (diff <= self.abs_tolerance)) and (
                    (self.rel_tolerance is None)
                    or (
                        diff / min(math.fabs(value), math.fabs(self.best_value))
                        <= self.rel_tolerance
                    )
                ):
                    tmp.append(item)
                else:
                    del self._solutions[item.id]
            heapq.heapify(tmp)
            self._heap = tmp

        assert len(self._solutions) == len(
            self._heap
        ), f"Num solutions is {len(self._solutions)} but the heap size is {len(self._heap)}"
        return soln.id


class PoolManager:
    """
    Manages one or more solution pools.

    The default solution pool has policy ``keep_best`` with name ``None``.
    If a new Solution pool is added without a name, then the ``None``
    pool is replaced.  Otherwise, if a solution pool is added with an
    existing name an error occurs.

    The pool manager always has an active pool.  The pool manager has the
    same API as a solution pool, and the envelope design pattern is used
    to expose the methods and data for the active pool.  The active pool
    defaults to the pool that was most recently added to the pool manager.

    Note that all pools share the same Counter object to enable overall
    solution count tracking and unique solution id values.

    """

    _policy_dispatcher = {
        PoolPolicy.keep_all: SolutionPool_KeepAll,
        PoolPolicy.keep_best: SolutionPool_KeepBest,
        PoolPolicy.keep_latest: SolutionPool_KeepLatest,
        PoolPolicy.keep_latest_unique: SolutionPool_KeepLatestUnique,
    }

    def __init__(self):
        self._name = None
        self._pools = {}
        self.add_pool(name=self._name)
        self._solution_counter = 0

    #
    # The following methods give the PoolManager the same API as a pool.
    # These methods pass-though and operate on the active pool.
    #

    @property
    def name(self):
        """
        Returns
        -------
        str
            The name of the active pool.
        """
        return self._name

    @property
    def metadata(self):
        """
        Returns
        -------
        Munch
            Metadata for the active pool.
        """
        return self.active_pool.metadata

    @property
    def policy(self):
        """
        Returns
        -------
        str
            The policy that is executed by the active pool.
        """
        return self.active_pool.policy

    @property
    def solutions(self):
        """
        Returns
        -------
        list
            The solutions in the active pool.
        """
        return self.active_pool.solutions.values()

    @property
    def last_solution(self):
        """
        Returns
        -------
        Solution
            The last solution added to the active pool.
        """
        return self.active_pool.last_solution

    @property
    def max_pool_size(self):
        """
        Returns
        -------
        int or None
            The maximum pool size value for the active pool, or None if this parameter is not by this pool.
        """
        return getattr(self.active_pool, 'max_pool_size', None)

    def to_dict(self):
        """
        Returns
        -------
        dict
            A dictionary representation of the active pool.
        """
        return self.active_pool.to_dict()

    def __iter__(self):
        """
        Yields
        -------
        Solution
            The solutions in the active pool.
        """
        for soln in self.active_pool.solutions:
            yield soln

    def __len__(self):
        """
        Returns
        -------
        int
            The number of solutions in the active pool.
        """
        return len(self.active_pool)

    def __getitem__(self, soln_id):
        """
        Returns
        -------
        Solution
            The specified solution in the active pool.
        """
        return self._pools[self._name][soln_id]

    def add(self, *args, **kwargs):
        """
        Adds a solution to the active pool.

        Returns
        ----------
        int
            The index of the solution that is added.
        """
        return self.active_pool.add(*args, **kwargs)

    #
    # The following methods support the management of multiple
    # pools within a PoolManager.
    #

    @property
    def active_pool(self):
        """
        Gets the underlying active SolutionPool in PoolManager

        Returns
        ----------
        SolutionPool
            Active pool object

        """
        assert self._name in self._pools, f"Unknown pool '{self._name}'"
        return self._pools[self._name]

    def add_pool(
        self, *, name=None, policy=PoolPolicy.keep_best, as_solution=None, **kwds
    ):
        """
        Initializes a new solution pool and adds it to this pool manager.

        The method expects parameters for the constructor of the corresponding solution pool.
        Supported pools are `keep_all`, `keep_best`, `keep_latest`, and `keep_latest_unique`.

        Parameters
        ----------
        name : str
            The name of the solution pool.  If name is already used then, then an error is generated.
        policy : PoolPolicy
            This enum value indicates the policy that is enforced new solution pool.
            (Default is PoolPolicy.keep_best.)
        as_solution : Callable[..., Solution] or None
            Method for converting inputs into Solution objects.
            A value of None will result in the default_as_solution function being used.
        **kwds
            Other associated arguments that are used to initialize the solution pool.

        Returns
        ----------
        dict
            Metadata for the newly create solution pool.
        """
        if name is None and None in self._pools:
            del self._pools[None]

        if name not in self._pools:
            # Delete the 'None' pool if it isn't being used
            if name is not None and None in self._pools and len(self._pools[None]) == 0:
                del self._pools[None]

            if not policy in self._policy_dispatcher:
                raise ValueError(f"Unknown pool policy: {policy} {type(policy)}")
            self._pools[name] = self._policy_dispatcher[policy](
                name=name, as_solution=as_solution, counter=weakref.proxy(self), **kwds
            )

        self._name = name
        return self.metadata

    def activate(self, name):
        """
        Sets the named SolutionPool to be the active pool in PoolManager

        Parameters
        ----------
        name : str
            name key to pick the SolutionPool in the PoolManager object to the active pool
            If name not a valid key then assertion error thrown
        Returns
        ----------
        dict
            Metadata attribute of the now active SolutionPool

        """
        if not name in self._pools:
            raise ValueError(f"Unknown pool '{name}'")
        self._name = name
        return self.metadata

    #
    # The following methods provide information about all
    # pools in the pool manager.
    #

    def get_pool_dicts(self):
        """
        Converts the set of pools to dictionary object with underlying dictionary of pools

        Returns
        ----------
        dict
            Keys are names of each pool in PoolManager
            Values are to_dict called on corresponding pool

        """
        return {k: v.to_dict() for k, v in self._pools.items()}

    def get_pool_names(self):
        """
        Returns the list of name keys for the pools in PoolManager

        Returns
        ----------
        List
            List of name keys of all pools in this PoolManager

        """
        return list(self._pools.keys())

    def get_pool_policies(self):
        """
        Returns the dictionary of name:policy pairs to identify policies in all Pools

        Returns
        ----------
        List
            List of name keys of all pools in this PoolManager

        """
        return {k: v.policy for k, v in self._pools.items()}

    def get_max_pool_sizes(self):
        """
        Returns the max_pool_size of all pools in the PoolManager as a dict.
        If a pool does not have a max_pool_size that value defaults to none

        Returns
        ----------
        dict
            keys as name of the pool
            values as max_pool_size attribute, if not defined, defaults to None

        """
        return {k: getattr(v, "max_pool_size", None) for k, v in self._pools.items()}

    def get_pool_sizes(self):
        """
        Returns the len of all pools in the PoolManager as a dict.

        Returns
        ----------
        dict
            keys as name of the pool
            values as the number of solutions in the underlying pool

        """
        return {k: len(v) for k, v in self._pools.items()}

    def write(self, json_filename, indent=None, sort_keys=True):
        """
        Dumps PoolManager to json file using json.dump method

        Parameters
        ----------
        json_filename : path-like
            Name of file output location
            If filename exists, will overwrite.
            If filename does not exist, will create.
        indent : int or str or None
            Pass through indent type for json.dump indent
        sort_keys : Boolean
            Pass through sort_keys for json.dump
            If true, keys from dict conversion will be sorted in json
            If false, no sorting

        """
        with open(json_filename, "w") as OUTPUT:
            json.dump(self.to_dict(), OUTPUT, indent=indent, sort_keys=sort_keys)

    def read(self, json_filename):
        """
        Reads in a json to construct the PoolManager pools

        Parameters
        ----------
        json_filename : path-like
            File name to read in as SolutionPools for this PoolManager
            If corresponding file does not exist, throws assertion error

        """
        # TODO: this does not set an active pool, should we do that?
        # TODO: this does not seem to update the counter value, possibly leading to non-unique ids
        assert os.path.exists(
            json_filename
        ), f"ERROR: file '{json_filename}' does not exist!"
        with open(json_filename, "r") as INPUT:
            try:
                data = json.load(INPUT)
            except ValueError as e:
                raise ValueError(f"Invalid JSON in file '{json_filename}': {e}")
            self._pools = data.solutions

    #
    # The following methods treat the PoolManager as a PoolCounter.
    # This allows the PoolManager to be used to provide a global solution count
    # for all pools that it manages.
    #

    @property
    def solution_counter(self):
        return self._solution_counter

    @solution_counter.setter
    def solution_counter(self, value):
        self._solution_counter = value


class PyomoPoolManager(PoolManager):
    """
    A subclass of PoolManager for handing pools of Pyomo solutions.

    This class redefines the add_pool method to use the
    default_as_pyomo_solution method to construct Solution objects.
    Otherwise, this class inherits from PoolManager.
    """

    def add_pool(
        self, *, name=None, policy=PoolPolicy.keep_best, as_solution=None, **kwds
    ):
        """
        Initializes a new solution pool and adds it to this pool manager.

        The method expects parameters for the constructor of the corresponding solution pool.
        Supported pools are `keep_all`, `keep_best`, `keep_latest`, and `keep_latest_unique`.

        Parameters
        ----------
        name : str
            The name of the solution pool.  If name is already used then, then an error is generated.
        policy : PoolPolicy
            This enum value indicates the policy that is enforced new solution pool.
            (Default is PoolPolicy.keep_best.)
        as_solution : Callable[..., Solution] or None
            Method for converting inputs into Solution objects.
            A value of None will result in the _as_pyomo_solution function being used.
        **kwds
            Other associated arguments that are used to initialize the solution pool.

        Returns
        ----------
        dict
            Metadata for the newly create solution pool.

        """
        if as_solution is None:
            as_solution = _as_pyomo_solution
        return PoolManager.add_pool(
            self, name=name, policy=policy, as_solution=as_solution, **kwds
        )
