import heapq
import collections
import dataclasses
import json
import weakref

from .aos_utils import MyMunch, _to_dict
from .solution import Solution, PyomoSolution

nan = float("nan")


def _as_solution(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0:
        assert type(args[0]) is Solution, "Expected a single solution"
        return args[0]
    return Solution(*args, **kwargs)


def _as_pyomo_solution(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0:
        assert type(args[0]) is Solution, "Expected a single solution"
        return args[0]
    return PyomoSolution(*args, **kwargs)


class PoolCounter:

    solution_counter = 0


class SolutionPoolBase:
    """
    A class for handing groups of solutions as pools
    This is the general base pool class.

    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of solution pools can be handled with the PoolManager class.

    Parameters
    ----------
    name : String
        String name of the pool object
    as_solution : Function or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default _as_solution method being used
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing
        A value of None will result in a new PoolCounter object being used
    policy : String
        String name for the pool construction policy
    """

    def __init__(self, name, as_solution, counter, policy="unspecified"):
        # TODO: what is the point of the metadata attribute? Can we add the policy to this
        # TODO: can we add subclass specific data to metadata object e.g. max_pool_size, abs_tolerance, objective
        self.metadata = MyMunch(context_name=name)
        self._solutions = {}
        self._policy = policy
        if as_solution is None:
            self._as_solution = _as_solution
        else:
            self._as_solution = as_solution
        if counter is None:
            self.counter = PoolCounter()
        else:
            self.counter = counter

    @property
    def solutions(self):
        return self._solutions.values()

    @property
    def last_solution(self):
        index = next(reversed(self._solutions.keys()))
        return self._solutions[index]

    @property
    def policy(self):
        return self._policy

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


class SolutionPool_KeepAll(SolutionPoolBase):
    """
    A subclass of SolutionPool with the policy of keeping all added solutions

    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of solution pools can be handled with the PoolManager class.

    Parameters
    ----------
    name : String
        String name of the pool object
    as_solution : Function or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default _as_solution method being used
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing
        A value of None will result in a new PoolCounter object being used
    """

    def __init__(self, name=None, as_solution=None, counter=None):
        super().__init__(name, as_solution, counter, policy="keep_all")

    def add(self, *args, **kwargs):
        """
        Add input solution to SolutionPool.
        Relies on the instance as_solution conversion method to convert inputs to Solution Object.
        Adds the converted Solution object to the pool dictionary.
        ID value for the solution genenerated as next increment of instance PoolCounter

        Parameters
        ----------
        General format accepted.
        Needs to match as_solution format

        Returns
        ----------
        int
            ID value for the added Solution object in the pool dictionary
        """
        soln = self._as_solution(*args, **kwargs)
        #
        soln.id = self._next_solution_counter()
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self._solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        """
        Converts SolutionPool to dictionary

        Returns
        ----------
        dict
            Dictionary of dictionaries for SolutionPool members
            metadata corresponding to _to_dict of self.metadata
            solutions corresponding to _to_dict of self._solutions
            pool_config corresponding to a dictionary of pool details with keys as member names
                including: self.policy
        """
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(policy=self._policy),
        )


class SolutionPool_KeepLatest(SolutionPoolBase):
    """
    A subclass of SolutionPool with the policy of keep the latest k solutions.
    Added solutions are not checked for uniqueness


    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of solution pools can be handled with the PoolManager class.

    Parameters
    ----------
    name : String
        String name of the pool object
    as_solution : Function or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default _as_solution method being used
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing
        A value of None will result in a new PoolCounter object being used
    max_pool_size : int
        The max_pool_size is the K value for keeping the latest K solutions.
        Must be a positive integer.
    """

    def __init__(self, name=None, as_solution=None, counter=None, *, max_pool_size=1):
        assert max_pool_size >= 1, "max_pool_size must be positive integer"
        super().__init__(name, as_solution, counter, policy="keep_latest")
        self.max_pool_size = max_pool_size
        self.int_deque = collections.deque()

    def add(self, *args, **kwargs):
        """
        Add input solution to SolutionPool.
        Relies on the instance as_solution conversion method to convert inputs to Solution Object.
        Adds the converted Solution object to the pool dictionary.
        ID value for the solution genenerated as next increment of instance PoolCounter
        When pool size < max_pool_size, new solution is added without deleting old solutions
        When pool size == max_pool_size, new solution is added and oldest solution deleted

        Parameters
        ----------
        General format accepted.
        Needs to match as_solution format

        Returns
        ----------
        int
            ID value for the added Solution object in the pool dictionary
        """
        soln = self._as_solution(*args, **kwargs)
        #
        soln.id = self._next_solution_counter()
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self.int_deque.append(soln.id)
        if len(self.int_deque) > self.max_pool_size:
            index = self.int_deque.popleft()
            del self._solutions[index]
        #
        self._solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        """
        Converts SolutionPool to dictionary

        Returns
        ----------
        dict
            Dictionary of dictionaries for SolutionPool members
            metadata corresponding to _to_dict of self.metadata
            solutions corresponding to _to_dict of self._solutions
            pool_config corresponding to a dictionary of self.policy and self.max_pool_size
        """
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(policy=self._policy, max_pool_size=self.max_pool_size),
        )


class SolutionPool_KeepLatestUnique(SolutionPoolBase):
    """
    A subclass of SolutionPool with the policy of keep the latest k unique solutions.
    Added solutions are checked for uniqueness


    This class is designed to integrate with the alternative_solution generation methods.
    Additionally, groups of solution pools can be handled with the PoolManager class.

    Parameters
    ----------
    name : String
        String name of the pool object
    as_solution : Function or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default _as_solution method being used
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing
        A value of None will result in a new PoolCounter object being used
    max_pool_size : int
        The max_pool_size is the K value for keeping the latest K solutions.
        Must be a positive integer.
    """

    def __init__(self, name=None, as_solution=None, counter=None, *, max_pool_size=1):
        assert max_pool_size >= 1, "max_pool_size must be positive integer"
        super().__init__(name, as_solution, counter, policy="keep_latest_unique")
        self.max_pool_size = max_pool_size
        self.int_deque = collections.deque()
        self.unique_solutions = set()

    def add(self, *args, **kwargs):
        """
        Add input solution to SolutionPool.
        Relies on the instance as_solution conversion method to convert inputs to Solution Object.
        If solution already present, new solution is not added.
        If input solution is new, the converted Solution object to the pool dictionary.
        ID value for the solution genenerated as next increment of instance PoolCounter
        When pool size < max_pool_size, new solution is added without deleting old solutions
        When pool size == max_pool_size, new solution is added and oldest solution deleted

        Parameters
        ----------
        General format accepted.
        Needs to match as_solution format

        Returns
        ----------
        None or int
            None value corresponds to solution was already present and is ignored
            int corresponds to ID value for the added Solution object in the pool dictionary
        """
        soln = self._as_solution(*args, **kwargs)
        #
        # Return None if the solution has already been added to the pool
        #
        tuple_repn = soln._tuple_repn()
        if tuple_repn in self.unique_solutions:
            return None
        self.unique_solutions.add(tuple_repn)
        #
        soln.id = self._next_solution_counter()
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self.int_deque.append(soln.id)
        if len(self.int_deque) > self.max_pool_size:
            index = self.int_deque.popleft()
            del self._solutions[index]
        #
        self._solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        """
        Converts SolutionPool to dictionary

        Returns
        ----------
        dict
            Dictionary of dictionaries for SolutionPool members
            metadata corresponding to _to_dict of self.metadata
            solutions corresponding to _to_dict of self._solutions
            pool_config corresponding to a dictionary of pool details with keys as member names
                including: self.policy, self.max_pool_size
        """
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(policy=self._policy, max_pool_size=self.max_pool_size),
        )


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
    name : String
        String name of the pool object
    as_solution : Function or None
        Method for converting inputs into Solution objects.
        A value of None will result in the default _as_solution method being used
    counter : PoolCounter or None
        PoolCounter object to manage solution indexing
        A value of None will result in a new PoolCounter object being used
    max_pool_size : int
        The max_pool_size is the K value for keeping the latest K solutions.
        Must be a positive integer.
    objective : None or Function
        The function to compare solutions based on.
        None results in use of the constant function 0
    abs_tolerance : None or int
        absolute tolerance from best solution based on objective beyond which to reject a solution
        None results in absolute tolerance test passing new solution
    rel_tolernace : None or int
        relative tolerance from best solution based on objective beyond which to reject a solution
        None results in relative tolerance test passing new solution
    keep_min : Boolean
        TODO: fill in
    best_value : float
        TODO: fill in
    """

    # TODO: pool design seems to assume problem sense as min, do we want to add sense to support max?
    def __init__(
        self,
        name=None,
        as_solution=None,
        counter=None,
        *,
        max_pool_size=None,
        objective=None,
        abs_tolerance=0.0,
        rel_tolerance=None,
        keep_min=True,
        best_value=nan,
    ):
        super().__init__(name, as_solution, counter, policy="keep_best")
        assert (max_pool_size is None) or (
            max_pool_size >= 1
        ), "max_pool_size must be None or positive integer"
        self.max_pool_size = max_pool_size
        self.objective = 0 if objective is None else objective
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance
        self.keep_min = keep_min
        self.best_value = best_value
        self.heap = []
        self.unique_solutions = set()

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        # Return None if the solution has already been added to the pool
        #
        tuple_repn = soln._tuple_repn()
        if tuple_repn in self.unique_solutions:
            return None
        self.unique_solutions.add(tuple_repn)
        #
        value = soln.objective(self.objective).value
        keep = False
        new_best_value = False
        if self.best_value is nan:
            self.best_value = value
            keep = True
        else:
            diff = value - self.best_value if self.keep_min else self.best_value - value
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

        if keep:
            soln.id = self._next_solution_counter()
            assert (
                soln.id not in self._solutions
            ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
            #
            self._solutions[soln.id] = soln
            #
            item = HeapItem(value=-value if self.keep_min else value, id=soln.id)
            if self.max_pool_size is None or len(self.heap) < self.max_pool_size:
                # There is room in the pool, so we just add it
                heapq.heappush(self.heap, item)
            else:
                # We add the item to the pool and pop the worst item in the pool
                item = heapq.heappushpop(self.heap, item)
                del self._solutions[item.id]

            if new_best_value:
                # We have a new best value, so we need to check that all existing solutions are close enough and re-heapify
                tmp = []
                for item in self.heap:
                    value = -item.value if self.keep_min else item.value
                    diff = (
                        value - self.best_value
                        if self.keep_min
                        else self.best_value - value
                    )
                    if (
                        (self.abs_tolerance is None) or (diff <= self.abs_tolerance)
                    ) and (
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
                self.heap = tmp

            assert len(self._solutions) == len(
                self.heap
            ), f"Num solutions is {len(self._solutions)} but the heap size is {len(self.heap)}"
            return soln.id

        return None

    def to_dict(self):
        """
        Converts SolutionPool to dictionary

        Returns
        ----------
        dict
            Dictionary of dictionaries for SolutionPool members
            metadata corresponding to _to_dict of self.metadata
            solutions corresponding to _to_dict of self._solutions
            pool_config corresponding to a dictionary of pool details with keys as member names
                including: self.policy, self.max_pool_size, self.objective
                    self.abs_tolerance, self.rel_tolerance
        """
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(
                policy=self._policy,
                max_pool_size=self.max_pool_size,
                objective=self.objective,
                abs_tolerance=self.abs_tolerance,
                rel_tolerance=self.rel_tolerance,
            ),
        )


class PoolManager:
    """
    A class for handing groups of SolutionPool objects
    Defaults to having a SolutionPool with policy KeepBest under name 'None'
    If a new SolutionPool is added while the 'None' pool is empty, 'None' pool is deleted

    When PoolManager has multiple pools, there is an active pool.
    PoolManager is designed ot have the same API as a pool for the active pool.
    Unless changed, the active pool defaults to the one most recently added to the PoolManager.

    All pools share the same Counter object to enable overall solution count tracking and unique solution id values.

    """

    def __init__(self):
        self._name = None
        self._pools = {}
        self.add_pool(self._name)
        self._solution_counter = 0

    #
    # The following methods give the PoolManager the same API as a pool.
    # These methods pass-though and operate on the active pool.
    #

    @property
    def metadata(self):
        return self.active_pool.metadata

    @property
    def solutions(self):
        return self.active_pool.solutions.values()

    @property
    def last_solution(self):
        return self.active_pool.last_solution

    def __iter__(self):
        for soln in self.active_pool.solutions:
            yield soln

    def __len__(self):
        return len(self.active_pool)

    def __getitem__(self, soln_id):
        return self._pools[self._name][soln_id]

    def add(self, *args, **kwargs):
        """
        Adds input to active SolutionPool

        Returns
        ----------
        Pass through for return value from calling add method on underlying pool
        """
        return self.active_pool.add(*args, **kwargs)

    # TODO as is this method works on all the pools, not the active pool, do we want to change this to enforce active pool API paradigm
    def to_dict(self):
        """
        Converts the set of pools to dictionary object with underlying dictionary of pools

        Returns
        ----------
        dict
            Keys are names of each pool in PoolManager
            Values are to_dict called on corresponding pool

        """
        return {k: v.to_dict() for k, v in self._pools.items()}

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

    def add_pool(self, name, *, policy="keep_best", as_solution=None, **kwds):
        """
        Initializes a new SolutionPool and adds it to the PoolManager.
        The method expects required parameters for the constructor of the corresponding SolutionPool except Counter.
        The counter object is provided by the PoolManager.
        Supported pools are KeepAll, KeepBest, KeepLatest, KeepLatestUnique

        Parameters
        ----------
        name : String
            name for the new pool.
            Acts as key for the new SolutionPool in the dictionary of pools maintained by PoolManager
            If name already used then sets that pool to active but makes no other changes
        policy : String
            String to choose which policy to enforce in the new SolutionPool
            Supported values are ['keep_all', 'keep_best', 'keep_latest', 'keep_latest_unique']
            Unsupported policy name will throw error.
            Default is 'keep_best'
        as_solution : None or Function
            Pass through method for as_solution conversion method to create Solution objects for the new SolutionPool
            Default is None for pass through default as_solution method
        **kwds
            Other associated arguments corresponding to the constructor for intended subclass of SolutionPoolBase

        Returns
        ----------
        dict
            Metadata attribute of the newly create SolutionPool

        """
        if name not in self._pools:
            # Delete the 'None' pool if it isn't being used
            if name is not None and None in self._pools and len(self._pools[None]) == 0:
                del self._pools[None]

            if policy == "keep_all":
                self._pools[name] = SolutionPool_KeepAll(
                    name=name, as_solution=as_solution, counter=weakref.proxy(self)
                )
            elif policy == "keep_best":
                self._pools[name] = SolutionPool_KeepBest(
                    name=name,
                    as_solution=as_solution,
                    counter=weakref.proxy(self),
                    **kwds,
                )
            elif policy == "keep_latest":
                self._pools[name] = SolutionPool_KeepLatest(
                    name=name,
                    as_solution=as_solution,
                    counter=weakref.proxy(self),
                    **kwds,
                )
            elif policy == "keep_latest_unique":
                self._pools[name] = SolutionPool_KeepLatestUnique(
                    name=name,
                    as_solution=as_solution,
                    counter=weakref.proxy(self),
                    **kwds,
                )
            else:
                raise ValueError(f"Unknown pool policy: {policy}")
        self._name = name
        return self.metadata

    def activate(self, name):
        """
        Sets the named SolutionPool to be the active pool in PoolManager

        Parameters
        ----------
        name : String
            name key to pick the SolutionPool in the PoolManager object to the active pool
            If name not a valid key then assertation error thrown
        Returns
        ----------
        dict
            Metadata attribute of the now active SolutionPool

        """
        assert name in self._pools, f"Unknown pool '{name}'"
        self._name = name
        return self.metadata

    def get_active_pool_name(self):
        """
        Returns the name string for the active pool

        Returns
        ----------
        String
            name key for the active pool

        """
        return self._name

    def get_active_pool_policy(self):
        """
        Returns the policy string for the active pool

        Returns
        ----------
        String
            policy in use for the active pool

        """
        return self.active_pool.policy

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

    def get_max_pool_size(self):
        """
        Returns the max_pool_size of the active pool if exists, else none

        Returns
        ----------
        int or None
            max_pool_size attribute of the active pool, if not defined, returns None

        """
        return getattr(self.active_pool, "max_pool_size", None)

    def get_max_pool_sizes(self):
        """
        Returns the max_pool_size of all pools in the PoolManager as a dict.
        If a pool does not have a max_pool_size that value defualts to none

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
        indent : int or String or None
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
            If corresponding file does not exist, throws assertation error

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
    A subclass of PoolManager for handing groups of SolutionPool objects.
    Uses default as_solution method _as_pyomo_solution instead of _as_solution

    Otherwise inherits from PoolManager
    """

    def add_pool(self, name, *, policy="keep_best", as_solution=None, **kwds):
        """
        Initializes a new SolutionPool and adds it to the PoolManager.
        The method expects required parameters for the constructor of the corresponding SolutionPool except Counter.
        The counter object is provided by the PoolManager.
        Supported pools are KeepAll, KeepBest, KeepLatest, KeepLatestUnique

        Parameters
        ----------
        name : String
            name for the new pool.
            Acts as key for the new SolutionPool in the dictionary of pools maintained by PoolManager
            If name already used then sets that pool to active but makes no other changes
        policy : String
            String to choose which policy to enforce in the new SolutionPool
            Supported values are ['keep_all', 'keep_best', 'keep_latest', 'keep_latest_unique']
            Unsupported policy name will throw error.
            Default is 'keep_best'
        as_solution : None or Function
            Pass through method for as_solution conversion method to create Solution objects for the new SolutionPool
            Default is None which results in using _as_pyomo_solution
        **kwds
            Other associated arguments corresponding to the constructor for intended subclass of SolutionPoolBase

        Returns
        ----------
        dict
            Metadata attribute of the newly create SolutionPool

        """
        if as_solution is None:
            as_solution = _as_pyomo_solution
        return PoolManager.add_pool(
            self, name, policy=policy, as_solution=as_solution, **kwds
        )
