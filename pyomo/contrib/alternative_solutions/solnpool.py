import heapq
import collections
import dataclasses
import json
import munch

from .aos_utils import MyMunch, _to_dict
from .solution import Solution

nan = float("nan")


class SolutionPoolBase:

    _id_counter = 0

    def __init__(self, name=None):
        self.metadata = MyMunch(context_name=name)
        self._solutions = {}

    @property
    def solutions(self):
        return self._solutions.values()

    @property
    def last_solution(self):
        index = next(reversed(self._solutions.keys()))
        return self._solutions[index]

    def __iter__(self):
        for soln in self._solutions.values():
            yield soln

    def __len__(self):
        return len(self._solutions)

    def __getitem__(self, soln_id):
        return self._solutions[soln_id]

    def _as_solution(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            assert type(args[0]) is Solution, "Expected a single solution"
            return args[0]
        return Solution(*args, **kwargs)


class SolutionPool_KeepAll(SolutionPoolBase):

    def __init__(self, name=None):
        super().__init__(name)

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        soln.id = SolutionPoolBase._id_counter
        SolutionPoolBase._id_counter += 1
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self._solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(policy="keep_all"),
        )


class SolutionPool_KeepLatest(SolutionPoolBase):

    def __init__(self, name=None, *, max_pool_size=1):
        super().__init__(name)
        self.max_pool_size = max_pool_size
        self.int_deque = collections.deque()

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        soln.id = SolutionPoolBase._id_counter
        SolutionPoolBase._id_counter += 1
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
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(policy="keep_latest", max_pool_size=self.max_pool_size),
        )


class SolutionPool_KeepLatestUnique(SolutionPoolBase):

    def __init__(self, name=None, *, max_pool_size=1):
        super().__init__(name)
        self.max_pool_size = max_pool_size
        self.int_deque = collections.deque()
        self.unique_solutions = set()

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        # Return None if the solution has already been added to the pool
        #
        tuple_repn = soln.tuple_repn()
        if tuple_repn in self.unique_solutions:
            return None
        self.unique_solutions.add(tuple_repn)
        #
        soln.id = SolutionPoolBase._id_counter
        SolutionPoolBase._id_counter += 1
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
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(policy="keep_latest_unique", max_pool_size=self.max_pool_size),
        )


@dataclasses.dataclass(order=True)
class HeapItem:
    value: float
    id: int = dataclasses.field(compare=False)


class SolutionPool_KeepBest(SolutionPoolBase):

    def __init__(
        self,
        name=None,
        *,
        max_pool_size=None,
        objective=None,
        abs_tolerance=0.0,
        rel_tolerance=None,
        keep_min=True,
        best_value=nan,
    ):
        super().__init__(name)
        self.max_pool_size = max_pool_size
        self.objective = objective
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance
        self.keep_min = keep_min
        self.best_value = best_value
        self.heap = []
        self.unique_solutions = set()
        self.objective = None

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        # Return None if the solution has already been added to the pool
        #
        tuple_repn = soln.tuple_repn()
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
            soln.id = SolutionPoolBase._id_counter
            SolutionPoolBase._id_counter += 1
            assert (
                soln.id not in self._solutions
            ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
            #
            self._solutions[soln.id] = soln
            #
            item = HeapItem(value=-value if self.keep_min else value, id=soln.id)
            #print(f"ADD {item.id} {item.value}")
            if self.max_pool_size is None or len(self.heap) < self.max_pool_size:
                # There is room in the pool, so we just add it
                heapq.heappush(self.heap, item)
            else:
                # We add the item to the pool and pop the worst item in the pool
                item = heapq.heappushpop(self.heap, item)
                #print(f"DELETE {item.id} {item.value}")
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
                        #print(f"DELETE? {item.id} {item.value}")
                        del self._solutions[item.id]
                heapq.heapify(tmp)
                self.heap = tmp

            assert len(self._solutions) == len(
                self.heap
            ), f"Num solutions is {len(self._solutions)} but the heap size is {len(self.heap)}"
            return soln.id

        return None

    def to_dict(self):
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(
                policy="keep_best",
                max_pool_size=self.max_pool_size,
                objective=self.objective,
                abs_tolerance=self.abs_tolerance,
                rel_tolerance=self.rel_tolerance,
            ),
        )


class PoolManager:

    def __init__(self):
        self._name = None
        self._pool = {}
        self.add_pool(self._name)

    def reset_solution_counter(self):
        SolutionPoolBase._id_counter = 0

    @property
    def pool(self):
        assert self._name in self._pool, f"Unknown pool '{self._name}'"
        return self._pool[self._name]

    @property
    def metadata(self):
        return self.pool.metadata

    @property
    def solutions(self):
        return self.pool.solutions.values()

    @property
    def last_solution(self):
        return self.pool.last_solution

    def __iter__(self):
        for soln in self.pool.solutions:
            yield soln

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, soln_id, name=None):
        if name is None:
            name = self._name
        return self._pool[name][soln_id]

    def add_pool(self, name, *, policy="keep_best", **kwds):
        if name not in self._pool:
            # Delete the 'None' pool if it isn't being used
            if name is not None and None in self._pool and len(self._pool[None]) == 0:
                del self._pool[None]

            if policy == "keep_all":
                self._pool[name] = SolutionPool_KeepAll(name=name)
            elif policy == "keep_best":
                self._pool[name] = SolutionPool_KeepBest(name=name, **kwds)
            elif policy == "keep_latest":
                self._pool[name] = SolutionPool_KeepLatest(name=name, **kwds)
            elif policy == "keep_latest_unique":
                self._pool[name] = SolutionPool_KeepLatestUnique(name=name, **kwds)
            else:
                raise ValueError(f"Unknown pool policy: {policy}")
        self._name = name
        return self.metadata

    def set_pool(self, name):
        assert name in self._pool, f"Unknown pool '{name}'"
        self._name = name
        return self.metadata

    def add(self, *args, **kwargs):
        return self.pool.add(*args, **kwargs)

    def to_dict(self):
        return {k: v.to_dict() for k, v in self._pool.items()}

    def write(self, json_filename, indent=None, sort_keys=True):
        with open(json_filename, "w") as OUTPUT:
            json.dump(self.to_dict(), OUTPUT, indent=indent, sort_keys=sort_keys)

    def read(self, json_filename):
        assert os.path.exists(
            json_filename
        ), f"ERROR: file '{json_filename}' does not exist!"
        with open(json_filename, "r") as INPUT:
            try:
                data = json.load(INPUT)
            except ValueError as e:
                raise ValueError(f"Invalid JSON in file '{json_filename}': {e}")
            self._pool = data.solutions
