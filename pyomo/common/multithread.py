from collections import defaultdict
from threading import get_ident, main_thread


class MultiThreadWrapper:
    """A python object proxy that wraps different instances for each thread.

    This is useful for handling thread-safe access to singleton objects without
    having to refactor the code that depends on them.

    Note that instances of the wrapped object are reused if two threads share the same
    identifier, because identifiers could be reused and are unique only for any given moment.
    See [get_ident()](https://docs.python.org/3/library/threading.html#threading.get_ident) for more information.
    """

    __slots__ = '_mtdict'

    def __init__(self, base):
        object.__setattr__(self, '_mtdict', defaultdict(base))

    def __getattr__(self, attr):
        return getattr(self._mtdict[get_ident()], attr)

    def __setattr__(self, attr, value):
        setattr(self._mtdict[get_ident()], attr, value)

    def __delattr__(self, attr):
        delattr(self._mtdict[get_ident()], attr)

    def __enter__(self):
        return self._mtdict[get_ident()].__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._mtdict[get_ident()].__exit__(exc_type, exc_value, traceback)

    def __dir__(self):
        return list(object.__dir__(self)) + list(self._mtdict[get_ident()].__dir__())

    def __str__(self):
        return self._mtdict[get_ident()].__str__()

    def __new__(cls, wrapped):
        return super().__new__(
            type(
                'MultiThreadMeta' + wrapped.__name__,
                (cls,),
                {'__doc__': wrapped.__doc__},
            )
        )


class MultiThreadWrapperWithMain(MultiThreadWrapper):
    """An extension of `MultiThreadWrapper` that exposes the wrapped instance
    corresponding to the [main_thread()](https://docs.python.org/3/library/threading.html#threading.main_thread)
    under the `.main_thread` field.

    This is useful for a falling back to a main instance when needed, but results
    in race conditions if used improperly.
    """

    def __init__(self, base):
        super().__init__(base)

    def __getattr__(self, attr):
        if attr == 'main_thread':
            return self._mtdict[main_thread().ident]
        return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        if attr == 'main_thread':
            raise ValueError('Setting `main_thread` attribute is not allowed')
        else:
            super().__setattr__(attr, value)

    def __dir__(self):
        return super().__dir__() + ['main_thread']
