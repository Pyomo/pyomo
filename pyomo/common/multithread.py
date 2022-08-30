from collections import defaultdict
from threading import get_ident, main_thread


class MultiThreadWrapper():
    """A python object proxy that wraps different instances for each thread.

    This is useful for handling thread-safe access to singleton objects without
    having to refactor the code that depends on them.

    Note that instances of the wrapped object are reused if two threads share the same
    identifier, because identifiers could be reused and are unique only for any given moment.
    See [get_ident()](https://docs.python.org/3/library/threading.html#threading.get_ident) for more information.
    """

    def __init__(self, base):
        self.__mtdict = defaultdict(base)

    def __getattr__(self, attr):
        return getattr(self.__mtdict[get_ident()], attr)
    
    def __setattr__(self, attr, value):
        if attr == '_MultiThreadWrapper__mtdict':
            object.__setattr__(self, attr, value)
        else:
            setattr(self.__mtdict[get_ident()], attr, value)
    
    def __enter__(self):
        return self.__mtdict[get_ident()].__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        return self.__mtdict[get_ident()].__exit__(exc_type, exc_value, traceback)


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
        if attr == '_MultiThreadWrapperWithMain__mtdict':
            return object.__getattribute__(self, '_MultiThreadWrapper__mtdict')
        elif attr == 'main_thread':
            return self.__mtdict[main_thread().ident]
        return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        if attr == 'main_thread':
            raise ValueError('Setting `main_thread` attribute is not allowed')
        else:
            super().__setattr__(attr, value)
