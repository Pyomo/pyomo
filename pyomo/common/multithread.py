from collections import defaultdict
from threading import get_ident, current_thread, main_thread


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
        id = get_ident()
        return getattr(self.__mtdict[id], attr)
    
    def __setattr__(self, attr, value):
        if attr == '_MultiThreadWrapper__mtdict':
            object.__setattr__(self, attr, value)
        else:
            id = get_ident()
            setattr(self.__mtdict[id], attr, value)
    
    def __enter__(self):
        id = get_ident()
        return self.__mtdict[id].__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        id = get_ident()
        return self.__mtdict[id].__exit__(exc_type, exc_value, traceback)


class MultiThreadWrapperWithMain(MultiThreadWrapper):
    """An extension of `MultiThreadWrapper` that exposes the wrapped instance
    corresponding to the [main_thread()](https://docs.python.org/3/library/threading.html#threading.main_thread)
    under the `.main_thread` field.

    This is useful for a falling back to a main instance when needed, but results
    in race conditions if used improperly.
    """
    def __init__(self, base):
        super().__init__(base)
        self.main_thread = base()

    def __getattr__(self, attr):
        if current_thread() is main_thread():
            return getattr(self.main_thread, attr)
        return super().__getattr__(attr)
    
    def __setattr__(self, attr, value):
        if attr == '_MultiThreadWrapper__mtdict':
            super().__setattr__(attr, value)
        elif attr == 'main_thread':
            object.__setattr__(self, attr, value)
        elif current_thread() is main_thread():
            setattr(self.main_thread, attr, value)
        else:
            super().__setattr__(attr, value)
    
    def __enter__(self):
        if current_thread() is main_thread():
            return self.main_thread.__enter__()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        if current_thread() is main_thread():
            return self.main_thread.__exit__(exc_type, exc_value, traceback)
        return super().__exit__(exc_type, exc_value, traceback)