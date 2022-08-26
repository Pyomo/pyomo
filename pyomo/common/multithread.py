from collections import defaultdict
from threading import get_ident, current_thread, main_thread


class MultiThreadWrapper():
    def __init__(self, base):
        self.__mtdict = defaultdict(base)

    def __getattr__(self, attr):
        id = get_ident()
        return getattr(self.__mtdict[id], attr)


class MultiThreadWrapperWithMain(MultiThreadWrapper):
    def __init__(self, base):
        super().__init__(base)
        self.main = base()

    def __getattr__(self, attr):
        if current_thread() is main_thread():
            return getattr(self.main, attr)
        return super().__getattr__(attr)
