from collections import defaultdict
from threading import get_ident


class MultiThreadWrapper(object):
    def __init__(self, base):
        self.__mtdict = defaultdict(base)

    def __getattr__(self, attr):
        try:
            return super(MultiThreadWrapper, self).__getattr__(attr)
        except AttributeError:
            id = get_ident()
            return getattr(self.__mtdict[id], attr)
