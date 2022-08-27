from collections import defaultdict
from threading import get_ident, current_thread, main_thread


class MultiThreadWrapper():
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


class MultiThreadWrapperWithMain(MultiThreadWrapper):
    def __init__(self, base):
        super().__init__(base)
        self.main = base()

    def __getattr__(self, attr):
        if current_thread() is main_thread():
            return getattr(self.main, attr)
        return super().__getattr__(attr)
    
    def __setattr__(self, attr, value):
        if attr == '_MultiThreadWrapper__mtdict':
            super().__setattr__(attr, value)
        elif attr == 'main':
            object.__setattr__(self, attr, value)
        elif current_thread() is main_thread():
            setattr(self.main, attr, value)
        else:
            super().__setattr__(attr, value)
