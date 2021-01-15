from collections.abc import MutableSet


class OrderedSet(MutableSet):
    def __init__(self, items=None):
        self._data = dict()  # we only support Python >= 3.7
        if items is not None:
            self.update(items)

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def add(self, item):
        self._data[item] = None

    def discard(self, item):
        self._data.pop(item, None)

    def __str__(self):
        s = '{'
        for i in self:
            s += str(i)
            s += ', '
        s += '}'
        return s

    def update(self, items):
        for i in items:
            self._data[i] = None

    def intersection(self, other):
        res = OrderedSet([i for i in self if i in other])
        return res

    def union(self, other):
        res = OrderedSet(self)
        res.update(other)
