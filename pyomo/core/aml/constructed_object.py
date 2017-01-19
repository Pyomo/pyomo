class IConstructedObject(object):
    """Interface for modeling objects that are constructed.

    This class assumes derived classes will define a
    _constructed attribute (and declare it as a slot if
    necessary).
    """
    __slots__ = ()

    @property
    def constructed(self):
        return self._constructed

    # for backwards compatibility
    def is_constructed(self):
        return self._constructed

    def construct(self, data=None):
        raise NotImplementedError     #pragma:nocover
