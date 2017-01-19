#
# Note, right now this class only supports a concrete
# set-like object (e.g., set, list, tuple) passed in
# as a single argument.
#
class _IndexedComponentContainerMixin(object):
    """Interface for modeling objects that are indexed
    by an external set object."""
    __slots__ = ()

    def __init__(self, arg):
        self._index = arg

    @property
    def index(self):
        return self._index

    def __getitem__(self, key):
        if key not in self.index:
            raise KeyError("The index '%s' is not valid for the "
                           "containers index set." % (str(key)))
        return super(_IndexedComponentContainerMixin, self).__getitem__(key)

    def __setitem__(self, key, val):
        if key not in self.index:
            raise KeyError("The index '%s' is not valid for the "
                           "containers index set." % (str(key)))
        return super(_IndexedComponentContainerMixin, self).__setitem__(key,
                                                                  val)
