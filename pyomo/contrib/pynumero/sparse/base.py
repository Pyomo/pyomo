

# ToDo: make this an abstract class
class SparseBase(object):

    def __init__(self):
        self._symmetric = False
        self._name = None

    @property
    def is_symmetric(self):
        return self._symmetric

    @is_symmetric.setter
    def is_symmetric(self, value):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, mname):
        self._name = mname

    def set_shape(self, shape):
        raise RuntimeError('set shape is not supported')

    def tofullmatrix(self):
        return NotImplemented
