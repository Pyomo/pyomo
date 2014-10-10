from coopr.pyomo import *

class Foo:
    pass
anotherObject = Foo()
anotherObject.x = Var([10])
anotherObject.x.value = 42
