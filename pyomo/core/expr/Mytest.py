#from pyomo.core.expr.numvalue import NonNumericValue, NumericValue, NumericConstant, as_numeric, ZeroConstant

class nonsense(object):
	__slots__ = ()
	def __init__(self):
		1 == 1

	def testmeth(self, word = 'yes', indi = True):
		if indi:
			print(word)	
			return 1
		else:
			print(word)
			return 0
"""
tmp = NumericValue()
print(tmp.name)
print(bool(abs(True*True-True)))
print("copy")
"""
tracer = nonsense()
print(type(tracer))
print(bool(tracer.testmeth()))

