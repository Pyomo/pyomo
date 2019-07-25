#testing only
import sys

class unode:

	def __init__(self,val=None): 
		if (val!=True and val!=False and val!=None):
			raise ValueError
		self.val = val 
		#self.next = unode(0)
		# val should be bool

	def __str__(self):
		res = "The value of this node is "
		res += str(self.getval())
		return res 	
		
	def getval(self):
		return self.val;



class operator:
	def __init__(self):
		raise Exception("trying to initialize abstract class")

class bnode:
	def __init__(self, oprt,frst, scd):
		pass
		
assert(True == 1)

#x = operator()
a = unode()
a = unode(1)
print(@a)
print(type(a.getval()))
print(a)
b = unode(0)
print("pass")