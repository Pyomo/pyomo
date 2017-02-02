# class.py

# @all:
class IntLocker:
    sint = None
    def __init__(self, i):
        self.set_value(i)
    def set_value(self, i):
        if type(i) is not int:
            print("Error: %d is not integer." % i)
        else:
            self.sint = i
    def pprint(self):
        print("The Int Locker has "+str(self.sint))
 
a = IntLocker(3)
a.pprint()
a.set_value(5)
a.pprint()
# @:all
