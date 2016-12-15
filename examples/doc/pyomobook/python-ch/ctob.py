# An example of a silly decorator to change 'c' to 'b' 
# in the return value of a function. 

def ctob_decorate(func):
   def func_wrapper(*args, **kwargs):
       retval = func(*args, **kwargs).replace('c','b')
       return retval.replace('C','B')
   return func_wrapper

@ctob_decorate
def Last_Words():
    return "Flying Circus"

print (Last_Words())
