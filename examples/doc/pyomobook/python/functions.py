# functions.py

# @all:
def Apply(f, a):
    r = []
    for i in range(len(a)):
        r.append(f(a[i]))
    return r
 
def SqifOdd(x):
    # if x is odd, 2*trunc(x/2) is not x
    # due to integer divide of x/2
    if 2*int(x/2) == x:
        return x
    else:
        return x*x
 
ShortList = range(4)
B = Apply(SqifOdd, ShortList)
print(B)
# @:all
