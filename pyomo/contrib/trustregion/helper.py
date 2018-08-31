from numpy import array, concatenate

def cloneXYZ(x, y, z):
    """
    This function is to create a hard copy of vector x, y, z.
    """
    x0 = array(x)
    y0 = array(y)
    z0 = array(z)
    return x0, y0, z0

def packXYZ(x, y, z):
    """
    This function concatenate x, y, x to one vector and return it.
    """
    t = concatenate([x, y, z])
    return t



def minIgnoreNone(a,b):
    if a is None:
        return b
    if b is None:
        return a
    if a<b:
        return a
    return b

def maxIgnoreNone(a,b):
    if a is None:
        return b
    if b is None:
        return a
    if a<b:
        return b
    return a
