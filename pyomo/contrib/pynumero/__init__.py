try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False

if numpy_available:
    from .sparse.intrinsic import *
    print("WARNING: Numpy not available. Install numpy before using pynumero")




