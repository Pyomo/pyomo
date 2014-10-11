import sys

try:
    import pyomo
    import pyomo.modeling
    import pyomo.core
    print("OK")
except Exception:
    e = sys.exc_info()[1]
    print("Pyomo package error: "+str(e))
