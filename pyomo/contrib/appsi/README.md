# APPSI: Auto Persistent Pyomo Solver Interfaces

APPSI is a collection of solver interfaces which can resolve Pyomo models quickly.

Design Decisisons
-----------------
* Fixing variables does not change the structure of a constraint (linear, quadratic, etc.)
  * Making this decision means we do not need to update constraints when variables in those constraints become fixed.
  * In the NL writer, we can fix variables in the bounds section (i.e., "4 value")
    * Ipopt handles this just fine
    * However, if all of the variables in a constraint are fixed, this can cause ipopt problems. Therefore, we do have
      to take care to exclude constraints with all variables fixed. Fortunately, this is pretty easy and efficient.
      