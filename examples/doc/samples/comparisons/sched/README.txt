= Comparing Pyomo with Commercial Modeling Tools: A Scheduling Problem =

This example shows a conference scheduling problem as formulated by 
several different modeling tools.


== Problem Description ==

 * [http://yetanothermathprogrammingconsultant.blogspot.com/2009/03/scheduling-problem.html Erwin Kalvelagen's original blog post describing this problem]

 * [http://brightsparc.wordpress.com/2009/03/31/scheduling-algorithm-in-gams-ampl-mfs-and-lp_solve/ Julian Bright's follow-up describing this problem in AMPL]

== Model Files ==

 * AMPL: [source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/sched/ampl/sched.mod sched.mod] [source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/sched/ampl/sched.dat sched.dat]

 * GAMS: [source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/sched/gams/sched.gms sched.gms]

 * !LpSolve: [source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/sched/lp_solve/srsched.lp srsched.lp] [source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/sched/lp_solve/srsched.mps srsched.mps] [source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/sched/lp_solve/srsched.txt srsched.txt]

 * Pyomo: [source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/sched/pyomo/sched.py sched.py] [source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/sched/pyomo/sched.dat sched.dat]

