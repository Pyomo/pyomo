# Coramin

Coramin is a Pyomo-based Python package that provides tools for
developing tailored algorithms for mixed-integer nonlinear programming
problems (MINLP's). This software includes classes for managing and
refining convex relaxations of nonconvex constraints. These classes
provide methods for updating the relaxation based on new variable
bounds, creating and managing piecewise relaxations (for multi-tree
based algorithms), and adding outer-approximation based cuts for
convex or concave constraints. These classes inherit from Pyomo
Blocks, so they can be easily integrated with Pyomo
models. Additionally, Coramin has functions for automatically
generating convex relaxations of general Pyomo models. Coramin also
has tools for domain reduction, including a parallel implementation
of optimization-based bounds tightening (OBBT) and various OBBT
filtering techniques.

## Primary Contributors
### [Michael Bynum](https://github.com/michaelbynum)
- Relaxation classes
- OBBT
- OBBT Filtering
- Factorable programming approach to generating relaxations

### [Carl Laird](https://github.com/carldlaird)
- Parallel OBBT
- McCormick and piecewise McCormick relaxations for bilinear terms
- Relaxations for univariate convex/concave functions

### [Anya Castillo](https://github.com/anyacastillo)
- Relaxation classes

### [Francesco Ceccon](https://github.com/fracek)
- Alpha-BB relaxation

## Relevant Packages

### [Suspect](https://github.com/cog-imperial/suspect)
Use of Coramin can be improved significantly by also utilizing
Suspect's convexity detection and feasibility-based bounds tightening
features. Future development of Coramin will directly use Suspect in
Coramin's factorable programming approach to generating relaxations.

