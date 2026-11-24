We are pleased to announce the release of Pyomo 6.11.0.

Pyomo is a collection of Python software packages that supports a
diverse set of optimization capabilities for formulating and analyzing
optimization models.

The following are highlights of the 6.11 release series:

- Add support for Python 3.15
- Updates to support new solver versions including XPRESS 10.0,
  Gurobi 14.0, and HiGHS 1.16
- Significant improvement in expression system performance including
  lower memory usage and faster model construction
- New solver interfaces:
  - `xpress` direct and persistent v2 solver interfaces
  - v2: Ongoing solver interface refactor
- ...and of course numerous minor bug fixes and performance enhancements

This release introduces two backwards incompatible changes in default
behaviors in order to make Pyomo more conservative and secure:

- **The current working directory is no longer searched by default when
  locating solver executables and shared libraries.** If you rely on a
  solver or library located in your working directory, either specify
  it with an explicit relative or absolute path (e.g.,
  `SolverFactory('ipopt', executable='./ipopt')` or
  `ExternalFunction(library='./mylib.so', ...)`), or add the directory
  to your `PATH` / `LD_LIBRARY_PATH`.
- **`ComponentUID` no longer uses `pickle` automatically.** Models that
  use custom (non-builtin) types in component indices and then build
  `ComponentUID`s will need to either re-enable pickle support or
  register a serializer/deserializer for their type.

A full list of updates and changes is available in the
[`CHANGELOG.md`](https://github.com/Pyomo/pyomo/blob/main/CHANGELOG.md).

Enjoy!

 - Pyomo Developer Team
 - pyomo-developers@googlegroups.com
 - https://www.pyomo.org


About Pyomo
-----------

The Pyomo home page provides resources for Pyomo users:

 * https://www.pyomo.org

Detailed documentation is hosted on Read the Docs:

 * https://pyomo.readthedocs.io/en/stable/

Pyomo development is hosted at GitHub:

 * https://github.com/Pyomo

Get help at:

 * StackOverflow: https://stackoverflow.com/questions/tagged/pyomo
 * Pyomo Forum:   https://groups.google.com/group/pyomo-forum/
