Pyomo CHANGELOG
===============


-------------------------------------------------------------------------------
Pyomo 6.6.1    (30 May 2023)
-------------------------------------------------------------------------------

- General
  - Update cmake builder for recent setuptools (#2847)
  - Fixing minor formatting for 6.6.0 release changes (#2842)
  - Silence deprecation warnings (#2854)
- Core
  - Update indentation handling in `config.StringFormatter` (#2853)
  - Restore slice API broken by #2829 (#2849)
  - Resolve handling of {}**0 in `LinearRepn`/`QuadraticRepn` (#2857)
- Solver Interfaces
  - NL writer: resolve error identifying vars in indexed SOS (#2852)
  - Manage Gurobi environments in GurobiDirect (#2680)
- Contributed Packages
  - cp: fix handling fixed BooleanVars in logical-to-disjunctive walker (#2850)
  - FBBT: Fix typo when handling GeneralExpression objects (#2848)
  - MindtPy: add support for cyipopt (#2830)

-------------------------------------------------------------------------------
Pyomo 6.6.0    (24 May 2023)
-------------------------------------------------------------------------------

- General
  - Remove `pyomo check`/`pyomo.checker` module (#2753)
  - Improve formatting of docstrings generated from `ConfigDict` (#2754)
  - Deprecate `add_docstring_list` (#2755)
  - Reapply `black` to previously completed directories (#2775)
  - Improve formatting for `DeveloperError`, `MouseTrap` messages (#2805)
- Core
  - Bugfix: component indexes specified as lists (#2765)
  - Remove the use of weakrefs in `SymbolMap` (#2791)
  - Improve conversions between Pyomo and Sympy expressions (#2806)
  - Rework expression generation to leverage multiple dispatch (#2722)
  - Improve robustness of `calculate_variable_from_constraint()` (#2812)
  - Add support for infix Boolean logical operators (#2835)
  - Improvements to Pyomo component iteration (#2829)
- Documentation
  - Copyright and Book Updates (#2750)
  - Link documentation in incidence_analysis README (#2759)
  - Update ReadtheDocs Configuration (#2780)
  - Correct import in community.rst (#2792)
  - Remove instructions for python <= 3.0 (#2822)
- Solvers Interfaces
  - NEOS: fix typo in `kestrelAMPL.kill()` argument (#2758)
  - Better handling of mutable parameters in HiGHS interface (#2763)
  - Improve linear data structure in NL writer (#2769)
  - Bugfix for shared named expressions in NL writer (#2790)
  - Resolve NPV constants in `LinearExpressions` in NL writer (#2811)
  - GAMS/Baron: ensure negative numbers are parenthesized (#2833)
  - Release LP version 2 (LPv2) writer (#2823, #2840)
- Testing
  - Rework Upload of Coverage Reports (#2761)
  - Update constant for visitor tests for python 3.11.2 (#2799)
  - Auto-Linting: Spelling Black Style Checker (#2800, #2818)
  - Skip MOSEK tests on NEOS (due to unknown NEOS error) (#2839)
- GDP
  - Add `gdp.bound_pretransformation` (#2824)
- Contributed Packages
  - APPSI: Improve logging consistency across solvers (#2787)
  - APPSI: Update `available` method in APPSI-Gurobi interface (#2828)
  - DoE: Release version 2 (#2794)
  - incidence_analysis: Remove strict usage of PyomoNLP (#2752)
  - incidence_analysis: Test `IndexedBlock` (#2789)
  - incidence_analysis: Use standard repn for incidence graph generation (#2834)
  - Parmest: Update for pandas 2.0.0 release (#2795)
  - piecewise: Add contrib.piecewise package (#2708, #2768, #2766, #2797, #2798,
    #2826)
  - PyNumero: Refactor CyIpopt interface to subclass `cyipopt.Problem` (#2760)
  - PyNumero: Fix CyIpopt interface when `load_solutions=False` (#2820)
  - PyROS: Fixes to PyROS Separation Routine (#2815)
  - PyROS: Fixes to Coefficient Matching and Timing Functionalities (#2837)

-------------------------------------------------------------------------------
Pyomo 6.5.0    (16 Feb 2023)
-------------------------------------------------------------------------------

- General
  - Apply `black` to enforce PEP8 standards in certain modules (#2737, #2738,
    #2733, #2732, #2731, #2728, #2730, #2729, #2720, #2721, #2719, #2718)
  - Add Developers' call information to README (#2665)
  - Deprecate `pyomo.checker` module (#2734)
  - Warn when infeasibility tools will not log output (#2666)
  - Separate identification from logging in `pyomo.util.infeasible.log_*` (#2669)
  - Convert subprocess timeout parameters to module attributes (#2672)
  - Resolve consistency issues in the Bunch class (#2685)
  - Remove GSL downloader from `download-extensions` (#2725)
  - Update enhancement GitHub issue template to link to wiki (#2739)
  - Add deprecation warning to `pyomo` command (#2740)
  - Require `version=` for all deprecation utilities (#2744)
  - Fix `pyomo --version` version string (#2743)
- Core
  - Fix minor typo in set.py (#2679)
  - Fix bugs in scaling transformation (#2678)
  - Rework handling of 'dimensionless' units in Pyomo (#2691)
- Solver Interfaces
  - Switch default NL writer to nlv2 and bug fixes (#2676, #2710, #2726)
  - Enable MOSEK10 warm-start flag and relocate var initialization (#2647)
  - Fix handling of POW in Baron writer (#2693)
  - Update GAMS license check to avoid exception when not available (#2697)
- Documentation
  - Fix incorrect documentation for sending options to solvers (#2688)
  - Fix Sphinx warnings (#2712)
  - Document Python Version Support policy (#2735)
  - Document deprecation and removal of functionality (#2741)
  - Document docstring formatting requirements (#2742)
- Testing
  - Skip failing Baron tests (#2694)
  - Remove residual `nose` references (#2736)
  - Update GHA setup-python version (#2705)
  - Improve GHA conda setup performance (#2701)
  - Add unit test for QCQO problems with MOSEK (#2682)
- DAE
  - Fix typo in `__init__.py` (#2683)
  - Add `active` filter to flattener (#2643)
- GDP
  - Add GDP-to-MIP transformation base class (#2687)
- Contributed Packages
  - DoE: New module for model-based design of experiments (#2294, #2711, #2527)
  - FBBT: Add tolerances to tests (#2675)
  - GDPopt: Switch a LBB test to use Gurobi as MINLP solver (#2686)
  - incidence_analysis: Add `plot` method to `IncidenceGraphInterface` (#2716)
  - incidence_analysis: Refactor to cache a graph instead of a matrix (#2715)
  - incidence_analysis: Add documentation and update API (#2727, #2745)
  - incidence_analysis: Add logging solve_strongly_connected_components (#2723)
  - MindtPy: Refactor to improve extensibility and maintainability (#2654)
  - Parmest: Suppress mpi-sppy output in import (#2692)
  - PyNumero: Add tee argument to Pyomo-SciPy square solvers (#2668)
  - PyNumero: Support implicit function solvers in ExternalPyomoModel (#2652)
  - PyROS: Fix user_time and wallclock_time bug (#2670)
  - PyROS: More judicious enforcement of PyROS Solver time limit (#2660, #2706)
  - PyROS: Update documentation (#2698, #2707)
  - PyROS: Adjust routine for loading DR polishing model solutions (#2700)
  - Viewer: Update to support PySide6 and display units and domain (#2689)

-------------------------------------------------------------------------------
Pyomo 6.4.4    (9 Dec 2022)
-------------------------------------------------------------------------------

- General
  - Convert `txt` to `md` files (`CHANGELOG`, `LICENSE`, `RELEASE`) (#2635)
  - Parallelize build of manylinux wheels (#2636)
  - Update source for Jenkins status badge (#2639, #2640)
  - Update relocated_module_attribute to work with cythonized modules (#2644)
  - Add utility methods to HierarchicalTimer (#2651)
- Core
  - Fix preservation of stale flags through clone/pickle (#2633)
  - Add support for local suffixes in scaling transformation (#2619)
- Solver Interfaces
  - Fix handling of nonconvex MIQCP problems in Xpress (#2625)
- Testing
  - Update GitHub actions to cancel jobs when new changes are pushed (#2634)
  - Remove requirement for a `pyutilib` directory in Jenkins driver (#2637)
  - Enable GitHub actions build on Windows Python 3.11 (#2638)
  - Add build services infrastructure status badge (#2646)
  - Add version upper bound on MOSEK warmstart test skip (#2649)
  - Improve compare.py handling of nosetests/pytest output (#2661)
- GDP
  - Add option to use multiple-bigm only on bound constraints (#2624)
  - Add logical_to_disjunctive and replace uses of logical_to_linear (#2627)
- Contributed Packages
  - FBBT: Fix bug with ExternalFunction expressions (#2657)
  - PyROS: Fix uncertain param bounds evaluation for FactorModelSet (#2620)
  - PyROS: Add origin attribute to BudgetSet (#2645)
  - PyROS: Fix UncertaintySet.bounding_problem method (#2659)

-------------------------------------------------------------------------------
Pyomo 6.4.3    (28 Nov 2022)
-------------------------------------------------------------------------------

- General
  - Update PauseGC to work in nested contexts (#2507)
  - Simplify deepcopy/pickle logic to speed up model clone (#2510)
  - Fix generate_standard_repn to handle unexpected NPV expressions (#2511)
  - Add thread safe proxies for PauseGC, TempFileManager singletons (#2514)
  - Fix ConstructionTimer bug for components indexed by nonfinite sets (#2518)
  - Add calculate_variable_from_constraint differentiation mode option (#2549)
  - Update URL for downloading GSL and GJH (#2556, #2588)
  - Update logic for retrying failed downloads (#2569)
  - Add support and testing for Python 3.11 (#2596, #2618)
  - Update deprecation utilities to improve user messages (#2606)
- Core
  - Refactor expression hierarchy, add RelationalExpression base class (#2499)
  - Support cloning individual blocks (#2504)
  - Block performance improvements (#2508)
  - Add support for creating a slice to a single ComponentData object (#2509)
  - Fix missing import of value in pyomo.core.base.external (#2525)
  - Improve handling of restricted words on Blocks (#2535)
  - Improve Reference() performance (#2537)
  - Fix mapping gradient/hessian for external functions with string args (#2539)
  - Fix bug for sum_product(Var, Param, Param) (#2551)
  - Add deprecation path for expression objects moved to relational_expr (#2554)
  - Exception when setting value of Expression to non-numeric expression (#2567)
  - Improve deepcopy performance (#2628)
- Documentation
  - Fix examples in working_models.rst (#2502)
- Solver Interfaces
  - Improve SCIP results object (#2462)
  - Improve warning message when LP writer raises KeyError (#2497)
  - Fix Gurobi work limit bug (#2530)
  - Updates and fixes for the NLv2 writer (#2540, #2622, #2568)
  - Fix Xpress when stopped due to MAXTIME or MAXNODES (#2553)
  - Add support for MOSEK 10 affine conic constraints (#2557)
  - Fix missing explicit space in GAMS put command (#2578)
  - Fix GAMS logfile storage location (#2580)
  - LP writer performance improvements (#2583, #2585)
  - Update handling of MOSEK Env and Python module (#2591)
  - Release MOSEK license when optimize raises a mosek.Error (#2593)
  - Update list of allowable characters in CPLEX filenames (#2597)
- Testing
  - Update performance driver to be usable outside of Pyomo (#2505)
  - Update the performance test driver (#2538)
  - Reduce amount of environment code cached in GitHub actions (#2565)
  - Update GitHub actions versions from v2 to v3 (#2566)
  - Allow nan to compare equal in assertStructuredAlmostEqual() (#2582)
  - Add test utilities for comparing expressions (#2590)
  - Skip a test in MOSEK 10 due to a bug in warm starting MIQPs (#2614)
  - Update skipped CBC test that works with CBC 2.10.6 (#2615)
  - Add SCIP to GitHub actions environment (#2602)
- GDP
  - Use OrderedSet instead of list in GDPTree to improve performance (#2516)
  - Reduce calls to logical_to_linear in GDP transformations (#2519)
  - Add utility for gathering BigM values after transformation (#2520)
  - Add tighter logical constraints in transformations of nested GDPs (#2550)
  - Fix pickling of transformed GDP models (#2576)
  - Add multiple-bigM transformation (#2592)
  - Improve performance of BigM transformation (#2605)
  - Remove weakref mapping Disjunctions to their algebraic_constraint (#2617)
- Contributed Packages
  - APPSI: Fix exception raised by appsi_gurobi during Python shutdown (#2498)
  - APPSI: Improve handling of Gurobi results (#2517)
  - APPSI: Add interface to HiGHS solver (#2561)
  - APPSI: Only release Gurobi license after deleting all instances (#2599)
  - APPSI: Patch IntEnum to preserve pre-3.11 formatting (#2607)
  - CP: New package for constraint programming (#2570, #2612)
  - GDPopt: Add warning when reporting results from LBB (#2534)
  - GDPopt: Delete dummy objective when we're done using it (#2552)
  - GDPopt: Add enumerate solution approach (#2559, #2575)
  - IIS: Add package for computing the IIS of an infeasible Pyomo model (#2512)
  - MindtPy: Fix bug in termination condition (#2587)
  - MindtPy: Fix bug in checking absolute and relative gap (#2608)
  - MPC: Data structures/utils for rolling horizon dynamic optimization (#2477)
  - Parmest: Solve square problem to initialize regression problem (#2438)
  - Parmest: Return ContinuousSet values from theta_est() (#2464)
  - PyNumero: Fix NumPy deprecation warnings (#2521)
  - PyNumero: Add interfaces to SciPy square solvers (#2523)
  - PyNumero: Check AmplInterface availability in SciPy solver tests (#2594)
  - PyNumero: Add ProjectedExtendedNLP class (#2601)
  - PyNumero: Add interface to SciPy scalar Newton solver (#2603)
  - PyROS: Rewrite UncertaintySet docstrings/improve validation (#2488)
  - PyROS: Updates to subproblem initialization and solver call routines (#2515)
  - PyROS: Fix collection of sub-solver solve times (#2543)

-------------------------------------------------------------------------------
Pyomo 6.4.2    (17 Aug 2022)
-------------------------------------------------------------------------------

- General
  - Resolve dill incompatibility with attempt_import (#2419)
  - Speed up book tests and improve import/logging infrastructure (#2449)
  - Make typing overloads available at runtime (#2471)
  - Update list of known TPLs to reflect Python 3.10 (#2478)
  - Add NL writer version 2 but don't activate by default (#2473)
  - Add deprecation.relocated_module utility (#2492)
- Core
  - Support cython functions through Initializer (#2421)
  - Ensure UnindexedComponent_set is correctly pickled/restored (#2416)
  - Update as_numeric() to raise an exception on non-is_numeric_type() Pyomo
    objects (#2444)
  - replace_expressions: support replacing variables with constants (#2410)
  - Update component_data_objects and component_data_iterindex to not
    return duplicate data (#2456)
  - Fix exception handling logic bug in quicksum (#2458)
  - Reduce recursion in model clone/deepcopy (#2487)
  - Resolve exception cloning empty LinearExpression objects (#2489)
  - Resolve errors from inplace operators on Expression objects (#2493)
- Documentation
  - Fix typos and incorrect link (#2420, #2482)
  - Fix Sphinx warnings in documentation builds (#2434)
- Solver Interfaces
  - Add support for SCIP 8.0 (#2409)
  - Add explicit space to GAMS put statements to support GAMS 39+ (#2450)
  - Catch exceptions in BARON interface when parsing objective bounds (#2467)
- Testing
  - Skip SCIP 8.0 tests on PyPy (#2424)
  - Force conda to use conda-forge channel in GHA (#2430)
  - Fix GHA win/3.10 Python version inconsistency (#2433)
  - Enable Pyomo.DAE Simulator tests on PyPy (#2443)
  - Abort the Jenkins build if virtualenv creation fails (#2451)
  - Remove scipy from GHA PyPy builds (#2474)
  - Move performance testing driver scripts to scripts/performance (#2476)
  - Improve common.timing test robustness (#2490)
- Contributed Packages
  - APPSI: Improve error message in nl writer when all variables fixed (#2407)
  - APPSI: Fix bug in Gurobi interface when modifying the objective (#2454)
  - APPSI: Solve blocks that use variables that are not on the block (#2453)
  - APPSI: Method to release Gurobi license from persistent interface (#2480)
  - community_detection: Trap exceptions from networkx (#2415)
  - FBBT: Add tolerances to tests (#2445)
  - GDPopt: Refactor to improve extensibility and maintainability (#2406)
  - incidence_analysis: Add method for weakly connected components (#2418)
  - MindtPy: Fix bug in copying variable values between models (#2432)
  - MindtPy: Add support for appsi_cplex and appsi_gurobi (#2373)
  - Parmest: Add utils submodule and support to convert Params to Vars (#2352)
  - Parmest: Fix use of relocated_module_attribute (#2472)
  - PyROS: Fixes to ellipsoidal sets (#2425)
  - PyROS: Account for variable domains (#2426)
  - PyROS: Fix objective validation (#2371)
  - PyROS: Avoid master feasibility problem unit consistency checks (#2459)
  - PyROS: Fix discrete uncertainty set separation (#2469)
  - PyROS: Update exception handling parsing BARON lower/upper bounds (#2486)

-------------------------------------------------------------------------------
Pyomo 6.4.1    (13 May 2022)
-------------------------------------------------------------------------------

- General
  - Extend linux distribution map logic (#2361)
  - Improve string processing in ConfigList, ConfigDict, and ListOf (#2360)
  - Update copyright assertion (#2368, #2378)
  - Deprecate name_buffer argument to getname (#2370)
  - Defer construction of the main Pyomo parser (#2385)
  - Improve error checking parsing ConfigDict string value (#2399)
- Core
  - Add indices to ComponentData objects (#2351)
  - Ignore SetOf, BuildAction, and BuildCheck when checking units (#2366)
  - Improve support for absolute value in differentiation and FBBT (#2347)
  - Allow relative tolerance when comparing unit dimensionality (#2395)
- Solver Interfaces
  - Fix bug in GLPK solver plugin (#2348)
  - Update BARON solution parser (#2367)
- Testing
  - Turn on failure for codecov-action (#2343)
  - Fixes to GHA due to updates to Ubuntu runner and Conda (#2356, #2384, #2396)
  - Update setup-python and pypy in GHA (#2364)
  - Pin version of openmpi used for testing (#2369)
- DAE Updates
  - Fix typo preventing a DAE test from running (#2349)
- Contributed Packages
  - APPSI: Minor improvements and generalizations (#2383, #2386, #2389,
    #2391, #2388, #2392)
  - incidence_analysis: Add option to ignore inequality constraints in
    IncidenceGraphInterface (#2350)
  - MC++: Update interface tests (#2400)
  - MindtPy: Add quadratic_strategy option to pass quadratic constraints
    or objectives to MIP solvers (#2338)
  - Parmest: Skip tests when seaborn and matplotlib are missing (#2394)
  - PyROS: Improve feasibility problem formulation, objective validation
    (#2353, #2371)

-------------------------------------------------------------------------------
Pyomo 6.4.0    (16 Mar 2022)
-------------------------------------------------------------------------------

- General
  - Distribute the ampl_function_demo C source / cmake files (#2319)
  - Add GitHub URL to setup.py for PyPI (#2325)
  - Use constant indentation for long lines in the HierarchicalTimer (#2336)
- Core
  - Remove incomplete pyomo/core/base/alias.py module (#2284)
  - Remove Python 3.6 support (#2312)
- Solver Interfaces
  - Remove PICO solver interface (#2341)
- Testing
  - Add Windows pip single test (#2334)
  - Fix test docstrings in piecewise to silence warnings (#2342)
- Contributed Packages
  - APPSI: Update Gurobi solver unavailable message (#2320)
  - APPSI: Remove distributables for Windows (#2326)
  - APPSI: Fix bugs when working with the Gurobi solution pool (#2335)
  - APPSI: Fix bug in persistent update for FBBT (#2340)
  - Preprocessing: Stop using polynomial_degree to check for linear and
    constant expressions (#2324)
  - PyNumero: Improve coverage of mpi block matrix tests (#2318)
  - PyNumero: Skip bound/domain validation in ExternalPyomoModel (#2323)
  - PyNumero: Remove deprecated usage of numpy.bool (#2339)
  - PyROS: Fix variable default initialization (#2331)

-------------------------------------------------------------------------------
Pyomo 6.3.0    (23 Feb 2022)
-------------------------------------------------------------------------------

ADVANCE CHANGE NOTICE
  - This will be the last release to support Python 3.6.

- General
  - Construct slices with normalized indices in slicing utility (#2223)
  - Remove hard-coded project name from default attempt_import() message (#2237)
  - Add --with-distributable-extensions option to setup.py (#2260)
  - Update Pyomo versioning to always include the micro version number (#2265)
  - Remove optional_packages from pyomo.environ (#2195)
  - Add Python 3.10 as an officially-supported interpreter (#2302)
  - `TempfileManager` updates (deletion order, add context) (#2297)
  - Add `report_scaling` utility to detect potential scaling issues (#2252)
- Core
  - Check Var bounds when setting values (#2196)
  - Fix bug in implicit Var construction (#2210)
  - Add support for user-defined Component List starting index (#2215)
  - Switch Var.value setter to validate values (#2214)
  - Overload component initializers with named keyword params (#2212, #2230)
  - Resolve errors when creating Reference to a scalar Set (#2229)
  - Standardize scalar-valued external function interfaces (#2231)
  - Resolve bugs in cloning _ImplicitAny domains (#2233)
  - Redesign Var.stale handling to use a global state flag (#2249)
  - Update differentiate() to accept string mode identifiers (#2266)
  - Add `as_quantity`: evaluate a Pyomo expression to a pint `Quantity` (#2222)
  - NPV_SumExpression to inherit from NPV_Mixin (#2209)
  - Add support for `abs` in numeric differentiation (#2232)
  - Search filesystem for AMPLExternalFunction library (#2305)
- Documentation
  - Updates to installation documentation (#2259)
  - Update GDP references (#2300)
  - Add example AMPL external function library (#2295)
- Solver Interfaces
  - Disable log file in gurobi_direct when keepfiles=False (#2203)
  - Remove old GLPK interfaces (#2256)
  - Support gurobi lp interface through gurobipy module (#2262)
  - Update CBC version identification (#2250)
  - Increase CBC timeout when getting version/asl compatibility (#2293)
  - Deprecate the `Alias` component (#2288)
  - Remove XPRESS interface (#2273)
  - Support string arguments in NL external functions (#2295)
  - Fix reversed NL operator codes for floor/ceil (#2216)
- Testing
  - Skip some fileutils tests on OSX Big Sur and Python<3.8 (#2224)
  - Fix Windows/Python 3.6 testing dependency (#2234)
  - Update test suite for BARON 22.1.19 (#2268)
  - Rework coverage uploads to use GitHub Actions (#2225)
  - Add option to test driver for turning off log capture (#2198)
  - Relaxing timing target for Windows GHA (#2303)
  - Resolve GDP test failure for missing solvers (#2306)
  - Testing infrastructure Refactor: `nosetests` to `pytest` (#2298)
  - Make time limit more robust for APPSI (#2296)
  - Resolve failing floating-point comparison failures in FBBT (#2244)
  - Check gurobipy availability in doctests (#2253)
- GDP Updates
  - Ensure gdp.fix_disjuncts always creates a fully algebraic model (#2263)
  - Add `partition_disjuncts` and `between_steps` transformations (#2221)
- Contributed Packages
  - APPSI: Fix bug with CBC options (#2243)
  - APPSI: Correctly identify changes to constraints (#2299)
  - APPSI: Improvements to persistent interface (#2246)
  - APPSI: Implement FBBT in C++ module (#2248)
  - APPSI: Resolve build errors on Windows (#2309)
  - GDPopt: Fix bugs in preprocessing (#2211)
  - GDPopt: Switch preprocessing to use FBBT (#2264)
  - incidence_analysis: General improvements (#2239, #2240)
  - MindtPy: Update online docs and logging (#2219, #2207)
  - MindtPy: Add Primal/Dual Integral (#2285)
  - MindtPy: Nonlinear sum reformulation of objective of MINLP (#2276)
  - Parmest: Covariance matrix (#2287), Add examples (#2274)
  - PyNumero: Add scaling factor support to ExternalPyomoModel (#2269)
  - PyNumero: Use projected NLPs for ExternalPyomoModel inner problems (#2283)
  - PyNumero: Add SQP example (#2291)
  - PyROS: Support ConstraintList and range constraints (#2206)
  - PyROS: Add optional bypassing of global separation subproblems (#2254)
  - TrustRegion: New implementation of Trust Region Framework (#2238, #2279)

-------------------------------------------------------------------------------
Pyomo 6.2    (17 Nov 2021)
-------------------------------------------------------------------------------

- General
  - Add ListOf domain validator for ConfigValue (#2103)
  - Remove deprecated calls to warnings.warning (#2106)
  - Add Bool and Integer domain validators (#2110)
  - Overhaul of Pyomo TempfileManager (#2109)
  - Allow passing tuples and expressions to
    calculate_variable_from_constraint() (#2133)
  - Add external functions to block in create_subsystem_block (#2142)
  - Track changes in pint 0.18 and Xpress 8.13.0 (#2168)
  - Remove code deprecated in Pyomo version 5.6 or older (#1992)
  - Add Linux aarch64 wheel build support (#2076)
  - Update wheel creation GitHub Actions version (#2192)
  - Remove optional_packages from pyomo.environ (#2195)
  - Ignore bounds/domain when updating variable value in
    calculate_variable_from_constraint() (#2177)
- Core
  - Fix Expression.getitem when initialized with Skip (#2136)
  - Support hierarchical models in logical_to_linear transformation (#2135)
  - Add ordered=True API for iterating over IndexedComponents (#2149)
  - Fix bug in differentiate module related to reused subexpressions (#2148)
  - Rework expression replacement and LinearExpression object (#2143)
  - Add support for custom pint registries (#2153)
  - Add support for solving blocks and targets in logical_to_linear
    transformation (#2147)
  - Add support for finding components with a ComponentUID (#2178)
  - Fix bugs with open NumericRanges (#2170, #2179)
  - Fix bugs with References (#2158)
  - Fix Initializer by treating pandas.Series as sequences (#2151)
  - Fix Initializer support for ConfigList/ConfigDict (#2200)
  - Add a DataFrameInitializer (#2150)
  - Add a public API for retrieving variable bound expressions (#2172)
  - Rework Var component to leverage Initializer (#2184)
- Documentation
  - Expand documentation of pyomo.common (#2104)
  - Minor documentation fixes (#2117)
- Solver Interfaces
  - Issue warning when using the Xpress LP/MPS interface (#2125)
  - Disable implicit repn cache for LP and NL writers (#2144)
  - Add option to force variables into an NL file (#2152)
  - Add the exception to the NEOS connection failure warning message (#2166)
  - Improve GAMS writer performance (#2191)
- Testing
  - Resolve test failures when no solvers are available (#2146)
  - Resolve NumericValue support in assertStructuredAlmostEqual (#2200)
  - Fix typo in booktest skip declaration (#2186)
- DAE Updates
  - Utility function to slice individual components in flatten module (#2141)
  - Fix errors and deprecation warnings in some DAE examples (#2189)
  - Fix bug allowing numpy.float64 types to end up in a ContinuousSet (#2193)
- GDP Updates
  - Update GDP transformations to handle logical constraints (#2175)
- Contributed Packages
  - APPSI: Add support for external functions in NL writer (#2111)
  - APPSI: Improved handling of immutable parameters (#2112)
  - PyNumero: Pass AMPLFUNC directly to pynumero_ASL and not through
    environment (#2114, #2129)
  - PyNumero: ExternalPyomoModel performance improvements (#2155)
  - PyNumero: Remove exception when variable bounds are equal (#2121)
  - Parmest: Add support for indexed variables (#2108)
  - incidence_analysis: Simplify extraction of diagonal blocks from a block
    triangularization (#2128)
  - GDPopt: Make line endings PEP8 compliant (#2173)
  - MindtPy: Add support for MIP solver solution pools (#2164)
  - MindtPy: Skip specific tests due to failure with Gurobi 9.5.0 on
    Windows (#2183)
  - MindtPy: Fix feasibility pump and fixed variable detection (#2185, #2187)
  - PyROS: Add uncertain variable bounds detection (#2159)

-------------------------------------------------------------------------------
Pyomo 6.1.2  (20 Aug 2021)
-------------------------------------------------------------------------------

- General
  - Add test for missing/extra __init__.py files (#2090)
  - Update deprecation warnings to only be emitted once per context (#2095)
- Core
  - Replace Set.card() with Set.at() (#2096)
- DAE
  - Remove use of OrderedSet.__getitem__ from contset.py (#2091)
- Contributed Packages
  - APPSI: Build/install APPSI extension outside of source tree (#2092)
  - MindtPy: Support gurobi_persistent in LP/NLP-based B&B algorithm (#2071)

-------------------------------------------------------------------------------
Pyomo 6.1.1  (17 Aug 2021)
-------------------------------------------------------------------------------

- General
  - Adding missing __init__.py files across Pyomo (#2086)

-------------------------------------------------------------------------------
Pyomo 6.1    (17 Aug 2021)
-------------------------------------------------------------------------------

- General
  - Support using TicTocTimer as a context manager (#2012)
  - Add optional dependency target in setup.py (#2017)
  - Added utilities for subsets of model components (#1998)
  - Improvements in deprecation warning system (#2045, #2058, #2056)
  - Resolve TeeStream deadlock issues (#2072)
  - Move PYOMO_CONFIG_DIR into pyomo.common.envvar (#2065)
  - Add a DynamicImplicitDomain domain validator (#2043)
  - Generate standard repn misidentifies nonlinear expr (#2074)
  - Add Module() validator to the config system (#2062)
- Core
  - Yield Iterator from IndexedComponent (#2007)
  - Use yield from in pyomo.core (#1984)
  - Improvements to sorted_robust and string formatting (#2020, #2024)
  - Integrating numpy with the Pyomo expression system (#2027, #2034, #2070)
  - Resolve Set bugs (#2048)
  - Prevent attaching global Sets to Blocks (#2042)
  - Correct scalar mutable param getitem implementation (#2066)
  - Fix Var bounds with unitted mutable Params (#2067)
  - Improve @disable_methods decorator (#2069)
  - Allow native numeric types in relational expressions (#2073)
  - Fixing symbol names in Initializer deprecation layer (#2079)
  - Deprecate OrderedSet.__getitem__ in favor of OrderedSet.card() (#2053)
- Documentation
  - Create a list of related packages (#2016)
- Solver Interfaces
  - Clean up dependencies and solver.available() IO (#2011)
  - Update Baron writer to recognize priority Suffix (#2035)
  - Add OCTERACT to the list of known NEOS solvers (#2064)
  - Resolve tee issues with Xpress 8.9+ (#2059)
  - Increase timeout for ASL version request (#2083)
- Testing
  - Miscellaneous testing fixes (#2023)
  - Improve management of tempfiles in tests (#2036)
  - Update GHA to ubuntu-latest; resolve ampl/mp build error (#2015)
  - Move constrained_layout tests to 'expensive' suite (#2018)
  - Update xpress, appsi tests to manage CWD (#2031)
  - Improve comparison of Book examples to baselines (#2044)
  - Add retry/timeout to powershell downloads (#2077)
- GDP Updates
  - gdp.hull: Only create one disaggregated variable for all Disjuncts in
    which a Var does not appear (#2005)
- Contributed Packages
  - FBBT: descend_into option for FBBT (#2025)
  - FBBT: Interval arithmetic edge cases for FBBT (#2026)
  - FBBT: Improved handling for product expressions (#2038)
  - incidence_analysis: methods for generating and solving strongly
    connected components of a square model (#2009)
  - incidence_analysis: Add Dulmage-Mendelsohn partition (#2063)
  - PETSc: Remove PETSc solver (#1986)
  - PyROS: Pyomo Robust Optimization Solver (#2006, #2061)
  - sensitivity_toolbox: Add kaug dsdp mode (#1613)
  - APPSI: API Cleanup (#1944), bug-fix in update_variables (#2051)
  - MindtPy: Increase code coverage (#2021)
  - MindtPy: Replace is not by != (#2081)
  - PyNumero: ExternalGreyBox subclass to embed implicit functions (#2022)


-------------------------------------------------------------------------------
Pyomo 6.0.1   (4 Jun 2021)
-------------------------------------------------------------------------------

- General
  - Catch non-standard file.write() implementations (#1999)
- Testing
  - Fix miscellaneous tests caught by peripheral test suites (#1991)
- GDP Updates
  - Switch bigm's M value calculation to always use fbbt (#2000)
- Contributed Packages
  - FBBT: Add support for LinearExpression (#1994)
  - PyNumero: Improve management of stdout in cyipopt solver (#2001)
  - GDPopt: Fix implicit conversion warnings (#2002)

-------------------------------------------------------------------------------
Pyomo 6.0    (20 May 2021)
-------------------------------------------------------------------------------

BACKWARDS COMPATIBILITY WARNINGS
  - Moved PySP into a separate package available at
    https://github.com/Pyomo/pysp
  - Renamed "Simple" components to "Scalar" components
  - Removed support for implicit ranged inequalities
  - Removed support for component reconstruct method
  - Removed support for casting relational expressions to bool
  - Removed support for Python 2.7, 3.5, and pypy2

- Drop PyUtilib Dependency
  - Replace pyutilib.misc.import_file with pyomo.common.fileutils (#1815)
  - Replace pyutilib.math with Python math package (#1809)
  - Remove or replaced pyutilib.misc items (#1814, #1822)
  - Replace pyutilib.subprocess with Python subprocess (#1812)
  - Replace pyutilib.th with unittest and pyomo.common.unittest (#1859, #1898)
  - Remove pyutilib.pyro (#1869)
  - Remove pyutilib dependencies from Pyomo script (#1976)
  - Remove hard pyutilib dependency (#1980)

- General
  - Drop support and testing for Python 2.7, 3.5, and pypy2 (#1763, #1818)
  - Remove PySP (#1819)
  - Clean up code for Python 2/3 cross-compatibility, remove use of six
    (#1829, #1833, #1901, #1908, #1910, #1911, #1876, #1915, #1919,
    #1931, #1938, #1937)
  - Standardize usage of string type comparison (#1826)
  - Switch default branch of Pyomo from master to main (#1843)
  - Cleanup console scripts (#1830)
  - Update import_file to recognize module context (#1871)
  - Remove 2nd edition Pyomo Book examples (#1883)
  - Move Container/Options/Bunch into just Bunch (#1838)
  - Improve handling of enum.Enum domains in ConfigValue (#1893)
  - Add MANIFEST.in file to include cpp files in PyPI package (#1851)
  - Add GitHub issue templates for bug reports and feature requests (#1920)
  - Fix identification of Pyomo root dir in unittest.runtests (#1922)
  - Update the Config system (#1916)
  - Resolve numerous deprecation warnings (#1935)
  - Timing and timing test improvements (#1939, #1894, #1945)
  - Overhaul @deprecated decorator to work better with class, enum, and
    sphinx (#1906)
  - Improve attempt_import mechanism (#1940, #1948)
  - Stop caching the picklability of function types (#1959)
  - Support explicit constants in ExternalFunction arguments (#1967)
  - Add RenameClass metaclass (#1973)
  - Add 3rd edition Pyomo Book examples and tests (#1914)
  - Promote assertStructuredAlmostEqual to a module function (#1979)
- Core
  - Improve sorted_robust for (nested) tuple items (#1842)
  - Update _component_data_iter to use sorted_robust (#1852)
  - Disable implicit ranged inequalities (#1840)
  - Improve error message when constructing a RangeSet with external
    data (#1875)
  - Fix bug in scaling transformation (#1854)
  - Fix bug in identify_mutable_parameters (#1878)
  - Fix inequality function to work with floats (#1841)
  - Update test_model to skip tests if GLPK is not available (#1917)
  - Rename param._NotValid to param.Param.NoValue (#1927)
  - Improve error message for disabled methods on Abstract components (#1896)
  - Propagate exception flag to _EvaluationVisitor (#1946)
  - Define pprint() for ExternalFunction components (#1965)
  - Support setting Var and Param values with unit expressions (#1966)
  - Allow assigning None to a unitted variable (#1977)
  - Update to only allow constant expressions to be cast to bool (#1954)
  - Rename "Simple" components to "Scalar" components (#1971)
  - Add utility for comparing expressions (#1981)
  - Improve quoting in component names and indices (#1982, #1985)
  - Remove component.reconstruct() (#1983)
- Documentation
  - Add documentation and doctests for pyomo.common.timing (#1924)
  - Update the ReadTheDocs configuration (#1929)
  - Improve robustness of doctest tests (#1942)
- Solver Interfaces
  - Track changes in NEOS/RAPOSa expected behavior (#1816)
  - Fix quadratic term handling for QCQO problems in mosek_direct (#1835)
  - Remove pyomo.opt.blackbox and the COLIN solver interface (#1872)
  - Add support for SHOT solver to GAMS interface (#1889)
  - Fix sign convention in CBC for maximization problems (#1848)
  - Update gurobi available() checks (#1921)
  - Update CBC interface to handle infinite gap (#1907)
  - Preserve CWD when xpress import fails (#1958)
  - Update baron interface to prevent infinite loop when no solution is
    found (#1963)
  - Remove pyomo.solvers.tests.core (#1897)
- Testing
  - Clean up Jenkins test driver (#1813)
  - Update  unittest.timeout to work on Windows (#1825)
  - Manually add commit HASH to codecov upload on GitHub Actions (#1824)
  - Relax tee/timeout test timing to improve robustness (#1831)
  - Update Mosek tests (#1844)
  - Add Casadi to GitHub Actions test environment (#1849)
  - Disable codecov uploads from forks (#1863)
  - Update IDAES-ext version used in our GitHub Actions jobs workflows (#1882)
  - Rewrite test driver script (#1870, #1902)
  - Rebuild GitHub Actions caches (#1934)
  - Update expected CBC test failures (#1943)
  - Update merged reader test to reduce failures in OSX (#1953)
  - Add non-fatal timeout for nightly NEOS test (#1964)
  - Do not download/build extensions for slim builds (#1988)
- DAE Updates
  - Split get_index_set_except into two functions (#1879)
- GDP Updates
  - Fix multiple References to local variables in nested disjuncts (#1968)
  - Fix GDP error when using a disjunction with nested disjunctions as a
    target (#1969)
  - Convert Disjunct.indicator_var from binary to boolean variable (#1960)
- Network Updates
  - Fix bug in cloning Ports (#1899)
  - Updates to improve interoperability with pyomo.gdp (#1961)
- Contributed Packages
  - PyNumero: add Hessian support for grey box models, support multiple
    grey box models in a single Pyomo model, add callback functionality
    to cyipopt interface, track changes in cyipopt API, expose nl file
    options in PyomoNLP, add documentation and doctests (#1810, #1832,
    #1873, #1913, #1930, #1936)
  - Update contrib.viewer tests (#1820)
  - Add new Auto-Persistent Pyomo Solver Interfaces (APPSI) package (#1793)
  - GDPopt: add option to solve a relaxation with integer cuts, remove default
    strategy (#1837)
  - Benders: rename master problem to root problem (#1828)
  - FBBT: Improve interval arithmetic for some special cases (#1949)
  - Parmest: remove dependency on PySP and add interface to mpi-sppy (#1778)
  - Add new incidence_analysis package for maximum matching and block
    triangularization (#1888)
  - MindtPy: general improvements and add feasibility pump (#1847)

-------------------------------------------------------------------------------
Pyomo 5.7.3   (29 Jan 2021)
-------------------------------------------------------------------------------

ADVANCE CHANGE NOTICE:
  - This is the last Pyomo release to support Python 2.7, 3.4, and 3.5
  - This is the last Pyomo release to bundle PySP
  - This is the last Pyomo release that will have a strict PyUtilib dependency

- General
  - Improve GHA workflow for uploading manylinux wheels (#1758, #1757)
  - Add Python 3.9 to OSX/Windows on the GHA release workflow (#1755)
  - Standardize file line endings (#1764)
  - Resolve matplotlib deferred import failure (#1770)
  - Move the ATTIC directory to an internal archived repository (#1762)
  - Rename log_active_constraints to log_model_constraints (#1788)
  - Remove references to ordereddict package (#1796)
  - Convert old deprecation messages to use pyomo.common.deprecation (#1798)
  - Update and standardize Pyomo logging configuration (#1797)
  - Add writer consistency checks between filenames and formats (#1804)
  - Initialize self.tempdir in TempfileManagerClass (#1806)
- Core
  - Add is_reference API (#1740)
  - Add option to not rename components in the scaling transformation (#1596)
  - Improve units documentation and performance (#1769, #1794, #1805)
- Documentation
  - Move the doc/attic directory to an internal archived repository (#1773)
- Solver Interfaces
  - Fix missing imports in the persistent solver interface (#1772)
  - Update NEOS interface to provide email addresses (#1782, #1783, #1795)
  - Add mipgap capture to CPLEX direct interface (#1775)
  - Update solver license_is_valid checks (#1789)
  - Update CPLEX solution parser for CPLEX 12.10 (#1792)
  - Resolve issues with the NEOS interface (#1802)
- DAE Updates
  - Extend flattener to allow multiple sets (#1768)
- GDP Updates
  - Fix subproblem initialiations in gdp.cuttingplane (#1780)
- Testing
  - Fix Jenkins test script (#1771)
  - Fix GHA tests to work with PyMySQL 1.0.0 (#1777)
- Contributed Packages
  - PyNumero: resolve portability issue with Cyipopt exception handling,
    add PyomoNLP methods for accessing equality/inequality constraints (#1781,
    #1791)
  - Parmest: update pairwise plot to use the covariance matrix (#1774)

-------------------------------------------------------------------------------
Pyomo 5.7.2   (17 Dec 2020)
-------------------------------------------------------------------------------

- General
  - Increase timeout when loading libraries through CtypesEnviron (#1644)
  - Add support for deferred import of submodules (#1683, #1729)
  - Overhaul imports to replace 'import *' and remove unused imports (#1607)
  - Move pyutilib.misc Bunch/Container/Options imports to
    pyomo.common.collections (#1708, #1735)
  - Move pyutilib.common exceptions to pyomo.common.errors (#1694, #1733)
  - Move pyutilib.misc.timing to pyomo.common.timing (#1688)
  - Move pyutilib.misc.config into pyomo.common.config (#1687)
  - Move pyutilib.factory to pyomo.common.factory (#1695)
  - Move pyutilib.misc.gc_manager to pyomo.common.gc_manager (#1716)
  - Move pyutilib LogHandler to pyomo.common.log (#1717)
  - Move pyutilib.services.tempfiles to pyomo.common.tempfiles (#1742)
  - Make logger and ostream public attributes in the TicTocTimer (#1714)
  - Update management of matplotlib imports (#1738)
  - Add utility function to get slice of component along certain set (#1639)
  - Remove unused pyomo.solvers.plugins.testdriver directory (#1750)
- Core
  - Use name buffer in add_slack_variables transformation (#1637)
  - Make Constraint.Feasible be an alias for Constraint.Skip (#1676)
  - Updates to slicing infrastructure and documentation (#1638, #1736)
  - Add public API for checking Param mutability (#1651)
  - Update methods for differentiating Pyomo expressions (#1663)
  - Add skip_stale_vars option to store_to function (#1679)
  - Improve Block initialization (#1703)
  - Update ComponentUID to improve serialization and leverage slicing
    (#1652, #1713, #1748)
  - Extend Reference function to accept Sequence and Mapping derivatives (#1739)
  - Fix bugs in deepcopying/pickling units (#1743)
  - Fix cloning AMPLExternalFunction objects after external library is
    loaded (#1745)
  - Improve log message when loading solution with 'warning' status (#1752)
- Solver Interfaces
  - Fix typo in Gurobi direct interface (#1649)
  - Explicitly close log files for cplex_direct (#1655)
  - Deprecate use of base64.decodestring in Python 3 NEOS interface (#1661)
  - Improvements to MOSEK direct interface, new persistent interface,
    add license check to tests, resolve incompatibilities with
    Python 2.7 (#1686, #1746)
- DAE Updates
  - Improve performance of discretization transformations (#1727)
  - Allow flattener to find constraints that skip the first index (#1720)
- GDP Updates
  - Rewrite of gdp.cuttingplane transformation (#1668)
  - Fix handling of local variables in gdp.hull
- PySP Updates
  - Defer import of profiling packages and guppy module (#1677, #1697)
  - Fix integer conversion of string arguments in ProgressiveHedging (#1684)
- Testing
  - Update Jenkins driver to resolve codecov report upload (#1670)
  - Update slim and cythonized builds in GitHub Actions and Jenkins
    drivers (#1674)
  - Fix security risk in GitHub Actions workflow (#1654)
  - Bundle GitHub Actions coverage report uploads (#1680)
  - Create Pyomo-specific unittest.TestCase and update fragile test (#1682)
  - Add python-louvain to testing infrastructure (#1646)
  - Update the idaes-ext download in the GitHub Actions workflows (#1698)
  - Make pyomo.common.timing tests more flexible (#1711)
  - Rename assertRelativeEqual to assertStructuredAlmostEqual (#1719)
  - Replace ubuntu-latest with ubuntu-18.04 in GitHub Actions workflows (#1709)
  - Change GitHub Action setup-miniconda action to version 2 (#1696)
  - GitHub Actions: add linux/conda and linux/parallel builds (#1725)
  - Timing updates (#1728)
  - Remove Travis-CI test driver and references to Travis-CI (#1722)
  - Resolve test failures with xpress 8.11 (#1706)
  - Fix installation of pymumps through conda in GitHub Actions (#1731)
  - Add TPLs, update solver PATH, and remove solvers from SLIM builds in GitHub
    Actions tests (#1732)
  - Add Gurobi to GitHub Actions workflows (#1721)
  - Add Python 3.9 to test suite (#1753, #1755)
- Documentation
  - Add docstrings in pyomo.util.config (#1707)
- Contributed Packages
  - Add new package for performing Louvain community detection on a
    Pyomo model (#1526)
  - FME: treat bounds of variables being projected as constraints in the
    transformation (#1664)
  - MindtPy: add extended cutting plane and global OA strategies (#1590)
  - Parmest: bug fix in return_values, resolve example incompatibility with
    Python<3.5, use CUID v2 representations, update handling of indexed
    stage variables (#1645, #1699, #1734, #1737)
  - PyNumero: add ENABLE_HSL cmake option, extend Cyipopt solver
    interface, improve MPI Matvec performance, update mumps interface to
    use attempt_import, performance enhancements to pynumero.sparse,
    misc updates (#1632, #1653, #1610, #1667, #1681, #1705, #1724)

-------------------------------------------------------------------------------
Pyomo 5.7.1   (15 Sep 2020)
-------------------------------------------------------------------------------

- General
  - Add functions for checking the solver termination condition to environ (#1540)
  - Remove appdirs dependency (#1558)
  - Setting version number for new deprecation warnings (#1524)
  - Add generic tarball creation to GitHub Actions Workflow for creating Wheels
    (#1522)
  - Deprecate pyomo install-extras (#1521)
  - Add test to monitor 'import pyomo.environ' time (#1577)
  - Move ComponentMap and ComponentSet to pyomo.common.collections (#1593)
  - Fix issue when parsing /etc/os-release (#1603)
  - Fix deprecation decorator tests (#1629)
- Core
  - Add a logical expression system (#1507)
  - Resolve table formatting with unicode character data (#1530)
  - Fix Model.compute_statistics method (#1553)
  - Deprecate CUID targets in add_slack_variables transformation (#1563)
  - Defer import of sympy in the cnf_walker (#1576)
  - Minor fixes to units (#1582, #1584)
  - Add IndexedSet.data method that was mistakenly removed (#1609)
  - Add support for class method initializers (#1608)
  - Rework constraint initialization (#1592)
  - Remove truthiness assumption for sequence in TuplizeValuesInitializer (#1620)
  - Add deprecated support for Var(within=RealSet) (#1619)
  - Fix standard repn of external functions with fixed arguments (#1623)
- Solver Interfaces
  - Use Baron option to more consistently return duals (#1538)
  - Add direct and persistent interfaces to Xpress (#1443)
  - Augment persistent solver interfaces to support column generation (#1568)
- DAE Updates
  - Add options to solve_consistent_initial_conditions (#1534, #1606)
  - Add find_nearest_index method to ContinuousSet (#1559)
  - Allow flattening of ctypes other than Var (#1583)
- GDP Updates
  - Fix a couple bugs in GDP basic steps (#1532)
  - Make key for calculated M values in bigM dictionary consistent (#1618)
- Testing
  - Fix to retry codecov upload if codecov.sh script fails (#1539)
  - Update to GitHub Actions tests to fix an OSX Python environment failure (#1555)
  - Allow failing codecov uploads on GitHub Actions tests (#1561)
  - Add doctests to the GitHub Actions builds (#1542)
  - Add manual job trigger for GitHub Actions Workflows (#1578)
  - Rebuild GitHub Actions download caches (#1598)
  - Disable OS package cache for OSX (#1625)
- Documentation
  - Clean up some old documentation on sparse sets (#1545)
  - Remove OnlineDocs spy files from the repository (#1549)
  - Resolve RTD documentation build warnings (#1552)
  - Fix typo in code snippet (#1624)
  - Update contribution guide (#1621)
- Contributed Packages
  - Defer the Z3 availability check in the satsolver (#1529)
  - MindtPy: add configuration arguments, bug fixes (#1500)
  - Parmest: fix solver options typo, update graphics module (#1531, #1622)
  - PyNumero: Add scaling to Cyipopt interface, set AMPLFUNC before loading NL
    files, improve MPI matvec performance (#1554, #1546, #1610)
  - Fix bugs in sensitivity toolbox (#1536, #1574)
  - Add integer arithmetic option to FME transformation (#1594)

-------------------------------------------------------------------------------
Pyomo 5.7.0   (19 Jun 2020)
-------------------------------------------------------------------------------

- General
  - Remove references to Python versions older than 2.7 (#1361)
  - Fix Python 3 compatibility issue in setup.py (#1354)
  - Updates to the FileDownloader, fix Python 3 type issue, add utility for
    identifying os platform, improve find_library method (#1353, #1368, #1373)
  - Fix tokenization, update errors in the DAT parser (#1366, #1516)
  - Fix typo in relax_integrality deprecation warning (#1385)
  - Promote __version__ identifier to the pyomo module scope (#1390)
  - Update to only compute is_fixed when necessary (#1402, #1415)
  - Add ConfigEnum class (#1418)
  - Prevent exception for transformations missing doc string (#1454)
  - Automate wheel creation using GitHub actions (#1409)
  - Replace uses of pyutilib.enum package (#1506)
- Core
  - Units support for Pyomo components and models (#1341, #1460, #1494, #1504)
  - Integrate new Set component implementation (#1319)
  - Add deprecation warning for Set.value_list (#1371)
  - Fix bug when constructing empty sets with dimen>1 (#1377)
  - Rename component.type() to component.ctype (#1376)
  - Fixes for linear expression handling (#1403, #1405)
  - Fix handling of deactivated blocks in relax_integer_vars transformation (#1428)
  - Fix component_data_objects for scalar components with no len() (#1436)
  - Ensure block rules are always called (#1438)
  - Fix construction of scalar derived blocks (#1459)
  - Updates to native Pyomo differentiation (#1423)
  - Expand expression template support (#1433)
  - Ensure sympy configuration happens (#1470)
  - Add deprecation wrapper for old StreamBasedExpressionVisitor API (#1488)
  - Identify squared linear sums as quadratic in generate_standard_repn (#1493)
  - Check for consistent dimensions when slicing (#1492)
- Solver Interfaces
  - Updates to GAMS interface, update GAMSDirect.available to catch unexpected
    exceptions, add GDX interface for returning solution values, improve error
    reporting from GAMS, fix handling of fixed variables in LinearExpressions
    (#1351, #1446, #1382, #1463)
  - CPLEXDirect performance improvements (#1416)
  - Improve Baron warning for ResName/TimName options (#1486)
- PySP Updates
  - Clean up PySP sizes example (#1395)
  - Remove use of pyutilib.enum package (#1464)
- GDP Updates
  - Rewrite of the Chull transformation (#1421)
  - Fix bug in bigm transformation for nested disjunctions (#1479)
  - Rename gdp.chull transformation to gdp.hull (#1471)
  - Deprecate GDP reclassify transformation (#1502)
- DAE Updates
  - Add DAE set manipulation utilities (#1288)
  - Add function for identifying and solving for consistent initial conditions (#1410)
  - Fix DAE flattener for non-time-indexed blocks (#1489)
- Network Updates
  - Update to cast fixed values back to float (#1469)
- Testing
  - Update Travis badge to reflect migration from .org to .com (#1364)
  - Add test coverage to linux/osx GitHub actions (#1359)
  - Update Baron, Ipopt, gjh_asl_json, Xpress solvers in GitHub actions
    (#1378, #1393, #1394, #1452)
  - Unified GitHub actions workflow (#1426)
  - Disable Appveyor testing (#1447)
  - Update coverage configuration and reporting (#1451, #1455, #1457, #1462)
  - Solver testing improvements (#1473)
- Documentation
  - Update RangeSet docstring (#1437)
  - Expand developer documentation (#1472)
  - Update README.txt for building the Sphinx documentation (#1480)
  - Add documentation for sensitivity_toolbox (#1481)
- Contributed Packages
  - PyNumero updates, add CyIpopt interface, enable build through
    build-extensions, add HSL interface, package reorganization, add operations
    for block vectors/matrices, fix broken examples (#1356, #1412, #1381, #1419,
    #1439, #1467, #1518)
  - Add Fourier-Motzkin elimination (FME) transformation (#1370, #1440)
  - GDPOpt updates, fix use of numeric derivatives, preserve implicit config
    options (#1422, #1432,)
  - Mindtpy updates, add lp/nlp algorithm, bug fixes, documentation, fix
    warmstart, add cycling check (#1391, #1476)
  - Fix and deprecate constraint tightener transformation in the preprocessing
    module (#1458, #1388)
  - FBBT updates, fix numerical issues (#1360)
  - parmest updates, add ScenarioCreator, update data in example to match paper
    results, calculate covariance matrix (#1363, #1474, #1475)
  - Add basic interior point algorithm based on PyNumero (#1450, #1505, #1495)

-------------------------------------------------------------------------------
Pyomo 5.6.9   (18 Mar 2020)
-------------------------------------------------------------------------------

- General
  - Fix bug and improve output formatting in pyomo.util.infeasible (#1226, #1234)
  - Add 'version' and 'remove_in' arguments to deprecation_warning (#1231)
  - Change NoArgumentGiven to a class and standardize usage (#1236)
  - Update GSL URL to track change in AMPL SSL certificate (#1245)
  - Clean up setup.py (#1227)
  - Remove legacy build/test/distribution scripts (#1263)
  - Use dict comprehension for constructing dictionaries (#1241)
  - Fix report_timing for constructing objects without index_set (#1298)
  - Add missing import for ftoa (#1320)
  - Add attempt_import and standardize yaml imports (#1328)
  - Add get_text_file method to the FileDownloader (#1330)
  - Add helper function to retrieve solver status (#1335)
  - Speed up import of pyomo.environ (#1344)
- Core
  - Update Units test to handle Pint 0.10 (#1246)
  - Move blockutil.py from pyomo/core to pyomo/util (#1238)
  - Deprecate pyomo.connectors (#1237)
  - Add initial implementation for a MatrixConstraint (#1242)
  - Fix _BlockData set_value() (#1249)
  - Raise error on failed Param validation (#1272)
  - Fix return value for component decorator (#1296)
  - Change mult. order in taylor_series_expansion for numpy compatibility (#1329)
  - Deprecate 'Any' being the default Param domain (#1266)
- Solver Interfaces
  - Update CPLEX direct interface to support CPLEX 12.10 (#1276)
  - Shorten GAMS ShortNameLabeler symbols (#1338)
  - Add branching priorities to CPLEXSHELL (#1300)
- PySP updates
  - Added a csvwriter test to the rapper tests (#1318)
  - Fix csvwriter when NetworkX used to specify the scenario tree (#1321)
- GDP updates
  - Update BigM estimation for nonlinear expressions (#1222)
  - Refactor GDP examples for easier testing (#1289)
  - Rewrite of BigM transformation (#1129)
- DAE updates
  - Add a flatten_dae_variables utility (#1315, #1334)
- Network updates
  - Allow disabling split_frac and correct bounds on duplicated variables (#1186)
- Testing
  - Remove 'nightly' tests from the 'expensive' suite (#1247)
  - Set up GitHub actions for Linux, OSX, and Windows testing (#1233, #1232,
    #1230, #1262, #1277, #1317, #1281, #1323, #1331, #1342)
  - Clean up Travis driver (#1264)
  - Update Appveyor driver (#1293, #1343)
  - Add GitHub Actions workflow for testing forks/branches (#1294)
  - Update tests to use sys.executable to launch python subprocesses (#1322)
  - Improve testing and coverage reporting for MPI tests (#1325)
  - Update codecov config to reduce failing coverage checks on PRs (#1345)
- Documentation
  - Remove CBC from installation documentation (#1303)
  - Add GitHub Actions documentation to the contribution guide (#1316)
  - Documentation for using indexed components in persistent solver interfaces
    (#1324)
  - Documentation for developers on using forks (#1326)
- Contributed Packages
  - Deprecate pyomo.contrib.simplemodel (#1250)
  - Updates to GDPopt, Merge GDPbb into GDPopt (#1255, #1268)
  - PyNumero updates, redesign of NLP interfaces API, support for Windows,
    updates to PyNumero.sparse, add MUMPS interface (#1253, #1271, #1273, #1285,
    #1314)
  - FBBT fixes and tests (#1291)
  - Updates to Parmest, support for leave-N-out sampling and data reconciliation,
    graphics and documentation improvements (#1337)
  - Fix Benders MPI logic bug and expand parallel test coverage (#1278)

-------------------------------------------------------------------------------
Pyomo 5.6.8   (13 Dec 2019)
-------------------------------------------------------------------------------

- General
  - Fix collections imports for Python 3.x (#1175)
  - Add verbose logging options for infeasibility testing (#1207)
- Core
  - Remove deprecated call to formatargspec in Python 3.x (#1174)
  - Remove old, unreachable code in PyomoModel.py (#1189)
  - Clean up default argument for variable fix method (#1221)
- Solver Interfaces
  - Fix incorrect precision warnings in the GAMS and BARON writers (#1166)
  - Fix typo in MOSEK problem status handling (#1170)
  - Fix bug in NEOS interface for Python 3.x (#1206)
- DAE updates
  - Fix bug in get_index_information (#1195)
  - Remove use of _implicit_subsets (#1197)
- PySP updates
  - Expose options for CC and CVaR in the rapper interface (#1211)
- Testing
  - Disable extras on Appveyor Python 3.7 build (#1180)
  - Add Python 3.8 to Travis test suite (#1182)
  - Updates to Jenkins build infrastructure (#1216)
- Documentation
  - Minor updates for PySP, GDP, stream-based walkers, and installation
    (#1168, #1191, #1204, #1208)
- Contributed Packages
  - Add MC++ compiled library version checking (#1172)
  - Fix minor type mismatch in PyNumero (#1214)
  - Fix deprecation warning in Mindtpy (#1188)
  - Add test skipping to trust region tests requiring IPOPT (#1220)

-------------------------------------------------------------------------------
Pyomo 5.6.7    (7 Nov 2019)
-------------------------------------------------------------------------------

- General
  - Update links to tutorials and examples in the README (#1082)
  - Verify Python 3.8 support (#1162)
- Core
  - Add a guard in getname for ComponentData with no parent (#1075)
  - Add support for sqrt in Python-based AD implementation (#1086)
  - Add methods to IndexedVar to set bounds for all indices (#1087)
  - Replace ReciprocalExpression with DivisionExpression (#989)
  - Fix bug where Set multiplication changed the original Sets (#1103)
  - Improve ease of use of LinearExpression constructor (#1097)
  - Tech preview of the Set component hierarchy rewrite (#1111)
  - Add a get_interval method for Sets (#1128)
  - Update pprint method for derived classes and add support for printing
    ComponentData (#1043)
  - Create a common interface for differentiation (#1059)
  - Add function to calculate the Taylor series approximation of an
    expression (#1059)
  - Fix in Sympy interface to avoid complex numbers (#1140)
  - Add support for log10 in differentiation (#1145)
- Solver Interfaces
  - Fix bug in BARON and GAMS writers for handling exponential expressions (#1114)
  - Improve Gurobi persistent interface, enable callbacks, lazy constraints,
    and more (#1137)
- MPEC updates
  - Update transformations to descend into disjuncts (#1085)
- PySP updates
  - Replace Graph.node with Graph.nodes to support networkx 2.4 (#1141)
- GDP updates
  - Use a name buffer when generating component names (#1149)
- Testing
  - Add pathos library to the Appveyor build (#1135)
- Contributed Packages
  - Model viewer improvements and bug fixes (#1071, #1092)
  - MindtPy outer approximation improvements (#983)
  - GDPOpt improvements, support for Ports (#1098, #1133)
  - FBBT improvements and bug fixes, updates to interval arithmetic, support
    for log10, replace math.inf with float('inf') (#1131, #1134, #1102, #1078,
    #1076, #1146)
- Documentation
  - Fix typo in working_models.rst (#1094)
  - Add documentation for user interface to LinearExpression (#1120)

-------------------------------------------------------------------------------
Pyomo 5.6.6   (21 Jun 2019)
-------------------------------------------------------------------------------

- Core
  - Remove a line in standard_repn that was unreachable (#1063)
- Solver Interfaces
  - Fix symbolic labels for BARON and GAMS (#1057)
  - Fix in cplex_direct to check for solution before extracting bounds (#1060)
  - Fix in NL writer where a single-term sum output a n-ary operator (#1067)
- Contributed Packages
  - FBBT improvements (#1002)
  - Benders improvements (#1061)
  - GDPopt updates (#922, #1065)
  - Model viewer improvements, add residual table, improved install/testing,
    support for AbstractModels (#955, #1054, #1066)

-------------------------------------------------------------------------------
Pyomo 5.6.5   (10 Jun 2019)
-------------------------------------------------------------------------------

- General
  - Remove non-ascii character from README.md (#1045)
- PySP
  - Fix docstring formatting (#1042)
- Testing
  - Resolve book test failures (#1038)
- Contributed Packages
  - Benders cut generator component (#1028)

-------------------------------------------------------------------------------
Pyomo 5.6.4   (24 May 2019)
-------------------------------------------------------------------------------

- General
  - Resolve project description rendering on PyPI

-------------------------------------------------------------------------------
Pyomo 5.6.3   (24 May 2019)
-------------------------------------------------------------------------------

- General
  - Update the README (#990, #991, #992)
  - Fix compilation with Cython
  - Change the default use_signal_handling from True to None (#1014)
- Testing
  - Add a license check for Mosek (#1007)
  - Add a Cythonization test to Travis (#1012)
- Core
  - Bug fix in IndexedComponent using a nonhashable object as an index (#1006)
  - Bug fix in fix_discrete transformation for domain comparison (#996)
  - Add read-only view of kernel matrix_constraint (#1011, #1013)
  - Add specialized conic constraints to kernel (#980, #1018)
  - Bug fix in standard repn when using a fixed variable as an exponent (#978)
  - Add _associativity to _ExpressionData (#1031)
- Solver Interfaces
  - Bug fix in CBC plugin related to certain objective values (#1004)
- Contributed Packages
  - FBBT improvements (#994)
  - MC++ variable bound guards (#1015)
  - Improvements to bounds_to_vars preprocessing transformation (#1016)
  - GDPbb improvements and cleanup (#982)

-------------------------------------------------------------------------------
Pyomo 5.6.2   (1 May 2019)
-------------------------------------------------------------------------------

- General
  - Fix setup.py for Windows installation with Python3 and Cython (#823)
  - Reorganize how Pyomo manages the version number (#854)
  - Updated build badges on main Github page (#867)
  - Fix bug in model size report utility (#904)
  - Catch eval errors in calculate_variable_from_constraint utility (#872)
  - Add project description file used by COIN-OR server (#891)
  - Add common utility for downloading files (#819, #927)
  - Add infrastructure for building compiled extensions (#940, #959)
  - Return the subcommand error code from the pyomo script (#958)
  - Fix the "pyomo install-extras" command (#981, #986)
  - Deprecate pyomo.bilevel and pyomo.duality (#943)
  - Update deprecation decorator to include version number (#943)
- Testing
  - Update Appveyor configuration to use --no-update-deps conda option (#837)
  - Enable publication of coverage results from Jenkins (#842, #892)
  - Update Travis configuration with new docker image location (#880)
  - Activate coverage for Anaconda Travis builds (#887)
  - Update DataPortal tests requiring PyYaml to not rely on file diffs (#931)
  - Add simplified Jenkins test driver (#949, #952, #960)
  - Restrict auto-build of PyomoGallery to master branch builds (#985)
- Core
  - Reference improvements to support sparse components (#830)
  - Fixed typo on Var docstring (#865)
  - Initial support for units handling (#833)
  - Make component slice objects picklable/deepcopyable (#914)
  - Allow variable identification for LinearExpression (#920)
  - Split expr_pyomo5.py into three files (#888)
  - Fix ordered sets not iterating in correct order (#936)
  - Add SimpleBlock to pyomo.core namespace (#941)
  - Kernel updates (#951)
  - Fix expression infix representation (#966)
- Solver Interfaces
  - Add logic to control signals in shellcmd solvers (#856)
  - Narrow the check for a valid Baron license (#857)
  - Add missing import to kestrel_plugin.py (#895)
  - Allow '~' in CPLEX file names (#925)
  - Updates to CBC plugin to handle all return statuses and support warm starts
    (#874, #946)
  - Add Mosek solver plugin (#971)
- GDP updates
  - Add nonconvex heat exchanger network example (#824)
  - Bugfix for GDP Var mover (#667)
  - Add basic step example (#848)
  - Add GDP logo (#873, #876)
  - Improve error message in GDP reclassification (#884)
  - Updates to Stickies example (#937, #948)
  - Simplify the fix_disjuncts transformation (#921)
- DAE updates
  - Use ConfigBlocks for discretization options handling and several bug fixes
    (#967, #973)
- Network updates
  - Fix sequential decomposition when ports contain References (#975)
- Contributed Packages
  - Parmest updates to make API more flexible, updated examples, documentation
    and tests (#814)
  - GDPopt algorithm enhancements, cut generation bug fix, add example
    to tests, time limit option support (#805, #826, #852, #970)
  - Multistart solver wrapper bug fix for identifying the Objective (#807)
  - Updates to MC++ interface and documentation (#827, #929, #939, #964, #968)
  - Add import guards for optional packages like Scipy and Pandas (#853, #889)
  - PETSc wrapper fixes and enhancements (#806)
  - Disjunctive bound tightening with GLOA (#877)
  - Use CBC for GDP_bounds OBBT computation (#933)
  - Set minimum Numpy version for Pynumero (#962)
  - New packages:
    - Mixed-Integer Decomposition Toolbox in Pyomo (MindtPy) (#386, #970)
    - Feasibility based bounds tightening (#825, #863, #883, #954)
    - GDP branch and bound solver (#847, #870, #886)
    - Satisfiability solver interface to z3 (#862, #885)
    - Automatic/symbolic differentiation of Pyomo expressions in Python
      (#818, #926)
    - Graphical model viewer/editor (#834, #878, #897, #911, #945)
- Documentation
  - Show how to extend an objective function (#820)
  - Show how to use PySP with a ConcreteModel (#831)
  - Update documentation on assigning values to a Param (#841)
  - Update Set documentation (#850)
  - Fixed typo in overview documentation (#864)
  - Show how to activate/deactivate constraints (#932)

-------------------------------------------------------------------------------
Pyomo 5.6.1   (18 Jan 2019)
-------------------------------------------------------------------------------

- General
  - Fix setup.py installation failure on Windows (#813)
- Testing
  - Add assertion method for comparing lists of floats (#800)
- Solver interfaces
  - Bugfix in GAMS writer related to splitting long lines (#797)
  - Allow ":" in cplex file names (#810)
  - Fixes to NEOS interface (#793)
- GDP updates
  - Fixed typo in GDP example (#801)
  - Add support for RangeSet in GDP transformations (#803)

-------------------------------------------------------------------------------
Pyomo 5.6     (19 Dec 2018)
-------------------------------------------------------------------------------

- General
  - Removing testing and support for Python 2.6 (#322)
  - Adding TerminationCondition and SolverStatus to pyomo.environ (#429)
  - Refactoring pyomo.util into pyomo.common for the general utilities that do
    not depend on the rest of Pyomo, and the user-focused modeling utilities in
    pyomo.util. (#502)
    - New utilities: Model size report (#579), calculating variable
      values (#763)
  - Fix pyomo help -s command (#551)
  - Fixes to avoid deprecation warnings (#760, #765)
- Core
  - Adding the Pyomo5 expression system, which supports PyPy (#272,
    #471, #475, #514, #520, #524, #536, #615)
    - Optimizations to minimize use of NumericConstant objects (#469)
    - Implemented a general-purpose expression visitor (#671)
  - Implicitly order Pyomo sets based on insertion order (#568)
  - Fix for overriding values in IndexedSets (#710)
  - Efficiency improvements when processing unhashable indexes (#467)
  - Replace PyUtilib factories with native Pyomo factories (#661)
  - unique_component_name now checks all attributes (not just components) (#497)
  - Adding relative_to argument for getname() (#495)
  - Add 'SubclassOf' to support finding components based on sub class (#723)
  - Move PseudoMap out of _BlockData class (#778)
  - Add Reference "component" (#655, #742)
  - Transformations map create_using arguments through the model clone (#497)
  - Implemented a scaling transformation (#694)
  - Improvements for External Functions
    - Resolved problems with ExternalFunctions with fixed arguments (#442)
    - Add method to return F,G,H from AMPL external functions (#666, #668)
  - Improvements to symbolic expressions
    - Add support for sqrt function in differentiate() (#642)
    - Update symbolic differentiation to be compatible with sympy 1.3
      and performance improvements (#715, #719)
    - Silence invalid error messages logged with Template Expressions (#634)
  - Kernel improvements
    - Move kernel util (#522)
    - Kernel updates (#761)
    - Kernel updates simplifying handling of active status on containers,
      save storage key on child, warn overwriting container index, fix
      linear_canonical_form flag on linear_constraint (#569, #570, #571,
      #589, #598)
- Solver interfaces
  - Fix to solver tempfiles (#590)
  - Prevent crash when solver returns suffixes but not values for
    variables (#596)
  - Resolving issue when path names with spaces appear in CPLEX run file (#485)
  - Fix for spaces in cplex file names (#586)
  - Bugfix in Gurobi direct interface (#635)
  - Fix NEOS interface when behind a proxy in Python 3.x (#644)
  - Fixed Baron writer sqrt error (#751)
  - Corrected Ipopt capabilities (#573)
  - GAMS writer
    - fix for power() expressions (#454)
    - fix for fixed variables (#510)
    - ShortNameLabeler to limit symbol names (#512)
    - Add warning about zero division in variable evaluation (#529)
    - Use tmpdir for subprocess files (#530)
    - Integrate options attribute (#558)
    - Change writer ctype checking (#593, #641)
    - Fix double-operator due to negative constant (#698)
    - Fix Windows stdout compatibility issue (#779)
  - LP writer
    - More efficient string manipulation in LP writer (#610)
    - Fix generation of bounds for fixed variables in LP writer (#623)
  - Cythonized writers in pyomo.repn (#675)
- Bilevel:
  - Create pyomo.dualize (#606)
  - Updates to dualization and bilevel solvers (#535)
- DAE updates
  - Added support for external functions (#526)
  - Bugfix in Casadi interface (#544)
  - Code overhaul and improved test coverage (#660)
- GDP updates
  - Implemented improper basic steps (#582)
  - Add examples from literature (#581, #702)
  - Improve BigM test robustness (#659)
  - Bug fixes (#490, #498)
- PySP updates
  - Python 3.7 support (#463)
  - Fix bugs in finance example (#564, #578)
  - Added a wrapper for PySP to create a scripting interface (#689, #727, #737)
  - Bug fixes (#736, #788)
- New packages:
  - DataPortal:
    - Move DataPortal up to a top-level pyomo.dataportal subpackage (#607)
    - Update options for pymysql (#678)
  - Network
    - Replaced Pyomo connector infrastructure with Pyomo.Network and
      added sequential modular simulation capability (#583, #648)
- Contributed packages:
  - GDPopt updates to use config blocks, support integer variables,
    documentation, callback renaming, new initialization strategy
    (#513, #541, #595, #599, #633, #650, #717)
  - Update trust region to use ConfigBlocks (#738, #785)
  - New packages:
    - Preprocessing transformation for variable aggregation (#533, #617)
    - Compute disjunctive variable bounds (#481)
    - Parmest package for parameter estimation (#706, #733, #769, #781)
    - PyNumero package for numerical optimization (#725, #775)
    - sensitivity_toolbox for interfacing with sIPOPT (#766)
    - PETSc AMPL wrapper (#774)
- Testing
  - Resolved several build errors on Appveyor (#438, #539, #552, #577, #705)
  - Reconfigured Travis CI to run tests using DockerHub images, work on PRs
    from forks, run Python 3.7 (#479, #517, #600)
  - Jenkins driver updates to fix errors when files are moved/renamed,
    delete coverage info from previous builds, decode unicode in
    booktests driver, test fixes, new drivers (#627, #629, #630, #631,
    #704, #735, #772, #771, #786)
- Documentation and Examples
  - Converted old asciidoc documentation to Sphinx (#486, #574)
  - Reorganized the Sphinx documentation to make it easier to navigate
    and merge old and new documentation sections (#699)
  - Create more prominent section for documentation on contrib packages (#729)
  - Documentation updates: scripting (#543, #748, #756), pyomo.network
    (#651, #745, #780), preprocessing transformations (#653), GAMS writer
    (#518), pyomo.kernel (#767), abstract models (#777), general (#782)
  - Document PR process (#628)
  - Updated README.md to include link for performance plots (#534)
  - Update examples (#436)

-------------------------------------------------------------------------------
Pyomo 5.5.1   (26 Oct 2018)
-------------------------------------------------------------------------------

- General
  - Adding support for Python 3.7

-------------------------------------------------------------------------------
Pyomo 5.5     (14 Apr 2018)
-------------------------------------------------------------------------------

- Move preprocessing transformations to contrib (#426)
  - Moves bounds_to_vars and propagate_zero_sum into contrib.preprocessing and
    changes aliases to contrib.bounds_to_vars and contrib.propagate_zero_sum.
- Improved external function support
  - Fixing AMPLFUNC environment variable management for "derived ASL
    solvers (ipopt, conopt, scipampl) (#423)
  - Adding support for using the AMPL GSL library (#427)
- PySP updates
  - Fixing atypical methods in PySP for generating a scenario tree and
    associated instances (#422, #424)
  - Adding support in PySP for new persistent solver API (#397)
  - Fixing bug in Eckstein-Combettes PySP-PH extension for
    handling unused model variables (#396)
- GDP updates
  - add support for implicit declaration of multi-constraint Disjuncts (#410)
  - switch to using ConfigBlocks for chull, bigm relaxations (#410)
  - fix for Big-M with disjuncts containing indexed blocks (#421)
- Fix for discretizing block hierarchical models with
  circular references (#406, #353)
- Fix formatting for small numbers in Baron file writer (#405)
- Allow retrieving a mutable Param even when no initial or
  default value is specified (#393)
- Python3 fix for Set.pprint
- Make handling of trivially non-binding constraints consistent across
  the various solver interfaces (#392)
- Fix to pprint for Params that store non-numeric values
- Removing use of namespace_packages in setup.py
- Modifying transformation that detects trivial constraints to maintain
  a list of the constraints that were deactivated (#385)
- Documentation updates (#425)

-------------------------------------------------------------------------------
Pyomo 5.4.3   (2 Mar 2018)
-------------------------------------------------------------------------------

- Another fix in the release process.

-------------------------------------------------------------------------------
Pyomo 5.4.2   (2 Mar 2018)
-------------------------------------------------------------------------------

- Misc fix in the release process.

-------------------------------------------------------------------------------
Pyomo 5.4.1   (28 Feb 2018)
-------------------------------------------------------------------------------

- Misc version increment to support pypi idiosyncrasies.

-------------------------------------------------------------------------------
Pyomo 5.4     (28 Feb 2018)
-------------------------------------------------------------------------------

=======
- Remove checks for is_indexed from PersistentSolver methods (#366)
- GDP rewrite (#354)
- Updated gdp.chull to handle named expressions (#318)
- Fixed Disjunction to support declaration without expr or rule (#241)
- Misc Gurobi and Cplex fixes (#368)
- Fix to gurobi_direct initialization when gurobipy is not available. (#363)
- Fixes to persistent solvers (#361)
- Fix to cplex 12.6 error management. (#357)
- Fix intermittent "unclonable" component error (#356)
- Fix GAMS interface ignoring `tee` (#351)
- Remove duplicated 'Key' for Expression.display() (#349)
- removing import from GUROBI_RUN.py (#344)
- Add logfile option to GAMS solver (#302)
- Improvements for Gurobi/Cplex Interfaces (#331)
- GDPopt solver initial implementation (#337)
- Fixed to update IndexedVar domain display (#5)
- Reorganize the API documentation (#333)
- Transformation to strip bounds from model (#220)
- Adding transformation to propagate equality-linked fixed variables (#192)
- Fixing Python 3.x compatibility for Sets (#305)
- Fix so that GAMS writer can safely ignore Connector objects (#310)
- Faster solution parsing for CBC (#311)
- Reworked pyomo.contrib and added the pyomo.contrib.example package (#299)
- Fixes for Python 2.6 compatibility in third-party packages
- Adding diagnostic functions for Pyomo models (#217)
- Allow None to be a valid value for Params (#301)
- Improved testing with Travis and Appveyor
- Fixes to Pyro management with PySP solvers
- Transformation of explicit constraints to variable bounds (#190)
- Corrected the settimelimit option with CBC (#265)
- Bug fixes in direct/persistent solver interfaces (#282)
- A major rework of the PySP solver interface
- Added testing for Sphinx code fragments
- Various updates to online documentation generated by Sphinx
- Add error message to model.write if guess_format returns None (#260)
- Ensure that generate_cuid_names descends into Disjunct objects (#176)
- Fix GAMS writer to better support the power function (#263)
- Added persistent interfaces for CPLEX and Gurobi (#262)
- Added additional timing information for model construction and solvers
- Logging overhaul and support for timing concrete models (#245)

-------------------------------------------------------------------------------
Pyomo 5.3     (21 Oct 2017)
-------------------------------------------------------------------------------

- Removed testing for Python 3.4
- Added exp() to symbolic module (#151)
- Resolved representation error with 1/var (#153)
- Added pyomo.core.kernel (#130)
- Various solver interface fixes: CBC, SCIP, IPOPT, GLPK
- Add docstring to apply function in Transformation (#174)
- Adding a TerminationCondition enum value for "Infeasible or Unbounded" (#171)
- New scripts for performance testing
- Use the has_lb() and has_ub() helper methods for vars and constraints.
- Added documentation tests.
- Updates to DAPS (#200)
- Fix KeyError message in NL writer (#189)
- New ODE/DAE simulator interface for pyomo.DAE (#180)
- Added deprecation decorator (#203)
- New transformation to fix nonnegative variables in equality constraints (#198)
- Deprecated BigM piecewise representations (#216)
- Added GAMS solver interface (#164, #215, #221, #240)
- Updates to testing configuration on Travis and Appveyor
- Tracking changes in pyutilib.th

-------------------------------------------------------------------------------
Pyomo 5.2     (14 May 2017)
-------------------------------------------------------------------------------

- Resolved timeout issues running NEOS solvers
- Changing the generic ASL solver plugin to use '-AMPL' rather than '-s'
- Fixed loading solutions into sub-blocks (#127)
- Cloning a model now preserves any previous solution(s).
- Fixing pickling of ModelSolutions objects. (#65)
- Allow cloning of blocks even when attributes cannot be deep copied
- Fixed how GUROBI_RUN reports the solve time.
- Added DAPS solver (#139)
- Adding support for Python 3.6. (#103)
- Restricting user defined component names on Blocks to avoid overwriting
  important Block methods. (#126)
- Updating pyomo.dae transformations to support Block-derived components.
  (#132, #129, #89)
- Fix rare issue where numeric constants in the left- and right-hand sides of a
  double inequality expression were incorrectly mapped to an equality
  expression even when the values were different.
- Creating an Ipopt solver plugin with additional functionality for sending
  options to Ipopt using an options file. Options beginning with 'OF_' will be
  interpreted as options that should appear in an options file. A warning is
  printed when this will cause an existing 'ipopt.opt' file in the current
  working directory to be ignored.
- Make the Ipopt solver plugin more gracefully handle the case when a solution
  file is not created by the solver (e.g., when the solver exits because there
  are too few degrees of freedom). (#135)
- Reduce time required to import pyomo.environ by delaying checks for
  availability of solver interfaces until they are used. (#109)
- Adding support for solving Blocks as if they were Models. (#110, #94)
- Adding support for declaring components on a model by decorating the rule
  instead of explicitly invoking the setattr method on the block. This
  eliminates the normal pattern of the component name appearing in three
  places. (#99)
- Fixes to pyomo.gdp transformations for nested blocks.
- Fixing case in block recursion where descend_into was not being honored.
- Bug fix that impacted non-serial solver managers.
- Make checks for is_indexed() more efficient. (#105)
- Fixes to files for PEP8 compliance.
- Fix to statues set by CPLEX plugin when problem is unbounded.
- Improving Param exception messages.
- Putting a daps release in PySP (#124)
- Updating the bilinear transform to avoid creating a Set `index`.

-------------------------------------------------------------------------------
Pyomo 5.1.1   (8 Jan 2017)
-------------------------------------------------------------------------------

- Monkeypatch to resolve (#95)

-------------------------------------------------------------------------------
Pyomo 5.1     (4 Jan 2017)
-------------------------------------------------------------------------------

- Added a CONOPT plugin to handle a custom SOL file output (#88)
- Changed 'pyomo solve' to use any Model object that is found, rather than
  requiring the default object to be named 'model' (#80)
- Reworked the solver testing infrastructure to enable enumeration of all
  test scenarios, which is used by 'pyomo test-solvers' (#78)
- Fixes for xpress solver options writer (#79)
- Resolved an issue where Pyomo was unnecessarily cloning most inequality
  expressions (#80)
- Reworked the Pyomo *.DAT file parser and post-parse processing logic. This
  eliminates parse ambiguities that were causing test failures.
- Finalized book test examples for the new edition of the Pyomo book
- Added pyomo.contrib to support integration of third-party Pyomo libraries
  in a standard manner
- Fixed TravisCI testing issues where the incorrect version of Python was being
  tested
- Fixed error messages in pyomo.dae (#81)
- Revised CBC interface to recognize intermediate non-integer solutions (#77)
- Added checks for IPOPT test failures based on version info (#72)
- Removed support for OpenOpt

-------------------------------------------------------------------------------
Pyomo 5.0.1   (16 Nov 2016)
-------------------------------------------------------------------------------

- Updating PyUtilib dependency

-------------------------------------------------------------------------------
Pyomo 5.0     (15 Nov 2016)
-------------------------------------------------------------------------------

- Added examples used in the Pyomo book to the Pyomo software repos
- Added support for automatically generating "template expressions"
- Fix tuple flattening in the case of unhashable SimpleParam in indexes
- Deactivated solver plugins before the are returned from the SolverFactory
- Significantly simplified the polynomial_degree() logic for Pyomo4 expressions
- Updates to expression logic
- Adding a 'feasible' attribute to the TerminationCondition Enum
- Updates to ExternalFunction logic
- Renamed the component name() method to getname() and added a 'name' property
- Fix to allow creating undefined (and uninitialized) mutable params
- Added more careful checks in Set() and Param() to ensure that an abstract set
  expression is not evaluated
- Added DAE examples to Pyomo

-------------------------------------------------------------------------------
Pyomo 4.4.1
-------------------------------------------------------------------------------

- Fixing PyUtilib dependency version.

-------------------------------------------------------------------------------
Pyomo 4.4
-------------------------------------------------------------------------------

- Output a warning in benders when the solver interface used to solve the
  master problem does not report a solution gap
- Document DAE transformations (#28)
- Allow use of ellipsis in the middle of the indices when slicing
- The Component slicing logic now uses simple slices (`:`) for matching a
  single index and the Ellipsis (`...`) for matching zero or more indices.
- Fixing Python 3 compatibility problems in pyomo.dae & dae examples.
- Support evaluating external functions in AMPL-stype shared libraries.
- Renaming "BasicExternalFunction" to "AMPLExternalFunction"
- Allowing Params to be used in setting bounds on a ContinuousSet.
- Use HTTPS with NEOS
- Allow a mutable Parameter that has not been initialized with a value to
  be used in the rhs of an equality constraint. (#13)
- Updates to SolverFactor to better support functionality offered by the
  'executable' keyword for SystemCallSolver.
- Updates to writers to enforce consistent output ordering in quadratic
  expressions.
- Improving the error message for Constraint rules that return None
- Adding PySP solver interface to the SD solver
- Adding an implicit SP representation to PySP that uses annotations
  for declaring mutable parameters as stochastic data and providing
  a distribution (only tables supported for now).
- Partial fix for infinite recursion reported (#6). This will support
  creating scalar (simple) components by passing a set([None]) in
  as the index.
- Extending component slicing ability to nested (hierarchical) slices.

-------------------------------------------------------------------------------
Pyomo 4.3 (4.3.11381)
-------------------------------------------------------------------------------

- Removed 'nose' from required packages during installation.
- Fix to be compatible with older versions of 'six'.

-------------------------------------------------------------------------------
Pyomo 4.3 (4.3.11377)
-------------------------------------------------------------------------------

- Restructured PySP tests to hide expected exception output
- Updated pyomo_install to support jython (experimental)
- Fixed a missing reference to 'six'
- More robust management of file I/O in GLPK interface

-------------------------------------------------------------------------------
Pyomo 4.3 (4.3.11345)
-------------------------------------------------------------------------------

- Various fixes to the NEOS solver plugin and to the SOL parser for
  certain NEOS solvers.
- Fixed a bug that caused an exception to occur when dynamically adding blocks
  to a ConcreteModel.
- Miscellaneous fixes to PySP scripting examples.

-------------------------------------------------------------------------------
Pyomo 4.3 (4.3.11328)
-------------------------------------------------------------------------------

- Misc bug fix

-------------------------------------------------------------------------------
Pyomo 4.3 (4.3.11327)
-------------------------------------------------------------------------------

- Resolved bug in pyomo_install when pyvenv cannot be found.
- Gracefully trap BARON terminations

-------------------------------------------------------------------------------
Pyomo 4.3 (4.3.11323)
-------------------------------------------------------------------------------

- Scripts
  - pyomo_install
    - Added --pip-options to specify options for PIP option
    - Added --index-url to specify the PyPI mirror
    - Restructured this script to make it more modular
    - Deprecated the --with-extras option
    - Added "--venv-only" option for setting up a local python virtual
      environment with no additional packages (apart from pip, setuptools, and
      potentially wheel).
    - Fixes when installing from zip files offline.
  - get-pyomo-extras.py
    - Misc fixes and updates
  - pyomo
    - Consolidated help information within the 'pyomo help' subcommand
    - Added 'pyomo install-extras' to install third-party dependencies
- pyomo.bilevel
  - Major changes to get the bilevel_ld solver working
  - Extended the logic in SubModel to allow for implicit declarations of
    upper-level variables.
- pyomo.check
  - More complete scoping checks
- pyomo.core
  - Changes to allow sorted output using DataPortal objects
  - Bug fix in linear dualization transformation
  - Adding a mechanism to load default configuration information from a
    standard location.
  - Changes to allow objectives to be used inside expressions, and to
    allow objectives to be sent through the solver plugins
  - Updating blocks so that models can be created and populated using rule options
  - Added a domain property method to IndexedVar so it cannot be treated as having
    a single domain
  - Added the Var() 'dense' option.  This defaults to True, which defines variables
    densely.  But sparse variables are defined when dense==False.
  - Change to define a SOS constraint over the declared index set for a variable
    unless an explicit index set is provided.
  - Changes to ensure that most components do not define their data densely.
- pyomo.dae
  - Updates to support Python3 and current Pyomo Transformation API
  - Fixed the collocation extension to reduce the degrees of freedom for certain variables
- pyomo.gdp
  - Many fixes to bilinear transformation
  - Fixed the Big-M estimation routine to work with constant expressions
- pyomo.neos
  - Added support for ephemeral solver options, which over-ride the
    options dictionary
  - Fixes to enable loading of solutions from NEOS solvers
- pyomo.opt
  - Sort variable names when printing results
  - Many fixes to sol reader.
- pyomo.pysp
  - Updates to ScenarioStructure.dat to handle a larger
    class of models that can have non-uniform variable sets
    in later time stages. Also updating parsing rules to
    allow for more convenient variable declarations
      - NodeVariables / NodeDerivedVariables can be used in
        place of StageVariables / StageDerivedVariables to
        declare variables on a per-node basis
      - Indexed variables can be declared without bracket
        notation (e.g., x[*,*] -> x)
      - Single or indexed block names can be used to
        conveniently declare all variables found on that
        block within a node or stage
  - Re-write of runef and runbenders based off of new PySP
    scripting tools
  - Adding option to runbenders for including one or more
    scenarios in the master benders problem
  - Fixes to csvsolutionwriter so that it works with
    distributed PH
  - Added new script (evaluate_xhat) that allows users to
    evaluate SP solutions on alternate scenario trees.
  - Added more extensive checking to PySP->SMPS conversion
    utility (pysp2smps) to alert users when conversion fails
    due to non-uniform problem structure or missing
    stochastic annotations
  - Added helper routine to convert a networkx directed
    graph into a PySP scenario tree model
  - Fixes to PH for better handling of user warmstarts
- pyomo.repn
  - Updates to the NL writer to handle new expression types.
  - Added an MPS writer
- pyomo.scripting
  - Updated the Pyro mip server to improve its efficiency
- pyomo.solvers
  - Updates to xpress plugin to use the ASL interface to xpress.
  - Fixed a major issue with the gap computation in the CPLEX direct/persistent
    solver plugins
  - Significant speed-up of the the CPLEX persistent plugin.
  - Added the 'mps' mode for various solvers.
  - Changes to the Pyro solver manager to allow out-of-order results collection

-------------------------------------------------------------------------------
Pyomo 4.2 (4.2.10784)
-------------------------------------------------------------------------------

- Changes to make the --json command-line option backwards compatible.

-------------------------------------------------------------------------------
Pyomo 4.2 (4.2.10782)
-------------------------------------------------------------------------------

- pyomo.core
  - Removed the 'initial' attribute from Var components.
  - Removed the 'reset()' method from core components.
  - Objective components now store sense for each objective.
  - Added support for slicing when indexing a component.
- pyomo.dae
  - Added methods to set and test the value of integral expressions
- pyomo.gdp
  - Added property methods to access disjunction expressions (e.g. lower, upper)
- pyomo.opt
  - Added support for ephemeral solver options that override the default
    values set in the solver object.
- pyomo.neos
  - Bug fixes to get data after calling NEOS
  - Resolving issues with python 3
- pyomo.scripting
  - Added --pyro-host and --pyro-port options
  - Added shutdown capabilities to pyro solver manager
  - Collect solve time for pyro solvers
- pyomo.pysp
  - Added --pyro-host and --pyro-port options to manage pyro-based execution
    in a more robust manner.
  - Adding utility for converting PySP models into SMPS input files (pysp2smps).
- pyomo.solver
  - Resolved serious issues with python direct interfaces:
    - CPLEX: constants in linear range constraints and all quadratic
      constraints were being excluded from the expression.
    - CPLEX and GUROBI: actually raise an exception when a nonlinear
      (non-quadratic) objective or constraint expression is encountered
      rather than just emitting the linear and quadratic part of the
      expression to the solver. This could lead to a naive user
      thinking they have solved a general nonlinear model with these
      solvers.
    - CPLEX and GUROBI: do not skip trivial constraints by default.
      Adding I/O option 'skip_trivial_constraints' to recover this
      behavior.
    - CPLEX: Merging as much of CPLEXPersistent with CPLEXDirect
      as possible to avoid repeating bug fixes. More should be done,
      but for now CPLEXDirect is a base class for CPLEXPersistent.
  - Fix to various solver plugins so that variable bounds set to
    float('inf') and float('-inf') are treated the same as variable
    bounds set to None.
  - Updates to Pyro solver managers
    - Major performance enhancements for Pyro-based solver managers.
      - Tasks can be uploaded to the dispatcher in bulk.
      - Workers no longer use a timeout when requesting tasks from the queue, which
        was wasting CPU cycles.
    - Compatibility fixes when Pyro4 is used.
- other
  - Updated the get-pyomo-extras.py script to install conditional dependencies
    for Pyomo (e.g. xlrd and ipython)
  - Adding logic to explicitly identify metasolvers.  This information is
    reflected in the 'pyomo help -s' command.
  - Deprecated 'pyomo.os', which was not being actively supported.
  - Added the 'pyomo info' subcommand to provide information about the Python
    installation.

-------------------------------------------------------------------------------
Pyomo 4.1 (4.1.10519)
-------------------------------------------------------------------------------

- Resolving bugs in NEOS solver interface due to change in solver options
  management.

-------------------------------------------------------------------------------
Pyomo 4.1 (4.1.10509)
-------------------------------------------------------------------------------

- Cleanup runbenders script to make it easier to test
- Cleanup temporary files during testing.
- Fixing PyUtilib version

-------------------------------------------------------------------------------
Pyomo 4.1 (4.1.10505)
-------------------------------------------------------------------------------

- Allow the dim() method to be called without the equality operator, which
  enables its use on unconstructed components
- RangeSet inherits from OrderedSimpleSet
- Added missing __slots__ declarations in set classes

-------------------------------------------------------------------------------
Pyomo 4.1 (4.1.10486)
-------------------------------------------------------------------------------

- API changes for model transformations
- Revised API for SOSConstraint, Suffix and Block components
- Optimization results are now loaded into models
- Removed explicit specification of model preprocessing
- Resolved many issues with writing and solving MPECs
- Changes to MPEC meta-solver names
- The solution output for runph has been changed to
- Pyomo subcommands can now use configuration files (e.g. pyomo solve config.json)
- New JSON/YAML format for parameter data
- Added a script to install pyomo.extras

-------------------------------------------------------------------------------
Pyomo 4.0 (4.0.9682)
-------------------------------------------------------------------------------

- Adding a more useful error message when a named objective is not
  present when load-solution is invoked on a model.
- Bug fix for pickling results objects.
- Performance fix for ordered sets
- Removed StateVar from DAE examples.
- Resolving a bug that disabled pyodbc support.
- Added preliminary support for pypyodbc.
- Fix to enable sets to be initialized with a tuple
- PySP test baseline updates
- Added PySP ddsip extension.
- Removed the 'pyomo bilevel' and 'pyomo mpec' subcommands
- Re-enabled the 'import' and 'export' Pyomo data commands, which
  are still deprecated.
- Various performance enhancements to avoid iterating over list of
  dictionary keys.
- Added a ComplementarityList component.

-------------------------------------------------------------------------------
Pyomo 4.0 (4.0.9629)
-------------------------------------------------------------------------------

This release rebrands Coopr as Pyomo, which reflects the fact that
users have consistently confused the Coopr software and the Pyomo
modeling language.  Pyomo 4.0 includes the following significant
changes:

- Pyomo provides a single source tree replacing all Coopr packages
- The 'pyomo' command replaces the 'coopr' command
- The 'pyomo solve' subcommand replaces the former 'pyomo' command
- The 'pyomo.environ' package is now used to import core Pyomo capabilities
- Robust support for Python 3.x

The following are highlights of this release:

- Modeling
   * Added a RealInterval domain
   * Major rework of coopr.dae. Can now represent higher order and partial
     differential equations. Also added more discretization schemes.

- Solvers
   * Added preliminary support for a Benders solver
   * Added support for the BARON solver
   * Preliminary support for MPEC solvers, including the PATH solver

- Transformations
   * Added explicit support for model transformations
   * The 'pyomo solve --transform' option specified model transformations
   * Created a streamline linear dual transformation

- Other
   * The 'pyomo help' command documents installed capabilities
   * Major rework and simplification of the 'pyomo_install' script
   * Added support for parallelism using Pyro4

-------------------------------------------------------------------------------
Pyomo 3.5.8787
-------------------------------------------------------------------------------

- pyomo.opt 2.12.2
  pyomo.core 3.6.4
  pyomo.pysp 3.5.5
  pyomo.solvers 3.2.1

-------------------------------------------------------------------------------
Pyomo 3.5.8748
-------------------------------------------------------------------------------

- pyomo.pysp 3.5.4

-------------------------------------------------------------------------------
Pyomo 3.5.8734
-------------------------------------------------------------------------------

- PyUtilib 4.7.3336

-------------------------------------------------------------------------------
Pyomo 3.5.8716
-------------------------------------------------------------------------------

- pyomo.core 3.6.3
  pyomo.pysp 3.5.3

-------------------------------------------------------------------------------
Pyomo 3.5.8706
-------------------------------------------------------------------------------

- pyomo.util 2.0.4
  pyomo.core 3.6.2

-------------------------------------------------------------------------------
Pyomo 3.5.8690
-------------------------------------------------------------------------------

- pyomo.neos 1.1.2
  pyomo.opt 2.12.1
  pyomo.core 3.6.1
  pyomo.pysp 3.5.2

- Added the pyomo_install script

-------------------------------------------------------------------------------
Pyomo 3.5.8669
-------------------------------------------------------------------------------

- PyUtilib 4.7.3311
  pyomo.util 2.0.3

-------------------------------------------------------------------------------
Pyomo 3.5.8663
-------------------------------------------------------------------------------

- PyUtilib 4.7.3305
  pyomo.util 2.0.2
  pyomo.environ 1.0.1

-------------------------------------------------------------------------------
Pyomo 3.5.8648
-------------------------------------------------------------------------------

- PyUtilib 4.7.3301
  pyomo.age 1.1.4
  pyomo.bilevel 1.0
  pyomo.util 2.0.1
  pyomo.dae 1.2
  pyomo.environ 1.0
  pyomo.gdp 1.2
  pyomo.util 2.8.2
  pyomo.mpec 1.0
  pyomo.neos 1.1.1
  pyomo.openopt 1.1.3
  pyomo.opt 2.12
  pyomo.os 1.0.4
  pyomo.core 3.6
  pyomo.pysos 2.0.9
  pyomo.pysp 3.5.1
  pyomo.solvers 3.2
  pyomo.sucasa 3.0

-------------------------------------------------------------------------------
Pyomo 3.4.7842
-------------------------------------------------------------------------------

- pyomo.dae 1.1
  pyomo.gdp 1.1.1
  pyomo.util 2.8.1
  pyomo.openopt 1.1.2
  pyomo.opt 2.11
  pyomo.os 1.0.3
  pyomo.plugins 3.1
  pyomo.core 3.5
  pyomo.pysp 3.4

-------------------------------------------------------------------------------
Pyomo 3.3.7114
-------------------------------------------------------------------------------

- pyomo.age 1.1.3
  pyomo.util 1.0.1
  pyomo.dae 1.0
  pyomo.gdp 1.1
  pyomo.util 2.7.2
  pyomo.openopt 1.1
  pyomo.opt 2.10
  pyomo.os 1.0.2
  pyomo.plugins 3.0
  pyomo.core 3.4
  pyomo.pysos 2.0.8
  pyomo.pysp 3.3
  pyomo.sucasa 2.1

-------------------------------------------------------------------------------
Pyomo 3.2.6148
-------------------------------------------------------------------------------

- pyomo.opt 2.9.1
  pyomo.core 3.3.2

-------------------------------------------------------------------------------
Pyomo 3.2.6124
-------------------------------------------------------------------------------

- pyomo.core 3.3.1

-------------------------------------------------------------------------------
Pyomo 3.2.6091
-------------------------------------------------------------------------------

- pyomo.gdp 1.0.4
  pyomo.openopt 1.0.3
  pyomo.opt 2.9
  pyomo.plugins 2.11
  pyomo.core 3.3
  pyomo.pysos 2.0.7
  pyomo.pysp 3.2
  pyomo.sucasa 2.0.6

-------------------------------------------------------------------------------
Pyomo 3.1.5746
-------------------------------------------------------------------------------

- pyomo.age 1.1.2
  pyomo.gdp 1.0.3
  pyomo.util 2.7
  pyomo.openopt 1.0.2
  pyomo.opt 2.8
  pyomo.os 1.0.1
  pyomo.plugins 2.9
  pyomo.core 3.1
  pyomo.pysp 3.1

-------------------------------------------------------------------------------
Pyomo 3.1.5409
-------------------------------------------------------------------------------

- Made the imports of pyomo.opt services more robust to the failure of
  individual services.

- Minor performance improvement

- Fixing import error when ordereddict is not available.

-------------------------------------------------------------------------------
Pyomo 3.1.5362
-------------------------------------------------------------------------------

- Bug fix for Python 2.7 installation.

-------------------------------------------------------------------------------
Pyomo 3.1.5325
-------------------------------------------------------------------------------

The following are highlights of this release:

- Solvers
   * Interfaces for OpenOpt solvers
   * Many solver interface improvements
   * A solver checker to validate solver interfaces
   * Improved support for SOS constraints (cplex, gurobi)
   * PH supports nonlinear models
   * PH-specific solver servers

- Modeling
   * Changes in rule semantics to limit rule return values
   * Changes in the expected order of rule arguments
   * Constant sums or products can now be used as constraint bounds
   * Added full support for the !ConstraintList modeling component.

- Usability enhancements
   * New 'pyomo' command has subcommands that consolidate Pyomo scripting
   * Added support to connect to databases with ODBC
   * Added comprehensive support for set expressions
   * Comprehensive rework of blocks and connectors for modular modeling
   * Made JSON the default results format

- Other
   * Efficiency improvements in model generation, memory, runtime, etc.
   * Preliminary support for black-box applications
   * Deprecated modeling syntax in Pyomo 3.0 is no longer legal


-------------------------------------------------------------------------------
Pyomo 3.0.4362
-------------------------------------------------------------------------------

- Solvers
   * More sophisticated logic for solver factory to find ASL and OS solvers
   * Various solver interface improvements
   * New Solver results object for efficient representation of variable values
   * New support for asynchronous progressive hedging

- Modeling
   * Changes in rule semantics to limit rule return values
   * Changes in the expected order of rule arguments
   * Constant sums or products can now be used as constraint bounds
   * Added full support for the ConstraintList modeling component.

- Usability enhancements
   * More explicit output from runph and runef commands
   * Added support in runef to write the extensive form in NL format
   * Add controls for garbage collection in PH

- Other
   * Efficiency improvements in generation of NL and LP files.
   * Significant efficiency improvements in parsing of Pyomo Data Files.
   * More robust MS Windows installer (does not use virtual python environment)

-------------------------------------------------------------------------------
Pyomo 2.5.3978
-------------------------------------------------------------------------------

- Performance improvements in Pyomo

- Bug fix when updating a results object that contains suffix data.

-------------------------------------------------------------------------------
Pyomo 2.5.3890
-------------------------------------------------------------------------------

- Solvers
    * MIP solver interface updates to use appropriate objective names
    * Added support for suffixes in GUROBI solver interface
    * Improved diagnostic analysis of PH solver for the extensive form

- Usability enhancements
    * Improved robustness of pyomo_install
    * Fixed Pyomo installation problem when using easy_install
    * Added a script to launch the PyomoAge GUI.
    * LP files now are written with the true objective name
    * Rework of pyomo command line to create a concise output
    * Many efficiency improvements during model generation!
    * Many improvements to diagnostic output and error handling
    * Expressions like "model.p > 1" can now be used within generation rules

- Modeling
    * Added support for generalized disjunctive programs (in pyomo.gdp)
    * Constraints can now be specified in "compound" form:  lb <= expr <= ub
    * Significant robustness enhancements for model expressions
    * Improved error handling for constraint generation

- Other
    * Python 2.5 is deprecated due to performance issues
    * Python versions 2.6 and 2.7 are supported
    * New MS Windows installer is now available

-------------------------------------------------------------------------------
Pyomo 2.4.3307
-------------------------------------------------------------------------------

- Solvers
  - Various fixes for Gurobi and CPLEX
  - Reorganized OS services in pyomo.os

- Usability enhancements
  - Improved robustness of pyomo_install
  - Default install of pyomo_install from PyPI

-------------------------------------------------------------------------------
Pyomo 2.4.3261
-------------------------------------------------------------------------------

- Updating dependencies for Pyomo.

-------------------------------------------------------------------------------
Pyomo 2.4.3209
-------------------------------------------------------------------------------

- Patch fix for pyomo.colin

-------------------------------------------------------------------------------
Pyomo 2.4.3199
-------------------------------------------------------------------------------

- Modeling
  - Concrete models are now supported
  - Nonlinear modeling extensions using the AMPL NL problem format
  - Add support for immutable versus mutable parameters.
  - Support for SOS1 and SOS2 constraints

- Data Integration
  - Can now import data from relational databases

- Solvers
  - Better support for Gurobi solver
  - Direct CPLEX solver interface
  - Interface to ipopt and nonlinear solvers via asl (just to be clear)
  - ASL solver interface can now be specified with the form
        --solver=asl:PICO

- Usability enhancements
  - Numerous bug fixes.
  - Updated messages to provide clearer indication of modeling errors

-------------------------------------------------------------------------------
Pyomo 2.3.2581
-------------------------------------------------------------------------------

- A preliminary Gurobi solver interface

- Extended syntax for data command files:
  'include' command to load other data command files
  'import' command to load data from other sources
  namespaces to next data declarations

- The pyomo_install script can install packages from Coin Bazaar

- New conversion scripts to generate LP or NL files from Pyomo models

- Solvers now extract standard suffix information

- Various fixes to PySP solvers

-------------------------------------------------------------------------------
Pyomo 2.0
-------------------------------------------------------------------------------

- Reorganization of Pyomo into separate packages:

  - pyomo.opt 2.0
  - pyomo.util 2.0
  - pyomo.plugins 2.0
  - pyomo.core 2.0
  - pyomo.pysos 2.0
  - pyomo.sucasa 2.0

-------------------------------------------------------------------------------
Pyomo 1.2
-------------------------------------------------------------------------------

- OPT

  - Added explicit support for a symbol_map, which is used to coordinate
    the symbols used in a converted problem with the symbols used in the
    original problem.

- PYOMO

  - Fixed bug in logic associated with index validation in ProductSets.

  - Changed the set ord() method to be 1-based

  - Added presolve diagnostics.

- SUCASA

  - Added draft SUCASA user manual.

- PYSP

  - Modified PH to support parallelism with Pyro.

  - Introduced the PH scripts, with many options that an end-user
    might want to try out.

  - Added a convergence criterion to PH based on the # of free
    discrete variables.

  - Flushed out PYSP 1.1 documentation.

  - Updated/improved timing reports and output in PH.

  - Setup binary quadratic term linearization.

  - Setup PH checkpointing using pickle.

  - Added 48-scenario test case for forestry problem.

  - Added PH option to specify a user-defined PH extension.

  - Fixes to PH proximal term linearization.

  - Created windows-friendly versions of core PYSP scripts.

- Plugins

  - Switched default CBC input file format to CPLEXLP

- General

  - Added a Pyomo 'getting started' manual.

  - Reorganized Pyomo package to use the 'pyomo' namespace.  The pyomo.util
    package was renamed pyomo.opt.

  - Added documentation on how to create/install plugins in Pyomo.

  - Added documentation for using pyomo_install.

  - Reworked pyomo_install to be created with the pyutilib.virtualenv script
    vpy_create.

  - Misc edits due to the PyUtilib reorg.

-------------------------------------------------------------------------------
Pyomo 1.1
-------------------------------------------------------------------------------

- OPT

  - Reorganized pyomo.opt to rely on plugins for most of its core
    functionality.

  - Changed the default output for PICO to be LP format, rather than
    .NL. This avoids issues with name-mapping that we haven't resolved yet.

  - Updated LP writer to not output integer/binary status for variables
    that aren't referenced in the model.

  - In LP output format, modified constraint names to be "suffixed"
    with the index in the same fashion as variables. The suffix is "None"
    if the constraint is a singleton. Will help debug models during
    development/prototyping.

  - For MIPs, added "slack" suffix to solutions and modified CPLEX solver
    interface to populate the suffix accordingly.

  - Improvement of the factory mechanism used to launch optimizers;
    solvers can be passed options during construction.

  - The CPLEX LP format assumes default variable bounds equal to 0 and
    +inf. These conflict with the more sane defaults in Pyomo, which are
    -inf and +inf. This leads to all kinds of silent, incorrect behavior
    in terms of getting very strange solutions where you are expecting a
    straightforward solve. The cpxlp writer was changed to always output
    bounds for variables, which is the safest route in any case.

  - Added a facility for managing asynchronous events. In particular,
    this facility has been setup to support the application of Pyomo
    solvers with subclasses of the AsynchronousSolverManager.

  - Created a distributed solver manager that uses the Pyro
    RPC package.

  - Rework of MIP solver interfaces for CPLEX, GLPK, PICO and CBC.

  - Using an explicit temporary file when launching the shell command
    subprocess, to avoid a buffer overflow.

  - A rework of the logic in shellcmd.py to segregate the solution
    into preprocess/solve/postprocess. This facilitates a fine-grain
    parallelization of just the IP solve, using Pyro.

  - If a variable name has the format x(1,a) or x[3,4,5] then create a
    dictionary attribute 'x' in the SolverResults object, which maps the
    tuple values to the corresponding value.

    For example:

      results.solution().variable.add('x(1,a)')
      print results.solution().variable.x[1,'a']

  - A change in the converter semantics. Now, the convert returns a
    tuple of filenames. In most cases, this is a singleton tuple. But
    in some important cases this is a non-singleton tuple (e.g. AMPL
    *.mod/*.dat files).


- PYOMO

  - Reworked Pyomo's management of components to use plugins.

  - Adding two new components BuildCheck and BuildAction. Their
    usage is similar, but their expected use is a it
    different. BuildAction  allows for the injection of arbitrary build
    actions during construction of the model, while BuildCheck is used
    to test conditions and generate exceptions when the build process
    goes awry.

  - There is also a subtle change to the Param component. The following
    is now legal:

        def f(model):
            return 1.3
        model.x = Param(initialize=f)

    That is, when the Param object does not have an explicit index,
    then no index is passed to an initializer function (as is expected).

  - Adding 'summation', a function that computes multi-vector
    product sums:

       sum_i x[i]*y[i]*z[i]

  - Adding automatic computation of variable bounds based on the domain.
    If the domain set supports simple bounds, then they will be used to
    initialize the bounds of a variable.

  - Adding logic to ignore the generation of constraints if the
    constructor rule returns either 'None' or '0'.

  - Changed default domain of a Param to be Any, to be consistent with Set.

  - Rework of Pyomo to enable preprocessor actions to manage all
    post-instance-generation activities. This includes a simple preprocessor
    plugin, which simply applies preprocessor actions in an order specified
    by the action-specific ranks.  The problem writers are no longer
    responsible for how preprocessor actions take place.

    NOTE: These preprocessors are very much tailored to the needs of the
    NL and LP writers. We need to figure out a mechanism to tailoring
    preprocessors to specific target solvers (e.g. tailoring preprocessing
    for the NL writer, or tailoring preprocessing for MILP models).

  _ The Constraint and Objective classes were reworked to integrate
    their data into a single _data dictionary. This simplified
    the management of id and label information.

  - Extending the Constraint(model.A, rule=f) constructor semantics. This
    change allows the rule to return a dictionary when the constraint is
    indexed by one or more sets.

  - A revision to Pyomo semantics. Now, expressions are not evaluated
    when performing arithmetic operations (plus, times, etc).

  - A major rework of how component attributes are managed for
    NumericValue  objects and subclasses of this class. This was driven
    by the desire to add the Var.declare_attribute() method, which declares
    attributes for variables (which are akin to ampl suffix's).

    Most of these changes were motivated by inconsistencies that were
    observed within Pyomo, and the desire to better protect declared
    attributes. Now, attributes are stored with names that are unlikely
    to be used by end-users or other Pyomo developers. Also, declared
    attributes can now only be referenced directly. Thus, you cannot
    refer to x._value, but instead you must use x.value.

  - Reworked the way that 'empty' objectives and constraints are dealt
    with. By default, objectives and constraints with expr/body
    values of None are provided, which facilitates some aspects of
    preprocessing. But this triggered exceptions, which are now disabled.

  - Reworked how the '_equality' constraint attribute is managed.  This is
    now in the ConstraintData class, which required various changes.

  - Changed the pprint output for equality constraints to make it clear
    that both the upper and lower bounds are equal for these constraints.

  - Fixed bug in the definition of parameter names. When using a rule
    to define parameter values, the parameter name now reflex the index
    values for the parameter.

  - Major change of NumericalValue subclass semantics. The 'value' data
    member is not supported for expressions, constraints and objectives.
    These classes need to use the call method.  This simplifies the
    logic in their getattr/setattr methods.

  - Various code optimizations to improve the runtime performance of Pyomo.

  - A major rework of the Pyomo core to eliminates the use
    of getattr and setattr methods. Removing these led to a 1/3 reduction
    in runtime for some largish p-median test problems.  This change
    has had the following impact on Pyomo functionality:

    . No validation that values assigned to parameters or variables are valid

    . Cannot set all suffixes simultaneously

    . Cannot set lower/upper bounds simultaneously

    . No validation that suffixes are valid when their value is set

    Also, this effort led to a rework of the Param and Var classes. These
    are now segregated between Element and Array classes (similar to the
    Set class). This led to further optimizations of the logic in these
    classes. Unfortunately, it also led to the use of the _VarBase and
    _ParamBase in the PyomoModel to index components.

  - Changed order in which parameters are validated. Validation needs
    to occur after parameters are set. This is due to the semantics of
    the validation function: we want to allow the validation function to
    refer to the value as if it were set.

  - Deprecating the use of the expression factory for algebraic expression
    types. These are now launched directly from the generate_expression()
    function.

  - Adding support for specifying options when launching solvers. For example:

    results = self.pico.solve(currdir+"bell3a.mps", options="maxCPUMinutes=0.1")

  - The value 'None' is accepted by all NumValue objects, and this
    indicates that the object has not been initialized.

  - ParamValue objects print their name, rather than their value.

  - Resolving an issue loading boolean data from a *.dat file.  Now,
    all true/false strings are changed to True/False values.

  - Rework of Pyomo to use the SolverManagerFactory. The default
    behavior is to use the 'serial' manager, which does local solves,
    but this supports extensibility for other solvers.


- SUCASA

  - Added an option to terminate SUCASA after the AMPL script is
    generated, but before PICO is called. This allows the AMPL script
    to be applied separately.

  - Rework of the files generated by SUCASA. Before, SUCASA generated
    three files:

    . app_milp.h/app_milp.cpp Define the derived MILP classes
    . app_extras.cpp Used to define the methods specialized by the user

    Now, SUCASA generates five files:

    . app_sucasa.h/app_sucasa.cpp Define the derived MILP classes
    . app_milp.h/app_milp.cpp Define a MILP from the code in *_sucasa.* files.
    . app_extras.cpp Used to define the methods specialized by the user

    The result, is that SUCASA now generates two derived solvers. BUT, the
    one that is exposed to the user can be customized without impacting
    the integration of the Info classes. Further, this segregation
    simplifies the class definitions that a user looks at; for example,
    there are no ugly ifdefs, and no references to the Info data.

  - A rework of the SUCASA API that is exported to the user. This new
    API allows users to register the vector of primal/dual values,
    which are access implicitly through the methods

      <name>_value()

  - Rework of the SUCASA Info API to ensure a consistent interface
    for sets, parameters, vars, etc.

  - Rework of the AMPL parser to guess the superset types for sets and
    parameters. This works on most common cases, but it's far from perfect.

  - Extended code generation to include parameter data.

  - Update of examples.


- PYSP

  - Initial integration of the Python Stochastic Programming (PySP) package.

  - Developed a variety of stochastic programming examples to illustrate
    the use of PYSP

  - Developed a PYSP user guide.


- General

  - Resolved Python 3.0 portability issues

  - Created a script create_pyomo_install, which uses virtualenv to
    automatically create the pyomo_install script, which automates the
    installation of pyomo.


-------------------------------------------------------------------------------
Pyomo 1.0
-------------------------------------------------------------------------------

- Initial release.
