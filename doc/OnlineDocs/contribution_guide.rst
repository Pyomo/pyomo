Contributing to Pyomo
=====================

We welcome all contributions including bug fixes, feature enhancements,
and documentation improvements. Pyomo manages source code contributions
via GitHub pull requests (PRs). 

Contribution Requirements
-------------------------

A PR should be 1 set of related changes. PRs for large-scale
non-functional changes (i.e. PEP8, comments) should be
separated from functional changes. This simplifies the review process
and ensures that functional changes aren't obscured by large amounts of
non-functional changes.

Coding Standards
++++++++++++++++
    
    * Required: 4 space indentation (no tabs)
    * Desired: PEP8
    * No use of __author__ 
    * Inside ``pyomo.contrib``: Contact information for the contribution
      maintainer (such as a Github ID) should be included in the Sphinx
      documentation

Sphinx-compliant documentation is required for:
    
    * Modules
    * Public and Private Classes
    * Public and Private Functions 

We also encourage you to include examples, especially for new features
and contributions to ``pyomo.contrib``.

Testing
+++++++

Pyomo uses ``unittest``, TravisCI, and Appveyor for testing and
continuous integration. Submitted code should include tests to establish
the validity of its results and/or effects. Unit tests are preferred but
we also accept integration tests. When test are run on a PR, we require
at least 70% coverage of the lines modified in the PR and prefer
coverage closer to 90%. We also require that all tests pass before a PR
will be merged.

At any point in the development cycle, a "work in progress" pull request
may be opened by including '[WIP]' at the beginning of the PR
title. This allows your code changes to be tested by Pyomo's automatic
testing infrastructure. Any pull requests marked '[WIP]' will not be
reviewed or merged by the core development team. In addition, any
'[WIP]' pull request left open for an extended period of time without
active development may be marked 'stale' and closed.

Review Process
--------------

After a PR is opened it will be reviewed by at least two members of the
core development team. The core development team consists of anyone with
write-access to the Pyomo repository. Pull requests opened by a core
developer only require one review. The reviewers will decide if they
think a PR should be merged or if more changes are necessary.

Reviewers look for:
    
    * Outside of ``pyomo.contrib``: Code rigor and standards, edge cases,
      side effects, etc.
    * Inside of ``pyomo.contrib``: No “glaringly obvious” problems with
      the code
    * Documentation and tests

The core development team tries to review pull requests in a timely
manner but we make no guarantees on review timeframes. In addition, PRs
might not be reviewed in the order they are opened in. 

Where to put contributed code 
----------------------------- 

In order to contribute to Pyomo, you must first make a fork of the Pyomo
git repository. Next, you should create a branch on your fork dedicated
to the development of the new feature or bug fix you're interested
in. Once you have this branch checked out, you can start coding. Bug
fixes and minor enhancements to existing Pyomo functionality should be
made in the appropriate files in the Pyomo code base. New examples,
features, and packages built on Pyomo should be placed in
``pyomo.contrib``. Follow the link below to find out if
``pyomo.contrib`` is right for your code.

.. toctree::
   :maxdepth: 1

   pyomo_contrib.rst
