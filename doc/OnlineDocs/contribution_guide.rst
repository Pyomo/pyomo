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

We do not squash and merge PRs so all commits in your branch will appear 
in the main history. In addition to well-documented PR descriptions,
we encourage modular/targeted commits with descriptive commit messages.

Coding Standards
++++++++++++++++
    
    * Required: `black <https://black.readthedocs.io/en/stable/>`_
    * No use of ``__author__`` 
    * Inside ``pyomo.contrib``: Contact information for the contribution
      maintainer (such as a Github ID) should be included in the Sphinx
      documentation

The first step of Pyomo's GitHub Actions workflow is to run
`black <https://black.readthedocs.io/en/stable/>`_ and a
`spell-checker <https://github.com/crate-ci/typos>`_ to ensure style
guide compliance and minimize typos. Before opening a pull request, please
run:

::

    # Auto-apply correct formatting
   pip install black
   black <path>
   # Find typos in files
   conda install typos
   typos --config .github/workflows/typos.toml <path>
   
If the spell-checker returns a failure for a word that is spelled
correctly, please add the word to the ``.github/workflows/typos.toml``
file. Note also that ``black`` reads from ``pyproject.toml`` to
determine correct configuration, so if you are running ``black``
indirectly (for example, using an IDE integration), please ensure you
are not overriding the project-level configuration set in that file.

Online Pyomo documentation is generated using `Sphinx <https://www.sphinx-doc.org/en/master/>`_
with the ``napoleon`` extension enabled. For API documentation we use of one of these 
`supported styles for docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_, 
but we prefer the NumPy standard. Whichever you choose, we require compliant docstrings for:
    
    * Modules
    * Public and Private Classes
    * Public and Private Functions

We also encourage you to include examples, especially for new features
and contributions to ``pyomo.contrib``.

Testing
+++++++

Pyomo uses `unittest <https://docs.python.org/3/library/unittest.html>`_,
`pytest <https://docs.pytest.org/>`_,
`GitHub Actions <https://docs.github.com/en/free-pro-team@latest/actions>`_,
and Jenkins
for testing and continuous integration. Submitted code should include 
tests to establish the validity of its results and/or effects. Unit 
tests are preferred but we also accept integration tests. We require 
at least 70% coverage of the lines modified in the PR and prefer coverage 
closer to 90%. We also require that all tests pass before a PR will be 
merged.

Tests must import the Pyomo test harness from
``pyomo.common.unittest`` instead of using Python's built-in
``unittest`` module directly. This wrapper extends the standard testing
framework with Pyomo-specific capabilities, including automatic plugin
discovery, solver availability checks, and improved test filtering
controls. Using the provided interface ensures that all tests run
consistently across Pyomo's multiple CI environments.
A small example is shown below:

.. code-block:: python

   import pyomo.common.unittest as unittest

   class TestSomething(unittest.TestCase):
       def test_basic(self):
           self.assertEqual(1 + 1, 2)

Developers can also use any of the predefined ``pytest`` markers to categorize
their tests appropriately.
Markers are declared in ``pyproject.toml``. Some commonly used markers are:

- ``expensive``: tests that take a long time to run
- ``mpi``: tests that require MPI
- ``solver(name)``: dynamic marker to label a test for a specific solver,
  e.g., ``@pytest.mark.solver("gurobi")``

.. note::
   If you are having issues getting tests to pass on your Pull Request,
   please tag any of the core developers to ask for help.

The Pyomo main branch provides a Github Actions workflow (configured
in the ``.github/`` directory) that will test any changes pushed to
a branch with a subset of the complete test harness that includes
multiple virtual machines (``ubuntu``, ``mac-os``, ``windows``)
and multiple Python versions. For existing forks, fetch and merge
your fork (and branches) with Pyomo's main. For new forks, you will
need to enable GitHub Actions in the 'Actions' tab on your fork.
This will enable the tests to run automatically with each push to your fork.

At any point in the development cycle, a "work in progress" pull request
may be opened by including '[WIP]' at the beginning of the PR
title. Any pull requests marked '[WIP]' or draft will not be
reviewed or merged by the core development team. However, any
'[WIP]' pull request left open for an extended period of time without
active development may be marked 'stale' and closed.

.. note::
   Draft and WIP Pull Requests will **NOT** trigger tests. This is an effort to
   reduce our CI backlog. Please make use of the provided
   branch test suite for evaluating / testing draft functionality.

Python Version Support
++++++++++++++++++++++

By policy, Pyomo supports and tests the currently supported Python versions,
as can be seen on `Status of Python Versions <https://devguide.python.org/versions/>`_.
It is expected that tests will pass for all of the supported and tested
versions of Python, unless otherwise stated.

At the time of the first Pyomo release after the end-of-life of a minor Python
version, we will remove testing and support for that Python version.

This will also result in a bump in the minor Pyomo version.

For example, assume Python 3.A is declared end-of-life while Pyomo is on
version 6.3.Y. After the release of Pyomo 6.3.(Y+1), Python 3.A will be removed,
and the next Pyomo release will be 6.4.0.

Working on Forks and Branches
-----------------------------

All Pyomo development should be done on forks of the Pyomo
repository. In order to fork the Pyomo repository, visit
https://github.com/Pyomo/pyomo, click the "Fork" button in the
upper right corner, and follow the instructions.

This section discusses two recommended workflows for contributing
pull-requests to Pyomo. The first workflow, labeled
:ref:`Working with my fork and the GitHub Online UI <forksgithubui>`,
does not require the use of 'remotes', and
suggests updating your fork using the GitHub online UI. The second
workflow, labeled
:ref:`Working with remotes and the git command-line <forksremotes>`, outlines
a process that defines separate remotes for your fork and the main
Pyomo repository.

More information on git can be found at
https://git-scm.com/book/en/v2. Section 2.5 has information on working
with remotes.


.. _forksgithubui:

Working with my fork and the GitHub Online UI
+++++++++++++++++++++++++++++++++++++++++++++

After creating your fork (per the instructions above), you can
then clone your fork of the repository with

::

   git clone https://github.com/<username>/pyomo.git

For new development, we strongly recommend working on feature
branches. When you have a new feature to implement, create
the branch with the following.

::

   cd pyomo/     # to make sure you are in the folder managed by git
   git branch <branch_name>
   git checkout <branch_name>

Development can now be performed. When you are ready, commit
any changes you make to your local repository. This can be
done multiple times with informative commit messages for
different tasks in the feature development.

::

   git add <filename>
   git status  # to check that you have added the correct files
   git commit -m 'informative commit message to describe changes'

In order to push the changes in your local branch to a branch on your fork, use

::

   git push origin <branch_name>


When you have completed all the changes and are ready for a pull request, make
sure all the changes have been pushed to the branch <branch_name> on your fork.

    * visit https://github.com/<username>/pyomo.
    * Just above the list of files and directories in the repository,
      you should see a button that says "Branch: main". Click on
      this button, and choose the correct branch.
    * Click the "New pull request" button just to the right of the
      "Branch: <branch_name>" button.
    * Fill out the pull request template and click the green "Create
      pull request" button.

At times during your development, you may want to merge changes from
the Pyomo main development branch into the feature branch on your
fork and in your local clone of the repository.

Using GitHub UI to merge Pyomo main into a branch on your fork
****************************************************************

To update your fork, you will actually be merging a pull-request from
the head Pyomo repository into your fork.

    * Visit https://github.com/Pyomo/pyomo.
    * Click on the "New pull request" button just above the list of
      files and directories.
    * You will see the title "Compare changes" with some small text
      below it which says "Compare changes across branches, commits,
      tags, and more below. If you need to, you can also compare
      across forks." Click the last part of this: "compare across
      forks".
    * You should now see four buttons just below this: "base
      repository: Pyomo/pyomo", "base: main", "head repository:
      Pyomo/pyomo", and "compare: main". Click the leftmost button
      and choose "<username>/Pyomo".
    * Then click the button which is second to the left, and choose
      the branch which you want to merge Pyomo main into. The four
      buttons should now read: "base repository: <username>/pyomo",
      "base: <branch_name>", "head repository: Pyomo/pyomo", and
      "compare: main". This is setting you up to merge a pull-request
      from Pyomo's main branch into your fork's <branch_name> branch.
    * You should also now see a pull request template. If you fill out
      the pull request template and click "Create pull request", this
      will create a pull request which will update your fork and
      branch with any changes that have been made to the main branch
      of Pyomo.
    * You can then merge the pull request by clicking the green "Merge
      pull request" button from your fork on GitHub.

.. _forksremotes:

Working with remotes and the git command-line
+++++++++++++++++++++++++++++++++++++++++++++

After you have created your fork, you can clone the fork and setup
git 'remotes' that allow you to merge changes from (and to) different
remote repositories. Below, we have included a set of recommendations,
but, of course, there are other valid GitHub workflows that you can
adopt.

The following commands show how to clone your fork and setup
two remotes, one for your fork, and one for the head Pyomo repository.

::
   
   git clone https://github.com/<username>/pyomo.git
   git remote rename origin my-fork
   git remote add head-pyomo https://github.com/pyomo/pyomo.git

Note, you can see a list of your remotes with

::

   git remote -v

The commands for creating a local branch and performing local commits
are the same as those listed in the previous section above. Below are
some common tasks based on this multi-remote setup.

If you have changes that have been committed to a local feature branch
(<branch_name>), you can push these changes to the branch on your fork
with,

::

   git push my-fork <branch_name>

In order to update a local branch with changes from a branch of the
Pyomo repository,

::

   git checkout <branch_to_update>
   git fetch head-pyomo
   git merge head-pyomo/<branch_to_update_from> --ff-only

The "--ff-only" only allows a merge if the merge can be done by a
fast-forward. If you do not require a fast-forward, you can drop this
option. The most common concrete example of this would be

::

   git checkout main
   git fetch head-pyomo
   git merge head-pyomo/main --ff-only

The above commands pull changes from the main branch of the head
Pyomo repository into the main branch of your local clone. To push
these changes to the main branch on your fork,

::

   git push my-fork main


Setting up your development environment
+++++++++++++++++++++++++++++++++++++++

After cloning your fork, you will want to install Pyomo from source.

Step 1 (recommended): Create a new ``conda`` environment.

::

   conda create --name pyomodev

You may change the environment name from ``pyomodev`` as you see fit.
Then activate the environment:

::
   
   conda activate pyomodev

Step 2 (optional): Install PyUtilib

The hard dependency on PyUtilib was removed in Pyomo 6.0.0. There is still a
soft dependency for any code related to ``pyomo.dataportal.plugins.sheet``.

If your contribution requires PyUtilib, you will likely need the main branch of
PyUtilib to contribute. Clone a copy of the repository in a new directory:

::

   git clone https://github.com/PyUtilib/pyutilib

Then in the directory containing the clone of PyUtilib run:

::

   python setup.py develop
   
Step 3: Install Pyomo

Finally, move to the directory containing the clone of your Pyomo fork and run:

::

  pip install -e .[tests,docs,optional]

These commands register the cloned code with the active python environment
(``pyomodev``). This way, your changes to the source code for ``pyomo`` are
automatically used by the active environment. You can create another conda
environment to switch to alternate versions of pyomo (e.g., stable).

Review Process
--------------

After a PR is opened it will be reviewed by at least two members of the
core development team. The core development team consists of anyone with
write-access to the Pyomo repository. PRs opened by a core
developer only require one review. The reviewers will decide if they
think a PR should be merged or if more changes are necessary.

Reviewers look for:

* **Core and Addons:** Code rigor, standards compliance, appropriate test
  coverage, and avoidance of unintended side effects
* **Devel:** Basic code correctness and clarity, with an understanding that
  these areas are experimental and evolving
* **All areas:** Reasonable documentation and tests

.. note::

   For more information about Pyomo's development principles and the
   stability expectations for ``addons`` and ``devel``, see
   :doc:`/principles`.

The core development team tries to review PRs in a timely
manner, but we make no guarantees on review timeframes. In addition, PRs
might not be reviewed in the order in which they are opened.


Where to put contributed code 
----------------------------- 

In order to contribute to Pyomo, you must first make a fork of the Pyomo
git repository. Next, you should create a branch on your fork dedicated
to the development of the new feature or bug fix you're interested
in. Once you have this branch checked out, you can start coding. Bug
fixes and minor enhancements to existing Pyomo functionality should be
made in the appropriate files in the Pyomo code base.

Larger features, new modeling components, or experimental functionality
should be placed in one of Pyomo's extension namespaces, described below.

Namespaces for Contributed and Experimental Code
++++++++++++++++++++++++++++++++++++++++++++++++

Pyomo has a long history of supporting community-developed extensions.
Historically, all such contributions were placed under the
``pyomo.contrib`` namespace. This structure allowed new modeling tools and
algorithms to be shared quickly, but over time it became difficult for users
to distinguish between more stable, supported functionality and
experimental or research-oriented code.

As a result, Pyomo is transitioning to a more structured contribution
model with two clear namespaces:

* ``pyomo.addons`` – For mostly stable, supported extensions that build on
  the Pyomo core. These packages are maintained by dedicated
  contributors, follow Pyomo's coding and testing standards, and adhere
  to the same deprecation policies as the rest of the codebase.

* ``pyomo.devel`` – For experimental or rapidly evolving
  contributions. These modules serve as early experimentation for research ideas,
  prototypes, or specialized modeling components. Functionality under
  this namespace may change or be removed between releases without
  deprecation warnings.

This two-tiered structure provides contributors a clear pathway from
**experimentation to supported integration**, while protecting users from
unexpected changes in stable areas of the codebase.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Namespace
     - Intended Use
     - Stability
   * - ``pyomo.devel``
     - Active research and experimental code
     - Unstable; APIs may change without warning
   * - ``pyomo.addons``
     - Mostly stable, supported extensions maintained by contributors
     - Most stable APIs; follow Pyomo's standards
   * - ``pyomo``
     - Core Pyomo modeling framework
     - Fully supported and versioned

For specific inclusion requirements and maintenance expectations for each
namespace, see:

* :doc:`../../pyomo/addons/README`
* :doc:`../../pyomo/devel/README`

Submitting a Contributed Package
--------------------------------

Including contributed packages in the Pyomo source tree provides a
convenient mechanism for defining new functionality that can be
optionally deployed by users. We expect this mechanism to include
Pyomo extensions and experimental modeling capabilities.  However,
contributed packages are treated as optional packages, which are not
maintained by the Pyomo developer team. Thus, it is the responsibility
of the code contributor to keep these packages up-to-date.

Contributed package will be considered as pull requests,
which will be reviewed by the Pyomo developer team. Specifically,
this review will consider the suitability of the proposed capability,
whether tests are available to check the execution of the code, and
whether documentation is available to describe the capability.
Contributed packages will be tested along with Pyomo. If test failures
arise, then these packages will be disabled and an issue will be
created to resolve these test failures.

When submitting a new contributed package (under either ``addons`` or
``devel``), please ensure that:

* The package has at least one maintainer responsible for its upkeep.
* The code includes tests that can be run through Pyomo's
  continuous integration framework.
* The package includes a README or module-level documentation that
  clearly describes its purpose and usage.
* Optional dependencies are properly declared in ``setup.py``
  under the appropriate ``[optional]`` section.
* The contribution passes all standard style and formatting checks.

Contributed Packages within Pyomo
+++++++++++++++++++++++++++++++++

Third-party contributions can be included directly within the
``pyomo.devel`` or ``pyomo.addons`` packages.
The ``pyomo/devel/example`` package
provides an example of how this can be done, including a directory
for plugins and package tests. For example, this package can be
imported as a subpackage of ``pyomo.devel``::

    import pyomo.environ as pyo
    from pyomo.devel.example import a

    # Print the value of 'a' defined by this package
    print(a)

Although ``pyomo.devel.example`` is included in the Pyomo source
tree, it is treated as an optional package.  Pyomo will attempt to
import this package, but if an import failure occurs, Pyomo will
silently ignore it.  Otherwise, this pyomo package will be treated
like any other.  Specifically:

* Plugin classes defined in this package are loaded when ``pyomo.environ`` is loaded.

* Tests in this package are run with other Pyomo tests.
