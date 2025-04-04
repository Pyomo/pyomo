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
   black -S -C <path> --exclude examples/pyomobook/python-ch/BadIndent.py
   # Find typos in files
   conda install typos
   typos --config .github/workflows/typos.toml <path>
   
If the spell-checker returns a failure for a word that is spelled correctly,
please add the word to the ``.github/workflows/typos.toml`` file.

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

  python setup.py develop

These commands register the cloned code with the active python environment
(``pyomodev``). This way, your changes to the source code for ``pyomo`` are
automatically used by the active environment. You can create another conda
environment to switch to alternate versions of pyomo (e.g., stable).

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

``pyomo.contrib``
-----------------

Pyomo uses the ``pyomo.contrib`` package to facilitate the inclusion
of third-party contributions that enhance Pyomo's core functionality.
The are two ways that ``pyomo.contrib`` can be used to integrate
third-party packages:

* ``pyomo.contrib`` can provide wrappers for separate Python packages, thereby
   allowing these packages to be imported as subpackages of pyomo.

* ``pyomo.contrib`` can include contributed packages that are developed and
   maintained outside of the Pyomo developer team.  

Including contrib packages in the Pyomo source tree provides a
convenient mechanism for defining new functionality that can be
optionally deployed by users.  We expect this mechanism to include
Pyomo extensions and experimental modeling capabilities.  However,
contrib packages are treated as optional packages, which are not
maintained by the Pyomo developer team.  Thus, it is the responsibility
of the code contributor to keep these packages up-to-date.

Contrib package contributions will be considered as pull-requests,
which will be reviewed by the Pyomo developer team.  Specifically,
this review will consider the suitability of the proposed capability,
whether tests are available to check the execution of the code, and
whether documentation is available to describe the capability.
Contrib packages will be tested along with Pyomo.  If test failures
arise, then these packages will be disabled and an issue will be
created to resolve these test failures.

Contrib Packages within Pyomo
+++++++++++++++++++++++++++++

Third-party contributions can be included directly within the
``pyomo.contrib`` package.  The ``pyomo/contrib/example`` package
provides an example of how this can be done, including a directory
for plugins and package tests.  For example, this package can be
imported as a subpackage of ``pyomo.contrib``::

    import pyomo.environ as pyo
    from pyomo.contrib.example import a

    # Print the value of 'a' defined by this package
    print(a)

Although ``pyomo.contrib.example`` is included in the Pyomo source
tree, it is treated as an optional package.  Pyomo will attempt to
import this package, but if an import failure occurs, Pyomo will
silently ignore it.  Otherwise, this pyomo package will be treated
like any other.  Specifically:

* Plugin classes defined in this package are loaded when ``pyomo.environ`` is loaded.

* Tests in this package are run with other Pyomo tests.

