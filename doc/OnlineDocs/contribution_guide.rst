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

The Pyomo master branch (as of `this commit <https://github.com/Pyomo/pyomo/commit/49e2ff171ddcd083c62ac28379afcf33af2549ae>`) provides a Github Action
workflow that will test any changes pushed to a branch using Ubuntu with
Python 3.7. For existing forks, fetch and merge your fork (and branches) with
Pyomo's master. For new forks, you will need to enable Github Actions
in the 'Actions' tab on your fork. Then the test will begin to run
automatically with each push to your fork.

At any point in the development cycle, a "work in progress" pull request
may be opened by including '[WIP]' at the beginning of the PR
title. This allows your code changes to be tested by Pyomo's automatic
testing infrastructure. Any pull requests marked '[WIP]' will not be
reviewed or merged by the core development team. In addition, any
'[WIP]' pull request left open for an extended period of time without
active development may be marked 'stale' and closed.

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
      you should see a button that says "Branch: master". Click on
      this button, and choose the correct branch.
    * Click the "New pull request" button just to the right of the
      "Branch: <branch_name>" button.
    * Fill out the pull request template and click the green "Create
      pull request" button.

At times during your development, you may want to merge changes from
the Pyomo master development branch into the feature branch on your
fork and in your local clone of the repository.

Using GitHub UI to merge Pyomo master into a branch on your fork
****************************************************************

To update your fork, you will actually be merging a pull-request from
the main Pyomo repository into your fork.

    * Visit https://github.com/Pyomo/pyomo.
    * Click on the "New pull request" button just above the list of
      files and directories.
    * You will see the title "Compare changes" with some small text
      below it which says "Compare changes across branches, commits,
      tags, and more below. If you need to, you can also compare
      across forks." Click the last part of this: "compare across
      forks".
    * You should now see four buttons just below this: "base
      repository: Pyomo/pyomo", "base: master", "head repository:
      Pyomo/pyomo", and "compare: master". Click the leftmost button
      and choose "<username>/Pyomo".
    * Then click the button which is second to the left, and choose
      the branch which you want to merge Pyomo master into. The four
      buttons should now read: "base repository: <username>/pyomo",
      "base: <branch_name>", "head repository: Pyomo/pyomo", and
      "compare: master". This is setting you up to merge a pull-request
      from Pyomo's master branch into your fork's <branch_name> branch.
    * You should also now see a pull request template. If you fill out
      the pull request template and click "Create pull request", this
      will create a pull request which will update your fork and
      branch with any changes that have been made to the master branch
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
two remotes, one for your fork, and one for the main Pyomo repository.

::
   
   git clone https://github.com/<username>/pyomo.git
   git remote rename origin my-fork
   git remote add main-pyomo https://github.com/pyomo/pyomo.git

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
   git fetch main-pyomo
   git merge main-pyomo/<branch_to_update_from> --ff-only

The "--ff-only" only allows a merge if the merge can be done by a
fast-forward. If you do not require a fast-forward, you can drop this
option. The most common concrete example of this would be

::

   git checkout master
   git fetch main-pyomo
   git merge main-pyomo/master --ff-only

The above commands pull changes from the master branch of the main
Pyomo repository into the master branch of your local clone. To push
these changes to the master branch on your fork,

::

   git push my-fork master
   

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

* ``pyomo.contrib`` can provide wrappers for separate Python packages, thereby allowing these packages to be imported as subpackages of pyomo.

* ``pyomo.contrib`` can include contributed packages that are developed and maintained outside of the Pyomo developer team.  

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

The following two examples illustrate the two ways
that ``pyomo.contrib`` can be used to integrate third-party
contributions.

Including External Packages
+++++++++++++++++++++++++++

The `pyomocontrib_simplemodel
<http://pyomocontrib-simplemodel.readthedocs.io/en/latest/>`_ package
is derived from Pyomo, and it defines the class SimpleModel that
illustrates how Pyomo can be used in a simple, less object-oriented
manner. Specifically, this class mimics the modeling style supported
by `PuLP <https://github.com/coin-or/pulp>`_.

While ``pyomocontrib_simplemodel`` can be installed and used separate
from Pyomo, this package is included in ``pyomo/contrib/simplemodel``.
This allows this package to be referenced as if were defined as a
subpackage of ``pyomo.contrib``.  For example::

    from pyomo.contrib.simplemodel import *
    from math import pi

    m = SimpleModel()

    r = m.var('r', bounds=(0,None))
    h = m.var('h', bounds=(0,None))

    m += 2*pi*r*(r + h)
    m += pi*h*r**2 == 355

    status = m.solve("ipopt")

This example illustrates that a package can be distributed separate
from Pyomo while appearing to be included in the ``pyomo.contrib``
subpackage.  Pyomo requires a separate directory be defined under
``pyomo/contrib`` for each such package, and the Pyomo developer
team will approve the inclusion of third-party packages in this
manner.


Contrib Packages within Pyomo
+++++++++++++++++++++++++++++

Third-party contributions can also be included directly within the
``pyomo.contrib`` package.  The ``pyomo/contrib/example`` package
provides an example of how this can be done, including a directory
for plugins and package tests.  For example, this package can be
imported as a subpackage of ``pyomo.contrib``::

    from pyomo.environ import *
    from pyomo.contrib.example import a

    # Print the value of 'a' defined by this package
    print(a)

Although ``pyomo.contrib.example`` is included in the Pyomo source
tree, it is treated as an optional package.  Pyomo will attempt to
import this package, but if an import failure occurs, Pyomo will
silently ignore it.  Otherwise, this pyomo package will be treated
like any other.  Specifically:

* Plugin classes defined in this package are loaded when `pyomo.environ` is loaded.

* Tests in this package are run with other Pyomo tests.

