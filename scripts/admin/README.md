# Admin Scripts

--------

## Contributors Script

The `contributors.py` script is intended to be used to determine contributors
to a public GitHub repository within a given time frame.

### Requirements

1. Python 3.9+
1. [PyGithub](https://pypi.org/project/PyGithub/)
1. A [GitHub Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) with `repo` access, exported to the environment variable `GH_TOKEN`

### Usage

```
Usage: contributors.py <Org/Repo> <start_date> <end_date>
   <Org/Repo>   : the GitHub organization/repository combo (e.g., Pyomo/pyomo)
   <start_date> : date from which to start exploring contributors in YYYY-MM-DD
   <end_date>   : date at which to stop exploring contributors in YYYY-MM-DD

ALSO REQUIRED: Please generate a GitHub token (with repo permissions) and export to the environment variable GH_TOKEN.
   Visit GitHub's official documentation for more details.
```

### Results

A list of contributors will print to the terminal upon completion. More detailed
information, including authors, committers, reviewers, and pull requests, can
be found in the `contributors-start_date-end_date.json` generated file. 


----------

## Big Wheel of Misfortune

The `bwom.py` script is intended to be used during weekly Dev Calls to generate
a list of random open issues so developers can more proactively review issues
in the backlog.

### Requirements

1. Python 3.10+
1. [PyGithub](https://pypi.org/project/PyGithub/)
1. A [GitHub Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) with `repo` access, exported to the environment variable `GH_TOKEN`

### Usage

```
Usage: bwom.py <Org/Repo> [num_issues]
   <Org/Repo>   : the GitHub organization/repository combo (e.g., Pyomo/pyomo)
   [Number of issues] : optional number of random open issues to return (default is 5)

ALSO REQUIRED: Please generate a GitHub token (with repo permissions) and export to the environment variable GH_TOKEN.
   Visit GitHub's official documentation for more details.
```

### Results

A list of `n` random open issues (default is 5) on the target repository.
This list includes the issue number, title, and URL. For example:

```
Randomly selected open issues from Pyomo/pyomo:
- Issue #2087: Add Installation Environment Test (URL: https://github.com/Pyomo/pyomo/issues/2087)
- Issue #1310: Pynumero.sparse transpose (URL: https://github.com/Pyomo/pyomo/issues/1310)
- Issue #2218: cyipopt does not support `symbolic_solver_labels` or `load_solutions=False` (URL: https://github.com/Pyomo/pyomo/issues/2218)
- Issue #2123: k_aug interface in Pyomo sensitivity toolbox reports wrong answer (URL: https://github.com/Pyomo/pyomo/issues/2123)
- Issue #1761: slow quadratic constraint creation (URL: https://github.com/Pyomo/pyomo/issues/1761)
```
