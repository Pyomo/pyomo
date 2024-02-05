# Contributors Script

The `contributors.py` script is intended to be used to determine contributors
to a public GitHub repository within a given time frame.

## Requirements

1. Python 3.7+
1. [PyGithub](https://pypi.org/project/PyGithub/)
1. A [GitHub Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) with `repo` access, exported to the environment variable `GH_TOKEN`

## Usage

```
Usage: contributors.py <Org/Repo> <start_date> <end_date>
   <Org/Repo>   : the GitHub organization/repository combo (e.g., Pyomo/pyomo)
   <start_date> : date from which to start exploring contributors in YYYY-MM-DD
   <end_date>   : date at which to stop exploring contributors in YYYY-MM-DD

ALSO REQUIRED: Please generate a GitHub token (with repo permissions) and export to the environment variable GH_TOKEN.
   Visit GitHub's official documentation for more details.
```

## Results

A list of contributors will print to the terminal upon completion. More detailed
information, including authors, committers, reviewers, and pull requests, can
be found in the `contributors-start_date-end_date.json` generated file. 
