#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
This script is intended to query the GitHub REST API and get a list of open
issues for a given repository, returning a random selection of n issues.

We colloquially call this the "Big Wheel of Misfortune" (BWOM)
"""

import sys
import random
import os

from github import Github, Auth


def get_random_open_issues(repository, number_to_return):
    """
    Return a random selection of open issues from a repository.

    Parameters
    ----------
    repository : String
        The org/repo combination for target repository (GitHub). E.g.,
        IDAES/idaes-pse.
    number_to_return : int
        The number of random open issues to return.

    Returns
    -------
    random_issues : List
        A list of dictionaries containing information about randomly selected open issues.
    """
    # Collect the authorization token from the user's environment
    token = os.environ.get('GH_TOKEN')
    auth_token = Auth.Token(token)
    # Create a connection to GitHub
    gh = Github(auth=auth_token)
    # Create a repository object for the requested repository
    repo = gh.get_repo(repository)
    # Get all open issues
    open_issues = repo.get_issues(state='open')
    open_issues_list = [issue for issue in open_issues if "pull" not in issue.html_url]

    # Randomly select the specified number of issues
    random_issues = random.sample(
        open_issues_list, min(number_to_return, len(open_issues_list))
    )

    return random_issues


def print_big_wheel():
    """Prints a specified ASCII art representation of a big wheel."""
    wheel = [
        "        .                  __",
        "       / \\             . ' || ' .",
        "       )J(          .`     ||     `.",
        "      (8)7)       .   \\    ||    /   .",
        "       (')     .'/ _   \\ .-''-. /   _ \\",
        "       (=)   .' J   `- .' .--. '. -`   L",
        "       (') .'   F======' ((<>)) '======J",
        "      )J('     L      '. `||' .'      F",
        "      (7(8)      \\  _.-  `-||-'  -._  /",
        "       \\'        .     /  ||  \\     .",
        "      / |           .  /   ||   \\  .",
        "     /  |             ` . _||_ . `",
        "    /   |___________    _.-||_________",
        "  (()\\.'|   ___.....''''   ||._      .'",
        "  \\.`- .'.                /__\\/    .'|",
        ".'_______________________________.' ||",
        "  |'---------------------------'|==.||",
        "  ||.' ||                      ||.' ||",
        "  ||===========================||  (__)",
        "  ||                           ||",
        " (__)                     LGB (__)",
        " Credit: ascii.co.uk/art/spinningwheel",
    ]
    for line in wheel:
        print(line)


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} <Org/Repo> [num_issues]")
        print(
            "   <Org/Repo>   : the GitHub organization/repository combo (e.g., Pyomo/pyomo)"
        )
        print(
            "   [Number of issues] : optional number of random open issues to return (default is 5)"
        )
        print("")
        print(
            "ALSO REQUIRED: Please generate a GitHub token (with repo permissions) "
            "and export to the environment variable GH_TOKEN."
        )
        print("   Visit GitHub's official documentation for more details.")
        sys.exit(1)

    repository = sys.argv[1]
    num_issues = 5
    if len(sys.argv) == 3:
        try:
            num_issues = int(sys.argv[2])
            if num_issues <= 0:
                raise (ValueError("Need a positive number; why did you try <= 0?"))
        except ValueError as e:
            print(
                "*** ERROR: You did something weird when declaring the number of issues. Defaulting to 5.\n"
                f"(For posterity, this is the error that was returned: {e})\n"
            )

    print("Spinning the Big Wheel of Misfortune...\n")
    print_big_wheel()

    random_issues = get_random_open_issues(repository, num_issues)

    print(f"\nRandomly selected open issues from {repository}:")
    for issue in random_issues:
        print(f"- Issue #{issue.number}: {issue.title} (URL: {issue.html_url})")


if __name__ == '__main__':
    main()
