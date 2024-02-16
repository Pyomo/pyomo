#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
This script is intended to query the GitHub REST API and get contributor
information for a given time period.
"""

import sys
import re
import json
import os

from datetime import datetime
from os import environ
from time import perf_counter
from github import Github, Auth


def collect_contributors(repository, start_date, end_date):
    """
    Return contributor information for a repository in a given timeframe

    Parameters
    ----------
    repository : String
        The org/repo combination for target repository (GitHub). E.g.,
        IDAES/idaes-pse
    start_date : String
        Start date in YYYY-MM-DD.
    end_date : String
        End date in YYYY-MM-DD.

    Returns
    -------
    contributor_information : Dict
        A dictionary with contributor information including Authors, Reviewers,
        Committers, and Pull Requests.
    tag_name_map : Dict
        A dictionary that maps GitHub handles to GitHub display names (if they
        exist).
    only_tag_available : List
        A list of the handles for contributors who do not have GitHub display names
        available.

    """
    # Create data structure
    contributor_information = {}
    contributor_information['Pull Requests'] = {}
    contributor_information['Authors'] = {}
    contributor_information['Reviewers'] = {}
    contributor_information['Commits'] = {}
    # Collect the authorization token from the user's environment
    token = environ.get('GH_TOKEN')
    auth_token = Auth.Token(token)
    # Create a connection to GitHub
    gh = Github(auth=auth_token)
    # Create a repository object for the requested repository
    repo = gh.get_repo(repository)
    commits = repo.get_commits(since=start_date, until=end_date)
    # Search the commits between the two dates for those that match the string;
    # this is the default pull request merge message. This works assuming that
    # a repo does not squash commits
    merged_prs = [
        int(
            commit.commit.message.replace('Merge pull request #', '').split(' from ')[0]
        )
        for commit in commits
        if commit.commit.message.startswith("Merge pull request")
    ]
    # If the search above returned nothing, it's likely that the repo squashes
    # commits when merging PRs. This is a different regex for that case.
    if not merged_prs:
        regex_pattern = '\(#.*\)'
        for commit in commits:
            results = re.search(regex_pattern, commit.commit.message)
            try:
                merged_prs.append(int(results.group().replace('(#', '').split(')')[0]))
            except AttributeError:
                continue
    # Count the number of commits from each person within the two dates
    for commit in commits:
        try:
            if commit.author.login in contributor_information['Commits'].keys():
                contributor_information['Commits'][commit.author.login] += 1
            else:
                contributor_information['Commits'][commit.author.login] = 1
        except AttributeError:
            # Sometimes GitHub returns an author who doesn't have a handle,
            # which seems impossible but happens. In that case, we just record
            # their "human-readable" name
            if commit.commit.author.name in contributor_information['Commits'].keys():
                contributor_information['Commits'][commit.commit.author.name] += 1
            else:
                contributor_information['Commits'][commit.commit.author.name] = 1

    author_tags = set()
    reviewer_tags = set()
    for num in merged_prs:
        try:
            # sometimes the commit messages can lie and give a PR number
            # for a different repository fork/branch.
            # We try to query it, and if it doesn't work, whatever, move on.
            pr = repo.get_pull(num)
        except:
            continue
        # Sometimes the user does not have a handle recorded by GitHub.
        # In this case, we replace it with "NOTFOUND" so the person running
        # the code knows to go inspect it manually.
        author_tag = pr.user.login
        if author_tag is None:
            author_tag = "NOTFOUND"
        # Count the number of PRs authored by each person
        if author_tag in author_tags:
            contributor_information['Authors'][author_tag] += 1
        else:
            contributor_information['Authors'][author_tag] = 1
        author_tags.add(author_tag)

        # Now we inspect all of the reviews to see who engaged in reviewing
        # this specific PR
        reviews = pr.get_reviews()
        review_tags = set(review.user.login for review in reviews)
        # Count how many PRs this person has reviewed
        for tag in review_tags:
            if tag in reviewer_tags:
                contributor_information['Reviewers'][tag] += 1
            else:
                contributor_information['Reviewers'][tag] = 1
        reviewer_tags.update(review_tags)
        contributor_information['Pull Requests'][num] = {
            'author': author_tag,
            'reviewers': review_tags,
        }
    # This portion replaces tags with human-readable names, if they are present,
    # so as to remove the step of "Who does that handle belong to?"
    all_tags = author_tags.union(reviewer_tags)
    tag_name_map = {}
    only_tag_available = []
    for tag in all_tags:
        if tag in tag_name_map.keys():
            continue
        name = gh.search_users(tag + ' in:login')[0].name
        # If they don't have a name listed, just keep the tag
        if name is not None:
            tag_name_map[tag] = name
        else:
            only_tag_available.append(tag)
    for key in tag_name_map.keys():
        if key in contributor_information['Authors'].keys():
            contributor_information['Authors'][tag_name_map[key]] = (
                contributor_information['Authors'].pop(key)
            )
        if key in contributor_information['Reviewers'].keys():
            contributor_information['Reviewers'][tag_name_map[key]] = (
                contributor_information['Reviewers'].pop(key)
            )
    return contributor_information, tag_name_map, only_tag_available


def set_default(obj):
    """
    Converts sets to list for JSON dump
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <Org/Repo> <start_date> <end_date>")
        print(
            "   <Org/Repo>   : the GitHub organization/repository combo (e.g., Pyomo/pyomo)"
        )
        print(
            "   <start_date> : date from which to start exploring contributors in YYYY-MM-DD"
        )
        print(
            "   <end_date>   : date at which to stop exploring contributors in YYYY-MM-DD"
        )
        print("")
        print(
            "ALSO REQUIRED: Please generate a GitHub token (with repo permissions) and export to the environment variable GH_TOKEN."
        )
        print("   Visit GitHub's official documentation for more details.")
        sys.exit(1)
    repository = sys.argv[1]
    repository_name = sys.argv[1].split('/')[1]
    try:
        start = sys.argv[2].split('-')
        year = int(start[0])
        try:
            month = int(start[1])
        except SyntaxError:
            month = int(start[1][1])
        try:
            day = int(start[2])
        except SyntaxError:
            day = int(start[2][1])
        start_date = datetime(year, month, day)
    except:
        print("Ensure that the start date is in YYYY-MM-DD format.")
        sys.exit(1)
    try:
        end = sys.argv[3].split('-')
        year = int(end[0])
        try:
            month = int(end[1])
        except SyntaxError:
            month = int(end[1][1])
        try:
            day = int(end[2])
        except SyntaxError:
            day = int(end[2][1])
        end_date = datetime(year, month, day)
    except:
        print("Ensure that the end date is in YYYY-MM-DD format.")
        sys.exit(1)
    json_filename = f"contributors-{repository_name}-{sys.argv[2]}-{sys.argv[3]}.json"
    if os.path.isfile(json_filename):
        raise FileExistsError(f'ERROR: The file {json_filename} already exists!')
    print('BEGIN DATA COLLECTION... (this can take some time)')
    tic = perf_counter()
    contrib_info, author_name_map, tags_only = collect_contributors(
        repository, start_date, end_date
    )
    toc = perf_counter()
    print(f"\nCOLLECTION COMPLETE. Time to completion: {toc - tic:0.4f} seconds")
    print(f"\nContributors between {sys.argv[2]} and {sys.argv[3]}:")
    for item in author_name_map.values():
        print(item)
    print("\nOnly GitHub handles are available for the following contributors:")
    for tag in tags_only:
        print(tag)
    with open(json_filename, 'w') as file:
        json.dump(contrib_info, file, default=set_default)
    print(f"\nDetailed information can be found in {json_filename}.")
