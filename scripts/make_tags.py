#!/usr/bin/env python3
"""
Fetch your starred repos and build a tag table:
repo_full_name | tag1, tag2, ...
"""
import os, sys, pathlib, json, time
from slugify import slugify
from github import Github

TOKEN = os.getenv("GH_TOKEN")         # set in workflow secrets

gh   = Github(TOKEN, per_page=100)
user = gh.get_user()

rows = []
for repo in user.get_starred():
    try:
        topics = repo.get_topics()         # may raise 403 on some orgs
    except GithubException as exc:
        if exc.status == 403:
            print(f"Skip {repo.full_name}: 403 from org policy")
            topics = []                    # fall back to language + owner tag
        else:
            raise

    tags = {slugify(t) for t in topics}
    if repo.language:
        tags.add(slugify(repo.language))
    tags.add(slugify(repo.owner.login))

    rows.append((repo.full_name, ", ".join(sorted(tags))))
    time.sleep(0.1)

# write markdown table
out = pathlib.Path("starred_repos_tags.md")
with out.open("w", encoding="utf-8") as fh:
    fh.write("| Repository | Tags |\n|---|---|\n")
    for name, tag_str in rows:
        fh.write(f"| {name} | {tag_str} |\n")

print(f"Wrote {out} with {len(rows)} rows")
