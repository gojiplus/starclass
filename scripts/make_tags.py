#!/usr/bin/env python3
"""
Fetch your starred repos and build a tag table:
repo_full_name | tag1, tag2, ...
"""
import os, sys, pathlib, json, time
from slugify import slugify
from github import Github

TOKEN = os.getenv("GH_TOKEN")         # set in workflow secrets

if not TOKEN or not USER:
    sys.exit("Missing GH_TOKEN or GH_USER env vars")

gh   = Github(TOKEN, per_page=100)
user = gh.get_user()

rows = []
for repo in user.get_starred():
    topics = repo.get_topics()              # list[str]
    lang   = repo.language or ""
    owner  = repo.owner.login

    # tiny taxonomy rule‑set
    tags   = {slugify(t) for t in topics}
    if lang:  tags.add(slugify(lang))
    tags.add(slugify(owner))

    rows.append((repo.full_name, ", ".join(sorted(tags))))
    time.sleep(0.1)                         # stay polite (<90 req/min)

# write markdown table
out = pathlib.Path("starred_repos_tags.md")
with out.open("w", encoding="utf-8") as fh:
    fh.write("| Repository | Tags |\n|---|---|\n")
    for name, tag_str in rows:
        fh.write(f"| {name} | {tag_str} |\n")

print(f"Wrote {out} with {len(rows)} rows")
