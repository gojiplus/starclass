#!/usr/bin/env python3
"""
Fetch starred repositories list and track changes incrementally.
Maintains starred_repos_list.json with repo metadata for change detection.
"""
import os, sys, json, pathlib, time
from datetime import datetime
from github import Github, GithubException

TOKEN = os.getenv("GH_TOKEN")
USER = os.getenv("GH_USER")

if not (TOKEN or USER):
    sys.exit("Set GH_USER or GH_TOKEN to identify account")

gh = Github(TOKEN or None, per_page=100)
user = gh.get_user() if TOKEN else gh.get_user(USER)

# load existing starred repos list
list_file = pathlib.Path("starred_repos_list.json")
existing_repos = {}
if list_file.exists():
    with list_file.open("r", encoding="utf-8") as fh:
        existing_repos = json.load(fh)

# fetch current starred repos
current_repos = {}
new_repos = []
updated_repos = []

print("Fetching starred repositories...")
for repo in user.get_starred():
    repo_key = repo.full_name
    repo_info = {
        "full_name": repo.full_name,
        "updated_at": repo.updated_at.isoformat(),
        "starred_at": datetime.now().isoformat(),  # when we first saw it starred
        "language": repo.language,
        "description": repo.description or ""
    }
    
    current_repos[repo_key] = repo_info
    
    # check if new or updated
    if repo_key not in existing_repos:
        new_repos.append(repo_key)
        print(f"  NEW: {repo_key}")
    elif existing_repos[repo_key]["updated_at"] != repo_info["updated_at"]:
        updated_repos.append(repo_key)
        print(f"  UPDATED: {repo_key}")
    
    time.sleep(0.05)  # gentle rate limiting

# find removed repos
removed_repos = set(existing_repos.keys()) - set(current_repos.keys())
if removed_repos:
    for repo_key in removed_repos:
        print(f"  REMOVED: {repo_key}")

# save updated list
with list_file.open("w", encoding="utf-8") as fh:
    json.dump(current_repos, fh, indent=2)

# save change summary
changes = {
    "timestamp": datetime.now().isoformat(),
    "total_starred": len(current_repos),
    "new_repos": new_repos,
    "updated_repos": updated_repos,
    "removed_repos": list(removed_repos)
}

changes_file = pathlib.Path("starred_repos_changes.json")
with changes_file.open("w", encoding="utf-8") as fh:
    json.dump(changes, fh, indent=2)

print(f"\nSummary:")
print(f"  Total starred: {len(current_repos)}")
print(f"  New: {len(new_repos)}")
print(f"  Updated: {len(updated_repos)}")
print(f"  Removed: {len(removed_repos)}")
print(f"Saved to {list_file} and {changes_file}")