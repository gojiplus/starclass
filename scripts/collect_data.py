#!/usr/bin/env python3
"""
Incrementally collect detailed data from starred repositories.
Only fetches data for new/updated repos based on starred_repos_changes.json.
"""
import os, sys, json, time, pathlib
from slugify import slugify
from github import Github, GithubException

TOKEN = os.getenv("GH_TOKEN")
USER = os.getenv("GH_USER")

if not (TOKEN or USER):
    sys.exit("Set GH_USER or GH_TOKEN to identify account")

gh = Github(TOKEN or None, per_page=100)

# load existing data
data_file = pathlib.Path("starred_repos_data.json")
existing_data = {}
if data_file.exists():
    with data_file.open("r", encoding="utf-8") as fh:
        existing_list = json.load(fh)
        existing_data = {repo["full_name"]: repo for repo in existing_list}

# load changes to know what to update
changes_file = pathlib.Path("starred_repos_changes.json")
if not changes_file.exists():
    print("No changes file found. Run fetch_starred.py first.")
    sys.exit(1)

with changes_file.open("r", encoding="utf-8") as fh:
    changes = json.load(fh)

# repos that need data collection
repos_to_process = set(changes["new_repos"] + changes["updated_repos"])

if not repos_to_process:
    print("No new or updated repositories to process.")
    # still save existing data in case file was missing
    repos_data = list(existing_data.values())
else:
    print(f"Processing {len(repos_to_process)} repositories...")
    
    for repo_name in repos_to_process:
        print(f"Processing {repo_name}...")
        
        try:
            repo = gh.get_repo(repo_name)
            
            # get basic repo info
            repo_info = {
                "full_name": repo.full_name,
                "name": repo.name,
                "description": repo.description or "",
                "language": repo.language or "",
                "topics": [],
                "readme": "",
                "url": repo.html_url,
                "last_updated": repo.updated_at.isoformat()
            }
            
            # get topics (handle 403s gracefully)
            try:
                topics = repo.get_topics()
                repo_info["topics"] = topics
            except GithubException as exc:
                if exc.status == 403:
                    print(f"  Skip topics for {repo.full_name}: 403")
                    repo_info["topics"] = []
                else:
                    print(f"  Error getting topics: {exc}")
                    repo_info["topics"] = []
            
            # get README content (handle 403s gracefully)
            try:
                readme = repo.get_readme()
                content = readme.decoded_content.decode('utf-8')
                repo_info["readme"] = content[:2000] if len(content) > 2000 else content
            except GithubException as exc:
                if exc.status == 403:
                    print(f"  Skip README for {repo.full_name}: 403")
                elif exc.status == 404:
                    print(f"  No README found for {repo.full_name}")
                else:
                    print(f"  Error reading README: {exc}")
            except Exception as e:
                print(f"  Error reading README: {e}")
            
            existing_data[repo_name] = repo_info
            time.sleep(0.2)  # increased delay for rate limiting
            
        except Exception as e:
            print(f"  Error processing {repo_name}: {e}")

    # remove repos that were unstarred
    for removed_repo in changes["removed_repos"]:
        existing_data.pop(removed_repo, None)
        print(f"Removed {removed_repo} from data")

    repos_data = list(existing_data.values())

# save updated data
with data_file.open("w", encoding="utf-8") as fh:
    json.dump(repos_data, fh, indent=2, ensure_ascii=False)

print(f"Data collection complete. Total repositories: {len(repos_data)}")
print(f"Saved to {data_file}")