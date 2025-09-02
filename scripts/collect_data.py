#!/usr/bin/env python3
"""
Collect data from starred repositories and save to JSON.
Fetches: repo name, description, topics, language, README content
"""
import os, sys, json, time, pathlib
from slugify import slugify
from github import Github, GithubException

TOKEN = os.getenv("GH_TOKEN")
USER = os.getenv("GH_USER")

if not (TOKEN or USER):
    sys.exit("Set GH_USER or GH_TOKEN to identify account")

gh = Github(TOKEN or None, per_page=100)
user = gh.get_user() if TOKEN else gh.get_user(USER)

repos_data = []
for repo in user.get_starred():
    print(f"Processing {repo.full_name}...")
    
    # get basic repo info
    repo_info = {
        "full_name": repo.full_name,
        "name": repo.name,
        "description": repo.description or "",
        "language": repo.language or "",
        "topics": [],
        "readme": "",
        "url": repo.html_url
    }
    
    # get topics
    try:
        topics = repo.get_topics()
        repo_info["topics"] = topics
    except GithubException as exc:
        if exc.status == 403:
            print(f"  Skip topics for {repo.full_name}: 403 from org policy")
        else:
            print(f"  Error getting topics: {exc}")
    
    # get README content (first 2000 chars to avoid huge files)
    try:
        readme = repo.get_readme()
        content = readme.decoded_content.decode('utf-8')
        repo_info["readme"] = content[:2000] if len(content) > 2000 else content
    except GithubException:
        print(f"  No README found for {repo.full_name}")
    except Exception as e:
        print(f"  Error reading README: {e}")
    
    repos_data.append(repo_info)
    time.sleep(0.1)  # rate limiting

# save to JSON
output_file = pathlib.Path("starred_repos_data.json")
with output_file.open("w", encoding="utf-8") as fh:
    json.dump(repos_data, fh, indent=2, ensure_ascii=False)

print(f"Collected data for {len(repos_data)} repositories")
print(f"Saved to {output_file}")