#!/usr/bin/env python3
"""
Fetch your starred repos and build a tag table:
repo_full_name | tag1, tag2, ...
"""
import os, sys, pathlib, json, time
from slugify import slugify
from github import Github, GithubException

TOKEN = os.getenv("GH_TOKEN")         # optional GitHub token
USER  = os.getenv("GH_USER")          # username for unauth'd requests

if not (TOKEN or USER):
    sys.exit("Set GH_USER or GH_TOKEN to identify account")

gh = Github(TOKEN or None, per_page=100)
user = gh.get_user() if TOKEN else gh.get_user(USER)

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

# semantic clustering using repository names and descriptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# collect repository info for semantic analysis
repo_data = []
repo_names = []
for name, tag_str in rows:
    # combine repo name and tags for clustering
    text = name.replace("/", " ").replace("-", " ").replace("_", " ") + " " + tag_str.replace(",", " ")
    repo_data.append(text)
    repo_names.append(name)

# create TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=100, stop_words='english', lowercase=True)
tfidf_matrix = vectorizer.fit_transform(repo_data)

# determine optimal number of clusters (max 8, min 2)
n_repos = len(repo_data)
n_clusters = min(8, max(2, n_repos // 3))

# perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

# generate cluster names based on most important terms
feature_names = vectorizer.get_feature_names_out()
cluster_names = {}
for i in range(n_clusters):
    # get centroid for this cluster
    centroid = kmeans.cluster_centers_[i]
    # get top 3 terms with highest weights
    top_indices = centroid.argsort()[-3:][::-1]
    top_terms = [feature_names[idx] for idx in top_indices]
    cluster_names[i] = " & ".join(top_terms).title()

# group repositories by clusters
categorized = {}
for i, (name, _) in enumerate(rows):
    cluster_id = cluster_labels[i]
    cluster_name = cluster_names[cluster_id]
    
    if cluster_name not in categorized:
        categorized[cluster_name] = []
    categorized[cluster_name].append(name)

# prepare output data
output_clusters = {}

# add topic clusters
for cluster_name, repos in categorized.items():
    if repos:
        output_clusters[cluster_name] = [name for name, _ in sorted(repos)]

# add language groups
for lang, repos in sorted(by_language.items()):
    if repos:
        output_clusters[f"{lang} Projects"] = [name for name, _ in sorted(repos)]

# add uncategorized
if uncategorized:
    output_clusters["Other"] = [name for name, _ in sorted(uncategorized)]

# write JSON output
json_out = pathlib.Path("starred_repos_clusters.json")
with json_out.open("w", encoding="utf-8") as fh:
    json.dump(output_clusters, fh, indent=2)

# write markdown output
md_out = pathlib.Path("starred_repos_tags.md")
with md_out.open("w", encoding="utf-8") as fh:
    fh.write("# Starred Repositories (Auto-Clustered)\n\n")
    
    for topic, repos in output_clusters.items():
        fh.write(f"## {topic}\n\n")
        for repo in repos:
            fh.write(f"- {repo}\n")
        fh.write("\n")

print(f"Wrote {out} with {len(rows)} rows")
