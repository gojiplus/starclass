#!/usr/bin/env python3
"""
Cluster starred repositories using semantic similarity.
Reads from starred_repos_data.json and outputs clusters.
"""
import json, pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# load repository data
data_file = pathlib.Path("starred_repos_data.json")
if not data_file.exists():
    print("Error: starred_repos_data.json not found. Run collect_data.py first.")
    exit(1)

with data_file.open("r", encoding="utf-8") as fh:
    repos_data = json.load(fh)

print(f"Loaded data for {len(repos_data)} repositories")

# prepare text for clustering
repo_texts = []
repo_names = []

for repo in repos_data:
    # combine multiple text sources for semantic analysis
    text_parts = [
        repo["name"].replace("-", " ").replace("_", " "),
        repo["description"],
        " ".join(repo["topics"]),
        repo["readme"][:500]  # first 500 chars of README
    ]
    
    combined_text = " ".join(filter(None, text_parts))
    repo_texts.append(combined_text)
    repo_names.append(repo["full_name"])

# create TF-IDF vectors
vectorizer = TfidfVectorizer(
    max_features=200, 
    stop_words='english', 
    lowercase=True,
    min_df=1,  # include all terms
    ngram_range=(1, 2)  # include bigrams for better context
)
tfidf_matrix = vectorizer.fit_transform(repo_texts)

# determine optimal number of clusters
n_repos = len(repos_data)
n_clusters = min(10, max(3, n_repos // 4))

print(f"Clustering {n_repos} repositories into {n_clusters} groups...")

# perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

# generate meaningful cluster names
feature_names = vectorizer.get_feature_names_out()
cluster_info = {}

for i in range(n_clusters):
    # get repos in this cluster
    cluster_repos = [repo_names[j] for j, label in enumerate(cluster_labels) if label == i]
    
    # get centroid and top terms
    centroid = kmeans.cluster_centers_[i]
    top_indices = centroid.argsort()[-5:][::-1]
    top_terms = [feature_names[idx] for idx in top_indices if centroid[idx] > 0]
    
    # create readable cluster name
    if top_terms:
        cluster_name = " & ".join(top_terms[:3]).title()
    else:
        cluster_name = f"Cluster {i+1}"
    
    cluster_info[cluster_name] = cluster_repos

# write JSON output
json_out = pathlib.Path("starred_repos_clusters.json")
with json_out.open("w", encoding="utf-8") as fh:
    json.dump(cluster_info, fh, indent=2)

# write markdown output
md_out = pathlib.Path("starred_repos_clusters.md")
with md_out.open("w", encoding="utf-8") as fh:
    fh.write("# Starred Repositories (Semantic Clusters)\n\n")
    
    for topic, repos in cluster_info.items():
        fh.write(f"## {topic}\n\n")
        for repo in sorted(repos):
            fh.write(f"- {repo}\n")
        fh.write("\n")

print(f"Created {len(cluster_info)} clusters")
print(f"Saved to {json_out} and {md_out}")