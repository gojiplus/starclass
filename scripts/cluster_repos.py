#!/usr/bin/env python3
"""
Smart clustering of starred repositories with incremental assignment.
Supports both full reclustering and assigning new repos to existing clusters.
"""
import json, pathlib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

MODE = os.getenv("CLUSTER_MODE", "auto")  # auto, full, assign

# load repository data
data_file = pathlib.Path("starred_repos_data.json")
if not data_file.exists():
    print("Error: starred_repos_data.json not found. Run collect_data.py first.")
    exit(1)

with data_file.open("r", encoding="utf-8") as fh:
    repos_data = json.load(fh)

print(f"Loaded data for {len(repos_data)} repositories")

# load existing clusters and model if available
clusters_file = pathlib.Path("starred_repos_clusters.json")
model_file = pathlib.Path("clustering_model.joblib")
existing_clusters = {}
saved_model = None

if clusters_file.exists() and model_file.exists() and MODE != "full":
    with clusters_file.open("r", encoding="utf-8") as fh:
        existing_clusters = json.load(fh)
    saved_model = joblib.load(model_file)
    print(f"Loaded existing model with {len(existing_clusters)} clusters")

# load changes to identify new repos
changes_file = pathlib.Path("starred_repos_changes.json")
new_repos = []
if changes_file.exists():
    with changes_file.open("r", encoding="utf-8") as fh:
        changes = json.load(fh)
    new_repos = changes.get("new_repos", [])

# determine clustering mode
if MODE == "full" or not existing_clusters or not saved_model:
    print("Performing full reclustering...")
    cluster_mode = "full"
elif new_repos and MODE == "assign":
    print(f"Assigning {len(new_repos)} new repositories to existing clusters...")
    cluster_mode = "assign"
elif new_repos and MODE == "auto":
    # auto: assign if few new repos, recluster if many
    if len(new_repos) <= max(3, len(repos_data) * 0.1):
        print(f"Auto mode: Assigning {len(new_repos)} new repos to existing clusters")
        cluster_mode = "assign"
    else:
        print(f"Auto mode: Too many new repos ({len(new_repos)}), performing full recluster")
        cluster_mode = "full"
else:
    print("No new repositories, using existing clusters")
    cluster_mode = "existing"

def prepare_text(repo):
    """Prepare repository text for vectorization"""
    text_parts = [
        repo["name"].replace("-", " ").replace("_", " "),
        repo["description"],
        " ".join(repo["topics"]),
        repo["readme"][:500]  # first 500 chars of README
    ]
    return " ".join(filter(None, text_parts))

if cluster_mode == "full":
    # full reclustering
    repo_texts = [prepare_text(repo) for repo in repos_data]
    repo_names = [repo["full_name"] for repo in repos_data]
    
    vectorizer = TfidfVectorizer(
        max_features=200, 
        stop_words='english', 
        lowercase=True,
        min_df=1,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(repo_texts)
    
    n_clusters = min(10, max(3, len(repos_data) // 4))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    # generate cluster names
    feature_names = vectorizer.get_feature_names_out()
    cluster_info = {}
    
    for i in range(n_clusters):
        cluster_repos = [repo_names[j] for j, label in enumerate(cluster_labels) if label == i]
        centroid = kmeans.cluster_centers_[i]
        top_indices = centroid.argsort()[-3:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices if centroid[idx] > 0]
        
        cluster_name = " & ".join(top_terms).title() if top_terms else f"Cluster {i+1}"
        cluster_info[cluster_name] = cluster_repos
    
    # save model for future incremental updates
    model_data = {
        "vectorizer": vectorizer,
        "kmeans": kmeans,
        "feature_names": feature_names
    }
    joblib.dump(model_data, model_file)
    
elif cluster_mode == "assign":
    # assign new repos to existing clusters
    model_data = saved_model
    vectorizer = model_data["vectorizer"]
    kmeans = model_data["kmeans"]
    
    # prepare new repo texts
    new_repo_data = [repo for repo in repos_data if repo["full_name"] in new_repos]
    new_texts = [prepare_text(repo) for repo in new_repo_data]
    
    if new_texts:
        # vectorize new repos using existing vocabulary
        new_vectors = vectorizer.transform(new_texts)
        # predict clusters for new repos
        new_labels = kmeans.predict(new_vectors)
        
        # get existing cluster names (ordered by cluster ID)
        cluster_names = list(existing_clusters.keys())
        
        # assign new repos to clusters
        cluster_info = existing_clusters.copy()
        for i, repo in enumerate(new_repo_data):
            cluster_id = new_labels[i]
            if cluster_id < len(cluster_names):
                cluster_name = cluster_names[cluster_id]
                if cluster_name not in cluster_info:
                    cluster_info[cluster_name] = []
                cluster_info[cluster_name].append(repo["full_name"])
                print(f"  Assigned {repo['full_name']} to '{cluster_name}'")
    else:
        cluster_info = existing_clusters
        
else:
    # use existing clusters
    cluster_info = existing_clusters

# write outputs
json_out = pathlib.Path("starred_repos_clusters.json")
with json_out.open("w", encoding="utf-8") as fh:
    json.dump(cluster_info, fh, indent=2)

md_out = pathlib.Path("starred_repos_clusters.md")
with md_out.open("w", encoding="utf-8") as fh:
    fh.write("# Starred Repositories (Semantic Clusters)\n\n")
    
    for topic, repos in cluster_info.items():
        fh.write(f"## {topic}\n\n")
        for repo in sorted(repos):
            fh.write(f"- {repo}\n")
        fh.write("\n")

print(f"Clustering complete. Mode: {cluster_mode}")
print(f"Total clusters: {len(cluster_info)}")
print(f"Saved to {json_out} and {md_out}")