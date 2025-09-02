# starclass

Automatically cluster your starred GitHub repositories using semantic similarity analysis. This tool intelligently tracks your starred repos, analyzes their content, and groups them into meaningful clusters with incremental updates to preserve your established organization.

## How it works

**Incremental Pipeline Architecture:**
1. **Repository Tracking** (`fetch_starred.py`) - Tracks starred repositories and identifies new/updated/removed repos
2. **Data Collection** (`collect_data.py`) - Incrementally fetches detailed metadata only for changed repositories
3. **Smart Clustering** (`cluster_repos.py`) - Uses semantic similarity with three modes:
   - **Auto**: Assigns new repos to existing clusters, reclusters if major changes
   - **Assign**: Only assigns new repos to existing clusters (preserves organization)
   - **Full**: Complete reclustering from scratch

## Usage

### GitHub Action
Add this action to your workflow:

```yaml
- name: Cluster starred repositories
  uses: gojiplus/starclass@main
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
    cluster-mode: 'auto'  # or 'assign' or 'full'
```

### Self-hosted Demo
This repository includes a demo workflow. Fork it and:

1. Set repository secrets:
   - `GH_PAT`: Personal access token with `user` scope  
2. Go to Actions tab → "Demo StarClass Action" → "Run workflow"

### Local Usage
```bash
# Step 1: Track starred repositories
GH_TOKEN=<your-token> GH_USER=<your-username> python scripts/fetch_starred.py

# Step 2: Collect detailed data (incremental)
GH_TOKEN=<your-token> GH_USER=<your-username> python scripts/collect_data.py

# Step 3: Generate clusters
CLUSTER_MODE=auto python scripts/cluster_repos.py
```

## Clustering Modes

- **auto** (default): Smart mode that assigns new repos to existing clusters, but reclusters everything if there are major changes
- **assign**: Only assigns new repositories to existing clusters, preserving your current organization
- **full**: Performs complete reclustering of all repositories from scratch

## Output Files
- `starred_repos_list.json`: Tracked starred repositories with metadata
- `starred_repos_changes.json`: Summary of new/updated/removed repositories
- `starred_repos_data.json`: Detailed repository data (cached)
- `starred_repos_clusters.json`: Clustered repositories by topic
- `starred_repos_clusters.md`: Human-readable clustered output
- `clustering_model.joblib`: Saved clustering model for incremental updates
