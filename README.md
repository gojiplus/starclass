# starclass

Automatically cluster your starred GitHub repositories using semantic similarity analysis. This tool fetches your starred repos, analyzes their names, descriptions, topics, and README content, then groups them into meaningful clusters.

## How it works

**Pipeline Architecture:**
1. **Data Collection** (`collect_data.py`) - Fetches starred repositories with metadata (topics, descriptions, README content) and caches to JSON
2. **Clustering** (`cluster_repos.py`) - Applies semantic similarity clustering using TF-IDF and K-means to group related repositories
3. **Output** - Generates both JSON and Markdown files with clustered results

## Usage

### GitHub Action
Add this action to your workflow:

```yaml
- name: Cluster starred repositories
  uses: gojiplus/starclass@main
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

### Self-hosted Demo
This repository includes a demo workflow. Fork it and:

1. Set repository secrets:
   - `GH_PAT`: Personal access token with `user` scope  
2. Go to Actions tab → "Demo StarClass Action" → "Run workflow"

### Local Usage
```bash
# Collect data
GH_TOKEN=<your-token> GH_USER=<your-username> python scripts/collect_data.py

# Generate clusters
python scripts/cluster_repos.py
```

## Output Files
- `starred_repos_data.json`: Raw repository data (cached)
- `starred_repos_clusters.json`: Clustered repositories by topic
- `starred_repos_clusters.md`: Human-readable clustered output
