# starclass
Fetch your starred repositories and generate a markdown table of tags.

## Usage

### GitHub Workflow (Recommended)
The repository includes a GitHub workflow that can be manually triggered to update the starred repos tags. It uses the built-in `GITHUB_TOKEN` and automatically commits the results.

To run: Go to Actions tab → "Update Starred Repos Tags" → "Run workflow"

### Local Usage
For personal use you can run the script locally:

```bash
GH_USER=<your-user> python scripts/make_tags.py
```

Providing `GH_TOKEN` is optional but increases the API rate limit.
