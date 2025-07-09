# starclass
Fetch your starred repositories and generate a markdown table of tags.

The workflow uses the built in `GITHUB_TOKEN`.  For personal use you can run
the script locally:

```bash
GH_USER=<your-user> python scripts/make_tags.py
```

Providing `GH_TOKEN` is optional but increases the API rate limit.
