# Way of working
## Code styleguide
- `black` + `flake8` with selected plugins

## Notebooks
- do not push notebooks to repo (it looks ugly in diff)
- use `jupytext` to convert notebooks to python scripts (you can work on notebooks and `jupytext` syncs them with scripts)

### `jupytext` [quickstart](https://jupytext.readthedocs.io/en/latest/using-cli.html)

> NOTE: general `jupytext` config will be set up soon (commands and usage will be simplified)

**Sync notebook**
```bash
jupytext --set-formats ipynb,py:percent notebook.ipynb
```

**Sync notebook with automatic `black`**
```bash
jupytext --set-formats ipynb,py:percent --pipe black notebook.ipynb
```


## Git
- use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) as a style guide for naming your commits
- do not push directly to `main` branch (work on a non-protected branch and create a PR)