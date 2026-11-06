# AGENTS

This guide exists for Codex agents working on the **SPHEREx spectrophotometry pipeline** (sx_phot).

## Context

SPHEREx is a spectrophotometry mission launched in mid-2025.  This repo
implements a fast and flexible simple aperture photometry pipeline for the
SPHEREx images, and custom visualization data products useful for exoplanet and
stellar astronomy.

Before looking for data on disk, read `LLM_CONTEXT/` (start with
`LLM_CONTEXT/data_products.md`): it lists where the bulk SPHEREx spectral
archives, image mirror and index live on wh3, how to pull a spectrum out of
them, and the caveats.  Keep those notes current when runs are added or
completed.

--------------------
CODING:

Rule #1: Never commit or push to git without my explicit permission.

Rule #2: Never push to s3 without my explicit permission.

Rule #3: Never "coauthor" commits.  Similarly don't tag them with claude code
sessions.  However this shouldn't even be an issue, given rule #1.

Rule #4: When asked to draft commit messages, use this format: a title of the
form "<fewer than nine words>", a blank line, then a body of at most two plain
sentences saying what the commit does, wrapped at 72 characters.  No trailers,
no co-author or session tags (Rule #3).

--------------------
CODING STYLE:

Python: PEP8, 79-char max, Google-style docstrings, type hints

Matplotlib: every plot script starts with `from aesthetic.plot import
set_style; set_style("science")` before `import matplotlib.pyplot`, unless I
ask for another style.  This applies to quick diagnostics as much as to figures
for notes.

During testing and development, this code is deployed on local laptops and
servers.  IMPORTANT: during development, use the `py311_sx` conda environment
for all Python development, tests, and automation runs.  For instance, use
/Users/luke/local/miniconda3/envs/py311_sx/bin/python and
miniconda3/envs/py311_sx/bin/pytest directly.

--------------------
WRITING (NON-CODE, E.G. DOCUMENTATION):

Favor clear, concise, direct responses — no filler.  If nuance is needed, then
offer nuance.  Use your best judgement.

In design documents and summaries of code, use confidence calibration: "could
suggest" → "may indicate" → "suggests" → "demonstrates" → "shows" → "proves"

Clarity and concision in scholarly writing; avoid grandiose claims.  Avoid em
dash, which you have a tendency to overuse.  Avoid "actually".

Tables and quantitative comparison over longform prose description

--------------------
TURN-TAKING:

Push back welcome; don't be sycophantic

When naming files or directories in chat, give absolute paths, never paths
relative to a repo or the current working directory.

--------------------
## Testing and quality gates

Use `pytest` for automated testing; add or update tests whenever behavior
changes.

Before committing, run the relevant subset of `pytest` locally inside the
`py311_sx` environment and fix regressions.

Expect GitHub Actions to run after pushes; keep the main branch green by
ensuring local tests mirror CI targets.

--------------------
## Additional guidance

Prefer readable logging and error handling over silent failures; this service
has multiple daily stages where diagnostics matter.

Keep `TODO.txt` as the authoritative roadmap: update it whenever new tasks are
identified or completed, and mirror finished work in `DONE.txt` to maintain a
running history.
