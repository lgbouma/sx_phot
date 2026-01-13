# AGENTS

This guide exists for Codex agents working on the **SPHEREx spectrophotometry pipeline** (sx_phot).

## Context

- SPHEREx is a spectrophotometry mission launched in mid-2025.  This repo implements a fast and flexible simple aperture photometry pipeline for the SPHEREx images, and custom visualization data products useful for exoplanet and stellar astronomy.

## Coding standards

- Write Python that passes PEP 8; keep lines readable and use descriptive names.
- Use Google-style docstrings for every public module, class, function, and method to describe arguments, returns, raises, and any non-obvious behavior.
- Use f-strings for logging messages.
- Favor small, testable functions and keep high-level routines thin when possible.

## Deployment

- At scale, this code is deployed on the open science grid.
- During testing and development, this code is deployed on local laptops and servers.  IMPORTANT: during development, use the `vbls` conda environment for all Python development, tests, and automation runs.  For instance, use /Users/luke/local/miniconda3/envs/vbls/bin/python and miniconda3/envs/vbls/bin/pytest directly.

## Workflow expectations

- Begin sessions by loading all `*.py`, `*.txt`, and `*.md` files into the context window.
- Commit directly to `main`; branches and pull requests are unnecessary for this solo workflow.
- Keep commits focused, describing the problem solved and mentioning tests run.
- Avoid rewriting history (no rebases on published work) unless the user explicitly requests it.

## Testing and quality gates

- Use `pytest` for automated testing; add or update tests whenever behavior changes.
- Before committing, run the relevant subset of `pytest` locally inside the `vbls` environment and fix regressions.
- Expect GitHub Actions to run after pushes; keep the main branch green by ensuring local tests mirror CI targets.

## Additional guidance

- Prefer readable logging and error handling over silent failures; this service has multiple daily stages where diagnostics matter.
- Document assumptions in code comments only when the implementation is non-obvious or relies on domain details (avoid noise).
- Coordinate deployment steps through GitHub pushes; no extra tooling is currently defined.
- Keep `TODO.txt` as the authoritative roadmap: update it whenever new tasks are identified or completed, and mirror finished work in `DONE.txt` to maintain a running history.
