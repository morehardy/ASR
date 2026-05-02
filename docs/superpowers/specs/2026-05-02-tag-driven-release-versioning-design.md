# Tag-Driven Release Versioning Design

## Purpose

Release versioning should have one source of truth: the Git release tag. A
published GitHub Release tagged `vX.Y.Z` should produce Python distributions
with version `X.Y.Z` without manually editing package version strings.

## Current Problems

- `pyproject.toml` contains a static project version.
- `src/asr/__init__.py` contains a separate hard-coded `__version__`.
- Releases require keeping multiple version references synchronized.

## Design

Use `hatch-vcs` with the existing Hatchling build backend. The project metadata
declares `version` as dynamic, and Hatchling derives the build version from Git
tags matching the existing `vX.Y.Z` release convention.

Runtime code reads `__version__` from installed package metadata instead of a
hard-coded string. Source checkouts that are not installed fall back to an
explicit unknown development version.

The PyPI publish workflow continues to publish only from GitHub Releases, but
checkout must fetch full Git history and tags so dynamic version detection sees
the release tag. The workflow should validate release tag shape rather than
comparing it to a static version in `pyproject.toml`.

## Release Flow

1. Merge the release-ready code to `main`.
2. Create a GitHub Release with a tag such as `v0.2.1`.
3. GitHub Actions checks out the full tag history.
4. Tests run.
5. `uv build` creates `echoalign_asr_mlx-X.Y.Z` distributions.
6. `twine check` validates the distributions.
7. Trusted Publishing uploads to PyPI.

## Testing

- Run the existing unit test suite.
- Run `uv build`.
- Verify distribution filenames include the current tag-derived version.
- Run `twine check dist/*`.

## Non-Goals

- No custom release script.
- No manual version bump command.
- No switch away from Hatchling.
