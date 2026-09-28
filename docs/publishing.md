# Build and publish

This guide is for a maintainer. It prepares a release; it does not upload it. Use Python 3.9 or newer. The project version is set in `pyproject.toml` and `src/zcartpole/_version.py`. Update them together before each release. A version uploaded to PyPI cannot be replaced with different files under the same version.

## Verify locally

From a clean project folder:

```bash
python -m pip install '.[dev,docs,opt,viz,speed]' build twine
python -m pytest -q
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

Inspect both the wheel and source archive. Install the built wheel in a clean environment and run a tiny balance JSON; install the source archive too if you plan to support it. Confirm the metadata version, `LICENSE`, and README rendering. The measured DaRUS CSV and derived reports must stay out of **both** PyPI artifacts. The source ZIP supplied separately can include them under `DATA_LICENSE.md`.

## Publish with your account

Choose a repository URL and project owner before adding `project.urls` to `pyproject.toml`; do not invent these links. Check whether the name is available on PyPI and TestPyPI at release time. Preferred publishing uses a trusted publisher linked to the actual source repository and release workflow. Upload a release candidate to TestPyPI first, inspect its rendered page, then install and smoke-test it using TestPyPI with an appropriate dependency index. After approval, publish the *same verified artifacts* to PyPI. If trusted publishing is unavailable, use a scoped upload token with `python -m twine upload dist/*`; keep credentials outside the project.

Do not upload old files left in `dist/`. Record the release tag, hashes, build environment, and test result. The sample physical parameters are illustrative; document tested platforms and any known limitations in the release notes. See [PyPA's packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) and [trusted publisher guide](https://docs.pypi.org/trusted-publishers/).
