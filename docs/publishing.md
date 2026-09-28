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

The repository is `GokhanTkr/Zcartpole`. The manual `.github/workflows/publish.yml` workflow builds a wheel and source archive, checks their metadata, and uploads to the selected index. It does not run on every commit. Check the `zcartpole` name on both indices before setup.

Create pending trusted publishers in the [TestPyPI account](https://test.pypi.org/manage/account/publishing/) and [PyPI account](https://pypi.org/manage/account/publishing/). Use project `zcartpole`, owner `GokhanTkr`, repository `Zcartpole`, workflow `publish.yml`, and environment `testpypi` or `pypi` respectively. Create matching GitHub environments, with a required reviewer for `pypi`. The two indexes need separate accounts. No API token is stored in the repository.

In GitHub Actions, run **Package release** with destination `testpypi`. Inspect the rendered package page and install/test the artifact. Then run it again from the same commit with destination `pypi` and approve the protected environment. Each run rebuilds the distributions from the selected commit; compare version and source revision before the production run. A version already uploaded to an index cannot be overwritten.

Record the release tag, hashes, build environment, and test result. The sample physical parameters are illustrative; document tested platforms and any known limitations in the release notes. See [PyPA's packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) and [trusted publisher guide](https://docs.pypi.org/trusted-publishers/).
