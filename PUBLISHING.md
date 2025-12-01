# Publishing safenax to PyPI

This guide walks you through publishing your package to PyPI using `uv`.

## Prerequisites

Before publishing, make sure you have:
1. A PyPI account (create one at https://pypi.org/account/register/)
2. A TestPyPI account for testing (create one at https://test.pypi.org/account/register/)
3. Updated the author information in `pyproject.toml`

## Step 1: Update Package Metadata

**Important**: Before publishing, update `pyproject.toml`:
- Replace `"Your Name"` with your actual name
- Replace `"your.email@example.com"` with your email
- Update version number if needed (currently `0.1.0`)

## Step 2: Build and Publish with `uv` (Easy Way!)

`uv` makes building and publishing super simple, just like Poetry:

### Test on TestPyPI first (Recommended):

```bash
# Build and publish to TestPyPI in one command
uv build
uv publish --publish-url https://test.pypi.org/legacy/
```

Then test installing it:
```bash
uv pip install --index-url https://test.pypi.org/simple/ --no-deps safenax
```

### Publish to PyPI:

```bash
# Build (if not already built)
uv build

# Publish to PyPI
uv publish
```

That's it! Just two commands. 🎉

You'll be prompted for your PyPI username and password (or token).

## Step 3: Verify Installation

After publishing, verify by installing from PyPI:

```bash
uv pip install safenax
# or
pip install safenax
```

And test the import:
```python
from safenax import PortfolioOptimizationV0, BinanceFeeTier
```

## Using API Tokens (Recommended)

For better security, use API tokens instead of passwords:

1. Generate a token on PyPI (Account Settings → API tokens)
2. Use it when prompted, or set environment variables:

```bash
export UV_PUBLISH_USERNAME=__token__
export UV_PUBLISH_PASSWORD=pypi-YOUR_TOKEN_HERE
```

Or create a `~/.pypirc` file:

```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

## Publishing Updates

When you want to publish a new version:

1. Update the version in `pyproject.toml`
2. Build and publish:
   ```bash
   uv build
   uv publish
   ```

That's it! `uv` handles cleaning old builds automatically.

## Alternative: Manual Method with build + twine

If you prefer the traditional approach or need more control:

```bash
# Install tools
pip install build twine

# Build
python -m build

# Publish to TestPyPI
python -m twine upload --repository testpypi dist/*

# Publish to PyPI
python -m twine upload dist/*
```

## CI/CD Automation (Optional)

Consider setting up GitHub Actions to automate publishing when you create a new release tag.

## After Publishing

Once published, users can install your package with:

```bash
pip install safenax
```

And use it like:

```python
from safenax import PortfolioOptimizationV0, PortfolioOptimizationGARCHV0, BinanceFeeTier

# Use your environments
env = PortfolioOptimizationV0(data_paths={"BTC": "path/to/btc.csv"})
```
