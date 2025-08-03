# Release Guide

This guide covers the release process for the Segmentation Robustness Framework.

## 🚀 Release Philosophy

### Release Principles

- **Stability First**: Ensure releases are stable and well-tested
- **Semantic Versioning**: Follow semantic versioning principles
- **Comprehensive Testing**: Test thoroughly before release
- **Clear Communication**: Document changes and migration guides
- **User-Focused**: Prioritize user experience and backward compatibility

### Release Types

1. **Patch Releases** (x.y.z+1): Bug fixes and minor improvements
2. **Minor Releases** (x.y+1.0): New features, backward compatible
3. **Major Releases** (x+1.0.0): Breaking changes, major features

## 📋 Pre-Release Checklist

### Code Quality

- [ ] **All tests pass**: `pytest` runs successfully
- [ ] **Code coverage**: >90% coverage maintained
- [ ] **Linting passes**: `ruff check .` passes
- [ ] **Documentation**: All new features documented
- [ ] **Examples updated**: Examples work with new features

### Documentation

- [ ] **API documentation**: Updated for new/changed APIs
- [ ] **User guide**: Updated with new features
- [ ] **Migration guide**: For breaking changes
- [ ] **Changelog**: Updated with all changes
- [ ] **README**: Updated if needed

### Testing

- [ ] **Tests**: All new code covered

## 🔄 Release Process

### Step 1: Prepare Release Branch

```bash
# Create release branch
git checkout -b release/v1.2.0

# Update version in pyproject.toml
# Change version from "1.1.0" to "1.2.0"

# Update version in segmentation_robustness_framework/__version__.py
# Change version from "1.1.0" to "1.2.0"

# Update changelog
# Add new section for v1.2.0
```

### Step 2: Update Version

```toml
# pyproject.toml
[tool.poetry]
name = "segmentation-robustness-framework"
version = "1.2.0"  # Update this
description = "Framework for evaluating segmentation model robustness"
```

```python
VERSION = (1, 2, 0)  # Update this

__version__ = ".".join(map(str, VERSION))
```

### Step 3: Update Changelog

```markdown
# CHANGELOG.md

## [1.2.0] - 2024-01-15

### Added
- New FGSM attack implementation
- Support for custom metrics
- GPU memory optimization
- Comprehensive logging system

### Changed
- Improved error messages for better debugging
- Updated default batch size to 8
- Enhanced documentation with more examples

### Fixed
- Memory leak in large batch processing
- Incorrect metric calculation for edge cases
- Documentation typos and broken links

### Deprecated
- `old_attack_function` - use `new_attack_function` instead

### Removed
- Support for Python 3.8 (end of life)
```

### Step 4: Run Release Tests

```bash
# Clean environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# Install fresh dependencies
pip install -e ".[dev]"

# Run full test suite
pytest

# Check documentation builds
mkdocs build

# Test installation from PyPI (if available)
pip install segmentation-robustness-framework==1.2.0
```

### Step 5: Create Release Tag

```bash
# Commit all changes
git add .
git commit -m "Release v1.2.0"

# Create and push tag
git tag v1.2.0
git push origin release/v1.2.0
git push origin v1.2.0
```

### Step 6: Build and Publish

```bash
# Build package
poetry build

# Check built package
tar -tzf dist/segmentation_robustness_framework-1.2.0.tar.gz

# Publish to PyPI
poetry publish

# Verify publication
pip install segmentation-robustness-framework==1.2.0
```

### Step 7: Create GitHub Release

1. **Go to GitHub**: Navigate to releases page
2. **Create new release**: Use tag `v1.2.0`
3. **Add release notes**: Copy from CHANGELOG.md
4. **Upload assets**: Add any additional files
5. **Publish release**: Make it public

## 📝 Release Notes Template

```markdown
# Release v1.2.0

## 🎉 What's New

This release introduces several new features and improvements:

### ✨ New Features

- **FGSM Attack**: Fast Gradient Sign Method implementation
- **Custom Metrics**: Support for user-defined evaluation metrics
- **GPU Optimization**: Improved memory management for large models
- **Enhanced Logging**: Comprehensive logging system for debugging

### 🔧 Improvements

- **Better Error Messages**: More descriptive error messages
- **Documentation**: Expanded examples and tutorials
- **Performance**: Faster evaluation pipeline
- **Compatibility**: Support for newer PyTorch versions

### 🐛 Bug Fixes

- Fixed memory leak in batch processing
- Corrected metric calculation for edge cases
- Fixed documentation typos and broken links
- Resolved CUDA memory issues on Windows

### 📚 Documentation

- Added comprehensive API documentation
- Updated user guide with new features
- Added migration guide for breaking changes
- Improved code examples

## 🚀 Installation

```bash
pip install segmentation-robustness-framework==1.2.0
```

## 🔄 Migration Guide

### Breaking Changes

If you're upgrading from v1.1.0:

1. **Update import statements**:
   ```python
   # Old
   from segmentation_robustness_framework.attacks import OldAttack
   
   # New
   from segmentation_robustness_framework.attacks import FGSM
   ```

2. **Update function calls**:
   ```python
   # Old
   attack = OldAttack(model, eps=0.02)
   
   # New
   attack = FGSM(model, eps=0.02)
   ```

### Deprecated Features

- `old_attack_function` is deprecated, use `new_attack_function` instead
- Support for Python 3.8 will be removed in v1.3.0

## 📊 Performance Improvements

- **30% faster** evaluation pipeline
- **50% less memory** usage for large models
- **Improved GPU** utilization

## 🧪 Testing

All tests pass on:
- Python 3.12
- PyTorch 2.0, 2.1
- CUDA 11.8, 12.1
- CPU-only environments

## 🙏 Thanks

Thanks to all contributors who made this release possible:
- @username1 for FGSM implementation
- @username2 for performance improvements
- @username3 for documentation updates

## 🔍 Post-Release Tasks

### Monitor Release

- [ ] **Check PyPI**: Verify package is available
- [ ] **Test installation**: Install in clean environment
- [ ] **Monitor issues**: Watch for user reports
- [ ] **Update documentation**: Deploy updated docs

### Communication

- [ ] **Announce release**: Social media, mailing list
- [ ] **Update website**: Update version numbers
- [ ] **Notify users**: Email major users if breaking changes
- [ ] **Blog post**: Write release blog post if major release

### Maintenance

- [ ] **Merge release branch**: Merge to main
- [ ] **Update development version**: Bump to next dev version
- [ ] **Plan next release**: Schedule and plan features
- [ ] **Archive old versions**: Clean up old releases

## 🚨 Emergency Releases

### When to Make Emergency Release

- **Security vulnerabilities**: Critical security issues
- **Major bugs**: Breaking bugs affecting many users
- **Compatibility issues**: Critical compatibility problems

### Emergency Release Process

```bash
# Create hotfix branch
git checkout -b hotfix/v1.2.1

# Make minimal fix
# Update version to 1.2.1
# Update changelog

# Test thoroughly
pytest

# Create tag and release
git tag v1.2.1
git push origin v1.2.1

# Build and publish immediately
poetry build
poetry publish
```

## 📊 Release Metrics

### Track These Metrics

- **Downloads**: PyPI download statistics
- **Issues**: Bug reports and feature requests
- **Adoption**: New users and usage patterns
- **Performance**: Benchmark results
- **Compatibility**: Platform compatibility reports

### Monitoring Tools

```bash
# Check PyPI downloads
pip install pypistats
pypistats overall segmentation-robustness-framework

# Monitor GitHub metrics
# Use GitHub Insights for repository analytics
```

## 🎯 Best Practices

### Release Planning

1. **Plan ahead**: Schedule releases in advance
2. **Feature freeze**: Stop adding features before release
3. **Test thoroughly**: Run comprehensive tests
4. **Document changes**: Keep changelog updated
5. **Communicate clearly**: Inform users of changes

### Quality Assurance

1. **Automated testing**: Use CI/CD for testing
2. **Manual testing**: Test on different platforms
3. **User testing**: Get feedback from users
4. **Performance testing**: Ensure no regressions
5. **Security review**: Check for vulnerabilities

### Communication

1. **Clear release notes**: Document all changes
2. **Migration guides**: Help users upgrade
3. **Timely announcements**: Notify users promptly
4. **Support**: Be ready to help users
5. **Feedback**: Collect and act on feedback

## 🚀 Next Steps

After reading this release guide:

1. **Set up release tools** and automation
2. **Practice the release process** with test releases
3. **Contribute to releases** by helping with testing
4. **Learn from each release** to improve the process
5. **Help maintain release quality** by following guidelines

Happy releasing! 🎉
