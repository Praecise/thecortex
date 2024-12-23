# Contributing to Cortex

Thank you for your interest in contributing to Cortex! This document outlines the process for contributing and our community guidelines.

## License

Cortex is licensed under the Apache License, Version 2.0. By contributing to Cortex, you agree to license your contributions under the same license.

## Getting Started

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/praecise/the-cortex.git
cd the-cortex
```
3. Set up your development environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development Workflow

1. Create a new branch for your work:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding standards

3. Run the test suite:
```bash
pytest tests/
```

4. Update documentation as needed

5. Commit your changes:
```bash
git commit -m "feat: description of your changes"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Pull Request Process

1. Update your branch with the latest main:
```bash
git fetch origin
git rebase origin/main
```

2. Push your changes:
```bash
git push origin feature/your-feature-name
```

3. Open a Pull Request through GitHub

4. Ensure your PR:
   - Has a clear title and description
   - Includes tests for new functionality
   - Updates relevant documentation
   - Passes all CI checks

## Code Style

We follow these coding standards:
- Python: PEP 8
- Documentation: Google style docstrings
- Type hints for all new code
- Maximum line length of 88 characters (using Black formatter)

Run code formatting:
```bash
black cortex/
isort cortex/
```

## Testing

- All new features must include tests
- Maintain or improve code coverage
- Include both unit and integration tests
- Test with `pytest tests/`

## Documentation

- Update documentation for any changed functionality
- Follow Google style docstrings
- Include examples for new features
- Keep API documentation up to date

## Review Process

PRs need:
- One approval from core maintainer
- All CI checks passing
- Up-to-date with main branch
- Documentation updated

## Copyright Notice

Every source file should include:
```python
# Copyright [Year] Cortex Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Release Process

1. Version Bumping
- Use semantic versioning
- Update CHANGELOG.md
- Update version in setup.py

2. Release Steps
- Create release branch
- Run full test suite
- Update documentation
- Create GitHub release
- Publish to PyPI

## Community

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and discussions
- Stack Overflow: Technical questions with 'cortex' tag

## Need Help?

- Check our documentation
- Ask in GitHub Discussions
- Join our community chat
- Email: maintainers@thecortex.xyz

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes for significant contributions
- Eligible for maintainer status after sustained contributions

Thank you for contributing to Cortex!