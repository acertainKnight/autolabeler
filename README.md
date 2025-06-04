# Project Name Template

## Using this template

1. Click "Use this template" above
2. Give your project a name
3. Clone your new repository
4. Initialize your project:
   ```bash
   # Run setup script if you have one
   ./setup.sh
   ```

## What's Included

- [List key features/structure]
- [List configurations]
- [List any prerequisites]

## Configuration

[Explain any configuration needed]

## License

[Your chosen license]

## Project Overview

This is repo-template for use with local or devx development. It handles a fairly wide
variety of concerns including pytests, linting, precommits, and docker set-ups.

It heavily utilizes Make targets. All packages are installed using uv.

## Development Setup

### Requirements

Please ensure your system meets the following requirements before continuing:

- Python 3.9+ installation
- [uv](https://astral.sh/uv) package manager installed ([installation guide](https://astral.sh/uv/install))
- Docker installed and configured
- Required environment variables: TRINO_USER TRINO_PW_DEV RUN_ENV DATA_ENV

### Initial Setup

1. Clone the repository locally
   ```sh
   git clone <your-repo-url>
   ```

2. Set up environment variables (create a `.env` file or set them in your environment)

3. Build the development environment:
   ```sh
   # Basic build
   make build

   # Build with specific tag
   make build TAG=v1.0.0

   # For additional build options run
   make help
   ```

4. Start the services in a dev shell:
   ```sh
   make develop
   ```

## Project Configuration

### Central Configuration Files

The project attempts to use centralized configuration to minimize the need for
modifying Docker files directly:

#### pyproject.toml
Contains core project configuration:

```toml
[project]
name = "your_project_name"  # Sets Docker container names
requires-python = ">=3.9"   # Determines Python version for Docker

dependencies = [
    'bloomberg.analysts.datalake',
    'bloomberg.analysts.bdh'
]

[project.optional-dependencies]
dev = [
    # These are all required for the standard dev workflow
    'black',
    'build',
    'codespell',
    'pre-commit',
    'pytest',
    'pytest-cov',
    'ruff',
    'twine',
# Place any other dev only dependencies for the project below
]

[tool.docker.base_images.default]
image = "artprod.dev.bloomberg.com/rhel8-dpkg-local-dev:latest"

[tool.docker.base_images.cuda]
image = "artprod.dev.bloomberg.com/rhel8-dpkg-local-dev:latest"
cuda_version = "12.1"
cudnn_version = "8.9.2.26"
nccl_version = "2.18.5"
```

#### .env File
Create a `.env` file in the project root for environment variables:

### Dependency Management

Dependencies are managed through the pyproject.toml according to current PEP:

```sh
make build
make build dev
make build gpu
make build custom
make build dev custom
```

### Docker Configuration

The Docker setup is designed to be configuration-driven, rarely requiring direct modifications:

- **Base Images**: Configured in pyproject.toml's `[tool.docker]` sections
- **Python Version**: Automatically detected from pyproject.toml
- **Project Name**: Derived from pyproject.toml or overridden via .env
- **Dependencies**: Managed through pyproject.toml
- **Build Arguments**: Handled through make build flags

This means for most new projects, you only need to:
1. Update pyproject.toml with your project details
2. Put your package code in src/YOUR_PROJECT_NAME
3. Set up your .env file
4. Use `make` commands for all Docker operations

## Development Workflow

### Building Options

The project supports various build configurations:

```sh
# Basic development build
make build dev

# Fresh build without cache
make build no-cache
```

### Development Commands

#### Start Development Shell
```sh
make develop  # or make shell
```

#### Code Quality
```sh
# Auto-format code
make autoformat

# Run pre-commit checks
make precommit

# Run linting
make lint

# Run all checks (linting and tests)
make check
```

#### Testing
```sh
# Run all tests
make test

# Run tests with specific arguments
make test ARGS="-v -k test_name"
```

#### Jupyter Lab
```sh
make lab  # or make jupyter
```

### Docker Management

```sh
# View running containers
make ps

# View container logs
make logs

# Stop all services
make stop

# Full cleanup (removes all Docker resources and cached files)
make clean
```

## Code Quality Standards

Our primary linting and formatting tools are:

- [Ruff](https://github.com/charliermarsh/ruff): Fast Python linter and formatter
- [Black](https://black.readthedocs.io/en/stable/): Code formatter
- [codespell](https://github.com/codespell-project/codespell): Spell checker
- [mdformat](https://mdformat.readthedocs.io/en/stable/): Markdown formatter
- [sqlfluff](https://docs.sqlfluff.com/en/stable/): SQL formatter

Key standards include:

- Documentation follows NumPy style guides
- Single quotes are standard
- 88 character line length
- SQL dialect set to `bigquery`
- SQL templating set to `jinja`
- SQL tab size set to `2`

### Code Migration

```sh
# Run normal migration
make migrate

# Add #noqa to failing lines
make migrate add-noqa=true
```

