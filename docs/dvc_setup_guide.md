# DVC Setup Guide for AutoLabeler

This guide explains how to set up and use Data Version Control (DVC) for managing datasets and models in AutoLabeler Phase 2.

## What is DVC?

DVC (Data Version Control) is an open-source tool for versioning large files, datasets, and ML models. It integrates with Git to provide version control for data science projects.

**Key Benefits:**
- Track and version datasets and models without bloating Git repos
- Collaborate on data with team members
- Reproduce experiments with exact data versions
- Store data in cloud storage (S3, Azure, GCS, etc.)
- Create reproducible ML pipelines

## Installation

### Install DVC

```bash
# Basic installation
pip install dvc

# With cloud storage support
pip install 'dvc[s3]'      # For AWS S3
pip install 'dvc[azure]'   # For Azure Blob Storage
pip install 'dvc[gs]'      # For Google Cloud Storage
pip install 'dvc[all]'     # All storage backends
```

### Verify Installation

```bash
dvc version
```

## Quick Start

### 1. Initialize DVC

```python
from autolabeler.core.versioning import DVCManager, DVCConfig

# Configure DVC
config = DVCConfig(
    repo_path='/path/to/autolabeler',
    remote_name='storage',
    remote_url='s3://my-bucket/autolabeler-data'
)

# Initialize
manager = DVCManager(config)
manager.init()
manager.configure_remote()
```

Or use the CLI:

```bash
cd /path/to/autolabeler
dvc init
dvc remote add -d storage s3://my-bucket/autolabeler-data
```

### 2. Track a Dataset

```python
# Add a dataset
metadata = manager.add_dataset(
    file_path='datasets/train.csv',
    version='v1.0',
    description='Initial training dataset',
    tags=['train', 'sentiment-analysis'],
    metadata={'num_samples': 10000, 'source': 'customer-reviews'}
)

# Push to remote storage
manager.push()
```

### 3. Track a Model

```python
# Add a trained model
metadata = manager.add_model(
    model_path='models/classifier_v1.pkl',
    version='v1.0',
    description='Baseline sentiment classifier',
    metrics={
        'accuracy': 0.85,
        'f1_score': 0.83,
        'precision': 0.84,
        'recall': 0.82
    },
    tags=['production', 'baseline'],
    metadata={
        'hyperparameters': {'learning_rate': 0.001, 'epochs': 10},
        'training_time_seconds': 3600
    }
)

# Push to remote storage
manager.push()
```

### 4. Retrieve Data/Models

```python
# Pull from remote storage
manager.pull()

# Checkout a specific version
manager.checkout_version('datasets/train.csv', version='v1.0')

# List all versions
versions = manager.list_versions(version_type='dataset')
for v in versions:
    print(f"{v.version}: {v.description} ({v.timestamp})")
```

## Remote Storage Configuration

### AWS S3

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID="your-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"

# Add remote
dvc remote add -d storage s3://my-bucket/dvc-cache
```

Or in Python:

```python
config = DVCConfig(
    repo_path='/path/to/autolabeler',
    remote_url='s3://my-bucket/dvc-cache'
)
manager = DVCManager(config)
manager.init()
manager.configure_remote()
```

### Azure Blob Storage

```bash
# Configure Azure credentials
export AZURE_STORAGE_ACCOUNT="myaccount"
export AZURE_STORAGE_KEY="mykey"

# Add remote
dvc remote add -d storage azure://mycontainer/dvc-cache
```

### Google Cloud Storage

```bash
# Configure GCS credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Add remote
dvc remote add -d storage gs://my-bucket/dvc-cache
```

### Local Storage (for testing)

```bash
# Use local directory as remote
dvc remote add -d storage /path/to/local/dvc-cache
```

## Advanced Features

### Version Lineage Tracking

Track the evolution of datasets and models:

```python
# Create base version
v1 = manager.add_dataset(
    'datasets/train.csv',
    version='v1.0',
    description='Initial dataset'
)

# Create derived version
v2 = manager.add_dataset(
    'datasets/train_cleaned.csv',
    version='v1.1',
    description='Cleaned and preprocessed',
    parent_version='v1.0'
)

# Get lineage
lineage = manager.get_lineage('v1.1')
print(f"Lineage: {' -> '.join(lineage)}")
# Output: Lineage: v1.0 -> v1.1
```

### Compare Versions

```python
comparison = manager.compare_versions('v1.0', 'v2.0', version_type='model')
print(f"Accuracy improvement: {comparison['metrics']['accuracy']['percent_change']:.2f}%")
print(f"Size difference: {comparison['size_diff_bytes'] / 1024 / 1024:.2f} MB")
```

### Export Metadata Reports

```python
# Export version metadata to CSV
manager.export_metadata_report(
    output_path='reports/version_history.csv',
    version_type='all'
)
```

### Metadata Queries

```python
# Get specific version info
version_info = manager.get_version_info('v1.0', version_type='model')
print(f"Metrics: {version_info.metrics}")
print(f"Tags: {version_info.tags}")
print(f"Size: {version_info.size_bytes / 1024 / 1024:.2f} MB")

# Filter by tags
all_versions = manager.list_versions()
production_models = [v for v in all_versions if 'production' in v.tags]
```

## Integration with Phase 2 Features

### DSPy Optimization

```python
# Track optimized prompts/programs
manager.add_model(
    'optimizers/sentiment_optimizer.pkl',
    version='v1.0',
    description='DSPy optimized sentiment classifier',
    metrics={
        'accuracy': 0.92,
        'cost_per_example': 0.003,
        'latency_ms': 150
    },
    tags=['dspy', 'optimized']
)
```

### Active Learning

```python
# Track labeled datasets through active learning iterations
for iteration in range(5):
    manager.add_dataset(
        f'datasets/active_learning_iter_{iteration}.csv',
        version=f'al_v{iteration}',
        description=f'Active learning iteration {iteration}',
        parent_version=f'al_v{iteration-1}' if iteration > 0 else None,
        metadata={
            'iteration': iteration,
            'newly_labeled': 100,
            'total_labeled': 100 * (iteration + 1)
        }
    )
```

### Weak Supervision

```python
# Track labeling functions and their outputs
manager.add_dataset(
    'weak_supervision/labeled_data.csv',
    version='ws_v1.0',
    description='Weakly supervised labels',
    metadata={
        'num_labeling_functions': 10,
        'coverage': 0.85,
        'label_model_accuracy': 0.78
    }
)
```

## Best Practices

### 1. Version Everything

- Track all datasets (train, validation, test)
- Track all models (checkpoints, final models)
- Track feature stores and embeddings
- Track labeling function definitions

### 2. Use Descriptive Versions

```python
# Good: Semantic versioning with context
version='v1.2.3-dspy-optimized'

# Good: Date-based for datasets
version='2025-01-15-customer-reviews'

# Avoid: Generic versions
version='model1', version='data2'
```

### 3. Add Rich Metadata

```python
manager.add_model(
    model_path='models/classifier.pkl',
    version='v2.0',
    description='DSPy optimized with active learning',
    metrics={
        'accuracy': 0.92,
        'f1_score': 0.90,
        'cost_reduction': 0.45,  # 45% cost reduction
        'label_efficiency': 0.85  # 85% fewer labels needed
    },
    tags=['production', 'dspy', 'active-learning'],
    metadata={
        'hyperparameters': {
            'temperature': 0.1,
            'max_tokens': 150
        },
        'training_samples': 1000,
        'optimization_method': 'BootstrapFewShot',
        'cost_per_sample': 0.002
    }
)
```

### 4. Regular Pushes

```python
# After adding versions, push to remote
manager.push()

# Or use Git hooks to automate
# (see CI/CD Integration section)
```

### 5. Tag Important Versions

```python
# Tag production models
tags=['production', 'v2.0', 'approved']

# Tag experimental versions
tags=['experiment', 'test', 'do-not-deploy']

# Tag by feature
tags=['dspy-optimized', 'graphrag', 'active-learning']
```

## CI/CD Integration

### GitHub Actions Workflow

Add to `.github/workflows/phase2-tests.yml`:

```yaml
jobs:
  dvc-tests:
    name: DVC Integration Tests
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Install DVC
        run: pip install 'dvc[s3]'

      - name: Configure DVC remote
        run: |
          dvc remote add -d storage s3://test-bucket/dvc-cache
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Pull DVC data
        run: dvc pull

      - name: Run tests with versioned data
        run: pytest tests/test_unit/test_dvc_manager.py
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Auto-push DVC data on commit

# Check for .dvc files in staging area
if git diff --cached --name-only | grep -q '\.dvc$'; then
    echo "Pushing DVC data to remote..."
    dvc push
fi
```

## Troubleshooting

### DVC Not Found

```bash
# Ensure DVC is installed
pip install dvc

# Check PATH
which dvc
```

### Remote Storage Access Denied

```bash
# Check credentials
aws s3 ls s3://my-bucket/  # For S3
az storage blob list --account-name myaccount  # For Azure

# Verify IAM permissions
# Ensure your credentials have read/write access to the remote
```

### Large Files in Git

If you accidentally committed large files to Git:

```bash
# Remove from Git, add to DVC
git rm --cached large_file.csv
dvc add large_file.csv
git add large_file.csv.dvc .gitignore
git commit -m "Move large file to DVC"
```

### Cache Issues

```bash
# Clear DVC cache
dvc cache clear

# Rebuild cache
dvc fetch
dvc checkout
```

## Performance Tips

### 1. Use Symlinks (Linux/Mac)

```python
config = DVCConfig(
    repo_path='/path/to/repo',
    use_symlinks=True  # Faster checkout
)
```

### 2. Partial Checkouts

```bash
# Only checkout specific files
dvc checkout datasets/train.csv
```

### 3. Shallow Clones

```bash
# Clone without full DVC data
git clone <repo-url>
# Then selectively pull data
dvc pull datasets/train.csv.dvc
```

## Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/start)
- [DVC with CI/CD](https://dvc.org/doc/use-cases/ci-cd-for-machine-learning)
- [DVC API Reference](https://dvc.org/doc/api-reference)

## Support

For issues or questions:
1. Check this guide
2. Review DVC documentation
3. Open an issue in the AutoLabeler repository
4. Contact the AutoLabeler team
