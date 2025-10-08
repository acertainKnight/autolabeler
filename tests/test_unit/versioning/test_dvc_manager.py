"""Unit tests for DVC Manager.

Tests cover:
- DVC initialization and configuration
- Dataset versioning
- Model versioning
- Remote storage operations
- Version metadata management
- Version comparison and lineage
- Error handling
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import pandas as pd

from autolabeler.core.versioning import DVCManager, DVCConfig, VersionMetadata


@pytest.fixture
def dvc_config(tmp_path):
    """Create test DVC configuration."""
    return DVCConfig(
        repo_path=tmp_path,
        remote_name='test-remote',
        remote_url='s3://test-bucket/dvc-cache',
        auto_stage=True
    )


@pytest.fixture
def dvc_manager(dvc_config):
    """Create DVC manager instance."""
    return DVCManager(dvc_config)


@pytest.fixture
def mock_dvc_available(monkeypatch):
    """Mock DVC availability check."""
    monkeypatch.setattr('shutil.which', lambda x: '/usr/bin/dvc' if x == 'dvc' else None)


@pytest.fixture
def sample_dataset_file(tmp_path):
    """Create a sample dataset file."""
    data_file = tmp_path / 'train.csv'
    df = pd.DataFrame({
        'text': ['sample 1', 'sample 2', 'sample 3'],
        'label': ['positive', 'negative', 'neutral']
    })
    df.to_csv(data_file, index=False)
    return data_file


@pytest.fixture
def sample_model_file(tmp_path):
    """Create a sample model file."""
    model_file = tmp_path / 'model.pkl'
    model_file.write_text('mock model content')
    return model_file


class TestDVCConfig:
    """Test DVC configuration."""

    def test_config_initialization(self, tmp_path):
        """Test basic config initialization."""
        config = DVCConfig(
            repo_path=tmp_path,
            remote_name='myremote',
            remote_url='s3://bucket/path'
        )

        assert config.repo_path == tmp_path
        assert config.remote_name == 'myremote'
        assert config.remote_url == 's3://bucket/path'
        assert config.auto_stage is True  # default
        assert config.use_symlinks is False  # default

    def test_config_path_conversion(self):
        """Test automatic path conversion."""
        config = DVCConfig(repo_path='/tmp/test')

        assert isinstance(config.repo_path, Path)
        assert config.repo_path == Path('/tmp/test')

    def test_config_cache_dir(self, tmp_path):
        """Test custom cache directory."""
        cache = tmp_path / 'cache'
        config = DVCConfig(
            repo_path=tmp_path,
            cache_dir=cache
        )

        assert config.cache_dir == cache
        assert isinstance(config.cache_dir, Path)

    def test_config_defaults(self, tmp_path):
        """Test default configuration values."""
        config = DVCConfig(repo_path=tmp_path)

        assert config.remote_name == 'storage'
        assert config.remote_url is None
        assert config.cache_dir is None
        assert config.auto_stage is True
        assert config.use_symlinks is False


class TestDVCManagerInitialization:
    """Test DVC manager initialization."""

    def test_manager_creation(self, dvc_manager, dvc_config):
        """Test manager creation with config."""
        assert dvc_manager.config == dvc_config
        assert dvc_manager.repo_path == dvc_config.repo_path
        assert dvc_manager.metadata_dir == dvc_config.repo_path / '.dvc' / 'metadata'

    def test_dvc_availability_check(self, dvc_manager):
        """Test DVC availability detection."""
        # Actual availability depends on environment
        assert isinstance(dvc_manager._dvc_available, bool)

    def test_dvc_not_available(self, dvc_manager, monkeypatch):
        """Test handling when DVC is not installed."""
        monkeypatch.setattr('shutil.which', lambda x: None)
        manager = DVCManager(dvc_manager.config)

        assert not manager._dvc_available

        with pytest.raises(RuntimeError, match='DVC is not installed'):
            manager._run_dvc_command(['init'])


class TestDVCCommands:
    """Test DVC command execution."""

    @patch('subprocess.run')
    def test_run_dvc_command_success(self, mock_run, dvc_manager, mock_dvc_available):
        """Test successful DVC command execution."""
        mock_run.return_value = Mock(returncode=0, stdout='success', stderr='')

        result = dvc_manager._run_dvc_command(['init'])

        assert result.returncode == 0
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert args[0][0] == ['dvc', 'init']
        assert args[1]['cwd'] == dvc_manager.repo_path

    @patch('subprocess.run')
    def test_run_dvc_command_failure(self, mock_run, dvc_manager, mock_dvc_available):
        """Test DVC command failure handling."""
        mock_run.return_value = Mock(returncode=1, stdout='', stderr='error')

        result = dvc_manager._run_dvc_command(['invalid'], check=False)

        assert result.returncode == 1

    @patch('subprocess.run')
    def test_run_dvc_command_with_check(self, mock_run, dvc_manager, mock_dvc_available):
        """Test DVC command with check=True."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'dvc')

        with pytest.raises(subprocess.CalledProcessError):
            dvc_manager._run_dvc_command(['invalid'], check=True)


class TestDVCInitialization:
    """Test DVC repository initialization."""

    @patch.object(DVCManager, '_run_dvc_command')
    def test_init_new_repo(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test initializing new DVC repo."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.init()

        assert result is True
        mock_cmd.assert_called_once_with(['init'])
        assert dvc_manager.metadata_dir.exists()

    @patch.object(DVCManager, '_run_dvc_command')
    def test_init_existing_repo(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test initializing existing DVC repo."""
        # Create .dvc directory
        (dvc_manager.repo_path / '.dvc').mkdir()

        result = dvc_manager.init()

        assert result is True
        mock_cmd.assert_not_called()  # Should not reinit

    @patch.object(DVCManager, '_run_dvc_command')
    def test_init_force_reinit(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test force reinitialize existing repo."""
        (dvc_manager.repo_path / '.dvc').mkdir()
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.init(force=True)

        assert result is True
        mock_cmd.assert_called_once_with(['init', '--force'])

    @patch.object(DVCManager, '_run_dvc_command')
    def test_init_failure(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test initialization failure."""
        mock_cmd.side_effect = subprocess.CalledProcessError(1, 'dvc')

        result = dvc_manager.init()

        assert result is False


class TestRemoteConfiguration:
    """Test DVC remote storage configuration."""

    @patch.object(DVCManager, '_run_dvc_command')
    def test_configure_remote_basic(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test basic remote configuration."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.configure_remote()

        assert result is True
        assert mock_cmd.call_count == 2  # add remote + set default
        calls = mock_cmd.call_args_list
        assert calls[0][0][0] == ['remote', 'add', '-f', 'test-remote', 's3://test-bucket/dvc-cache']
        assert calls[1][0][0] == ['remote', 'default', 'test-remote']

    @patch.object(DVCManager, '_run_dvc_command')
    def test_configure_remote_no_default(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test remote configuration without setting default."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.configure_remote(default=False)

        assert result is True
        assert mock_cmd.call_count == 1  # only add remote

    @patch.object(DVCManager, '_run_dvc_command')
    def test_configure_remote_custom(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test remote configuration with custom values."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.configure_remote(
            remote_name='custom',
            remote_url='azure://container/path'
        )

        assert result is True
        calls = mock_cmd.call_args_list
        assert 'custom' in calls[0][0][0]
        assert 'azure://container/path' in calls[0][0][0]

    def test_configure_remote_no_url(self, dvc_manager):
        """Test remote configuration without URL."""
        dvc_manager.config.remote_url = None

        result = dvc_manager.configure_remote()

        assert result is False

    @patch.object(DVCManager, '_run_dvc_command')
    def test_configure_remote_failure(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test remote configuration failure."""
        mock_cmd.side_effect = subprocess.CalledProcessError(1, 'dvc')

        result = dvc_manager.configure_remote()

        assert result is False


class TestDatasetVersioning:
    """Test dataset versioning operations."""

    @patch.object(DVCManager, '_run_dvc_command')
    def test_add_dataset_success(self, mock_cmd, dvc_manager, sample_dataset_file, mock_dvc_available):
        """Test successfully adding a dataset."""
        mock_cmd.return_value = Mock(returncode=0)

        metadata = dvc_manager.add_dataset(
            file_path=sample_dataset_file,
            version='v1.0',
            description='Test dataset',
            tags=['train', 'test'],
            metadata={'source': 'test'}
        )

        assert metadata is not None
        assert metadata.version == 'v1.0'
        assert metadata.description == 'Test dataset'
        assert metadata.tags == ['train', 'test']
        assert metadata.metadata['source'] == 'test'
        assert metadata.file_path == str(sample_dataset_file)
        assert metadata.size_bytes > 0

        # Check command was called
        mock_cmd.assert_called_once()
        assert 'add' in mock_cmd.call_args[0][0]

    @patch.object(DVCManager, '_run_dvc_command')
    def test_add_dataset_with_parent(self, mock_cmd, dvc_manager, sample_dataset_file, mock_dvc_available):
        """Test adding dataset with parent version."""
        mock_cmd.return_value = Mock(returncode=0)

        metadata = dvc_manager.add_dataset(
            file_path=sample_dataset_file,
            version='v1.1',
            description='Updated dataset',
            parent_version='v1.0'
        )

        assert metadata.version == 'v1.1'
        assert metadata.parent_version == 'v1.0'

    def test_add_dataset_file_not_found(self, dvc_manager):
        """Test adding non-existent dataset."""
        metadata = dvc_manager.add_dataset(
            file_path='/nonexistent/file.csv',
            version='v1.0',
            description='Should fail'
        )

        assert metadata is None

    @patch.object(DVCManager, '_run_dvc_command')
    def test_add_dataset_dvc_failure(self, mock_cmd, dvc_manager, sample_dataset_file, mock_dvc_available):
        """Test dataset addition with DVC command failure."""
        mock_cmd.side_effect = subprocess.CalledProcessError(1, 'dvc')

        metadata = dvc_manager.add_dataset(
            file_path=sample_dataset_file,
            version='v1.0',
            description='Should fail'
        )

        assert metadata is None


class TestModelVersioning:
    """Test model versioning operations."""

    @patch.object(DVCManager, '_run_dvc_command')
    def test_add_model_success(self, mock_cmd, dvc_manager, sample_model_file, mock_dvc_available):
        """Test successfully adding a model."""
        mock_cmd.return_value = Mock(returncode=0)

        metadata = dvc_manager.add_model(
            model_path=sample_model_file,
            version='v1.0',
            description='Test model',
            metrics={'accuracy': 0.85, 'f1': 0.83},
            tags=['production'],
            metadata={'hyperparameters': {'lr': 0.001}}
        )

        assert metadata is not None
        assert metadata.version == 'v1.0'
        assert metadata.description == 'Test model'
        assert metadata.metrics == {'accuracy': 0.85, 'f1': 0.83}
        assert metadata.tags == ['production']
        assert metadata.metadata['hyperparameters']['lr'] == 0.001

    @patch.object(DVCManager, '_run_dvc_command')
    def test_add_model_with_parent(self, mock_cmd, dvc_manager, sample_model_file, mock_dvc_available):
        """Test adding model with parent version."""
        mock_cmd.return_value = Mock(returncode=0)

        metadata = dvc_manager.add_model(
            model_path=sample_model_file,
            version='v2.0',
            description='Improved model',
            parent_version='v1.0',
            metrics={'accuracy': 0.90}
        )

        assert metadata.version == 'v2.0'
        assert metadata.parent_version == 'v1.0'
        assert metadata.metrics['accuracy'] == 0.90

    def test_add_model_file_not_found(self, dvc_manager):
        """Test adding non-existent model."""
        metadata = dvc_manager.add_model(
            model_path='/nonexistent/model.pkl',
            version='v1.0',
            description='Should fail'
        )

        assert metadata is None


class TestVersionCheckout:
    """Test version checkout operations."""

    @patch.object(DVCManager, '_run_dvc_command')
    def test_checkout_specific_version(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test checking out a specific version."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.checkout_version('data/train.csv', version='v1.0')

        assert result is True
        mock_cmd.assert_called_once()
        cmd = mock_cmd.call_args[0][0]
        assert 'checkout' in cmd
        assert 'data/train.csv' in cmd
        assert '--rev' in cmd
        assert 'v1.0' in cmd

    @patch.object(DVCManager, '_run_dvc_command')
    def test_checkout_latest_version(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test checking out latest version."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.checkout_version('data/train.csv')

        assert result is True
        cmd = mock_cmd.call_args[0][0]
        assert 'checkout' in cmd
        assert '--rev' not in cmd

    @patch.object(DVCManager, '_run_dvc_command')
    def test_checkout_failure(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test checkout failure."""
        mock_cmd.side_effect = subprocess.CalledProcessError(1, 'dvc')

        result = dvc_manager.checkout_version('data/train.csv', version='v1.0')

        assert result is False


class TestRemoteOperations:
    """Test push/pull operations."""

    @patch.object(DVCManager, '_run_dvc_command')
    def test_push_default_remote(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test pushing to default remote."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.push()

        assert result is True
        mock_cmd.assert_called_once_with(['push'])

    @patch.object(DVCManager, '_run_dvc_command')
    def test_push_specific_remote(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test pushing to specific remote."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.push(remote='custom-remote')

        assert result is True
        cmd = mock_cmd.call_args[0][0]
        assert cmd == ['push', '--remote', 'custom-remote']

    @patch.object(DVCManager, '_run_dvc_command')
    def test_push_failure(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test push failure."""
        mock_cmd.side_effect = subprocess.CalledProcessError(1, 'dvc')

        result = dvc_manager.push()

        assert result is False

    @patch.object(DVCManager, '_run_dvc_command')
    def test_pull_default_remote(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test pulling from default remote."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.pull()

        assert result is True
        mock_cmd.assert_called_once_with(['pull'])

    @patch.object(DVCManager, '_run_dvc_command')
    def test_pull_specific_remote(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test pulling from specific remote."""
        mock_cmd.return_value = Mock(returncode=0)

        result = dvc_manager.pull(remote='custom-remote')

        assert result is True
        cmd = mock_cmd.call_args[0][0]
        assert cmd == ['pull', '--remote', 'custom-remote']

    @patch.object(DVCManager, '_run_dvc_command')
    def test_pull_failure(self, mock_cmd, dvc_manager, mock_dvc_available):
        """Test pull failure."""
        mock_cmd.side_effect = subprocess.CalledProcessError(1, 'dvc')

        result = dvc_manager.pull()

        assert result is False


class TestVersionListing:
    """Test version listing and retrieval."""

    def test_list_versions_empty(self, dvc_manager):
        """Test listing versions when none exist."""
        versions = dvc_manager.list_versions()

        assert versions == []

    def test_list_versions_with_data(self, dvc_manager):
        """Test listing versions with saved metadata."""
        # Create metadata directory
        dvc_manager.metadata_dir.mkdir(parents=True)
        dataset_dir = dvc_manager.metadata_dir / 'dataset'
        dataset_dir.mkdir()

        # Save test metadata
        metadata1 = VersionMetadata(
            version='v1.0',
            timestamp='2025-01-01T10:00:00',
            description='Test v1',
            file_path='data/train.csv'
        )
        metadata2 = VersionMetadata(
            version='v2.0',
            timestamp='2025-01-02T10:00:00',
            description='Test v2',
            file_path='data/train.csv'
        )

        with open(dataset_dir / 'v1.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v1.0',
                    'timestamp': '2025-01-01T10:00:00',
                    'description': 'Test v1',
                    'file_path': 'data/train.csv',
                    'file_hash': None,
                    'size_bytes': None,
                    'metrics': {},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )
        with open(dataset_dir / 'v2.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v2.0',
                    'timestamp': '2025-01-02T10:00:00',
                    'description': 'Test v2',
                    'file_path': 'data/train.csv',
                    'file_hash': None,
                    'size_bytes': None,
                    'metrics': {},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        versions = dvc_manager.list_versions(version_type='dataset')

        assert len(versions) == 2
        # Should be sorted by timestamp (newest first)
        assert versions[0].version == 'v2.0'
        assert versions[1].version == 'v1.0'

    def test_list_versions_by_type(self, dvc_manager):
        """Test filtering versions by type."""
        dvc_manager.metadata_dir.mkdir(parents=True)

        # Create dataset version
        dataset_dir = dvc_manager.metadata_dir / 'dataset'
        dataset_dir.mkdir()
        with open(dataset_dir / 'v1.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v1.0',
                    'timestamp': '2025-01-01T10:00:00',
                    'description': 'Dataset',
                    'file_path': 'data/train.csv',
                    'file_hash': None,
                    'size_bytes': None,
                    'metrics': {},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        # Create model version
        model_dir = dvc_manager.metadata_dir / 'model'
        model_dir.mkdir()
        with open(model_dir / 'v1.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v1.0',
                    'timestamp': '2025-01-01T10:00:00',
                    'description': 'Model',
                    'file_path': 'models/clf.pkl',
                    'file_hash': None,
                    'size_bytes': None,
                    'metrics': {},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        # Test filtering
        dataset_versions = dvc_manager.list_versions(version_type='dataset')
        model_versions = dvc_manager.list_versions(version_type='model')
        all_versions = dvc_manager.list_versions(version_type='all')

        assert len(dataset_versions) == 1
        assert len(model_versions) == 1
        assert len(all_versions) == 2


class TestVersionInfo:
    """Test getting version information."""

    def test_get_version_info_not_found(self, dvc_manager):
        """Test getting info for non-existent version."""
        info = dvc_manager.get_version_info('v99.0')

        assert info is None

    def test_get_version_info_success(self, dvc_manager):
        """Test getting version info successfully."""
        # Setup metadata
        dvc_manager.metadata_dir.mkdir(parents=True)
        dataset_dir = dvc_manager.metadata_dir / 'dataset'
        dataset_dir.mkdir()

        with open(dataset_dir / 'v1.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v1.0',
                    'timestamp': '2025-01-01T10:00:00',
                    'description': 'Test version',
                    'file_path': 'data/train.csv',
                    'file_hash': 'abc123',
                    'size_bytes': 1024,
                    'metrics': {'accuracy': 0.85},
                    'tags': ['production'],
                    'parent_version': None,
                    'metadata': {'key': 'value'}
                },
                f
            )

        info = dvc_manager.get_version_info('v1.0')

        assert info is not None
        assert info.version == 'v1.0'
        assert info.description == 'Test version'
        assert info.metrics['accuracy'] == 0.85
        assert 'production' in info.tags


class TestVersionComparison:
    """Test version comparison functionality."""

    def test_compare_versions_not_found(self, dvc_manager):
        """Test comparison with non-existent versions."""
        result = dvc_manager.compare_versions('v1.0', 'v2.0')

        assert 'error' in result

    def test_compare_versions_success(self, dvc_manager):
        """Test successful version comparison."""
        # Setup metadata
        dvc_manager.metadata_dir.mkdir(parents=True)
        dataset_dir = dvc_manager.metadata_dir / 'dataset'
        dataset_dir.mkdir()

        # V1.0
        with open(dataset_dir / 'v1.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v1.0',
                    'timestamp': '2025-01-01T10:00:00',
                    'description': 'Version 1',
                    'file_path': 'data/train.csv',
                    'file_hash': None,
                    'size_bytes': 1000,
                    'metrics': {'accuracy': 0.80},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        # V2.0
        with open(dataset_dir / 'v2.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v2.0',
                    'timestamp': '2025-01-02T10:00:00',
                    'description': 'Version 2',
                    'file_path': 'data/train.csv',
                    'file_hash': None,
                    'size_bytes': 2000,
                    'metrics': {'accuracy': 0.90},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        result = dvc_manager.compare_versions('v1.0', 'v2.0')

        assert result['version1'] == 'v1.0'
        assert result['version2'] == 'v2.0'
        assert result['size_diff_bytes'] == 1000
        assert result['metrics']['accuracy']['v1'] == 0.80
        assert result['metrics']['accuracy']['v2'] == 0.90
        assert result['metrics']['accuracy']['diff'] == 0.10

    def test_compare_versions_percent_change(self, dvc_manager):
        """Test percent change calculation in comparison."""
        # Setup metadata
        dvc_manager.metadata_dir.mkdir(parents=True)
        model_dir = dvc_manager.metadata_dir / 'model'
        model_dir.mkdir()

        # V1.0
        with open(model_dir / 'v1.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v1.0',
                    'timestamp': '2025-01-01T10:00:00',
                    'description': 'Version 1',
                    'file_path': 'model.pkl',
                    'file_hash': None,
                    'size_bytes': None,
                    'metrics': {'accuracy': 0.80},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        # V2.0
        with open(model_dir / 'v2.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v2.0',
                    'timestamp': '2025-01-02T10:00:00',
                    'description': 'Version 2',
                    'file_path': 'model.pkl',
                    'file_hash': None,
                    'size_bytes': None,
                    'metrics': {'accuracy': 0.96},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        result = dvc_manager.compare_versions('v1.0', 'v2.0', version_type='model')

        assert result['metrics']['accuracy']['percent_change'] == pytest.approx(20.0)


class TestVersionLineage:
    """Test version lineage tracking."""

    def test_lineage_single_version(self, dvc_manager):
        """Test lineage for version with no parent."""
        # Setup metadata
        dvc_manager.metadata_dir.mkdir(parents=True)
        dataset_dir = dvc_manager.metadata_dir / 'dataset'
        dataset_dir.mkdir()

        with open(dataset_dir / 'v1.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v1.0',
                    'timestamp': '2025-01-01T10:00:00',
                    'description': 'Base version',
                    'file_path': 'data/train.csv',
                    'file_hash': None,
                    'size_bytes': None,
                    'metrics': {},
                    'tags': [],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        lineage = dvc_manager.get_lineage('v1.0')

        assert lineage == ['v1.0']

    def test_lineage_multiple_versions(self, dvc_manager):
        """Test lineage chain across multiple versions."""
        # Setup metadata
        dvc_manager.metadata_dir.mkdir(parents=True)
        dataset_dir = dvc_manager.metadata_dir / 'dataset'
        dataset_dir.mkdir()

        # Create version chain: v1.0 -> v1.1 -> v1.2
        for version, parent in [('v1.0', None), ('v1.1', 'v1.0'), ('v1.2', 'v1.1')]:
            with open(dataset_dir / f'{version}.json', 'w') as f:
                json.dump(
                    {
                        'version': version,
                        'timestamp': '2025-01-01T10:00:00',
                        'description': f'Version {version}',
                        'file_path': 'data/train.csv',
                        'file_hash': None,
                        'size_bytes': None,
                        'metrics': {},
                        'tags': [],
                        'parent_version': parent,
                        'metadata': {}
                    },
                    f
                )

        lineage = dvc_manager.get_lineage('v1.2')

        assert lineage == ['v1.0', 'v1.1', 'v1.2']

    def test_lineage_not_found(self, dvc_manager):
        """Test lineage for non-existent version."""
        lineage = dvc_manager.get_lineage('v99.0')

        assert lineage == []


class TestMetadataReport:
    """Test metadata export functionality."""

    @patch.object(DVCManager, '_run_dvc_command')
    def test_export_metadata_report_success(self, mock_cmd, dvc_manager, tmp_path, mock_dvc_available):
        """Test successful metadata report export."""
        # Setup metadata
        dvc_manager.metadata_dir.mkdir(parents=True)
        dataset_dir = dvc_manager.metadata_dir / 'dataset'
        dataset_dir.mkdir()

        with open(dataset_dir / 'v1.0.json', 'w') as f:
            json.dump(
                {
                    'version': 'v1.0',
                    'timestamp': '2025-01-01T10:00:00',
                    'description': 'Test version',
                    'file_path': 'data/train.csv',
                    'file_hash': 'abc123',
                    'size_bytes': 1024000,
                    'metrics': {'accuracy': 0.85, 'f1': 0.83},
                    'tags': ['production'],
                    'parent_version': None,
                    'metadata': {}
                },
                f
            )

        output_file = tmp_path / 'report.csv'
        result = dvc_manager.export_metadata_report(output_file)

        assert result is True
        assert output_file.exists()

        # Verify CSV content
        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert df['version'].iloc[0] == 'v1.0'
        assert df['metric_accuracy'].iloc[0] == 0.85
        assert df['metric_f1'].iloc[0] == 0.83
        assert 'production' in df['tags'].iloc[0]

    def test_export_metadata_report_no_versions(self, dvc_manager, tmp_path):
        """Test export with no versions available."""
        output_file = tmp_path / 'report.csv'
        result = dvc_manager.export_metadata_report(output_file)

        assert result is False
        assert not output_file.exists()


class TestHelperMethods:
    """Test helper methods."""

    def test_get_size_file(self, tmp_path):
        """Test getting file size."""
        test_file = tmp_path / 'test.txt'
        test_file.write_text('hello world')

        manager = DVCManager(DVCConfig(repo_path=tmp_path))
        size = manager._get_size(test_file)

        assert size == 11  # "hello world" is 11 bytes

    def test_get_size_directory(self, tmp_path):
        """Test getting directory size."""
        test_dir = tmp_path / 'testdir'
        test_dir.mkdir()
        (test_dir / 'file1.txt').write_text('hello')
        (test_dir / 'file2.txt').write_text('world')

        manager = DVCManager(DVCConfig(repo_path=tmp_path))
        size = manager._get_size(test_dir)

        assert size == 10  # 5 + 5 bytes

    def test_save_metadata(self, dvc_manager):
        """Test saving metadata to disk."""
        metadata = VersionMetadata(
            version='v1.0',
            timestamp='2025-01-01T10:00:00',
            description='Test',
            file_path='data/test.csv',
            metrics={'accuracy': 0.85}
        )

        dvc_manager._save_metadata(metadata, 'dataset')

        # Verify file was created
        meta_file = dvc_manager.metadata_dir / 'dataset' / 'v1.0.json'
        assert meta_file.exists()

        # Verify content
        with open(meta_file) as f:
            data = json.load(f)
            assert data['version'] == 'v1.0'
            assert data['metrics']['accuracy'] == 0.85


@pytest.mark.unit
class TestDVCManagerIntegration:
    """Integration tests for DVC manager (unit level)."""

    @patch.object(DVCManager, '_run_dvc_command')
    def test_full_workflow(self, mock_cmd, dvc_manager, sample_dataset_file, mock_dvc_available):
        """Test complete workflow: init -> add -> push -> list."""
        mock_cmd.return_value = Mock(returncode=0)

        # Initialize
        assert dvc_manager.init() is True

        # Configure remote
        assert dvc_manager.configure_remote() is True

        # Add dataset
        metadata = dvc_manager.add_dataset(
            file_path=sample_dataset_file,
            version='v1.0',
            description='Initial dataset',
            tags=['train']
        )
        assert metadata is not None

        # Push
        assert dvc_manager.push() is True

        # List versions
        versions = dvc_manager.list_versions()
        assert len(versions) == 1
        assert versions[0].version == 'v1.0'

    @patch.object(DVCManager, '_run_dvc_command')
    def test_version_evolution_workflow(self, mock_cmd, dvc_manager, sample_dataset_file, tmp_path, mock_dvc_available):
        """Test evolving versions with lineage."""
        mock_cmd.return_value = Mock(returncode=0)

        # V1.0
        v1 = dvc_manager.add_dataset(
            file_path=sample_dataset_file,
            version='v1.0',
            description='Initial dataset'
        )

        # V1.1 (derived from v1.0)
        dataset_v2 = tmp_path / 'train_v2.csv'
        dataset_v2.write_text('updated data')

        v2 = dvc_manager.add_dataset(
            file_path=dataset_v2,
            version='v1.1',
            description='Cleaned dataset',
            parent_version='v1.0'
        )

        # Check lineage
        lineage = dvc_manager.get_lineage('v1.1')
        assert lineage == ['v1.0', 'v1.1']
