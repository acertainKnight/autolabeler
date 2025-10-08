"""
DVC Manager for Dataset and Model Versioning.

This module provides a comprehensive interface for managing data and model versions
using DVC (Data Version Control). It supports:
- Dataset versioning with metadata tracking
- Model versioning with lineage tracking
- Remote storage integration (S3, Azure, GCS)
- Reproducible experiments
"""

import json
import subprocess
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from loguru import logger
import pandas as pd


@dataclass
class DVCConfig:
    """Configuration for DVC manager.

    Attributes:
        repo_path: Root path of the DVC repository
        remote_name: Name of the remote storage (e.g., 'myremote')
        remote_url: URL of the remote storage (e.g., 's3://bucket/path')
        cache_dir: Local cache directory for DVC
        auto_stage: Automatically stage changes in Git
        use_symlinks: Use symlinks for cached files
    """

    repo_path: Path
    remote_name: str = 'storage'
    remote_url: Optional[str] = None
    cache_dir: Optional[Path] = None
    auto_stage: bool = True
    use_symlinks: bool = False

    def __post_init__(self):
        """Validate and convert paths."""
        if isinstance(self.repo_path, str):
            self.repo_path = Path(self.repo_path)
        if self.cache_dir and isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class VersionMetadata:
    """Metadata for a versioned dataset or model.

    Attributes:
        version: Version identifier (e.g., 'v1.0', commit hash)
        timestamp: When this version was created
        description: Human-readable description
        metrics: Performance metrics (for models)
        tags: Searchable tags
        parent_version: Previous version for lineage tracking
        file_path: Path to the versioned file/directory
        file_hash: DVC hash for the file
        size_bytes: File size in bytes
        metadata: Additional custom metadata
    """

    version: str
    timestamp: str
    description: str
    file_path: str
    file_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_version: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DVCManager:
    """Manager for DVC-based dataset and model versioning.

    This class provides high-level operations for version control of
    data and models using DVC, with additional metadata tracking.

    Example:
        >>> config = DVCConfig(
        ...     repo_path='/path/to/repo',
        ...     remote_url='s3://mybucket/dvc-storage'
        ... )
        >>> manager = DVCManager(config)
        >>> manager.init()
        >>> manager.add_dataset('data/train.csv', 'v1.0', 'Initial training data')
    """

    def __init__(self, config: DVCConfig):
        """Initialize DVC manager.

        Args:
            config: DVC configuration
        """
        self.config = config
        self.repo_path = config.repo_path
        self.metadata_dir = self.repo_path / '.dvc' / 'metadata'
        self._dvc_available = self._check_dvc_available()

    def _check_dvc_available(self) -> bool:
        """Check if DVC is installed and available."""
        return shutil.which('dvc') is not None

    def _run_dvc_command(
        self,
        command: list[str],
        check: bool = True,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a DVC command.

        Args:
            command: DVC command parts (e.g., ['add', 'data.csv'])
            check: Raise exception on non-zero exit
            capture_output: Capture stdout/stderr

        Returns:
            CompletedProcess with command results

        Raises:
            RuntimeError: If DVC is not available
            subprocess.CalledProcessError: If command fails and check=True
        """
        if not self._dvc_available:
            raise RuntimeError(
                'DVC is not installed. Install it with: pip install dvc'
            )

        full_command = ['dvc'] + command
        logger.debug(f'Running DVC command: {" ".join(full_command)}')

        result = subprocess.run(
            full_command,
            cwd=self.repo_path,
            check=check,
            capture_output=capture_output,
            text=True
        )

        if result.returncode != 0:
            logger.error(f'DVC command failed: {result.stderr}')
        else:
            logger.debug(f'DVC command output: {result.stdout}')

        return result

    def init(self, force: bool = False) -> bool:
        """Initialize DVC in the repository.

        Args:
            force: Reinitialize if already initialized

        Returns:
            True if initialization successful
        """
        dvc_dir = self.repo_path / '.dvc'

        if dvc_dir.exists() and not force:
            logger.info('DVC already initialized')
            return True

        try:
            self._run_dvc_command(['init', '--force'] if force else ['init'])
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'Initialized DVC in {self.repo_path}')
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to initialize DVC: {e}')
            return False

    def configure_remote(
        self,
        remote_name: Optional[str] = None,
        remote_url: Optional[str] = None,
        default: bool = True
    ) -> bool:
        """Configure remote storage for DVC.

        Args:
            remote_name: Name for the remote (uses config if None)
            remote_url: Remote URL (uses config if None)
            default: Set as default remote

        Returns:
            True if configuration successful
        """
        remote_name = remote_name or self.config.remote_name
        remote_url = remote_url or self.config.remote_url

        if not remote_url:
            logger.warning('No remote URL provided')
            return False

        try:
            # Add or modify remote
            self._run_dvc_command([
                'remote', 'add', '-f', remote_name, remote_url
            ])

            # Set as default if requested
            if default:
                self._run_dvc_command([
                    'remote', 'default', remote_name
                ])

            logger.info(f'Configured remote {remote_name}: {remote_url}')
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to configure remote: {e}')
            return False

    def add_dataset(
        self,
        file_path: str | Path,
        version: str,
        description: str,
        tags: Optional[list[str]] = None,
        parent_version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> Optional[VersionMetadata]:
        """Add and version a dataset.

        Args:
            file_path: Path to the dataset file/directory
            version: Version identifier
            description: Description of this version
            tags: Optional tags for categorization
            parent_version: Previous version for lineage
            metadata: Additional metadata

        Returns:
            VersionMetadata if successful, None otherwise
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f'File not found: {file_path}')
            return None

        try:
            # Add file to DVC
            result = self._run_dvc_command(['add', str(file_path)])

            # Get file hash from .dvc file
            dvc_file = file_path.with_suffix(file_path.suffix + '.dvc')
            file_hash = None
            if dvc_file.exists():
                with open(dvc_file) as f:
                    dvc_data = json.load(f) if dvc_file.suffix == '.json' else {}
                    # DVC files are YAML, simplified for now
                    file_hash = 'dvc-tracked'

            # Create metadata
            version_meta = VersionMetadata(
                version=version,
                timestamp=datetime.now().isoformat(),
                description=description,
                file_path=str(file_path),
                file_hash=file_hash,
                size_bytes=self._get_size(file_path),
                tags=tags or [],
                parent_version=parent_version,
                metadata=metadata or {}
            )

            # Save metadata
            self._save_metadata(version_meta, 'dataset')

            logger.info(f'Added dataset version {version}: {file_path}')
            return version_meta

        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to add dataset: {e}')
            return None

    def add_model(
        self,
        model_path: str | Path,
        version: str,
        description: str,
        metrics: Optional[dict[str, float]] = None,
        tags: Optional[list[str]] = None,
        parent_version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> Optional[VersionMetadata]:
        """Add and version a model.

        Args:
            model_path: Path to the model file/directory
            version: Version identifier
            description: Description of this version
            metrics: Performance metrics
            tags: Optional tags for categorization
            parent_version: Previous version for lineage
            metadata: Additional metadata (e.g., hyperparameters)

        Returns:
            VersionMetadata if successful, None otherwise
        """
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f'Model not found: {model_path}')
            return None

        try:
            # Add model to DVC
            self._run_dvc_command(['add', str(model_path)])

            # Create metadata
            version_meta = VersionMetadata(
                version=version,
                timestamp=datetime.now().isoformat(),
                description=description,
                file_path=str(model_path),
                file_hash='dvc-tracked',
                size_bytes=self._get_size(model_path),
                metrics=metrics or {},
                tags=tags or [],
                parent_version=parent_version,
                metadata=metadata or {}
            )

            # Save metadata
            self._save_metadata(version_meta, 'model')

            logger.info(f'Added model version {version}: {model_path}')
            return version_meta

        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to add model: {e}')
            return None

    def checkout_version(
        self,
        file_path: str | Path,
        version: Optional[str] = None
    ) -> bool:
        """Checkout a specific version of a file.

        Args:
            file_path: Path to the file
            version: Git revision/tag to checkout (None for latest)

        Returns:
            True if successful
        """
        try:
            if version:
                # Checkout specific git version first
                self._run_dvc_command(['checkout', str(file_path), '--rev', version])
            else:
                # Checkout current version
                self._run_dvc_command(['checkout', str(file_path)])

            logger.info(f'Checked out {file_path} at version {version or "latest"}')
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to checkout version: {e}')
            return False

    def push(self, remote: Optional[str] = None) -> bool:
        """Push tracked data to remote storage.

        Args:
            remote: Remote name (uses default if None)

        Returns:
            True if successful
        """
        try:
            cmd = ['push']
            if remote:
                cmd.extend(['--remote', remote])

            self._run_dvc_command(cmd)
            logger.info('Pushed data to remote')
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to push: {e}')
            return False

    def pull(self, remote: Optional[str] = None) -> bool:
        """Pull tracked data from remote storage.

        Args:
            remote: Remote name (uses default if None)

        Returns:
            True if successful
        """
        try:
            cmd = ['pull']
            if remote:
                cmd.extend(['--remote', remote])

            self._run_dvc_command(cmd)
            logger.info('Pulled data from remote')
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to pull: {e}')
            return False

    def list_versions(
        self,
        version_type: str = 'all'
    ) -> list[VersionMetadata]:
        """List all tracked versions.

        Args:
            version_type: Type of versions ('dataset', 'model', or 'all')

        Returns:
            List of version metadata
        """
        versions = []

        if not self.metadata_dir.exists():
            return versions

        types_to_check = []
        if version_type in ('dataset', 'all'):
            types_to_check.append('dataset')
        if version_type in ('model', 'all'):
            types_to_check.append('model')

        for vtype in types_to_check:
            type_dir = self.metadata_dir / vtype
            if type_dir.exists():
                for meta_file in type_dir.glob('*.json'):
                    try:
                        with open(meta_file) as f:
                            data = json.load(f)
                            versions.append(VersionMetadata(**data))
                    except Exception as e:
                        logger.warning(f'Failed to load metadata {meta_file}: {e}')

        return sorted(versions, key=lambda v: v.timestamp, reverse=True)

    def get_version_info(
        self,
        version: str,
        version_type: str = 'all'
    ) -> Optional[VersionMetadata]:
        """Get information about a specific version.

        Args:
            version: Version identifier
            version_type: Type of version ('dataset', 'model', or 'all')

        Returns:
            VersionMetadata if found, None otherwise
        """
        versions = self.list_versions(version_type)
        for v in versions:
            if v.version == version:
                return v
        return None

    def compare_versions(
        self,
        version1: str,
        version2: str,
        version_type: str = 'all'
    ) -> dict[str, Any]:
        """Compare two versions.

        Args:
            version1: First version identifier
            version2: Second version identifier
            version_type: Type of versions to compare

        Returns:
            Dictionary with comparison results
        """
        v1 = self.get_version_info(version1, version_type)
        v2 = self.get_version_info(version2, version_type)

        if not v1 or not v2:
            return {'error': 'One or both versions not found'}

        comparison = {
            'version1': version1,
            'version2': version2,
            'timestamp_diff': v2.timestamp > v1.timestamp,
            'size_diff_bytes': (v2.size_bytes or 0) - (v1.size_bytes or 0),
        }

        # Compare metrics if both have them
        if v1.metrics and v2.metrics:
            metric_diffs = {}
            all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
            for metric in all_metrics:
                m1 = v1.metrics.get(metric, 0)
                m2 = v2.metrics.get(metric, 0)
                metric_diffs[metric] = {
                    'v1': m1,
                    'v2': m2,
                    'diff': m2 - m1,
                    'percent_change': ((m2 - m1) / m1 * 100) if m1 != 0 else None
                }
            comparison['metrics'] = metric_diffs

        return comparison

    def get_lineage(self, version: str, version_type: str = 'all') -> list[str]:
        """Get the lineage chain for a version.

        Args:
            version: Version identifier
            version_type: Type of version

        Returns:
            List of versions in lineage chain (oldest to newest)
        """
        lineage = []
        current = self.get_version_info(version, version_type)

        while current:
            lineage.insert(0, current.version)
            if current.parent_version:
                current = self.get_version_info(current.parent_version, version_type)
            else:
                break

        return lineage

    def _save_metadata(self, metadata: VersionMetadata, version_type: str):
        """Save version metadata to disk.

        Args:
            metadata: Version metadata to save
            version_type: Type of version ('dataset' or 'model')
        """
        type_dir = self.metadata_dir / version_type
        type_dir.mkdir(parents=True, exist_ok=True)

        meta_file = type_dir / f'{metadata.version}.json'
        with open(meta_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

    def _get_size(self, path: Path) -> int:
        """Get size of file or directory in bytes.

        Args:
            path: Path to measure

        Returns:
            Size in bytes
        """
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0

    def export_metadata_report(
        self,
        output_path: str | Path,
        version_type: str = 'all'
    ) -> bool:
        """Export metadata report to CSV.

        Args:
            output_path: Path for output CSV file
            version_type: Type of versions to export

        Returns:
            True if successful
        """
        versions = self.list_versions(version_type)

        if not versions:
            logger.warning('No versions found to export')
            return False

        try:
            # Convert to DataFrame
            data = []
            for v in versions:
                row = {
                    'version': v.version,
                    'timestamp': v.timestamp,
                    'description': v.description,
                    'file_path': v.file_path,
                    'size_mb': (v.size_bytes or 0) / 1024 / 1024,
                    'parent_version': v.parent_version or '',
                    'tags': ','.join(v.tags),
                }
                # Add metrics as columns
                for key, value in v.metrics.items():
                    row[f'metric_{key}'] = value
                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

            logger.info(f'Exported metadata report to {output_path}')
            return True

        except Exception as e:
            logger.error(f'Failed to export metadata report: {e}')
            return False
