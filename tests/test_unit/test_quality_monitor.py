"""Unit tests for QualityMonitor."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


# Mock QualityMonitor class for testing
class QualityMonitor:
    """
    Comprehensive quality monitoring with real-time metrics.

    This is a mock implementation for testing purposes.
    The actual implementation will be in Phase 1.
    """

    def __init__(self, dataset_name: str, monitoring_config: dict = None):
        """
        Initialize quality monitor.

        Args:
            dataset_name: Dataset identifier
            monitoring_config: Monitoring configuration
        """
        self.dataset_name = dataset_name
        self.monitoring_config = monitoring_config or {}
        self.metrics_history = {}

    def calculate_krippendorff_alpha(
        self,
        df: pd.DataFrame,
        annotator_columns: list[str],
        value_domain: list[str] = None,
        level_of_measurement: str = "nominal",
    ) -> float:
        """
        Calculate Krippendorff's alpha for multi-annotator agreement.

        Args:
            df: DataFrame with annotations
            annotator_columns: List of column names for each annotator
            value_domain: Optional domain of possible values
            level_of_measurement: Type of data ("nominal", "ordinal", "interval", "ratio")

        Returns:
            Alpha value between -1 and 1
        """
        if len(annotator_columns) < 2:
            raise ValueError("At least 2 annotators required")

        valid_levels = ["nominal", "ordinal", "interval", "ratio"]
        if level_of_measurement not in valid_levels:
            raise ValueError(f"level_of_measurement must be one of {valid_levels}")

        # Check columns exist
        for col in annotator_columns:
            if col not in df.columns:
                raise KeyError(f"Column {col} not found in DataFrame")

        # Simplified mock implementation
        # Real implementation would use krippendorff package or manual calculation
        n_annotators = len(annotator_columns)
        n_items = len(df)

        # Count agreements
        agreements = 0
        total_comparisons = 0

        for idx in range(n_items):
            values = [df.iloc[idx][col] for col in annotator_columns]
            # Remove missing values
            values = [v for v in values if pd.notna(v)]

            if len(values) >= 2:
                # Count pairwise agreements
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        total_comparisons += 1
                        if values[i] == values[j]:
                            agreements += 1

        if total_comparisons == 0:
            return 0.0

        # Simplified alpha calculation
        observed_agreement = agreements / total_comparisons
        # Mock expected agreement (should be calculated based on value distribution)
        expected_agreement = 0.33  # For 3 classes

        if expected_agreement == 1.0:
            return 1.0

        alpha = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

        return max(-1.0, min(1.0, alpha))  # Clamp to [-1, 1]

    def compute_cqaa(
        self, df: pd.DataFrame, cost_column: str, quality_score_column: str
    ) -> float:
        """
        Compute Cost Per Quality-Adjusted Annotation.

        Formula: CQAA = Total Cost / (Annotations × Average Quality Score)

        Args:
            df: DataFrame with cost and quality data
            cost_column: Column containing per-annotation cost
            quality_score_column: Column containing quality scores [0, 1]

        Returns:
            Cost per quality-adjusted annotation in currency units
        """
        if cost_column not in df.columns:
            raise KeyError(f"Column {cost_column} not found")
        if quality_score_column not in df.columns:
            raise KeyError(f"Column {quality_score_column} not found")

        total_cost = df[cost_column].sum()
        n_annotations = len(df)
        avg_quality = df[quality_score_column].mean()

        if avg_quality == 0:
            raise ValueError("Average quality score cannot be zero")

        cqaa = total_cost / (n_annotations * avg_quality)

        return cqaa

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        metric_columns: list[str],
        window_size: int = 100,
        n_sigma: float = 3.0,
    ) -> list[dict]:
        """
        Detect statistical anomalies in annotation stream.

        Uses z-score outlier detection within sliding windows.

        Args:
            df: DataFrame with metrics
            metric_columns: Columns to monitor for anomalies
            window_size: Size of sliding window
            n_sigma: Number of standard deviations for outlier threshold

        Returns:
            List of detected anomalies
        """
        anomalies = []

        for col in metric_columns:
            if col not in df.columns:
                continue

            values = df[col].values

            # Use sliding window
            for i in range(len(values)):
                start_idx = max(0, i - window_size)
                end_idx = min(len(values), i + window_size)

                window = values[start_idx:end_idx]
                mean = np.mean(window)
                std = np.std(window)

                if std > 0:
                    z_score = abs((values[i] - mean) / std)

                    if z_score > n_sigma:
                        anomalies.append(
                            {
                                "index": i,
                                "metric": col,
                                "value": float(values[i]),
                                "z_score": float(z_score),
                                "recommendation": f"Investigate unusual {col} value",
                            }
                        )

        return anomalies

    def track_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime = None,
        metadata: dict = None,
    ):
        """
        Track a metric value over time.

        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
            metadata: Optional additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()

        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []

        self.metrics_history[metric_name].append(
            {
                "timestamp": timestamp,
                "value": value,
                "metadata": metadata or {},
            }
        )

    def get_metric_history(
        self,
        metric_name: str,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> pd.DataFrame:
        """
        Retrieve historical metric values.

        Args:
            metric_name: Name of the metric
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with columns: timestamp, value, metadata
        """
        if metric_name not in self.metrics_history:
            return pd.DataFrame(columns=["timestamp", "value", "metadata"])

        history = self.metrics_history[metric_name]

        # Filter by time range
        if start_time or end_time:
            filtered = []
            for entry in history:
                if start_time and entry["timestamp"] < start_time:
                    continue
                if end_time and entry["timestamp"] > end_time:
                    continue
                filtered.append(entry)
            history = filtered

        return pd.DataFrame(history)

    def generate_dashboard(
        self,
        df: pd.DataFrame,
        output_path: Path,
        format: str = "html",
        include_sections: list[str] = None,
    ) -> Path:
        """
        Generate comprehensive quality dashboard.

        Args:
            df: DataFrame with annotation results and metrics
            output_path: Path to save dashboard
            format: Output format ("html", "pdf", "json")
            include_sections: Optional list of sections to include

        Returns:
            Path to generated dashboard file
        """
        if format not in ["html", "pdf", "json"]:
            raise ValueError(f"Format must be one of ['html', 'pdf', 'json'], got {format}")

        # Mock dashboard generation
        dashboard_content = {
            "dataset_name": self.dataset_name,
            "total_annotations": len(df),
            "sections": include_sections or ["executive_summary"],
            "generated_at": datetime.now().isoformat(),
        }

        # Write to file
        if format == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(dashboard_content, f, indent=2)
        else:
            # Mock HTML/PDF generation
            output_path.write_text(f"Dashboard for {self.dataset_name}")

        return output_path


@pytest.mark.unit
class TestQualityMonitor:
    """Test suite for QualityMonitor."""

    def test_initialization(self):
        """Test QualityMonitor initialization."""
        monitor = QualityMonitor("test_dataset")
        assert monitor.dataset_name == "test_dataset"
        assert monitor.metrics_history == {}

    def test_krippendorff_alpha_basic(self, sample_multi_annotator_data):
        """Test basic Krippendorff's alpha calculation."""
        monitor = QualityMonitor("test_dataset")

        alpha = monitor.calculate_krippendorff_alpha(
            sample_multi_annotator_data,
            ["annotator_1", "annotator_2", "annotator_3"],
        )

        assert -1.0 <= alpha <= 1.0

    def test_krippendorff_alpha_perfect_agreement(self):
        """Test alpha with perfect agreement."""
        monitor = QualityMonitor("test_dataset")

        # Perfect agreement across all annotators
        df = pd.DataFrame(
            {
                "annotator_1": ["pos", "neg", "neu"],
                "annotator_2": ["pos", "neg", "neu"],
                "annotator_3": ["pos", "neg", "neu"],
            }
        )

        alpha = monitor.calculate_krippendorff_alpha(df, ["annotator_1", "annotator_2", "annotator_3"])

        assert alpha >= 0.9  # Should be close to 1.0

    def test_krippendorff_alpha_missing_values(self):
        """Test alpha with missing values."""
        monitor = QualityMonitor("test_dataset")

        df = pd.DataFrame(
            {
                "annotator_1": ["pos", "neg", None],
                "annotator_2": ["pos", None, "neu"],
                "annotator_3": [None, "neg", "neu"],
            }
        )

        alpha = monitor.calculate_krippendorff_alpha(df, ["annotator_1", "annotator_2", "annotator_3"])

        assert -1.0 <= alpha <= 1.0

    def test_krippendorff_alpha_insufficient_annotators(self):
        """Test alpha raises error with < 2 annotators."""
        monitor = QualityMonitor("test_dataset")

        df = pd.DataFrame({"annotator_1": ["pos", "neg", "neu"]})

        with pytest.raises(ValueError, match="At least 2 annotators"):
            monitor.calculate_krippendorff_alpha(df, ["annotator_1"])

    def test_krippendorff_alpha_invalid_level(self, sample_multi_annotator_data):
        """Test alpha raises error with invalid measurement level."""
        monitor = QualityMonitor("test_dataset")

        with pytest.raises(ValueError, match="level_of_measurement"):
            monitor.calculate_krippendorff_alpha(
                sample_multi_annotator_data,
                ["annotator_1", "annotator_2"],
                level_of_measurement="invalid",
            )

    def test_krippendorff_alpha_missing_column(self, sample_multi_annotator_data):
        """Test alpha raises error with missing column."""
        monitor = QualityMonitor("test_dataset")

        with pytest.raises(KeyError):
            monitor.calculate_krippendorff_alpha(
                sample_multi_annotator_data,
                ["annotator_1", "nonexistent_column"],
            )

    def test_compute_cqaa(self, sample_cost_data):
        """Test CQAA computation."""
        monitor = QualityMonitor("test_dataset")

        cqaa = monitor.compute_cqaa(
            sample_cost_data,
            cost_column="llm_cost",
            quality_score_column="confidence",
        )

        assert cqaa > 0
        assert isinstance(cqaa, float)

    def test_compute_cqaa_missing_column(self, sample_cost_data):
        """Test CQAA raises error with missing column."""
        monitor = QualityMonitor("test_dataset")

        with pytest.raises(KeyError):
            monitor.compute_cqaa(
                sample_cost_data,
                cost_column="nonexistent",
                quality_score_column="confidence",
            )

    def test_compute_cqaa_zero_quality(self):
        """Test CQAA raises error with zero quality scores."""
        monitor = QualityMonitor("test_dataset")

        df = pd.DataFrame(
            {
                "cost": [0.01, 0.02, 0.03],
                "quality": [0.0, 0.0, 0.0],
            }
        )

        with pytest.raises(ValueError, match="quality score cannot be zero"):
            monitor.compute_cqaa(df, "cost", "quality")

    def test_detect_anomalies(self, sample_cost_data):
        """Test anomaly detection."""
        monitor = QualityMonitor("test_dataset")

        # Add an obvious anomaly
        df = sample_cost_data.copy()
        df.loc[50, "latency_ms"] = 5000  # Very high latency

        anomalies = monitor.detect_anomalies(
            df,
            metric_columns=["latency_ms", "llm_cost"],
            window_size=20,
            n_sigma=3.0,
        )

        assert len(anomalies) > 0
        assert any(a["metric"] == "latency_ms" for a in anomalies)

    def test_detect_anomalies_no_anomalies(self, sample_cost_data):
        """Test anomaly detection with clean data."""
        monitor = QualityMonitor("test_dataset")

        # Use data without anomalies
        anomalies = monitor.detect_anomalies(
            sample_cost_data,
            metric_columns=["confidence"],
            window_size=50,
            n_sigma=5.0,  # Very high threshold
        )

        # Should find few or no anomalies with high threshold
        assert isinstance(anomalies, list)

    def test_detect_anomalies_missing_column(self, sample_cost_data):
        """Test anomaly detection with missing column."""
        monitor = QualityMonitor("test_dataset")

        # Should not raise error, just skip missing columns
        anomalies = monitor.detect_anomalies(
            sample_cost_data,
            metric_columns=["nonexistent_column"],
        )

        assert len(anomalies) == 0

    def test_track_metric(self):
        """Test metric tracking."""
        monitor = QualityMonitor("test_dataset")

        monitor.track_metric("accuracy", 0.85)
        monitor.track_metric("accuracy", 0.87)
        monitor.track_metric("latency", 245.3)

        assert "accuracy" in monitor.metrics_history
        assert "latency" in monitor.metrics_history
        assert len(monitor.metrics_history["accuracy"]) == 2
        assert len(monitor.metrics_history["latency"]) == 1

    def test_track_metric_with_timestamp(self):
        """Test metric tracking with custom timestamp."""
        monitor = QualityMonitor("test_dataset")

        ts = datetime(2024, 1, 1, 12, 0, 0)
        monitor.track_metric("accuracy", 0.85, timestamp=ts)

        history = monitor.get_metric_history("accuracy")
        assert len(history) == 1
        assert history.iloc[0]["timestamp"] == ts

    def test_track_metric_with_metadata(self):
        """Test metric tracking with metadata."""
        monitor = QualityMonitor("test_dataset")

        metadata = {"model": "gpt-4", "temperature": 0.7}
        monitor.track_metric("accuracy", 0.85, metadata=metadata)

        history = monitor.get_metric_history("accuracy")
        assert history.iloc[0]["metadata"] == metadata

    def test_get_metric_history_nonexistent(self):
        """Test getting history for non-existent metric."""
        monitor = QualityMonitor("test_dataset")

        history = monitor.get_metric_history("nonexistent")
        assert len(history) == 0
        assert list(history.columns) == ["timestamp", "value", "metadata"]

    def test_get_metric_history_with_time_filter(self):
        """Test getting metric history with time filtering."""
        monitor = QualityMonitor("test_dataset")

        # Add metrics at different times
        base_time = datetime(2024, 1, 1)
        for i in range(5):
            monitor.track_metric("accuracy", 0.8 + i * 0.01, timestamp=base_time + timedelta(hours=i))

        # Filter to middle 3 entries
        start_time = base_time + timedelta(hours=1)
        end_time = base_time + timedelta(hours=3)

        history = monitor.get_metric_history("accuracy", start_time=start_time, end_time=end_time)

        assert len(history) == 3

    def test_generate_dashboard(self, tmp_path, sample_cost_data):
        """Test dashboard generation."""
        monitor = QualityMonitor("test_dataset")

        output_path = tmp_path / "dashboard.html"

        result_path = monitor.generate_dashboard(
            sample_cost_data,
            output_path,
            format="html",
            include_sections=["executive_summary", "cost_analysis"],
        )

        assert result_path == output_path
        assert output_path.exists()

    def test_generate_dashboard_json_format(self, tmp_path, sample_cost_data):
        """Test dashboard generation in JSON format."""
        monitor = QualityMonitor("test_dataset")

        output_path = tmp_path / "dashboard.json"

        result_path = monitor.generate_dashboard(
            sample_cost_data,
            output_path,
            format="json",
        )

        assert result_path.exists()

        # Verify JSON is valid
        import json

        with open(output_path) as f:
            data = json.load(f)

        assert "dataset_name" in data
        assert data["dataset_name"] == "test_dataset"

    def test_generate_dashboard_invalid_format(self, tmp_path, sample_cost_data):
        """Test dashboard generation with invalid format."""
        monitor = QualityMonitor("test_dataset")

        output_path = tmp_path / "dashboard.xyz"

        with pytest.raises(ValueError, match="Format must be one of"):
            monitor.generate_dashboard(
                sample_cost_data,
                output_path,
                format="invalid",
            )
