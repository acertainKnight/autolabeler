"""
Streamlit dashboard for quality monitoring and confidence calibration.

Run with: streamlit run quality_dashboard.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Install with: pip install plotly")

from ..core.quality import ConfidenceCalibrator, QualityMonitor


# Page configuration
st.set_page_config(
    page_title="AutoLabeler Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV or Parquet file."""
    try:
        path = Path(file_path)
        if path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        else:
            st.error(f"Unsupported file format: {path.suffix}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()


def plot_calibration_curve(calibration_data: dict[str, Any]) -> None:
    """Plot calibration curve showing confidence vs accuracy."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly required for visualization")
        return

    bins = calibration_data.get("bins", [])
    if not bins:
        st.warning("No calibration data available")
        return

    # Extract data
    bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
    bin_confidence = [b["bin_confidence"] for b in bins]
    bin_accuracy = [b["bin_accuracy"] for b in bins]
    bin_sizes = [b["n_samples"] for b in bins]

    # Create figure
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(color="gray", dash="dash"),
    ))

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=bin_confidence,
        y=bin_accuracy,
        mode="markers+lines",
        name="Model Calibration",
        marker=dict(
            size=[s / 10 for s in bin_sizes],  # Size proportional to bin size
            color=bin_centers,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Confidence"),
        ),
        text=[f"Bin: {c:.2f}<br>Samples: {s}" for c, s in zip(bin_centers, bin_sizes)],
        hovertemplate="Confidence: %{x:.3f}<br>Accuracy: %{y:.3f}<br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Predicted Confidence",
        yaxis_title="Actual Accuracy",
        hovermode="closest",
        width=800,
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_confidence_distribution(df: pd.DataFrame, confidence_column: str) -> None:
    """Plot distribution of confidence scores."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly required for visualization")
        return

    confidences = df[confidence_column].dropna()

    fig = px.histogram(
        confidences,
        nbins=50,
        title="Confidence Score Distribution",
        labels={"value": "Confidence", "count": "Frequency"},
    )

    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_annotator_performance(metrics: dict[str, dict[str, Any]]) -> None:
    """Plot per-annotator performance metrics."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly required for visualization")
        return

    # Extract data
    data = []
    for annotator_id, m in metrics.items():
        if "accuracy" in m:
            data.append({
                "Annotator": annotator_id,
                "Accuracy": m["accuracy"],
                "Annotations": m.get("n_annotations", 0),
                "Cohen's Kappa": m.get("cohen_kappa", None),
            })

    if not data:
        st.warning("No accuracy data available for annotators")
        return

    df_metrics = pd.DataFrame(data)

    # Create bar chart
    fig = px.bar(
        df_metrics,
        x="Annotator",
        y="Accuracy",
        title="Annotator Performance",
        color="Annotations",
        hover_data=["Cohen's Kappa"],
        color_continuous_scale="Blues",
    )

    fig.add_hline(
        y=df_metrics["Accuracy"].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text="Mean Accuracy",
    )

    fig.update_layout(
        xaxis_title="Annotator ID",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_krippendorff_trend(quality_snapshots: list[dict[str, Any]]) -> None:
    """Plot Krippendorff's alpha over time."""
    if not PLOTLY_AVAILABLE or not quality_snapshots:
        return

    # Extract data
    timestamps = [s["timestamp"] for s in quality_snapshots]
    alphas = [s["alpha"] for s in quality_snapshots]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=alphas,
        mode="lines+markers",
        name="Krippendorff's Alpha",
        line=dict(color="green", width=2),
        marker=dict(size=8),
    ))

    # Add quality bands
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="green", opacity=0.1, annotation_text="Excellent", annotation_position="right")
    fig.add_hrect(y0=0.67, y1=0.8, fillcolor="yellow", opacity=0.1, annotation_text="Good", annotation_position="right")
    fig.add_hrect(y0=0.0, y1=0.67, fillcolor="red", opacity=0.1, annotation_text="Poor", annotation_position="right")

    fig.update_layout(
        title="Inter-Annotator Agreement Trend",
        xaxis_title="Time",
        yaxis_title="Krippendorff's Alpha",
        yaxis_range=[-0.1, 1.1],
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard application."""
    st.title("📊 AutoLabeler Quality Dashboard")
    st.markdown("""
    Monitor annotation quality, confidence calibration, and inter-annotator agreement
    in real-time for your labeling workflows.
    """)

    # Sidebar
    st.sidebar.header("Configuration")

    # Dataset selection
    dataset_name = st.sidebar.text_input("Dataset Name", value="my_dataset")

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv", "parquet"],
        help="Upload your labeled dataset for analysis",
    )

    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_parquet(uploaded_file)

        st.sidebar.success(f"Loaded {len(df)} rows")

        # Column selection
        st.sidebar.subheader("Column Mapping")
        columns = df.columns.tolist()

        confidence_col = st.sidebar.selectbox("Confidence Column", ["None"] + columns, index=0)
        true_label_col = st.sidebar.selectbox("True Label Column", ["None"] + columns, index=0)
        pred_label_col = st.sidebar.selectbox("Predicted Label Column", ["None"] + columns, index=0)

        # Annotator columns (for Krippendorff's alpha)
        st.sidebar.subheader("Inter-Annotator Agreement")
        annotator_cols = st.sidebar.multiselect(
            "Annotator Columns",
            columns,
            help="Select columns representing different annotators' labels",
        )

        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Confidence Calibration",
            "👥 Inter-Annotator Agreement",
            "🎯 Annotator Performance",
            "⚠️ Quality Anomalies",
        ])

        # Tab 1: Confidence Calibration
        with tab1:
            st.header("Confidence Calibration Analysis")

            if confidence_col != "None" and true_label_col != "None" and pred_label_col != "None":
                col1, col2 = st.columns(2)

                with col1:
                    calibration_method = st.selectbox(
                        "Calibration Method",
                        ["temperature", "platt"],
                    )

                with col2:
                    n_bins = st.slider("Number of Bins", min_value=5, max_value=20, value=10)

                if st.button("Run Calibration Analysis", type="primary"):
                    with st.spinner("Analyzing calibration..."):
                        # Create calibrator
                        calibrator = ConfidenceCalibrator(method=calibration_method, n_bins=n_bins)

                        # Get valid data
                        valid_mask = (
                            (~df[confidence_col].isna()) &
                            (~df[true_label_col].isna()) &
                            (~df[pred_label_col].isna())
                        )
                        valid_df = df[valid_mask]

                        if len(valid_df) < 10:
                            st.error("Not enough valid data for calibration (need at least 10 samples)")
                        else:
                            # Split into train/test
                            train_size = int(len(valid_df) * 0.7)
                            train_df = valid_df.iloc[:train_size]
                            test_df = valid_df.iloc[train_size:]

                            # Fit calibrator
                            calibrator.fit(
                                train_df[confidence_col].values,
                                train_df[true_label_col].values,
                                train_df[pred_label_col].values,
                            )

                            # Evaluate before calibration
                            metrics_before = calibrator.evaluate_calibration(
                                test_df[confidence_col].values,
                                test_df[true_label_col].values,
                                test_df[pred_label_col].values,
                                apply_calibration=False,
                            )

                            # Evaluate after calibration
                            metrics_after = calibrator.evaluate_calibration(
                                test_df[confidence_col].values,
                                test_df[true_label_col].values,
                                test_df[pred_label_col].values,
                                apply_calibration=True,
                            )

                            # Display metrics
                            st.subheader("Calibration Metrics")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "ECE (Before)",
                                    f"{metrics_before['expected_calibration_error']:.4f}",
                                )
                                st.metric(
                                    "ECE (After)",
                                    f"{metrics_after['expected_calibration_error']:.4f}",
                                    delta=f"{metrics_after['expected_calibration_error'] - metrics_before['expected_calibration_error']:.4f}",
                                    delta_color="inverse",
                                )

                            with col2:
                                st.metric(
                                    "Brier Score (Before)",
                                    f"{metrics_before['brier_score']:.4f}",
                                )
                                st.metric(
                                    "Brier Score (After)",
                                    f"{metrics_after['brier_score']:.4f}",
                                    delta=f"{metrics_after['brier_score'] - metrics_before['brier_score']:.4f}",
                                    delta_color="inverse",
                                )

                            with col3:
                                st.metric(
                                    "Calibration Gap (Before)",
                                    f"{metrics_before['calibration_gap']:.4f}",
                                )
                                st.metric(
                                    "Calibration Gap (After)",
                                    f"{metrics_after['calibration_gap']:.4f}",
                                    delta=f"{metrics_after['calibration_gap'] - metrics_before['calibration_gap']:.4f}",
                                    delta_color="inverse",
                                )

                            # Plot calibration curves
                            st.subheader("Before Calibration")
                            plot_calibration_curve(metrics_before)

                            st.subheader("After Calibration")
                            plot_calibration_curve(metrics_after)

                            # Confidence distribution
                            st.subheader("Confidence Distribution")
                            plot_confidence_distribution(df, confidence_col)

            else:
                st.info("Please select confidence, true label, and predicted label columns in the sidebar.")

        # Tab 2: Inter-Annotator Agreement
        with tab2:
            st.header("Inter-Annotator Agreement")

            if len(annotator_cols) >= 2:
                if st.button("Calculate Krippendorff's Alpha", type="primary"):
                    with st.spinner("Calculating inter-annotator agreement..."):
                        monitor = QualityMonitor(dataset_name=dataset_name)

                        result = monitor.calculate_krippendorff_alpha(df, annotator_cols)

                        # Display alpha
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            alpha = result["alpha"]
                            color = "green" if alpha > 0.8 else "orange" if alpha > 0.67 else "red"
                            st.metric("Krippendorff's Alpha", f"{alpha:.4f}")

                        with col2:
                            st.metric("Annotators", result["n_annotators"])

                        with col3:
                            st.metric("Items", result["n_items"])

                        with col4:
                            st.metric("Mean Pairwise Agreement", f"{result['mean_pairwise_agreement']:.4f}")

                        # Interpretation
                        st.subheader("Interpretation")
                        if alpha > 0.8:
                            st.success("✅ Excellent agreement - Data is highly reliable")
                        elif alpha > 0.67:
                            st.warning("⚠️ Good agreement - Data is tentatively reliable")
                        else:
                            st.error("❌ Poor agreement - Data quality issues detected")

                        # Agreement statistics
                        st.subheader("Agreement Statistics")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Items with Full Agreement", result["items_with_full_agreement"])

                        with col2:
                            st.metric("Items with Disagreement", result["items_with_disagreement"])

                        # Pairwise agreement
                        if result["pairwise_agreement"]:
                            st.subheader("Pairwise Agreement Matrix")
                            pairwise_df = pd.DataFrame([
                                {"Pair": k, "Agreement": v}
                                for k, v in result["pairwise_agreement"].items()
                            ])
                            st.dataframe(pairwise_df, use_container_width=True)

            else:
                st.info("Please select at least 2 annotator columns in the sidebar to calculate inter-annotator agreement.")

        # Tab 3: Annotator Performance
        with tab3:
            st.header("Per-Annotator Performance")

            annotator_id_col = st.selectbox("Annotator ID Column", ["None"] + columns, index=0)
            label_col = st.selectbox("Label Column", ["None"] + columns, index=0)
            gold_col = st.selectbox("Gold Standard Column (optional)", ["None"] + columns, index=0)

            if annotator_id_col != "None" and label_col != "None":
                if st.button("Analyze Annotator Performance", type="primary"):
                    with st.spinner("Analyzing annotator performance..."):
                        monitor = QualityMonitor(dataset_name=dataset_name)

                        gold_col_val = None if gold_col == "None" else gold_col
                        conf_col_val = None if confidence_col == "None" else confidence_col

                        metrics = monitor.track_annotator_metrics(
                            df,
                            annotator_id_col,
                            label_col,
                            gold_col_val,
                            conf_col_val,
                        )

                        # Display overview
                        st.subheader("Overview")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total Annotators", len(metrics))

                        with col2:
                            total_annotations = sum(m["n_annotations"] for m in metrics.values())
                            st.metric("Total Annotations", total_annotations)

                        with col3:
                            if gold_col_val:
                                accuracies = [m["accuracy"] for m in metrics.values() if "accuracy" in m]
                                if accuracies:
                                    st.metric("Mean Accuracy", f"{np.mean(accuracies):.4f}")

                        # Plot performance
                        if gold_col_val:
                            plot_annotator_performance(metrics)

                        # Detailed metrics table
                        st.subheader("Detailed Metrics")
                        metrics_df = pd.DataFrame([
                            {
                                "Annotator": ann_id,
                                "Annotations": m["n_annotations"],
                                "Accuracy": m.get("accuracy", "N/A"),
                                "Cohen's Kappa": m.get("cohen_kappa", "N/A"),
                                "Mean Confidence": m.get("confidence_stats", {}).get("mean", "N/A"),
                            }
                            for ann_id, m in metrics.items()
                        ])
                        st.dataframe(metrics_df, use_container_width=True)

            else:
                st.info("Please select annotator ID and label columns.")

        # Tab 4: Quality Anomalies
        with tab4:
            st.header("Quality Anomaly Detection")

            if annotator_id_col != "None" and label_col != "None":
                col1, col2, col3 = st.columns(3)

                with col1:
                    acc_threshold = st.slider("Accuracy Threshold", 0.0, 1.0, 0.7, 0.05)

                with col2:
                    conf_std_threshold = st.slider("Confidence Std Threshold", 0.0, 1.0, 0.3, 0.05)

                with col3:
                    rate_zscore = st.slider("Annotation Rate Z-Score", 1.0, 5.0, 2.5, 0.5)

                if st.button("Detect Anomalies", type="primary"):
                    with st.spinner("Detecting anomalies..."):
                        monitor = QualityMonitor(dataset_name=dataset_name)

                        gold_col_val = None if gold_col == "None" else gold_col
                        conf_col_val = None if confidence_col == "None" else confidence_col

                        # Get metrics
                        metrics = monitor.track_annotator_metrics(
                            df,
                            annotator_id_col,
                            label_col,
                            gold_col_val,
                            conf_col_val,
                        )

                        # Detect anomalies
                        anomalies = monitor.detect_anomalies(
                            metrics,
                            accuracy_threshold=acc_threshold,
                            confidence_std_threshold=conf_std_threshold,
                            annotation_rate_zscore_threshold=rate_zscore,
                        )

                        # Display results
                        if anomalies:
                            st.warning(f"⚠️ Detected {len(anomalies)} anomalies")

                            for anomaly in anomalies:
                                severity = anomaly["severity"]
                                icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"

                                with st.expander(f"{icon} {anomaly['annotator_id']} - {anomaly['issue']}"):
                                    st.write(f"**Issue:** {anomaly['issue']}")
                                    st.write(f"**Value:** {anomaly['value']}")
                                    st.write(f"**Severity:** {severity}")
                                    if "note" in anomaly:
                                        st.write(f"**Note:** {anomaly['note']}")
                        else:
                            st.success("✅ No anomalies detected!")

            else:
                st.info("Please select annotator ID and label columns.")

    else:
        st.info("👈 Upload a dataset in the sidebar to get started")

        # Show example
        st.subheader("Example Dataset Format")
        example_df = pd.DataFrame({
            "text": ["Example 1", "Example 2", "Example 3"],
            "true_label": ["positive", "negative", "positive"],
            "predicted_label": ["positive", "negative", "negative"],
            "confidence": [0.92, 0.85, 0.63],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["positive", "negative", "positive"],
            "annotator3": ["positive", "positive", "positive"],
        })
        st.dataframe(example_df, use_container_width=True)


if __name__ == "__main__":
    main()
