# AutoLabeler Installation Guide

## Quick Start

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autolabeler.git
cd autolabeler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install basic AutoLabeler
pip install -e .
```

This installs AutoLabeler with all Phase 1 and Phase 2 core features including:
- Basic labeling with RAG
- Ensemble methods
- Confidence calibration
- Quality metrics (Krippendorff's alpha)
- Advanced RAG (GraphRAG, RAPTOR)
- Active Learning framework

---

## Installation Options

### Option 1: Basic Installation (Recommended for most users)

```bash
pip install -e .
```

**Includes:**
- ✅ Core labeling functionality
- ✅ RAG-based example retrieval
- ✅ Ensemble methods
- ✅ Phase 1: Confidence calibration, quality metrics
- ✅ Phase 2: GraphRAG, RAPTOR, Active Learning

**Does NOT include:**
- ❌ Quality dashboard (Streamlit)
- ❌ DSPy prompt optimization
- ❌ Data versioning (DVC)

---

### Option 2: With Dashboard

```bash
pip install -e ".[dashboard]"
```

**Adds:**
- ✅ Streamlit quality dashboard
- ✅ Plotly interactive visualizations
- ✅ Real-time quality monitoring

**Use when:** You want visual quality monitoring and analysis

---

### Option 3: With Phase 2 Features

```bash
pip install -e ".[phase2]"
```

**Adds:**
- ✅ DSPy prompt optimization framework
- ✅ Data versioning with DVC

**Use when:** You want algorithmic prompt optimization and experiment tracking

**Note:** DSPy requires additional setup (see DSPy Configuration below)

---

### Option 4: Complete Installation (All Features)

```bash
pip install -e ".[all]"
```

**Includes everything:**
- ✅ All core features
- ✅ Quality dashboard
- ✅ DSPy optimization
- ✅ DVC versioning with all remote storage options (S3, Azure, GCS)

**Use when:** You want the complete AutoLabeler experience

---

### Option 5: Development Installation

```bash
pip install -e ".[dev,all]"
```

**Adds development tools:**
- Black (code formatting)
- Ruff (linting)
- pytest (testing)
- pre-commit hooks

**Use when:** Contributing to AutoLabeler development

---

## Verifying Installation

### Check Core Features

```python
from autolabeler import AutoLabeler
from autolabeler.core.labeling import OptimizedLabelingService
from autolabeler.core.active_learning import ActiveLearningSampler

print("✅ AutoLabeler installed successfully!")
```

### Check Phase 1 Features

```python
from autolabeler.core.quality.calibrator import ConfidenceCalibrator
from autolabeler.core.quality.monitor import QualityMonitor

print("✅ Phase 1 features available!")
```

### Check Phase 2 Features

```python
from autolabeler.core.optimization.dspy_optimizer import DSPyOptimizer
from autolabeler.core.rag.graph_rag import GraphRAG
from autolabeler.core.rag.raptor_rag import RAPTORRAG
from autolabeler.core.versioning.dvc_manager import DVCManager

print("✅ Phase 2 features available!")
```

---

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```env
# Required: LLM Provider API Key
OPENROUTER_API_KEY=your_openrouter_key
# OR
OPENAI_API_KEY=your_openai_key

# Optional: Corporate proxy
CORPORATE_BASE_URL=https://your-proxy.com/v1
CORPORATE_API_KEY=your_corporate_key

# Optional: Model settings
DEFAULT_MODEL=meta-llama/llama-3.1-8b-instruct:free
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: Logging
LOG_LEVEL=INFO
```

### 2. DSPy Configuration (Optional)

If you installed DSPy (`pip install -e ".[phase2]"` or `".[all]"`):

```python
import dspy
from autolabeler.config import Settings

settings = Settings()

# Configure DSPy with your LLM
lm = dspy.OpenAI(
    model="gpt-3.5-turbo",
    api_key=settings.openai_api_key
)
dspy.settings.configure(lm=lm)
```

See `examples/phase2_dspy_optimization_example.py` for complete usage.

### 3. DVC Configuration (Optional)

If you installed DVC (`pip install -e ".[versioning]"` or `".[all]"`):

```bash
# Initialize DVC in your project
cd /path/to/your/project
dvc init

# Configure remote storage (choose one)
# S3
dvc remote add -d myremote s3://mybucket/path

# Azure
dvc remote add -d myremote azure://mycontainer/path

# GCS
dvc remote add -d myremote gs://mybucket/path
```

See `docs/dvc_setup_guide.md` for detailed setup instructions.

---

## Running the Dashboard

If you installed the dashboard feature:

```bash
# Start the quality monitoring dashboard
streamlit run src/autolabeler/dashboard/quality_dashboard.py

# Dashboard will be available at: http://localhost:8501
```

---

## Testing Your Installation

### Quick Test

```bash
# Run a subset of tests
pytest tests/ -k "not requires_api" -x

# This runs all tests that don't require external API access
```

### Full Test Suite

```bash
# Run all tests (requires API keys configured)
pytest tests/ -v

# Run only Phase 2 tests
pytest tests/test_phase2/ -v
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'autolabeler'`

**Solution:**
```bash
# Make sure you installed in editable mode
pip install -e .

# Check installation
pip list | grep autolabeler
```

### Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'scipy'` or similar

**Solution:**
```bash
# Reinstall with all dependencies
pip install -e ".[all]"

# Or install specific feature group
pip install -e ".[dashboard]"  # for dashboard features
pip install -e ".[phase2]"     # for DSPy and DVC
```

### DSPy Not Working

**Problem:** `DSPy not installed` warning or import errors

**Solution:**
```bash
# Install DSPy explicitly
pip install dspy-ai>=2.5.0

# Or install phase2 features
pip install -e ".[phase2]"
```

**Note:** DSPy requires additional configuration. See the DSPy Configuration section above.

### DVC Not Working

**Problem:** `dvc: command not found` or DVC operations fail

**Solution:**
```bash
# Install DVC with remote support
pip install -e ".[versioning]"

# Or install specific remote
pip install dvc dvc-s3  # for S3
pip install dvc dvc-azure  # for Azure
pip install dvc dvc-gs  # for GCS

# Initialize DVC
dvc init
```

### API Key Issues

**Problem:** `API key not found` errors

**Solution:**
```bash
# Create .env file in project root
cat > .env << EOF
OPENAI_API_KEY=your_api_key_here
EOF

# Or set environment variable
export OPENAI_API_KEY=your_api_key_here
```

---

## Dependency Overview

### Core Dependencies (always installed)

| Package | Version | Purpose |
|---------|---------|---------|
| pydantic | >=2 | Configuration and validation |
| pandas | >=2 | Data handling |
| langchain | >=0.1.0 | LLM orchestration |
| openai | >=1.0 | OpenAI API client |
| sentence-transformers | >=4.1.0 | Embeddings |
| faiss-cpu | >=1.7 | Vector similarity search |
| scikit-learn | >=1.3.0 | ML utilities |
| scipy | >=1.11.0 | Statistical functions (Phase 1) |
| krippendorff | >=0.8.1 | Inter-annotator agreement (Phase 1) |
| networkx | >=3.0 | Graph operations (Phase 2) |
| python-louvain | >=0.16 | Community detection (Phase 2) |
| umap-learn | >=0.5.0 | Dimensionality reduction (Phase 2) |
| rank-bm25 | >=0.2.2 | BM25 search (Phase 2) |

### Optional Dependencies

| Feature | Package | Version | Install With |
|---------|---------|---------|--------------|
| **Dashboard** | streamlit | >=1.43.0 | `[dashboard]` |
| | plotly | >=5.24.0 | `[dashboard]` |
| **DSPy** | dspy-ai | >=2.5.0 | `[dspy]` or `[phase2]` |
| **Versioning** | dvc | >=3.0.0 | `[versioning]` or `[phase2]` |
| | dvc-s3 | >=3.0.0 | `[versioning]` (S3 support) |
| | dvc-azure | >=3.0.0 | `[versioning]` (Azure support) |
| | dvc-gs | >=3.0.0 | `[versioning]` (GCS support) |

---

## Upgrade Instructions

### From Phase 1 to Phase 2

If you already have Phase 1 installed:

```bash
# Pull latest changes
git pull origin main

# Reinstall with new dependencies
pip install -e . --upgrade

# Or install with all features
pip install -e ".[all]" --upgrade
```

### Verify Upgrade

```python
# Check Phase 2 imports work
from autolabeler.core.optimization.dspy_optimizer import DSPyOptimizer
from autolabeler.core.rag.graph_rag import GraphRAG
from autolabeler.core.active_learning import ActiveLearningSampler

print("✅ Successfully upgraded to Phase 2!")
```

---

## Next Steps

After installation:

1. **Configure API keys:** Set up `.env` file with your LLM provider credentials
2. **Try examples:** Run `python examples/phase2_dspy_optimization_example.py`
3. **Read documentation:** Check out `PHASE2_COMPLETE.md` for feature overview
4. **Run tests:** Validate installation with `pytest tests/test_phase2/ -v`
5. **Start dashboard:** Launch quality monitoring with `streamlit run ...`

---

## Getting Help

- **Documentation:** See `PHASE2_COMPLETE.md` for comprehensive feature documentation
- **Examples:** Check `examples/` directory for usage examples
- **Tests:** Review `tests/` directory for code examples
- **Issues:** Report bugs at https://github.com/yourusername/autolabeler/issues

---

## System Requirements

- **Python:** 3.10, 3.11, or 3.12
- **OS:** Linux, macOS, Windows (WSL recommended)
- **Memory:** 4GB minimum, 8GB recommended
- **Storage:** 500MB for installation, additional for datasets

---

**Installation complete!** 🎉

Your AutoLabeler is ready with state-of-the-art annotation capabilities.
