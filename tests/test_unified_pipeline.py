"""Integration test for unified labeling pipeline.

Tests the complete flow:
1. Load dataset config
2. Load prompts
3. Initialize pipeline
4. Label sample texts (mocked LLM calls)
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock the dependencies before importing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sibyls.core.dataset_config import DatasetConfig
from src.sibyls.core.labeling.pipeline import LabelingPipeline
from src.sibyls.core.prompts.registry import PromptRegistry
from src.sibyls.core.quality.confidence_scorer import ConfidenceScorer


@pytest.fixture
def mock_config():
    """Create a minimal test config."""
    from src.sibyls.core.dataset_config import ModelConfig
    
    config = DatasetConfig(
        name="test_dataset",
        labels=["0", "1", "-1"],
        text_column="text",
        label_column="label",
        batch_size=5,
        jury_models=[
            ModelConfig(
                name="test-model-1",
                provider="openai",
                model="gpt-4o-mini",
                has_logprobs=True,
            ),
            ModelConfig(
                name="test-model-2",
                provider="anthropic",
                model="claude-3-5-haiku-20241022",
                has_logprobs=False,
            ),
        ],
        jury_temperature=0.0,
        use_relevancy_gate=False,
        use_candidate_annotation=False,
    )
    return config


@pytest.fixture
def mock_prompts(tmp_path):
    """Create minimal test prompts."""
    prompts_dir = tmp_path / "prompts" / "test_dataset"
    prompts_dir.mkdir(parents=True)
    
    (prompts_dir / "system.md").write_text("You are a test classifier.")
    (prompts_dir / "rules.md").write_text("# Classification Rules\n\nLabel as 0, 1, or -1.")
    (prompts_dir / "examples.md").write_text("# Examples\n\n**Example 1**: Text -> 1")
    (prompts_dir / "mistakes.md").write_text("# Common Mistakes\n\nDon't confuse 0 with 1.")
    
    registry = PromptRegistry("test_dataset", base_dir=tmp_path / "prompts")
    return registry


def test_confidence_scorer_logprobs():
    """Test confidence extraction from logprobs."""
    scorer = ConfidenceScorer()
    
    logprobs = {"0": 0.8, "1": 0.15, "-1": 0.05}
    conf = scorer.from_logprobs(logprobs, label=0)
    
    assert conf == 0.8


def test_confidence_scorer_verbal():
    """Test verbal confidence mapping."""
    scorer = ConfidenceScorer()
    
    assert scorer.from_verbal("high") == 0.9
    assert scorer.from_verbal("medium") == 0.7
    assert scorer.from_verbal("low") == 0.5


def test_confidence_scorer_calibration():
    """Test isotonic calibration."""
    scorer = ConfidenceScorer()
    
    # Create fake calibration data
    raw_confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, False, False, False, True, True, True, False, False]
    
    scorer.fit_calibrator(raw_confs, correct)
    
    # Calibrated score should be different
    calibrated = scorer.calibrate(0.9)
    assert 0.0 <= calibrated <= 1.0


def test_prompt_registry(mock_prompts):
    """Test prompt loading and assembly."""
    system, user = mock_prompts.build_labeling_prompt("Test text", "")
    
    assert "You are a test classifier" in system
    assert "Classification Rules" in system
    assert "Test text" in user


@pytest.mark.asyncio
async def test_pipeline_integration_mock(mock_config, mock_prompts):
    """Test full pipeline with mocked LLM calls."""
    scorer = ConfidenceScorer()
    pipeline = LabelingPipeline(mock_config, mock_prompts, scorer)
    
    # Mock the jury providers to return fake responses
    from src.sibyls.core.llm_providers.providers import LLMResponse
    
    mock_response_1 = LLMResponse(
        text='{"label": 1, "reasoning": "Test reason", "confidence": "high"}',
        parsed_json={"label": 1, "reasoning": "Test reason", "confidence": "high"},
        logprobs={"0": 0.1, "1": 0.8, "-1": 0.1},
        cost=0.001,
    )
    
    mock_response_2 = LLMResponse(
        text='{"label": 1, "reasoning": "Test reason 2", "confidence": "medium"}',
        parsed_json={"label": 1, "reasoning": "Test reason 2", "confidence": "medium"},
        logprobs=None,
        cost=0.001,
    )
    
    # Mock each provider's call method
    for i, provider in enumerate(pipeline.jury_providers):
        provider.call = AsyncMock(return_value=mock_response_1 if i == 0 else mock_response_2)
    
    # Test labeling one text
    result = await pipeline.label_one("Test headline about policy")
    
    assert result.label == 1
    assert result.tier in ["ACCEPT", "ACCEPT-M", "SOFT", "QUARANTINE"]
    assert len(result.jury_labels) == 2
    assert all(label == 1 for label in result.jury_labels)
    assert result.agreement == "unanimous"


def test_dataset_config_yaml(tmp_path):
    """Test loading dataset config from YAML."""
    config_path = tmp_path / "test.yaml"
    config_path.write_text("""
name: test_dataset
labels: ["0", "1", "-1"]
text_column: text
label_column: label
batch_size: 10

jury_models:
  - name: gpt-4o-mini
    provider: openai
    model: gpt-4o-mini
    has_logprobs: true

jury_temperature: 0.0
use_relevancy_gate: false
use_candidate_annotation: false
""")
    
    config = DatasetConfig.from_yaml(config_path)
    
    assert config.name == "test_dataset"
    assert len(config.labels) == 3
    assert config.batch_size == 10
    assert len(config.jury_models) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
