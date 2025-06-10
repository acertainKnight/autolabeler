"""Unit tests for the RuleGenerator class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from autolabeler import Settings, RuleGenerator
from autolabeler.models import (
    LabelingRule,
    RuleGenerationResult,
    RuleSet,
    RuleUpdateResult,
)


@pytest.fixture
def sample_data():
    """Sample labeled data for testing."""
    return pd.DataFrame({
        "text": [
            "I love this product! It's amazing.",
            "This is terrible quality. Very disappointed.",
            "The product is okay, nothing special.",
            "Excellent service and fast delivery!",
            "Poor quality and expensive.",
            "It's an average product for the price.",
            "Outstanding quality and design!",
            "Worst purchase ever. Complete waste.",
            "Decent product, meets basic needs.",
            "Perfect quality and value!",
        ],
        "sentiment": [
            "positive", "negative", "neutral",
            "positive", "negative", "neutral",
            "positive", "negative", "neutral",
            "positive"
        ]
    })


@pytest.fixture
def settings():
    """Test settings with mock API key."""
    return Settings(
        openrouter_api_key="test-key",
        llm_model="test-model",
        max_examples_per_query=3,
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for rule generation."""
    mock_rule = LabelingRule(
        rule_id="test_rule_1",
        label="positive",
        pattern_description="Positive sentiment with enthusiastic language",
        conditions=["Contains positive adjectives", "Expresses satisfaction"],
        indicators=["love", "amazing", "excellent"],
        examples=["I love this product!", "Excellent service!"],
        confidence=0.85,
        frequency=3,
        creation_timestamp="2024-01-01T00:00:00",
        last_updated="2024-01-01T00:00:00",
    )

    mock_ruleset = RuleSet(
        dataset_name="test_dataset",
        task_description="Test sentiment classification",
        label_categories=["positive", "negative", "neutral"],
        rules=[mock_rule],
        general_guidelines=["Consider overall sentiment"],
        version="1.0.0",
        creation_timestamp="2024-01-01T00:00:00",
        last_updated="2024-01-01T00:00:00",
    )

    return RuleGenerationResult(
        ruleset=mock_ruleset,
        generation_metadata={"test": "metadata"},
        data_analysis={"total_examples": 10},
    )


class TestRuleGenerator:
    """Test cases for RuleGenerator class."""

    def test_initialization(self, settings):
        """Test RuleGenerator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "rulesets"

            generator = RuleGenerator(
                dataset_name="test_dataset",
                settings=settings,
                storage_path=storage_path,
            )

            assert generator.dataset_name == "test_dataset"
            assert generator.settings == settings
            assert generator.storage_path == storage_path
            assert storage_path.exists()

    @patch('autolabeler.rule_generator.OpenRouterClient')
    def test_llm_client_initialization(self, mock_openrouter, settings):
        """Test LLM client initialization."""
        mock_client = Mock()
        mock_openrouter.return_value = mock_client
        mock_client.with_structured_output.return_value = Mock()

        generator = RuleGenerator("test_dataset", settings)

        mock_openrouter.assert_called_once_with(
            api_key=settings.openrouter_api_key,
            model=settings.llm_model,
        )
        assert mock_client.with_structured_output.call_count == 2  # generation and update LLMs

    def test_analyze_data(self, settings, sample_data):
        """Test data analysis functionality."""
        generator = RuleGenerator("test_dataset", settings)

        analysis = generator._analyze_data(sample_data, "text", "sentiment")

        assert analysis["total_examples"] == 10
        assert analysis["unique_labels"] == 3
        assert "positive" in analysis["label_distribution"]
        assert "negative" in analysis["label_distribution"]
        assert "neutral" in analysis["label_distribution"]
        assert "label_patterns" in analysis
        assert "data_hash" in analysis

    def test_extract_indicators(self, settings):
        """Test indicator extraction from examples."""
        generator = RuleGenerator("test_dataset", settings)

        examples = [
            "I love this amazing product!",
            "This excellent service is outstanding!",
            "Amazing quality and fantastic design!"
        ]

        indicators = generator._extract_indicators(examples)

        assert isinstance(indicators, list)
        assert len(indicators) <= 5  # Should limit to top 5
        # Should include common words like "amazing"
        assert any("amazing" in indicator.lower() for indicator in indicators)

    def test_find_common_prefixes(self, settings):
        """Test common prefix identification."""
        generator = RuleGenerator("test_dataset", settings)

        texts = [
            "I love this product",
            "I hate this service",
            "I think this is okay"
        ]

        prefixes = generator._find_common_prefixes(texts)

        assert isinstance(prefixes, list)
        assert any("I" in prefix for prefix in prefixes)

    def test_find_common_suffixes(self, settings):
        """Test common suffix identification."""
        generator = RuleGenerator("test_dataset", settings)

        texts = [
            "This product is great!",
            "This service is terrible!",
            "This item is okay!"
        ]

        suffixes = generator._find_common_suffixes(texts)

        assert isinstance(suffixes, list)
        assert any("!" in suffix for suffix in suffixes)

    def test_rules_similarity(self, settings):
        """Test rule similarity detection."""
        generator = RuleGenerator("test_dataset", settings)

        rule1 = LabelingRule(
            rule_id="rule1",
            label="positive",
            pattern_description="Test rule 1",
            conditions=["condition1"],
            indicators=["love", "great", "amazing"],
            examples=["example1"],
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        rule2 = LabelingRule(
            rule_id="rule2",
            label="positive",
            pattern_description="Test rule 2",
            conditions=["condition2"],
            indicators=["love", "excellent", "amazing"],  # 2/3 overlap
            examples=["example2"],
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        rule3 = LabelingRule(
            rule_id="rule3",
            label="negative",  # Different label
            pattern_description="Test rule 3",
            conditions=["condition3"],
            indicators=["love", "great", "amazing"],
            examples=["example3"],
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        # Same label with overlapping indicators should be similar
        assert generator._rules_are_similar(rule1, rule2) == True

        # Different labels should not be similar
        assert generator._rules_are_similar(rule1, rule3) == False

    def test_consolidate_rules(self, settings):
        """Test rule consolidation."""
        generator = RuleGenerator("test_dataset", settings)

        # Create similar rules that should be merged
        rule1 = LabelingRule(
            rule_id="rule1",
            label="positive",
            pattern_description="Positive rule 1",
            conditions=["condition1"],
            indicators=["love", "great"],
            examples=["example1"],
            confidence=0.8,
            frequency=5,
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        rule2 = LabelingRule(
            rule_id="rule2",
            label="positive",
            pattern_description="Positive rule 2",
            conditions=["condition2"],
            indicators=["love", "amazing"],  # Overlapping with rule1
            examples=["example2"],
            confidence=0.9,
            frequency=3,
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        rule3 = LabelingRule(
            rule_id="rule3",
            label="negative",  # Different label, shouldn't merge
            pattern_description="Negative rule",
            conditions=["condition3"],
            indicators=["hate", "terrible"],
            examples=["example3"],
            confidence=0.7,
            frequency=4,
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        rules = [rule1, rule2, rule3]
        consolidated = generator._consolidate_rules(rules)

        # Should have fewer rules after consolidation
        assert len(consolidated) <= len(rules)
        # Should have rules for both labels
        labels = {rule.label for rule in consolidated}
        assert "positive" in labels
        assert "negative" in labels

    def test_hash_dataframe(self, settings, sample_data):
        """Test DataFrame hashing for change detection."""
        generator = RuleGenerator("test_dataset", settings)

        hash1 = generator._hash_dataframe(sample_data, ["text", "sentiment"])
        hash2 = generator._hash_dataframe(sample_data, ["text", "sentiment"])

        # Same data should produce same hash
        assert hash1 == hash2

        # Modified data should produce different hash
        modified_data = sample_data.copy()
        modified_data.loc[0, "sentiment"] = "changed"
        hash3 = generator._hash_dataframe(modified_data, ["text", "sentiment"])

        assert hash1 != hash3

    def test_increment_version(self, settings):
        """Test version increment functionality."""
        generator = RuleGenerator("test_dataset", settings)

        assert generator._increment_version("1.0.0") == "1.0.1"
        assert generator._increment_version("1.0.5") == "1.0.6"
        assert generator._increment_version("invalid") == "1.0.1"

    @patch('autolabeler.rule_generator.RuleGenerator._generate_rules_for_batch')
    def test_generate_rules_from_data(self, mock_generate_batch, settings, sample_data, mock_llm_response):
        """Test end-to-end rule generation."""
        # Mock the batch generation
        mock_rule = LabelingRule(
            rule_id="test_rule",
            label="positive",
            pattern_description="Test rule",
            conditions=["test condition"],
            indicators=["test"],
            examples=["test example"],
            confidence=0.8,
            frequency=2,
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )
        mock_generate_batch.return_value = [mock_rule]

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "rulesets"
            generator = RuleGenerator("test_dataset", settings, storage_path)

            result = generator.generate_rules_from_data(
                df=sample_data,
                text_column="text",
                label_column="sentiment",
                task_description="Test sentiment classification",
                batch_size=5,
                min_examples_per_rule=2,
            )

            assert isinstance(result, RuleGenerationResult)
            assert result.ruleset.dataset_name == "test_dataset"
            assert len(result.ruleset.rules) > 0
            assert "generation_metadata" in result.model_dump()
            assert "data_analysis" in result.model_dump()

    def test_export_markdown_guidelines(self, settings):
        """Test Markdown export functionality."""
        # Create a test ruleset
        rule = LabelingRule(
            rule_id="test_rule",
            label="positive",
            pattern_description="Positive sentiment rule",
            conditions=["Contains positive words"],
            indicators=["good", "great", "excellent"],
            examples=["This is good!", "Great product!"],
            confidence=0.9,
            frequency=5,
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        ruleset = RuleSet(
            dataset_name="test_dataset",
            task_description="Test classification",
            label_categories=["positive", "negative"],
            rules=[rule],
            general_guidelines=["Be consistent"],
            version="1.0.0",
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = RuleGenerator("test_dataset", settings, Path(temp_dir))
            output_path = Path(temp_dir) / "guidelines.md"

            generator._export_markdown_guidelines(ruleset, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "# Annotation Guidelines" in content
            assert "positive" in content
            assert "Positive sentiment rule" in content

    def test_save_and_load_ruleset(self, settings):
        """Test saving and loading rulesets."""
        rule = LabelingRule(
            rule_id="test_rule",
            label="positive",
            pattern_description="Test rule",
            conditions=["test"],
            indicators=["good"],
            examples=["good example"],
            confidence=0.8,
            frequency=3,
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        ruleset = RuleSet(
            dataset_name="test_dataset",
            task_description="Test task",
            label_categories=["positive"],
            rules=[rule],
            general_guidelines=["test guideline"],
            version="1.0.0",
            creation_timestamp="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "rulesets"
            generator = RuleGenerator("test_dataset", settings, storage_path)

            # Save ruleset
            generator._save_ruleset(ruleset)

            # Verify file was created
            ruleset_files = list(storage_path.glob("test_dataset_ruleset_*.json"))
            assert len(ruleset_files) == 1

            # Load ruleset
            loaded_ruleset = generator.load_latest_ruleset()

            assert loaded_ruleset.dataset_name == ruleset.dataset_name
            assert loaded_ruleset.version == ruleset.version
            assert len(loaded_ruleset.rules) == len(ruleset.rules)

    def test_load_nonexistent_ruleset(self, settings):
        """Test loading a ruleset that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "rulesets"
            generator = RuleGenerator("nonexistent_dataset", settings, storage_path)

            with pytest.raises(FileNotFoundError):
                generator.load_latest_ruleset()


if __name__ == "__main__":
    pytest.main([__file__])
