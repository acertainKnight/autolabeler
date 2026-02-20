#!/usr/bin/env python3
"""Unit tests for the cascaded model escalation strategy.

Tests the CascadeStrategy logic without making any API calls.
Validates tier ordering, acceptance decisions, and escalation behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.autolabeler.core.dataset_config import DatasetConfig, ModelConfig
from src.autolabeler.core.labeling.cascade import CascadeStrategy, EscalationResult


def make_config(
    use_cascade: bool = True,
    cascade_confidence_threshold: float = 0.85,
    cascade_agreement_threshold: float = 0.80,
    jury_models: list[ModelConfig] | None = None,
) -> DatasetConfig:
    """Create a test config with cascade settings."""
    if jury_models is None:
        jury_models = [
            ModelConfig(provider="openai", model="gpt-4o", name="GPT-4o",
                        has_logprobs=True, cost_tier=2),
            ModelConfig(provider="anthropic", model="claude-sonnet-4-5", name="Claude",
                        has_logprobs=False, cost_tier=2),
            ModelConfig(provider="google", model="gemini-2.5-flash", name="Gemini-Flash",
                        has_logprobs=False, cost_tier=1),
        ]
    
    return DatasetConfig(
        name="test",
        labels=["0", "1", "2"],
        use_cascade=use_cascade,
        cascade_confidence_threshold=cascade_confidence_threshold,
        cascade_agreement_threshold=cascade_agreement_threshold,
        jury_models=jury_models,
    )


def test_tier_ordering():
    """Models should be ordered by cost_tier, cheapest first."""
    config = make_config()
    strategy = CascadeStrategy(config)
    
    tiers = strategy.tiers()
    
    # Tier 0 should contain model index 2 (Gemini-Flash, cost_tier=1)
    assert tiers[0] == [2], f"Tier 0 should be [2], got {tiers[0]}"
    
    # Tier 1 should contain model indices 0 and 1 (cost_tier=2)
    assert sorted(tiers[1]) == [0, 1], f"Tier 1 should be [0,1], got {tiers[1]}"
    
    print("✓ test_tier_ordering passed")


def test_single_model_confident_accept():
    """A single confident model should trigger early acceptance."""
    config = make_config(cascade_confidence_threshold=0.85)
    strategy = CascadeStrategy(config)
    
    results = [
        {"label": "1", "confidence": 0.92, "model_name": "Gemini-Flash"}
    ]
    
    accept, reason = strategy.should_accept(results, tier_index=0)
    assert accept is True, f"Should accept, got {accept}"
    assert "single_model_confident" in reason
    
    print("✓ test_single_model_confident_accept passed")


def test_single_model_low_confidence_escalate():
    """A single model with low confidence should escalate."""
    config = make_config(cascade_confidence_threshold=0.85)
    strategy = CascadeStrategy(config)
    
    results = [
        {"label": "1", "confidence": 0.60, "model_name": "Gemini-Flash"}
    ]
    
    accept, reason = strategy.should_accept(results, tier_index=0)
    assert accept is False, f"Should escalate, got {accept}"
    assert "low_confidence" in reason
    
    print("✓ test_single_model_low_confidence_escalate passed")


def test_two_model_agreement_accept():
    """Two models agreeing above threshold should accept."""
    config = make_config(cascade_agreement_threshold=0.80)
    strategy = CascadeStrategy(config)
    
    results = [
        {"label": "1", "confidence": 0.88, "model_name": "Gemini-Flash"},
        {"label": "1", "confidence": 0.85, "model_name": "GPT-4o"},
    ]
    
    accept, reason = strategy.should_accept(results, tier_index=1)
    assert accept is True, f"Should accept, got {accept}"
    assert "agreement" in reason
    
    print("✓ test_two_model_agreement_accept passed")


def test_two_model_disagreement_escalate():
    """Two models disagreeing should escalate."""
    config = make_config()
    strategy = CascadeStrategy(config)
    
    results = [
        {"label": "1", "confidence": 0.88, "model_name": "Gemini-Flash"},
        {"label": "0", "confidence": 0.85, "model_name": "GPT-4o"},
    ]
    
    accept, reason = strategy.should_accept(results, tier_index=1)
    assert accept is False, f"Should escalate, got {accept}"
    assert "disagreement" in reason
    
    print("✓ test_two_model_disagreement_escalate passed")


def test_two_model_agreement_low_confidence_escalate():
    """Two models agreeing but with low confidence should escalate."""
    config = make_config(cascade_agreement_threshold=0.80)
    strategy = CascadeStrategy(config)
    
    results = [
        {"label": "1", "confidence": 0.55, "model_name": "Gemini-Flash"},
        {"label": "1", "confidence": 0.60, "model_name": "GPT-4o"},
    ]
    
    accept, reason = strategy.should_accept(results, tier_index=1)
    assert accept is False, f"Should escalate, got {accept}"
    assert "low_confidence" in reason
    
    print("✓ test_two_model_agreement_low_confidence_escalate passed")


def test_supermajority_accept():
    """3 models with 2 agreeing (supermajority) should not accept (need 75%)."""
    config = make_config(cascade_agreement_threshold=0.80)
    strategy = CascadeStrategy(config)
    
    results = [
        {"label": "1", "confidence": 0.88, "model_name": "Gemini-Flash"},
        {"label": "1", "confidence": 0.85, "model_name": "GPT-4o"},
        {"label": "0", "confidence": 0.70, "model_name": "Claude"},
    ]
    
    # 2/3 = 66% < 75% threshold, so no supermajority shortcut
    accept, reason = strategy.should_accept(results, tier_index=2)
    assert accept is False, f"Should not accept 2/3 as supermajority, got {accept}"
    
    print("✓ test_supermajority_accept passed")


def test_three_model_unanimous_accept():
    """3 models unanimously agreeing should accept."""
    config = make_config(cascade_agreement_threshold=0.80)
    strategy = CascadeStrategy(config)
    
    results = [
        {"label": "1", "confidence": 0.88, "model_name": "Gemini-Flash"},
        {"label": "1", "confidence": 0.85, "model_name": "GPT-4o"},
        {"label": "1", "confidence": 0.82, "model_name": "Claude"},
    ]
    
    accept, reason = strategy.should_accept(results, tier_index=2)
    assert accept is True, f"Should accept unanimous, got {accept}"
    assert "agreement" in reason
    
    print("✓ test_three_model_unanimous_accept passed")


def test_no_results_escalate():
    """Empty results should escalate."""
    config = make_config()
    strategy = CascadeStrategy(config)
    
    accept, reason = strategy.should_accept([], tier_index=0)
    assert accept is False
    
    print("✓ test_no_results_escalate passed")


def test_escalation_result_cost_savings():
    """EscalationResult should calculate cost savings correctly."""
    config = make_config()
    strategy = CascadeStrategy(config)
    
    result = strategy.build_escalation_result(
        results=[{"label": "1"}],
        models_called=1,
        early_exit=True,
        reason="single_model_confident",
    )
    
    # 1 of 3 models called = 66.7% saved
    assert result.early_exit is True
    assert abs(result.cost_saved_pct - 66.67) < 1.0, (
        f"Expected ~66.7% savings, got {result.cost_saved_pct}"
    )
    
    print("✓ test_escalation_result_cost_savings passed")


def test_all_same_tier():
    """When all models have same cost_tier, first tier has all models."""
    models = [
        ModelConfig(provider="openai", model="a", name="A", cost_tier=1),
        ModelConfig(provider="openai", model="b", name="B", cost_tier=1),
        ModelConfig(provider="openai", model="c", name="C", cost_tier=1),
    ]
    config = make_config(jury_models=models)
    strategy = CascadeStrategy(config)
    
    tiers = strategy.tiers()
    assert len(tiers) == 1, f"Should have 1 tier, got {len(tiers)}"
    assert sorted(tiers[0]) == [0, 1, 2]
    
    print("✓ test_all_same_tier passed")


def test_three_tiers():
    """Three distinct cost tiers should produce three escalation levels."""
    models = [
        ModelConfig(provider="openai", model="expensive", name="A", cost_tier=3),
        ModelConfig(provider="openai", model="mid", name="B", cost_tier=2),
        ModelConfig(provider="openai", model="cheap", name="C", cost_tier=1),
    ]
    config = make_config(jury_models=models)
    strategy = CascadeStrategy(config)
    
    tiers = strategy.tiers()
    assert len(tiers) == 3, f"Should have 3 tiers, got {len(tiers)}"
    assert tiers[0] == [2], "Tier 0 should be cheap model (index 2)"
    assert tiers[1] == [1], "Tier 1 should be mid model (index 1)"
    assert tiers[2] == [0], "Tier 2 should be expensive model (index 0)"
    
    print("✓ test_three_tiers passed")


def main() -> int:
    """Run all tests."""
    print("=" * 60)
    print("CASCADE STRATEGY UNIT TESTS")
    print("=" * 60)
    print()
    
    tests = [
        test_tier_ordering,
        test_single_model_confident_accept,
        test_single_model_low_confidence_escalate,
        test_two_model_agreement_accept,
        test_two_model_disagreement_escalate,
        test_two_model_agreement_low_confidence_escalate,
        test_supermajority_accept,
        test_three_model_unanimous_accept,
        test_no_results_escalate,
        test_escalation_result_cost_savings,
        test_all_same_tier,
        test_three_tiers,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
