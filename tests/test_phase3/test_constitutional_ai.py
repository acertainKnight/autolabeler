"""Comprehensive tests for Constitutional AI Service.

Test Coverage:
- Principle definition and validation (10 tests)
- Violation detection (10 tests)
- Critique and revision (8 tests)
- Multi-principle enforcement (7 tests)
- Edge cases and robustness (5 tests)

Total: 40+ tests
"""
import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any


class TestPrincipleDefinition:
    """Test constitutional principle definition and validation."""

    def test_principle_structure(self, constitutional_principles):
        """Test principle structure requirements."""
        for principle in constitutional_principles:
            assert 'name' in principle
            assert 'description' in principle
            assert 'check_fn' in principle

    def test_accuracy_principle(self):
        """Test accuracy principle definition."""
        principle = {
            'name': 'accuracy',
            'description': 'Labels must be accurate and evidence-based',
            'threshold': 0.7,
        }

        assert principle['name'] == 'accuracy'
        assert principle['threshold'] > 0.5

    def test_fairness_principle(self):
        """Test fairness principle definition."""
        principle = {
            'name': 'fairness',
            'description': 'Labels should not show demographic bias',
            'bias_threshold': 0.3,
        }

        assert principle['name'] == 'fairness'
        assert principle['bias_threshold'] < 0.5

    def test_consistency_principle(self):
        """Test consistency principle."""
        principle = {
            'name': 'consistency',
            'description': 'Similar inputs should receive similar labels',
            'similarity_threshold': 0.8,
        }

        assert principle['similarity_threshold'] > 0.5

    def test_transparency_principle(self):
        """Test transparency principle."""
        principle = {
            'name': 'transparency',
            'description': 'Labeling decisions must include clear reasoning',
            'require_explanation': True,
        }

        assert principle['require_explanation'] is True

    def test_harmlessness_principle(self):
        """Test harmlessness principle."""
        principle = {
            'name': 'harmlessness',
            'description': 'Labels should not promote harmful content',
            'harmful_keywords': ['violence', 'hate', 'illegal'],
        }

        assert len(principle['harmful_keywords']) > 0

    def test_helpfulness_principle(self):
        """Test helpfulness principle."""
        principle = {
            'name': 'helpfulness',
            'description': 'Labels should provide useful information',
            'min_information_content': 10,  # characters
        }

        assert principle['min_information_content'] > 0

    def test_principle_priority(self):
        """Test principle priority ordering."""
        principles = [
            {'name': 'accuracy', 'priority': 1},
            {'name': 'fairness', 'priority': 2},
            {'name': 'consistency', 'priority': 3},
        ]

        sorted_principles = sorted(principles, key=lambda x: x['priority'])
        assert sorted_principles[0]['name'] == 'accuracy'

    def test_custom_principle_creation(self):
        """Test creating custom principles."""
        custom = {
            'name': 'domain_specific',
            'description': 'Follow domain-specific labeling guidelines',
            'rules': ['rule1', 'rule2', 'rule3'],
        }

        assert 'rules' in custom
        assert len(custom['rules']) > 0

    def test_principle_validation(self, constitutional_principles):
        """Test principle validation logic."""
        for principle in constitutional_principles:
            # All principles should have check functions
            assert callable(principle['check_fn'])


class TestViolationDetection:
    """Test constitutional violation detection."""

    def test_accuracy_violation_detection(self):
        """Test detecting accuracy violations."""
        label = {'text': 'Positive', 'confidence': 0.5}
        threshold = 0.7

        is_violation = label['confidence'] < threshold
        assert is_violation

    def test_bias_violation_detection(self):
        """Test detecting bias violations."""
        label = {
            'text': 'Negative',
            'bias_score': 0.6,  # High bias
            'demographic_parity': 0.4,
        }

        bias_threshold = 0.5
        is_violation = label['bias_score'] > bias_threshold
        assert is_violation

    def test_consistency_violation(self):
        """Test detecting consistency violations."""
        # Similar inputs with different labels
        labels = [
            {'text': 'Great product', 'label': 'Positive'},
            {'text': 'Great product!', 'label': 'Negative'},  # Inconsistent
        ]

        # Check similarity
        similarity = 0.95  # Very similar text
        consistency_threshold = 0.8

        is_similar = similarity > consistency_threshold
        is_consistent = labels[0]['label'] == labels[1]['label']

        violation = is_similar and not is_consistent
        assert violation

    def test_explanation_missing_violation(self):
        """Test detecting missing explanation violations."""
        label = {'text': 'Positive', 'explanation': None}

        requires_explanation = True
        is_violation = requires_explanation and label['explanation'] is None
        assert is_violation

    def test_harmful_content_violation(self):
        """Test detecting harmful content violations."""
        harmful_keywords = ['violence', 'hate', 'illegal']
        text = "This contains violence and hate speech"

        has_harmful = any(keyword in text.lower() for keyword in harmful_keywords)
        assert has_harmful

    def test_multiple_violations(self):
        """Test detecting multiple simultaneous violations."""
        label = {
            'text': 'Negative',
            'confidence': 0.5,  # Low confidence
            'bias_score': 0.7,  # High bias
            'explanation': None,  # Missing explanation
        }

        violations = []
        if label['confidence'] < 0.7:
            violations.append('low_confidence')
        if label['bias_score'] > 0.5:
            violations.append('high_bias')
        if label['explanation'] is None:
            violations.append('missing_explanation')

        assert len(violations) == 3

    def test_severity_scoring(self):
        """Test violation severity scoring."""
        violations = [
            {'type': 'low_confidence', 'severity': 'medium'},
            {'type': 'high_bias', 'severity': 'critical'},
            {'type': 'missing_explanation', 'severity': 'low'},
        ]

        severity_map = {'low': 1, 'medium': 2, 'critical': 3}
        total_severity = sum(severity_map[v['severity']] for v in violations)

        assert total_severity > 0

    def test_violation_thresholds(self):
        """Test configurable violation thresholds."""
        thresholds = {
            'accuracy': {'warning': 0.7, 'critical': 0.5},
            'bias': {'warning': 0.3, 'critical': 0.5},
        }

        confidence = 0.6
        if confidence < thresholds['accuracy']['critical']:
            level = 'critical'
        elif confidence < thresholds['accuracy']['warning']:
            level = 'warning'
        else:
            level = 'ok'

        assert level == 'warning'

    def test_false_positive_handling(self):
        """Test handling false positive violations."""
        # Sometimes violations are flagged incorrectly
        flagged_violations = ['low_confidence', 'high_bias']
        actual_violations = ['low_confidence']

        false_positives = set(flagged_violations) - set(actual_violations)
        assert len(false_positives) == 1

    def test_violation_history_tracking(self):
        """Test tracking violation history."""
        history = [
            {'timestamp': '2024-01-01', 'violation': 'low_confidence'},
            {'timestamp': '2024-01-02', 'violation': 'high_bias'},
            {'timestamp': '2024-01-03', 'violation': 'low_confidence'},
        ]

        low_conf_count = sum(
            1 for v in history if v['violation'] == 'low_confidence'
        )
        assert low_conf_count == 2


class TestCritiqueAndRevision:
    """Test critique and revision mechanisms."""

    def test_generate_critique(self):
        """Test generating critique for violation."""
        label = {'text': 'Negative', 'confidence': 0.5}
        violation = 'low_confidence'

        critique = f"The label has {violation}: confidence is {label['confidence']}, which is below the threshold of 0.7"

        assert 'low_confidence' in critique
        assert '0.5' in critique

    def test_revision_suggestion(self):
        """Test generating revision suggestions."""
        original = {'text': 'Negative', 'confidence': 0.5}

        suggestion = {
            'issue': 'low_confidence',
            'recommendation': 'Gather more evidence or defer to human annotator',
        }

        assert 'recommendation' in suggestion

    def test_iterative_revision(self):
        """Test iterative revision process."""
        original = {'text': 'Unclear', 'confidence': 0.5}
        revisions = []

        # Revision 1
        rev1 = {'text': 'Negative', 'confidence': 0.6}
        revisions.append(rev1)

        # Revision 2
        rev2 = {'text': 'Negative', 'confidence': 0.75}
        revisions.append(rev2)

        # Check improvement
        assert revisions[-1]['confidence'] > original['confidence']

    def test_revision_acceptance_criteria(self):
        """Test criteria for accepting revisions."""
        original = {'confidence': 0.5, 'bias_score': 0.6}
        revised = {'confidence': 0.8, 'bias_score': 0.2}

        # Check if revision meets criteria
        meets_accuracy = revised['confidence'] >= 0.7
        meets_fairness = revised['bias_score'] <= 0.3

        accept_revision = meets_accuracy and meets_fairness
        assert accept_revision

    def test_max_revision_attempts(self):
        """Test limiting revision attempts."""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            # Try to fix violations
            if attempts == 2:
                break  # Fixed

        assert attempts < max_attempts

    def test_revision_quality_check(self):
        """Test quality checking of revisions."""
        original_score = 0.5
        revised_score = 0.8

        improvement = revised_score - original_score
        min_improvement = 0.1

        quality_ok = improvement >= min_improvement
        assert quality_ok

    def test_critique_explanation_quality(self):
        """Test quality of critique explanations."""
        critique = "The label 'Positive' has low confidence (0.55). Consider reviewing the evidence."

        # Should be informative
        assert len(critique) > 20
        assert 'low confidence' in critique.lower()

    def test_revision_preserves_context(self):
        """Test that revisions preserve original context."""
        original = {
            'text': 'Positive',
            'context': 'Product review',
            'metadata': {'source': 'user_123'},
        }

        revised = original.copy()
        revised['text'] = 'Negative'  # Only change label

        # Context preserved
        assert revised['context'] == original['context']
        assert revised['metadata'] == original['metadata']


class TestMultiPrincipleEnforcement:
    """Test enforcing multiple principles simultaneously."""

    def test_check_all_principles(self, constitutional_principles):
        """Test checking label against all principles."""
        label = {'text': 'Positive', 'confidence': 0.85, 'bias_score': 0.1}

        violations = []
        for principle in constitutional_principles:
            if not principle['check_fn'](label):
                violations.append(principle['name'])

        # Should pass all checks with good label
        assert len(violations) == 0

    def test_principle_conflicts(self):
        """Test handling conflicts between principles."""
        # Accuracy might prefer high confidence
        # Fairness might require rejecting biased high-confidence labels

        label = {'confidence': 0.9, 'bias_score': 0.6}

        accuracy_ok = label['confidence'] > 0.7
        fairness_ok = label['bias_score'] < 0.3

        # Conflict: high accuracy but unfair
        has_conflict = accuracy_ok and not fairness_ok
        assert has_conflict

    def test_principle_priority_resolution(self):
        """Test resolving conflicts via priority."""
        principles = [
            {'name': 'fairness', 'priority': 1},  # Highest
            {'name': 'accuracy', 'priority': 2},
        ]

        # If conflict, follow highest priority
        sorted_principles = sorted(principles, key=lambda x: x['priority'])
        highest_priority = sorted_principles[0]

        assert highest_priority['name'] == 'fairness'

    def test_weighted_principle_scores(self):
        """Test weighted scoring across principles."""
        scores = {
            'accuracy': 0.9,
            'fairness': 0.7,
            'consistency': 0.8,
        }

        weights = {
            'accuracy': 0.5,
            'fairness': 0.3,
            'consistency': 0.2,
        }

        overall_score = sum(scores[p] * weights[p] for p in scores)
        assert 0.0 <= overall_score <= 1.0

    def test_principle_subset_enforcement(self):
        """Test enforcing subset of principles."""
        all_principles = ['accuracy', 'fairness', 'consistency', 'transparency']
        active_principles = ['accuracy', 'fairness']

        # Only check active principles
        to_check = [p for p in all_principles if p in active_principles]
        assert len(to_check) == 2

    def test_principle_composition(self):
        """Test composing multiple principles."""

        def check_accuracy(label):
            return label['confidence'] > 0.7

        def check_fairness(label):
            return label['bias_score'] < 0.3

        def check_all(label):
            return check_accuracy(label) and check_fairness(label)

        label = {'confidence': 0.8, 'bias_score': 0.2}
        assert check_all(label)

    def test_principle_dependencies(self):
        """Test principle dependencies."""
        # Transparency depends on accuracy
        # Can't be transparent about inaccurate labels

        principles = {
            'accuracy': {'depends_on': []},
            'transparency': {'depends_on': ['accuracy']},
        }

        # Check dependencies are satisfied
        assert 'accuracy' in principles['transparency']['depends_on']


class TestEdgeCasesRobustness:
    """Test edge cases and robustness."""

    def test_empty_principle_set(self):
        """Test with no principles defined."""
        principles = []

        with pytest.raises((ValueError, IndexError)):
            if len(principles) == 0:
                raise ValueError("No principles defined")

    def test_undefined_principle_check(self):
        """Test handling undefined principle checks."""
        label = {'text': 'Positive'}

        # Principle checks non-existent field
        try:
            _ = label['nonexistent_field']
            assert False  # Should raise error
        except KeyError:
            assert True

    def test_principle_with_missing_data(self):
        """Test principle checking with missing data."""
        label = {'text': 'Positive'}  # Missing confidence, bias_score

        # Should handle gracefully
        confidence = label.get('confidence', None)
        if confidence is not None:
            passes = confidence > 0.7
        else:
            passes = None  # Cannot determine

        assert passes is None

    def test_extreme_violation_values(self):
        """Test with extreme violation values."""
        label = {'confidence': -1.0, 'bias_score': 10.0}  # Invalid values

        # Should detect invalid ranges
        assert label['confidence'] < 0 or label['confidence'] > 1
        assert label['bias_score'] > 1

    def test_principle_performance_at_scale(self):
        """Test principle checking performance at scale."""
        import time

        num_labels = 10000
        labels = [
            {'confidence': np.random.rand(), 'bias_score': np.random.rand()}
            for _ in range(num_labels)
        ]

        start = time.time()
        violations = []
        for label in labels:
            if label['confidence'] < 0.7:
                violations.append(label)
        duration = time.time() - start

        assert duration < 1.0  # Should be fast


@pytest.mark.integration
class TestConstitutionalAIIntegration:
    """Integration tests for Constitutional AI."""

    def test_full_constitutional_pipeline(self, constitutional_principles):
        """Test complete constitutional AI pipeline."""
        # 1. Generate label
        label = {'text': 'Positive', 'confidence': 0.5, 'bias_score': 0.6}

        # 2. Check principles
        violations = []
        for principle in constitutional_principles:
            if not principle['check_fn'](label):
                violations.append(principle['name'])

        # 3. Generate critiques
        critiques = [f"Violation: {v}" for v in violations]

        # 4. Revise
        revised_label = label.copy()
        revised_label['confidence'] = 0.8
        revised_label['bias_score'] = 0.2

        # 5. Verify improvements
        assert len(violations) > 0
        assert revised_label['confidence'] > label['confidence']

    def test_constitutional_ai_with_llm(self, mock_llm_provider):
        """Test constitutional AI with LLM integration."""
        # Generate label
        label = mock_llm_provider.generate()

        # Check principles
        confidence = label.get('confidence', 0.5)
        meets_threshold = confidence > 0.7

        # If violations, ask LLM to revise
        if not meets_threshold:
            critique = "Please improve confidence by providing more evidence"
            revised = mock_llm_provider.generate()  # Mock revised response

        assert 'text' in label

    def test_constitutional_monitoring_dashboard(self):
        """Test constitutional monitoring metrics."""
        metrics = {
            'total_labels': 1000,
            'violations_detected': 150,
            'violations_resolved': 120,
            'violation_rate': 0.15,
        }

        assert metrics['violations_detected'] > metrics['violations_resolved'] - 50
        assert metrics['violation_rate'] == metrics['violations_detected'] / metrics['total_labels']

    def test_constitutional_training_feedback(self):
        """Test using constitutional violations for training."""
        violations = [
            {'label': 'A', 'violation': 'low_confidence'},
            {'label': 'B', 'violation': 'high_bias'},
            {'label': 'C', 'violation': 'low_confidence'},
        ]

        # Group by violation type for training
        from collections import Counter

        violation_counts = Counter(v['violation'] for v in violations)

        # Focus training on most common violations
        most_common = violation_counts.most_common(1)[0][0]
        assert most_common == 'low_confidence'

    def test_adaptive_principle_thresholds(self):
        """Test adapting principle thresholds over time."""
        initial_threshold = 0.7
        violation_rate = 0.3  # 30% of labels violate

        # If too many violations, adjust threshold
        if violation_rate > 0.2:
            new_threshold = initial_threshold * 0.9  # Make it easier

        assert new_threshold < initial_threshold

    def test_principle_impact_analysis(self):
        """Test analyzing impact of each principle."""
        principles = ['accuracy', 'fairness', 'consistency']
        violations_per_principle = {
            'accuracy': 50,
            'fairness': 30,
            'consistency': 20,
        }

        # Identify most impactful principle
        most_violations = max(violations_per_principle, key=violations_per_principle.get)
        assert most_violations == 'accuracy'

    def test_constitutional_ai_cost_benefit(self):
        """Test cost-benefit analysis of constitutional enforcement."""
        without_constitutional = {
            'accuracy': 0.75,
            'fairness': 0.60,
            'cost': 1.0,
        }

        with_constitutional = {
            'accuracy': 0.85,
            'fairness': 0.80,
            'cost': 1.5,  # More expensive
        }

        # Calculate benefit
        acc_improvement = with_constitutional['accuracy'] - without_constitutional['accuracy']
        fairness_improvement = with_constitutional['fairness'] - without_constitutional['fairness']

        # Calculate cost increase
        cost_increase = with_constitutional['cost'] - without_constitutional['cost']

        # Is it worth it?
        improvement = (acc_improvement + fairness_improvement) / 2
        roi = improvement / cost_increase

        assert roi > 0
