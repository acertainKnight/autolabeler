"""Confidence scoring and calibration for LLM outputs.

Implements hybrid confidence estimation using:
1. Logprob extraction (OpenAI) - P(label | text) from token probabilities
2. Self-consistency sampling (Claude, Gemini) - agreement across n samples
3. Isotonic regression calibration - post-hoc calibration for reliability

Evidence base:
- Logprob confidence with calibration reduces ECE by ~46% (Amazon 2024)
- Verbal confidence (high/medium/low) has ECE 0.13-0.43 (severely miscalibrated)
- Self-consistency provides reliable confidence estimates without logprobs
"""

import asyncio
from collections import Counter
from typing import Any

import numpy as np
from loguru import logger


class ConfidenceScorer:
    """Hybrid confidence scoring with calibration.
    
    Extracts confidence from:
    - Logprobs (OpenAI): Direct token probability
    - Self-consistency (Claude, Gemini): Agreement across n samples
    - Verbal fallback: Maps high/medium/low to probabilities
    
    Optionally calibrates via isotonic regression for better reliability.
    
    Example:
        >>> scorer = ConfidenceScorer()
        >>> # From logprobs
        >>> conf = scorer.from_logprobs({"0": -0.1, "1": -3.2}, label=0)
        >>> print(conf)  # 0.90
        >>> 
        >>> # From self-consistency
        >>> async def model_call(prompt, temp):
        ...     return {"label": 0}
        >>> conf = await scorer.from_self_consistency_async(model_call, "...", n=3)
        >>> print(conf)  # (0, 1.0)  - label 0, 100% agreement
    """
    
    def __init__(self):
        """Initialize confidence scorer.
        
        Calibrator is None until fit_calibrator() is called.
        """
        self.calibrator = None
    
    def from_logprobs(self, logprobs: dict[str, float], label: int | str) -> float:
        """Extract confidence from token logprobs.
        
        Parameters:
            logprobs: Dictionary mapping label tokens to probabilities (not log-probs!)
            label: The predicted label
            
        Returns:
            Confidence score (0-1)
            
        Example:
            >>> scorer = ConfidenceScorer()
            >>> conf = scorer.from_logprobs({"0": 0.92, "1": 0.05, "-1": 0.03}, label=0)
            >>> print(conf)  # 0.92
        """
        label_str = str(label)
        
        if label_str in logprobs:
            return float(logprobs[label_str])
        
        # Fallback: uncertain
        logger.warning(f"Label {label_str} not found in logprobs: {logprobs.keys()}")
        return 0.5
    
    async def from_self_consistency_async(
        self,
        model_call_func: Any,
        prompt: str,
        n_samples: int = 3,
        temperature: float = 0.3,
    ) -> tuple[int | str, float]:
        """Measure confidence via self-consistency sampling.
        
        Calls the model n times with higher temperature and measures
        agreement rate. High agreement = high confidence.
        
        Parameters:
            model_call_func: Async function that calls the model
            prompt: Prompt to send to model
            n_samples: Number of samples to take
            temperature: Sampling temperature (0.3 recommended for variance)
            
        Returns:
            Tuple of (majority_label, agreement_rate)
            
        Example:
            >>> async def call_model(prompt, temp):
            ...     # Your model call here
            ...     return {"label": 0}
            >>> label, conf = await scorer.from_self_consistency_async(
            ...     call_model, "Classify this...", n=3
            ... )
        """
        # Run n samples in parallel
        tasks = [
            model_call_func(prompt, temperature)
            for _ in range(n_samples)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        labels = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Self-consistency sample failed: {r}")
                continue
            if isinstance(r, dict) and "label" in r:
                labels.append(r["label"])
            elif hasattr(r, "parsed_json") and r.parsed_json and "label" in r.parsed_json:
                labels.append(r.parsed_json["label"])
        
        if not labels:
            logger.error("All self-consistency samples failed")
            return (None, 0.0)
        
        # Calculate majority and agreement
        counts = Counter(labels)
        majority_label = counts.most_common(1)[0][0]
        agreement_rate = counts[majority_label] / len(labels)
        
        return (majority_label, agreement_rate)
    
    def from_verbal(self, verbal: str) -> float:
        """Map verbal confidence to numeric score.
        
        Parameters:
            verbal: Confidence string ("high", "medium", "low")
            
        Returns:
            Numeric confidence score
            
        Note:
            Verbal confidence is known to be miscalibrated (ECE 0.13-0.43).
            Use logprobs or self-consistency when possible.
        """
        mapping = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5,
        }
        
        verbal_lower = verbal.lower().strip()
        return mapping.get(verbal_lower, 0.7)  # default to medium
    
    def fit_calibrator(
        self,
        raw_confidences: list[float],
        correct: list[bool],
    ) -> None:
        """Fit isotonic regression calibrator.
        
        Use high-confidence correct predictions from an existing model
        to build a calibration curve.
        
        Parameters:
            raw_confidences: Raw confidence scores from model
            correct: Whether each prediction was correct (True/False)
            
        Example:
            >>> scorer = ConfidenceScorer()
            >>> # Get calibration data
            >>> confidences = [0.9, 0.8, 0.7, 0.6]
            >>> correct = [True, True, False, False]
            >>> scorer.fit_calibrator(confidences, correct)
            >>> # Now calibrate new scores
            >>> calibrated = scorer.calibrate(0.85)
        """
        from sklearn.isotonic import IsotonicRegression
        
        if len(raw_confidences) < 10:
            logger.warning(
                f"Only {len(raw_confidences)} calibration samples. "
                "Need at least 10 for reliable calibration."
            )
            return
        
        self.calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        
        try:
            self.calibrator.fit(
                raw_confidences,
                [int(c) for c in correct]
            )
            logger.info(
                f"Fitted calibrator on {len(raw_confidences)} samples. "
                f"Mean correct: {np.mean(correct):.2%}"
            )
        except Exception as e:
            logger.error(f"Calibrator fitting failed: {e}")
            self.calibrator = None
    
    def calibrate(self, raw_confidence: float) -> float:
        """Apply calibration to raw confidence score.
        
        Parameters:
            raw_confidence: Raw confidence score
            
        Returns:
            Calibrated confidence score
            
        Note:
            Returns raw confidence if calibrator hasn't been fitted.
        """
        if self.calibrator is None:
            return raw_confidence
        
        try:
            calibrated = float(self.calibrator.predict([raw_confidence])[0])
            return np.clip(calibrated, 0.0, 1.0)
        except Exception as e:
            logger.warning(f"Calibration failed: {e}. Returning raw confidence.")
            return raw_confidence
    
    def calculate_ece(
        self,
        confidences: list[float],
        correct: list[bool],
        n_bins: int = 10,
    ) -> float:
        """Calculate Expected Calibration Error (ECE).
        
        ECE measures how well-calibrated confidence scores are.
        Lower is better (0 = perfect calibration).
        
        Parameters:
            confidences: Confidence scores
            correct: Whether each prediction was correct
            n_bins: Number of bins for binning confidences
            
        Returns:
            ECE score (0-1)
            
        Example:
            >>> scorer = ConfidenceScorer()
            >>> confidences = [0.9, 0.8, 0.7]
            >>> correct = [True, True, False]
            >>> ece = scorer.calculate_ece(confidences, correct)
            >>> print(f"ECE: {ece:.3f}")
        """
        confidences = np.array(confidences)
        correct = np.array(correct, dtype=float)
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins[:-1]) - 1
        
        ece = 0.0
        total_samples = len(confidences)
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_confidence = confidences[mask].mean()
            bin_accuracy = correct[mask].mean()
            bin_size = mask.sum()
            
            ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)
        
        return float(ece)
    
    def __repr__(self) -> str:
        """String representation."""
        status = "calibrated" if self.calibrator is not None else "uncalibrated"
        return f"ConfidenceScorer(status={status})"
