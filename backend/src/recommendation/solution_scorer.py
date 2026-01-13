"""Solution scoring for multi-criteria recommendation ranking.

Scores deployment configurations on 4 criteria (0-100 scale):
- Accuracy/Quality: Model capability (from Artificial Analysis benchmarks or param count fallback)
- Price: Cost efficiency (inverse of cost, normalized)
- Latency: SLO compliance with capped scoring (from Andre's PostgreSQL benchmarks)
- Complexity: Deployment simplicity (fewer GPUs = simpler)

INTEGRATION NOTE:
- Quality scoring: Uses Yuval's weighted_scores CSVs (Artificial Analysis benchmarks)
- Latency/Price/Complexity: Uses Andre's scoring logic and benchmark data
- Latency scoring uses min/max ranges from usecase_slo_workload.json to cap scoring
  (no extra credit for latencies below the "min" threshold)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import use-case quality scorer
try:
    from .usecase_quality_scorer import score_model_quality
    USE_CASE_QUALITY_AVAILABLE = True
except ImportError:
    USE_CASE_QUALITY_AVAILABLE = False


class SolutionScorer:
    """Score deployment configurations on 4 criteria (0-100 scale)."""

    # Accuracy tiers based on model parameter count (in billions)
    # Larger models generally have higher accuracy/capability
    ACCURACY_TIERS = {
        3: 40,
        4: 45,
        7: 55,
        8: 60,
        9: 62,
        14: 70,
        17: 72,
        20: 75,
        24: 78,
        27: 80,
        70: 85,
        120: 90,
        405: 95,
        480: 98,
    }

    # Complexity scores based on total GPU count
    COMPLEXITY_SCORES = {
        1: 100,
        2: 90,
        3: 82,
        4: 75,
        5: 70,
        6: 65,
        7: 62,
        8: 60,
    }

    # Default weights for balanced score
    DEFAULT_WEIGHTS = {
        "accuracy": 0.40,
        "price": 0.40,
        "latency": 0.10,
        "complexity": 0.10,
    }

    def __init__(self):
        """Initialize the SolutionScorer with SLO range data."""
        self.slo_ranges = self._load_slo_ranges()

    def _load_slo_ranges(self) -> dict:
        """
        Load use-case SLO min/max ranges from usecase_slo_workload.json.

        These ranges define "excellent" (min) to "acceptable" (max) latency targets.
        Latencies at or below "min" receive the same maximum score (no extra credit).

        Returns:
            Dict mapping use_case to SLO target ranges
        """
        # Path is: backend/src/recommendation/solution_scorer.py
        # Need to go up 4 levels to get to project root, then into data/
        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "business_context"
            / "use_case"
            / "configs"
            / "usecase_slo_workload.json"
        )
        try:
            with open(config_path) as f:
                data = json.load(f)
            logger.debug(f"Loaded SLO ranges from {config_path}")
            return data.get("use_case_slo_workload", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load SLO ranges from {config_path}: {e}")
            return {}

    def _calculate_capped_latency_score(
        self, actual_ms: float, min_ms: float, max_ms: float
    ) -> float:
        """
        Score latency with ceiling at "min" threshold.

        This implements a capped scoring approach where:
        - At or below min_ms: 100 (no extra credit for going lower)
        - Between min_ms and max_ms: linear interpolation 100→60
        - Above max_ms: penalty zone with scores below 60

        Args:
            actual_ms: Actual latency in milliseconds
            min_ms: "Excellent" threshold - latencies at or below get max score
            max_ms: "Acceptable" threshold - latencies at this point get score of 60

        Returns:
            Score 0-100
        """
        if actual_ms <= min_ms:
            # Capped at 100 - no extra credit for going below min
            return 100.0
        elif actual_ms <= max_ms:
            # Linear interpolation from 100 (at min) to 60 (at max)
            ratio = (actual_ms - min_ms) / (max_ms - min_ms)
            return 100 - (ratio * 40)
        else:
            # Above max - penalty zone
            # Score drops linearly, reaching 0 at 2x max
            overage_ratio = actual_ms / max_ms
            return max(0, 60 - (overage_ratio - 1) * 60)

    def score_accuracy(self, model_size_str: str, model_name: Optional[str] = None,
                        use_case: Optional[str] = None) -> int:
        """
        Score model accuracy/quality.
        
        Priority:
        1. Use-case specific benchmark score (Artificial Analysis data) if available
        2. Fallback to model size-based heuristic (Andre's original logic)

        Args:
            model_size_str: Model size string (e.g., "8B", "70B", "8x7B")
            model_name: Optional model name for use-case-specific scoring
            use_case: Optional use case for benchmark-based scoring

        Returns:
            Score 0-100
        """
        # Try use-case-specific quality scoring first (Yuval's contribution)
        if USE_CASE_QUALITY_AVAILABLE and model_name and use_case:
            quality_score = score_model_quality(model_name, use_case)
            if quality_score > 0:
                logger.debug(f"Quality score for {model_name} ({use_case}): {quality_score:.1f}")
                return int(quality_score)
        
        # Fallback to size-based heuristic (Andre's original logic)
        return self._score_accuracy_by_size(model_size_str)
    
    def _score_accuracy_by_size(self, model_size_str: str) -> int:
        """
        Score model accuracy based on parameter count tier (fallback).

        Args:
            model_size_str: Model size string (e.g., "8B", "70B", "8x7B")

        Returns:
            Score 0-100
        """
        param_count = self._extract_param_count(model_size_str)

        # Find the closest tier at or below the param count
        best_score = 40  # minimum score
        for tier_size, tier_score in sorted(self.ACCURACY_TIERS.items()):
            if param_count >= tier_size:
                best_score = tier_score
            else:
                break

        logger.debug(f"Accuracy score for {model_size_str} ({param_count}B): {best_score}")
        return best_score

    def score_price(self, cost_per_month: float, min_cost: float, max_cost: float) -> int:
        """
        Score price using non-linear formula for better differentiation.

        Enhanced Formula: 100 * (1 - (Monthly_Cost / Max_Monthly_Cost)^0.7)
        
        This creates more spread between configurations:
        - 1x A100: ~$1,100/mo → Score: 95
        - 2x A100: ~$2,200/mo → Score: 85  
        - 4x H100: ~$7,900/mo → Score: 60
        - 8x H100: ~$15,800/mo → Score: 35
        
        The power of 0.7 creates non-linear scaling that:
        - Rewards cheaper configurations more significantly
        - Creates meaningful gaps between similar-cost options
        - Penalizes expensive multi-GPU setups appropriately

        Args:
            cost_per_month: Configuration cost in USD/month
            min_cost: Minimum cost among all configurations
            max_cost: Maximum cost among all configurations

        Returns:
            Score 0-100 (100 = cheapest, 0 = most expensive)
        """
        import math
        
        if max_cost == 0:
            return 100
            
        if max_cost == min_cost:
            # All configs have same cost - give them high score
            return 95

        # Clamp cost to range
        cost = max(min_cost, min(max_cost, cost_per_month))
        
        # Non-linear scoring formula
        # Power of 0.7 creates better spread than linear
        cost_ratio = cost / max_cost
        score = int(100 * (1 - math.pow(cost_ratio, 0.7)))
        
        # Ensure minimum score of 5 for any valid config
        score = max(5, min(100, score))
        
        logger.debug(
            f"Price score for ${cost_per_month:,.0f}/mo: {score} "
            f"(ratio: {cost_ratio:.2f}, min: ${min_cost:,.0f}, max: ${max_cost:,.0f})"
        )
        return score

    def score_latency(
        self,
        predicted_ttft_ms: int,
        predicted_itl_ms: int,
        predicted_e2e_ms: int,
        target_ttft_ms: int,
        target_itl_ms: int,
        target_e2e_ms: int,
        use_case: str = None,
        near_miss_tolerance: float = 0.0,
    ) -> tuple[int, str]:
        """
        Score latency using CAPPED RANGE SCORING.

        Uses min/max ranges from usecase_slo_workload.json to cap latency scoring:
        - Latencies at or below "min" threshold get max score (100) - no extra credit
        - Latencies between min and max get linearly interpolated scores (100→60)
        - Latencies above max get penalty scores (<60)

        This prevents the system from over-rewarding low latency configurations
        that exceed user requirements.

        Args:
            predicted_ttft_ms: Predicted TTFT p95 in ms
            predicted_itl_ms: Predicted ITL p95 in ms
            predicted_e2e_ms: Predicted E2E p95 in ms
            target_ttft_ms: Target TTFT p95 in ms (used for SLO compliance check)
            target_itl_ms: Target ITL p95 in ms (used for SLO compliance check)
            target_e2e_ms: Target E2E p95 in ms (used for SLO compliance check)
            use_case: Use case for looking up SLO ranges
            near_miss_tolerance: How much over SLO to consider "near_miss" vs "exceeds"

        Returns:
            Tuple of (score 0-100, slo_status)
            - slo_status: "compliant", "near_miss", or "exceeds"
        """
        # ===== STEP 1: Calculate SLO compliance status =====
        # Configs are pre-filtered by find_configurations_meeting_slo(), but we still
        # need to determine if each is compliant vs near-miss for scoring purposes
        ratios = []
        if target_ttft_ms > 0:
            ratios.append(predicted_ttft_ms / target_ttft_ms)
        if target_itl_ms > 0:
            ratios.append(predicted_itl_ms / target_itl_ms)
        if target_e2e_ms > 0:
            ratios.append(predicted_e2e_ms / target_e2e_ms)

        if not ratios:
            # All targets are zero - this is a configuration error
            logger.error(
                f"All SLO targets are zero (ttft={target_ttft_ms}, itl={target_itl_ms}, "
                f"e2e={target_e2e_ms}). Cannot score latency."
            )
            return 0, "exceeds"

        worst_ratio = max(ratios)

        # Determine SLO status using the tolerance passed from capacity_planner
        if worst_ratio <= 1.0:
            slo_status = "compliant"
        elif worst_ratio <= (1.0 + near_miss_tolerance):
            slo_status = "near_miss"
        else:
            # This shouldn't happen if find_configurations_meeting_slo() is working correctly
            logger.error(
                f"Config exceeds SLO by {worst_ratio:.2f}x but passed database filter. "
                f"This indicates a bug in find_configurations_meeting_slo()."
            )
            return 0, "exceeds"

        # ===== STEP 2: Get SLO ranges for this use case =====
        slo_range = self.slo_ranges.get(use_case, {}).get("slo_targets", {})

        # If no SLO ranges available for this use case, all compliant configs get max score
        # We don't use arbitrary defaults - that would defeat use-case-specific scoring
        if not slo_range:
            logger.warning(
                f"No SLO ranges found for use_case='{use_case}'. "
                f"All compliant configs will receive max latency score."
            )
            return 100, slo_status

        # Get min/max ranges for each metric
        ttft_range = slo_range.get("ttft_ms", {})
        itl_range = slo_range.get("itl_ms", {})
        e2e_range = slo_range.get("e2e_ms", {})

        # Require both min and max for each metric
        if not all([
            ttft_range.get("min") and ttft_range.get("max"),
            itl_range.get("min") and itl_range.get("max"),
            e2e_range.get("min") and e2e_range.get("max"),
        ]):
            logger.warning(
                f"Incomplete SLO ranges for use_case='{use_case}'. "
                f"All compliant configs will receive max latency score."
            )
            return 100, slo_status

        ttft_min, ttft_max = ttft_range["min"], ttft_range["max"]
        itl_min, itl_max = itl_range["min"], itl_range["max"]
        e2e_min, e2e_max = e2e_range["min"], e2e_range["max"]

        # ===== STEP 3: Calculate CAPPED latency scores =====
        # Each metric scored using the capped approach:
        # - At or below min: 100 (no extra credit)
        # - Between min and max: linear 100→60
        # - Above max: penalty (<60)
        ttft_score = self._calculate_capped_latency_score(
            predicted_ttft_ms, ttft_min, ttft_max
        )
        itl_score = self._calculate_capped_latency_score(
            predicted_itl_ms, itl_min, itl_max
        )
        e2e_score = self._calculate_capped_latency_score(
            predicted_e2e_ms, e2e_min, e2e_max
        )

        # Weight: TTFT 34%, ITL 33%, E2E 33%
        final_score = ttft_score * 0.34 + itl_score * 0.33 + e2e_score * 0.33

        # Safety clamp in case weights are changed and don't sum to 1.0
        score = int(max(0, min(100, final_score)))

        logger.debug(
            f"Latency score: {score} ({slo_status}) [use_case={use_case or 'default'}] - "
            f"Capped scores: TTFT={ttft_score:.0f} (vs {ttft_min}-{ttft_max}ms), "
            f"ITL={itl_score:.0f} (vs {itl_min}-{itl_max}ms), "
            f"E2E={e2e_score:.0f} (vs {e2e_min}-{e2e_max}ms), "
            f"Predicted: TTFT={predicted_ttft_ms}, ITL={predicted_itl_ms}, E2E={predicted_e2e_ms}"
        )
        return score, slo_status

    def score_complexity(self, total_gpu_count: int) -> int:
        """
        Score complexity based on deployment topology.

        Args:
            total_gpu_count: Total GPUs required (tensor_parallel * replicas)

        Returns:
            Score 0-100 (100 = simplest, lower = more complex)
        """
        # Use predefined scores or calculate for larger counts
        if total_gpu_count in self.COMPLEXITY_SCORES:
            score = self.COMPLEXITY_SCORES[total_gpu_count]
        elif total_gpu_count > 8:
            # Linear decay for very large deployments
            score = max(40, 60 - (total_gpu_count - 8) * 2)
        else:
            score = 60

        logger.debug(f"Complexity score for {total_gpu_count} GPUs: {score}")
        return score

    def score_balanced(
        self,
        accuracy_score: int,
        price_score: int,
        latency_score: int,
        complexity_score: int,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Calculate weighted composite score.

        Args:
            accuracy_score: Accuracy score (0-100)
            price_score: Price score (0-100)
            latency_score: Latency score (0-100)
            complexity_score: Complexity score (0-100)
            weights: Optional custom weights (default: 40% accuracy, 40% price,
                     10% latency, 10% complexity)

        Returns:
            Weighted composite score (0-100)
        """
        w = weights or self.DEFAULT_WEIGHTS

        balanced = (
            accuracy_score * w["accuracy"]
            + price_score * w["price"]
            + latency_score * w["latency"]
            + complexity_score * w["complexity"]
        )

        logger.debug(
            f"Balanced score: {balanced:.1f} "
            f"(A={accuracy_score}, P={price_score}, L={latency_score}, C={complexity_score})"
        )
        return round(balanced, 1)

    def _extract_param_count(self, size_str: str) -> float:
        """
        Extract approximate parameter count from size string.

        Args:
            size_str: Size string (e.g., "8B", "70B", "8x7B")

        Returns:
            Approximate parameter count in billions
        """
        try:
            # Handle "8B", "70B" format
            if "B" in size_str and "x" not in size_str.lower():
                # Extract number before B
                match = re.search(r"(\d+\.?\d*)\s*B", size_str, re.IGNORECASE)
                if match:
                    return float(match.group(1))

            # Handle "8x7B" MoE format (use total params)
            if "x" in size_str.lower() and "B" in size_str.upper():
                match = re.search(r"(\d+)\s*x\s*(\d+\.?\d*)\s*B", size_str, re.IGNORECASE)
                if match:
                    return float(match.group(1)) * float(match.group(2))

            # Fallback: try to extract any number
            match = re.search(r"(\d+\.?\d*)", size_str)
            if match:
                return float(match.group(1))

            return 10.0  # Default fallback
        except Exception:
            logger.warning(f"Could not parse size string: {size_str}")
            return 10.0
