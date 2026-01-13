"""Capacity planning and GPU configuration recommendation.

IMPORTANT: PostgreSQL Migration (Phase 1):
- Uses traffic profile-based exact matching on (prompt_tokens, output_tokens)
- Queries benchmarks by exact traffic profile (512→256, 1024→1024, 4096→512, 10240→1536)
- Filters by p95 SLO compliance (TTFT, ITL, E2E)
- Uses pre-calculated e2e_p95 from benchmarks (not dynamic calculation)

Benchmarks collected using GuideLLM with fixed traffic profiles:
- Batching: vLLM continuous batching (dynamic, auto-configured)
- KV cache: enabled (vLLM default)
- Request pattern: steady-state load

TODO (Phase 2+): Parametric Performance Models
- Train regression models: f(prompt_tokens, output_tokens) → (ttft_p95, itl_p95, e2e_p95)
- Support arbitrary traffic profiles beyond the 4 GuideLLM defaults
- Interpolate for in-range predictions with confidence intervals
"""

import logging
import math

from ..context_intent.schema import (
    ConfigurationScores,
    DeploymentIntent,
    DeploymentRecommendation,
    GPUConfig,
    SLOTargets,
    TrafficProfile,
)
from ..knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository
from ..knowledge_base.model_catalog import ModelCatalog, ModelInfo
from .ranking_service import get_task_bonus
from .solution_scorer import SolutionScorer

logger = logging.getLogger(__name__)


class CapacityPlanner:
    """Plan GPU capacity to meet SLO targets and traffic requirements."""

    def __init__(
        self, benchmark_repo: BenchmarkRepository | None = None, catalog: ModelCatalog | None = None
    ):
        """
        Initialize capacity planner.

        Args:
            benchmark_repo: Benchmark repository
            catalog: Model catalog
        """
        self.benchmark_repo = benchmark_repo or BenchmarkRepository()
        self.catalog = catalog or ModelCatalog()

    def _calculate_required_replicas(self, qps_per_replica: float, required_qps: float) -> int:
        """
        Calculate number of replicas needed for traffic.

        Args:
            qps_per_replica: QPS capacity per replica
            required_qps: Required QPS to handle

        Returns:
            Number of replicas (minimum 1)
        """
        # Add 20% headroom for safety
        headroom_factor = 1.2
        required_capacity = required_qps * headroom_factor

        replicas = math.ceil(required_capacity / qps_per_replica)
        return max(1, replicas)

    def _generate_reasoning_from_bench(
        self,
        bench: BenchmarkData,
        gpu_config: GPUConfig,
        intent: DeploymentIntent,
        model: ModelInfo | None = None,
    ) -> str:
        """Generate explanation for recommendation from benchmark data.

        Args:
            bench: Benchmark data
            gpu_config: GPU configuration
            intent: Deployment intent
            model: Model info (optional, may be None if not in catalog)

        Returns:
            Reasoning string
        """
        reasons = []

        # Model selection
        if model:
            reasons.append(
                f"Selected {model.name} ({model.size_parameters}) for {intent.use_case} use case"
            )
        else:
            reasons.append(
                f"Selected {bench.model_hf_repo} for {intent.use_case} use case"
            )

        # GPU configuration
        if gpu_config.tensor_parallel > 1:
            reasons.append(
                f"Using {gpu_config.tensor_parallel}x tensor parallelism on {gpu_config.gpu_type} "
                f"for improved latency"
            )
        else:
            reasons.append(f"Deploying on {gpu_config.gpu_type} GPUs")

        # Scaling
        if gpu_config.replicas > 1:
            reasons.append(
                f"{gpu_config.replicas} independent replicas to handle {intent.user_count} users"
            )

        # Performance
        ttft_p95 = int(bench.ttft_p95) if bench.ttft_p95 else 0
        itl_p95 = int(bench.itl_p95) if bench.itl_p95 else 0
        reasons.append(f"Expected performance: TTFT={ttft_p95}ms (p95), ITL={itl_p95}ms (p95)")

        return ". ".join(reasons)

    def plan_all_capacities(
        self,
        traffic_profile: TrafficProfile,
        slo_targets: SLOTargets,
        intent: DeploymentIntent,
        include_near_miss: bool = False,  # Strict SLO filtering - no tolerance
        near_miss_tolerance: float = 0.0,  # No near-miss tolerance
        weights: dict[str, int] | None = None,  # Custom weights for balanced score
    ) -> list[DeploymentRecommendation]:
        """
        Plan GPU capacity and return ALL viable configurations meeting SLO.

        Queries benchmarks for all (model, GPU) configurations meeting SLO targets,
        then scores each on accuracy, price, latency, and complexity.

        Args:
            traffic_profile: Traffic characteristics (prompt_tokens, output_tokens)
            slo_targets: p95 SLO targets
            intent: Original deployment intent
            include_near_miss: Whether to include configs within tolerance of SLO
            near_miss_tolerance: How much over SLO to allow (0.2 = 20%)
            weights: Custom weights for balanced score (0-10 scale)
                     Keys: accuracy, price, latency, complexity

        Returns:
            List of DeploymentRecommendations with scores attached
        """
        scorer = SolutionScorer()
        all_configs: list[DeploymentRecommendation] = []

        # Determine SLO thresholds for query
        # If including near-miss, relax thresholds by tolerance
        if include_near_miss:
            query_ttft = int(slo_targets.ttft_p95_target_ms * (1 + near_miss_tolerance))
            query_itl = int(slo_targets.itl_p95_target_ms * (1 + near_miss_tolerance))
            query_e2e = int(slo_targets.e2e_p95_target_ms * (1 + near_miss_tolerance))
        else:
            query_ttft = slo_targets.ttft_p95_target_ms
            query_itl = slo_targets.itl_p95_target_ms
            query_e2e = slo_targets.e2e_p95_target_ms

        # Get percentile from SLO targets (default to p95 for backwards compatibility)
        percentile = getattr(slo_targets, 'percentile', 'p95')
        
        # Query PostgreSQL for configurations meeting relaxed SLO targets
        matching_configs = self.benchmark_repo.find_configurations_meeting_slo(
            prompt_tokens=traffic_profile.prompt_tokens,
            output_tokens=traffic_profile.output_tokens,
            ttft_p95_max_ms=query_ttft,
            itl_p95_max_ms=query_itl,
            e2e_p95_max_ms=query_e2e,
            min_qps=0,
            percentile=percentile,
        )

        if not matching_configs:
            logger.warning(
                f"No configurations found for traffic profile "
                f"({traffic_profile.prompt_tokens}→{traffic_profile.output_tokens})"
            )
            return []

        # Build model lookup from catalog for scoring
        # Models not in catalog will get accuracy score = 0
        all_models = self.catalog.get_all_models()
        model_lookup = {m.model_id.lower(): m for m in all_models}

        # Process each matching benchmark (no pre-filtering by model list)
        for bench in matching_configs:
            # Look up model in catalog (may be None if not in catalog)
            model = model_lookup.get(bench.model_hf_repo.lower())

            # Calculate required replicas to handle traffic
            replicas = self._calculate_required_replicas(
                bench.requests_per_second, traffic_profile.expected_qps or 1.0
            )

            # Create GPU config - gpu_count is PER REPLICA, not total
            gpu_config = GPUConfig(
                gpu_type=bench.hardware,
                gpu_count=bench.hardware_count,  # Per-replica GPU count
                tensor_parallel=bench.hardware_count,
                replicas=replicas,
            )

            # Calculate cost using TOTAL GPUs (per-replica * replicas)
            total_gpus = bench.hardware_count * replicas
            cost_per_hour = self.catalog.calculate_gpu_cost(
                bench.hardware, total_gpus, hours_per_month=1
            )

            if cost_per_hour is None:
                logger.warning(f"Could not calculate cost for {bench.hardware}")
                continue

            cost_per_month = cost_per_hour * 730  # ~30 days

            # Calculate latency score and SLO status
            predicted_ttft = int(bench.ttft_p95) if bench.ttft_p95 else 0
            predicted_itl = int(bench.itl_p95) if bench.itl_p95 else 0
            predicted_e2e = int(bench.e2e_p95) if bench.e2e_p95 else 0

            latency_score, slo_status = scorer.score_latency(
                predicted_ttft_ms=predicted_ttft,
                predicted_itl_ms=predicted_itl,
                predicted_e2e_ms=predicted_e2e,
                target_ttft_ms=slo_targets.ttft_p95_target_ms,
                target_itl_ms=slo_targets.itl_p95_target_ms,
                target_e2e_ms=slo_targets.e2e_p95_target_ms,
                use_case=intent.use_case,
                near_miss_tolerance=near_miss_tolerance,
            )

            # Skip if exceeds SLO and we're not including near-miss
            if slo_status == "exceeds" and not include_near_miss:
                continue

            # Calculate accuracy score - USE RAW AA BENCHMARK SCORE
            # This is the actual model accuracy from Artificial Analysis benchmarks
            # NOT a composite score with latency/budget bonuses
            from .usecase_quality_scorer import score_model_quality
            
            # Try to get raw AA score using the benchmark model name
            model_name_for_scoring = model.name if model else bench.model_hf_repo
            raw_accuracy = score_model_quality(model_name_for_scoring, intent.use_case)
            
            # If no score found, try with benchmark's model_hf_repo
            if raw_accuracy == 0 and bench.model_hf_repo:
                raw_accuracy = score_model_quality(bench.model_hf_repo, intent.use_case)
            
            accuracy_score = int(raw_accuracy)

            # Apply task-specific bonus to accuracy score
            # This boosts models that are well-suited for the specific use case
            task_bonus = get_task_bonus(model_name_for_scoring, intent.use_case)
            accuracy_score = min(accuracy_score + task_bonus, 100)  # Cap at 100

            complexity_score = scorer.score_complexity(total_gpus)  # Use total GPUs for complexity

            # Determine model_id and model_name
            # Use catalog info if available, otherwise use benchmark model_hf_repo
            model_id = model.model_id if model else bench.model_hf_repo
            model_name = model.name if model else bench.model_hf_repo

            # Build benchmark_metrics with all percentile values for UI display
            benchmark_metrics = {
                "ttft_mean": int(bench.ttft_mean) if bench.ttft_mean else 0,
                "ttft_p90": int(bench.ttft_p90) if bench.ttft_p90 else 0,
                "ttft_p95": int(bench.ttft_p95) if bench.ttft_p95 else 0,
                "ttft_p99": int(bench.ttft_p99) if bench.ttft_p99 else 0,
                "itl_mean": int(bench.itl_mean) if bench.itl_mean else 0,
                "itl_p90": int(bench.itl_p90) if bench.itl_p90 else 0,
                "itl_p95": int(bench.itl_p95) if bench.itl_p95 else 0,
                "itl_p99": int(bench.itl_p99) if bench.itl_p99 else 0,
                "e2e_mean": int(bench.e2e_mean) if bench.e2e_mean else 0,
                "e2e_p90": int(bench.e2e_p90) if bench.e2e_p90 else 0,
                "e2e_p95": int(bench.e2e_p95) if bench.e2e_p95 else 0,
                "e2e_p99": int(bench.e2e_p99) if bench.e2e_p99 else 0,
                "tps_mean": float(bench.tps_mean) if bench.tps_mean else 0,
                "tps_p90": float(bench.tps_p90) if bench.tps_p90 else 0,
                "tps_p95": float(bench.tps_p95) if bench.tps_p95 else 0,
                "tps_p99": float(bench.tps_p99) if bench.tps_p99 else 0,
                # RPS per replica from benchmark (for card display)
                "requests_per_second": float(bench.requests_per_second) if bench.requests_per_second else 0,
                # Data validation flag: True = estimated/interpolated, False = real benchmark
                "estimated": getattr(bench, 'estimated', False),
            }
            
            # Build recommendation (price score calculated later after we know min/max)
            recommendation = DeploymentRecommendation(
                intent=intent,
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                model_id=model_id,
                model_name=model_name,
                gpu_config=gpu_config,
                predicted_ttft_p95_ms=predicted_ttft,
                predicted_itl_p95_ms=predicted_itl,
                predicted_e2e_p95_ms=predicted_e2e,
                predicted_throughput_qps=bench.requests_per_second * replicas,
                cost_per_hour_usd=cost_per_hour,
                cost_per_month_usd=cost_per_month,
                meets_slo=(slo_status == "compliant"),
                reasoning=self._generate_reasoning_from_bench(bench, gpu_config, intent, model),
                benchmark_metrics=benchmark_metrics,  # All percentile data for UI
                # Temporary scores without price (will be updated below)
                scores=ConfigurationScores(
                    accuracy_score=accuracy_score,
                    price_score=0,  # Placeholder
                    latency_score=latency_score,
                    complexity_score=complexity_score,
                    balanced_score=0.0,  # Placeholder
                    slo_status=slo_status,
                ),
            )

            all_configs.append(recommendation)

        if not all_configs:
            logger.warning("No viable configurations found for any model")
            return []

        # Now calculate price scores (need min/max across all configs)
        costs = [rec.cost_per_month_usd for rec in all_configs if rec.cost_per_month_usd]
        if costs:
            min_cost = min(costs)
            max_cost = max(costs)

            for rec in all_configs:
                if rec.scores and rec.cost_per_month_usd:
                    # Update price score
                    price_score = scorer.score_price(
                        rec.cost_per_month_usd, min_cost, max_cost
                    )
                    rec.scores.price_score = price_score

                    # Calculate base balanced score with user weights
                    # Weights from UI are 0-10 integers, normalize to fractions
                    normalized_weights = None
                    if weights:
                        total = sum(weights.values()) or 1  # Avoid division by zero
                        normalized_weights = {
                            k: v / total for k, v in weights.items()
                        }

                    base_balanced = scorer.score_balanced(
                        accuracy_score=rec.scores.accuracy_score,
                        price_score=price_score,
                        latency_score=rec.scores.latency_score,
                        complexity_score=rec.scores.complexity_score,
                        weights=normalized_weights,
                    )
                    
                    # Apply scalability penalty based on replica count
                    # Configs needing many replicas are less efficient for high workloads
                    replicas = rec.gpu_config.replicas if rec.gpu_config else 1
                    if replicas <= 1:
                        scalability_factor = 1.0  # No penalty
                    elif replicas <= 3:
                        scalability_factor = 0.98  # 2% penalty
                    elif replicas <= 6:
                        scalability_factor = 0.95  # 5% penalty
                    elif replicas <= 10:
                        scalability_factor = 0.90  # 10% penalty
                    elif replicas <= 20:
                        scalability_factor = 0.80  # 20% penalty
                    else:
                        scalability_factor = 0.65  # 35% penalty for very large deployments
                    
                    rec.scores.balanced_score = round(base_balanced * scalability_factor, 1)

        # Count unique models in configurations
        unique_models = {rec.model_id for rec in all_configs}
        logger.info(f"Found {len(all_configs)} viable configurations across {len(unique_models)} models")
        return all_configs
