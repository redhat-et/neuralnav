"""Data schemas for deployment intent and specifications."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TrafficProfile(BaseModel):
    """GuideLLM traffic profile for the deployment."""

    prompt_tokens: int = Field(..., description="Target prompt length in tokens (GuideLLM config)")
    output_tokens: int = Field(..., description="Target output length in tokens (GuideLLM config)")
    expected_qps: float | None = Field(None, description="Expected queries per second")


class SLOTargets(BaseModel):
    """Service Level Objective targets for the deployment."""

    ttft_p95_target_ms: int = Field(..., description="Time to First Token target (ms)")
    itl_p95_target_ms: int = Field(..., description="Inter-Token Latency target (ms/token)")
    e2e_p95_target_ms: int = Field(..., description="End-to-end latency target (ms)")
    percentile: str = Field(default="p95", description="Percentile for SLO comparison (mean, p90, p95, p99)")


class GPUConfig(BaseModel):
    """GPU configuration specification."""

    gpu_type: str = Field(..., description="GPU type (e.g., NVIDIA-L4, NVIDIA-A100-80GB)")
    gpu_count: int = Field(..., description="Total number of GPUs")
    tensor_parallel: int = Field(1, description="Tensor parallelism degree")
    replicas: int = Field(1, description="Number of independent replicas")


class ConfigurationScores(BaseModel):
    """Scores for a deployment configuration (0-100 scale)."""

    accuracy_score: int = Field(..., description="Model accuracy/capability score (0-100)")
    price_score: int = Field(..., description="Cost efficiency score - inverse of cost (0-100)")
    latency_score: int = Field(..., description="SLO headroom score (0-100)")
    complexity_score: int = Field(..., description="Deployment simplicity score (0-100)")
    balanced_score: float = Field(..., description="Weighted composite score (0-100)")
    slo_status: Literal["compliant", "near_miss", "exceeds"] = Field(
        ..., description="SLO compliance status"
    )


class DeploymentIntent(BaseModel):
    """Extracted deployment requirements from user conversation."""

    use_case: Literal[
        "chatbot_conversational",
        "code_completion",
        "code_generation_detailed",
        "translation",
        "content_generation",
        "summarization_short",
        "document_analysis_rag",
        "long_document_summarization",
        "research_legal_analysis",
    ] = Field(..., description="Primary use case type")

    experience_class: Literal["instant", "conversational", "interactive", "deferred", "batch"] = Field(
        ..., description="User experience class defining latency expectations"
    )

    user_count: int = Field(..., description="Number of users or scale")

    latency_requirement: Literal["very_high", "high", "medium", "low"] = Field(
        ...,
        description="Latency sensitivity (very_high=sub-500ms, high=sub-2s, medium=2-5s, low=>5s)",
    )

    throughput_priority: Literal["very_high", "high", "medium", "low"] = Field(
        default="medium", description="Importance of high request volume"
    )

    budget_constraint: Literal["strict", "moderate", "flexible", "none"] = Field(
        default="moderate", description="Cost sensitivity"
    )

    domain_specialization: list[str] = Field(
        default_factory=lambda: ["general"],
        description="Domain requirements (general, code, multilingual, enterprise)",
    )

    # Hardware preference extracted from natural language
    preferred_gpu_type: str = Field(
        default="Any GPU",
        description="User's preferred GPU type if mentioned (e.g., H100, H200, A100, L4) or 'Any GPU' if not specified"
    )

    # Priority hints extracted from natural language (used for weight calculation)
    accuracy_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Accuracy/quality importance"
    )
    cost_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Cost sensitivity (high = very cost sensitive)"
    )
    latency_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Latency importance"
    )
    complexity_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Preference for simpler deployments"
    )

    additional_context: str | None = Field(
        None, description="Any other relevant details from conversation"
    )


class DeploymentRecommendation(BaseModel):
    """Complete deployment recommendation with all specifications."""

    model_config = {"protected_namespaces": ()}

    # Input intent
    intent: DeploymentIntent

    # Generated specifications
    traffic_profile: TrafficProfile
    slo_targets: SLOTargets

    # Recommended configuration (None when no viable config found)
    model_id: str | None = Field(None, description="Recommended model identifier")
    model_name: str | None = Field(None, description="Human-readable model name")
    gpu_config: GPUConfig | None = None

    # Performance predictions (None when no viable config found)
    predicted_ttft_p95_ms: int | None = None
    predicted_itl_p95_ms: int | None = None
    predicted_e2e_p95_ms: int | None = None
    predicted_throughput_qps: float | None = None
    
    # All percentile metrics from benchmark (for UI to display based on user selection)
    benchmark_metrics: Optional[dict] = Field(default=None, description="All percentile metrics from benchmark")

    # Cost estimation (None when no viable config found)
    cost_per_hour_usd: float | None = None
    cost_per_month_usd: float | None = None

    # Metadata
    meets_slo: bool = Field(False, description="Whether configuration meets SLO targets")
    reasoning: str = Field(..., description="Explanation of recommendation choice or error message")
    alternative_options: list[dict] | None = Field(
        default=None, description="Alternative configurations with trade-offs"
    )

    # Multi-criteria scores (added for Solution Ranking feature)
    scores: ConfigurationScores | None = Field(
        default=None, description="Multi-criteria scores for ranking"
    )

    def to_alternative_dict(self) -> dict:
        """
        Convert recommendation to alternative option format.

        This is used when building the alternative_options list to avoid
        code duplication across capacity_planner.py and workflow.py.

        Returns:
            Dictionary with all fields needed for alternative comparison
        """
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "gpu_config": self.gpu_config.model_dump() if self.gpu_config else None,
            "predicted_ttft_p95_ms": self.predicted_ttft_p95_ms,
            "predicted_itl_p95_ms": self.predicted_itl_p95_ms,
            "predicted_e2e_p95_ms": self.predicted_e2e_p95_ms,
            "predicted_throughput_qps": self.predicted_throughput_qps,
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "cost_per_month_usd": self.cost_per_month_usd,
            "reasoning": self.reasoning,
            "scores": self.scores.model_dump() if self.scores else None,
            "benchmark_metrics": self.benchmark_metrics,
        }


class DeploymentSpecification(BaseModel):
    """
    Deployment specification generated from user intent.

    This is always generated successfully, even if no viable configurations exist.
    Contains the extracted intent, traffic profile, and SLO targets.
    """

    # User intent
    intent: DeploymentIntent

    # Generated specifications
    traffic_profile: TrafficProfile
    slo_targets: SLOTargets


class ConversationMessage(BaseModel):
    """Single message in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str | None = None


class RankedRecommendationsResponse(BaseModel):
    """Response containing multiple ranked recommendation lists.

    Provides 5 different views of the same configurations, each sorted
    by a different criterion to help users explore trade-offs.
    """

    # Filters applied
    min_accuracy_threshold: int | None = Field(
        default=None, description="Minimum accuracy score filter applied"
    )
    max_cost_ceiling: float | None = Field(
        default=None, description="Maximum monthly cost filter applied (USD)"
    )
    include_near_miss: bool = Field(
        default=True, description="Whether near-SLO configurations are included"
    )

    # Original specification
    specification: DeploymentSpecification | None = Field(
        default=None, description="The generated deployment specification"
    )

    # Ranked lists (top 5 each, sorted by respective criterion)
    best_accuracy: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by accuracy score"
    )
    lowest_cost: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by price score"
    )
    lowest_latency: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by latency score"
    )
    simplest: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by complexity score"
    )
    balanced: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by weighted composite score"
    )

    # Statistics
    total_configs_evaluated: int = Field(
        default=0, description="Total number of configurations evaluated"
    )
    configs_after_filters: int = Field(
        default=0, description="Number of configurations after applying filters"
    )
