"""
A/B Testing framework for prompt experiments.

Provides deterministic user bucketing and experiment management
for testing different prompt variants and configurations.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Definition of an A/B test experiment.

    Attributes:
        name: Unique experiment identifier
        variants: Dict mapping variant names to their configurations
        traffic_split: Dict mapping variant names to traffic percentages (0.0-1.0)
        enabled: Whether the experiment is active
    """
    name: str
    variants: Dict[str, dict]
    traffic_split: Dict[str, float]
    enabled: bool = True

    def __post_init__(self):
        # Validate traffic split sums to 1.0
        total = sum(self.traffic_split.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Traffic split for {self.name} must sum to 1.0, got {total}"
            )


class ABTestManager:
    """Manages A/B test experiments and user assignment.

    Uses deterministic hashing to ensure users always see the same variant.
    """

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self._assignment_cache: Dict[str, tuple[str, dict]] = {}

    def register_experiment(self, experiment: Experiment) -> None:
        """Register a new experiment.

        Args:
            experiment: Experiment configuration to register
        """
        self.experiments[experiment.name] = experiment
        logger.info(
            f"Registered experiment '{experiment.name}' with variants: "
            f"{list(experiment.variants.keys())}"
        )

    def unregister_experiment(self, name: str) -> bool:
        """Unregister an experiment.

        Args:
            name: Experiment name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self.experiments:
            del self.experiments[name]
            # Clear cached assignments for this experiment
            self._assignment_cache = {
                k: v for k, v in self._assignment_cache.items()
                if not k.startswith(f"{name}:")
            }
            return True
        return False

    def get_variant(
        self, experiment_name: str, user_id: int
    ) -> tuple[str, dict]:
        """Get the variant for a user in an experiment.

        Uses deterministic hashing based on user_id and experiment name
        to ensure consistent assignment.

        Args:
            experiment_name: Name of the experiment
            user_id: User's Telegram ID

        Returns:
            Tuple of (variant_name, variant_config)
            Returns ("control", {}) if experiment not found or disabled
        """
        cache_key = f"{experiment_name}:{user_id}"

        # Check cache
        if cache_key in self._assignment_cache:
            return self._assignment_cache[cache_key]

        # Get experiment
        exp = self.experiments.get(experiment_name)
        if not exp or not exp.enabled:
            return "control", {}

        # Deterministic hash based on user_id and experiment name
        hash_input = f"{user_id}:{experiment_name}"
        hash_val = int(
            hashlib.md5(hash_input.encode()).hexdigest(), 16
        )
        bucket = (hash_val % 10000) / 10000.0  # 0.0 to 0.9999

        # Find the variant based on traffic split
        cumulative = 0.0
        for variant_name, split in exp.traffic_split.items():
            cumulative += split
            if bucket < cumulative:
                result = (variant_name, exp.variants.get(variant_name, {}))
                self._assignment_cache[cache_key] = result
                return result

        # Fallback to first variant
        first_variant = list(exp.variants.keys())[0]
        result = (first_variant, exp.variants[first_variant])
        self._assignment_cache[cache_key] = result
        return result

    def is_in_variant(
        self, experiment_name: str, user_id: int, variant_name: str
    ) -> bool:
        """Check if a user is in a specific variant.

        Args:
            experiment_name: Name of the experiment
            user_id: User's Telegram ID
            variant_name: Variant to check

        Returns:
            True if user is in the specified variant
        """
        assigned_variant, _ = self.get_variant(experiment_name, user_id)
        return assigned_variant == variant_name

    def get_all_experiments(self) -> Dict[str, dict]:
        """Get info about all registered experiments.

        Returns:
            Dict mapping experiment names to their info
        """
        return {
            name: {
                "variants": list(exp.variants.keys()),
                "traffic_split": exp.traffic_split,
                "enabled": exp.enabled,
            }
            for name, exp in self.experiments.items()
        }


# Global manager instance
_ab_manager: Optional[ABTestManager] = None


def get_ab_manager() -> ABTestManager:
    """Get or create the global A/B test manager."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager()
        _register_default_experiments(_ab_manager)
    return _ab_manager


def _register_default_experiments(manager: ABTestManager) -> None:
    """Register default experiments.

    These can be modified or extended based on testing needs.
    """
    # Example: System prompt variants
    manager.register_experiment(Experiment(
        name="system_prompt_style",
        variants={
            "control": {
                "style": "default",
                "temperature": 0.0,
            },
            "friendly": {
                "style": "friendly",
                "temperature": 0.1,
                "prompt_prefix": "Отвечай дружелюбно и с эмпатией.",
            },
            "concise": {
                "style": "concise",
                "temperature": 0.0,
                "prompt_prefix": "Отвечай кратко и по существу.",
            },
        },
        traffic_split={
            "control": 0.34,
            "friendly": 0.33,
            "concise": 0.33,
        },
        enabled=False,  # Disabled by default, enable when ready to test
    ))

    # Example: Retrieval strategy
    manager.register_experiment(Experiment(
        name="retrieval_strategy",
        variants={
            "control": {
                "use_hyde": True,
                "use_decomposition": False,
                "top_k": 6,
            },
            "aggressive_hyde": {
                "use_hyde": True,
                "use_decomposition": True,
                "top_k": 8,
            },
            "no_hyde": {
                "use_hyde": False,
                "use_decomposition": False,
                "top_k": 6,
            },
        },
        traffic_split={
            "control": 0.50,
            "aggressive_hyde": 0.25,
            "no_hyde": 0.25,
        },
        enabled=False,
    ))

    # Example: Re-ranking experiment
    manager.register_experiment(Experiment(
        name="reranking",
        variants={
            "control": {"use_reranking": True},
            "no_reranking": {"use_reranking": False},
        },
        traffic_split={
            "control": 0.5,
            "no_reranking": 0.5,
        },
        enabled=False,
    ))


async def log_ab_metric(
    session_factory,
    user_id: int,
    experiment_name: str,
    metric_name: str,
    metric_value: float,
) -> None:
    """Log a metric for an A/B test.

    Args:
        session_factory: DB session factory
        user_id: User's Telegram ID
        experiment_name: Name of the experiment
        metric_name: Name of the metric (e.g., "latency", "feedback_positive")
        metric_value: Numeric value of the metric
    """
    from bot.db import add_ab_test_log

    manager = get_ab_manager()
    variant, _ = manager.get_variant(experiment_name, user_id)

    await add_ab_test_log(
        session_factory,
        tg_id=user_id,
        experiment_name=experiment_name,
        variant=variant,
        metric_name=metric_name,
        metric_value=metric_value,
    )
