"""Telemetry manager for tracking metrics and logging to Weights & Biases."""

import logging
from typing import Any, Dict, Optional
import wandb

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Manages telemetry logging to WandB and console."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        exp_config = config.get("experiment", config)
        self.enabled = exp_config.get("telemetry", {}).get("wandb_enabled", True)
        self.project = exp_config.get("telemetry", {}).get(
            "project", "fedmaq-experiments"
        )
        self.run_name = exp_config.get("telemetry", {}).get("run_name", None)
        self.run = None
        self.cumulative_bytes = 0

    def init_wandb(self) -> None:
        """Initialize WandB connection if enabled."""
        if not self.enabled:
            logger.info("WandB telemetry is disabled.")
            return

        # Flatten config dict for wandb configuration logging
        flat_config = self._flatten_dict(self.config)

        exp_config = self.config.get("experiment", self.config)
        try:
            self.run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=flat_config,
                mode=exp_config.get("telemetry", {}).get("mode", "online"),
            )
            logger.info(
                f"WandB run initialized: {self.run.name if self.run else 'offline'}"
            )
        except Exception as exc:
            logger.warning(
                f"Could not initialize WandB: {exc}. Telemetry will be console-only."
            )
            self.enabled = False

    def log(
        self,
        round_num: int,
        test_loss: float,
        test_acc: float,
        round_bytes: int,
        simulated_time: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log key metrics for a communication round."""
        self.cumulative_bytes += round_bytes
        cumulative_kb = self.cumulative_bytes / 1024.0
        cumulative_mb = cumulative_kb / 1024.0

        metrics = {
            "round": round_num,
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "communication/round_bytes": round_bytes,
            "communication/cumulative_bytes": self.cumulative_bytes,
            "communication/cumulative_mb": cumulative_mb,
            "system/simulated_time_sec": simulated_time,
        }

        if additional_metrics:
            metrics.update(additional_metrics)

        # Print to console log
        logger.info(
            f"Round {round_num:3d} | Test Acc: {test_acc*100:6.2f}% | "
            f"Test Loss: {test_loss:6.4f} | Comm: {cumulative_mb:7.3f} MB | "
            f"Sim Time: {simulated_time:8.2f}s"
        )

        if self.enabled and self.run is not None:
            # We log against step = round_num
            self.run.log(metrics, step=round_num)

    def finish(self) -> None:
        """Close the WandB run."""
        if self.enabled and self.run is not None:
            self.run.finish()
            logger.info("WandB run finished.")

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Helper to flatten nested dictionaries (such as Hydra Omegaconf)."""
        items: Dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
