"""Custom Flower Strategy extending FedAvg with simulated physical time tracking and telemetry."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np
from fedmaq.core.telemetry import TelemetryManager

logger = logging.getLogger(__name__)


class TelemetryFedAvg(FedAvg):
    """Custom FedAvg strategy tracking bandwidth delays, simulated time, and logging to telemetry."""

    def __init__(
        self,
        telemetry_manager: TelemetryManager,
        config: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.telemetry_manager = telemetry_manager
        self.config = config

        # Simulation parameters
        exp_config = config.get("experiment", config)
        self.num_clients = exp_config.get("num_clients", 10)
        self.simulated_time = 0.0
        self.cumulative_bytes = 0

        # Draw bandwidths from uniform U(bw_min, bw_max) in Mbps
        bw_cfg = exp_config.get("heterogeneity", {}).get("bandwidth", {})
        bw_min = float(bw_cfg.get("min_mbps", 5.0))
        bw_max = float(bw_cfg.get("max_mbps", 20.0))

        # Draw compute speeds from uniform U(comp_min, comp_max) in samples/second
        comp_cfg = exp_config.get("heterogeneity", {}).get("compute", {})
        comp_min = float(comp_cfg.get("min_samples_per_sec", 100.0))
        comp_max = float(comp_cfg.get("max_samples_per_sec", 500.0))

        # Seeded generator for reproducibility
        seed = config.get("seed", 42)
        rng = np.random.default_rng(seed)

        self.client_upload_bw = rng.uniform(bw_min, bw_max, size=self.num_clients)
        self.client_download_bw = rng.uniform(bw_min, bw_max, size=self.num_clients)
        self.client_comp_speed = rng.uniform(comp_min, comp_max, size=self.num_clients)

        logger.info(
            f"Initialized TelemetryFedAvg for {self.num_clients} clients. "
            f"Bandwidth: U({bw_min}, {bw_max}) Mbps. Compute: U({comp_min}, {comp_max}) samples/s."
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call FedAvg's aggregation
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if not results:
            return aggregated_parameters, metrics

        # Compute model size in bytes (based on the aggregated parameters)
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model_size_bytes = sum(arr.nbytes for arr in ndarrays)
        else:
            model_size_bytes = 0

        round_delays = []
        round_bytes_uploaded = 0
        round_bytes_downloaded = 0

        # Read epochs from fit config (defaults to 5)
        exp_config = self.config.get("experiment", self.config)
        epochs = exp_config.get("local_epochs", 5)

        for client_proxy, fit_res in results:
            # Map client to 0-indexed partition ID using client metrics, with fallback
            cid = int(fit_res.metrics.get("partition_id", -1))
            if cid < 0 or cid >= self.num_clients:
                cid = hash(client_proxy.cid) % self.num_clients

            # Bandwidth (Mbps -> bytes per second)
            upload_speed = (self.client_upload_bw[cid] * 10**6) / 8.0
            download_speed = (self.client_download_bw[cid] * 10**6) / 8.0
            comp_speed = self.client_comp_speed[cid]

            # 1. Download Delay
            t_download = model_size_bytes / download_speed
            round_bytes_downloaded += model_size_bytes

            # 2. Upload Delay (based on compressed/uncompressed sizes returned by client)
            bytes_uploaded = int(
                fit_res.metrics.get("bytes_uploaded", model_size_bytes)
            )
            t_upload = bytes_uploaded / upload_speed
            round_bytes_uploaded += bytes_uploaded

            # 3. Local Training Delay
            num_samples = fit_res.num_examples
            t_train = (num_samples * epochs) / comp_speed

            # Total round time for client
            client_total_time = t_download + t_train + t_upload
            round_delays.append(client_total_time)

        # For a synchronous server round, the round time is determined by the slowest client
        round_time = max(round_delays) if round_delays else 0.0
        self.simulated_time += round_time

        # Track total communication bytes for this round
        round_total_bytes = round_bytes_downloaded + round_bytes_uploaded
        self.cumulative_bytes += round_total_bytes
        self.last_round_bytes = round_total_bytes

        # Pass round stats in metrics dict
        metrics["round_time"] = round_time
        metrics["round_bytes"] = round_total_bytes

        return aggregated_parameters, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Perform global evaluation via FedAvg
        eval_res = super().evaluate(server_round, parameters)

        if eval_res is not None:
            loss, metrics = eval_res
            acc = float(metrics.get("accuracy", 0.0))

            # Retrieve round metrics tracked in self.aggregate_fit
            round_bytes = (
                getattr(self, "last_round_bytes", 0) if server_round > 0 else 0
            )

            # Log to console and WandB
            self.telemetry_manager.log(
                round_num=server_round,
                test_loss=loss,
                test_acc=acc,
                round_bytes=round_bytes,
                simulated_time=self.simulated_time,
            )

        return eval_res
