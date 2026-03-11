"""CQL reward environment for NeMo RL.

Minimal environment: valid CQL syntax → 1.0, invalid → 0.0.
Follows the same pattern as nemo_rl.environments.math_environment.MathEnvironment.
"""

import sys
from pathlib import Path
from typing import Any, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

# Add project root so we can import utils/
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.cql_validator import validate


class CQLEnvConfig(TypedDict):
    num_workers: int


class CQLEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)
class CQLEnvironment(EnvironmentInterface[CQLEnvironmentMetadata]):
    """Reward = 1.0 if generated CQL is syntactically valid, else 0.0."""

    def __init__(self, cfg: CQLEnvConfig):
        self.cfg = cfg

    def shutdown(self) -> None:
        pass

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CQLEnvironmentMetadata],
        **kwargs,
    ) -> EnvironmentReturn[CQLEnvironmentMetadata]:
        results = []
        for conversation in message_log_batch:
            response = "".join(
                str(m["content"]) for m in conversation if m["role"] == "assistant"
            )
            result = validate(response)
            results.append(1.0 if result.is_valid else 0.0)

        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()
        observations = [
            {"role": "environment", "content": f"reward={r}"} for r in results
        ]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics
