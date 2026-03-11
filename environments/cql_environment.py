"""CQL reward environment for NeMo RL — R1-style multi-component rewards.

Four reward components (DeepSeek-R1 pattern):
  1. Format    — proper <think>...</think> reasoning tags
  2. Structure — Jaccard of pipeline function names (right operations)
  3. Fields    — F1 of tags, field names, string literals (right data)
  4. Execution — placeholder for Docker LogScale compilation

Weights configurable via env config. Default: format=0.1, structure=0.3, fields=0.6, execution=0.0.
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

from utils.cql_rewards import compute_combined_reward


class CQLEnvConfig(TypedDict, total=False):
    num_workers: int
    reward_weights: dict[str, float]


class CQLEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)
class CQLEnvironment(EnvironmentInterface[CQLEnvironmentMetadata]):
    """R1-style reward: format + structure + fields + execution (placeholder)."""

    def __init__(self, cfg: CQLEnvConfig):
        self.cfg = cfg
        self.weights = cfg.get("reward_weights", {
            "format": 0.1,
            "structure": 0.3,
            "fields": 0.6,
            "execution": 0.0,
        })

    def shutdown(self) -> None:
        pass

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CQLEnvironmentMetadata],
        **kwargs,
    ) -> EnvironmentReturn[CQLEnvironmentMetadata]:
        rewards_list = []
        answers = []

        for conversation, meta in zip(message_log_batch, metadata):
            response = "".join(
                str(m["content"]) for m in conversation if m["role"] == "assistant"
            )
            reference = meta.get("ground_truth", "")
            result = compute_combined_reward(response, reference, self.weights)

            rewards_list.append(result["reward"])
            answers.append(result["extracted_cql"])

        rewards = torch.tensor(rewards_list).cpu()
        done = torch.ones_like(rewards).cpu()
        observations = [
            {"role": "environment", "content": f"reward={r:.3f}"} for r in rewards_list
        ]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards,
            terminateds=done,
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        metrics = {
            "mean_reward": batch["rewards"].mean().item(),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics
