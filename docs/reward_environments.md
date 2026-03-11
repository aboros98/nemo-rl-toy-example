# Creating Custom Reward Environments — NeMo RL

How to build, register, and use your own reward environments for GRPO training.

---

## Architecture

```
GRPO loop:
  1. Sample prompts from dataset
  2. Generate N rollouts per prompt (vLLM)
  3. Send rollouts to YOUR ENVIRONMENT → get rewards    ← you write this
  4. Compute advantages, update policy
```

Your environment is a **Ray actor**. NeMo RL calls `step()` with a batch of conversations, you return rewards. That's it.

---

## Minimal Template

```python
"""environments/my_reward.py"""

import ray
import torch
from typing import Any, TypedDict
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class MyMetadata(TypedDict):
    ground_truth: str          # whatever your env needs per sample
    # add more fields as needed


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MyRewardEnv(EnvironmentInterface[MyMetadata]):

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Load models, validators, API clients, etc.

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MyMetadata],
        **kwargs,
    ) -> EnvironmentReturn[MyMetadata]:
        rewards = []
        responses = []
        for conversation, meta in zip(message_log_batch, metadata):
            # Extract the model's response (last assistant message)
            response = "".join(
                str(m["content"]) for m in conversation if m["role"] == "assistant"
            )
            responses.append(response)

            # ---- YOUR REWARD LOGIC HERE ----
            score = self.compute_reward(response, meta["ground_truth"])
            rewards.append(score)

        rewards_t = torch.tensor(rewards, dtype=torch.float32).cpu()
        terminateds = torch.ones(len(rewards), dtype=torch.float32).cpu()

        observations = [
            {"role": "environment", "content": f"reward={r:.2f}"}
            for r in rewards
        ]
        answers = [r.strip() for r in responses]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_t,
            terminateds=terminateds,
            answers=answers,
        )

    def compute_reward(self, response: str, ground_truth: str) -> float:
        """Replace with your actual reward logic."""
        return 1.0 if response.strip() == ground_truth.strip() else 0.0

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        # IMPORTANT: mask rewards for sequences that didn't end properly
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "fraction_ended": batch["is_end"].float().mean().item(),
            "mean_gen_length": batch["generation_lengths"].float().mean().item(),
        }
        return batch, metrics

    def shutdown(self) -> None:
        pass
```

---

## How to Register & Use

In your training script, **before** calling the GRPO loop:

```python
from nemo_rl.environments.utils import register_env

# FQN = fully qualified name: "module.path.ClassName"
register_env("my_reward", "environments.my_reward.MyRewardEnv")
```

In your YAML config:

```yaml
env:
  my_reward:         # must match the name in register_env()
    num_workers: 8   # Ray actors for parallel reward eval
```

That's it. NeMo RL handles actor creation, batching, and lifecycle.

---

## The Two Methods You Must Implement

### `step()` — Score a batch of rollouts

**Input:**
- `message_log_batch`: list of conversations, each is a list of `{"role": ..., "content": ...}` dicts
- `metadata`: list of per-sample metadata (ground truth, test cases, etc.)

**Output:** `EnvironmentReturn` with:
| Field | Type | Description |
|-------|------|-------------|
| `observations` | `list[dict]` | Feedback messages (logged, not used for training) |
| `metadata` | `list[dict]` | Updated metadata (pass through if single-turn) |
| `next_stop_strings` | `list[None]` | Stop strings for next turn (`[None]*N` for single-turn) |
| `rewards` | `torch.Tensor` | Shape `[batch_size]`, **must be on CPU** |
| `terminateds` | `torch.Tensor` | Shape `[batch_size]`, 1.0 = episode done |
| `answers` | `list[str]` | Extracted answers (model responses or post-processed text) |

**Typical pattern for single-turn reward (most common):**
```python
rewards = torch.tensor([score1, score2, ...]).cpu()
terminateds = torch.ones_like(rewards).cpu()  # always done after 1 turn
```

### `global_post_process_and_metrics()` — Compute logging metrics

Called once per training step after all rollouts are scored. The `batch` dict contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `rewards` | `[B]` | Raw rewards from `step()` |
| `is_end` | `[B]` | Whether generation ended properly (hit EOS/stop string) |
| `generation_lengths` | `[B]` | Token count of each generation |
| `prompt_lengths` | `[B]` | Token count of each prompt |

**Critical line** — always include this:
```python
batch["rewards"] = batch["rewards"] * batch["is_end"]
```
This zeros out rewards for sequences that didn't terminate properly (e.g., hit max length without producing a complete answer). Without this, you reward incomplete garbage.

The returned `metrics` dict is logged to TensorBoard automatically.

---

## Reward Design Patterns

### Pattern 1: Binary (exact match)
```python
def compute_reward(self, response, ground_truth):
    return 1.0 if response.strip() == ground_truth.strip() else 0.0
```
Good for: math, factual QA, code that must match exactly.

### Pattern 2: Graded (partial credit)
```python
def compute_reward(self, response, ground_truth):
    score = 0.0
    if is_syntactically_valid(response):
        score += 0.3
    if passes_unit_tests(response):
        score += 0.4
    score += 0.3 * similarity(response, ground_truth)
    return min(score, 1.0)
```
Good for: code generation, structured output, CQL.

### Pattern 3: LLM-as-judge
```python
def __init__(self, cfg):
    self.judge_client = openai.Client(base_url=cfg["judge_url"])

def compute_reward(self, response, ground_truth):
    result = self.judge_client.chat.completions.create(
        model="judge-model",
        messages=[{"role": "user", "content": f"Rate this: {response}"}],
    )
    return float(result.choices[0].message.content) / 10.0
```
Good for: open-ended tasks, style, helpfulness. Slow — use sparingly.

### Pattern 4: Multi-component with hard constraints
```python
def compute_reward(self, response, ground_truth):
    # Hard constraint: invalid syntax → capped at 0.0
    if not is_valid_syntax(response):
        return max(-0.5 + 0.1 * ngram_sim(response, ground_truth), -1.0)

    # Soft rewards for valid syntax
    base = 0.3
    base += 0.3 * passes_execution(response)
    base += 0.4 * ngram_sim(response, ground_truth)
    return base
```
Good for: CQL, SQL, any domain with verifiable constraints. The key invariant: **invalid must always score lower than valid**.

---

## Adding External Dependencies

Your `__init__` can load anything — models, API clients, databases:

```python
@ray.remote(max_restarts=-1, max_task_retries=-1)
class LogScaleRewardEnv(EnvironmentInterface[MyMetadata]):
    def __init__(self, cfg):
        self.cfg = cfg
        # Connect to LogScale sandbox for execution testing
        import requests
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {cfg['api_token']}"
        self.logscale_url = cfg["logscale_url"]

    def execute_query(self, cql: str) -> bool:
        resp = self.session.post(
            f"{self.logscale_url}/api/v1/repositories/{self.cfg['repo']}/query",
            json={"queryString": cql, "start": "1h", "end": "now"},
            timeout=10,
        )
        return resp.status_code == 200
```

Pass config via YAML:
```yaml
env:
  logscale_reward:
    num_workers: 4
    api_token: ${oc.env:LOGSCALE_TOKEN}
    logscale_url: "https://cloud.community.humio.com"
    repo: "sandbox"
```

---

## Parallelizing Heavy Rewards

If reward computation is expensive (API calls, code execution), use worker actors:

```python
@ray.remote
class RewardWorker:
    def __init__(self):
        # Load model, connect to API, etc.
        pass

    def score_batch(self, items):
        return [self.score_one(item) for item in items]

@ray.remote(max_restarts=-1, max_task_retries=-1)
class ParallelRewardEnv(EnvironmentInterface[MyMetadata]):
    def __init__(self, cfg):
        self.workers = [RewardWorker.remote() for _ in range(cfg["num_workers"])]

    def step(self, message_log_batch, metadata, **kwargs):
        # Split work across workers
        chunk_size = max(1, len(message_log_batch) // len(self.workers))
        chunks = [
            message_log_batch[i:i + chunk_size]
            for i in range(0, len(message_log_batch), chunk_size)
        ]
        futures = [
            self.workers[i % len(self.workers)].score_batch.remote(chunk)
            for i, chunk in enumerate(chunks)
        ]
        results = []
        for batch_scores in ray.get(futures):
            results.extend(batch_scores)

        rewards = torch.tensor(results).cpu()
        # ... rest is same
```

---

## Custom Metrics for TensorBoard

Everything you return in the `metrics` dict from `global_post_process_and_metrics` gets logged:

```python
def global_post_process_and_metrics(self, batch):
    batch["rewards"] = batch["rewards"] * batch["is_end"]

    rewards = batch["rewards"]
    is_end = batch["is_end"]

    metrics = {
        # Standard
        "accuracy": rewards.mean().item(),
        "fraction_ended": is_end.float().mean().item(),
        "mean_gen_length": batch["generation_lengths"].float().mean().item(),

        # Custom — add whatever you want
        "reward_std": rewards.std().item(),
        "reward_max": rewards.max().item(),
        "reward_min": rewards.min().item(),
        "pct_perfect": (rewards == 1.0).float().mean().item(),
        "pct_zero": (rewards == 0.0).float().mean().item(),
    }

    # Pass@k (fraction of prompts with at least one correct answer)
    # If you have num_generations_per_prompt rollouts grouped together:
    if "num_generations_per_prompt" in batch:
        n = batch["num_generations_per_prompt"]
        grouped = rewards.view(-1, n)
        metrics["pass@k"] = (grouped.max(dim=1).values > 0.5).float().mean().item()

    return batch, metrics
```

View in TensorBoard:
```bash
tensorboard --logdir logs/ --port 6006
```

---

## Existing Example: CQL Environment

See `environments/cql_environment.py` — R1-style multi-component reward (~98 lines):
- **Format reward** (0.2 weight): `<think>...</think>` tag presence → 0.0 / 0.5 / 1.0
- **N-gram reward** (0.8 weight): bigram F1 vs reference CQL using semantic tokenizer
- **Execution reward** (0.0 weight): placeholder for Docker LogScale compilation

Reward logic lives in `utils/cql_rewards.py` (pure Python, no GPU deps — testable on Mac).

```python
from utils.cql_rewards import compute_combined_reward

result = compute_combined_reward(response, reference_cql, weights={"format": 0.2, "ngram": 0.8, "execution": 0.0})
# result = {"reward": 0.84, "format": 1.0, "ngram": 0.8, "execution": 0.0, "extracted_cql": "...", "has_thinking": True}
```

---

## Checklist

- [ ] Class decorated with `@ray.remote(max_restarts=-1, max_task_retries=-1)`
- [ ] Inherits `EnvironmentInterface[YourMetadata]`
- [ ] `step()` returns `EnvironmentReturn` with **all 6 fields** including `answers`
- [ ] All tensors (`rewards`, `terminateds`) on **CPU**
- [ ] `global_post_process_and_metrics()` multiplies `rewards * is_end`
- [ ] `shutdown()` method exists (can be empty)
- [ ] Registered via `register_env("name", "module.ClassName")` before training starts
- [ ] Config YAML has `env: your_name: num_workers: N`
- [ ] Rewards bounded and consistent
