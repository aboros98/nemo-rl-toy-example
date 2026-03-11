# Creating Reward Environments for GRPO Training

This doc explains how to write a reward environment — the component that scores
model outputs during GRPO training. Two approaches exist: **NeMo RL environments**
(Ray actors, what we use) and **NeMo Gym resource servers** (HTTP/FastAPI). Both
work with NeMo RL training.

---

## Quick Orientation

During GRPO training, this is what happens every step:

```
1. Dataloader picks prompts from train.jsonl
   → each prompt has metadata (e.g., ground_truth CQL query)

2. vLLM generates N rollouts per prompt
   → model produces text like "<think>...</think>\n#event=..."

3. YOUR ENVIRONMENT scores each rollout        ← you write this
   → receives: conversation + metadata
   → returns: reward float per rollout

4. GRPO computes advantages, updates policy weights
```

**You only write step 3.** Everything else is handled by NeMo RL.

### Two Approaches

| | NeMo RL Environment | NeMo Gym Resource Server |
|---|---|---|
| **Repo** | [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) |
| **Pattern** | Python class (Ray actor) | FastAPI HTTP server |
| **Method you write** | `step(conversations, metadata) → rewards` | `async verify(request) → reward` |
| **Communication** | In-process via Ray | HTTP POST to `/verify` |
| **Best for** | Fast pure-Python rewards | External services, LLM judges, sandboxed execution |
| **We use** | ✅ `environments/cql_environment.py` | Planned for LogScale Docker compilation |

---

## Approach 1: NeMo RL Environment (What We Use)

A NeMo RL environment is a Python class decorated with `@ray.remote` that
implements two methods:
- `step()` — score a batch of model outputs
- `global_post_process_and_metrics()` — compute logging metrics

### What the Data Looks Like

When NeMo RL calls your `step()`, here is what you receive:

```python
# message_log_batch — a list of conversations (one per rollout)
# Each conversation is a list of message dicts:
message_log_batch = [
    # Rollout 1:
    [
        {"role": "user", "content": "Generate a CQL query that finds DNS requests..."},
        {"role": "assistant", "content": "<think>I need to filter DnsRequest events...</think>\n"
                                      "#event_simpleName=DnsRequest | where DomainName=/.*malware.*/"},
    ],
    # Rollout 2 (same prompt, different model response):
    [
        {"role": "user", "content": "Generate a CQL query that finds DNS requests..."},
        {"role": "assistant", "content": "SELECT * FROM dns WHERE domain LIKE '%malware%'"},
    ],
    # ... more rollouts
]

# metadata — one dict per rollout, comes from your data processor
metadata = [
    {"ground_truth": "#event_simpleName=DnsRequest | where DomainName=/.*malware.*/"},
    {"ground_truth": "#event_simpleName=DnsRequest | where DomainName=/.*malware.*/"},
]
```

Your job: compare each assistant response to the ground truth and return a reward.

### Step-by-Step: Create Your Environment

#### 1. Write the environment class

```python
"""environments/my_reward.py"""
import ray
import torch
from typing import Any, TypedDict
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class MyMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MyRewardEnv(EnvironmentInterface[MyMetadata]):
    """Called by NeMo RL to score rollouts during GRPO training."""

    def __init__(self, cfg: dict):
        # cfg comes from your YAML: env.my_reward.*
        self.cfg = cfg

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MyMetadata],
        **kwargs,
    ) -> EnvironmentReturn[MyMetadata]:
        rewards = []
        answers = []

        for conversation, meta in zip(message_log_batch, metadata):
            # 1. Extract the model's response (concatenate all assistant messages)
            response = "".join(
                str(m["content"]) for m in conversation if m["role"] == "assistant"
            )

            # 2. Score it against the ground truth
            ground_truth = meta["ground_truth"]
            score = 1.0 if response.strip() == ground_truth.strip() else 0.0

            rewards.append(score)
            answers.append(response.strip())

        # 3. Return all 6 required fields
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": f"reward={r:.2f}"} for r in rewards],
            metadata=metadata,                                  # pass through (single-turn)
            next_stop_strings=[None] * len(message_log_batch),  # no stop strings (single-turn)
            rewards=torch.tensor(rewards).cpu(),                # MUST be on CPU
            terminateds=torch.ones(len(rewards)).cpu(),         # 1.0 = episode done
            answers=answers,                                    # extracted answers for logging
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        # Zero out rewards for rollouts that didn't finish properly
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        return batch, {
            "accuracy": batch["rewards"].mean().item(),
            "fraction_ended": batch["is_end"].float().mean().item(),
            "mean_gen_length": batch["generation_lengths"].float().mean().item(),
        }

    def shutdown(self) -> None:
        pass  # cleanup if needed
```

#### 2. Register it in your training script

```python
# scripts/run_grpo.py — BEFORE calling grpo_train()
from nemo_rl.environments.utils import register_env
register_env("my_reward", "environments.my_reward.MyRewardEnv")
```

#### 3. Configure it in YAML

```yaml
# Two things must connect:

# A. Environment config — passed as cfg dict to __init__()
env:
  my_reward:           # must match register_env("my_reward", ...)
    num_workers: 8

# B. Data config — tells which env to use for this dataset
data:
  default:
    env_name: "my_reward"   # must match the key under env:
    processor: "my_data_processor"
```

That is it. NeMo RL creates Ray actors, batches rollouts, calls `step()`, and
feeds rewards into the GRPO loss.

---

### Real Example: Our CQL Environment

Here is the actual flow through `environments/cql_environment.py`, annotated:

```python
@ray.remote(max_restarts=-1, max_task_retries=-1)
class CQLEnvironment(EnvironmentInterface[CQLEnvironmentMetadata]):

    def __init__(self, cfg: CQLEnvConfig):
        self.cfg = cfg
        # Weights come from YAML: env.cql.reward_weights
        self.weights = cfg.get("reward_weights", {
            "format": 0.1, "structure": 0.3, "fields": 0.6, "execution": 0.0,
        })

    def step(self, message_log_batch, metadata, **kwargs):
        rewards_list, answers = [], []

        for conversation, meta in zip(message_log_batch, metadata):
            # Concatenate all assistant turns into one string
            response = "".join(
                str(m["content"]) for m in conversation if m["role"] == "assistant"
            )
            reference = meta.get("ground_truth", "")

            # compute_combined_reward is pure Python — same function used
            # by test_rewards_local.py and the reward playground
            result = compute_combined_reward(response, reference, self.weights)
            rewards_list.append(result["reward"])
            answers.append(result["extracted_cql"])

        return EnvironmentReturn(
            observations=[...],  # see full code in environments/cql_environment.py
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards_list).cpu(),
            terminateds=torch.ones_like(torch.tensor(rewards_list)).cpu(),
            answers=answers,
        )
```

**Concrete walkthrough — what happens when rollouts arrive:**

```
Golden query:
  #event_simpleName=ProcessRollup2 | where FileName="cmd.exe" | groupBy(ComputerName)

─────────────────────────────────────────────────────────────────────

Rollout A (perfect match with reasoning):
  "<think>I need process events for cmd.exe</think>
   #event_simpleName=ProcessRollup2 | where FileName="cmd.exe" | groupBy(ComputerName)"

  Step 1: extract_cql_from_response() strips <think>...</think> tags
  Step 2: compute_format_reward()     → 1.0  (both <think> and </think> present)
  Step 3: compute_structure_reward()  → 1.0  (functions: [where, groupBy] — exact match)
  Step 4: compute_field_reward()      → 1.0  (entities: {processrollup2, filename, "cmd.exe",
                                               computername} — all match)
  Combined = 0.1 × 1.0 + 0.3 × 1.0 + 0.6 × 1.0 = 1.0

─────────────────────────────────────────────────────────────────────

Rollout B (partial match — wrong event type):
  "<think>Let me try</think>
   #event_simpleName=DnsRequest | where FileName="cmd.exe""

  format:     1.0  (tags present)
  structure:  0.33 (functions: [where] — missing groupBy)
  fields:     0.57 (entities: {dnsrequest, filename, "cmd.exe"} — wrong event, missing computername)
  Combined = 0.1 × 1.0 + 0.3 × 0.33 + 0.6 × 0.57 = 0.54

─────────────────────────────────────────────────────────────────────

Rollout C (SQL instead of CQL — wrong language):
  "SELECT * FROM processes WHERE name = 'cmd.exe'"

  format:     0.0  (no <think>...</think> tags)
  structure:  0.0  (no CQL functions found)
  fields:     0.0  (no matching entities)
  Combined = 0.0
```

**GRPO uses the reward differences** — the model learns that Rollout A (1.0) is
better than B (0.54), which is better than C (0.0). Over training, it stops
generating SQL and starts producing correct CQL with reasoning.

---

### The Four Reward Components (Our CQL System)

All reward logic lives in `utils/cql_rewards.py` (pure Python, no GPU needed):

| Component | Weight | Measures | Range |
|-----------|--------|----------|-------|
| **Format** | 0.1 | `<think>...</think>` tag presence | 0.0 (none), 0.5 (one tag), 1.0 (both) |
| **Structure** | 0.3 | Jaccard similarity of pipeline function names | 0.0 – 1.0 |
| **Fields** | 0.6 | F1 score of tags, field names, string literals | 0.0 – 1.0 |
| **Execution** | 0.0 | Placeholder for Docker LogScale compilation | Always 0.0 |

**Structure reward** answers: "Did the model use the right operations?"
- Extracts ordered function names like `[where, groupBy, count]` from both queries
- Computes Jaccard similarity: |intersection| / |union|
- Example: golden has `[where, groupBy]`, model has `[where, count]` → 1/3 = 0.33

**Fields reward** answers: "Did the model reference the right data?"
- Extracts entities: #tag values, field names (identifiers), string literals
- Computes F1 score between the two entity sets
- Example: golden `{processrollup2, filename, "cmd.exe"}`, model `{dnsrequest, filename, "cmd.exe"}` → precision 2/3, recall 2/3, F1 = 0.67

Weights are configurable in YAML:
```yaml
env:
  cql:
    reward_weights:
      format: 0.1      # encourage chain-of-thought reasoning
      structure: 0.3    # right CQL operations
      fields: 0.6      # right data/entities
      execution: 0.0    # disabled until we have LogScale Docker
```

---

### EnvironmentReturn Fields Reference

`step()` must return a NamedTuple with exactly these 6 fields:

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `observations` | `list[dict]` | `{"role": "environment", "content": "..."}` — logged, not used for training |
| 2 | `metadata` | `list[dict]` | Pass through unchanged for single-turn; update for multi-turn |
| 3 | `next_stop_strings` | `list[None]` | `[None] * batch_size` for single-turn tasks |
| 4 | `rewards` | `Tensor` | Shape `[batch_size]` — **must be on CPU**, float32 |
| 5 | `terminateds` | `Tensor` | Shape `[batch_size]` — `1.0` = episode done (always 1.0 for single-turn) |
| 6 | `answers` | `list[str or None]` | Extracted answers — the "answer" part of each response |

### The `global_post_process_and_metrics()` Method

Called once per training step after all rollouts are scored.

**Critical line** — always include this:
```python
batch["rewards"] = batch["rewards"] * batch["is_end"]
```
This zeros out rewards for rollouts that did not terminate properly (e.g., hit max
token length without producing EOS). Without it, you reward incomplete garbage.

The `batch` dict contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `rewards` | `[B]` | Rewards from your `step()` |
| `is_end` | `[B]` | 1.0 if generation ended properly, 0.0 otherwise |
| `generation_lengths` | `[B]` | Number of tokens generated per rollout |
| `prompt_lengths` | `[B]` | Number of tokens in each prompt |

Everything you return in the `metrics` dict gets logged to TensorBoard/W&B:

```python
return batch, {
    "mean_reward": batch["rewards"].mean().item(),
    "reward_std": batch["rewards"].std().item(),
    "pct_perfect": (batch["rewards"] == 1.0).float().mean().item(),
    "fraction_ended": batch["is_end"].float().mean().item(),
    "generation_lengths": batch["generation_lengths"].float().mean().item(),
}
```

---

### Reward Design Patterns

#### Pattern 1: Binary (exact match)
```python
score = 1.0 if response.strip() == ground_truth.strip() else 0.0
```
Good for: math, factual QA. This is what NeMo RL's built-in `MathEnvironment` uses.

#### Pattern 2: Multi-component (partial credit)
```python
score  = 0.1 * format_score     # did the model show reasoning?
score += 0.3 * structure_score   # right operations/functions?
score += 0.6 * field_score       # right data/entities?
```
Good for: structured output (CQL, SQL, code). This is what our CQL environment uses.
The key invariant: **invalid must always score lower than valid**.

#### Pattern 3: LLM-as-judge
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

#### Pattern 4: Execution-based
```python
score = 1.0 if run_in_sandbox(response) == expected_output else 0.0
```
Good for: code generation (unit tests), SQL (execute and compare results).
NeMo RL's `CodeEnvironment` does this.

---

### Testing Locally (No GPU Needed)

The reward logic lives in pure Python (`utils/cql_rewards.py`), testable on Mac:

```bash
# Run all reward unit tests (31 tests)
python3 -m pytest utils/ -v

# Test with first training example as golden query
python3 scripts/test_rewards_local.py

# Test with your own golden query — see full breakdown of each component
python3 scripts/test_rewards_local.py \
  --golden '#event_simpleName=ProcessRollup2 | where FileName="cmd.exe" | groupBy(ComputerName)'

# Experiment with different weight distributions
python3 scripts/test_rewards_local.py --golden '...' \
  --weights '{"format":0.0,"structure":0.5,"fields":0.5,"execution":0.0}'
```

The test script shows full breakdowns — which functions matched/missed, which
entities were hallucinated or missing, component-by-component scoring:

```
=== Perfect match (with think tags) ===
Response: <think>Need process events for cmd.exe</think>
          #event_simpleName=ProcessRollup2 | where FileName="cmd.exe" | groupBy(ComputerName)

Format:    1.000  (both <think> and </think> found)
Structure: 1.000  (shared: where, groupBy | missing: none)
Fields:    1.000  (shared: processrollup2, filename, "cmd.exe", computername | missing: none)
Combined:  1.000  (= 0.1*1.0 + 0.3*1.0 + 0.6*1.0)
```

---

### Advanced: External Dependencies

Your `__init__` can load anything — models, API clients, databases:

```python
@ray.remote(max_restarts=-1, max_task_retries=-1)
class LogScaleRewardEnv(EnvironmentInterface[MyMetadata]):
    def __init__(self, cfg):
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
    logscale_url: "https://cloud.community.humio.com"
    repo: "sandbox"
```

### Advanced: Parallelizing Heavy Rewards

If reward computation is expensive (API calls, code execution), use worker actors:

```python
@ray.remote
class RewardWorker:
    def score_batch(self, items):
        return [self.score_one(item) for item in items]

@ray.remote(max_restarts=-1, max_task_retries=-1)
class ParallelRewardEnv(EnvironmentInterface[MyMetadata]):
    def __init__(self, cfg):
        self.workers = [RewardWorker.remote() for _ in range(cfg["num_workers"])]

    def step(self, message_log_batch, metadata, **kwargs):
        chunks = [message_log_batch[i::len(self.workers)] for i in range(len(self.workers))]
        futures = [w.score_batch.remote(c) for w, c in zip(self.workers, chunks)]
        results = []
        for batch_scores in ray.get(futures):
            results.extend(batch_scores)
        # ... build EnvironmentReturn as usual
```

---

## Approach 2: NeMo Gym Resource Server (FastAPI)

NeMo Gym ([NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)) uses a different
pattern. Instead of a Ray actor, you write a **FastAPI HTTP server** that exposes
a `/verify` endpoint. NeMo Gym (or NeMo RL via its `nemo_gym` adapter) sends
each model output to your server and gets back a reward.

```
┌──────────────────┐     HTTP POST /verify      ┌─────────────────────────┐
│  NeMo RL         │ ──────────────────────────► │  Your Resource Server   │
│  (GRPO training) │     {response, prompt}      │  (FastAPI app)          │
│                  │ ◄────────────────────────── │                         │
│                  │     {reward: 0.85}           │  - parse response       │
└──────────────────┘                              │  - compute reward       │
                                                  │  - return score         │
                                                  └─────────────────────────┘
```

### When to Use This Instead of Approach 1

| Situation | Use |
|---|---|
| Reward is pure Python, fast to compute | Approach 1 (NeMo RL) |
| Need external services (Docker, LLM judge, database) | Approach 2 (NeMo Gym) |
| Want to test rewards via HTTP independently of training | Approach 2 (NeMo Gym) |
| Multi-step agent tasks (tool use, browsing) | Approach 2 (NeMo Gym) |
| Simple single-turn scoring | Approach 1 (NeMo RL) |

### Minimal Template

```python
"""resources_servers/my_cql_reward/app.py"""
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MyCQLServerConfig(BaseResourcesServerConfig):
    name: str = "cql_reward"


class MyCQLResourcesServer(SimpleResourcesServer):
    config: MyCQLServerConfig

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # body.response.output_text = the model's full text output
        # body.responses_create_params.input = the original prompt messages
        model_output = body.response.output_text or ""

        # --- Your reward logic ---
        reward = 1.0 if "|" in model_output else 0.0  # has pipe operators?

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    MyCQLResourcesServer.run_webserver()
```

### Adding Task-Specific Fields

Subclass the request/response models to pass extra data:

```python
from pydantic import ConfigDict

class CQLVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    cql: str              # ground truth CQL query
    nl_query: str         # natural language question
    schema_context: str   # event type schema

class CQLVerifyResponse(BaseVerifyResponse):
    extracted_cql: str | None = None
    structure_score: float = 0.0
    fields_score: float = 0.0

class CQLResourcesServer(SimpleResourcesServer):
    async def verify(self, body: CQLVerifyRequest) -> CQLVerifyResponse:
        model_output = body.response.output_text or ""
        ground_truth = body.cql

        # Reuse the same reward logic we test locally
        from utils.cql_rewards import compute_combined_reward
        result = compute_combined_reward(model_output, ground_truth)

        return CQLVerifyResponse(
            **body.model_dump(),
            reward=result["reward"],
            extracted_cql=result["extracted_cql"],
            structure_score=result["structure"],
            fields_score=result["fields"],
        )
```

### Running and Testing

```bash
# Start the server
python resources_servers/my_cql_reward/app.py
# Listening on http://0.0.0.0:8080

# Test with curl
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -d '{
    "responses_create_params": {"input": [{"role": "user", "content": "Find DNS requests"}]},
    "response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "#event=DnsRequest | groupBy(DomainName)"}]}]},
    "cql": "#event_simpleName=DnsRequest | groupBy(DomainName)"
  }'
# Returns: {"reward": 0.72, "structure_score": 1.0, "fields_score": 0.5, ...}
```

### Bridging NeMo Gym Servers into NeMo RL Training

NeMo RL has a built-in `nemo_gym` adapter. In your YAML:

```yaml
env:
  nemo_gym:
    model_name: ${policy.model_name}
    base_urls: ["http://localhost:8080"]     # your resource server URL
    initial_global_config_dict: {}           # NeMo Gym config
```

The `NemoGym` adapter handles generation and training. Your server just scores.

### Real Examples from NeMo Gym Repo

| Server | What it does | Reward type |
|--------|-------------|-------------|
| `text_to_sql/` | LLM-as-judge SQL equivalence | Binary (0 or 1) with swap-check |
| `code_gen/` | Code execution + unit tests | Binary (passes/fails) |
| `math_with_judge/` | LLM judges math solutions | Binary |
| `instruction_following/` | Checks format constraints | Multi-component |

Source: [`resources_servers/`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers) in NVIDIA-NeMo/Gym.

---

## Checklist

### NeMo RL Environment (Approach 1)
- [ ] Class decorated with `@ray.remote(max_restarts=-1, max_task_retries=-1)`
- [ ] Inherits `EnvironmentInterface[YourMetadata]`
- [ ] `step()` returns `EnvironmentReturn` with **all 6 fields**
- [ ] All tensors (`rewards`, `terminateds`) on **CPU**
- [ ] `global_post_process_and_metrics()` includes `batch["rewards"] *= batch["is_end"]`
- [ ] `shutdown()` method exists (can be empty)
- [ ] Registered via `register_env("name", "module.ClassName")` before training starts
- [ ] YAML has both `env.your_name` and `data.default.env_name` pointing to it
- [ ] Reward logic testable locally without GPU

### NeMo Gym Resource Server (Approach 2)
- [ ] Subclasses `SimpleResourcesServer`
- [ ] Overrides `async def verify()` returning `BaseVerifyResponse` with `reward` field
- [ ] Runs standalone: `python app.py` starts the HTTP server
- [ ] Testable via `curl` or `pytest` without NeMo RL
- [ ] If using with NeMo RL: `env.nemo_gym.base_urls` points to server
