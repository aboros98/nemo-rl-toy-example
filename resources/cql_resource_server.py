#!/usr/bin/env python3
"""CQL Resource Server — FastAPI reward server for NL-to-CQL training.

Modular reward system with independent, pluggable components:
  - syntax:    Binary — is the CQL syntactically valid?
  - execution: Binary — does the query execute without errors?
  - ngram:     Continuous — bigram similarity to the golden query.

Each component returns [0, 1]. Combined via weighted sum.
Easy to add/remove reward components — just define a function and add a weight.

Hard invariant: an invalid query must NEVER score higher than a valid one.
Enforced by weight design: syntax_weight > ngram_weight.

Exposes /verify and /compute_rewards endpoints matching NeMo Gym's API.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cql_tokenizer import bigram_similarity
from utils.cql_validator import validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Reward weights — each component is [0, 1], combined via weighted sum.
#
# INVARIANT: syntax_weight > ngram_weight ensures that a valid query with
# zero ngram similarity (0.4) always beats an invalid query with perfect
# ngram similarity (0.3). No clamps or ceilings needed.
#
# To add a new reward component:
#   1. Write a function that returns [0, 1]
#   2. Add it to DEFAULT_REWARD_WEIGHTS
#   3. Add it to _compute_components()
# ============================================================================
DEFAULT_REWARD_WEIGHTS = {
    "syntax": 0.4,
    "execution": 0.3,
    "ngram": 0.3,
}

MOCK_EXECUTION_SUCCESS_RATE = 0.8

# ============================================================================
# Data models
# ============================================================================


class TaskItem(BaseModel):
    """A single NL-to-CQL task."""
    nl_query: str
    cql_query: str  # Golden CQL query
    schema_context: str = ""
    source: str = ""
    tags: list[str] = []


class PromptRequest(BaseModel):
    """Request for a batch of prompts."""
    batch_size: int = 4
    step: int = 0


class BatchRewardRequest(BaseModel):
    """Batch reward request."""
    completions: list[dict]


class VerifyRequest(BaseModel):
    """NeMo Gym verify request."""
    prompt: str = ""
    response: str = ""
    metadata: dict = {}


class VerifyResponse(BaseModel):
    """NeMo Gym verify response."""
    reward: float
    breakdown: dict


# ============================================================================
# Task dataset
# ============================================================================


class TaskDataset:
    """Loads and serves tasks from JSONL data."""

    def __init__(self, data_path: str):
        self.data: list[TaskItem] = []
        self._index = 0
        self._load(data_path)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Data file not found: {path}")
            return
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self.data.append(TaskItem(**rec))
                except Exception as e:
                    logger.warning(f"Skipping malformed record: {e}")
        logger.info(f"Loaded {len(self.data)} tasks from {path}")
        random.shuffle(self.data)

    def get_batch(self, batch_size: int) -> list[TaskItem]:
        """Get a batch of tasks, cycling through the dataset."""
        batch = []
        for _ in range(batch_size):
            if not self.data:
                break
            if self._index >= len(self.data):
                self._index = 0
                random.shuffle(self.data)
            batch.append(self.data[self._index])
            self._index += 1
        return batch


# ============================================================================
# Individual reward components — each returns [0.0, 1.0]
# ============================================================================


def reward_syntax(generated_cql: str) -> tuple[float, dict]:
    """Binary: 1.0 if syntactically valid, 0.0 otherwise."""
    result = validate(generated_cql)
    return (
        1.0 if result.is_valid else 0.0,
        {"valid": result.is_valid, "errors": result.errors},
    )


def reward_execution(generated_cql: str, syntax_valid: bool, mock: bool = True) -> float:
    """Binary: 1.0 if query executes, 0.0 otherwise.

    Returns 0.0 for syntactically invalid queries (can't execute).
    """
    if not syntax_valid:
        return 0.0
    if mock:
        logger.warning("[MOCK EXECUTION] Simulating query execution")
        return 1.0 if random.random() < MOCK_EXECUTION_SUCCESS_RATE else 0.0
    # Placeholder for real LogScale sandbox execution
    return 1.0


def reward_ngram(generated_cql: str, golden_cql: str) -> float:
    """Continuous [0, 1]: bigram similarity to the golden query."""
    return bigram_similarity(generated_cql, golden_cql)


# ============================================================================
# Combined reward
# ============================================================================


def compute_reward(
    generated_cql: str,
    golden_cql: str,
    mock_execution: bool = True,
    weights: dict[str, float] | None = None,
) -> dict:
    """Compute modular reward for a generated CQL query.

    Each component scores [0, 1] independently. Combined via weighted sum.
    Weights default to DEFAULT_REWARD_WEIGHTS. To add a new reward, define
    a scoring function and add its weight here.

    Returns:
        Dict with 'reward' (float in [0, 1]) and 'breakdown' (component details).
    """
    if weights is None:
        weights = DEFAULT_REWARD_WEIGHTS

    # Score each component
    syntax_score, syntax_info = reward_syntax(generated_cql)
    exec_score = reward_execution(generated_cql, syntax_info["valid"], mock_execution)
    ngram_score = reward_ngram(generated_cql, golden_cql)

    scores = {
        "syntax": syntax_score,
        "execution": exec_score,
        "ngram": ngram_score,
    }

    # Weighted sum
    reward = sum(weights[k] * scores[k] for k in scores)
    reward = round(max(0.0, min(1.0, reward)), 4)

    breakdown = {
        "syntax_valid": syntax_info["valid"],
        "syntax_errors": syntax_info["errors"],
        "execution_success": exec_score == 1.0,
        "ngram_similarity": round(ngram_score, 4),
        "component_scores": {k: round(v, 4) for k, v in scores.items()},
        "weights": weights,
        "reward": reward,
        "mock_execution": mock_execution,
    }

    return {"reward": reward, "breakdown": breakdown}


# ============================================================================
# FastAPI server
# ============================================================================


def create_app(data_path: str = None) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="CQL Resource Server", version="0.1.0")

    # Resolve data path
    if data_path is None:
        data_path = os.environ.get(
            "CQL_DATA_PATH",
            str(Path(__file__).resolve().parent.parent / "data" / "train.jsonl"),
        )

    dataset = TaskDataset(data_path)

    @app.get("/health")
    async def health():
        return {"status": "ok", "tasks_loaded": len(dataset.data)}

    @app.post("/get_prompts")
    async def get_prompts(request: PromptRequest):
        """Serve a batch of NL-to-CQL prompts."""
        batch = dataset.get_batch(request.batch_size)
        prompts = []
        for task in batch:
            prompt_text = _format_prompt(task)
            prompts.append({
                "prompt": prompt_text,
                "metadata": {
                    "nl_query": task.nl_query,
                    "golden_cql": task.cql_query,
                    "schema_context": task.schema_context,
                    "source": task.source,
                },
            })
        return {"prompts": prompts}

    @app.post("/compute_rewards")
    async def compute_rewards(request: BatchRewardRequest):
        """Compute rewards for a batch of completions."""
        rewards = []
        breakdowns = []
        for comp in request.completions:
            golden_cql = comp.get("metadata", {}).get("golden_cql", "")
            generated_cql = comp.get("completion", comp.get("response", ""))
            result = compute_reward(generated_cql, golden_cql, mock_execution=True)
            rewards.append(result["reward"])
            breakdowns.append(result["breakdown"])
        return {"rewards": rewards, "breakdowns": breakdowns}

    @app.post("/verify")
    async def verify(request: VerifyRequest):
        """NeMo Gym verify endpoint.

        Receives prompt + response, computes reward with full breakdown.

        NOTE: In real NeMo Gym, this would subclass SimpleResourcesServer and
        receive BaseVerifyRequest (with .response.choices[0].message.content).
        See docs/nemo_gym_rl_guide.md for the transition path.
        """
        golden_cql = request.metadata.get("golden_cql", "")
        generated_cql = request.response
        result = compute_reward(generated_cql, golden_cql, mock_execution=True)
        return VerifyResponse(
            reward=result["reward"],
            breakdown=result["breakdown"],
        )

    @app.post("/seed_session")
    async def seed_session():
        """NeMo Gym session seeding (no-op for CQL)."""
        return {}

    @app.post("/shutdown")
    async def shutdown():
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
        return {"status": "shutting_down"}

    return app


def _load_system_prompt() -> str:
    """Load system prompt from file, cached."""
    prompt_path = Path(__file__).resolve().parent / "cql_system_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text().strip()
    # Fallback if file missing
    return (
        "You are a CQL query expert. Output ONLY the query — "
        "no explanations, no markdown."
    )


_SYSTEM_PROMPT: str | None = None


def _get_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _load_system_prompt()
    return _SYSTEM_PROMPT


def _format_prompt(task: TaskItem) -> str:
    """Format a task into a prompt for the model.

    In production NeMo RL, the system prompt is injected via system_prompt_file
    and the user message comes from the JSONL 'prompt' field. This function
    combines both for the dummy training loop and /get_prompts endpoint.
    """
    # User message (matches _format_user_prompt in fetch_data.py)
    parts = []
    if task.schema_context and task.schema_context != "general CQL query":
        parts.append(f"Schema: {task.schema_context}")
    parts.append(f"Request: {task.nl_query}")
    user_msg = "\n".join(parts)

    return f"{_get_system_prompt()}\n\n{user_msg}\n\nQuery:\n"


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="CQL Resource Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    app = create_app(args.data_path)
    uvicorn.run(app, host=args.host, port=args.port)
