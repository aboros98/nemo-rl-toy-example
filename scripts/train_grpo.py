#!/usr/bin/env python3
"""GRPO Training Script for CQL RLVR.

Connects NeMo Gym (reward environment) with NeMo RL (GRPO training)
for reinforcement learning from verifiable rewards on NL-to-CQL generation.

Supports:
  --dry-run    Validate configs and data, print summary, exit
  --steps N    Override max training steps
  --gym-config Path to NeMo Gym config YAML
  --nemo-rl-config Path to NeMo RL GRPO config YAML

In dry-run mode: validates configs, checks data loading, prints summary.
In normal mode: runs GRPO training loop with reward logging.
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_grpo")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    p = Path(path)
    if not p.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)

    with open(p) as f:
        return yaml.safe_load(f)


def validate_data(data_dir: Path) -> dict:
    """Validate that training data exists and is well-formed."""
    stats = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            logger.warning(f"Data file not found: {path}")
            stats[split] = 0
            continue
        count = 0
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "nl_query" in rec and "cql_query" in rec:
                        count += 1
                except json.JSONDecodeError:
                    pass
        stats[split] = count
    return stats


def validate_configs(gym_config: dict, rl_config: dict) -> list[str]:
    """Validate configuration files and return warnings."""
    warnings = []

    # Check gym config
    if "reward" not in gym_config and "cql_resources_server" not in gym_config:
        warnings.append("Gym config missing reward or server configuration")

    # Check RL config
    grpo = rl_config.get("grpo", {})
    if not grpo:
        warnings.append("RL config missing 'grpo' section")
    else:
        steps = grpo.get("max_num_steps", 0)
        if steps < 1:
            warnings.append(f"max_num_steps is {steps}, should be >= 1")

    policy = rl_config.get("policy", {})
    if not policy.get("model_name"):
        warnings.append("RL config missing policy.model_name")

    model_name = policy.get("model_name", "").lower()

    # Check LoRA config
    dtensor = policy.get("dtensor_cfg", {})
    lora = dtensor.get("lora_cfg", {})
    if lora.get("enabled"):
        logger.info(
            f"  LoRA enabled: rank={lora.get('dim', '?')}, "
            f"alpha={lora.get('alpha', '?')}"
        )

        # Mamba2 out_proj check — critical for Nemotron-30B-A3B
        is_mamba_model = any(
            kw in model_name for kw in ["nemotron-3-nano", "30b-a3b", "mamba"]
        )
        exclude = lora.get("exclude_modules", [])
        has_outproj_exclude = any("out_proj" in str(m) for m in exclude)

        if is_mamba_model and not has_outproj_exclude:
            warnings.append(
                "CRITICAL: Mamba2 model detected but LoRA exclude_modules "
                "does not contain '*out_proj*'. Mamba2 out_proj modules have "
                "ZERO gradient with cuda_kernels_forward — training will "
                "silently fail. Add: exclude_modules: ['*out_proj*']"
            )

        if is_mamba_model and lora.get("use_triton", False):
            warnings.append(
                "Mamba2 model detected with use_triton: true. "
                "Triton LoRA kernels don't work with Mamba2 architecture. "
                "Set use_triton: false."
            )

    # Check vLLM TP vs cluster GPU count
    gen = policy.get("generation", {})
    vllm = gen.get("vllm_cfg", {})
    vllm_tp = vllm.get("tensor_parallel_size", 1)
    cluster = rl_config.get("cluster", {})
    gpus = cluster.get("gpus_per_node", 1)
    nodes = cluster.get("num_nodes", 1)
    total_gpus = gpus * nodes

    if vllm_tp > gpus:
        warnings.append(
            f"vLLM tensor_parallel_size ({vllm_tp}) exceeds "
            f"gpus_per_node ({gpus})"
        )

    logger.info(f"  Cluster: {nodes} node(s) x {gpus} GPU(s) = {total_gpus} total")
    if vllm_tp > 1:
        logger.info(f"  vLLM generation: TP={vllm_tp}")

    # Memory estimate for known models
    if "30b" in model_name:
        param_gb = 60  # 30B × 2 bytes (bf16)
        per_gpu_gb = param_gb / total_gpus
        logger.info(
            f"  Memory estimate: ~{param_gb}GB model weights (bf16), "
            f"~{per_gpu_gb:.0f}GB per GPU with FSDP2 sharding"
        )

    return warnings


def print_summary(
    gym_config: dict,
    rl_config: dict,
    data_stats: dict,
    steps: int,
    warnings: list[str],
) -> None:
    """Print a formatted configuration summary."""
    grpo = rl_config.get("grpo", {})
    policy = rl_config.get("policy", {})
    loss = rl_config.get("loss_fn", {})

    print("\n" + "=" * 70)
    print("CQL RLVR TRAINING CONFIGURATION SUMMARY")
    print("=" * 70)

    print(f"\n  Model:                {policy.get('model_name', 'unknown')}")
    print(f"  Max steps:            {steps}")
    print(f"  Prompts per step:     {grpo.get('num_prompts_per_step', '?')}")
    print(f"  Generations per prompt: {grpo.get('num_generations_per_prompt', '?')}")
    print(f"  Max sequence length:  {policy.get('max_total_sequence_length', '?')}")
    print(f"  Precision:            {policy.get('precision', '?')}")

    dtensor = policy.get("dtensor_cfg", {})
    lora = dtensor.get("lora_cfg", {})
    if lora.get("enabled"):
        print(f"  LoRA:                 rank={lora.get('dim')}, alpha={lora.get('alpha')}")
    else:
        print("  LoRA:                 disabled (full fine-tuning)")

    print(f"\n  GRPO clip ratio:      {loss.get('ratio_clip_min', '?')}")
    print(f"  GRPO clip high:       {loss.get('ratio_clip_max', '?')}")
    print(f"  KL coefficient:       {loss.get('reference_policy_kl_penalty', '?')}")
    print(f"  Token-level loss:     {loss.get('token_level_loss', '?')}")
    print(f"  Dynamic sampling:     {grpo.get('use_dynamic_sampling', '?')}")

    print(f"\n  Data splits:")
    for split, count in data_stats.items():
        print(f"    {split:10s}  {count:6d} examples")

    if warnings:
        print(f"\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    print("=" * 70)


class DummyTrainingLoop:
    """Simulated GRPO training loop for infrastructure validation.

    Uses the resource server for reward computation but simulates
    model generation (since we may not have the model weights locally).
    Logs to TensorBoard when enabled in config.
    """

    def __init__(
        self,
        gym_config: dict,
        rl_config: dict,
        data_dir: Path,
        log_dir: Path,
        max_steps: int,
    ):
        self.gym_config = gym_config
        self.rl_config = rl_config
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.metrics: list[dict] = []

        # Initialize TensorBoard writer if enabled
        self.tb_writer = None
        logger_cfg = rl_config.get("logger", {})
        if logger_cfg.get("tensorboard_enabled", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = log_dir / "tensorboard"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info(f"TensorBoard logging → {tb_dir}")
            except ImportError:
                # No torch — write metrics as JSON for later import.
                # Real NeMo RL has torch and uses SummaryWriter natively.
                logger.info(
                    "torch not installed (expected in dummy mode). "
                    "Metrics logged to CSV + JSON. "
                    "In production, NeMo RL logs to TensorBoard via torch.utils.tensorboard."
                )

        # Load training data
        self.train_data = []
        train_path = data_dir / "train.jsonl"
        if train_path.exists():
            with open(train_path) as f:
                for line in f:
                    self.train_data.append(json.loads(line))

    def _get_reward_server_url(self) -> str:
        """Get reward server URL from config."""
        env = self.rl_config.get("env", {})
        cql_env = env.get("cql", {})
        return cql_env.get("reward_server_url", "http://localhost:8080")

    def _simulate_generation(self, task: dict) -> list[str]:
        """Simulate model generation (returns dummy CQL queries).

        In the real pipeline, this would call the policy model via vLLM.
        """
        golden = task.get("cql_query", "")
        num_gens = self.rl_config.get("grpo", {}).get(
            "num_generations_per_prompt", 4
        )
        generations = []
        for i in range(num_gens):
            r = random.random()
            if r < 0.3:
                # Return golden query (good generation)
                generations.append(golden)
            elif r < 0.6:
                # Return slightly modified query
                parts = golden.split("|")
                if len(parts) > 1:
                    # Shuffle pipeline stages
                    shuffled = [parts[0]] + random.sample(
                        parts[1:], len(parts) - 1
                    )
                    generations.append("|".join(shuffled))
                else:
                    generations.append(golden)
            elif r < 0.85:
                # Return syntactically valid but different query
                event_types = [
                    "ProcessRollup2", "DnsRequest", "NetworkConnectIP4",
                ]
                et = random.choice(event_types)
                generations.append(
                    f"#event_simpleName={et} | count() | head(10)"
                )
            else:
                # Return invalid query
                generations.append("| | broken query (( no closing")

        return generations

    def _compute_rewards_local(
        self, generations: list[str], golden: str
    ) -> list[dict]:
        """Compute rewards locally using the resource server module."""
        from resources.cql_resource_server import compute_reward

        results = []
        for gen in generations:
            result = compute_reward(gen, golden, mock_execution=True)
            results.append(result)
        return results

    def _compute_rewards_remote(
        self, generations: list[str], golden: str
    ) -> list[dict]:
        """Compute rewards via the remote resource server."""
        import requests

        url = self._get_reward_server_url()
        completions = [
            {
                "completion": gen,
                "metadata": {"golden_cql": golden},
            }
            for gen in generations
        ]
        try:
            resp = requests.post(
                f"{url}/compute_rewards",
                json={"completions": completions},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {"reward": r, "breakdown": b}
                for r, b in zip(data["rewards"], data["breakdowns"])
            ]
        except Exception as e:
            logger.warning(
                f"Remote reward server unavailable ({e}), using local computation"
            )
            return self._compute_rewards_local(generations, golden)

    def run(self, use_remote: bool = True) -> None:
        """Run the dummy training loop."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.log_dir / "training_metrics.csv"

        logger.info(f"Starting GRPO training loop ({self.max_steps} steps)")
        logger.info("[MOCK] Simulating model generation — no real model loaded")
        logger.info(
            f"[MOCK] Using {'remote' if use_remote else 'local'} reward computation"
        )

        random.seed(42)

        for step in range(1, self.max_steps + 1):
            step_start = time.time()

            # Sample prompts for this step
            num_prompts = self.rl_config.get("grpo", {}).get(
                "num_prompts_per_step", 4
            )
            if self.train_data:
                tasks = random.choices(self.train_data, k=num_prompts)
            else:
                logger.error("No training data available!")
                break

            # Collect metrics across all prompts
            all_rewards = []
            syntax_valid_count = 0
            exec_success_count = 0
            ngram_sims = []
            total_gens = 0

            for task in tasks:
                # Generate completions
                generations = self._simulate_generation(task)
                golden = task.get("cql_query", "")

                # Compute rewards
                if use_remote:
                    results = self._compute_rewards_remote(generations, golden)
                else:
                    results = self._compute_rewards_local(generations, golden)

                for result in results:
                    total_gens += 1
                    reward = result["reward"]
                    breakdown = result.get("breakdown", {})
                    all_rewards.append(reward)

                    if breakdown.get("syntax_valid"):
                        syntax_valid_count += 1
                    if breakdown.get("execution_success"):
                        exec_success_count += 1
                    ngram_sims.append(breakdown.get("ngram_similarity", 0.0))

            # Compute step metrics
            step_time = time.time() - step_start

            step_metrics = {
                "step": step,
                "reward_mean": round(statistics.mean(all_rewards), 4),
                "reward_std": round(
                    statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0,
                    4,
                ),
                "reward_min": round(min(all_rewards), 4),
                "reward_max": round(max(all_rewards), 4),
                "syntax_valid_pct": round(
                    100 * syntax_valid_count / total_gens if total_gens else 0, 1
                ),
                "exec_success_pct": round(
                    100 * exec_success_count / total_gens if total_gens else 0, 1
                ),
                "ngram_sim_mean": round(
                    statistics.mean(ngram_sims) if ngram_sims else 0.0, 4
                ),
                "step_time_s": round(step_time, 2),
                "num_generations": total_gens,
            }
            self.metrics.append(step_metrics)

            # Log to TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar("train/reward_mean", step_metrics["reward_mean"], step)
                self.tb_writer.add_scalar("train/reward_std", step_metrics["reward_std"], step)
                self.tb_writer.add_scalar("train/reward_min", step_metrics["reward_min"], step)
                self.tb_writer.add_scalar("train/reward_max", step_metrics["reward_max"], step)
                self.tb_writer.add_scalar("train/syntax_valid_pct", step_metrics["syntax_valid_pct"], step)
                self.tb_writer.add_scalar("train/exec_success_pct", step_metrics["exec_success_pct"], step)
                self.tb_writer.add_scalar("train/ngram_sim_mean", step_metrics["ngram_sim_mean"], step)
                self.tb_writer.add_scalar("perf/step_time_s", step_metrics["step_time_s"], step)

            # Log
            logger.info(
                f"Step {step:3d}/{self.max_steps} | "
                f"reward={step_metrics['reward_mean']:+.3f} "
                f"(std={step_metrics['reward_std']:.3f}) | "
                f"syntax={step_metrics['syntax_valid_pct']:.0f}% | "
                f"exec={step_metrics['exec_success_pct']:.0f}% | "
                f"ngram={step_metrics['ngram_sim_mean']:.3f} | "
                f"t={step_metrics['step_time_s']:.1f}s"
            )

        # Save metrics to CSV
        self._save_metrics(metrics_path)

        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()
            logger.info("TensorBoard logs flushed. Run: tensorboard --logdir logs/tensorboard")

        # Print summary table
        self._print_summary()

    def _save_metrics(self, path: Path) -> None:
        """Save metrics to CSV."""
        if not self.metrics:
            return
        fieldnames = list(self.metrics[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics)
        logger.info(f"Metrics saved to {path}")

    def _print_summary(self) -> None:
        """Print clean summary table."""
        if not self.metrics:
            print("\nNo metrics collected.")
            return

        rewards = [m["reward_mean"] for m in self.metrics]
        syntax = [m["syntax_valid_pct"] for m in self.metrics]
        exec_s = [m["exec_success_pct"] for m in self.metrics]
        ngram = [m["ngram_sim_mean"] for m in self.metrics]

        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"  Steps completed:    {len(self.metrics)}")
        print(f"  Reward mean:        {statistics.mean(rewards):+.4f}")
        print(f"  Reward std:         {statistics.stdev(rewards) if len(rewards) > 1 else 0:.4f}")
        print(f"  Reward range:       [{min(rewards):+.4f}, {max(rewards):+.4f}]")
        print(f"  Syntax valid (avg): {statistics.mean(syntax):.1f}%")
        print(f"  Exec success (avg): {statistics.mean(exec_s):.1f}%")
        print(f"  N-gram sim (avg):   {statistics.mean(ngram):.4f}")

        # Check for NaN
        has_nan = any(math.isnan(r) for r in rewards)
        if has_nan:
            print("\n  WARNING: NaN rewards detected!")
        else:
            print("\n  All rewards are finite (no NaN).")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training for CQL RLVR"
    )
    parser.add_argument(
        "--gym-config",
        default=str(PROJECT_ROOT / "configs" / "cql_gym_config.yaml"),
        help="Path to NeMo Gym config",
    )
    parser.add_argument(
        "--nemo-rl-config",
        default=str(PROJECT_ROOT / "configs" / "cql_nemo_rl_config.yaml"),
        help="Path to NeMo RL GRPO config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and data, print summary, exit",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--local-rewards",
        action="store_true",
        help="Use local reward computation (no server needed)",
    )
    args = parser.parse_args()

    # Load configs
    logger.info("Loading configurations...")
    gym_config = load_yaml_config(args.gym_config)
    rl_config = load_yaml_config(args.nemo_rl_config)

    # Determine steps
    steps = args.steps or rl_config.get("grpo", {}).get("max_num_steps", 10)

    # Validate data
    data_dir = PROJECT_ROOT / "data"
    data_stats = validate_data(data_dir)

    # Validate configs
    warnings = validate_configs(gym_config, rl_config)

    # Print summary
    print_summary(gym_config, rl_config, data_stats, steps, warnings)

    if args.dry_run:
        logger.info("Dry run complete. No training performed.")
        if data_stats.get("train", 0) == 0:
            logger.warning(
                "No training data found. Run scripts/fetch_data.py first."
            )
            sys.exit(1)
        if warnings:
            logger.warning(f"{len(warnings)} config warning(s) found.")
        else:
            logger.info("Configuration looks good!")
        sys.exit(0)

    # Check data
    if data_stats.get("train", 0) == 0:
        logger.error("No training data. Run scripts/fetch_data.py first.")
        sys.exit(1)

    # Run training
    log_dir = PROJECT_ROOT / "logs"
    trainer = DummyTrainingLoop(
        gym_config=gym_config,
        rl_config=rl_config,
        data_dir=data_dir,
        log_dir=log_dir,
        max_steps=steps,
    )
    trainer.run(use_remote=not args.local_rewards)


if __name__ == "__main__":
    main()
