#!/usr/bin/env python3
"""GRPO Training for CQL RLVR — Real NeMo RL API.

This script follows the exact pattern from NVIDIA-NeMo/RL/examples/run_grpo.py.
It registers our custom CQL data processor, loads config, starts Ray, and runs
grpo_train() — the real NeMo RL training loop.

Usage (inside NeMo RL container or with NeMo RL installed):
    # Default config (production Nemotron-30B):
    uv run python scripts/run_grpo_cql.py

    # With custom config:
    uv run python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_config.yaml

    # Override any config key via Hydra-style CLI:
    uv run python scripts/run_grpo_cql.py ++grpo.max_num_steps=50 ++policy.optimizer.kwargs.lr=1e-5

    # Dry-run (validate config and data without NeMo RL dependencies):
    python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_config.yaml --dry-run
"""

import argparse
import os
import pprint
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="CQL RLVR — GRPO Training via NeMo RL")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "cql_nemo_rl_nemotron30b.yaml"),
        help="Path to GRPO YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data, print summary, exit (no NeMo RL required)",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def dry_run(config_path: str) -> None:
    """Validate config and data without NeMo RL dependencies."""
    import json
    import yaml

    print(f"[DRY RUN] Validating config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required top-level sections
    required = ["grpo", "loss_fn", "policy", "data", "env", "cluster"]
    for key in required:
        assert key in config, f"Missing required config section: {key}"
    print(f"  ✓ All required sections present: {required}")

    # Validate data files exist
    for split in ["train", "validation"]:
        dp = config["data"].get(split, {}).get("data_path")
        if dp:
            p = Path(dp) if os.path.isabs(dp) else PROJECT_ROOT / dp
            if p.exists():
                count = sum(1 for _ in open(p))
                print(f"  ✓ {split}: {p} ({count} examples)")
            else:
                print(f"  ✗ {split}: {p} NOT FOUND")

    # Validate system prompt
    sp = config["data"].get("default", {}).get("system_prompt_file")
    if sp:
        sp_path = Path(sp) if os.path.isabs(sp) else PROJECT_ROOT / sp
        if sp_path.exists():
            tokens_est = len(sp_path.read_text().split())
            print(f"  ✓ System prompt: {sp_path} (~{tokens_est} words)")
        else:
            print(f"  ✗ System prompt: {sp_path} NOT FOUND")

    # Validate model and LoRA
    model = config["policy"]["model_name"]
    print(f"  Model: {model}")

    dtensor = config["policy"].get("dtensor_cfg", {})
    lora = dtensor.get("lora_cfg", {}) if dtensor.get("enabled") else {}
    if lora.get("enabled"):
        print(f"  Training mode: LoRA (rank={lora.get('dim')}, alpha={lora.get('alpha')})")
        exclude = lora.get("exclude_modules", [])
        if "30B" in model or "Nano" in model:
            if not any("out_proj" in m for m in exclude):
                print("  ⚠ WARNING: Mamba2 model detected but *out_proj* not excluded from LoRA!")
            else:
                print(f"  ✓ LoRA excludes: {exclude}")
    else:
        print(f"  Training mode: Full fine-tuning (all parameters)")
        if dtensor.get("enabled"):
            ac = dtensor.get("activation_checkpointing", False)
            print(f"  Activation checkpointing: {'ON' if ac else 'OFF'}")
            if not ac and ("30B" in model or "Nano" in model):
                print("  ⚠ Consider activation_checkpointing: true for full-FT on 30B")

    # Validate cluster
    cluster = config["cluster"]
    gpus = cluster.get("gpus_per_node", 1) * cluster.get("num_nodes", 1)
    print(f"  Cluster: {cluster.get('num_nodes', 1)} node(s) × {cluster.get('gpus_per_node', 1)} GPUs = {gpus} total")

    # Scheduler info
    sched = config["policy"].get("scheduler")
    if sched:
        if isinstance(sched, list):
            names = [s["name"].split(".")[-1] for s in sched if "name" in s]
            print(f"  Scheduler: SequentialLR({' → '.join(names)})")
        elif isinstance(sched, dict):
            print(f"  Scheduler: {sched.get('name', 'unknown').split('.')[-1]}")

    print(f"\n[DRY RUN] Config valid. To train: uv run python {__file__} --config {config_path}")


def main() -> None:
    args, overrides = parse_args()

    if args.dry_run:
        dry_run(args.config)
        return

    # -- Real NeMo RL imports (only when actually training) --
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )
    from nemo_rl.utils.logger import get_next_experiment_dir

    # Register our custom CQL data processor before data loading
    from utils.cql_data_processor import register_cql_processor
    register_cql_processor()

    # Load and resolve config
    register_omegaconf_resolvers()
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")
    print("Final config:")
    pprint.pprint(config)

    # Set up experiment directories
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    # Initialize Ray cluster
    init_ray()

    # Set up tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Set up data
    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config["data"], config["env"]
    )

    # Set up policy, generation, cluster, etc.
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Check async mode
    if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
        unsupported = ["use_dynamic_sampling", "reward_scaling", "reward_shaping"]
        for feat in unsupported:
            if feat not in config["grpo"]:
                continue
            if feat == "use_dynamic_sampling":
                if config["grpo"][feat]:
                    raise NotImplementedError(f"{feat} is not supported with async GRPO")
            else:
                if config["grpo"][feat]["enabled"]:
                    raise NotImplementedError(f"{feat} is not supported with async GRPO")

        if config["data"].get("use_multiple_dataloader"):
            raise NotImplementedError(
                "use_multiple_dataloader is not supported with async GRPO"
            )

        from nemo_rl.algorithms.grpo import async_grpo_train

        print("🚀 Running async GRPO training for CQL")
        async_config = config["grpo"]["async_grpo"]
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        print("🚀 Running synchronous GRPO training for CQL")
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )

    print("✅ CQL GRPO training complete!")


if __name__ == "__main__":
    main()
