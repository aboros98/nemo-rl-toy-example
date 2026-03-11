#!/usr/bin/env python3
"""SFT Training for CQL — Real NeMo RL API.

Supervised fine-tuning on CQL data as a warmup step before GRPO.
Follows the exact pattern from NVIDIA-NeMo/RL/examples/run_sft.py.

Usage (inside NeMo RL container or with NeMo RL installed):
    # Default config:
    uv run python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml

    # Override any config key:
    uv run python scripts/run_sft_cql.py ++sft.max_num_steps=100 ++policy.optimizer.kwargs.lr=2e-5

    # Dry-run:
    python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml --dry-run
"""

import argparse
import os
import pprint
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="CQL SFT Training via NeMo RL")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "sft_cql_config.yaml"),
        help="Path to SFT YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data, print summary, exit",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def dry_run(config_path: str) -> None:
    """Validate config and data without NeMo RL dependencies."""
    import yaml

    print(f"[DRY RUN] Validating SFT config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    required = ["sft", "policy", "data", "cluster"]
    for key in required:
        assert key in config, f"Missing required config section: {key}"
    print(f"  ✓ All required sections present: {required}")

    for split in ["train", "validation"]:
        dp = config["data"].get(split, {}).get("data_path")
        if dp:
            p = Path(dp) if os.path.isabs(dp) else PROJECT_ROOT / dp
            if p.exists():
                count = sum(1 for _ in open(p))
                print(f"  ✓ {split}: {p} ({count} examples)")
            else:
                print(f"  ✗ {split}: {p} NOT FOUND")

    print(f"  Model: {config['policy']['model_name']}")
    print(f"  Steps: {config['sft'].get('max_num_steps', '?')}")
    print(f"  Batch: {config['policy'].get('train_global_batch_size', '?')}")

    # LoRA / full-FT check
    model = config["policy"]["model_name"]
    dtensor = config["policy"].get("dtensor_cfg", {})
    lora = dtensor.get("lora_cfg", {}) if dtensor.get("enabled") else {}
    if lora.get("enabled"):
        print(f"  Training mode: LoRA (rank={lora.get('dim')}, alpha={lora.get('alpha')})")
        exclude = lora.get("exclude_modules", [])
        if "30B" in model or "Nano" in model:
            if not any("out_proj" in m for m in exclude):
                print("  ⚠ WARNING: Mamba2 model — *out_proj* not excluded from LoRA!")
            else:
                print(f"  ✓ LoRA excludes: {exclude}")
    else:
        print(f"  Training mode: Full fine-tuning (all parameters)")

    print(f"\n[DRY RUN] Config valid. To train: uv run python {__file__} --config {config_path}")


def setup_data(tokenizer, data_config):
    """Set up SFT data — follows examples/run_sft.py pattern."""
    from datasets import concatenate_datasets

    from nemo_rl.data.datasets import (
        AllTaskProcessedDataset,
        load_response_dataset,
        update_single_dataset_config,
    )

    assert "train" in data_config, (
        "The dataset config structure is updated. See "
        "https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md#datasets"
    )

    print("\n▶ Setting up data...")
    task_data_processors = {}
    task_data_preprocessors = {}
    data_list = []

    if isinstance(data_config["train"], dict):
        data_config["train"] = [data_config["train"]]

    for cfg in data_config["train"]:
        if "default" in data_config and data_config["default"] is not None:
            update_single_dataset_config(cfg, data_config["default"])
        data = load_response_dataset(cfg)
        data_list.append(data)
        data_processor = partial(
            data.processor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        )
        task_data_processors[data.task_name] = (data.task_spec, data_processor)
        if hasattr(data, "preprocessor") and data.preprocessor is not None:
            task_data_preprocessors[data.task_name] = data.preprocessor

    merged_data = concatenate_datasets([data.dataset for data in data_list])
    dataset = AllTaskProcessedDataset(
        merged_data,
        tokenizer,
        None,
        task_data_processors,
        task_data_preprocessors=task_data_preprocessors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Training dataset loaded with {len(dataset)} samples.")

    # Validation
    val_task_data_processors = {}
    val_task_data_preprocessors = {}
    val_data_list = []

    for data in data_list:
        if hasattr(data, "val_dataset") and data.val_dataset is not None:
            val_data_list.append(data.val_dataset)
            task_name = data.task_name
            val_task_data_processors[task_name] = task_data_processors[task_name]
            if task_name in task_data_preprocessors:
                val_task_data_preprocessors[task_name] = task_data_preprocessors[task_name]

    if "validation" in data_config and data_config["validation"] is not None:
        if isinstance(data_config["validation"], dict):
            data_config["validation"] = [data_config["validation"]]

        for cfg in data_config["validation"]:
            if "default" in data_config and data_config["default"] is not None:
                update_single_dataset_config(cfg, data_config["default"])
            val_data = load_response_dataset(cfg)
            val_data_list.append(val_data.dataset)
            val_data_processor = partial(
                val_data.processor,
                add_bos=data_config["add_bos"],
                add_eos=data_config["add_eos"],
                add_generation_prompt=data_config["add_generation_prompt"],
            )
            val_task_data_processors[val_data.task_name] = (val_data.task_spec, val_data_processor)
            if hasattr(val_data, "preprocessor") and val_data.preprocessor is not None:
                val_task_data_preprocessors[val_data.task_name] = val_data.preprocessor

    val_dataset = None
    if len(val_data_list) > 0:
        merged_val_data = concatenate_datasets(val_data_list)
        val_dataset = AllTaskProcessedDataset(
            merged_val_data,
            tokenizer,
            None,
            val_task_data_processors,
            task_data_preprocessors=val_task_data_preprocessors,
            max_seq_length=data_config["max_input_seq_length"],
        )
        print(f"  ✓ Validation dataset loaded with {len(val_dataset)} samples.")

    return dataset, val_dataset


def main() -> None:
    args, overrides = parse_args()

    if args.dry_run:
        dry_run(args.config)
        return

    from omegaconf import OmegaConf

    from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )
    from nemo_rl.utils.logger import get_next_experiment_dir

    # Register our custom CQL data processor
    from utils.cql_data_processor import register_cql_processor
    register_cql_processor()

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

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    dataset, val_dataset = setup_data(tokenizer, config["data"])

    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    print("🚀 Running SFT training for CQL")
    sft_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        sft_save_state,
    )

    print("✅ CQL SFT training complete!")


if __name__ == "__main__":
    main()
