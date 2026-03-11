#!/usr/bin/env python3
"""SFT Training for CQL — NeMo RL.

Usage:
    uv run python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml
    uv run python scripts/run_sft_cql.py --config configs/sft_cql_full_config.yaml ++sft.max_num_steps=100
    python scripts/run_sft_cql.py --dry-run  # validate config without NeMo RL
"""

import argparse
import os
import pprint
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="CQL SFT Training")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "sft_cql_config.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    return parser.parse_known_args()


def dry_run(config_path):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key in ["sft", "policy", "data", "cluster"]:
        assert key in config, f"Missing: {key}"
    print(f"[DRY RUN] Config OK: {config_path}")
    print(f"  Model: {config['policy']['model_name']}")
    print(f"  Steps: {config['sft']['max_num_steps']}")
    lora = config["policy"].get("dtensor_cfg", {}).get("lora_cfg", {})
    print(f"  Mode: {'LoRA (rank=' + str(lora.get('dim')) + ')' if lora.get('enabled') else 'Full fine-tuning'}")


def setup_data(tokenizer, data_config):
    """Set up SFT data — matches official examples/run_sft.py."""
    from datasets import concatenate_datasets
    from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset, update_single_dataset_config

    assert "train" in data_config

    task_data_processors, task_data_preprocessors, data_list = {}, {}, []
    if isinstance(data_config["train"], dict):
        data_config["train"] = [data_config["train"]]

    for cfg in data_config["train"]:
        if "default" in data_config and data_config["default"] is not None:
            update_single_dataset_config(cfg, data_config["default"])
        data = load_response_dataset(cfg)
        data_list.append(data)
        data_processor = partial(data.processor, add_bos=data_config["add_bos"], add_eos=data_config["add_eos"], add_generation_prompt=data_config["add_generation_prompt"])
        task_data_processors[data.task_name] = (data.task_spec, data_processor)
        if hasattr(data, "preprocessor") and data.preprocessor is not None:
            task_data_preprocessors[data.task_name] = data.preprocessor

    dataset = AllTaskProcessedDataset(
        concatenate_datasets([d.dataset for d in data_list]), tokenizer, None,
        task_data_processors, task_data_preprocessors=task_data_preprocessors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    # Validation
    val_task_data_processors, val_task_data_preprocessors, val_data_list = {}, {}, []
    for data in data_list:
        if hasattr(data, "val_dataset") and data.val_dataset is not None:
            val_data_list.append(data.val_dataset)
            val_task_data_processors[data.task_name] = task_data_processors[data.task_name]
            if data.task_name in task_data_preprocessors:
                val_task_data_preprocessors[data.task_name] = task_data_preprocessors[data.task_name]

    if "validation" in data_config and data_config["validation"] is not None:
        if isinstance(data_config["validation"], dict):
            data_config["validation"] = [data_config["validation"]]
        for cfg in data_config["validation"]:
            if "default" in data_config and data_config["default"] is not None:
                update_single_dataset_config(cfg, data_config["default"])
            val_data = load_response_dataset(cfg)
            val_data_list.append(val_data.dataset)
            val_processor = partial(val_data.processor, add_bos=data_config["add_bos"], add_eos=data_config["add_eos"], add_generation_prompt=data_config["add_generation_prompt"])
            val_task_data_processors[val_data.task_name] = (val_data.task_spec, val_processor)
            if hasattr(val_data, "preprocessor") and val_data.preprocessor is not None:
                val_task_data_preprocessors[val_data.task_name] = val_data.preprocessor

    val_dataset = None
    if val_data_list:
        val_dataset = AllTaskProcessedDataset(
            concatenate_datasets(val_data_list), tokenizer, None,
            val_task_data_processors, task_data_preprocessors=val_task_data_preprocessors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    return dataset, val_dataset


def main():
    args, overrides = parse_args()

    if args.dry_run:
        dry_run(args.config)
        return

    from omegaconf import OmegaConf
    from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
    from nemo_rl.utils.logger import get_next_experiment_dir

    # Register custom CQL processor
    from nemo_rl.data.processors import register_processor
    from utils.cql_data_processor import cql_data_processor
    register_processor("cql_data_processor", cql_data_processor)

    register_omegaconf_resolvers()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    dataset, val_dataset = setup_data(tokenizer, config["data"])

    policy, cluster, train_dataloader, val_dataloader, loss_fn, logger, checkpointer, sft_save_state, master_config = setup(config, tokenizer, dataset, val_dataset)
    sft_train(policy, train_dataloader, val_dataloader, tokenizer, loss_fn, master_config, logger, checkpointer, sft_save_state)


if __name__ == "__main__":
    main()
