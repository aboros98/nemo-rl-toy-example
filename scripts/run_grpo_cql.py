#!/usr/bin/env python3
"""GRPO Training for CQL — NeMo RL.

Usage:
    uv run python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b.yaml
    uv run python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_config.yaml ++grpo.max_num_steps=50
    python scripts/run_grpo_cql.py --dry-run  # validate config without NeMo RL
"""

import argparse
import os
import pprint
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="CQL GRPO Training")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "cql_nemo_rl_nemotron30b.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    return parser.parse_known_args()


def dry_run(config_path):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key in ["grpo", "loss_fn", "policy", "data", "env", "cluster"]:
        assert key in config, f"Missing: {key}"
    print(f"[DRY RUN] Config OK: {config_path}")
    print(f"  Model: {config['policy']['model_name']}")
    print(f"  Steps: {config['grpo']['max_num_steps']}")
    print(f"  GPUs: {config['cluster'].get('num_nodes', 1) * config['cluster'].get('gpus_per_node', 1)}")
    lora = config["policy"].get("dtensor_cfg", {}).get("lora_cfg", {})
    print(f"  Mode: {'LoRA (rank=' + str(lora.get('dim')) + ')' if lora.get('enabled') else 'Full fine-tuning'}")


def main():
    args, overrides = parse_args()

    if args.dry_run:
        dry_run(args.config)
        return

    from omegaconf import OmegaConf
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.environments.utils import register_env
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
    from nemo_rl.utils.logger import get_next_experiment_dir

    # Register custom CQL components
    from nemo_rl.data.processors import register_processor
    from utils.cql_data_processor import cql_data_processor
    register_processor("cql_data_processor", cql_data_processor)
    register_env("cql", "environments.cql_environment.CQLEnvironment")

    # Load config
    register_omegaconf_resolvers()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(config["policy"]["generation"], tokenizer)

    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(tokenizer, config["data"], config["env"])
    policy, policy_generation, cluster, dataloader, val_dataloader, loss_fn, logger, checkpointer, grpo_state, master_config = setup(config, tokenizer, dataset, val_dataset)

    if config["grpo"].get("async_grpo", {}).get("enabled"):
        from nemo_rl.algorithms.grpo import async_grpo_train
        async_grpo_train(
            policy=policy, policy_generation=policy_generation, dataloader=dataloader,
            val_dataloader=val_dataloader, tokenizer=tokenizer, loss_fn=loss_fn,
            task_to_env=task_to_env, val_task_to_env=val_task_to_env,
            logger=logger, checkpointer=checkpointer, grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=config["grpo"]["async_grpo"]["max_trajectory_age_steps"],
        )
    else:
        grpo_train(policy, policy_generation, dataloader, val_dataloader, tokenizer, loss_fn, task_to_env, val_task_to_env, logger, checkpointer, grpo_state, master_config)


if __name__ == "__main__":
    main()
