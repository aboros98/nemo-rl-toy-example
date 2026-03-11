# Hyperparameters Reference — CQL RLVR

All tunables for GRPO and SFT training with NeMo RL. Organized by category.

---

## 1. GRPO Algorithm

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| Prompts per step | `grpo.num_prompts_per_step` | 64 | Total prompts sampled each training step |
| Generations per prompt | `grpo.num_generations_per_prompt` | 4 | Rollouts per prompt (GRPO group size) |
| Max training steps | `grpo.max_num_steps` | 200 | Total training iterations |
| KL penalty coefficient | `grpo.kl_coeff` | 0.0 | 0 = no KL penalty (pure GRPO). >0 adds reference policy KL |
| Dynamic sampling | `grpo.dynamic_sampling.enabled` | true | Skip zero-variance reward groups (all correct or all wrong) |
| Token-level loss | `grpo.token_level_loss.enabled` | true | Per-token advantage weighting vs per-sequence |
| PPO clip ratio | `grpo.ratio_clip_min` | 0.8 | Lower clip bound (= 1 - ε, typically ε=0.2) |
| PPO clip ratio high | `grpo.ratio_clip_max` | 1.28 | Upper clip bound (Clip-Higher variant, >1+ε). Set to 1.2 for standard PPO |
| Entropy bonus | `grpo.entropy_bonus_coeff` | 0.0 | Encourage exploration. Usually 0 for GRPO |

**What to tune first**: `num_prompts_per_step` × `num_generations_per_prompt` = total generations per step. Start with 64×4=256. If rewards are noisy, increase group size to 8 or 16.

---

## 2. Loss & Optimization

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| Learning rate | `policy.optimizer.kwargs.lr` | 1e-6 (GRPO), 2e-5 (SFT) | LoRA can use 2-5× higher LR |
| Weight decay | `policy.optimizer.kwargs.weight_decay` | 0.01 | Standard AdamW |
| Adam β1, β2 | `policy.optimizer.kwargs.betas` | [0.9, 0.999] | |
| Gradient clip | `policy.max_grad_norm` | 1.0 | Global norm clipping |
| Warmup steps | `policy.lr_scheduler.warmup_steps` | 10 | Linear warmup |
| Scheduler type | `policy.lr_scheduler.type` | "cosine" | Options: "cosine", "linear", "constant", "constant_with_warmup" |
| Min LR ratio | `policy.lr_scheduler.min_lr_ratio` | 0.1 | Minimum LR as fraction of peak (cosine only) |
| Train batch size | `policy.train_global_batch_size` | 256 (GRPO), 32 (SFT) | GRPO: should equal num_prompts × num_generations |
| Micro batch size | `policy.train_micro_batch_size` | 4 | Per-GPU batch size for gradient accumulation |

**Key relationship**: `train_global_batch_size / train_micro_batch_size / num_gpus = gradient_accumulation_steps`

**Scheduler**: NeMo RL supports: `cosine` (recommended for GRPO), `linear`, `constant`, `constant_with_warmup`. Change via `policy.lr_scheduler.type`.

---

## 3. LoRA Configuration

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| Enable LoRA | `policy.dtensor_cfg.lora_cfg.enabled` | true | false = full fine-tuning |
| Rank | `policy.dtensor_cfg.lora_cfg.dim` | 256 | Higher = more capacity. 64-256 typical |
| Alpha | `policy.dtensor_cfg.lora_cfg.alpha` | 512 | Effective scale = alpha/rank. Keep alpha=2×rank |
| Dropout | `policy.dtensor_cfg.lora_cfg.dropout` | 0.05 | |
| DoRA | `policy.dtensor_cfg.lora_cfg.use_dora` | true | Decomposed weight adaptation. Better than vanilla LoRA |
| Exclude modules | `policy.dtensor_cfg.lora_cfg.exclude_modules` | ['*out_proj*'] | **MANDATORY for Mamba2** — SSM out_proj must be excluded |
| Match all linear | `policy.dtensor_cfg.lora_cfg.match_all_linear` | false | true = apply LoRA to all linear layers |
| Use Triton | `policy.dtensor_cfg.lora_cfg.use_triton` | false | **Must be false for Mamba2** |

**LoRA vs Full-FT**: LoRA rank=256 uses ~2% of parameters. Full-FT needs more memory but can achieve higher quality. Start with LoRA, try full-FT if LoRA plateaus.

**Mamba2 constraints** (Nemotron-3-Nano-30B-A3B):
- `exclude_modules: ['*out_proj*']` — MANDATORY
- `match_all_linear: false`
- `use_triton: false`
- `sequence_packing.enabled: false`

---

## 4. Generation / vLLM

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| Temperature | `generation.temperature` | 0.7 | Higher = more diverse rollouts. 0.6-1.0 typical for GRPO |
| Top-p | `generation.top_p` | 0.95 | Nucleus sampling threshold |
| Max new tokens | `generation.max_new_tokens` | 512 | Max response length. CQL queries rarely exceed 256 |
| Stop strings | `generation.stop_strings` | ["\n\n", "```"] | Stop generation early on these tokens |
| vLLM tensor parallel | `generation.vllm_cfg.tensor_parallel_size` | 4 | GPUs for inference. Must divide total GPUs |
| GPU memory util | `generation.vllm_cfg.gpu_memory_utilization` | 0.5 | Fraction of GPU memory for vLLM KV cache |
| Mamba SSM cache dtype | `generation.vllm_cfg.mamba_ssm_cache_dtype` | "float32" | **Required for Mamba2** |

**Temperature tuning**: Low temp (0.3-0.5) = less exploration, faster convergence but may get stuck. High temp (0.8-1.2) = more diverse, better for GRPO reward signal but slower convergence.

---

## 5. Cluster & Parallelism

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| Num GPUs | `cluster.num_gpus` | 8 | Total GPUs across all nodes |
| GPUs per node | `cluster.gpus_per_node` | 8 | Must match SLURM `--gpus-per-node` |
| Num nodes | `cluster.num_nodes` | 1 | For multi-node training |
| FSDP sharding | `policy.dtensor_cfg.tensor_parallel_size` | 1 | Usually 1 (FSDP handles sharding) |
| Activation checkpointing | `policy.dtensor_cfg.activation_checkpointing` | true | Saves memory at cost of ~30% speed |
| CPU offload | `policy.dtensor_cfg.cpu_offload` | false | Offload optimizer states to CPU. Saves GPU memory |
| Compilation | `policy.compilation_config.use_inductor` | false | **Must be false for Mamba2**. true for transformer models |

**Memory budget (8×B200, 192GB each)**:
- Nemotron-30B in bf16: ~60GB model weights
- LoRA: model weights + ~2GB adapters + ~30GB optimizer states
- Full-FT: model weights + ~180GB optimizer states (needs FSDP + CPU offload)
- vLLM inference: separate from training, uses `gpu_memory_utilization` fraction

---

## 6. Environment (GRPO only)

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| Env workers | `env.cql.num_workers` | 8 | Ray actors for reward computation |

The CQL environment returns: 1.0 for syntactically valid CQL, 0.0 for invalid. Future: add ngram similarity, mock execution scoring.

---

## 7. SFT-Specific

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| Max input seq length | `data.max_input_seq_length` | 4096 | Truncate inputs longer than this |
| Sequence packing | `data.default.sequence_packing.enabled` | false | **Must be false for Mamba2** |
| Add BOS | `data.add_bos` | false | Tokenizer-specific. Usually false for modern models |
| Add EOS | `data.add_eos` | true | Add EOS token to targets |
| Add generation prompt | `data.add_generation_prompt` | false | For SFT, usually false (we have the full conversation) |
| Validation interval | `sft.val_every_num_steps` | 50 | Run validation every N steps |

---

## 8. Logging & Checkpointing

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| Log dir | `logger.log_dir` | "logs/cql_grpo" | TensorBoard logs saved here |
| Log interval | `logger.log_every_n_steps` | 1 | Log metrics every N steps |
| Checkpoint enabled | `checkpointing.enabled` | true | Save model checkpoints |
| Checkpoint dir | `checkpointing.checkpoint_dir` | "checkpoints/cql_grpo" | Where to save checkpoints |
| Save interval | `checkpointing.save_every_n_steps` | 50 | Save checkpoint every N steps |
| Keep N checkpoints | `checkpointing.num_to_keep` | 3 | Rolling window of recent checkpoints |

**TensorBoard**: NeMo RL logs to TensorBoard by default. View with:
```bash
tensorboard --logdir logs/ --port 6006
```

Key metrics to watch:
- `reward/mean` — average reward (should increase)
- `reward/std` — reward variance (GRPO needs non-zero variance)
- `policy/loss` — policy gradient loss
- `policy/ratio_mean` — importance sampling ratio (should stay near 1.0)
- `policy/kl_mean` — KL divergence from reference (if kl_coeff > 0)
- `generation/mean_response_length` — response length (watch for length hacking)

---

## Quick-Start Recipes

### Conservative (safe first run)
```bash
sbatch scripts/slurm/grpo.sh  # uses default config as-is
```

### Aggressive exploration
```bash
OVERRIDES="++generation.temperature=1.0 ++grpo.num_generations_per_prompt=8 ++grpo.ratio_clip_max=1.5" \
  sbatch scripts/slurm/grpo.sh
```

### Full fine-tuning
```bash
CONFIG=configs/cql_nemo_rl_nemotron30b_full.yaml sbatch scripts/slurm/grpo.sh
```

### Higher learning rate with LoRA
```bash
OVERRIDES="++policy.optimizer.kwargs.lr=5e-6" sbatch scripts/slurm/grpo.sh
```
