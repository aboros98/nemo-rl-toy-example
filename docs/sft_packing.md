# Sequence Packing for SFT — Complete Guide

You've done SFT with TRL. This doc covers packing in both NeMo RL and TRL so you can
compare directly, then pick the right setup for CQL fine-tuning.

---

## Why Pack?

Without packing, every sequence in a batch is padded to the longest sequence:

```
Without packing (50-70% wasted):        With packing (~5% wasted):
┌─────────────────────┐                 ┌─────────────────────┐
│ ████░░░░░░░░░░░░░░░ │                 │ ████▓▓▓▓▓▓▓██▓▓▓░░ │
│ ██████████████████░░ │                 │ ██████████████████░░ │
│ ██░░░░░░░░░░░░░░░░░ │                 │ ██████████████████░░ │
│ ██████░░░░░░░░░░░░░ │                 └─────────────────────┘
└─────────────────────┘                  3 sequences in 2 rows
 4 sequences, tons of padding            vs 4 sequences in 4 rows
```

**Impact:** 2-3× faster training, same convergence, same final quality.

CQL queries average ~50 tokens but range from 10-500. That's Zipf-distributed →
packing gives huge wins.

---

## NeMo RL SFT Packing

### Config (from actual `sft.yaml`)

```yaml
policy:
  model_name: "meta-llama/Llama-3.2-1B"
  train_global_batch_size: 32
  train_micro_batch_size: 1          # Always 1 with packing!
  max_total_sequence_length: 1024
  precision: "bfloat16"

  # ── Enable packing ──────────────────────────────────────
  sequence_packing:
    enabled: true
    train_mb_tokens: 1024            # max_seq_len × micro_batch = bin capacity
    algorithm: "modified_first_fit_decreasing"  # Best efficiency
    sequence_length_round: 64        # Pad to multiple of 64 (hardware alignment)

  # ── OR dynamic batching (mutually exclusive) ────────────
  # dynamic_batching:
  #   enabled: true
  #   train_mb_tokens: 1024
  #   sequence_length_round: 64
```

**Key rule:** `sequence_packing` and `dynamic_batching` are mutually exclusive. Pick one.

### Packing Algorithms (what NeMo RL provides)

| Algorithm | How It Works | When to Use |
|-----------|-------------|-------------|
| `modified_first_fit_decreasing` | Johnson & Garey 1985. 5-phase: classify by size → large bins → add medium → pair smalls → greedy remainder | **Default. Best efficiency.** |
| `first_fit_decreasing` | Sort desc by length, place each in first bin that fits | Good general-purpose |
| `first_fit_shuffle` | Random shuffle then first-fit | When order doesn't matter |
| `concatenative` | Sequential concat until full | Debugging only |

### How It Works Internally

NeMo RL does **not** pre-pack your dataset into `.npy` files. Instead:

1. `BatchedDataDict.shard_by_batch_size()` takes a batch + the packing config
2. It runs the packing algorithm to compute **bin assignments** (which sequences go together)
3. Reorders the batch and creates metadata (cumulative sequence lengths)
4. Workers call `make_microbatch_iterator_for_packable_sequences()` to get actual packed tensors
5. FlashAttention-2 uses `cu_seqlens_q`/`cu_seqlens_kv` to attend within each sequence only

**Cross-contamination?** Impossible — FlashAttention-2 uses cumulative sequence lengths, so
each sequence's attention is strictly isolated. No custom mask needed.

### Loss Function

NeMo RL wraps losses with `SequencePackingLossWrapper`:
- Takes packed logits → unpacks per sequence → runs loss on each → averages
- The loss function code doesn't need to know about packing at all

### With Context Parallelism (CP)

If you use CP > 1 (long sequences), each packed sequence is individually CP-sharded for
load balancing:

```
# CP=2, sequences [A=5tok, B=8tok, C=1tok, D=3tok]
# First pad each to multiple of 2*CP*TP = 4:
A: 5→8,  B: 8→8,  C: 1→4,  D: 3→4

# CP-shard each sequence, then pack per rank:
CP0: [A_chunk0, B_chunk0, C_chunk0, D_chunk0]  → pack
CP1: [A_chunk1, B_chunk1, C_chunk1, D_chunk1]  → pack
```

### With Pipeline Parallelism (PP)

Works, but: all packed sequences are padded to the max packed length (PP needs fixed
buffer sizes). In practice this barely matters because packing is already so efficient.

### Complete SFT + Packing Config for CQL

```yaml
sft:
  max_num_epochs: 3
  max_num_steps: 500
  val_period: 50
  val_batches: 8
  val_global_batch_size: 32
  val_micro_batch_size: 1
  seed: 42

checkpointing:
  enabled: true
  checkpoint_dir: "results/sft-cql"
  metric_name: "val:val_loss"
  higher_is_better: false
  keep_top_k: 3
  save_period: 50

policy:
  model_name: "nvidia/Nemotron-Mini-4B-Instruct"
  tokenizer:
    name: ${policy.model_name}
    chat_template: >
      {% for message in messages %}
      {%- if message['role'] == 'system' %}{{ message['content'].strip() }}
      {%- elif message['role'] == 'user' %}{{ '\nQuestion: ' + message['content'].strip() + '\nCQL:' }}
      {%- elif message['role'] == 'assistant' %}{{ ' ' + message['content'].strip() }}
      {%- endif %}
      {% endfor %}

  train_global_batch_size: 32
  train_micro_batch_size: 1        # Must be 1 with packing
  max_total_sequence_length: 512   # CQL queries are short
  precision: "bfloat16"

  sequence_packing:
    enabled: true
    train_mb_tokens: 512           # = max_total_sequence_length × micro_batch_size
    algorithm: "modified_first_fit_decreasing"
    sequence_length_round: 64

  max_grad_norm: 1.0

  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 2.0e-5                   # Higher than RL (SFT standard)
      weight_decay: 0.1
      betas: [0.9, 0.98]
      eps: 1e-5
      foreach: false
      fused: false

  dtensor_cfg:
    _v2: true
    enabled: true
    tensor_parallel_size: 1
    context_parallel_size: 1
    activation_checkpointing: false
    lora_cfg:
      enabled: true
      match_all_linear: true
      dim: 16
      alpha: 32
      dropout: 0.0

data:
  max_input_seq_length: 512
  add_bos: true
  add_eos: true
  shuffle: true
  train:
    data_path: "data/train.jsonl"
  validation:
    data_path: "data/val.jsonl"
  default:
    processor: "sft_processor"
    input_key: "nl_query"
    output_key: "cql_query"

cluster:
  gpus_per_node: 1
  num_nodes: 1
```

---

## TRL SFT Packing (for comparison)

Since you've used TRL before, here's the equivalent:

### Option 1: `packing=True` on SFTTrainer (simple)

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="./sft-cql",
    max_seq_length=512,
    packing=True,                    # ← This is it
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

**How it works:** TRL's `ConstantLengthDataset` concatenates tokenized examples with EOS
separators until `max_seq_length` is reached. Simple greedy bin-packing.

**⚠️ Warning:** This uses standard causal attention across the entire packed sequence.
Sequences CAN cross-attend to each other. For most SFT this is fine empirically, but
it's technically cross-contamination.

### Option 2: `padding_free=True` (proper isolation)

```python
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

collator = DataCollatorForCompletionOnlyLM(
    instruction_template="### Question:",
    response_template="### CQL:",
    tokenizer=tokenizer,
    padding_free=True,               # ← FlashAttention-2 variable-length
)

config = SFTConfig(
    output_dir="./sft-cql",
    max_seq_length=512,
    per_device_train_batch_size=4,
    # packing=False when using padding_free collator
    bf16=True,
    attn_implementation="flash_attention_2",  # Required
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    data_collator=collator,
)
```

**How it works:** Uses FlashAttention-2's variable-length API. Each sequence gets its own
attention scope — no cross-contamination. This is the same approach NeMo RL uses.

### TRL vs NeMo RL Comparison

| Feature | TRL `packing=True` | TRL `padding_free=True` | NeMo RL `sequence_packing` |
|---------|--------------------|-----------------------|---------------------------|
| Cross-contamination | ⚠️ Yes (cross-attend) | ✅ No (FA2 isolation) | ✅ No (FA2 isolation) |
| Packing algorithm | Greedy concat | Greedy concat | 4 algorithms (MFFD best) |
| Pre-processing | In DataLoader | In DataCollator | Online per batch |
| Attention kernel | Standard/FA2 | FA2 required | FA2 required |
| Pipeline parallel | N/A | N/A | ✅ Supported |
| Context parallel | N/A | N/A | ✅ Supported |
| Distributed | FSDP/DeepSpeed | FSDP/DeepSpeed | DTensor (FSDP2) / Megatron |
| Speedup | ~2× | ~2× | 2-3× |

---

## Dynamic Batching (the alternative)

If your model/attention doesn't support packing (rare), NeMo RL also offers dynamic batching:

```yaml
policy:
  dynamic_batching:
    enabled: true
    train_mb_tokens: 1024          # Target tokens per microbatch
    sequence_length_round: 64      # Pad to multiple of this
```

How it works:
1. Sort sequences by length within each chunk
2. Group into microbatches by total token count
3. Each microbatch is padded only to its own max length (not the global max)

Less efficient than packing (~1.5× vs 2-3×), but simpler and compatible with everything.

**Cannot be used with Megatron + Pipeline Parallelism** (PP needs fixed buffer sizes).

---

## Practical Tips

### 1. Always set `micro_batch_size: 1` with packing

Each "row" in the microbatch is already a packed sequence containing multiple examples.
Setting mbs > 1 with packing wastes memory.

### 2. Adjust `train_mb_tokens` for your GPU

```
train_mb_tokens = max_seq_length × 1   # Start here
# If OOM, reduce. If GPU underutilized, increase.
# Monitor with: nvidia-smi + packing efficiency metrics
```

### 3. Monitor packing efficiency

NeMo RL logs packing metrics:
- **Packing efficiency**: ratio of useful tokens to total tokens in packed batches
- **Sequences per pack**: average number of sequences packed together
- Target: >85% efficiency. If lower, try a longer `max_total_sequence_length`.

### 4. CQL-specific considerations

CQL queries are short (median ~50 tokens, max ~300). With `max_seq_length=512`:
- Expect 5-10 CQL examples packed per row
- Packing efficiency >90% easily
- This makes packing especially valuable — without it, you'd waste ~80% of compute

### 5. SFT before GRPO

The typical pipeline is:

```
Base Model → SFT (with packing) → GRPO (with packing in generation)
                ↓                        ↓
         Learn CQL syntax          Learn to optimize rewards
         + common patterns         + execution correctness
```

SFT first teaches the model basic CQL structure. GRPO then optimizes for correctness.
Both stages benefit from packing since CQL is short-form.

### 6. Packing + LoRA

Works perfectly with both DTensor LoRA and Megatron LoRA. No special config needed —
just enable both:

```yaml
policy:
  sequence_packing:
    enabled: true
    algorithm: "modified_first_fit_decreasing"
  dtensor_cfg:
    lora_cfg:
      enabled: true
      dim: 16
```

---

## Quick Reference

```
┌─────────────────────────────────────────────────────┐
│              Decision Tree: Packing                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Variable-length data?                               │
│    ├─ No → Don't bother                              │
│    └─ Yes                                            │
│         ├─ FlashAttention-2 available?               │
│         │    ├─ Yes → Use sequence_packing (MFFD)    │
│         │    └─ No  → Use dynamic_batching           │
│         └─ Using Pipeline Parallel?                  │
│              ├─ Yes → Use sequence_packing only      │
│              └─ No  → Either works, packing better   │
│                                                      │
│  Framework?                                          │
│    ├─ NeMo RL → sequence_packing.enabled: true       │
│    ├─ TRL     → padding_free=True (best)             │
│    │            or packing=True (simple)              │
│    └─ NeMo FW → prepare_packed_ft_dataset.py         │
│                                                      │
└─────────────────────────────────────────────────────┘
```
