# NeMo RL Model Setup: Chat Templates, Logging, and Correct Training

## 1. Data Format — System / Question / Answer Separation

Your JSONL has **two fields that matter** for NeMo RL:

```jsonl
{"nl_query": "Schema: DnsRequest (DomainName, ...)\nCount DNS requests by domain", "cql_query": "#event_simpleName=DnsRequest | groupBy(DomainName, function=count())", "source": "query_hub", ...}
```

| Role | Source | What NeMo RL Does |
|------|--------|-------------------|
| **System** | `resources/cql_system_prompt.txt` | Loaded via `system_prompt_file` config → injected as `{"role": "system"}` message by `cql_data_processor` |
| **User** (question) | `nl_query` field in JSONL | Read via `input_key: "nl_query"` → becomes `{"role": "user"}` |
| **Assistant** (answer) | `cql_query` field in JSONL | Read via `output_key: "cql_query"` → stored in `extra_env_info["ground_truth"]` for reward computation. **NOT sent to model** — the model generates its own response |

The system prompt lives in a file, not in the JSONL. Clean separation.

---

## 2. How the Chat Template Works

NeMo RL applies the model's chat template at the **processor level**, not at the data level. Here's the exact flow:

```
JSONL record
    ↓
ResponseDataset.format_data()
    → messages = [{"role": "user", "content": nl_query},
                  {"role": "assistant", "content": cql_query}]
    ↓
cql_data_processor()    ← our custom processor
    1. Loads system_prompt from system_prompt_file
    2. tokenizer.apply_chat_template([{"role": "system", ...}])
       → "<|start_header_id|>system<|end_header_id|>\n\nYou are a log query expert..."
    3. tokenizer.apply_chat_template([{"role": "user", ...}], add_generation_prompt=True)
       → "<|start_header_id|>user<|end_header_id|>\n\nSchema: ...\nCount DNS...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    4. cql_query → extra_env_info["ground_truth"]  (for reward, not for model)
    ↓
Model sees: [system tokens] [user tokens] [assistant header]
Model generates: the CQL query
    ↓
Reward server scores generation vs ground_truth
```

### Why this matters:
- The tokenizer's `chat_template` (Jinja2) adds the correct special tokens (`<|start_header_id|>`, `<|eot_id|>`, etc.)
- Different models have different templates — Nemotron vs Llama vs Qwen all differ
- NeMo RL handles this automatically when you set `tokenizer.name` in the config
- The `add_generation_prompt=True` flag adds the assistant header so the model knows to start generating

### To verify the chat template is correct:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
messages = [
    {"role": "system", "content": "You are a CQL expert."},
    {"role": "user", "content": "Count DNS requests by domain"},
]
print(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
```
This prints the exact token sequence your model will see during training.

---

## 3. Custom Data Processor — `cql_data_processor`

File: `utils/cql_data_processor.py`

NeMo RL's **default processor** (`math_hf_data_processor`) ignores `system_prompt_file`. Only `math_data_processor` handles system prompts, but it expects `problem`/`expected_answer` fields (not `messages`).

Our custom `cql_data_processor`:
- Properly creates a 3-role message log: system → user → (model generates assistant)
- Tokenizes each role with `apply_chat_template` (correct special tokens)
- Stores golden CQL in `extra_env_info` for the reward function
- Handles overflow (truncates and masks if prompt > `max_seq_length`)

### Registration (in your launch script):
```python
from utils.cql_data_processor import register_cql_processor
register_cql_processor()  # Registers "cql_data_processor" in NeMo RL's processor registry
```

### Config reference:
```yaml
data:
  default:
    processor: "cql_data_processor"
    system_prompt_file: "resources/cql_system_prompt.txt"
```

---

## 4. TensorBoard Logging — YES, NeMo RL Supports It

NeMo RL has a unified `Logger` that supports **TensorBoard, W&B, MLflow, and SwanLab** simultaneously.

### Config:
```yaml
logger:
  log_dir: "logs"
  tensorboard_enabled: true        # ← enables TensorBoard
  wandb_enabled: false             # or true for both
  mlflow_enabled: false
  swanlab_enabled: false
  monitor_gpus: true               # logs GPU memory/utilization
  tensorboard: {}                  # uses log_dir by default
  gpu_monitoring:
    collection_interval: 10        # seconds between GPU metric samples
    flush_interval: 10
```

### What NeMo RL logs automatically:

| Metric | Description |
|--------|-------------|
| `loss/policy_avg` | GRPO policy loss |
| `objective/kl` | KL divergence from reference policy |
| `objective/scores` | Raw reward scores from environment |
| `policy/approxkl_avg` | Approximate KL between consecutive policies |
| `policy/clipfrac_avg` | Fraction of updates where ratio clipping fired |
| `policy/entropy_avg` | Policy entropy (exploration signal) |
| `val/ratio` | Mean importance sampling ratio |
| `val:accuracy` | Validation accuracy (if configured) |
| `train/step_timing` | Wall-clock time per step |
| `gpu/memory_used` | GPU memory (if `monitor_gpus: true`) |
| `gpu/utilization` | GPU utilization % |

### To view:
```bash
tensorboard --logdir logs/
```

### Grad norms:
NeMo RL clips gradients via `max_grad_norm: 1.0` (configured in `policy`). The **gradient norm before clipping** is logged by NeMo RL's training loop — you'll see it in TensorBoard as `train/grad_norm`. No extra configuration needed.

---

## 5. Ensuring Correct Training

### Checklist:

1. **Chat template matches model**: Set `tokenizer.name` to the instruct variant (has chat template built in).
   - Nemotron-30B: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (instruct tokenizer)
   - This is different from the model name (`-Base-BF16`)

2. **System prompt is separate**: Via `system_prompt_file`, not baked into data.
   The custom processor injects it as `role: "system"` with proper tokenization.

3. **Stop strings prevent runaway**: `["\n\n", "```", "Explanation:", "Note:"]`
   — model stops after generating the query, doesn't add explanations.

4. **Golden answer → reward only**: The `output_key` (cql_query) goes to `extra_env_info`,
   not to the model. The model generates freely; reward scores against golden.

5. **LoRA excludes out_proj** (Nemotron-30B only):
   `exclude_modules: ['*out_proj*']` — Mamba2 out_proj has zero gradient.

6. **Token-level loss** (`token_level_loss: true`):
   Better for variable-length CQL queries vs sequence-level.

7. **Dynamic sampling** (`use_dynamic_sampling: true`):
   Filters zero-variance reward groups (all-correct or all-wrong prompts).

### What the training loop looks like at runtime:

```
For each step:
  1. Sample prompts from data/train.jsonl
  2. Format: system_prompt + nl_query → tokenize with chat template
  3. vLLM generates N completions per prompt (num_generations_per_prompt)
  4. Each completion → reward server → {syntax, execution, ngram} → combined reward
  5. GRPO advantage = reward - mean(rewards for this prompt)
  6. Policy gradient update with clipped ratio (Clip-Higher: 0.2/0.28)
  7. Log metrics to TensorBoard / W&B
```

---

## 6. Logger Architecture (from NeMo RL source)

```
nemo_rl.utils.logger.Logger
├── TensorboardLogger (torch.utils.tensorboard.SummaryWriter)
├── WandbLogger (wandb.init)
├── MLflowLogger (mlflow)
└── SwanLabLogger (swanlab)

All loggers share the same interface:
  .log_metrics(metrics_dict, step)
  .log_histogram(values, step, name)
  .log_hyperparams(params)
  .log_plot(figure, step, name)
```

You can enable **multiple loggers simultaneously** — TB + W&B is common for production.

---

## Summary

| Concern | How it's handled |
|---------|-----------------|
| System prompt | `system_prompt_file` → separate `role: "system"` in chat template |
| User question | `input_key: "nl_query"` → `role: "user"` |
| Golden answer | `output_key: "cql_query"` → `extra_env_info` (reward only) |
| Chat template | Applied by tokenizer automatically; processor calls `apply_chat_template()` per role |
| TensorBoard | `tensorboard_enabled: true` in logger config; metrics logged every step |
| Grad norms | Logged automatically by NeMo RL as `train/grad_norm` |
| W&B | `wandb_enabled: true` (can run alongside TB) |
| GPU monitoring | `monitor_gpus: true` → memory + utilization logged |
