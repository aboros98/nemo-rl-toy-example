# CQL RLVR — Rewards & Training Strategy

Technical deep-dive into the reward design and GRPO training strategy for NL-to-CQL generation.

## Why GRPO, Not DPO

DPO needs preference pairs (chosen vs rejected). For code generation you'd need to pre-label which CQL query is "better" — expensive and noisy when both queries are valid but structurally different.

GRPO sidesteps this: generate N completions per prompt, score them all with a verifiable reward function, use the group's mean as the baseline. The advantage is `reward_i - mean(rewards)`. No reference model forward pass needed (KL=0 in our config), no preference data curation. This is exactly the GSM8K pattern you've done — swap the math verifier for a CQL verifier.

The key difference from GSM8K: math has a binary reward (correct/incorrect answer). CQL has a **continuous** reward — a query can be partially right (valid syntax, wrong structure, close ngram match). This richer signal is why we decompose the reward into independent components rather than using a single binary check.

## Reward Architecture

```
reward = 0.4 × syntax + 0.3 × execution + 0.3 × ngram
         ────────────    ───────────────    ────────────
         binary 0/1      binary 0/1         continuous [0,1]
```

### Why These Three Components

| Component | What it measures | Signal type | Why it matters |
|-----------|-----------------|-------------|----------------|
| **syntax** | Does the CQL parse? (balanced delimiters, known functions, valid pipe structure) | Binary gate | Without valid syntax, nothing else matters. This is the "format reward" equivalent from math RLVR. |
| **execution** | Does the query run against LogScale without errors? | Binary gate | A query can be syntactically valid but reference nonexistent fields or use wrong types. Currently **mocked at 80% random** — in production, hit a real LogScale sandbox. |
| **ngram** | How similar is the generated query to the gold reference? | Continuous gradient | This is the "soft" signal. Even if the query is wrong, being structurally close to the answer should be rewarded. Uses bigram Dice coefficient on semantic CQL tokens (functions, tags, operators are atomic). |

### Why These Weights (0.4 / 0.3 / 0.3)

The weights serve a dual purpose:

1. **Signal priority**: syntax > execution ≈ ngram. Getting the syntax right is the first thing the model should learn.

2. **Invariant enforcement by construction**: `syntax_weight (0.4) > ngram_weight (0.3)` guarantees that the worst valid query (score = 0.4) always beats the best invalid query (score = 0.3). No clamps, no ceilings, no floor — the math just works.

### Reward Landscape

```
                    syntax  execution  ngram   → reward
─────────────────────────────────────────────────────────
Invalid, 0% match    0.0      0.0      0.0    → 0.00  ← worst possible
Invalid, 100% match  0.0      0.0      1.0    → 0.30  ← best invalid
Valid, exec fail, 0% 1.0      0.0      0.0    → 0.40  ← worst valid
Valid, exec fail, ↑  1.0      0.0      0.5    → 0.55
Valid, exec pass, 0% 1.0      1.0      0.0    → 0.70
Valid, exec pass, ↑  1.0      1.0      0.5    → 0.85
Valid, exec pass, ✓  1.0      1.0      1.0    → 1.00  ← perfect
─────────────────────────────────────────────────────────
                                          gap: 0.10
                                    (0.40 - 0.30 = invariant margin)
```

Every row of this table has a distinct reward. The model always has gradient signal to improve, whether it's learning syntax, getting execution right, or matching the reference more closely.

### Comparison with GSM8K GRPO

| Aspect | GSM8K GRPO | CQL GRPO |
|--------|-----------|----------|
| Reward | Binary: answer matches or not | Continuous: weighted 3-component sum |
| Verifier | String match on final number | Syntax validator + executor + ngram |
| Variance per group | High (all-or-nothing) | Lower (continuous gradient) |
| Dynamic sampling | Filters groups where all gens are 0 or all 1 | Filters zero-variance groups (rare since reward is continuous) |
| KL penalty | Usually 0.01-0.05 | 0.0 for dummy, 0.01-0.05 for production |

The continuous reward is both a blessing and a curse:
- **Blessing**: more gradient signal, faster early learning, model always has something to chase
- **Curse**: GRPO's advantage normalization within groups can compress differences when all generations score similarly (e.g. all valid, all ~0.6). Dynamic sampling helps here.

## GRPO Hyperparameters Explained

```yaml
ratio_clip_min: 0.2    # Standard PPO-style clipping (ε = 0.2)
ratio_clip_max: 0.28   # "Clip-Higher" — asymmetric clip for positive advantages
                        # Lets good actions update slightly more aggressively
token_level_loss: true  # Loss computed per-token, not per-sequence
                        # Critical for code: teaches which tokens matter
KL penalty: 0.0         # No KL for dummy. In production: 0.01-0.05
                        # Start at 0, add if model degenerates
```

**Clip-Higher (0.28 vs 0.2):** Standard GRPO clips the importance ratio symmetrically at 1±ε. Clip-Higher uses a wider upper clip (1+0.28) than lower clip (1-0.2), allowing the model to learn more aggressively from high-advantage tokens. Empirically helps on code tasks where correct tokens are sparse.

**Token-level loss:** Instead of one reward per sequence, the reward signal is distributed across all generated tokens. This is important for CQL because a single wrong token (missing `)`, wrong function name) can flip the syntax reward from 1→0. Token-level loss helps the model learn which specific tokens caused the reward drop.

## Adding New Reward Components

The system is designed to be pluggable. To add a new reward:

```python
# 1. Write a function returning [0, 1]
def reward_schema_accuracy(generated_cql: str, schema: dict) -> float:
    """Does the query reference valid fields for the event type?"""
    # ... your logic ...
    return score  # [0, 1]

# 2. Add weight (must sum to 1 with others, or renormalize)
DEFAULT_REWARD_WEIGHTS = {
    "syntax": 0.3,
    "execution": 0.2,
    "ngram": 0.2,
    "schema_accuracy": 0.3,  # new
}

# 3. Add to compute_reward()
scores["schema_accuracy"] = reward_schema_accuracy(generated_cql, schema)
```

**Candidate rewards for production:**

| Reward | Type | What it measures | When to add |
|--------|------|------------------|-------------|
| `schema_accuracy` | continuous | Do field names match the event type schema? | When you have real schema definitions |
| `execution_result_quality` | continuous | Does the query return reasonable results? (not empty, not too many) | When LogScale sandbox is live |
| `structural_similarity` | continuous | Jaccard of function pipeline (already implemented in tokenizer) | If ngram alone isn't enough |
| `format_compliance` | binary | No markdown, no explanation, just CQL | If model keeps adding prose around the query |

## Training Phases (Recommended Strategy)

### Phase 1: Dummy Pipeline (this repo)
- Mock execution, mock generation
- Validates: data pipeline, reward server, config, metrics logging
- **You are here**

### Phase 2: SFT Warmup
- Standard SFT on the (nl_query, cql_query) pairs
- Gets the model to output CQL-shaped text
- Use the same data from `data/train.jsonl`
- 1-3 epochs, standard cross-entropy loss
- **Don't skip this** — GRPO works much better when the model already produces valid CQL most of the time

### Phase 3: GRPO with Mock Execution
- Real model (Nemotron-Mini-4B-Instruct + LoRA)
- Real generation via vLLM
- Mock execution (80% random)
- Purpose: validate the RL loop with real model outputs
- 100-500 steps, watch syntax_valid_pct climb

### Phase 4: GRPO with Real Execution
- Swap mock execution for real LogScale sandbox
- This is where the execution reward becomes real signal
- Add KL penalty (0.01) to prevent reward hacking
- 1000+ steps
- Watch for: model finding "exploits" (valid queries that score high but are meaningless)

### Phase 5: Scale
- Bigger model or full fine-tune (drop LoRA)
- More rollouts (8-16 per prompt)
- Multi-node (see `docs/slurm_multinode.md`)
- Add schema_accuracy reward component
- Add execution_result_quality reward

## What Can Go Wrong

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Reward collapses to 0.3 | Model outputs only invalid CQL | SFT warmup first; lower learning rate |
| Reward stuck at ~0.4 | All valid but exec always fails | Check mock rate; lower execution weight |
| Reward at 1.0 everywhere | Reward function is broken / model memorized | Check for data leakage; add KL penalty |
| High variance across steps | Too few rollouts per prompt | Increase `num_generations_per_prompt` to 8-16 |
| NaN rewards | Numerical issue in reward or in GRPO loss | Check for empty queries; add epsilon to divisions |
| Model outputs prose | Prompt template not strong enough | Add format_compliance reward; stricter prompt |

## File Reference

| File | What it does |
|------|-------------|
| `resources/cql_resource_server.py` | Reward functions + FastAPI server |
| `utils/cql_tokenizer.py` | Semantic CQL tokenizer (used by ngram reward) |
| `utils/cql_validator.py` | Syntax validator (used by syntax reward) |
| `configs/cql_nemo_rl_config.yaml` | Full NeMo RL GRPO config (ready for real NeMo RL) |
| `configs/cql_gym_config.yaml` | NeMo Gym environment config |
| `scripts/train_grpo.py` | Training script (dummy loop for now) |
