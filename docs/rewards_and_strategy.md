# CQL RLVR — Rewards & Training Strategy

Technical deep-dive into the reward design and GRPO training strategy for NL-to-CQL generation.

## Why GRPO, Not DPO

DPO needs preference pairs (chosen vs rejected). For code generation you'd need to pre-label which CQL query is "better" — expensive and noisy when both queries are valid but structurally different.

GRPO sidesteps this: generate N completions per prompt, score them all with a verifiable reward function, use the group's mean as the baseline. The advantage is `reward_i - mean(rewards)`. No reference model forward pass needed (KL=0 in our config), no preference data curation. This is exactly the GSM8K pattern you've done — swap the math verifier for a CQL verifier.

The key difference from GSM8K: math has a binary reward (correct/incorrect answer). CQL has a **continuous** reward — a query can be partially right (close ngram match, proper format). This richer signal is why we decompose the reward into independent components rather than using a single binary check.

## Reward Architecture

```
reward = 0.2 × format + 0.8 × ngram + 0.0 × execution
         ────────────    ────────────    ────────────────
         discrete 0/0.5/1  continuous [0,1]  placeholder (always 0)
```

Source of truth: `utils/cql_rewards.py` (pure Python, no GPU deps — testable on Mac).

### Why These Three Components

| Component | What it measures | Signal type | Why it matters |
|-----------|-----------------|-------------|----------------|
| **format** | Does the model use `<think>...</think>` tags? | Discrete (0 / 0.5 / 1.0) | R1-style format reward. Encourages chain-of-thought reasoning before CQL output. |
| **ngram** | How similar is the generated CQL to the gold reference? | Continuous gradient | Bigram Dice coefficient on semantic CQL tokens. Even slightly wrong queries get partial credit. |
| **execution** | Does the query compile in LogScale? | Placeholder (0.0) | Not yet implemented. Set weight=0 until Docker LogScale sandbox is ready. |

### Format Scoring

| Condition | Score |
|-----------|-------|
| No `<think>` or `</think>` tags | 0.0 |
| Has one tag but not both | 0.5 |
| Has both `<think>` and `</think>` | 1.0 |

### Why These Weights (0.2 / 0.8 / 0.0)

- **N-gram dominates (0.8)**: accuracy is the primary signal. The model must produce correct CQL.
- **Format is secondary (0.2)**: worth +0.20 max. Enough to reward `<think>` tags but won't overpower accuracy.
- **Execution disabled (0.0)**: placeholder until Docker LogScale sandbox is ready.
- When execution is added, recommended rebalance: format=0.1, ngram=0.5, execution=0.4.

### Reward Landscape

```
                        format  ngram   → reward
─────────────────────────────────────────────────────
No tags, wrong CQL       0.0    0.0    → 0.000  ← worst
No tags, perfect CQL     0.0    1.0    → 0.800  ← correct but no reasoning
One tag, partial CQL     0.5    0.3    → 0.340
Both tags, wrong CQL     1.0    0.0    → 0.200  ← format only
Both tags, partial CQL   1.0    0.5    → 0.600
Both tags, perfect CQL   1.0    1.0    → 1.000  ← best possible
─────────────────────────────────────────────────────
```

Key insight: the model can get 0.800 without think tags (bare perfect CQL) but needs tags to reach 1.0. This creates a smooth incentive to learn format AFTER accuracy.

### How CQL is Extracted

`extract_cql_from_response()` in `utils/cql_rewards.py`:
1. If `<think>...</think>` tags found: extract text AFTER `</think>` as CQL, text inside as thinking
2. If no tags: the entire response is treated as CQL
3. If tags found but nothing after `</think>`: CQL is empty string (ngram=0)

This means the ngram reward only scores the CQL portion, not the thinking. The model cannot game ngram by stuffing reference tokens inside `<think>` tags.

### Comparison with GSM8K GRPO

| Aspect | GSM8K GRPO | CQL GRPO |
|--------|-----------|----------|
| Reward | Binary: answer matches or not | Continuous: weighted 2-component sum |
| Verifier | String match on final number | Bigram similarity + format check |
| Variance per group | High (all-or-nothing) | Moderate (continuous gradient) |
| Dynamic sampling | Filters groups where all gens are 0 or all 1 | Filters zero-variance groups (rare since reward is continuous) |
| Format reward | Optional (boxed answer) | 0.2 weight (think tags) |

The continuous reward is both a blessing and a curse:
- **Blessing**: more gradient signal, faster early learning, model always has something to chase
- **Curse**: GRPO's advantage normalization within groups can compress differences when all generations score similarly (e.g. all ~0.6). Dynamic sampling helps here.

## GRPO Hyperparameters Explained

```yaml
ratio_clip_min: 0.2    # Standard PPO-style clipping (ε = 0.2)
ratio_clip_max: 0.28   # "Clip-Higher" — asymmetric clip for positive advantages
                        # Lets good actions update slightly more aggressively
token_level_loss: true  # Loss computed per-token, not per-sequence
                        # Critical for code: teaches which tokens matter
KL penalty: 0.0         # No KL for now. In production: 0.01-0.05
                        # Start at 0, add if model degenerates
```

**Clip-Higher (0.28 vs 0.2):** Standard GRPO clips the importance ratio symmetrically at 1±ε. Clip-Higher uses a wider upper clip (1+0.28) than lower clip (1-0.2), allowing the model to learn more aggressively from high-advantage tokens. Empirically helps on code tasks where correct tokens are sparse.

**Token-level loss:** Instead of one reward per sequence, the reward signal is distributed across all generated tokens. This is important for CQL because a single wrong token (missing `)`, wrong function name) can flip the ngram reward significantly. Token-level loss helps the model learn which specific tokens caused the reward change.

## Adding New Reward Components

The system is designed to be pluggable. To add a new reward:

```python
# In utils/cql_rewards.py:

# 1. Write a function returning [0, 1]
def compute_execution_reward(cql: str) -> float:
    """Does the query compile in LogScale?"""
    resp = requests.post(LOGSCALE_URL, json={"query": cql}, timeout=10)
    return 1.0 if resp.status_code == 200 else 0.0

# 2. Add weight in configs/cql_nemo_rl_nemotron30b.yaml
#    reward_weights: {format: 0.1, ngram: 0.5, execution: 0.4}

# 3. Update compute_combined_reward() to include the new component
```

**Candidate rewards for production:**

| Reward | Type | What it measures | When to add |
|--------|------|------------------|-------------|
| `execution` | binary | Does the CQL compile in LogScale? | When Docker sandbox is ready |
| `schema_accuracy` | continuous | Do field names match the event type schema? | When you have real schema definitions |
| `structural_similarity` | continuous | Jaccard of function pipeline (already in tokenizer) | If ngram alone isn't enough |
| `result_quality` | continuous | Does the query return reasonable results? | When LogScale sandbox is live |

## Training Phases (Recommended Strategy)

### Phase 1: SFT Warmup
- Standard SFT on the (nl_query, cql_query) pairs
- Gets the model to output CQL-shaped text
- Use the same data from `data/train.jsonl`
- 1-3 epochs, standard cross-entropy loss
- **Don't skip this** — GRPO works much better when the model already produces valid CQL most of the time

### Phase 2: GRPO with N-gram + Format Rewards
- Real model (Nemotron-30B + LoRA) with SFT checkpoint
- Real generation via vLLM
- Format (0.2) + ngram (0.8) rewards
- 100-500 steps, watch mean_reward climb
- Tune: if model ignores `<think>` tags, increase format weight to 0.3

### Phase 3: GRPO with Execution Reward
- Add Docker LogScale sandbox for compilation checking
- Rebalance: format=0.1, ngram=0.5, execution=0.4
- Add KL penalty (0.01) to prevent reward hacking
- 1000+ steps
- Watch for: model finding "exploits" (valid queries that compile but are meaningless)

### Phase 4: Scale
- Full fine-tune (drop LoRA) if GPU memory allows
- More rollouts (8-16 per prompt)
- Add schema_accuracy reward component
- Add result_quality reward

## What Can Go Wrong

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Mean reward stuck at ~0.2 | Model only learned format, CQL is garbage | SFT warmup first; increase ngram weight |
| Mean reward at 0.8 everywhere | All rollouts produce similar CQL (no variance) | Enable dynamic sampling; increase rollouts per prompt |
| Reward at 1.0 everywhere | Reward function bug or data leakage | Check ngram scores individually; verify extraction |
| High variance across steps | Too few rollouts per prompt | Increase `num_generations_per_prompt` to 8-16 |
| NaN rewards | Numerical issue in tokenizer or division | Check for empty queries; add epsilon to divisions |
| Model outputs prose around CQL | Prompt template not strong enough | Add a format penalty for text after CQL; stricter prompt |
| Think tags but empty reasoning | Format reward is too easy (tags only) | Add minimum length check to format reward |

## File Reference

| File | What it does |
|------|-------------|
| `utils/cql_rewards.py` | **Reward logic** — format + ngram + execution (pure Python) |
| `environments/cql_environment.py` | NeMo RL Ray actor wrapper (imports from cql_rewards.py) |
| `scripts/test_rewards_local.py` | Local reward testing (imports from cql_rewards.py) |
| `utils/cql_tokenizer.py` | Semantic CQL tokenizer (used by ngram reward) |
| `resources/cql_system_prompt.txt` | System prompt with `<think>` tag instructions |
| `configs/cql_nemo_rl_nemotron30b.yaml` | GRPO config with reward_weights section |
| `scripts/run_grpo_cql.py` | GRPO training script |
