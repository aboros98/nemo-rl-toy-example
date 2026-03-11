# GRPO vs GSPO vs DAPO vs Dr. GRPO — Algorithm Deep-Dive

A practical comparison of the four GRPO-family algorithms with math, pseudocode,
and NeMo RL config for each. Written for someone who's trained GRPO on GSM8K and
wants to understand what the variants actually change under the hood.

---

## Table of Contents

1. [The Shared Foundation](#the-shared-foundation)
2. [GRPO — Vanilla](#grpo--vanilla)
3. [Dr. GRPO — Length-Debiased](#dr-grpo--length-debiased)
4. [GSPO — Sequence-Level](#gspo--sequence-level)
5. [DAPO — The Kitchen Sink](#dapo--the-kitchen-sink)
6. [Side-by-Side Comparison](#side-by-side-comparison)
7. [Pseudocode for All Four](#pseudocode-for-all-four)
8. [NeMo RL Configs](#nemo-rl-configs)
9. [Which One Should I Use?](#which-one-should-i-use)
10. [References](#references)

---

## The Shared Foundation

All four algorithms share this structure:

```
For each training step:
  1. Sample B prompts from the dataset
  2. For each prompt, generate K completions from the current policy π_θ
  3. Score each completion with a reward function → r_1, ..., r_K
  4. Compute advantages within each group (how good is this one vs the rest?)
  5. Compute importance sampling ratio: how much did the policy change?
  6. Compute clipped surrogate loss (PPO-style)
  7. Backprop, update weights
```

The differences are in **steps 4-6**: how advantages are computed, at what granularity
importance sampling happens, and how the loss is aggregated.

---

## GRPO — Vanilla

**Paper**: DeepSeek-R1 (2024)
**Key idea**: PPO without a critic. Use the group of K completions as a baseline.

### Math

**Advantage** (group-relative, normalized):
```
μ = mean(r_1, ..., r_K)
σ = std(r_1, ..., r_K) + ε
A_i = (r_i - μ) / σ                    # Same advantage for ALL tokens in response i
```

**Importance sampling** (token-level):
```
ρ_t = π_θ(token_t | context) / π_θ_old(token_t | context)
```

**Loss** (per-response, then averaged):
```
L_i = (1/|o_i|) * Σ_t min(ρ_t * A_i, clip(ρ_t, 1-ε, 1+ε) * A_i)
L = (1/G) * Σ_i L_i + β * KL(π_θ ‖ π_ref)
```

### The Problem

The `1/|o_i|` normalization creates **length bias**: a long wrong answer gets divided
by many tokens (small gradient), while a short wrong answer gets divided by few tokens
(large gradient). This inadvertently encourages verbose outputs.

Also: dividing by `σ` introduces **difficulty bias** — easy prompts (all correct, σ≈0)
get enormous advantages, hard prompts get suppressed.

### Pseudocode

```python
def grpo_loss(prompts, completions, rewards, old_logprobs, new_logprobs, ref_logprobs):
    losses = []
    for prompt_idx in range(len(prompts)):
        group_rewards = rewards[prompt_idx]  # shape: [K]
        mu = group_rewards.mean()
        sigma = group_rewards.std() + 1e-8
        advantages = (group_rewards - mu) / sigma  # shape: [K]

        for k in range(K):
            # Token-level importance sampling ratio
            ratio = torch.exp(new_logprobs[prompt_idx][k] - old_logprobs[prompt_idx][k])
            # shape: [seq_len]

            A = advantages[k]  # scalar, same for all tokens
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            surrogate = torch.min(ratio * A, clipped * A)

            # Per-token loss, averaged over response length
            loss_k = -surrogate.mean()  # ← the 1/|o_i| normalization
            losses.append(loss_k)

    # KL penalty
    kl = (old_logprobs - ref_logprobs).mean()
    return torch.stack(losses).mean() + beta * kl
```

---

## Dr. GRPO — Length-Debiased

**Paper**: "Understanding R1-Zero-Like Training" (Sea AI Lab, 2025, arXiv:2503.20783)
**Key idea**: Remove the two biases from vanilla GRPO.

### What Changes

Two simple changes:

1. **Remove length normalization**: Don't divide by `|o_i|`. Sum over tokens, then
   average over the batch globally.
2. **Remove std normalization**: Don't divide advantages by `σ`. Just center them.

### Math

**Advantage** (centered, NOT normalized by std):
```
μ = mean(r_1, ..., r_K)
A_i = r_i - μ                          # No division by σ
```

**Loss** (token-sum, NOT token-mean):
```
L_i = Σ_t min(ρ_t * A_i, clip(ρ_t, 1-ε, 1+ε) * A_i)    # NO 1/|o_i|
L = (1/G) * Σ_i L_i
```

### Why It Works

- Without `1/|o_i|`: a 100-token correct answer and a 10-token correct answer
  contribute gradient proportional to their length — the optimizer treats each
  token equally regardless of which response it belongs to.
- Without `/σ`: easy and hard prompts contribute proportionally to their reward
  difference, not amplified by small σ.

### Pseudocode

```python
def dr_grpo_loss(prompts, completions, rewards, old_logprobs, new_logprobs):
    losses = []
    for prompt_idx in range(len(prompts)):
        group_rewards = rewards[prompt_idx]
        mu = group_rewards.mean()
        advantages = group_rewards - mu  # ← NO /σ

        for k in range(K):
            ratio = torch.exp(new_logprobs[prompt_idx][k] - old_logprobs[prompt_idx][k])
            A = advantages[k]
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            surrogate = torch.min(ratio * A, clipped * A)

            loss_k = -surrogate.sum()  # ← .sum() NOT .mean()
            losses.append(loss_k)

    return torch.stack(losses).mean()  # Average over batch
```

---

## GSPO — Sequence-Level

**Paper**: "Group Sequence Policy Optimization" (Qwen/Alibaba, 2025, arXiv:2507.18071)
**Key idea**: Importance sampling should match the reward granularity — sequence-level.

### The Insight

In GRPO, rewards are **per-sequence** (the whole response gets one reward), but
importance sampling ratios are **per-token**. This mismatch causes high variance:
a single token with a wild ratio can dominate the gradient for the entire response.

GSPO fixes this by computing the importance ratio **at the sequence level**.

### Math

**Importance sampling** (sequence-level, geometric mean of token ratios):
```
w_i = exp( (1/|o_i|) * Σ_t log(π_θ(t) / π_old(t)) )
    = (Π_t π_θ(t) / π_old(t))^(1/|o_i|)
```

This is the geometric mean of token-level ratios — a single number per sequence.
Then the loss is:

```
L_i = min(w_i * A_i, clip(w_i, 1-ε, 1+ε) * A_i)
L = (1/G) * Σ_i L_i
```

### Why Geometric Mean?

The product of token probabilities IS the sequence probability:
`π(sequence) = Π_t π(token_t | context_t)`

Taking the geometric mean (= `exp(mean(log_ratios))`) gives a length-normalized
sequence-level ratio. This automatically handles length normalization too —
you don't need the `1/|o_i|` hack.

### Pseudocode

```python
def gspo_loss(prompts, completions, rewards, old_logprobs, new_logprobs):
    losses = []
    for prompt_idx in range(len(prompts)):
        group_rewards = rewards[prompt_idx]
        mu = group_rewards.mean()
        sigma = group_rewards.std() + 1e-8
        advantages = (group_rewards - mu) / sigma

        for k in range(K):
            log_ratio = new_logprobs[prompt_idx][k] - old_logprobs[prompt_idx][k]
            # shape: [seq_len]

            # Sequence-level ratio: geometric mean of token ratios
            seq_log_ratio = log_ratio.mean()  # mean of log = log of geometric mean
            w = torch.exp(seq_log_ratio)      # scalar!

            A = advantages[k]
            clipped = torch.clamp(w, 1 - eps, 1 + eps)
            loss_k = -torch.min(w * A, clipped * A)  # scalar loss

            losses.append(loss_k)

    return torch.stack(losses).mean()
```

### Key Difference from GRPO

```python
# GRPO: token-level ratio, applied per-token
ratio = torch.exp(log_ratio)           # shape: [seq_len]
loss = -(ratio * A).mean()             # average over tokens

# GSPO: sequence-level ratio, single scalar
w = torch.exp(log_ratio.mean())        # shape: scalar
loss = -(w * A)                        # single scalar loss
```

---

## DAPO — The Kitchen Sink

**Paper**: "DAPO: An Open-Source LLM RL System" (2025, arXiv:2503.14476)
**Key idea**: Fix GRPO's four practical problems with four targeted innovations.

### The Four Fixes

#### Fix 1: Clip-Higher (Asymmetric Clipping)

Standard PPO clips symmetrically: `[1-ε, 1+ε]`. This limits how much the policy can
change in BOTH directions equally. But for exploration, you want to allow aggressive
updates when the advantage is positive (found something good) while being conservative
when negative (don't unlearn too much).

```
# Standard PPO:
clip(ratio, 1 - 0.2, 1 + 0.2)         → [0.8, 1.2]

# DAPO Clip-Higher:
clip(ratio, 1 - 0.2, 1 + 0.28)        → [0.8, 1.28]
```

Wider upper bound = more exploration, less entropy collapse.

#### Fix 2: Dynamic Sampling

If all K completions for a prompt get the same reward (all correct or all wrong),
`σ = 0` and all advantages are zero → no gradient. Wasted compute.

Dynamic sampling detects these zero-variance groups and replaces them with new
prompts that produce diverse rewards:

```python
while batch_has_zero_variance_groups and retries < max_retries:
    resample_zero_variance_prompts()
    regenerate_completions()
    recompute_rewards()
```

#### Fix 3: Token-Level Policy Gradient with Ratio Clamping

DAPO adds a second clamp on individual token ratios BEFORE the sequence-level
surrogate computation. This prevents a single token with a huge ratio from
dominating the gradient:

```python
# Standard: just surrogate clipping
ratio = exp(new_logprob - old_logprob)
surrogate = min(ratio * A, clip(ratio, 1-ε_lo, 1+ε_hi) * A)

# DAPO: also clamp individual ratios
ratio = exp(new_logprob - old_logprob)
ratio = clamp(ratio, 1 - c, 1 + c)    # ← token-level clamp (ratio_clip_c)
surrogate = min(ratio * A, clip(ratio, 1-ε_lo, 1+ε_hi) * A)
```

#### Fix 4: Overlong Reward Shaping

Responses approaching `max_length` without a stop token are probably degenerate.
Apply a linearly increasing penalty:

```python
if len(response) > max_length - buffer:
    penalty = penalty_factor * (len(response) - max_length + buffer) / buffer
    reward -= penalty
```

### Full Pseudocode

```python
def dapo_loss(prompts, completions, rewards, old_logprobs, new_logprobs,
              eps_lo=0.2, eps_hi=0.28, ratio_clip_c=0.2,
              max_length=512, overlong_buffer=128, overlong_penalty=1.0):

    # --- Fix 4: Overlong reward shaping ---
    for i, completion in enumerate(completions):
        if len(completion) > max_length - overlong_buffer:
            overshoot = len(completion) - (max_length - overlong_buffer)
            rewards[i] -= overlong_penalty * (overshoot / overlong_buffer)

    # --- Fix 2: Dynamic sampling ---
    for prompt_idx in range(len(prompts)):
        group_r = rewards[prompt_idx]
        if group_r.std() < 1e-8:  # Zero variance → resample
            # In practice: regenerate completions for this prompt
            # or swap in a different prompt
            continue

    losses = []
    for prompt_idx in range(len(prompts)):
        group_r = rewards[prompt_idx]
        mu = group_r.mean()
        sigma = group_r.std() + 1e-8
        advantages = (group_r - mu) / sigma

        for k in range(K):
            log_ratio = new_logprobs[prompt_idx][k] - old_logprobs[prompt_idx][k]
            ratio = torch.exp(log_ratio)

            # --- Fix 3: Token-level ratio clamping ---
            ratio = torch.clamp(ratio, 1 - ratio_clip_c, 1 + ratio_clip_c)

            A = advantages[k]

            # --- Fix 1: Asymmetric (Clip-Higher) surrogate ---
            clipped = torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi)
            surrogate = torch.min(ratio * A, clipped * A)

            # Token-level loss, averaged
            loss_k = -surrogate.mean()
            losses.append(loss_k)

    return torch.stack(losses).mean()
```

---

## Side-by-Side Comparison

| | **GRPO** | **Dr. GRPO** | **GSPO** | **DAPO** |
|---|---|---|---|---|
| **Origin** | DeepSeek (2024) | Sea AI Lab (2025) | Qwen/Alibaba (2025) | Community (2025) |
| **Paper** | DeepSeek-R1 | arXiv:2503.20783 | arXiv:2507.18071 | arXiv:2503.14476 |
| **Importance sampling** | Per-token | Per-token | **Per-sequence** (geometric mean) | Per-token |
| **Clipping** | Symmetric `[1-ε, 1+ε]` | Symmetric | Symmetric | **Asymmetric** (Clip-Higher) |
| **Length normalization** | Yes (`1/\|o_i\|`) — biased | **No** — unbiased | Implicit (geometric mean) | Yes (token mean) |
| **Std normalization** | Yes (`/σ`) — biased | **No** | Yes | Yes |
| **Dynamic sampling** | No | No | No | **Yes** — skips zero-variance groups |
| **Token ratio clamping** | No | No | N/A (sequence-level) | **Yes** (`ratio_clip_c`) |
| **Overlong shaping** | No | No | No | **Yes** — penalty near max length |
| **KL penalty** | Optional (β) | None | Optional | **None** (relies on clipping) |
| **Length bias** | ✅ Present | ❌ Removed | ❌ Removed | ⚠️ Mitigated by overlong shaping |
| **Difficulty bias** | ✅ Present | ❌ Removed | ✅ Present | ✅ Present (but dynamic sampling helps) |
| **Variance** | Medium | Medium | **Low** (sequence-level) | Medium (but more stable from fixes) |
| **NeMo RL support** | ✅ | Partial (`token_level_loss: true`) | ✅ | ✅ |

### The One-Line Summary

- **GRPO**: Baseline. Simple but biased.
- **Dr. GRPO**: GRPO minus two bugs (length bias + difficulty bias). Two-line code change.
- **GSPO**: Move importance sampling from token→sequence level. Principled fix for the ratio-reward mismatch.
- **DAPO**: Throw four orthogonal fixes at GRPO. Most practically robust.

---

## Pseudocode for All Four

Here's the critical inner loop for each, stripping away the shared scaffolding:

```python
# ═══════════════════════════════════════════════════
# GRPO — token-level ratio, token-mean loss
# ═══════════════════════════════════════════════════
A = (r - r.mean()) / (r.std() + eps)              # normalized advantage
ratio = exp(new_logp - old_logp)                   # [seq_len]
clip_ratio = clamp(ratio, 1-ε, 1+ε)
loss = -min(ratio * A, clip_ratio * A).mean()      # .mean() over tokens


# ═══════════════════════════════════════════════════
# Dr. GRPO — token-level ratio, token-SUM loss, no std
# ═══════════════════════════════════════════════════
A = r - r.mean()                                   # NOT divided by std
ratio = exp(new_logp - old_logp)                   # [seq_len]
clip_ratio = clamp(ratio, 1-ε, 1+ε)
loss = -min(ratio * A, clip_ratio * A).sum()        # .sum() NOT .mean()


# ═══════════════════════════════════════════════════
# GSPO — sequence-level ratio
# ═══════════════════════════════════════════════════
A = (r - r.mean()) / (r.std() + eps)
log_ratio = (new_logp - old_logp).mean()           # mean of log = geometric mean
w = exp(log_ratio)                                  # scalar
clip_w = clamp(w, 1-ε, 1+ε)
loss = -min(w * A, clip_w * A)                     # scalar


# ═══════════════════════════════════════════════════
# DAPO — token-level, Clip-Higher, ratio clamped
# ═══════════════════════════════════════════════════
A = (r - r.mean()) / (r.std() + eps)
ratio = exp(new_logp - old_logp)
ratio = clamp(ratio, 1-c, 1+c)                     # token-level pre-clamp
clip_ratio = clamp(ratio, 1-ε_lo, 1+ε_hi)          # asymmetric surrogate
loss = -min(ratio * A, clip_ratio * A).mean()
```

---

## NeMo RL Configs

### GRPO (vanilla)

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2              # Symmetric
  ratio_clip_c: null               # No token-level clamping
  token_level_loss: true           # token-mean (vanilla GRPO)
  reference_policy_kl_penalty: 0.01

grpo:
  normalize_rewards: true          # /σ normalization
  use_leave_one_out_baseline: true
  use_dynamic_sampling: false
  reward_shaping:
    enabled: false
```

### Dr. GRPO

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2
  ratio_clip_c: null
  token_level_loss: true           # NeMo RL averages over tokens by default
  reference_policy_kl_penalty: 0.0 # Dr. GRPO drops KL

grpo:
  normalize_rewards: false         # ← KEY: no std normalization
  use_leave_one_out_baseline: false
  use_dynamic_sampling: false
  reward_shaping:
    enabled: false
```

> **Note**: Full Dr. GRPO requires token-sum (not token-mean). NeMo RL's
> `token_level_loss: true` does token-mean. You get most of the benefit from
> removing std normalization. True token-sum may need a custom loss.

### GSPO

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2
  importance_sampling_level: "sequence"  # ← KEY: sequence-level ratios
  token_level_loss: false
  reference_policy_kl_penalty: 0.01

grpo:
  normalize_rewards: true
  use_leave_one_out_baseline: true
  use_dynamic_sampling: false
  reward_shaping:
    enabled: false
```

### DAPO

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.28             # ← Clip-Higher (asymmetric)
  ratio_clip_c: 0.2               # ← Token-level pre-clamping
  token_level_loss: true
  reference_policy_kl_penalty: 0.0 # DAPO drops KL, relies on clipping

grpo:
  normalize_rewards: true
  use_leave_one_out_baseline: true
  use_dynamic_sampling: true       # ← Dynamic sampling
  dynamic_sampling_max_gen_batches: 5
  reward_shaping:
    enabled: true                  # ← Overlong penalty
    overlong_buffer_length: 128
    overlong_buffer_penalty: 1.0
    max_response_length: 512
```

### Our CQL Config (DAPO-lite)

We use DAPO's Clip-Higher + dynamic sampling but skip overlong shaping
(CQL queries are short):

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.28             # Clip-Higher
  token_level_loss: true
  reference_policy_kl_penalty: 0.0

grpo:
  use_dynamic_sampling: true
  reward_shaping:
    enabled: false                 # CQL is short, no overlong needed
```

---

## Which One Should I Use?

### Decision Tree

```
Is your task binary-reward (pass/fail)?
├── YES → DAPO (dynamic sampling handles zero-variance groups)
└── NO (continuous reward like ours)
    │
    ├── Are responses long (>100 tokens)?
    │   ├── YES → DAPO (overlong shaping) or GSPO (sequence-level)
    │   └── NO → GRPO or Dr. GRPO is fine
    │
    └── Are you seeing length hacking?
        ├── YES → Dr. GRPO (removes length bias) or GSPO
        └── NO → Vanilla GRPO works
```

### For CQL specifically

CQL queries are **short** (10-50 tokens), our rewards are **continuous** [0, 1],
and we have **three reward components** (syntax, execution, ngram). This means:

- Length bias is minimal → Dr. GRPO's fix doesn't matter much
- Zero-variance groups are rare (continuous rewards) → dynamic sampling helps less
- Overlong shaping is unnecessary → CQL queries are naturally short

**Our choice: DAPO-lite** (Clip-Higher for better exploration, dynamic sampling as
insurance, skip overlong shaping). This gives us the exploration benefits without
unnecessary complexity.

---

## References

- **GRPO**: [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2401.02954)
- **Dr. GRPO**: [Understanding R1-Zero-Like Training (arXiv:2503.20783)](https://arxiv.org/abs/2503.20783)
- **GSPO**: [Group Sequence Policy Optimization (arXiv:2507.18071)](https://arxiv.org/abs/2507.18071) | [Qwen Blog](https://qwen.ai/blog?id=gspo)
- **DAPO**: [DAPO (arXiv:2503.14476)](https://arxiv.org/abs/2503.14476) | [NeMo RL Guide](https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html)
- **λ-GRPO**: [Learnable Token Preferences (arXiv:2510.06870)](https://arxiv.org/abs/2510.06870)
- **Technical deep-dive (all algorithms)**: [HuggingFace Forum](https://discuss.huggingface.co/t/offering-a-technical-deep-dive-on-grpo-dapo-dr-grpo-algorithms/154480)
- **The Illustrated GRPO**: [Visual guide](https://abderrahmanskiredj.github.io/the-illustrated-grpo/)
- **verl GRPO docs**: [verl.readthedocs.io](https://verl.readthedocs.io/en/latest/algo/grpo.html)

---

*See also: [NeMo RL Reference](nemo_rl_reference.md) |
[Training Parameters Deep-Dive](nemo_gym_rl_guide.md#training-parameters-deep-dive) |
[Reward Design & Strategy](rewards_and_strategy.md)*
