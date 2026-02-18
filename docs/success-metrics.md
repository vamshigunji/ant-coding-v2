# Success Metrics Framework — ant-coding

## The Core Problem

Comparing a single agent to a multi-agent system on raw token count is meaningless. A 3-agent system will **always** use more tokens than a 1-agent system — it has 3x the conversation turns by construction. The research question isn't "which uses fewer tokens" but **"which delivers more value per token spent."**

---

## The Metrics That Actually Matter

### Tier 1: Primary Metrics (What Decides "Better")

These are the metrics that answer your research question.

#### 1. Task Resolution Rate (%)

```
resolution_rate = tasks_where_all_tests_pass / total_tasks
```

The most important metric. Did the system actually solve the problem? Binary: tests pass or they don't.

| Realistic | Optimistic |
|-----------|------------|
| 25-40% on SWE-bench Lite (single agent baselines are ~20-30%) | 50-60% on SWE-bench Lite |
| 60-80% on custom/easy tasks | 90%+ on custom/easy tasks |

**Why this range:** Current state-of-the-art single agents (Aider, SWE-agent) score 20-35% on SWE-bench Lite. Multi-agent should beat that if the architecture adds real value. If your multi-agent system scores *below* a single agent, the architecture is adding overhead, not intelligence.

#### 2. Cost Per Resolved Task ($)

```
cost_per_resolution = total_cost_usd / tasks_resolved
```

**This is the single most important efficiency metric.** It captures the quality-efficiency tradeoff in one number.

| Architecture | Cost/Task | Resolution Rate | Cost/Resolution |
|-------------|-----------|----------------|-----------------|
| 1 agent | $0.15 | 30% | **$0.50** |
| 3 agents (shared) | $0.40 | 55% | **$0.73** |
| 3 agents (isolated) | $0.45 | 40% | **$1.13** |
| 3 agents (hybrid) | $0.38 | 58% | **$0.66** |

In this example, hybrid memory wins — not because it uses fewer tokens, but because it resolves more tasks per dollar. The single agent is cheapest per task but most expensive per *resolved* task only if multi-agent resolution rate is high enough.

**The breakeven question:** At what resolution rate does the multi-agent cost-per-resolution beat the single-agent cost-per-resolution?

```
breakeven_resolution_rate = (multi_agent_cost_per_task / single_agent_cost_per_resolution)
```

If single agent costs $0.50/resolution and multi-agent costs $0.40/task, multi-agent needs >80% resolution rate to beat it. If multi-agent costs $0.40/task and achieves 55% resolution, that's $0.73/resolution — **worse** than single agent. This is the honest math.

#### 3. First-Attempt Resolution Rate (pass@1)

```
pass_at_1 = probability_of_solving_on_first_try
```

More important than pass@3 or pass@5 in practice — you usually can't run a task 5 times in production. Multi-agent architectures should improve first-attempt reliability through review loops, not just through brute-force retries.

| Realistic | Optimistic |
|-----------|------------|
| 15-25% single agent on SWE-bench | 35-50% multi-agent on SWE-bench |
| 50-65% single agent on custom tasks | 75-90% multi-agent on custom tasks |

---

### Tier 2: Efficiency Metrics (How Much Value Per Token)

These explain *why* one architecture beats another.

#### 4. Useful Token Ratio

```
useful_token_ratio = tokens_in_successful_runs / total_tokens_across_all_runs
```

A system that uses 50K tokens to solve 8/10 tasks wastes 20% of tokens on failures. A system that uses 80K tokens to solve 9/10 tasks wastes 11% on failures. The second system uses more tokens total but wastes fewer proportionally.

| Realistic | Optimistic |
|-----------|------------|
| 40-60% (lots of tokens burned on failures) | 70-85% (most token spend produces results) |

#### 5. Token Overhead Ratio

```
overhead_ratio = multi_agent_tokens / single_agent_tokens  (on same task)
```

Multi-agent systems add coordination overhead. Measure it.

| Overhead | Interpretation |
|----------|---------------|
| 1.0-1.5x | Excellent — minimal coordination cost |
| 1.5-2.5x | Typical — acceptable if resolution rate increase justifies it |
| 2.5-4.0x | Expensive — architecture is chatty, agents duplicating work |
| 4.0x+ | Red flag — agents likely in loops or sharing too much context |

The healthy range is 1.5-2.5x overhead WITH a corresponding 15-30% resolution rate increase.

#### 6. Tokens Per Resolved Task

```
tokens_per_resolution = total_tokens / tasks_resolved
```

Unlike raw token count, this normalizes by success. A system using 100K tokens to solve 10 tasks (10K/resolution) is more efficient than one using 60K tokens to solve 4 tasks (15K/resolution).

---

### Tier 3: Quality Metrics (Beyond Pass/Fail)

Binary pass/fail misses nuance. Two patches can both pass tests but differ wildly in quality.

#### 7. Patch Quality Score (1-5, LLM-as-Judge)

Four dimensions:

| Dimension | What It Measures | Why It Matters |
|-----------|-----------------|----------------|
| **Correctness** | Does it fix the root cause, not just symptoms? | A patch can pass tests by coincidence |
| **Minimality** | How surgical is the change? | 3-line fix > 50-line refactor for a bug |
| **Code Quality** | Is it idiomatic, readable, maintainable? | Production-grade vs hacky |
| **Completeness** | Does it handle edge cases? | Passes provided tests vs robust |

**Realistic range:** 2.5-3.5 average across dimensions  
**Optimistic range:** 3.5-4.5 average

**Key hypothesis to test:** Multi-agent systems (especially with a Reviewer agent) should score higher on code quality and completeness than single-agent systems, even if resolution rates are similar. The reviewer catches issues the coder misses.

#### 8. Patch Size Ratio

```
patch_size_ratio = generated_patch_lines / gold_standard_patch_lines
```

| Ratio | Interpretation |
|-------|---------------|
| 0.8-1.2 | Ideal — similar scope to human fix |
| 1.2-2.0 | Acceptable — some extra changes |
| 2.0-5.0 | Bloated — agent over-modified the codebase |
| 5.0+ | Agent rewrote things it shouldn't have |

---

### Tier 4: Robustness Metrics (Consistency and Failure Analysis)

#### 9. Resolution Variance (Consistency)

```
variance = std_dev(pass_rates_across_runs) / mean(pass_rates_across_runs)
```

Low variance = reliable architecture. High variance = architecture is gambling.

**Multi-agent hypothesis:** Multi-agent systems should have LOWER variance than single agents because multiple agents catch each other's mistakes. If variance is higher, the architecture is introducing new failure modes.

| Coefficient of Variation | Interpretation |
|--------------------------|---------------|
| < 0.15 | Highly consistent |
| 0.15-0.30 | Moderate — some run-to-run variation |
| 0.30+ | Unreliable — results depend heavily on luck |

#### 10. Error Recovery Rate

```
recovery_rate = tasks_that_failed_then_succeeded_after_retry / tasks_that_failed_initially
```

When an agent gets a test failure, does the system recover? This is where multi-agent architectures (especially Loop + Debugger) should shine vs single-agent.

**Realistic:** 20-35% recovery rate  
**Optimistic:** 50-70% recovery rate

#### 11. Failure Category Breakdown

Not all failures are equal. Categorize them:

| Category | Description | What It Reveals |
|----------|-------------|-----------------|
| **Planning failure** | Wrong approach chosen | Planner agent needs better context |
| **Implementation failure** | Right approach, wrong code | Coder agent capability issue |
| **Integration failure** | Code works in isolation, breaks in context | Agents not sharing enough state |
| **Hallucination cascade** | Agent A hallucinated, B built on it | Shared memory propagating errors |
| **Timeout** | Ran out of tokens/time | Architecture too slow for task complexity |
| **Tool failure** | Code execution or file ops failed | Infrastructure issue, not architecture |

**Key research signal:** If shared memory has more "hallucination cascade" failures than isolated memory, that's evidence that shared memory has a downside: error propagation.

---

## The Honest Comparison Framework

### Single Agent vs Multi-Agent: Fair Comparison Rules

| Rule | Why |
|------|-----|
| Same model for all agents | Don't compare Claude-3-agent vs GPT-1-agent |
| Same token budget | Give single agent the SAME total budget as multi-agent |
| Same tasks, same order | Eliminate task-ordering effects |
| Same temperature (0.0) | Eliminate randomness |
| Multiple runs (k≥3) | Reduce noise, enable pass@k |

### The Comparison Table You'll Actually Produce

```markdown
| Metric                    | Single Agent | 2-Agent Seq | 3-Agent Seq | 3-Agent Loop | p-value |
|---------------------------|-------------|-------------|-------------|--------------|---------|
| Resolution Rate (%)       | 30.0        | 42.0        | 48.0        | 55.0         | 0.003   |
| Cost Per Resolution ($)   | 0.50        | 0.71        | 0.83        | 0.73         | 0.041   |
| pass@1 (%)                | 25.0        | 35.0        | 40.0        | 50.0         | 0.008   |
| Useful Token Ratio (%)    | 45.0        | 55.0        | 52.0        | 65.0         | 0.012   |
| Overhead Ratio            | 1.0x        | 1.8x        | 2.4x        | 2.1x         | —       |
| Tokens/Resolution (K)     | 12.5        | 14.2        | 16.7        | 12.8         | 0.087   |
| Patch Quality (1-5)       | 3.1         | 3.4         | 3.8         | 3.9          | 0.022   |
| Variance (CV)             | 0.28        | 0.22        | 0.19        | 0.15         | 0.034   |
| Error Recovery Rate (%)   | 0.0*        | 15.0        | 25.0        | 45.0         | 0.001   |
```

*Single agent has no recovery — it gets one shot per run.

### Reading This Table: What "Winning" Looks Like

**Realistic win for multi-agent:**
- Resolution rate +10-20% over single agent
- Cost per resolution within 1.5x of single agent
- Lower variance (more consistent)
- Higher patch quality scores
- Error recovery > 20%

**Optimistic win for multi-agent:**
- Resolution rate +25-40% over single agent
- Cost per resolution LOWER than single agent (extra agents pay for themselves)
- Variance cut in half
- Patch quality 4.0+
- Error recovery > 50%

**Multi-agent loses if:**
- Resolution rate < single agent (architecture adds confusion, not intelligence)
- Cost per resolution > 2x single agent (overhead too high for the quality gain)
- Higher variance than single agent (new failure modes outweigh benefits)

---

## The Metrics That Answer Your Research Questions

Mapping back to your original research questions:

| Research Question | Primary Metric | Supporting Metrics |
|-------------------|---------------|-------------------|
| Does shared memory outperform isolated? | Resolution rate + Cost per resolution | Useful token ratio, overhead ratio |
| Is multi-agent worth the extra tokens? | Cost per resolution (multi vs single) | Tokens per resolution, breakeven analysis |
| Which architecture is most reliable? | Resolution variance + pass@1 | Error recovery rate, failure categories |
| Does the reviewer agent add value? | Resolution rate delta (with/without reviewer) | Patch quality score, hallucination cascade count |
| Is the effect model-dependent? | Resolution rate across models (same architecture) | Cost per resolution across models |
| Where are the diminishing returns? | Resolution rate vs agent count curve | Cost per resolution vs agent count |

---

## Metric Collection Checklist

For every experiment, collect and record ALL of these:

```yaml
# In experiments/registry.yml
outcome:
  # Tier 1: Primary
  resolution_rate: null           # %
  cost_per_resolution: null       # $
  pass_at_1: null                 # %
  pass_at_3: null                 # %
  
  # Tier 2: Efficiency
  total_tokens: null
  total_cost: null                # $
  useful_token_ratio: null        # %
  overhead_ratio: null            # x (vs single agent baseline)
  tokens_per_resolution: null     # K
  
  # Tier 3: Quality
  avg_patch_quality: null         # 1-5
  avg_patch_size_ratio: null      # x
  
  # Tier 4: Robustness
  resolution_variance_cv: null    # coefficient of variation
  error_recovery_rate: null       # %
  failure_categories:
    planning: null
    implementation: null
    integration: null
    hallucination_cascade: null
    timeout: null
    tool_failure: null
```

---

## One More Thing: The "So What" Test

After each experiment, ask yourself:

> If I showed this result to someone skeptical of multi-agent systems, would they be convinced?

If the answer is "no" — you need either better metrics, more data, or an honest acknowledgment that the architecture didn't help for this task type. **Negative results are valuable.** "Shared memory doesn't help for simple bug fixes but significantly helps for multi-file refactoring tasks" is a more useful finding than "multi-agent is better."
