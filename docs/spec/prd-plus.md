# PRD+ : Additional Requirements

**Applies to:** `docs/spec/prd.md`  
**Status:** This document extends the PRD. Where they conflict, PRD+ wins.  
**Why this exists:** The original PRD designed the framework layers but underspecified the metrics collection. This addendum ensures the framework can collect all 11 success metrics defined in `docs/spec/success-metrics.md`.

---

## 1. Expanded Type Definitions

### 1.1 ExperimentMetrics — Additional Fields

**Extends:** PRD Section 10.2

Add these fields to the existing `ExperimentMetrics` dataclass. Existing fields remain unchanged.

```python
@dataclass
class ExperimentMetrics:
    # ── All original fields from PRD Section 10.2 remain ──

    # ── NEW: Tier 1 — Primary ──
    cost_per_resolution: float = 0.0
    # total_cost_usd / tasks_passed. Infinity if 0 passed.
    # This is the single most important efficiency metric.

    # ── NEW: Tier 2 — Efficiency ──
    useful_token_ratio: float = 0.0
    # tokens_in_successful_runs / total_tokens.
    # Answers: what % of token spend produced results?

    overhead_ratio: float = 0.0
    # total_tokens / baseline_experiment_tokens.
    # 0.0 if no baseline configured. Answers: how much does multi-agent coordination cost?

    tokens_per_resolution: float = 0.0
    # total_tokens / tasks_passed. Infinity if 0 passed.
    # Normalizes token count by success.

    # ── NEW: Tier 3 — Quality ──
    avg_patch_quality: float = 0.0
    # Mean of LLM judge "overall" scores across all tasks (1-5 scale).

    avg_patch_size_ratio: float = 0.0
    # Mean of (generated_patch_lines / gold_patch_lines) across tasks where gold exists.
    # 0.0 if no gold standard patches available. Ideal range: 0.8-1.2.

    # ── NEW: Tier 4 — Robustness ──
    resolution_variance_cv: float = 0.0
    # Coefficient of variation of per-task pass rates across runs.
    # Lower = more consistent. Requires runs_per_task >= 2.

    error_recovery_rate: float = 0.0
    # tasks_that_recovered / tasks_that_initially_failed.
    # "Recovered" = intermediate_test_results starts with False, ends with True.

    failure_categories: dict = field(default_factory=lambda: {
        "planning": 0,
        "implementation": 0,
        "integration": 0,
        "hallucination_cascade": 0,
        "timeout": 0,
        "tool_failure": 0,
    })
    # Count of failed tasks per failure category. Filled by FailureClassifier.
```

### 1.2 TaskResult — Additional Fields

**Extends:** PRD Section 4.2

Add these fields to the existing `TaskResult` dataclass. Existing fields remain unchanged.

```python
@dataclass
class TaskResult:
    # ── All original fields from PRD Section 4.2 remain ──

    # NEW: Intermediate test results (for loop/iterative patterns)
    intermediate_test_results: list[bool] = field(default_factory=list)
    # Tracks pass/fail at each iteration within a single run.
    # Example: [False, False, True] = failed twice, passed on 3rd attempt.
    # For non-loop patterns: [True] or [False] (single attempt).
    # Used to calculate error_recovery_rate.

    # NEW: Failure classification
    failure_category: Optional[str] = None
    # One of: "planning", "implementation", "integration",
    # "hallucination_cascade", "timeout", "tool_failure"
    # Only set for failed tasks. Filled by FailureClassifier after run.

    # NEW: Patch size comparison
    generated_patch_lines: int = 0
    gold_patch_lines: int = 0
    # gold_patch_lines = 0 if no gold standard available (custom tasks).
    # For SWE-bench tasks, populated from the dataset's expected patch.

    # NEW: LLM Judge scores (per-task)
    judge_scores: Optional[dict] = None
    # {"correctness": 4, "minimality": 3, "code_quality": 4,
    #  "completeness": 3, "overall": 3.5, "reasoning": "..."}
    # Set by LLMJudge.evaluate(). Aggregated into ExperimentMetrics.avg_patch_quality.
```

---

## 2. Experiment Config Addition

### 2.1 Baseline Reference

**Extends:** PRD Section 12.2

Add one optional field to the experiment YAML schema:

```yaml
experiment:
  # ... all existing fields remain ...

  # NEW: Reference experiment for overhead calculation
  baseline_experiment_id: null
  # Optional. When set, the eval harness loads metrics from
  # results/{baseline_experiment_id}/metrics.json
  # and computes: overhead_ratio = this.total_tokens / baseline.total_tokens
  #
  # Set this to your single-agent experiment ID.
  # Example: baseline_experiment_id: "single-agent-claude"
```

Add this to the `ExperimentConfig` Pydantic model as:

```python
baseline_experiment_id: Optional[str] = None
```

---

## 3. LLM Judge Dimension Change

**Overrides:** PRD Section 10.3

The original PRD defined 4 judge dimensions: correctness, code_quality, completeness, efficiency.

**Replace with:** correctness, minimality, code_quality, completeness.

| Dimension | What It Measures | 5 (best) | 3 (mid) | 1 (worst) |
|-----------|-----------------|----------|---------|-----------|
| **Correctness** | Fixes root cause, not just symptoms | Root cause fixed | Symptom patched | Wrong fix |
| **Minimality** | How surgical is the change | Minimal diff | Some extra changes | Rewrote everything |
| **Code Quality** | Idiomatic, readable, maintainable | Production-grade | Acceptable | Hacky |
| **Completeness** | Handles edge cases | Comprehensive | Main case only | Incomplete |

**Why:** "Efficiency" was removed because it overlaps with minimality, and minimality is easier to score consistently. The judge prompt should ask for a JSON response with these 4 scores plus "overall" (weighted average) and "reasoning" (string).

---

## 4. New Component: FailureClassifier

**Adds to:** PRD Section 10 (Evaluation Harness)

**File:** `src/ant_coding/eval/failure_classifier.py`

```python
class FailureClassifier:
    """
    Classifies WHY a task failed. Runs on every failed task after the experiment.
    Uses a cheap/fast model (default: Gemini Flash) since this runs at scale.

    Input: task description, generated patch, test output, event log summary
    Output: one of 6 categories

    Categories:
    ┌──────────────────────────┬──────────────────────────────────────────────┐
    │ Category                 │ Meaning                                      │
    ├──────────────────────────┼──────────────────────────────────────────────┤
    │ "planning"               │ Wrong approach chosen. Plan was flawed.       │
    │ "implementation"         │ Right approach, wrong code. Bugs, syntax.     │
    │ "integration"            │ Code works alone, breaks in context.          │
    │ "hallucination_cascade"  │ Agent A hallucinated, B built on it.          │
    │ "timeout"                │ Ran out of tokens or wall time.               │
    │ "tool_failure"           │ Code exec, file ops, or git failed.           │
    └──────────────────────────┴──────────────────────────────────────────────┘

    The classifier prompt includes:
    - Task description (what should have been fixed)
    - Generated patch (or "no patch generated")
    - Test output (the failure evidence)
    - Last 20 events from event log (the agent's decision trail)
    - Memory access summary (reads that returned None — information gaps)

    Error handling: If the classifier LLM returns malformed output,
    default to "implementation" and log a warning.
    """

    def __init__(self, model: str = "gemini/gemini-2.5-flash"): ...

    async def classify(
        self,
        task: Task,
        result: TaskResult,
        events: list[Event],
    ) -> str: ...
```

---

## 5. New Orchestration Pattern: SingleAgent

**Adds to:** PRD Section 6.4 (Reference Implementations)

**File:** `src/ant_coding/orchestration/examples/single_agent.py`

```python
@OrchestrationRegistry.register
class SingleAgent(OrchestrationPattern):
    """
    Single agent baseline — the control group for all multi-agent experiments.

    Architecture:
    - 1 agent with full tool access
    - No memory interaction (nothing to share with)
    - No multi-turn agent coordination
    - System prompt + task description + tools → solution

    Purpose:
    - Establishes the performance floor
    - Provides the denominator for overhead_ratio calculation
    - Answers: "Is multi-agent worth the extra tokens?"

    Every multi-agent experiment should set baseline_experiment_id
    to the corresponding SingleAgent experiment (same model, same tasks).
    """

    def name(self) -> str:
        return "single-agent"

    def description(self) -> str:
        return "Single agent with full tool access. Baseline for all multi-agent comparisons."

    def get_agent_definitions(self) -> list[dict]:
        return [{"name": "SoloAgent", "role": "End-to-end task solver"}]

    async def solve(self, task, model, memory, tools, workspace_dir) -> TaskResult:
        # 1. Build system prompt with task + file context
        # 2. Single model.complete() call with tools
        # 3. Execute tool calls (file edits, code execution)
        # 4. Run tests
        # 5. Return TaskResult
        #
        # Memory: write task result to memory for consistency with the
        # framework contract, but there's no other agent to read it.
        ...
```

---

## 6. Metrics Calculation Logic

**Extends:** PRD Section 10.2

The `calculate_metrics()` function in `eval/harness.py` must compute all tiers. Here is the calculation spec for the new fields:

```python
def calculate_metrics(
    results: list[TaskResult],
    config: ExperimentConfig,
    baseline_metrics: Optional[ExperimentMetrics] = None,
) -> ExperimentMetrics:
    m = ExperimentMetrics(...)  # existing calculation for original fields

    # ── Tier 1 ──
    m.cost_per_resolution = (
        m.total_cost_usd / m.tasks_passed
        if m.tasks_passed > 0 else float('inf')
    )

    # ── Tier 2 ──
    success_tokens = sum(r.total_tokens for r in results if r.success)
    m.useful_token_ratio = success_tokens / m.total_tokens if m.total_tokens > 0 else 0.0

    m.tokens_per_resolution = (
        m.total_tokens / m.tasks_passed
        if m.tasks_passed > 0 else float('inf')
    )

    if baseline_metrics and baseline_metrics.total_tokens > 0:
        m.overhead_ratio = m.total_tokens / baseline_metrics.total_tokens

    # ── Tier 3 ──
    quality_scores = [r.judge_scores["overall"] for r in results if r.judge_scores]
    m.avg_patch_quality = statistics.mean(quality_scores) if quality_scores else 0.0

    ratios = [
        r.generated_patch_lines / r.gold_patch_lines
        for r in results if r.gold_patch_lines > 0
    ]
    m.avg_patch_size_ratio = statistics.mean(ratios) if ratios else 0.0

    # ── Tier 4 ──
    # Resolution variance: group by task_id, compute per-task pass rate, then CV
    per_task = {}
    for r in results:
        per_task.setdefault(r.task_id, []).append(1.0 if r.success else 0.0)
    task_rates = [statistics.mean(runs) for runs in per_task.values()]
    if task_rates and statistics.mean(task_rates) > 0:
        m.resolution_variance_cv = statistics.stdev(task_rates) / statistics.mean(task_rates)

    # Error recovery: initially failed (first intermediate = False) but finally succeeded
    recovered = sum(
        1 for r in results
        if len(r.intermediate_test_results) > 1
        and not r.intermediate_test_results[0]
        and r.success
    )
    initially_failed = sum(
        1 for r in results
        if r.intermediate_test_results and not r.intermediate_test_results[0]
    )
    m.error_recovery_rate = recovered / initially_failed if initially_failed > 0 else 0.0

    # Failure categories: count from TaskResult.failure_category
    for r in results:
        if not r.success and r.failure_category:
            m.failure_categories[r.failure_category] += 1

    return m
```

---

## 7. Breakeven Analysis

**Adds to:** PRD Section 10.4 (StatisticalComparison)

```python
def breakeven_analysis(
    single_agent_metrics: ExperimentMetrics,
    multi_agent_cost_per_task: float,
) -> dict:
    """
    At what resolution rate does multi-agent break even with single-agent?

    breakeven_rate = multi_agent_cost_per_task / single_agent_cost_per_resolution
    """
    sa_cpr = single_agent_metrics.cost_per_resolution
    breakeven = multi_agent_cost_per_task / sa_cpr if sa_cpr > 0 and sa_cpr != float('inf') else 1.0
    return {
        "single_agent_cost_per_resolution": sa_cpr,
        "multi_agent_cost_per_task": multi_agent_cost_per_task,
        "breakeven_resolution_rate": min(breakeven, 1.0),
        "interpretation": (
            f"Multi-agent needs >{breakeven:.0%} resolution rate "
            f"to beat single-agent on cost-per-resolution"
        ),
    }
```

Include breakeven analysis in the comparison report when a baseline experiment is configured.

---

## 8. Story Impact Summary

These additions affect specific stories. Claude should read this document before starting any of these:

| PRD+ Section | Story Affected | What Changes |
|-------------|---------------|--------------|
| 1.1, 1.2 (Types) | **S1-E1-S05**: Core Type Definitions | Add new fields to ExperimentMetrics and TaskResult |
| 2.1 (Config) | **S1-E1-S04**: YAML Config Loader | Add `baseline_experiment_id` to ExperimentConfig |
| 3 (Judge) | **S5-E2-S02**: LLM-as-Judge | Dimensions → correctness, minimality, code_quality, completeness |
| 4 (Classifier) | **S5-E2-S05**: Failure Classification | New story — build FailureClassifier |
| 5 (SingleAgent) | **S4-E1-S04**: Reference Implementations | Add SingleAgent alongside Parallel and Loop |
| 6 (Calculation) | **S5-E2-S01**: Metrics Calculation | Implement all 4 tiers in calculate_metrics() |
| 7 (Breakeven) | **S5-E3-S01**: Paired Statistical Tests | Add breakeven_analysis() method |
