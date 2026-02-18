# Sprint 2 — Epic 1: Model Abstraction Layer

**Epic ID:** S2-E1  
**Sprint:** 2  
**Priority:** P0 — Foundation  
**Goal:** Build the LiteLLM-based model provider that lets any LLM be called through a single interface. After this epic, `ModelProvider.complete()` works with Claude, GPT, and Gemini via config-only swaps.

**Dependencies:** S1-E1 (config system, type definitions)

---

## Story S2-E1-S01: ModelConfig and ModelProvider Core

**Branch:** `feature/S2-E1-S01`  
**Points:** 5

**Description:**  
Implement the ModelProvider class that wraps LiteLLM for unified async completions with automatic token/cost tracking.

**Acceptance Criteria:**

```gherkin
Given a ModelConfig with litellm_model="anthropic/claude-sonnet-4-5-20250929"
When I create ModelProvider(config) and call await provider.complete(messages=[...])
Then it returns a dict with choices[0].message.content
And provider.get_usage() shows prompt_tokens > 0 and completion_tokens > 0

Given a ModelProvider with temperature=0.0
When I call complete() twice with identical messages
Then both responses have temperature=0.0 in the underlying LiteLLM call

Given a ModelProvider
When the LiteLLM call fails with a transient error
Then it retries up to 3 times with exponential backoff
And if all retries fail, it raises a ModelError with the original error message

Given a ModelProvider with token tracking
When I call complete() 3 times
Then get_usage()["total_tokens"] equals the sum of all 3 calls
And get_usage()["total_cost_usd"] is calculated from cost_per_input/output_token

Given a ModelProvider
When I call reset_usage()
Then get_usage() returns all zeros
```

**Files to Create:**
- `src/ant_coding/models/provider.py`

---

## Story S2-E1-S02: Token Budget Enforcement

**Branch:** `feature/S2-E1-S02`  
**Points:** 3

**Description:**  
Add token budget tracking to ModelProvider. When cumulative tokens exceed the budget, further calls are rejected.

**Acceptance Criteria:**

```gherkin
Given a ModelProvider with a token_budget of 1000
When cumulative tokens reach 950 after a completion call
And I make another call that would use ~200 tokens
Then it raises TokenBudgetExceeded with the current total and the budget limit

Given a ModelProvider with no token_budget set (None)
When I make unlimited completion calls
Then no TokenBudgetExceeded is ever raised

Given a TokenBudgetExceeded error is raised
When I inspect the error
Then it contains: current_tokens, budget_limit, and last_call_tokens
```

**Files to Modify:**
- `src/ant_coding/models/provider.py` (add budget logic)

---

## Story S2-E1-S03: ModelRegistry with YAML Loading

**Branch:** `feature/S2-E1-S03`  
**Points:** 3

**Description:**  
Build the registry that loads model configs from `configs/models/` and instantiates ModelProvider on demand.

**Acceptance Criteria:**

```gherkin
Given YAML config files exist in configs/models/ for claude-sonnet, gpt-4o, gemini-flash
When I call registry.load_from_yaml("configs/models/")
Then registry.list_available() returns ["claude-sonnet", "gpt-4o", "gemini-flash"]

Given a loaded registry
When I call registry.get("claude-sonnet")
Then it returns a ModelProvider instance with litellm_model="anthropic/claude-sonnet-4-5-20250929"

Given a loaded registry
When I call registry.get("nonexistent-model")
Then it raises ModelNotFoundError with message containing "nonexistent-model"

Given a loaded registry
When I call registry.get("claude-sonnet") twice
Then each call returns a NEW ModelProvider instance (fresh token counters)
```

**Files to Create:**
- `src/ant_coding/models/registry.py`

---

## Story S2-E1-S04: Model Layer Unit Tests

**Branch:** `feature/S2-E1-S04`  
**Points:** 3

**Description:**  
Write comprehensive tests for the model layer. Use mocking for LiteLLM calls — tests must not make real API calls.

**Acceptance Criteria:**

```gherkin
Given the test suite runs with mocked LiteLLM
When I run `pytest tests/test_models.py -v`
Then all tests pass
And no real API calls are made (verified via mock assertions)

Given the test file
When I count test functions
Then there are at least 10 test cases covering:
  - Successful completion
  - Token tracking accumulation
  - Cost calculation
  - Retry on transient errors
  - Token budget enforcement
  - Registry loading from YAML
  - Registry get with valid/invalid names
  - Reset usage
  - Error handling for malformed responses
```

**Files to Create:**
- `tests/test_models.py`

---

## Epic Completion Checklist

- [ ] ModelProvider makes async LiteLLM calls with token/cost tracking
- [ ] Token budget enforcement works
- [ ] ModelRegistry loads from YAML configs
- [ ] `pytest tests/test_models.py` passes (all mocked, no API calls)
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated
