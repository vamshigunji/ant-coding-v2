# Branch Summary: feature/S5-E2-S02

## Story
S5-E2-S02: LLM-as-Judge Scoring (PRD+ Dimensions)

## What Changed
- Added `src/ant_coding/eval/llm_judge.py` with `LLMJudge` class
- Scores patches on 4 PRD+ dimensions: correctness, minimality, code_quality, completeness (1-5 scale)
- Computes weighted overall score
- Handles malformed judge responses gracefully (returns default scores with error note)
- Strips markdown code fences from responses, clamps scores to 1-5

## Key Decisions
- Default judge model: gemini/gemini-2.5-flash (cheap, fast for scale)
- Default weights: correctness=0.4, minimality/code_quality/completeness=0.2 each
- On parse failure, returns all 1s with error note rather than raising
- Temperature=0.0 for deterministic scoring

## Files Touched
- `src/ant_coding/eval/llm_judge.py` (new)
- `.agent/sprint.yml` (S5-E2-S02 done)

## How to Verify
```bash
pytest tests/ -v  # full suite: 131 passed, 1 skipped
```

## Notes for Reviewer
- Tests for LLMJudge will be in S5-E2-S05 (Evaluation Tests) with mocked LLM calls.
