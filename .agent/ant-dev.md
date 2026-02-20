# CLAUDE.md — ant-coding

## Project

ant-coding is a benchmarking framework that empirically compares multi-agent LLM architectures for software engineering tasks. You are building the experiment harness — not the experiments themselves. The human (Vamshi) designs orchestration experiments using the plugin interface you build.

**Repository:** https://github.com/vamshigunji/ant-coding-v2

**Naming note:** The PRD (`docs/spec/prd.md`) uses the old project name "AgentForge" in places. The project is **ant-coding**. The Python package is **`ant_coding`**. Use `ant_coding` everywhere in code (imports, module names, error classes). Ignore "AgentForge" in the PRD — treat it as "ant-coding".

---

## What to Read and When

### Every session (always, no exceptions)

1. **`.agent/sprint.yml`** — Your compass. Find the current sprint and the first story with status `not-started`. This is what you work on.
2. **The epic file referenced in sprint.yml** — e.g., `stories_and_epics/Sprint-2-Epic-1.md`. Contains your acceptance criteria (Given/When/Then). Do not write code until you've read these.

### First session (one-time)

3. **This file** — You're reading it now.
4. **`docs/spec/prd.md`** — Full technical specification.
5. **`docs/guides/contributing.md`** — Branching rules, commit convention, BRANCH_SUMMARY template.

### Triggered by context

| Trigger | Read This |
|---------|-----------|
| Story touches a **new layer** you haven't built yet | `docs/spec/prd.md` — relevant section (Sections 4-11) |
| Story involves **metrics, eval, types, or config** | `docs/spec/prd-plus.md` — additional requirements extending the PRD |
| Unsure how **layers connect** or what depends on what | `docs/architecture/system-overview.md` |
| Building anything in the **memory layer** | `docs/architecture/memory-architecture.md` |
| Wiring the **ExperimentRunner** or connecting layers | `docs/architecture/layer-interactions.md` and `docs/architecture/experiment-lifecycle.md` |
| Unsure about **branch naming, commit format, or merge process** | `docs/guides/contributing.md` |
| Writing a **BRANCH_SUMMARY.md** | `docs/guides/contributing.md` — template at the bottom |
| An epic is complete and you're doing **self-review** | Re-read the epic file's completion checklist |
| Understanding the **experiment config schema** | `docs/spec/prd.md` Section 12 |
| Understanding **evaluation metrics or pass@k** | `docs/spec/prd.md` Section 10 + `docs/spec/prd-plus.md` Section 6 |

**Rule: when in doubt, read the relevant doc. It's faster than guessing wrong and refactoring later.**

---

## Workflow

### Starting a Story

1. Read `.agent/sprint.yml` → find first `not-started` story in current sprint
2. Read the epic file → understand acceptance criteria
3. Read relevant PRD/PRD+ sections if touching a new layer
4. Update sprint.yml → set story status to `in-progress`
5. Create feature branch → `git checkout dev && git pull && git checkout -b feature/{story-id}`

### During Development

6. Write code in atomic conventional commits:
   ```
   <type>(<scope>): <description>
   ```
   - **Types:** `feat`, `fix`, `test`, `refactor`, `docs`, `chore`
   - **Scopes:** `config`, `models`, `memory`, `tools`, `orchestration`, `eval`, `observability`, `runner`, `scaffold`
7. Write tests alongside implementation — never finish a story without tests
8. Run tests after every logical change — `pytest tests/ -v --tb=short`
9. Run linter — `ruff check src/ && ruff format src/`

### Completing a Story

10. Verify ALL acceptance criteria from the epic file
11. Write `BRANCH_SUMMARY.md` in repo root (template below)
12. Merge to dev:
    ```bash
    git checkout dev && git pull
    git merge feature/{story-id} --no-ff -m "merge: {story-id} — {story name}"
    git push
    ```
    Do NOT delete the feature branch.
13. Update sprint.yml → set story status to `done`
14. Start next story → back to step 1

### Completing an Epic

15. When all stories in an epic are `done`:
    - Run the epic completion checklist from the epic file
    - Run full test suite — `pytest tests/ -v`
    - Run linter — `ruff check src/`
    - Self-review: read through all code changes for the epic. Check for missing error handling, untested edge cases, inconsistent naming, missing docstrings, unresolved TODOs.
    - Update sprint.yml → set epic status to `review`
    - Commit: `chore(sprint): mark {epic-id} as review`

### Moving to Next Sprint

16. When all epics in a sprint are `review` or `done`:
    - Update `current_sprint` in sprint.yml
    - Continue with first story of next sprint

---

## Code Standards

### Python
- Every public function has a docstring (what, params, return)
- Every module has a module-level docstring
- Type hints on all function signatures — no `Any` unless truly necessary
- No bare `except:` — catch specific exceptions
- No `# TODO` without a story ID
- Python 3.11+, async/await, Pydantic for config, dataclasses for data

### Testing
- Write tests BEFORE or DURING implementation, not after
- Each story's tests pass independently — `pytest tests/test_{module}.py -v`
- Use pytest fixtures for shared setup
- Mock external dependencies — no real API calls (LiteLLM, Google ADK)
- Test both happy path and error cases

### Git
- Never commit directly to `dev` or `main` — always feature branches
- Never force-push
- Each commit = one logical unit. Aim for 3-8 commits per story.
- Test suite must pass before merging to dev
- Never delete feature branches after merge

### Decisions
- Follow the PRD specification — it's your source of truth
- If PRD is ambiguous → make a reasonable decision, document in BRANCH_SUMMARY.md
- If PRD seems wrong → implement as specified, note concern in BRANCH_SUMMARY.md
- If a dependency isn't built yet → stub/mock it, note in BRANCH_SUMMARY.md

### Never Do
- Merge to `main` — only the human does that
- Skip tests
- Leave failing tests and move on
- Change the orchestration base class contract without documenting why
- Delete feature branches
- Update sprint.yml status without completing the work

---

## BRANCH_SUMMARY.md Template

Write this in repo root before merging each story. Overwrite the previous one (git history preserves old versions).

```markdown
# Branch Summary: feature/{story-id}

## Story
{story-id}: {story name}

## What Changed
- Brief list of what was added/modified

## Key Decisions
- Non-obvious choices and why
- Deviations from PRD (if any) and rationale

## Files Touched
- New and modified files

## How to Verify
\```bash
pytest tests/test_{module}.py -v
\```

## Notes for Reviewer
- Anything the human should pay attention to
- Known limitations or future improvements
```

---

## sprint.yml Updates

Change ONLY the `status` field. Valid transitions:
- `not-started` → `in-progress` (starting a story)
- `in-progress` → `done` (merged to dev, tests pass)
- All stories done → epic status becomes `review`

---

## Error Recovery

| Situation | Action |
|-----------|--------|
| Tests fail after merge to dev | Create `fix/{story-id}` branch, fix, merge |
| Merge conflict | Resolve in feature branch, re-test, then merge |
| Story depends on unfinished work | Stub/mock the dependency, note in BRANCH_SUMMARY.md |
| Acceptance criteria seem impossible | Implement what you can, document gap in BRANCH_SUMMARY.md |
| Design mistake in earlier sprint | Create `refactor/{description}` branch, fix, note in sprint.yml |

---

## Quick Reference

```bash
# What to work on
cat .agent/sprint.yml | grep "status: not-started" -B3 | head -20

# Start story
git checkout dev && git pull && git checkout -b feature/S1-E1-S01

# Commit
git commit -m "feat(memory): add MemoryManager with shared mode routing"

# Test
pytest tests/ -v --tb=short

# Lint
ruff check src/ && ruff format src/

# Merge
git checkout dev && git pull
git merge feature/S1-E1-S01 --no-ff -m "merge: S1-E1-S01 — Initialize Python Project"
git push

# Progress
grep -c "status: done" .agent/sprint.yml
```
