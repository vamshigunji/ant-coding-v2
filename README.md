# ant-coding

A benchmarking framework to empirically compare multi-agent LLM architectures for software engineering tasks — specifically whether **shared memory** outperforms **isolated memory** across token efficiency, output quality, and task completion rates.

---

## How This Repo Works

This repo is built by a **dev agent (Forge)** and reviewed by a **human (Vamshi)**. The docs, stories, and sprint plan exist from day 0. The code is created incrementally by the dev agent following the sprint plan.

### Two Roles

| Role | Who | Responsibility |
|------|-----|----------------|
| **Human / Master** | Vamshi | Reviews completed epics, merges `dev` → `main`, designs orchestration experiments |
| **Dev Agent (Forge)** | OpenClaw | Builds the framework story-by-story, pushes to `dev`, self-reviews before marking epic complete |

---

## For the Dev Agent (Forge)

**Start here →** `.agent/prompt.md`

This is your system prompt. It defines your persona, workflow, behavioral rules, and exactly how to navigate this repo. Read it fully on your first session.

**Your compass →** `.agent/sprint.yml`

Read this at the start of **every session**. It tells you:
- Which sprint is current
- Which story to work on next (first `not-started`)
- Where to find the story details (epic file path)

**Your workflow in 30 seconds:**
1. Read `.agent/sprint.yml` → find next story
2. Read the epic file → understand acceptance criteria (Given/When/Then)
3. Read relevant PRD section if touching a new layer → `docs/prd.md`
4. Create feature branch from `dev` → `feature/{story-id}`
5. Code with atomic conventional commits
6. Write tests, run them, run linter
7. Write `BRANCH_SUMMARY.md`
8. Merge to `dev`, update `sprint.yml` to `done`
9. Repeat

**When an epic is fully done:** Run the epic completion checklist, self-review, mark epic as `review` in sprint.yml.

---

## For the Human / Master

### Reviewing Forge's Work

Three zoom levels to understand what happened while you were away:

| Zoom Level | Where to Look | What You See |
|------------|--------------|--------------|
| 🔭 **Progress overview** | `.agent/sprint.yml` | Which stories/epics are done, in-progress, or waiting for review |
| 🔍 **Story decisions** | `BRANCH_SUMMARY.md` (or `git log -p -- BRANCH_SUMMARY.md`) | What changed, why, key decisions, how to verify |
| 🔬 **Exact coding steps** | `git log --oneline feature/{story-id}` | Commit-by-commit coding process |

### Merge Process

```bash
# 1. Check what's ready for review
grep "status: review" .agent/sprint.yml -B3

# 2. Review the dev branch
git diff main..dev

# 3. Read branch summaries for context
git log -p -- BRANCH_SUMMARY.md

# 4. Run tests
pytest tests/ -v

# 5. Merge when satisfied
git checkout main
git merge dev --no-ff -m "merge: Sprint X epics reviewed and approved"
git push
```

### Your Part: Orchestration Experiments

Layer 3 (Orchestration) is a **plugin interface**. Forge builds the framework; you design the experiments. After Sprint 4, you can create new orchestration patterns by:

1. Subclass `OrchestrationPattern` from `src/ant_coding/orchestration/base.py`
2. Implement `name()`, `description()`, `solve()`
3. Register with `@OrchestrationRegistry.register`
4. Create an experiment YAML in `configs/experiments/`
5. Run: `python scripts/run_experiment.py configs/experiments/your-experiment.yaml`

---

## Repo Structure

```
ant-coding-v2/
├── .agent/                          # Dev agent workspace
│   ├── prompt.md                    #   Agent system prompt
│   └── sprint.yml                   #   Sprint tracker (agent updates this)
│
├── docs/                            # Project documentation
│   ├── prd.md                       #   Product Requirements Document
│   └── architecture/                #   Mermaid architecture diagrams
│       ├── system-overview.md       #     8 layers + relationships
│       ├── layer-interactions.md    #     How layers call each other
│       ├── memory-architecture.md   #     Shared/Isolated/Hybrid deep dive
│       ├── experiment-lifecycle.md  #     Config → execution → results
│       └── git-workflow.md          #     Branching, commits, review process
│
├── stories_and_epics/               # Sprint backlog (one file per epic)
│   ├── Sprint-1-Epic-1.md          #   through
│   └── Sprint-6-Epic-3.md          #   13 epic files total
│
├── BRANCH_SUMMARY.md               # Agent writes per story (git tracks history)
│
│   ── below created by Forge as it builds ──
│
├── src/ant_coding/                  # Framework source code
├── configs/                         # YAML configs (models, memory, experiments)
├── tests/                           # Test suite
├── scripts/                         # CLI entry points
├── tasks/                           # Benchmark task definitions
├── results/                         # Experiment outputs (gitignored)
├── pyproject.toml                   # Project dependencies
└── .env.example                     # API key template
```

---

## Branching Strategy

```
main          ← Human merges here after review (production-ready)
  └── dev     ← Forge merges stories here (integration branch)
       ├── feature/S1-E1-S01  ← One branch per story
       ├── feature/S1-E1-S02
       └── ...
```

- **Forge** works on feature branches, merges to `dev`
- **Human** reviews `dev`, merges to `main`
- Feature branches are **never deleted** (preserved for review history)

---

## Sprint Plan Summary

| Sprint | Focus | Epics | Stories | Points |
|--------|-------|-------|---------|--------|
| 1 | Project Bootstrap & Configuration | 1 | 6 | 14 |
| 2 | Model & Memory Layers | 2 | 11 | 33 |
| 3 | Tasks & Tools | 2 | 11 | 36 |
| 4 | Orchestration & Runner | 2 | 9 | 36 |
| 5 | Evaluation & Observability | 3 | 12 | 37 |
| 6 | Polish & Integration | 3 | 11 | 41 |
| **Total** | | **11 epics** | **52 stories** | **175 points** |

---

## Quick Reference

```bash
# What's the current sprint status?
cat .agent/sprint.yml | head -20

# What stories are done?
grep "status: done" .agent/sprint.yml | wc -l

# What's ready for review?
grep "status: review" .agent/sprint.yml -B3

# Run all tests
pytest tests/ -v --tb=short

# Lint check
ruff check src/ && ruff format --check src/

# Run a specific experiment
python scripts/run_experiment.py configs/experiments/baseline-sequential.yaml

# Compare two experiments
python scripts/compare_results.py results/exp-a results/exp-b
```
