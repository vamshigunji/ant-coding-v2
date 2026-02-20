"""
Tests for experiment registry: add, update, lineage, validate.
"""

import tempfile
from pathlib import Path


from ant_coding.core.experiment_registry import ExperimentRegistry
from ant_coding.eval.metrics import ExperimentMetrics


def _tmp_registry():
    """Create a registry with a temp file."""
    tmpdir = tempfile.mkdtemp()
    path = str(Path(tmpdir) / "registry.yml")
    return ExperimentRegistry(path)


# ── Add Experiment ──


def test_add_experiment():
    """add_experiment creates entry with planned status."""
    reg = _tmp_registry()
    entry = reg.add_experiment(
        "baseline-claude",
        config_path="configs/baseline.yml",
        hypothesis="Single agent baseline",
    )
    assert entry["id"] == "baseline-claude"
    assert entry["status"] == "planned"
    assert entry["outcome"]["pass_rate"] is None


def test_add_with_parent():
    """add_experiment with parent records lineage."""
    reg = _tmp_registry()
    reg.add_experiment("baseline")
    entry = reg.add_experiment(
        "baseline--add-reviewer",
        parent="baseline",
        variable_changed="architecture: added reviewer agent",
        hypothesis="Reviewer improves quality",
    )
    assert entry["parent"] == "baseline"
    assert entry["variable_changed"] == "architecture: added reviewer agent"


# ── Update Status ──


def test_update_status():
    """update_status transitions status."""
    reg = _tmp_registry()
    reg.add_experiment("exp-1")
    reg.update_status("exp-1", "running")
    assert reg.get_experiment("exp-1")["status"] == "running"


# ── Update Outcome ──


def test_update_outcome():
    """update_outcome populates metrics and sets complete."""
    reg = _tmp_registry()
    reg.add_experiment("exp-1")

    metrics = ExperimentMetrics(
        experiment_id="exp-1",
        total_tasks=10,
        successful_tasks=6,
        failed_tasks=4,
        pass_rate=0.6,
        total_tokens=50000,
        total_cost=3.5,
        cost_per_resolution=0.58,
        useful_token_ratio=0.7,
    )
    reg.update_outcome("exp-1", metrics)

    entry = reg.get_experiment("exp-1")
    assert entry["status"] == "complete"
    assert entry["outcome"]["pass_rate"] == 0.6
    assert entry["outcome"]["total_tokens"] == 50000


def test_update_outcome_infinity():
    """Infinity values stored as None."""
    reg = _tmp_registry()
    reg.add_experiment("exp-1")

    metrics = ExperimentMetrics(
        experiment_id="exp-1",
        cost_per_resolution=float("inf"),
        tokens_per_resolution=float("inf"),
    )
    reg.update_outcome("exp-1", metrics)

    entry = reg.get_experiment("exp-1")
    assert entry["outcome"]["cost_per_resolution"] is None
    assert entry["outcome"]["tokens_per_resolution"] is None


# ── Lineage ──


def test_lineage_chain():
    """get_lineage returns root-to-leaf chain."""
    reg = _tmp_registry()
    reg.add_experiment("root")
    reg.add_experiment("root--add-a", parent="root", variable_changed="add A")
    reg.add_experiment("root--add-a--add-b", parent="root--add-a", variable_changed="add B")

    lineage = reg.get_lineage("root--add-a--add-b")
    assert len(lineage) == 3
    assert lineage[0]["id"] == "root"
    assert lineage[1]["id"] == "root--add-a"
    assert lineage[2]["id"] == "root--add-a--add-b"


def test_lineage_single():
    """Baseline has lineage of just itself."""
    reg = _tmp_registry()
    reg.add_experiment("baseline")
    lineage = reg.get_lineage("baseline")
    assert len(lineage) == 1
    assert lineage[0]["id"] == "baseline"


def test_lineage_not_found():
    """Missing experiment returns empty lineage."""
    reg = _tmp_registry()
    assert reg.get_lineage("nonexistent") == []


# ── Validate ──


def test_validate_missing_variable():
    """Warns when non-baseline experiment has no variable_changed."""
    reg = _tmp_registry()
    reg.add_experiment("child", parent="parent")
    warnings = reg.validate()
    assert any("variable_changed" in w for w in warnings)


def test_validate_stale_planned():
    """Warns about planned experiments older than 7 days."""
    reg = _tmp_registry()
    entry = reg.add_experiment("old-exp")
    # Manually backdate
    entry["date"] = "2020-01-01"
    reg._save()

    # Reload
    reg2 = ExperimentRegistry(str(reg._path))
    warnings = reg2.validate()
    assert any("planned" in w and "old-exp" in w for w in warnings)


def test_validate_clean():
    """No warnings for valid registry."""
    reg = _tmp_registry()
    reg.add_experiment("baseline")
    reg.add_experiment("baseline--test", parent="baseline", variable_changed="test change")
    assert reg.validate() == []


# ── Suggest ID ──


def test_suggest_id():
    """suggest_id produces correct slug."""
    reg = _tmp_registry()
    suggested = reg.suggest_id("baseline-claude", "memory: shared → isolated")
    assert suggested == "baseline-claude--memory-shared-isolated"


def test_suggest_id_cleanup():
    """suggest_id handles special characters."""
    reg = _tmp_registry()
    suggested = reg.suggest_id("parent", "add reviewer agent")
    assert suggested == "parent--add-reviewer-agent"


# ── Persistence ──


def test_persistence():
    """Registry persists across reloads."""
    tmpdir = tempfile.mkdtemp()
    path = str(Path(tmpdir) / "registry.yml")

    reg1 = ExperimentRegistry(path)
    reg1.add_experiment("exp-1")

    reg2 = ExperimentRegistry(path)
    assert reg2.get_experiment("exp-1") is not None
    assert reg2.get_experiment("exp-1")["id"] == "exp-1"


def test_list_experiments():
    """list_experiments returns all entries."""
    reg = _tmp_registry()
    reg.add_experiment("a")
    reg.add_experiment("b")
    assert len(reg.list_experiments()) == 2
