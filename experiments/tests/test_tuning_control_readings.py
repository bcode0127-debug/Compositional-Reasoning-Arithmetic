"""Schema and logic tests for experiments/scripts/tuning_control_readings.py on synthetic runs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tuning_control_readings", REPO_ROOT / "experiments/scripts/tuning_control_readings.py")
tcr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcr)


def write_run(d, enc, lr, wd, acc, train_acc, seed, n_epochs=60, sel=40):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + int(acc * 1000))
    flags = (rng.random(1000) < acc / 100).astype(int).tolist()
    val = [1.0 + i * 0.1 for i in range(n_epochs)]
    val[sel] = acc + 1.0
    val[sel + 1:] = [1.0] * (n_epochs - sel - 1)
    train = [10.0] * n_epochs
    train[sel] = train_acc
    (d / "history.json").write_text(json.dumps({"train_losses": [1.0] * n_epochs, "val_losses": [1.5] * n_epochs,
                                                "train_accuracies": train, "val_accuracies": val, "wall_time_seconds": 100.0}))
    (d / "test_results.json").write_text(json.dumps({
        "test_accuracy": sum(flags) / 10, "n": 1000, "positional_encoding": enc, "epoch": sel,
        "val_accuracy_fingerprint": val[sel], "per_example_correct": flags, "model_type": "transformer", "seed": seed}))
    (d / "environment.json").write_text(json.dumps({"gpu": "test", "git_commit": "x", "learning_rate": lr, "weight_decay": wd,
                                                    "optimizer": "AdamW" if wd > 0 else "Adam"}))
    (d / "ood_results.json").write_text(json.dumps({f: 1.0 for f in tcr.OOD_FILES}))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("tuning")
    accs = {"R1": 8, "E1a": 5, "E1b": 12, "E3a": 8, "R2": 20, "E2a": 14, "E2b": 22, "E3b": 8}
    for c, (enc, lr, wd) in tcr.CELLS.items():
        for k in tcr.SEEDS:
            train = 30.0 if (c == "E1b" and k == 0) else 85.0  # one non-fitting run
            write_run(tcr.run_dir(root, c, k), enc, lr, wd, accs[c], train, k)
    sweep = tmp_path_factory.mktemp("sweep")
    for a, (enc, _) in tcr.DESC_RUNS.items():
        for k in tcr.SEEDS:
            write_run(tcr.run_dir(sweep, a, k), enc, 3e-4, 0.0, 8, 85.0, k)
    lstm = {k: {"label": f"lstm/{k}", "test_accuracy": 35.0, "val_accuracy": 35.0, "selected_epoch": 1, "fits": True, "band": False,
                "diverged": False, "epoch_cap": False, "ood": {f: 5.0 for f in tcr.OOD_FILES},
                "flags": (np.random.default_rng(k).random(1000) < 0.35).astype(float)} for k in tcr.SEEDS}
    cells, desc = tcr.load_cells(root), tcr.load_descriptive(sweep)
    return tcr.build(cells, desc, lstm), cells


def test_readings_schema(synthetic):
    doc, _ = synthetic
    tcr.validate_rule(doc)
    json.dumps(doc)  # serialisable
    assert tcr.md_report(doc).splitlines()[0] == "Tuning control per docs/design/tuning-control.md (pre-registered at TUNE_COMMIT)."


def test_flags_and_eligibility(synthetic):
    doc, cells = synthetic
    assert cells["E1b"][0]["fits_by_rule"]["max_train"] is False and cells["E1b"][1]["fits_by_rule"]["max_train"] is True
    assert doc["cells"]["E1b"]["status"]["n_fitting_seeds"] == 2 and not doc["cells"]["E1b"]["status"]["uninformative"]
    elig = doc["reading1_tuning_effect"]["E1b"]["all_fitting_seeds"]
    assert [s["eligible"] for s in elig["per_seed"].values()] == [False, True, True] and elig["n_eligible"] == 2


def test_selection_is_on_validation_not_test(synthetic):
    doc, _ = synthetic
    assert doc["reading2_best_over_grid"]["rope"]["best_configuration_by_mean_validation"] == "E2b"
    assert doc["reading3_architecture_gap"]["sinusoidal_cell"] == "E1b"


def test_best_configuration_and_evp_cover_all_cells(synthetic):
    doc, _ = synthetic
    sin = doc["reading2_best_over_grid"]["sinusoidal"]
    assert sin["configurations_covered"] == ["E1a", "R1", "E1b", "E3a", "E3b"] and len(sin["expected_validation_performance"]) == 15
    assert len(doc["reading2_best_over_grid"]["rope"]["expected_validation_performance"]) == 9
    d = doc["reading3_architecture_gap"]["descriptive_best_configuration_overall"]
    assert d["label"] == "descriptive" and d["sinusoidal_cell"] == sin["best_configuration_by_mean_validation"]


def test_compare_flags_detects_mismatch():
    rec = {"per_example_correct": [1, 0, 1, 0], "test_accuracy": 50.0}
    assert tcr.compare_flags("x", rec, [1, 0, 1, 0], 50.0)["match"] is True
    bad = tcr.compare_flags("x", rec, [1, 1, 1, 0], 50.0)
    assert bad["match"] is False and bad["n_flag_differences"] == 1 and bad["first_differences"] == [1]


def test_pod_thread_setting_recorded():
    assert tcr.POD_THREADS["omp_num_threads_per_run"] == 3 and tcr.POD_THREADS["nproc_shim"] == 6


def test_uninformative_cell_gets_no_verdict(tmp_path):
    root = tmp_path / "r"
    for c, (enc, lr, wd) in tcr.CELLS.items():
        for k in tcr.SEEDS:
            write_run(tcr.run_dir(root, c, k), enc, lr, wd, 8, 40.0 if c == "E3b" and k < 2 else 85.0, k)
    cells = tcr.under_rule(tcr.load_cells(root), "max_train")
    assert tcr.cell_status(cells["E3b"])["uninformative"] is True


def test_evp_is_monotone_and_starts_at_mean():
    v = [1.0, 2.0, 3.0, 6.0]
    e = tcr.evp(v)
    assert abs(e["1"] - np.mean(v)) < 1e-12 and all(e[str(n + 1)] >= e[str(n)] for n in range(1, 4))


def test_both_fit_rules(tmp_path):
    """Max training accuracy over the run (primary) versus the selected-epoch value (pre-registered)."""
    root = tmp_path / "r"
    for c, (enc, lr, wd) in tcr.CELLS.items():
        for k in tcr.SEEDS:
            write_run(tcr.run_dir(root, c, k), enc, lr, wd, 8, 85.0, k)
    d = tcr.run_dir(root, "R1", 0)
    h = json.loads((d / "history.json").read_text())
    h["train_accuracies"] = [10.0] * 60
    h["train_accuracies"][40] = 75.0   # selected epoch below 80
    h["train_accuracies"][55] = 90.0   # later peak above 80
    (d / "history.json").write_text(json.dumps(h))
    r = tcr.load_cells(root)["R1"][0]
    assert r["selected_epoch"] == 40 and r["train_accuracy_at_selected"] == 75.0 and r["max_train_accuracy"] == 90.0
    assert r["fits_by_rule"] == {"max_train": True, "selected_epoch": False}
    assert r["band_by_rule"] == {"max_train": False, "selected_epoch": True}
    cells = tcr.load_cells(root)
    assert tcr.under_rule(cells, "selected_epoch")["R1"][0]["fits"] is False
    assert tcr.under_rule(cells, "max_train")["R1"][0]["fits"] is True


def test_selected_epoch_is_first_argmax_of_validation(tmp_path):
    d = tmp_path / "run"
    write_run(d, "sinusoidal", 3e-4, 0.0, 8, 85.0, 0)
    h = json.loads((d / "history.json").read_text())
    h["val_accuracies"][50] = h["val_accuracies"][40]  # a tie later in the run: the first index still wins
    (d / "history.json").write_text(json.dumps(h))
    assert tcr.load_run(d, "x")["selected_epoch"] == 40


def test_full_document_has_both_rules_and_validates(synthetic, tmp_path):
    _, cells = synthetic
    lstm = {k: {"label": "l", "test_accuracy": 35.0, "val_accuracy": 35.0, "selected_epoch": 1, "fits": True, "band": False,
                "diverged": False, "epoch_cap": False, "ood": {f: 5.0 for f in tcr.OOD_FILES},
                "flags": (np.random.default_rng(k).random(1000) < 0.35).astype(float)} for k in tcr.SEEDS}
    sweep = tmp_path / "s"
    for a, (enc, _) in tcr.DESC_RUNS.items():
        for k in tcr.SEEDS:
            write_run(tcr.run_dir(sweep, a, k), enc, 3e-4, 0.0, 8, 85.0, k)
    desc = tcr.load_descriptive(sweep)
    full = {"first_line": tcr.FIRST_LINE, "fit_rules": tcr.RULE_TEXT, "primary_rule": "max_train", "pre_registered_rule": "selected_epoch",
            **{r: tcr.build(cells, desc, lstm, None, r) for r in tcr.RULES}}
    tcr.validate(full)
    assert "maximum training accuracy" in tcr.md_report(full["max_train"])


def test_spike_flag_threshold(tmp_path):
    for gap, flagged in ((10.01, True), (10.00, False)):
        d = tmp_path / f"run{gap}"
        write_run(d, "sinusoidal", 3e-4, 0.0, 8, 10.0 + gap, 0)  # final train accuracy is 10.0, the maximum is 10.0 + gap
        r = tcr.load_run(d, "x")
        assert abs(r["max_minus_final"] - gap) < 1e-9 and r["spike_flag"] is flagged


def test_classification_differs_line(tmp_path):
    """E3b seeds 0 and 1 sit at 40% at the selected epoch but peak at 90% later: informative under max_train, uninformative under selected_epoch."""
    root = tmp_path / "r"
    for c, (enc, lr, wd) in tcr.CELLS.items():
        for k in tcr.SEEDS:
            write_run(tcr.run_dir(root, c, k), enc, lr, wd, 8, 85.0, k)
    for k in (0, 1):  # E3b: low value at the selected epoch, a later peak above 80
        d = tcr.run_dir(root, "E3b", k)
        h = json.loads((d / "history.json").read_text())
        h["train_accuracies"][40], h["train_accuracies"][55] = 40.0, 90.0
        (d / "history.json").write_text(json.dumps(h))
    cells = tcr.load_cells(root)
    sweep = tmp_path / "s"
    for a, (enc, _) in tcr.DESC_RUNS.items():
        for k in tcr.SEEDS:
            write_run(tcr.run_dir(sweep, a, k), enc, 3e-4, 0.0, 8, 85.0, k)
    desc = tcr.load_descriptive(sweep)
    lstm = {k: {"label": "l", "test_accuracy": 35.0, "val_accuracy": 35.0, "selected_epoch": 1, "fits": True, "band": False,
                "diverged": False, "epoch_cap": False, "ood": {f: 5.0 for f in tcr.OOD_FILES},
                "flags": (np.random.default_rng(k).random(1000) < 0.35).astype(float)} for k in tcr.SEEDS}
    full = {"first_line": tcr.FIRST_LINE, "fit_rules": tcr.RULE_TEXT, "primary_rule": "max_train", "pre_registered_rule": "selected_epoch",
            **{r: tcr.build(cells, desc, lstm, None, r) for r in tcr.RULES}}
    cls, ver = tcr.differences(full)
    assert cls.startswith("Classification differs between rules: E3b (informative under max_train, uninformative under selected_epoch)")
    assert ver.startswith("Verdict differs between rules:")
    md = tcr.md_full(full)
    assert cls in md and ver in md and md.count("## Cells: fit count and classification under each rule") == 1


def _synthetic_full_inputs(tmp_path, tweak=None):
    root = tmp_path / "r"
    for c, (enc, lr, wd) in tcr.CELLS.items():
        for k in tcr.SEEDS:
            write_run(tcr.run_dir(root, c, k), enc, lr, wd, 8 + (4 if c == "E1b" else 0), 85.0, k)
    if tweak:
        tweak(root)
    sweep = tmp_path / "s"
    for a, (enc, _) in tcr.DESC_RUNS.items():
        for k in tcr.SEEDS:
            write_run(tcr.run_dir(sweep, a, k), enc, 3e-4, 0.0, 8, 85.0, k)
    lstm = {k: {"label": "l", "test_accuracy": 35.0, "val_accuracy": 35.0, "selected_epoch": 1, "fits": True, "band": False,
                "diverged": False, "epoch_cap": False, "ood": {f: 5.0 for f in tcr.OOD_FILES},
                "flags": (np.random.default_rng(k).random(1000) < 0.35).astype(float)} for k in tcr.SEEDS}
    return tcr.load_cells(root), tcr.load_descriptive(sweep), lstm


def test_selection_pool_same_under_both_rules_when_a_grid_cell_is_uninformative_under_one(tmp_path):
    def tweak(root):  # E1b: a low value at the selected epoch for two seeds, a later peak above 80
        for k in (0, 1):
            d = tcr.run_dir(root, "E1b", k)
            h = json.loads((d / "history.json").read_text())
            h["train_accuracies"][40], h["train_accuracies"][55] = 40.0, 90.0
            (d / "history.json").write_text(json.dumps(h))
    cells, desc, lstm = _synthetic_full_inputs(tmp_path, tweak)
    a, b = tcr.under_rule(cells, "max_train"), tcr.under_rule(cells, "selected_epoch")
    assert tcr.cell_status(a["E1b"])["uninformative"] is False and tcr.cell_status(b["E1b"])["uninformative"] is True
    assert tcr.best_by_val(a, "sinusoidal") == tcr.best_by_val(b, "sinusoidal") == "E1b"
    full = {"first_line": tcr.FIRST_LINE, "fit_rules": tcr.RULE_TEXT, "primary_rule": "max_train", "pre_registered_rule": "selected_epoch",
            **{r: tcr.build(cells, desc, lstm, None, r) for r in tcr.RULES}}
    tcr.validate(full)
    assert full["max_train"]["reading3_architecture_gap"]["selection_pool"] == full["selected_epoch"]["reading3_architecture_gap"]["selection_pool"]
    md = tcr.md_full(full)
    assert tcr.SAME_LINE in md and "Selection pool, sinusoidal" in md and "Selection pool, rope" in md


def test_non_fit_kind(tmp_path):
    for n_epochs, expected in ((200, "epoch cap"), (60, "stopped before cap")):
        d = tmp_path / f"run{n_epochs}"
        write_run(d, "sinusoidal", 3e-4, 0.0, 8, 30.0, 0, n_epochs=n_epochs)  # never fits
        r = tcr.under_rule({0: tcr.load_run(d, "x")}, "max_train")[0]
        assert tcr.non_fit_kind(r) == expected
