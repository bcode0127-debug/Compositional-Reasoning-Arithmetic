"""Tuning-control flags (design: docs/design/tuning-control.md): optimizer factory, train_v2 arguments, runner."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import train_v2  # noqa: E402
from utils.trainer import build_optimizer  # noqa: E402

CFG = train_v2.TRAINING_CONFIG_V2


def small_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def test_build_optimizer_defaults_is_plain_adam():
    opt = build_optimizer(small_model(), CFG["learning_rate"]["transformer"])
    assert type(opt) is torch.optim.Adam
    g = opt.param_groups[0]
    assert g["lr"] == CFG["learning_rate"]["transformer"] and g["weight_decay"] == 0


def test_build_optimizer_weight_decay_is_adamw():
    opt = build_optimizer(small_model(), learning_rate=1e-4, weight_decay=0.01)
    assert type(opt) is torch.optim.AdamW
    g = opt.param_groups[0]
    assert g["lr"] == 1e-4 and g["weight_decay"] == 0.01


def test_default_optimizer_step_identical_to_adam():
    x, y = torch.randn(16, 4, generator=torch.Generator().manual_seed(1)), torch.randn(16, 2, generator=torch.Generator().manual_seed(2))
    a, b = small_model(), small_model()
    oa = build_optimizer(a, 3e-4)
    ob = torch.optim.Adam(b.parameters(), lr=3e-4)
    for m, o in ((a, oa), (b, ob)):
        o.zero_grad()
        nn.functional.mse_loss(m(x), y).backward()
        o.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


@pytest.mark.parametrize("model", ["lstm", "transformer"])
def test_train_v2_default_learning_rate_is_config(model):
    args = train_v2.build_parser().parse_args(["--model", model, "--study", "1", "--seed", "0"])
    assert train_v2.optimizer_settings(args) == {"learning_rate": CFG["learning_rate"][model], "weight_decay": 0.0,
                                                 "optimizer": "Adam"}


def test_train_v2_flags_override():
    args = train_v2.build_parser().parse_args(["--model", "transformer", "--study", "1", "--seed", "0",
                                               "--learning-rate", "1e-4", "--weight-decay", "0.01"])
    assert train_v2.optimizer_settings(args) == {"learning_rate": 1e-4, "weight_decay": 0.01, "optimizer": "AdamW"}


def test_train_v2_refuses_negative_weight_decay(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["train_v2.py", "--model", "transformer", "--study", "1", "--seed", "0",
                                      "--weight-decay", "-0.1", "--output-dir", str(tmp_path / "run")])
    with pytest.raises(SystemExit):
        train_v2.main()
    assert not (tmp_path / "run").exists()


def test_run_records_optimizer_settings(tmp_path, monkeypatch):
    seen = {}

    def fake_train_model(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after recording")

    out = tmp_path / "run"
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(train_v2, "train_model", fake_train_model)
    monkeypatch.setattr(sys, "argv", ["train_v2.py", "--model", "transformer", "--study", "1", "--seed", "0",
                                      "--epochs", "1", "--max-train-examples", "8", "--device", "cpu",
                                      "--learning-rate", "1e-4", "--weight-decay", "0.01", "--output-dir", str(out)])
    with pytest.raises(RuntimeError, match="stop after recording"):
        train_v2.main()
    env = json.loads((out / "environment.json").read_text())
    assert (env["learning_rate"], env["weight_decay"], env["optimizer"]) == (1e-4, 0.01, "AdamW")
    assert (seen["learning_rate"], seen["weight_decay"]) == (1e-4, 0.01)


def test_run_tuning_control_dry_run():
    r = subprocess.run(["bash", "scripts/training/run_tuning_control.sh", "--dry-run"], cwd=REPO_ROOT,
                       capture_output=True, text=True, env={**os.environ, "PYTHON": "python3"})
    assert r.returncode == 0
    lines = r.stdout.splitlines()
    assert len(lines) == 18
    cells = [("E1a", "sinusoidal", "1e-4", "0"), ("E1b", "sinusoidal", "1e-3", "0"),
             ("E3a", "sinusoidal", "3e-4", "0.01"), ("E2a", "rope", "1e-4", "0"), ("E2b", "rope", "1e-3", "0"),
             ("E3b", "sinusoidal", "3e-4", "0.1")]
    expected = [f"python3 -u train_v2.py --model transformer --study 1 --seed {s} --epochs 200 --patience 50 "
                f"--positional_encoding {enc} --learning-rate {lr} --weight-decay {wd} "
                f"--output-dir results_v2/tuning_control/{cell}/study1/seed{s}"
                for cell, enc, lr, wd in cells for s in range(3)]
    assert lines == expected


def test_adamw_zero_weight_decay_step_identical_to_adam():
    x = torch.randn(16, 4, generator=torch.Generator().manual_seed(1))
    y = torch.randn(16, 2, generator=torch.Generator().manual_seed(2))
    a, b = small_model(), small_model()
    oa = torch.optim.AdamW(a.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ob = torch.optim.Adam(b.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    for m, o in ((a, oa), (b, ob)):
        o.zero_grad()
        nn.functional.mse_loss(m(x), y).backward()
        o.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
