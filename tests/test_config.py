"""Stage-8 config file tests: JSON overrides, guards, CLI wiring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from primordia.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_overrides_apply_on_defaults(tmp_path: Path) -> None:
    cfg = load_config(write_json(tmp_path / "c.json", {
        "metabolic_cost": 0.2,
        "food_growth_rate": 0.01,
    }))
    assert cfg.metabolic_cost == 0.2
    assert cfg.food_growth_rate == 0.01
    # untouched fields keep their defaults
    assert cfg.eat_rate == Config().eat_rate
    assert cfg.max_creatures == Config().max_creatures


def test_base_survives_and_override_wins(tmp_path: Path) -> None:
    base = Config(max_creatures=100, initial_creatures=40)
    path = write_json(tmp_path / "a.json", {"food_growth_rate": 0.02})
    cfg = load_config(path, base)
    assert cfg.initial_creatures == 40  # from base
    assert cfg.food_growth_rate == 0.02  # from JSON

    path2 = write_json(tmp_path / "b.json", {"initial_creatures": 10})
    cfg2 = load_config(path2, base)
    assert cfg2.initial_creatures == 10  # JSON wins over base


def test_food_patch_fields_validation() -> None:
    """Stage-9 patch knobs: bounds are enforced with clear messages."""
    Config(food_patch_amplitude=0.0)  # uniform field (stage-8 behavior)
    Config(food_patch_amplitude=1.0, food_patch_scale=25.0)
    with pytest.raises(ValueError, match="food_patch_amplitude"):
        Config(food_patch_amplitude=1.5)
    with pytest.raises(ValueError, match="food_patch_amplitude"):
        Config(food_patch_amplitude=-0.1)
    with pytest.raises(ValueError, match="food_patch_scale"):
        Config(food_patch_scale=0.0)


def test_turn_cost_validation() -> None:
    """Stage 9.5 steering cost: >= 0, default > 0, max_turn still bounded."""
    Config(turn_cost=0.0)  # free turning stays available for experiments
    assert Config().turn_cost > 0.0
    assert Config().max_turn == 0.3
    with pytest.raises(ValueError, match="turn_cost"):
        Config(turn_cost=-0.01)


def test_environment_validation() -> None:
    """Stage 10 environment knobs: bounds are enforced with clear messages."""
    Config(terrain_amplitude=0.0)  # terrain off = pre-stage-10 world
    Config(season_amp=0.0, day_amp=0.0)  # cycles off
    with pytest.raises(ValueError, match="terrain_amplitude"):
        Config(terrain_amplitude=1.5)
    with pytest.raises(ValueError, match="terrain_scale"):
        Config(terrain_scale=0.0)
    with pytest.raises(ValueError, match="water_move_cost"):
        Config(water_move_cost=0.5)  # water must never be cheaper than land
    with pytest.raises(ValueError, match="season_period"):
        Config(season_period=1)
    with pytest.raises(ValueError, match="day_period"):
        Config(day_period=1)
    with pytest.raises(ValueError, match="season_amp"):
        Config(season_amp=-0.1)
    with pytest.raises(ValueError, match="day_amp"):
        Config(day_amp=1.2)


def test_biology_validation() -> None:
    """Stage 11 biology knobs: bounds are enforced with clear messages."""
    Config(mate_range=0.001, senescence_rate=0.0)  # senescence off allowed
    with pytest.raises(ValueError, match="mate_range"):
        Config(mate_range=0.0)
    with pytest.raises(ValueError, match="senescence_rate"):
        Config(senescence_rate=-0.1)
    with pytest.raises(ValueError, match="carrion_yield"):
        Config(carrion_yield=1.5)
    with pytest.raises(ValueError, match="meat_decay"):
        Config(meat_decay=1.0)  # decay < 1 so meat never flips sign


def test_unknown_field_fails_loudly(tmp_path: Path) -> None:
    path = write_json(tmp_path / "bad.json", {"food_growht_rate": 0.01})
    with pytest.raises(ValueError, match="unknown Config fields"):
        load_config(path)


def test_invalid_value_fails_validation(tmp_path: Path) -> None:
    path = write_json(tmp_path / "bad.json", {"food_growth_rate": 2.0})
    with pytest.raises(ValueError, match="food_growth_rate"):
        load_config(path)


def test_non_object_json_fails(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_config(str(path))


def test_cli_config_and_conflicts(tmp_path: Path) -> None:
    good = write_json(tmp_path / "ok.json", {"metabolic_cost": 0.5})
    proc = subprocess.run(
        [sys.executable, "run.py", "--headless", "--ticks", "1",
         "--config", good, "--save", str(tmp_path / "w.npz")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr

    bad = write_json(tmp_path / "typo.json", {"nope": 1})
    proc = subprocess.run(
        [sys.executable, "run.py", "--headless", "--ticks", "1", "--config", bad],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode != 0 and "unknown Config fields" in proc.stderr

    # --config with --load: the save owns its Config
    proc = subprocess.run(
        [sys.executable, "run.py", "--load", str(tmp_path / "w.npz"),
         "--config", good, "--headless", "--ticks", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 2 and "cannot be combined" in proc.stderr
