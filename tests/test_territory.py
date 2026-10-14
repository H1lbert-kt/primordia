"""Stage-25 territory: banked houses, guarded raids, mark comfort, store input."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from primordia import step
from primordia.config import Config, load_config
from primordia.genome import get_numba_rng_state, set_numba_rng_state
from primordia.io import FORMAT_VERSION, load_world, save_world
from primordia.sensors import input_groups
from primordia.stats import StatsRecorder
from primordia.world import World


def make_config(**overrides) -> Config:
    """Tiny quiet world: flat terrain, no cycles, cheap metabolism.

    ``initial_food=0`` keeps the eat phase free of accidental forage so the
    territory flows are the only energy moving; ``food_capacity`` stays at
    its default because perceive divides by it.
    """
    base = dict(
        max_creatures=8,
        initial_creatures=2,
        terrain_amplitude=0.0,
        season_amp=0.0,
        day_amp=0.0,
        senescence_rate=0.0,
        houses_enabled=True,
        initial_food=0.0,
    )
    base.update(overrides)
    return Config(**base)


def _force_house(
    w: World,
    x: float,
    y: float,
    owner_id: int,
    u: float = 1.0,
    deposit: float = 0.0,
) -> int:
    """Materialize a structure directly (bypassing the build gate)."""
    s = int(w.struct_free_count) - 1
    assert s >= 0
    w.struct_free_count = s
    w.struct_pos[s, 0] = x
    w.struct_pos[s, 1] = y
    w.struct_type[s] = 1
    w.struct_store[s] = u
    w.struct_deposit[s] = deposit
    w.struct_owner[s] = owner_id
    w.struct_alive[s] = True
    w.rebuild_struct_fields()
    return s


def _place(w: World, slot: int, x: float, y: float, energy: float,
           gate: float | None = None) -> None:
    w.pos[slot, 0] = x
    w.pos[slot, 1] = y
    w.energy[slot] = energy
    w.vel[slot] = 0.0
    if gate is not None:
        w.actions[slot, 2] = gate


def _bank_pair(**cfg_kw) -> tuple[World, int, int, int]:
    """Owner on its banked house, foreign creature a few units away."""
    knobs = dict(deposit_rate=0.5, raid_rate=0.3, guard_range=3.0)
    knobs.update(cfg_kw)
    cfg = make_config(**knobs)
    w = World(cfg, seed=0)
    a, b = np.flatnonzero(w.alive)
    _place(w, a, 500.0, 500.0, 150.0)
    _place(w, b, 520.0, 520.0, 50.0)
    s = _force_house(w, 500.0, 500.0, int(w.creature_id[a]), deposit=40.0)
    step.phase_rebuild_counts(w)
    return w, int(a), int(b), s


# --- config -----------------------------------------------------------------


def test_layout_and_default_knobs() -> None:
    cfg = Config()
    assert cfg.sensor_input_dim == 38
    assert cfg.brain_params == 448
    assert cfg.genome_size == 454
    assert cfg.plastic_params == 8 * 38 + 8 * 8
    assert cfg.deposit_rate == pytest.approx(0.5)
    assert cfg.raid_rate == pytest.approx(0.3)
    assert cfg.guard_range == pytest.approx(3.0)
    assert cfg.comfort == pytest.approx(0.3)
    assert cfg.mark_threshold == pytest.approx(1.0)
    assert cfg.deposit_threshold_frac == pytest.approx(1.1)
    assert cfg.store_scale == pytest.approx(100.0)
    label, group = input_groups(cfg)[-1]
    assert label == "store"
    assert group == slice(3 * cfg.n_rays + 16, 3 * cfg.n_rays + 17)


def test_config_validation() -> None:
    for bad, match in (
        (dict(deposit_rate=-0.1), "deposit_rate"),
        (dict(raid_rate=-0.1), "raid_rate"),
        (dict(guard_range=-1.0), "guard_range"),
        (dict(comfort=1.5), "comfort"),
        (dict(comfort=-0.1), "comfort"),
        (dict(mark_threshold=-1.0), "mark_threshold"),
        (dict(deposit_threshold_frac=0.0), "deposit_threshold_frac"),
        (dict(store_scale=0.0), "store_scale"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(Config(), **bad)


# --- deposits ---------------------------------------------------------------


def test_deposit_moves_energy_and_conserves() -> None:
    w, a, _b, s = _bank_pair(deposit_threshold_frac=1.0)
    e0 = float(w.energy[a])
    d0 = float(w.struct_deposit[s])
    step.phase_eat(w)
    assert w.deposited > 0.0
    assert float(w.struct_deposit[s]) > d0
    assert float(w.energy[a]) < e0
    assert float(w.energy[a]) + float(w.struct_deposit[s]) == pytest.approx(
        e0 + d0, abs=1e-4
    )


def test_deposit_floors_at_reproduce_threshold() -> None:
    w, a, _b, s = _bank_pair(deposit_threshold_frac=1.0, deposit_rate=100.0)
    step.phase_eat(w)
    assert float(w.energy[a]) == pytest.approx(
        w.config.reproduce_threshold, abs=1e-4
    )


def test_deposit_needs_own_complete_house() -> None:
    # foreign house: the stranger never banks; move the owner away so only
    # the foreigner is in play
    w, a, b, s = _bank_pair(deposit_threshold_frac=1.0)
    _place(w, a, 100.0, 100.0, 150.0)
    step.phase_rebuild_counts(w)
    e0 = float(w.energy[b])
    step.phase_eat(w)
    assert float(w.struct_deposit[s]) == pytest.approx(40.0)
    assert float(w.energy[b]) == e0  # foreign: untouched by the bank
    # incomplete house (u < 1) never accepts deposits
    cfg = make_config(deposit_threshold_frac=1.0)
    w2 = World(cfg, seed=0)
    a = int(np.flatnonzero(w2.alive)[0])
    _place(w2, a, 500.0, 500.0, 150.0)
    s2 = _force_house(w2, 500.0, 500.0, int(w2.creature_id[a]), u=0.5)
    step.phase_rebuild_counts(w2)
    step.phase_eat(w2)
    assert float(w2.struct_deposit[s2]) == 0.0
    assert w2.deposited == 0.0


def test_disabled_off_is_neutral() -> None:
    # rates at zero: no new energy ever moves, and a pre-seeded bank below
    # the mark threshold leaves the ground unmarked (the mark is a physical
    # reading of stored energy, not a rate-gated rule)
    cfg = make_config(
        deposit_rate=0.0, raid_rate=0.0, comfort=0.0,
        deposit_threshold_frac=1.0,
    )
    w = World(cfg, seed=3)
    a = int(np.flatnonzero(w.alive)[0])
    _place(w, a, 500.0, 500.0, 500.0)
    s = _force_house(w, 500.0, 500.0, int(w.creature_id[a]), deposit=0.5)
    step.phase_rebuild_counts(w)
    for _ in range(20):
        step.phase_eat(w)
        step.phase_age_and_metabolize(w)
    assert w.deposited == 0.0 and w.raided == 0.0
    assert float(w.struct_deposit[s]) == pytest.approx(0.5)
    assert np.all(w.mark == 0.0)  # 0.5 < mark_threshold=1.0


# --- withdraw / raid --------------------------------------------------------


def test_withdraw_pays_from_own_bank() -> None:
    w, a, _b, s = _bank_pair()
    _place(w, a, 500.0, 500.0, 50.0, gate=1.0)  # under threshold, mouth open
    e0 = float(w.energy[a])
    d0 = float(w.struct_deposit[s])
    step.phase_eat(w)
    assert float(w.struct_deposit[s]) == pytest.approx(
        d0 - w.config.deposit_rate, abs=1e-4
    )
    assert float(w.energy[a]) == pytest.approx(
        e0 + w.config.deposit_rate, abs=1e-4
    )
    assert w.deposited > 0.0  # the withdraw is booked as a bank flow


def test_raid_blocked_while_owner_nearby() -> None:
    w, a, b, s = _bank_pair()
    # owner alive at the house (guard_range 3): raider standing on it too
    _place(w, b, 500.5, 500.0, 50.0, gate=1.0)
    step.phase_eat(w)
    assert w.raided == 0.0
    assert float(w.struct_deposit[s]) > 0.0
    assert int(w.raid_of[b]) == -1


def test_raid_drains_unguarded_house() -> None:
    w, a, b, s = _bank_pair()
    # move the owner far away, raider on the house with an open mouth
    _place(w, a, 100.0, 100.0, 150.0)
    _place(w, b, 500.0, 500.0, 50.0, gate=1.0)
    step.phase_rebuild_counts(w)  # owner moved: refresh the CSR guard scan
    e0 = float(w.energy[b])
    d0 = float(w.struct_deposit[s])
    step.phase_eat(w)
    assert w.raided > 0.0
    assert float(w.struct_deposit[s]) == pytest.approx(
        d0 - w.config.raid_rate, abs=1e-4
    )
    assert float(w.energy[b]) == pytest.approx(
        e0 + w.config.raid_rate, abs=1e-4
    )
    assert int(w.raid_of[b]) == s
    assert int(w.raid_of[a]) == -1


def test_full_flows_conserve_energy() -> None:
    """Deposit + withdraw + raid together move energy, never create it."""
    cfg = make_config(
        deposit_rate=0.5, raid_rate=0.3, guard_range=0.0,
        deposit_threshold_frac=1.0, metabolic_cost=0.0, move_cost=0.0,
        turn_cost=0.0, max_age=10_000, food_capacity=0.0, initial_food=0.0,
    )
    w = World(cfg, seed=5)
    a, b = np.flatnonzero(w.alive)
    # owner banks and withdraws at its own house far away...
    _place(w, a, 100.0, 100.0, 200.0, gate=1.0)
    _force_house(w, 100.0, 100.0, int(w.creature_id[a]), deposit=10.0)
    # ...while the foreigner raids an unguarded bank under its own feet
    _place(w, b, 500.0, 500.0, 10.0, gate=1.0)
    s = _force_house(w, 500.0, 500.0, int(w.creature_id[a]), deposit=30.0)
    step.phase_rebuild_counts(w)
    total0 = float(w.energy.sum() + w.struct_deposit[s] + w.food.sum()
                   + w.meat.sum())
    for _ in range(50):
        step.phase_eat(w)
    total1 = float(w.energy.sum() + w.struct_deposit[s] + w.food.sum()
                   + w.meat.sum())
    assert total1 == pytest.approx(total0, abs=1e-3)
    assert w.deposited > 0.0 and w.raided > 0.0


# --- store input ------------------------------------------------------------


def test_store_input_signed_and_saturating() -> None:
    cfg = make_config(store_scale=50.0)
    w = World(cfg, seed=0)
    a, b = np.flatnonzero(w.alive)
    s = _force_house(w, 500.0, 500.0, int(w.creature_id[a]), deposit=80.0)
    _place(w, a, 500.0, 500.0, 50.0)
    _place(w, b, 500.0, 500.0, 50.0)  # foreigner on the same cell
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    store_in = 3 * cfg.n_rays + 16
    # mine: +min(80, 50)/50 = +1.0 (saturated); foreigner: -1.0
    assert float(w.sensor_buf[a, store_in]) == pytest.approx(1.0)
    assert float(w.sensor_buf[b, store_in]) == pytest.approx(-1.0)
    # empty ground reads exact 0
    _place(w, b, 800.0, 800.0, 50.0)
    step.phase_perceive(w)
    assert float(w.sensor_buf[b, store_in]) == 0.0
    # a banked fire (type 0) is never a bank: flip and re-stamp
    w.struct_type[s] = 0
    w.rebuild_struct_fields()
    _place(w, a, 500.0, 500.0, 50.0)
    step.phase_perceive(w)
    assert float(w.sensor_buf[a, store_in]) == 0.0


# --- mark + comfort ---------------------------------------------------------


def test_mark_ring_and_own_comfort_discount() -> None:
    cfg = make_config(comfort=0.3, mark_threshold=1.0, house_radius=5.0)
    w = World(cfg, seed=0)
    a, b = np.flatnonzero(w.alive)
    _place(w, a, 500.0, 500.0, 50.0)
    _place(w, b, 560.0, 560.0, 50.0)
    _force_house(w, 500.0, 500.0, int(w.creature_id[a]), deposit=5.0)
    # mark stamped by rebuild: owner's cell is marked, distant cell is not
    assert float(w.mark[100, 100]) == 1.0
    assert int(w.mark_owner[100, 100]) == int(w.creature_id[a])
    assert float(w.mark[112, 112]) == 0.0
    e_a0, e_b0 = float(w.energy[a]), float(w.energy[b])
    step.phase_age_and_metabolize(w)
    cost = cfg.metabolic_cost
    # owner on own mark pays (1 - comfort) x cost; the stranger pays full
    assert float(w.energy[a]) == pytest.approx(
        e_a0 - cost * (1.0 - cfg.comfort), abs=1e-4
    )
    assert float(w.energy[b]) == pytest.approx(e_b0 - cost, abs=1e-4)


def test_foreign_mark_gives_no_comfort() -> None:
    cfg = make_config(comfort=0.3)
    w = World(cfg, seed=0)
    _a, b = np.flatnonzero(w.alive)
    _place(w, b, 500.0, 500.0, 50.0)
    # someone ELSE owns the banked house under the stranger's feet
    other = int(w.creature_id[b]) + 99
    _force_house(w, 500.0, 500.0, other, deposit=5.0)
    e0 = float(w.energy[b])
    step.phase_age_and_metabolize(w)
    assert float(w.energy[b]) == pytest.approx(e0 - cfg.metabolic_cost)


def test_comfort_zero_skips_lookup() -> None:
    cfg = make_config(comfort=0.0)
    w = World(cfg, seed=0)
    a = int(np.flatnonzero(w.alive)[0])
    _place(w, a, 500.0, 500.0, 50.0)
    _force_house(w, 500.0, 500.0, int(w.creature_id[a]), deposit=5.0)
    e0 = float(w.energy[a])
    step.phase_age_and_metabolize(w)
    assert float(w.energy[a]) == pytest.approx(e0 - cfg.metabolic_cost)


# --- determinism / save / stats ---------------------------------------------


def _territory_config() -> Config:
    return Config(
        max_creatures=32,
        initial_creatures=16,
        houses_enabled=True,
        build_house_bias=0.5,  # some houses actually get built
        deposit_rate=0.5,
        raid_rate=0.3,
        comfort=0.3,
        deposit_threshold_frac=1.0,
        mate_range=1e6,
        metabolic_cost=0.02,
        max_age=200,
        day_period=7,
        season_period=13,
    )


def test_determinism_two_worlds() -> None:
    cfg = _territory_config()
    a = World(cfg, seed=11)
    for _ in range(150):
        step.advance(a)
    b = World(cfg, seed=11)
    for _ in range(150):
        step.advance(b)
    for name in ("pos", "energy", "genome", "struct_deposit", "mark",
                 "sensor_buf", "struct_owner"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.deposited == b.deposited and a.raided == b.raided


def test_save_load_v12_continuation(tmp_path: Path) -> None:
    cfg = _territory_config()
    a = World(cfg, seed=42)
    for _ in range(60):
        step.advance(a)
    path = tmp_path / "mid.npz"
    save_world(a, str(path))
    saved_stream = get_numba_rng_state()

    b = load_world(str(path))
    for _ in range(40):
        step.advance(b)
    set_numba_rng_state(saved_stream)
    for _ in range(40):
        step.advance(a)

    assert a.tick == b.tick == 100
    for name in ("pos", "vel", "angle", "energy", "age", "genome",
                 "hidden_prev", "dw", "struct_deposit"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.births == b.births
    assert a.deposited == b.deposited and a.raided == b.raided


def test_v11_save_is_rejected_clearly(tmp_path: Path) -> None:
    cfg = make_config()
    w = World(cfg, seed=1)
    path = tmp_path / "old.npz"
    save_world(w, str(path))
    with np.load(path, allow_pickle=False) as f:
        payload = {name: f[name] for name in f.files}
    payload["format_version"] = np.int32(FORMAT_VERSION - 1)
    old = tmp_path / "v11.npz"
    np.savez(old, **payload)
    with pytest.raises(ValueError, match="format"):
        load_world(str(old))


def test_stats_columns_record_flows() -> None:
    cfg = _territory_config()
    w = World(cfg, seed=7)
    rec = StatsRecorder()
    rec.record(w)
    for col in ("deposited", "raided", "energy_in_stores", "mark_cells"):
        assert col in rec.to_dict()
    for _ in range(40):
        step.advance(w)
        rec.record(w)
    data = rec.to_dict()
    assert data["deposited"].dtype == np.float32
    assert data["mark_cells"].dtype == np.int32
    assert float(data["energy_in_stores"][-1]) >= 0.0
    assert int(data["mark_cells"][-1]) >= 0


# --- render guards ----------------------------------------------------------


def test_raid_flash_lifecycle() -> None:
    """raid_of -> EventTracker.raid_lines: age 0 while raiding, fade, drop."""
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    from dataclasses import replace

    from primordia.render.draw import EventTracker
    from primordia.render.settings import Settings

    w, a, b, s = _bank_pair()
    _place(w, a, 100.0, 100.0, 150.0)  # owner far away: house unguarded
    _place(w, b, 500.0, 500.0, 50.0, gate=1.0)
    step.phase_rebuild_counts(w)
    settings = replace(Settings(), flash_duration=5)
    tracker = EventTracker(w)

    step.phase_eat(w)
    tracker.update(w, settings)
    assert len(tracker.raid_lines) == 1
    line = tracker.raid_lines[0]
    assert (line.x1, line.y1) == (500.0, 500.0)  # the raider stands on the bank
    assert (line.x2, line.y2) == tuple(float(v) for v in w.struct_pos[s])
    assert line.age == 0

    # raids stop (mouth closed): the next eat pass clears raid_of, and the
    # quiet ticks fade the line (same lifecycle as donations)
    _place(w, b, 500.0, 500.0, 50.0, gate=0.0)
    for _ in range(settings.flash_duration - 1):
        step.phase_eat(w)  # clears raid_of at the top of each pass
        tracker.update(w, settings)
    assert tracker.raid_lines  # still fading
    assert tracker.raid_lines[0].age > 0
    step.phase_eat(w)
    tracker.update(w, settings)
    assert not tracker.raid_lines
    assert not tracker._raids


def test_territory_overlay_does_not_mutate_world() -> None:
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    from primordia.render.app import _rebuild_territory_overlay
    from primordia.render.settings import Settings

    cfg = make_config(deposit_threshold_frac=1.0)
    w = World(cfg, seed=5)
    a = int(np.flatnonzero(w.alive)[0])
    _place(w, a, 500.0, 500.0, 50.0)
    _force_house(w, 500.0, 500.0, int(w.creature_id[a]), deposit=5.0)
    gh, gw = w.food.shape
    rgb = np.zeros((gh, gw, 3), dtype=np.float32)
    alpha = np.zeros((gh, gw, 1), dtype=np.float32)
    mark, owner = w.mark.copy(), w.mark_owner.copy()
    pos, genome, energy = w.pos.copy(), w.genome.copy(), w.energy.copy()
    _rebuild_territory_overlay(w, rgb, alpha, Settings())
    assert np.array_equal(w.mark, mark)
    assert np.array_equal(w.mark_owner, owner)
    assert np.array_equal(w.pos, pos)
    assert np.array_equal(w.genome, genome)
    assert np.array_equal(w.energy, energy)
    # and it actually filled something: the marked cell is tinted
    assert float(alpha[100, 100, 0]) > 0.0


def test_deposit_bars_skip_fires_and_incomplete() -> None:
    """Only complete banked houses draw a bar; fires/candidates stay blank."""
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    from primordia.render.camera import Camera
    from primordia.render.draw import stamp_deposit_bars
    from primordia.render.settings import Settings

    settings = Settings()
    w = World(make_config(), seed=5)
    a = int(np.flatnonzero(w.alive)[0])
    _place(w, a, 500.0, 500.0, 50.0)
    # a banked fire (type 0) and an incomplete house must both be skipped
    fire = _force_house(w, 500.0, 500.0, int(w.creature_id[a]), deposit=9.0)
    w.struct_type[fire] = 0
    w.rebuild_struct_fields()
    surf = pygame.Surface(
        (settings.viewport_width, settings.viewport_height), pygame.SRCALPHA
    )
    camera = Camera(
        settings, w.config.width, w.config.height,
        float(w.cell_w), float(w.cell_h),
        gw=w.food.shape[1], gh=w.food.shape[0],
    )
    stamp_deposit_bars(surf, w, camera, settings)
    arr = np.frombuffer(
        pygame.image.tobytes(surf, "RGBA"), dtype=np.uint8
    ).reshape(settings.viewport_height, settings.viewport_width, 4)
    assert int(arr[..., 3].sum()) == 0  # nothing drawn

    # completing the house makes the bar appear
    w.struct_type[fire] = 1
    w.struct_store[fire] = 1.0
    w.rebuild_struct_fields()
    stamp_deposit_bars(surf, w, camera, settings)
    arr = np.frombuffer(
        pygame.image.tobytes(surf, "RGBA"), dtype=np.uint8
    ).reshape(settings.viewport_height, settings.viewport_width, 4)
    assert int(arr[..., 3].sum()) > 0


def test_render_raids_uses_raid_color() -> None:
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    from primordia.render.camera import Camera
    from primordia.render.draw import Transfer, render_raids
    from primordia.render.settings import Settings

    settings = Settings()
    surf = pygame.Surface(
        (settings.viewport_width, settings.viewport_height), pygame.SRCALPHA
    )
    surf.fill((0, 0, 0, 0))
    w = World(make_config(), seed=5)
    camera = Camera(
        settings, w.config.width, w.config.height,
        float(w.cell_w), float(w.cell_h),
        gw=w.food.shape[1], gh=w.food.shape[0],
    )
    cx, cy = w.config.width / 2.0, w.config.height / 2.0
    render_raids(surf, [Transfer(cx, cy, cx + 4.0, cy, age=0)], camera, settings)
    mx, my = camera.world_to_screen(np.array([[cx + 2.0, cy]], dtype=np.float32))
    arr = np.frombuffer(
        pygame.image.tobytes(surf, "RGBA"), dtype=np.uint8
    ).reshape(settings.viewport_height, settings.viewport_width, 4)
    mid = arr[int(my[0]), int(mx[0])]
    assert mid[3] > 0
    assert tuple(mid[:3]) == settings.flash_raid
