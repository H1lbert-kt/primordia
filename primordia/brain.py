"""Vectorized MLP forward pass and output-to-motion conversion.

Genome layout per creature (float32 row, offsets derived from ``Config``):

    W1 (hidden, inputs)    [0, hid*in)          # hidden-major: inner loop contiguous
    b1                     [hid*in, +hid)
    W2 (outputs, hidden)   [+hid, +hid + out*hid)
    b2                     [+out, +out + out)
    W_rec (hidden, hidden) [+out, +hid*hid)     # Elman recurrence
    body traits            [brain_params, +6)   # max_speed, size, vision_range,
                                                # diet, tag_hue, plasticity_eta

``think`` is an Elman network: each hidden unit also reads the previous
tick's hidden state (``hidden_prev``) through ``W_rec``, then the new state
is committed back — short-term memory without any new randomness. Since
stage 21 the forward reads *effective* synapses ``genome + dw`` for ``W1``
and ``W_rec`` when ``use_dw`` is true (the Oja delta lives in its own
scratch, never in the genome), and the commit of ``hidden_prev`` moved to
``step.phase_plasticity`` — the update rule needs the pre-commit hidden
state as the recurrent presynaptic side, so it runs between the forward and
the commit (dead slots still keep their stale row until respawn zeroes it).
With ``use_dw`` false the branch is skipped entirely and the pass is
bit-identical to the pre-stage-21 baseline.
``think`` stores raw output logits in ``actions``; ``apply_actions`` maps them
to motion (cinematic: vel = dir(angle) * max_speed * accel), converts the
eat, emit, donate, gather, build_fire and build_house logits into gates in
``(0, 1)`` (stages 13/16/17 added outputs 3-7, all decoded with the same
scaled sigmoid), and drains ``turn_cost`` energy per radian turned (steering
is a physical cost, stage 9.5). Because the gate slots are decoded in place,
a completed tick leaves ``actions[i, 2:]`` holding *gates*, not logits —
consumers that display or average them must not sigmoid again.
No random numbers are drawn in this module.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def think(
    sensor_buf: np.ndarray,
    hidden_buf: np.ndarray,
    hidden_prev: np.ndarray,
    actions: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    dw: np.ndarray,
    use_dw: bool,
    n_in: np.int32,
    n_hid: np.int32,
    n_out: np.int32,
) -> None:
    """One recurrent forward pass: inputs + hidden_prev -> hidden -> logits.

    ``use_dw`` selects effective synapses ``genome + dw`` for ``W1`` and
    ``W_rec`` (stage 21); false skips the branch entirely so the pass is
    bit-identical to the baseline. The hidden state is *not* committed
    here — ``step.phase_plasticity`` commits it after the Oja update, which
    needs the pre-commit state as the recurrent presynaptic side.
    """
    w1_off = 0
    b1_off = n_hid * n_in
    w2_off = b1_off + n_hid
    b2_off = w2_off + n_hid * n_out
    wrec_off = b2_off + n_out
    dw_w1 = 0  # dw layout: W1 block, then W_rec block (no biases, no W2)
    dw_rec = n_hid * n_in

    for i in prange(alive.size):
        if not alive[i]:
            continue
        for j in range(n_hid):
            acc = genome[i, b1_off + j]
            row = w1_off + j * n_in
            drow = dw_w1 + j * n_in
            if use_dw:
                for k in range(n_in):
                    acc = acc + sensor_buf[i, k] * (
                        genome[i, row + k] + dw[i, drow + k]
                    )
            else:
                for k in range(n_in):
                    acc = acc + sensor_buf[i, k] * genome[i, row + k]
            rec = wrec_off + j * n_hid
            drec = dw_rec + j * n_hid
            if use_dw:
                for h in range(n_hid):
                    acc = acc + hidden_prev[i, h] * (
                        genome[i, rec + h] + dw[i, drec + h]
                    )
            else:
                for h in range(n_hid):
                    acc = acc + hidden_prev[i, h] * genome[i, rec + h]
            hidden_buf[i, j] = np.tanh(acc)
        for o in range(n_out):
            acc = genome[i, b2_off + o]
            row = w2_off + o * n_hid
            for j in range(n_hid):
                acc = acc + hidden_buf[i, j] * genome[i, row + j]
            actions[i, o] = acc


@njit(cache=True, parallel=True)
def apply_actions(
    actions: np.ndarray,
    angle: np.ndarray,
    vel: np.ndarray,
    energy: np.ndarray,
    age: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    max_turn: np.float32,
    turn_cost: np.float32,
    traits_off: np.int32,
    senescence_rate: np.float32,
) -> None:
    """Map logits to steering: angle += turn, vel = dir * speed, gates = sigmoid.

    Turning drains ``turn_cost * |turn| * max_turn`` energy (radians actually
    rotated), so perpetual spinning starves — a physical law, never a fitness
    term (stage 9.5). Senescence (stage 11): effective top speed shrinks by
    the wear factor ``w = exp(senescence_rate * age)`` — old bodies move
    slower; rate 0 restores stage-10 speed exactly. The eat gate (output 2),
    emit gate (output 3, stage 13), donate gate (output 4, stage 16) and the
    gather/fire/house gates (outputs 5-7, stage 17) are all mapped with the
    same scaled sigmoid into ``(0, 1)``; step decides what each one spends.
    The three stage-17 gates are *decisions*, not rates: step attempts the
    action when the decoded gate exceeds 0.5 (raw logit positive).
    """
    tau = np.float32(2.0 * np.pi)
    for i in prange(alive.size):
        if not alive[i]:
            continue
        accel = np.tanh(actions[i, 0])
        turn = np.tanh(actions[i, 1])
        # scaled sigmoid: (0, 1), equals 0.5 at logit 0 (cheaper than exp)
        actions[i, 2] = np.float32(0.5) * (np.float32(1.0) + np.tanh(actions[i, 2]))
        # emit gate (stage 13): same transform as the eat gate; step's
        # emit_signal phase reads actions[i, 3] as the deposit fraction
        actions[i, 3] = np.float32(0.5) * (np.float32(1.0) + np.tanh(actions[i, 3]))
        # donate gate (stage 16): same transform; step's donate phase reads
        # actions[i, 4] as the gift fraction of donate_rate
        actions[i, 4] = np.float32(0.5) * (np.float32(1.0) + np.tanh(actions[i, 4]))
        # gather gate (stage 17): fraction of gather_rate harvested per tick
        actions[i, 5] = np.float32(0.5) * (np.float32(1.0) + np.tanh(actions[i, 5]))
        # build gates (stage 17): step acts when the decoded value > 0.5
        actions[i, 6] = np.float32(0.5) * (np.float32(1.0) + np.tanh(actions[i, 6]))
        actions[i, 7] = np.float32(0.5) * (np.float32(1.0) + np.tanh(actions[i, 7]))

        a = angle[i] + turn * max_turn
        while a >= tau:
            a = a - tau
        while a < 0.0:
            a = a + tau
        angle[i] = a
        energy[i] = energy[i] - turn_cost * abs(turn) * max_turn

        speed = accel * genome[i, traits_off]
        if senescence_rate > 0.0:
            speed = speed / np.exp(senescence_rate * np.float32(age[i]))
        c = np.cos(a)
        s = np.sin(a)
        vel[i, 0] = c * speed
        vel[i, 1] = s * speed


@njit(cache=True, parallel=True)
def plasticity(
    sensor_buf: np.ndarray,
    hidden_buf: np.ndarray,
    hidden_prev: np.ndarray,
    energy: np.ndarray,
    e_last_plastic: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    dw: np.ndarray,
    eta_fixed: np.float32,
    evolvable: bool,
    eta_off: np.int32,
    credit: bool,
    credit_strength: np.float32,
    interval_ok: bool,
    decay: np.float32,
    clip: np.float32,
    cost: np.float32,
    n_in: np.int32,
    n_hid: np.int32,
    n_out: np.int32,
) -> None:
    """Local Oja update on ``dw`` plus the hidden-state commit (stage 21).

    The rule writes only into the ``dw`` scratch — the genome is never
    modified inside a life (Baldwin, not Lamarck) — so the effective
    synapses are ``W_eff = genome + dw``:

        dW = eta * (x * h - h^2 * W_eff)

    The ``-h^2 * W_eff`` term acts only while the postsynaptic neuron is
    active (h ~ 0 => step ~ 0: it is not idle-synapse atrophy); it keeps
    ``W_eff`` around the Oja equilibrium instead of letting it run away.
    On top of the rule the delta decays by ``decay`` every application —
    unlearned adaptation slides back to the evolved blueprint — and is
    clamped to ``+/-clip``, so evolved weights are never erased. ``W1`` is
    updated from the current sensors, ``W_rec`` from ``hidden_prev``
    (still the *pre-commit* state: the commit below runs last), and ``W2``
    is never plastic.

    Learning is not free: the energy drain ``cost * |dw moved|`` is debited
    here, so with the evolvable eta (``evolvable=True``, spawned at 0)
    selection decides whether turning plasticity on earns its price. The
    optional ``credit`` modulation scales eta by ``1 + strength *
    tanh(energy - e_last_plastic)`` — LOCAL CREDIT ASSIGNMENT, the ablation
    variant only; it lets an individual's own payoff shape its learning,
    which is a reward-like signal and deliberately off by default.

    Every alive slot is also committed into ``hidden_prev`` (the job think
    had before stage 21) and ``e_last_plastic`` is refreshed, so the phase
    is a no-op for learning — never for the commit — on ticks where the
    interval gate or eta says no. No random numbers are drawn.
    """
    b1_off = n_hid * n_in
    w2_off = b1_off + n_hid
    b2_off = w2_off + n_hid * n_out
    wrec_off = b2_off + n_out
    dw_rec = n_hid * n_in  # dw layout: W1 block, then W_rec block

    for i in prange(alive.size):
        if not alive[i]:
            continue
        eta = np.float32(0.0)
        if interval_ok:
            if evolvable:
                eta = genome[i, eta_off]
                if eta < 0.0:
                    eta = 0.0
            else:
                eta = eta_fixed
            if credit:
                d_e = energy[i] - e_last_plastic[i]
                eta = eta * (np.float32(1.0) + credit_strength * np.tanh(d_e))
                if eta < 0.0:
                    eta = 0.0
        moved = np.float32(0.0)
        if eta > 0.0:
            keep = np.float32(1.0) - decay
            # W1: presynaptic x = this tick's sensors, postsynaptic h = this
            # tick's hidden (both just written by think)
            for j in range(n_hid):
                h = hidden_buf[i, j]
                hh = h * h
                row = j * n_in
                for k in range(n_in):
                    d = row + k
                    w = genome[i, d] + dw[i, d]
                    delta = eta * (sensor_buf[i, k] * h - hh * w)
                    new = (dw[i, d] + delta) * keep
                    if new > clip:
                        new = clip
                    elif new < -clip:
                        new = -clip
                    moved = moved + np.abs(new - dw[i, d])
                    dw[i, d] = new
            # W_rec: presynaptic = pre-commit hidden_prev (the commit below
            # runs after this block — that is why it moved out of think)
            for j in range(n_hid):
                h = hidden_buf[i, j]
                hh = h * h
                drow = dw_rec + j * n_hid
                grow = wrec_off + j * n_hid
                for k in range(n_hid):
                    d = drow + k
                    w = genome[i, grow + k] + dw[i, d]
                    delta = eta * (hidden_prev[i, k] * h - hh * w)
                    new = (dw[i, d] + delta) * keep
                    if new > clip:
                        new = clip
                    elif new < -clip:
                        new = -clip
                    moved = moved + np.abs(new - dw[i, d])
                    dw[i, d] = new
            if cost > 0.0 and moved > 0.0:
                energy[i] = energy[i] - cost * moved
        # hidden-state commit (think's job before stage 21): every alive
        # slot, every tick, exactly what think used to do
        for j in range(n_hid):
            hidden_prev[i, j] = hidden_buf[i, j]
        e_last_plastic[i] = energy[i]
