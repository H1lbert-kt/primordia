"""Vectorized MLP forward pass and output-to-motion conversion.

Genome layout per creature (float32 row, offsets derived from ``Config``):

    W1 (hidden, inputs)  [0, hid*in)          # hidden-major: inner loop contiguous
    b1                   [hid*in, +hid)
    W2 (outputs, hidden) [+hid, +hid + out*hid)
    b2                   [+out, +out + out)
    body traits          [brain_params, +4)   # max_speed, size, vision_range, diet

``think`` stores raw output logits in ``actions``; ``apply_actions`` maps them
to motion (cinematic: vel = dir(angle) * max_speed * accel) and converts the
eat logit into a gate in ``(0, 1)`` that ``step`` applies when foraging.
No random numbers are drawn in this module.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def think(
    sensor_buf: np.ndarray,
    hidden_buf: np.ndarray,
    actions: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    n_in: np.int32,
    n_hid: np.int32,
    n_out: np.int32,
) -> None:
    """One forward pass per living creature: inputs -> hidden -> raw logits."""
    w1_off = 0
    b1_off = n_hid * n_in
    w2_off = b1_off + n_hid
    b2_off = w2_off + n_hid * n_out

    for i in prange(alive.size):
        if not alive[i]:
            continue
        for j in range(n_hid):
            acc = genome[i, b1_off + j]
            row = w1_off + j * n_in
            for k in range(n_in):
                acc = acc + sensor_buf[i, k] * genome[i, row + k]
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
    alive: np.ndarray,
    genome: np.ndarray,
    max_turn: np.float32,
    traits_off: np.int32,
) -> None:
    """Map logits to steering: angle += turn, vel = dir * speed, gate = sigmoid."""
    tau = np.float32(2.0 * np.pi)
    for i in prange(alive.size):
        if not alive[i]:
            continue
        accel = np.tanh(actions[i, 0])
        turn = np.tanh(actions[i, 1])
        # scaled sigmoid: (0, 1), equals 0.5 at logit 0 (cheaper than exp)
        actions[i, 2] = np.float32(0.5) * (np.float32(1.0) + np.tanh(actions[i, 2]))

        a = angle[i] + turn * max_turn
        while a >= tau:
            a = a - tau
        while a < 0.0:
            a = a + tau
        angle[i] = a

        speed = accel * genome[i, traits_off]
        c = np.cos(a)
        s = np.sin(a)
        vel[i, 0] = c * speed
        vel[i, 1] = s * speed
