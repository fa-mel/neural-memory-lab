"""
Hopfield Network with optional Dale's Principle inhibition.

Key improvement over the original: a single `recall` method that tracks
energy incrementally (O(N) per step, not O(N²)) and optionally records
frame snapshots + 2-pattern trajectory in one pass.
"""
import numpy as np


class HopfieldNetwork:
    def __init__(self, num_neurons: int, inhibitory_fraction: float = 0.0):
        self.N = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons), dtype=np.float64)
        self.inhibitory_indices = np.array([], dtype=int)

        if inhibitory_fraction > 0:
            n_inh = int(self.N * inhibitory_fraction)
            self.inhibitory_indices = np.random.choice(
                self.N, size=n_inh, replace=False
            )

    # ── Training (Hebbian) ────────────────────────────────────────────────
    def train(self, patterns):
        M = len(patterns)
        if M == 0:
            return
        P = np.array(patterns, dtype=np.float64)
        self.weights = (P.T @ P) / M
        np.fill_diagonal(self.weights, 0.0)
        if len(self.inhibitory_indices) > 0:
            self.weights[self.inhibitory_indices, :] *= -1

    # ── Unified recall ────────────────────────────────────────────────────
    def recall(self, initial_pattern, max_steps=1500,
               snapshot_every=0, trajectory_patterns=None, seed=None):
        """
        Single recall pass that returns everything in one dict.

        Parameters
        ----------
        initial_pattern : array-like
            Starting state (784-vector).
        max_steps : int
            Number of asynchronous single-spin updates.
        snapshot_every : int
            If > 0, store a copy of the state every N steps (for animation).
        trajectory_patterns : tuple(array, array) or None
            (pattern_a, pattern_b) — if given, record overlap with each
            every 20 steps for 2D state-space plots.
        seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        dict with keys:
            state        — final state vector
            energy       — np.array of energy at every step
            snapshots    — list of state vectors (only if snapshot_every > 0)
            traj_a       — np.array of overlaps with pattern_a
            traj_b       — np.array of overlaps with pattern_b
        """
        if seed is not None:
            np.random.seed(seed)

        state = np.ascontiguousarray(initial_pattern, dtype=np.float64).flatten()
        W = self.weights
        N = self.N

        # Incremental energy tracking: O(N) per step instead of O(N²)
        energy_val = float(-0.5 * state @ W @ state)
        energy = np.empty(max_steps, dtype=np.float64)

        snapshots = [state.copy()] if snapshot_every > 0 else []

        track_traj = trajectory_patterns is not None
        traj_a, traj_b = [], []
        if track_traj:
            pat_a, pat_b = trajectory_patterns
            traj_a.append(float(np.dot(state, pat_a) / N))
            traj_b.append(float(np.dot(state, pat_b) / N))

        for step in range(max_steps):
            idx = int(np.random.randint(0, N))
            h = float(W[idx].dot(state))          # local field — O(N)
            old_spin = state[idx]
            new_spin = 1.0 if h >= 0.0 else -1.0

            if new_spin != old_spin:
                # ΔH = -(s_new - s_old) · h_i  (exact for w_ii = 0)
                energy_val += -(new_spin - old_spin) * h
                state[idx] = new_spin

            energy[step] = energy_val

            if snapshot_every > 0 and (step + 1) % snapshot_every == 0:
                snapshots.append(state.copy())

            if track_traj and step % 20 == 0:
                traj_a.append(float(np.dot(state, pat_a) / N))
                traj_b.append(float(np.dot(state, pat_b) / N))

        result = {"state": state, "energy": energy}
        if snapshot_every > 0:
            result["snapshots"] = snapshots
        if track_traj:
            result["traj_a"] = np.array(traj_a)
            result["traj_b"] = np.array(traj_b)
        return result

    def energy(self, state):
        s = np.asarray(state, dtype=np.float64).flatten()
        return float(-0.5 * s @ self.weights @ s)
