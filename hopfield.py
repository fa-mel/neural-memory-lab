import numpy as np
from numba import jit


@jit(nopython=True)
def _fast_recall_loop(current_state, weights, max_steps, record_energy):
    N = len(current_state)
    energy_history = np.zeros(max_steps)

    initial_field = weights.astype(np.float64) @ current_state.astype(np.float64)
    current_energy = -0.5 * np.dot(current_state.astype(np.float64), initial_field)

    for step in range(max_steps):
        idx = np.random.randint(0, N)
        local_field = 0.0
        for j in range(N):
            local_field += weights[idx, j] * current_state[j]

        old_spin = current_state[idx]
        new_spin = 1.0 if local_field >= 0 else -1.0

        if new_spin != old_spin:
            current_state[idx] = new_spin
            if record_energy:
                col_sum = 0.0
                for k in range(N):
                    col_sum += weights[k, idx] * current_state[k]
                delta_s = new_spin - old_spin
                delta_term = delta_s * (local_field + col_sum)
                current_energy -= 0.5 * delta_term

        if record_energy:
            energy_history[step] = current_energy

    return current_state, energy_history


class HopfieldNetwork:
    def __init__(self, num_neurons, inhibitory_fraction=0.0):
        self.N = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons), dtype=np.float64)
        self.inhibitory_neurons = []

        if inhibitory_fraction > 0:
            num_inhibitory = int(self.N * inhibitory_fraction)
            self.inhibitory_neurons = np.random.choice(
                range(self.N), size=num_inhibitory, replace=False
            )

    def train(self, patterns):
        M = len(patterns)
        if M == 0:
            return
        self.weights = np.sum(
            [np.outer(p, p) for p in patterns], axis=0
        ).astype(np.float64)
        self.weights /= M
        np.fill_diagonal(self.weights, 0)

        if len(self.inhibitory_neurons) > 0:
            self.weights[self.inhibitory_neurons, :] *= -1

    def recall(self, initial_pattern, max_steps=1500, record_energy=True):
        state = initial_pattern.astype(np.float64).copy()
        final_state, history = _fast_recall_loop(
            state, self.weights, int(max_steps), record_energy
        )
        return final_state, history

    def calculate_energy(self, state):
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))
