import numpy as np


class HopfieldNetwork:
    def __init__(self, num_neurons, inhibitory_fraction=0.0):
        self.N = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons), dtype=np.float64)
        self.inhibitory_neurons = np.array([], dtype=int)

        if inhibitory_fraction > 0:
            num_inhibitory = int(self.N * inhibitory_fraction)
            self.inhibitory_neurons = np.random.choice(
                self.N, size=num_inhibitory, replace=False
            )

    def train(self, patterns):
        M = len(patterns)
        if M == 0:
            return
        patterns_arr = np.array(patterns, dtype=np.float64)
        self.weights = (patterns_arr.T @ patterns_arr) / M
        np.fill_diagonal(self.weights, 0.0)
        if len(self.inhibitory_neurons) > 0:
            self.weights[self.inhibitory_neurons, :] *= -1

    def recall(self, initial_pattern, max_steps=1500, record_energy=True):
        state = np.ascontiguousarray(initial_pattern, dtype=np.float64).flatten()
        W = self.weights
        energy_history = np.zeros(max_steps)

        for step in range(max_steps):
            idx = int(np.random.randint(0, self.N))
            local_field = float(W[idx].dot(state))
            new_spin = 1.0 if local_field >= 0.0 else -1.0
            state[idx] = new_spin
            if record_energy:
                energy_history[step] = self.calculate_energy(state)

        return state, energy_history

    def recall_with_trajectory(self, initial_pattern, pattern_a, pattern_b,
                                max_steps=1500, sample_every=20):
        """Recall while recording 2D state-space trajectory (overlap with two patterns)."""
        state = np.ascontiguousarray(initial_pattern, dtype=np.float64).flatten()
        W = self.weights
        N = self.N
        traj_a, traj_b = [], []
        energy_history = np.zeros(max_steps)

        for step in range(max_steps):
            idx = int(np.random.randint(0, N))
            local_field = float(W[idx].dot(state))
            state[idx] = 1.0 if local_field >= 0.0 else -1.0
            energy_history[step] = self.calculate_energy(state)
            if step % sample_every == 0:
                traj_a.append(float(np.dot(state, pattern_a) / N))
                traj_b.append(float(np.dot(state, pattern_b) / N))

        return state, energy_history, np.array(traj_a), np.array(traj_b)

    def calculate_energy(self, state):
        return float(-0.5 * state @ self.weights @ state)
