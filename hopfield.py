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
        patterns_arr = np.array(patterns, dtype=np.float64)  # (M, N)
        self.weights = (patterns_arr.T @ patterns_arr) / M
        np.fill_diagonal(self.weights, 0.0)

        if len(self.inhibitory_neurons) > 0:
            self.weights[self.inhibitory_neurons, :] *= -1

    def recall(self, initial_pattern, max_steps=1500, record_energy=True):
        state = np.array(initial_pattern, dtype=np.float64).copy()
        energy_history = np.zeros(max_steps)

        if record_energy:
            energy_history[0] = self.calculate_energy(state)

        for step in range(max_steps):
            idx = np.random.randint(0, self.N)
            local_field = self.weights[idx] @ state
            new_spin = 1.0 if local_field >= 0 else -1.0

            if new_spin != state[idx]:
                state[idx] = new_spin

            if record_energy:
                energy_history[step] = self.calculate_energy(state)

        return state, energy_history

    def calculate_energy(self, state):
        return -0.5 * float(state @ self.weights @ state)

    def calculate_energy(self, state):
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))
