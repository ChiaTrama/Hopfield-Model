import numpy as np

class HopfieldModelnD:
    def __init__(self, patterns, update_method='montecarlo', learning_rule='hebb', R=None, verbose=True):
        """
        Initialize the Hopfield model.
        
        Args:
            patterns (list of np.array): List of patterns to store (e.g. 2D images with values ±1).
            update_method (str): 'synchronous', 'asynchronous' or 'montecarlo'.
            learning_rule (str): 'hebb' (default), 'storkey' or 'local'.
            R (float): Radius for local coupling (needed if learning_rule=='local').
            verbose (bool): If True, print convergence info.
        """
        self.input_shape = patterns[0].shape
        self.dimension = patterns[0].size
        self.update_method = update_method
        self.learning_rule = learning_rule
        self.R = R
        self.verbose = verbose
        self.stored_patterns = [p.flatten() for p in patterns]
        self.J = np.zeros((self.dimension, self.dimension))
        self.train(patterns)
    
    def train(self, patterns):
        if self.learning_rule == 'hebb':
            self.J = np.zeros((self.dimension, self.dimension))
            for pattern in patterns:
                p = pattern.flatten()
                self.J += np.outer(p, p)
            self.J /= self.dimension
            np.fill_diagonal(self.J, 0)
        elif self.learning_rule == 'turkey':
            self.J = np.zeros((self.dimension, self.dimension))
            for pattern in patterns:
                xi = pattern.flatten()
                h = np.dot(self.J, xi)
                delta = (np.outer(xi, xi) - np.outer(xi, h) - np.outer(h, xi)) / self.dimension
                self.J += delta
                np.fill_diagonal(self.J, 0)
        elif self.learning_rule == 'local':
            if self.R is None:
                raise ValueError("For 'local', R must be specified.")
            if len(self.input_shape) != 2:
                raise ValueError("'local' is only implemented for 2D patterns.")
            self.J = np.zeros((self.dimension, self.dimension))
            height, width = self.input_shape
            positions = np.array([(i, j) for i in range(height) for j in range(width)])
            dists = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :])**2, axis=2))
            local_mask = dists <= self.R
            for pattern in patterns:
                xi = pattern.flatten()
                self.J += np.outer(xi, xi) * local_mask
            self.J /= self.dimension
            np.fill_diagonal(self.J, 0)
        else:
            raise ValueError("Invalid learning rule.")
    
    def update(self, state, steps=1, temperature=0.0, alpha=0.0, global_iter=None, schedule='classic'):
        pattern = state.flatten()
        if self.update_method == 'synchronous':
            for _ in range(steps):
                new_pattern = np.sign(np.dot(self.J, pattern))
                new_pattern[new_pattern == 0] = pattern[new_pattern == 0]
                pattern = new_pattern
            return pattern
        elif self.update_method == 'asynchronous':
            for _ in range(steps):
                for _ in range(self.dimension):
                    i = np.random.randint(0, self.dimension)
                    h = np.dot(self.J[i, :], pattern)
                    pattern[i] = np.sign(h) if h != 0 else pattern[i]
            return pattern
        elif self.update_method == 'montecarlo':
            if global_iter is not None:
                if schedule == 'classic':
                    T = temperature / (1 + alpha * global_iter)
                elif schedule == 'exponential':
                    T = temperature * np.exp(-alpha * global_iter)
                elif schedule == 'logarithmic':
                    T = temperature / (1 + alpha * np.log(1+global_iter))
                else:
                    raise ValueError("Schedule not recognized. Use 'classic', 'exponential' or 'logarithmic'.")
                for _ in range(self.dimension):
                    i = np.random.randint(0, self.dimension)
                    h = np.dot(self.J[i, :], pattern)
                    delta_E = 2 * pattern[i] * h
                    if delta_E <= 0:
                        pattern[i] = -pattern[i]
                    else:
                        if T > 0 and np.random.rand() < np.exp(-delta_E / T):
                            pattern[i] = -pattern[i]
                return pattern
            else:
                for t in range(steps):
                    if schedule == 'classic':
                        T = temperature / (1 + alpha * t)
                    elif schedule == 'exponential':
                        T = temperature * np.exp(-alpha * t)
                    else:
                        raise ValueError("Schedule not recognized. Use 'classic' or 'exponential'.")
                    for _ in range(self.dimension):
                        i = np.random.randint(0, self.dimension)
                        h = np.dot(self.J[i, :], pattern)
                        delta_E = 2 * pattern[i] * h
                        if delta_E <= 0:
                            pattern[i] = -pattern[i]
                        else:
                            if T > 0 and np.random.rand() < np.exp(-delta_E / T):
                                pattern[i] = -pattern[i]
                return pattern
        else:
            raise ValueError("Invalid update_method.")
    
    def energy(self, state):
        s = state.flatten()
        return -0.5 * np.dot(s, np.dot(self.J, s))
    
    def storage_limit(self):
        return 0.138 * self.dimension

    def overlap(self, pattern1, pattern2, absolute=False):
        val = np.dot(pattern1.flatten(), pattern2.flatten()) / self.dimension
        return abs(val) if absolute else val
    
    def memory_interference(self):
        P = len(self.stored_patterns)
        if P < 2:
            return 0.0
        total = 0.0
        count = 0
        for mu in range(P):
            for nu in range(mu+1, P):
                ov = np.dot(self.stored_patterns[mu], self.stored_patterns[nu]) / self.dimension
                total += abs(ov)
                count += 1
        return total / count
    
    def correct(self, corrupted_pattern, max_iter=100, convergence_check=1, temperature=0.0, alpha=0.0, schedule='classic'):
        pattern = corrupted_pattern.flatten()
        check = 0
        for i in range(max_iter):
            new_pattern = self.update(pattern, steps=1, temperature=temperature, alpha=alpha, global_iter=i, schedule=schedule)
            if np.array_equal(pattern, new_pattern):
                check += 1
                if check >= convergence_check:
                    break
            else:
                check = 0
            pattern = new_pattern
        if self.verbose:
            print(f"Converged after {i+1} iterations")
        return pattern
    
    def correct_patterns(self, corrupted_patterns, max_iter=100, convergence_check=1, temperature=0.0, alpha=0.0, schedule='classic'):
        corrected_patterns = np.zeros_like(corrupted_patterns)
        n = corrupted_patterns.shape[0]
        for i in range(n):
            corrected = self.correct(corrupted_patterns[i], max_iter, convergence_check, temperature, alpha, schedule)
            corrected_patterns[i] = corrected.reshape(self.input_shape)
        return corrected_patterns