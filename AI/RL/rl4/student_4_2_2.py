#!/usr/bin/env python3
# rewards: [golden_fish, jellyfish_1, jellyfish_2, ... , step]
rewards = [100, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -10]

# Q learning learning rate
alpha = 0.6

# Q learning discount rate
gamma = 0.8

# Epsilon initial
epsilon_initial = 1

# Epsilon final
epsilon_final = 0.1

# Annealing timesteps
annealing_timesteps = 5000

# threshold
threshold = 1e-6
