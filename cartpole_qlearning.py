import gymnasium as gym
import numpy as np

# Initialize the environment
env = gym.make("CartPole-v1", render_mode='human')
n_actions = env.action_space.n

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
min_epsilon = 0.1  # Minimum exploration rate
episodes = 500  # Number of episodes
bins = [10, 10, 10, 10]  # Discretization bins for each state dimension

# Discretization helper functions
def discretize_state(state):
    """Discretize the continuous state into discrete bins."""
    state_lower_bounds = env.observation_space.low
    state_upper_bounds = env.observation_space.high
    state_upper_bounds[1] = 0.5  # Clip velocity to reasonable ranges
    state_lower_bounds[1] = -0.5
    state_upper_bounds[3] = np.radians(50)
    state_lower_bounds[3] = -np.radians(50)
    
    discrete_state = []
    for i in range(len(state)):
        scaled = (state[i] - state_lower_bounds[i]) / (state_upper_bounds[i] - state_lower_bounds[i])
        scaled = np.clip(scaled, 0, 0.999)  # Ensure within binning range
        discrete_state.append(int(scaled * bins[i]))
    return tuple(discrete_state)

# Initialize Q-table with zeros
q_table = np.zeros(bins + [n_actions])

# Define the epsilon-greedy policy
def choose_action(state):
    """Choose an action based on the epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Training loop
for episode in range(episodes):
    state = discretize_state(env.reset()[0])  # Reset environment and discretize state
    done = False
    total_reward = 0

    while not done:
        action = choose_action(state)  # Choose action
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)
        done = terminated or truncated

        # Update Q-table
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error

        # Update state and accumulate reward
        state = next_state
        total_reward += reward

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print(f"Episode {episode + 1}/{episodes}: Total Reward: {total_reward}")

# Close the environment
env.close()

# Save the trained Q-table for future use
np.save("q_table_cartpole.npy", q_table)
print("Training completed and Q-table saved as 'q_table_cartpole.npy'")
