import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Initialize environment
env = gym.make("CartPole-v1", render_mode="human")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
alpha = 0.001  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Exploration decay
min_epsilon = 0.1  # Minimum exploration rate
episodes = 5000  # Number of episodes
batch_size = 64  # Replay batch size
memory_size = 100000  # Replay memory size
target_update_frequency = 10  # Update target network every n episodes

# Neural network for Q-function approximation
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize networks
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and replay memory
optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
replay_memory = deque(maxlen=memory_size)

# Function to choose an action (epsilon-greedy)
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.argmax(policy_net(state)).item()  # Exploit

# Function to update the policy network
def optimize_model():
    if len(replay_memory) < batch_size:
        return

    batch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Compute current Q-values and target Q-values
    current_q_values = policy_net(states).gather(1, actions)
    max_next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition in replay memory
        replay_memory.append((state, action, reward, next_state, float(done)))

        # Update state and accumulate reward
        state = next_state
        total_reward += reward

        # Optimize the model
        optimize_model()

    # Update epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Update target network
    if episode % target_update_frequency == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode + 1}/{episodes}: Total Reward: {total_reward}")

env.close()
