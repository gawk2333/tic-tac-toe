import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
from collections import namedtuple, deque
import random
from torch import nn
from torch import optim
from torch.functional import F
import matplotlib.pyplot as plt
import matplotlib
from itertools import count

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.board = torch.zeros((3, 3), dtype=torch.int32)  # Initialize an empty 3x3 board
        self.observation_space = spaces.MultiBinary(9)  # 3x3 board, each cell is binary
        self.action_space = spaces.Discrete(9)  # 9 possible moves (0-8)

    def reset(self):
        self.board = torch.zeros((3, 3), dtype=torch.int32)
        return self.board.flatten()

    def step(self, action, player):
        board = self.board
        row, col = divmod(action, 3)
        if self.board[row, col] == 0:
            self.board[row, col] = player 
        else:
            return torch.tensor(self.board.flatten(), dtype=torch.int32), -1, True  # Invalid move
        done, reward = self.check_game(player)
        return torch.tensor(self.board.flatten(), dtype=torch.int32), reward, done

    # def render(self, mode='human'):
    #     # Visualize the Tic-Tac-Toe board (optional)
    #     pass

    def check_game(self,player):
        board = self.board
        if player == -1:
            board = torch.where(board == 1, -1, torch.where(board == -1, 1, board))
        for i in range(3):
            if all(board[i, :] == player) or all(board[:, i] == player):
                return True, player 
        if torch.all(torch.diag(board) == player) or torch.all(torch.diag(torch.fliplr(board)) == player):
            return True, player
        if torch.all(board != 0):
            return True, 0  # Draw
        return False, 0  # Game ongoing

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        # Convert values to float32 tensors
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()  # Remove the oldest transition
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
  def __init__(self, n_observations, n_actions):
      super(DQN, self).__init__()
      self.layer1 = nn.Linear(n_observations, 128)
      self.layer2 = nn.Linear(128, 128)
      self.layer3 = nn.Linear(128, n_actions)
  def forward(self, x):
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      return self.layer3(x)
  
env = TicTacToeEnv()
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
board = env.reset()
n_observations = len(board)
print("n_observations:",n_observations)
print("n_actions:",n_actions)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).clone().detach()
    
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.max(1)[1]  # Get the action with the highest Q-value
            return action
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.int32)

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(list(batch.state))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_batch = state_batch.view(-1, 9)
    num_classes = 9
    # Convert indices to one-hot encoded vectors
    one_hot = torch.zeros((len(action_batch), num_classes))
    one_hot.scatter_(1, action_batch.unsqueeze(1).long(), 1)
    state_action_values = policy_net(state_batch).gather(1, one_hot.to(torch.int64))
    non_final_next_states = non_final_next_states.view(-1,9)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 4000

for i_episode in range(num_episodes):
    board = env.reset()
    state = board
    print('episode:',i_episode)
    for t in count():
        player = -1 if t % 2 == 0 else 1
        action = select_action(state)
       
        next_state, reward, done = env.step(action.item(),player)
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the policy network)
        optimize_model()
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
torch.save(policy_net.state_dict(), 'policy_net.pth')
torch.save(target_net.state_dict(), 'target_net.pth')