import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def seed_torch(seed=2022):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class dqn(nn.Module):
    def __init__(self, input_shape, n_actions) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(self, n_actions, n_features, lr=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=4000, batch_size=32, e_greedy_decay=0.001, e_greedy_min=0.05) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = torch.tensor(reward_decay, dtype=torch.float32).to(device)
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_decay = e_greedy_decay
        self.epsilon = self.epsilon_max
        self.epsilon_min = e_greedy_min
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        self.net = dqn(n_features, n_actions).to(device)
        self.tgt_net = dqn(n_features, n_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(device)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, train=True):
        observation = [observation[np.newaxis, :]]
        observation = torch.tensor(observation, dtype=torch.float32).to(device)
        if np.random.uniform() < self.epsilon and train:
            action = np.random.randint(0, self.n_actions)
        else:
            action_value = self.net(observation).detach().cpu().squeeze(0)
            action = np.argmax(action_value)

        return action

    def learn(self):
        self.net.train()
        self.tgt_net.train()
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.tgt_net.load_state_dict(self.net.state_dict())
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        s_ = torch.tensor(
            batch_memory[:, -self.n_features:], dtype=torch.float32).to(device)
        s = torch.tensor(
            batch_memory[:, :self.n_features], dtype=torch.float32).to(device)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # action
        reward = torch.tensor(
            batch_memory[:, self.n_features+1], dtype=torch.float32).to(device)
        q_next = self.tgt_net(s_)
        q_eval = self.net(s)
        # q_next = q_next.cpu()
        # q_eval = q_eval.cpu()
        q_target = q_eval.clone()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        max_, _ = torch.max(q_next, dim=1)
        q_target[batch_index, eval_act_index] = reward+self.gamma*max_
        self.optimizer.zero_grad()
        loss = self.criterion(q_eval, q_target)
        loss.backward()
        self.optimizer.step()
        self.epsilon = self.epsilon - \
            self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        self.learn_step_counter += 1
