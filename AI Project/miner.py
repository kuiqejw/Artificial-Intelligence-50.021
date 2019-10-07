import torch
import random
import numpy as np
import torch.nn as nn



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN_CONV(nn.Module):
    def __init__(self, out_length):
        super(DQN_CONV, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3) #conv2d takes in nSamplesxnChannelsxHeightxWidth 
        #N X M X 10 matrix. first  to 8th channel is the different no of mines, ninth has 1 if unknown, 0 otherwise
        self.out = nn.Linear(360, out_length)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out

class DQN_LINEAR(nn.Module):
    def __init__(self, n_states):
        super(DQN_LINEAR, self).__init__()
        # self.conv1 = nn.Conv2d(1,10,3) #starter to init N x M x2 channels

        self.fc1 = nn.Linear(n_states, 288)
        self.fc2 = nn.Linear(288, 220)
        self.fc3 = nn.Linear(220, n_states) #0 to 63 actions that can be taken
    def forward(self,x):
        p = nn.ReLU()
        # print('initial ', x.shape)
        x = p(self.fc1(x))
        # print('fc1', x.shape)
        x = self.fc2(x)
        # print('fc2', x.shape)
        x = self.fc3(x)
        # print('fc3', x.shape)
        return x

class Miner(object):
    def __init__(self,epsilon, memory_capacity, target_replace_iter, batch_size, gamma, conv_bool):
        self.epsilon = epsilon
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size
        self.gamma = gamma
        self.conv = conv_bool

        self.out_length = 64
        if self.conv == 0:
            self.eval_net, self.target_net = DQN_CONV(self.out_length).to(device), DQN_CONV(self.out_length).to(device)
        else:
            self.eval_net, self.target_net = DQN_LINEAR(self.out_length).to(device), DQN_LINEAR(self.out_length).to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, self.out_length * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):

        #for DQN_CONV
        if self.conv == 0:
            x = torch.FloatTensor(x).to(device)
        # # print(x.shape) gets a 8 x8 matrix

            x = torch.unsqueeze(x, 0).to(device)#creates a singleton dimension
        # # print(x.shape) #gets a 1 x 8 x8 
            x = torch.unsqueeze(x, 0).to(device)#creates another singleton dimension
        # print(x.shape) #gets a 1x1x8x8 


        #for DQN_LINEAR
        else:  
            x = x.flatten()
            x = torch.FloatTensor(x).to(device)
        # print(x.shape) gets a 8 x8 matrix
        if random.random() < self.epsilon:
            actions = self.eval_net(x)
            if self.conv == 0:
                action = torch.max(actions, 1)[1].data.numpy()[0]
            else:
                _, action = torch.max(actions, 0)
        else:
            action = random.randint(1, self.out_length)
            action -= 1
        return action

    def store_transition(self, s, a, r, s_):
        s = s.reshape(-1)
        s_ = s_.reshape(-1)
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def optimize_model(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = random.sample(range(self.memory_capacity), self.batch_size)
        b_memory = self.memory[sample_index, :]

        if self.conv == 0:
            b_s = torch.FloatTensor(b_memory[:, :self.out_length]).reshape((self.batch_size, 1, 8, 8)).to(device)
            b_a = torch.LongTensor(b_memory[:, self.out_length:self.out_length+1].astype(int)).to(device)
            b_r = torch.FloatTensor(b_memory[:, self.out_length+1:self.out_length+2]).to(device)
            b_s_ = torch.FloatTensor(b_memory[:, -self.out_length:]).reshape((self.batch_size, 1, 8, 8)).to(device)
            q_eval = self.eval_net(b_s).gather(1, b_a).to(device)
            q_next = self.target_net(b_s_).detach().to(device)
            q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            loss = self.loss_func(q_eval, q_target)
        else:
            b_s = torch.FloatTensor(b_memory[:, :self.out_length]).to(device)
            b_a = torch.LongTensor(b_memory[:, self.out_length:self.out_length+1].astype(int)).to(device)
            b_r = torch.FloatTensor(b_memory[:, self.out_length+1:self.out_length+2]).to(device)
            b_s_ = torch.FloatTensor(b_memory[:, -self.out_length:]).to(device)
            q_eval = self.eval_net(b_s).gather(1, b_a).to(device)
            q_next = self.target_net(b_s_).detach().to(device)
            q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_params(self):
        torch.save(self.eval_net.state_dict(), 'eval.pth')

    def load_params(self, path):
        self.eval_net.load_state_dict(torch.load(path))

# miner = Miner('easy', 0.9, 2000, 100, 32, 0.9)
# x = torch.zeros((1, 8, 8))
# miner.choose_action(x)
# miner.store_transition(x, 1, 1, x)
# miner.learn()