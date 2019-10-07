#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import torch
import torchvision
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# In[2]:


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[7]:


def select_action(state):
    global steps_done # some global variable was made...
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def plot_stuff():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(total_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.savefig("item.jpg")


# In[4]:


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

    
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
#     print(type(batch.next_state))
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# In[5]:



class edited_DQN(nn.Module):

    def __init__(self, outputs):
        super(edited_DQN, self).__init__()
        """self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        """
        #linear_input_size = convw * convh * 32
        self.leaky = nn.LeakyReLU()
        #self.sigmoid = nn.Sigmoid()
        self.head1 = nn.Linear(8, 128)
        self.head2 = nn.Linear(128, 128)
        self.head3 = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
        x = self.head1(x)
        #return self.head(x.view(x.size(0), -1))
        x = self.leaky(x)
        x = self.head2(x)
        x = self.leaky(x)
        return self.head3(x)


# In[9]:


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
# named tuples essentially return a class.
# the new class is called "Transition"
# it has a few fields, i.e state, action, next_state, reward.
#it can be accessed like a tuple, i.e if you have a = Transition(1,2,3,"hello"), and you call a[3] you will get "hello".
#or you could do a.reward, which will also return "hello"
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

#env = gym.make('CartPole-v0').unwrapped
env = gym.make('LunarLander-v2').unwrapped
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion() #turns on interactive mode....

# if gpu is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()


# init_screen = get_screen()
#init_state = env.state
# _, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = edited_DQN(n_actions).to(device)
target_net = edited_DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(),lr=5e-3,weight_decay=1e-20)
memory = ReplayMemory(10000)

steps_done = 0

num_episodes = 2000
total_rewards = []
episode_durations = []

for i_episode in range(num_episodes):
    # Initialize the environment and state
    
#     last_screen = get_screen()
#     current_screen = get_screen()
#     state = current_screen - last_screen
    ##############################################################################
    state = torch.tensor([env.reset()]).to(device)
    total=0
    for t in count():
        
        # Select and perform an action
        action = select_action(torch.tensor(state).view(1,-1))
        next_state, reward, done, _ = env.step(action.item())
        total+=reward
        total = torch.tensor([total])
        reward = torch.tensor([reward], device=device).float()
        
        if done:
            next_state=None
        else:
            next_state = torch.tensor([next_state]).to(device).float()
            
        # Store the transition in memory
        memory.push(state, action, next_state, total)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        optimize_model()
        
        
        if done:
            episode_durations.append(t + 1)
            total_rewards.append(total)
            #plot_durations()
            break
        ##############################################################################
        
    if i_episode%100==0 and i_episode!=0: ## I ADDED THIS TO MAKE IT PRINT EVERY 100...
        print("current episode: ",i_episode)
        
        totes = torch.tensor(total_rewards, dtype=torch.float)
        meanlist = torch.cat((torch.zeros(99),totes.unfold(0, 100, 1).mean(1).view(-1))) # list of all means
        # use the last mean.
        torch.save(policy_net.state_dict(), 'policy-episode-reward{}.pt'.format(meanlist[len(meanlist)-1])) # save this bit
        torch.save(target_net.state_dict(), 'target-episode-reward{}.pt'.format(meanlist[len(meanlist)-1])) # save me too
        # append the scores at the end of them.
        
    if i_episode%100!=0 and i_episode%10==0:
        print("current episode: ",i_episode)
    if i_episode%200==0 and i_episode!=0:
        plot_stuff()
        
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
totes = torch.tensor(total_rewards, dtype=torch.float)
meanlist = torch.cat((torch.zeros(99),totes.unfold(0, 100, 1).mean(1).view(-1)))
    
torch.save(policy_net.state_dict(), 'policy-episode-reward{}.pt'.format(meanlist[len(meanlist)-1]))
torch.save(target_net.state_dict(), 'target-episode-reward{}.pt'.format(meanlist[len(meanlist)-1]))
print('Complete')
env.render()
env.close()
plt.ioff()
plot_stuff()
# plt.show()
plt.savefig('landed.png')

