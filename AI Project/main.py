import torch
import argparse
from miner import Miner
from minesweeper import Minesweeper
import numpy
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=0.9, help='the probability to choose from memories')
parser.add_argument('--memory_capacity', type=int, default=50000, help='the capacity of memories')
parser.add_argument('--target_replace_iter', type=int, default=100, help='the iter to update the target net')
parser.add_argument('--batch_size', type=int, default=16, help='sample amount')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=20000, help='training epoch number')
parser.add_argument('--n_critic', type=int, default=100, help='evaluation point')
parser.add_argument('--test', type=int, default=0, help='whether execute test')
parser.add_argument('--conv', type=int, default=0, help = 'choose between linear and convolution')
opt = parser.parse_args()
print(opt)

miner = Miner(opt.epsilon, opt.memory_capacity, opt.target_replace_iter, opt.batch_size, opt.lr, opt.conv)
print('collecting experience...')


def movingaverage(interval, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

def plot_durations(success,ylabel, name):
    fig = plt.figure(2)
    axes = plt.gca()
    # axes.set_ylim([0, 500])
    plt.clf()
    plt.xlabel('Test Number')
    plt.ylabel(ylabel)
    plt.plot(success)
    print(success)
    z = movingaverage(success,10)
    #chop off the remaining 10
    z = z[:-10]
    z = numpy.concatenate((numpy.zeros(10), z))
    plt.plot(z)
    plt.savefig(name)

if opt.test:
    miner.load_params('eval.pth')
    game = Minesweeper()
    #to be changed upon GUI
    game.action(0)
    s = game.get_state()
    game.show()
    while game.get_status() == 0:
        a = miner.choose_action(s)
        game.action(a)
        game.show()
else:
    win_num = 0
    fail_num = 0
    avg_rewards = []
    success = []
    for epoch in range(opt.n_epochs):
        game = Minesweeper()
        game.action(0)
        s = game.get_state()
        if game.get_status() == 1:
            continue #gets out of the loop if all have been uncovered
        critic_r = 0
        ep_r = 0
        last_r = 0
        # game.show()
        while True:
            a = miner.choose_action(s)
            game.action(a)
            s_ = game.get_state() #returns the board with covered/uncovered/mine 
            status = game.get_status() #-1 if mined, 1 if success, 0 otherwise

            progress = s_ - s
            if status == 1:
                win_num += 1
                r = 1 #reward if won
            elif status == -1:
                fail_num += 1
                r = -1#reward if loss
            elif progress.sum() != 0:
                r = 0.9 #some progress is made
            else:
                r = -0.3 #nothing
            #we wanted to reward YOLO bombing as this is a proven strategy if stuck. 

            miner.store_transition(s, a, r, s_)

            ep_r += r
            if miner.memory_counter > opt.memory_capacity:
                miner.optimize_model()
                if game.get_status() != 0:
                    print('Ep: ', epoch,
                          '| Ep_r: ', round(ep_r, 2))

            if status != 0:
                break

            s = s_.copy()

        critic_r += ep_r
        if (epoch+1) % opt.n_critic == 0:
            print('=====evaluation=====')
            print('Epochs:', epoch)
            print('win number:', win_num)
            print('fail number:', fail_num)
            print('win rate:', win_num / (win_num + fail_num))
            print('total reward:', critic_r)
            avg_rewards.append(critic_r)
            success.append(win_num)
            win_num = 0
            fail_num = 0
            critic_r = 0
    plot_durations(success, 'Accuracy','batch_size16_rmspropdirect.png' )
    plot_durations(avg_rewards, 'Critic Reward','batch_size16_rmspropreward.png' )

    miner.save_params()
