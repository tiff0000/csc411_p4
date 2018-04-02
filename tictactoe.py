from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done


class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=32, output_size=9):
        super(Policy, self).__init__()
        # TODO
        self.linear_f1 = nn.Linear(input_size, hidden_size)
        self.linear_f2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # TODO
        h = F.relu(self.linear_f1(x))
        out = F.softmax(self.linear_f2(h))
        return out


def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob


def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    # TODO
    l = len(rewards)
    rewards = np.array(rewards)
    gammas = np.array([gamma ** (i) for i in range(l)])

    G = []
    for i in range(l):
        G.append(sum(rewards[i:] * gammas[:l - i]))
    return G

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step


def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
        Environment.STATUS_VALID_MOVE: 5,
        Environment.STATUS_INVALID_MOVE: -50,
        Environment.STATUS_WIN: 100,
        Environment.STATUS_TIE: -10,
        Environment.STATUS_LOSE: -100
    }[status]

    # part6.1
    # return {
    #     Environment.STATUS_VALID_MOVE: 1,
    #     Environment.STATUS_INVALID_MOVE: -25,
    #     Environment.STATUS_WIN: 50,
    #     Environment.STATUS_TIE: 0,
    #     Environment.STATUS_LOSE: -2
    # }[status]


def train(policy, env, gamma=0.9, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    
    episode = []
    avg_return = []

    win_rates, loss_rates, tie_rates = [], [], []
    
    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            episode.append(i_episode)
            avg_return.append(running_reward/log_interval)

            won_rate, lost_rate, tie_rate = play_against_random(policy, env)
            win_rates.append(won_rate)
            loss_rates.append(lost_rate)
            tie_rates.append(tie_rate)

            invalid_moves = get_invalid_moves(policy, env)

            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            running_reward = 0
            print('Win rate{}\tLose rate{}\tTie rate{}\tInvalid Moves{}'.format(
                won_rate,
                lost_rate,
                tie_rate,
                invalid_moves
            ))

        if i_episode % log_interval == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0:  # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        if i_episode == 50000:
            fig = plt.figure()
            # plt.plot(episode, avg_return)
            # plt.xlabel("Episodes")
            # plt.ylabel("Average return")
            # plt.title("Training curve of the tictactoe model")
            # plt.savefig("part5b_hidden32.png")
            # plt.savefig("part5b_hidden128.png")
            # plt.savefig("part5b_hidden16.png")

            # part 6
            # plt.figure()
            plt.plot(episode, win_rates, label="Win rate")
            plt.plot(episode, loss_rates, label="Loss rate")
            plt.plot(episode, tie_rates, label="Tie rate")
            plt.xlabel("Episodes")
            plt.ylabel("Win/lose/tie rates")
            plt.title("Changes of win/lose/tie rates throughout the training")
            plt.legend()
            plt.savefig("part6.1.png")

        if i_episode > 50000:
            return 0


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


# part 5d
def play_against_random(policy, env):
    """
    Play 100 games against random and return the rate of wins, losses and ties of the agent.
    """

    won_num = 0
    lost_num = 0
    tie_num = 0

    num_games = 100
    for i in range(num_games):
        state = env.reset()
        done = False
        while not done:
            action, probability = select_action(policy, state)
            state, status, done = env.play_against_random(action)

        won_num += status == Environment.STATUS_WIN
        lost_num += status == Environment.STATUS_LOSE
        tie_num += status == Environment.STATUS_TIE
    print("wons: {}\tlosses: {}\tties:{}".format(
        won_num, lost_num, tie_num
    ))
    won_num /= 100.0
    lost_num /= 100.0
    tie_num /= 100.0

    return won_num, lost_num, tie_num


def get_invalid_moves(policy, env):
    """
    Return the number of invalid moves
    """
    invalid = 0
    num_games = 100

    for i in range(num_games):
        state = env.reset()
        done = False
        while not done:
            action, probability = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            invalid += status == env.STATUS_INVALID_MOVE
    return invalid


if __name__ == '__main__':
    import sys
    policy = Policy()
    env = Environment()
    
    # part 2
    # env.render()
    # state = np.array([1,0,1,2,1,0,1,0,1])
    # state = torch.from_numpy(state).long().unsqueeze(0)
    # state = torch.zeros(3,9).scatter_(0, state, 1).view(1, 27)
    # print(state)
    
    # part 5a plot training curve
    train(policy, env)

    # part 5b. Try with different sizes of hidden units.
    # hidden_units = [16, 32, 128]
    # for h in hidden_units:
    #     env = Environment()
    #     policy = Policy(hidden_size=h)
    #     train(policy, env)
    #     print(play_against_random(policy, env))

    # part 5d
    print(play_against_random(policy, env))


