from environment import stock
from DQN import DQN, seed_torch
import matplotlib.pyplot as plt
import pandas as pd
from GRU import GRU
import torch


def game_step(observation, step=None, train=True, show_log=False):
    action = RL.choose_action(observation, train)
    observation_, reward, done = env.step(action, show_log=show_log)
    RL.store_transition(observation, action, reward, observation_)
    if step and step > 200 and step % 5 == 0:
        RL.learn()
    return observation_, done


def run(max_round: int):
    step = 0
    for episode in range(max_round):
        observation = env.reset()
        while True:
            observation, done = game_step(observation, step=step)
            if done:
                break
            step += 1
        if env.total_profit > env.best_profit:
            env.best_profit = env.total_profit
            env.draw()
        print('epoch:%d, total_profit:%.3f' % (episode+1, env.total_profit))


def BackTest(env: stock, show_log=True):
    observation = env.reset()
    while True:
        observation, done = game_step(
            observation, train=False, show_log=show_log)
        if done:
            break
    if env.total_profit > env.best_profit:
        env.best_profit = env.total_profit
        env.draw()
    print('total_profit:%.3f' % (env.total_profit))
    return env


if __name__ == "__main__":
    seed_torch()
    max_round = 300
    file_path = 'data/BCHAIN-MKPRU.csv'
    df = pd.read_csv(file_path)
    gru = torch.load('gru_bit.pth')
    env = stock(df, gru, alpha=0.01, init_money=1000.0)
    RL = DQN(env.n_actions, env.n_features, lr=0.01, reward_decay=0.9,
             e_greedy=0.9, replace_target_iter=200, memory_size=5000, batch_size=512)
    run(max_round)
    env = BackTest(env, show_log=False)
    print(env.best_profit)
