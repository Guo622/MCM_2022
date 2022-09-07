import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GRU import GRU
import torch


class stock:
    def __init__(self, df: pd.DataFrame, gru: GRU, alpha, init_money=1000.0, window_size: int = 7) -> None:
        self.n_actions = 3
        self.n_features = window_size+2
        self.trend = df[df.columns[1]].values
        self.mean = self.trend.mean()
        self.std = self.trend.std()
        self.df = df
        self.init_money = init_money
        self.window_size = window_size
        self.half_window = window_size//2
        self.buy_rate = alpha
        self.sell_rate = alpha
        self.best_profit = 0.0
        self.gru = gru

    def get_state(self, t):
        window_size = self.window_size
        d = t-window_size+1
        if d < 0:
            block = []
            for i in range(-d):
                block.append(self.trend[0])
            for i in range(t+1):
                block.append(self.trend[i])
        else:
            block = self.trend[d:t+1]
        x = torch.tensor(block).float().view(1, 7, 1).cuda()
        pred = self.gru((x-self.mean)/self.std)
        pred = torch.squeeze(pred)
        pred = [x.item()*self.std+self.mean for x in pred.cpu()]
        block = np.hstack((block, pred))

        res = []
        for i in range(window_size+3-1):
            res.append((block[i+1]-block[i])/(block[i]+0.0001))
        return np.array(res)

    def reset(self):
        self.hold_money = self.init_money
        self.buy_num = 0.0
        self.hold_num = 0.0
        self.stock_value = 0.0
        self.maket_value = 0.0
        self.last_value = self.init_money
        self.total_profit = 0.0
        self.t = self.window_size//2
        self.reward = 0.0
        self.states_sell = []
        self.states_buy = []
        self.states_hold = []
        self.profit_rate_account = []
        self.profit_rate_stock = []
        return self.get_state(self.t)

    def buy(self):
        self.buy_num = self.hold_money/2/self.trend[self.t]
        tmp_money = self.trend[self.t]*self.buy_num
        service_change = tmp_money*self.buy_rate
        self.hold_num += self.buy_num
        self.hold_money = self.hold_money - tmp_money - service_change
        self.states_buy.append(self.t)

    def sell(self):
        tmp_money = self.hold_num*self.trend[self.t]*0.8
        service_change = tmp_money*self.sell_rate
        self.hold_money = self.hold_money+tmp_money-service_change
        self.hold_num = self.hold_num*0.2
        self.states_sell.append(self.t)

    def step(self, action, show_log=True):
        if action == 1 and self.t < (len(self.trend)-self.half_window) and self.hold_money > 1e-10:
            self.buy()
            if show_log:
                print('day:%d, buy price:%f, buy num:%.3f, hold num:%.3f, hold money:%.3f' %
                      (self.t, self.trend[self.t], self.buy_num, self.hold_num, self.hold_money))
        elif action == 2 and self.hold_num > 1e-10:
            self.sell()
            if show_log:
                print('day:%d, sell price:%f, hold num:%.3f, hold money:%.3f'
                      % (self.t, self.trend[self.t], self.hold_num, self.hold_money))
        else:
            self.states_hold.append(self.t)
            if show_log:
                print('day:%d, hold num:%.3f hold money:%.3f' %
                      (self.t, self.hold_num, self.hold_money))
        self.stock_value = self.trend[self.t]*self.hold_num
        self.maket_value = self.stock_value+self.hold_money
        self.total_profit = self.maket_value-self.init_money

        reward = (self.trend[self.t+1]-self.trend[self.t])/self.trend[self.t]
        if np.abs(reward) <= 0.015:
            reward = reward*0.2
        elif np.abs(reward) <= 0.03:
            reward = reward*0.7
        elif np.abs(reward) >= 0.05:
            if reward < 0:
                reward = (reward+0.05)*0.1-0.05
            else:
                reward = (reward-0.05)*0.1+0.05

        if self.hold_num > 1e-10 or action == 2:
            self.reward = reward
            if action == 2:
                self.reward = -self.reward
        else:
            self.reward = -self.reward*0.1
        self.last_value = self.maket_value
        self.profit_rate_account.append(
            (self.maket_value-self.init_money)/self.init_money)
        self.profit_rate_stock.append(
            (self.trend[self.t]-self.trend[0])/self.trend[0])
        done = False
        self.t = self.t+1
        if self.t == len(self.trend)-2:
            done = True
        state = self.get_state(self.t)
        reward = self.reward
        return state, reward, done

    def get_info(self):
        return self.states_sell, self.states_buy, self.states_hold, self.profit_rate_account, self.profit_rate_stock

    def draw(self, save_name1='a.png', save_name2='b.png'):
        states_sell, states_buy, states_hold, profit_rate_account, profit_rate_stock = self.get_info()
        invest = profit_rate_account[-1]
        total_gains = self.total_profit
        close = self.trend
        fig = plt.figure(figsize=(15, 5))
        plt.plot(close, color='b', lw=2.)
        plt.plot(close, 'v', markersize=4, color='r',
                 label='selling signal', markevery=states_sell)
        plt.plot(close, '^', markersize=4, color='g',
                 label='buying signal', markevery=states_buy)
        plt.plot(close, '_', markersize=6, color='gray',
                 label='holding signal', markevery=states_hold)
        plt.title('total gains %f' %
                  (total_gains))
        plt.legend()
        plt.savefig(save_name1)
        plt.close()

        fig = plt.figure(figsize=(15, 5))
        plt.plot(profit_rate_account, label='my account')
        plt.plot(profit_rate_stock, label='stock')
        plt.legend()
        plt.savefig(save_name2)
        plt.close()
