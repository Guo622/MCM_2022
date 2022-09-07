import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class stock:
    def __init__(self, df: pd.DataFrame, alpha1=0.02, alpha2=0.01, init_money=1000.0, window_size: int = 7) -> None:
        self.n_actions = 3
        self.n_features = window_size
        self.trend1 = df['bit'].values
        self.trend2 = df['gold'].values
        self.last_gold = self.trend2[1]
        self.df = df
        self.init_money = init_money
        self.window_size = window_size
        self.half_window = window_size//2
        self.buy_rate1 = self.sell_rate1 = alpha1
        self.buy_rate2 = self.sell_rate2 = alpha2
        self.best_profit = 0.0

    def get_state(self, t):
        window_size = self.window_size+1
        d = t-window_size+1
        block1, block2 = [], []
        if d < 0:
            for i in range(-d):
                block1.append(self.trend1[0])
                block2.append(self.trend2[1])
            for i in range(t+1):
                block1.append(self.trend1[i])
                if self.trend2[i] < 0:
                    block2.append(self.last_gold)
                else:
                    block2.append(self.trend2[i])
                    self.last_gold = self.trend2[i]
        else:
            block1 = self.trend1[d:t+1]
            for i in range(d, t+1):
                if self.trend2[i] < 0:
                    block2.append(self.last_gold)
                else:
                    block2.append(self.trend2[i])
                    self.last_gold = self.trend2[i]
        res1, res2 = [], []
        for i in range(window_size-1):
            res1.append((block1[i+1]-block1[i])/(block1[i]+0.0001))
            res2.append((block2[i+1]-block2[i])/(block2[i]+0.0001))

        return np.array(res1), np.array(res2)

    def reset(self):
        self.hold_money = self.init_money
        self.buy_num1, self.buy_num2 = 0.0, 0.0
        self.hold_num1, self.hold_num2 = 0.0, 0.0
        self.stock_value1, self.stock_value2 = 0.0, 0.0
        self.maket_value = 0.0
        self.total_profit = 0.0
        self.last_gold = self.trend2[1]
        self.t = self.window_size//2
        self.reward = 0.0
        self.states_sell1, self.states_sell2 = [], []
        self.states_buy1, self.states_buy2 = [], []
        self.states_hold1, self.states_hold2 = [], []
        return self.get_state(self.t)

    def buy(self, flag=1):
        if flag == 1:
            self.buy_num1 = self.hold_money/2/self.trend1[self.t]
            tmp_money = self.trend1[self.t]*self.buy_num1
            service_change = tmp_money*self.buy_rate1
            self.hold_num1 += self.buy_num1
            self.hold_money = self.hold_money - \
                self.trend1[self.t]*self.buy_num1-service_change
            self.states_buy1.append(self.t)
        else:
            self.buy_num2 = self.hold_money/2/self.trend2[self.t]
            tmp_money = self.trend2[self.t]*self.buy_num2
            service_change = tmp_money*self.buy_rate2
            self.hold_num2 += self.buy_num2
            self.hold_money = self.hold_money - \
                self.trend2[self.t]*self.buy_num2-service_change
            self.states_buy2.append(self.t)

    def sell(self, flag=1):
        if flag == 1:
            tmp_money = self.hold_num1*self.trend1[self.t]*0.6
            service_change = tmp_money*self.sell_rate1
            self.hold_money = self.hold_money+tmp_money-service_change
            self.hold_num1 = self.hold_num1*0.4
            self.states_sell1.append(self.t)
        else:
            tmp_money = self.hold_num2*self.trend2[self.t]*0.6
            service_change = tmp_money*self.sell_rate2
            self.hold_money = self.hold_money+tmp_money-service_change
            self.hold_num2 = self.hold_num2*0.4
            self.states_sell2.append(self.t)

    def step(self, action1, action2, show_log=True):
        if action1 == 1 and self.t < (len(self.trend1)-self.half_window) and self.hold_money > 1e-10:
            self.buy(flag=1)
            # if show_log:
            #     print('day:%d, buy price:%f, buy num:%.3f, hold num:%.3f, hold money:%.3f' %
            #           (self.t, self.trend[self.t], self.buy_num, self.hold_num, self.hold_money))
        elif action1 == 2 and self.hold_num1 > 1e-10:
            self.sell(flag=1)
            # if show_log:
            #     print('day:%d, sell price:%f, hold num:%.3f, hold money:%.3f'
            #           % (self.t, self.trend[self.t], self.hold_num, self.hold_money))
        else:
            self.states_hold1.append(self.t)
            # if show_log:
            #     print('day:%d, hold num:%.3f hold money:%.3f' %
            #           (self.t, self.hold_num, self.hold_money))
        if action2 == 1 and self.t < (len(self.trend2)-self.half_window) and self.hold_money > 1e-10:
            self.buy(flag=2)
        elif action2 == 2 and self.hold_num2 > 1e-10:
            self.sell(flag=2)
        else:
            self.states_hold2.append(self.t)

        self.stock_value = self.trend1[self.t] * \
            self.hold_num1+self.trend2[self.t]*self.hold_num2
        self.maket_value = self.stock_value+self.hold_money
        self.total_profit = self.maket_value-self.init_money

        reward1 = (self.trend1[self.t+1] -
                   self.trend1[self.t])/self.trend1[self.t]
        if np.abs(reward1) <= 0.015:
            reward1 = reward1*0.2
        elif np.abs(reward1) <= 0.03:
            reward1 = reward1*0.7
        elif np.abs(reward1) >= 0.05:
            if reward1 < 0:
                reward1 = (reward1+0.05)*0.1-0.05
            else:
                reward1 = (reward1-0.05)*0.1+0.05

        if self.hold_num1 > 1e-10 or action1 == 2:
            self.reward1 = reward1
            if action1 == 2:
                self.reward1 = -self.reward1
        else:
            self.reward1 = -self.reward1*0.1

        if action2 == 0 or self.trend2[self.t+1] < 0:
            self.reward2 = 0.0
        else:
            reward2 = (self.trend2[self.t+1] -
                       self.trend2[self.t])/self.trend2[self.t]
            if np.abs(reward2) <= 0.015:
                reward1 = reward2*0.2
            elif np.abs(reward2) <= 0.03:
                reward2 = reward2*0.7
            elif np.abs(reward2) >= 0.05:
                if reward2 < 0:
                    reward2 = (reward2+0.05)*0.1-0.05
                else:
                    reward2 = (reward2-0.05)*0.1+0.05

            if self.hold_num2 > 1e-10 or action2 == 2:
                self.reward2 = reward2
                if action2 == 2:
                    self.reward2 = -self.reward2
            else:
                self.reward2 = -self.reward2*0.1

        done = False
        self.t = self.t+1
        if self.t == len(self.trend1)-2:
            done = True
        state1, state2 = self.get_state(self.t)
        reward1, reward2 = self.reward1, self.reward2
        return state1, state2, reward1, reward2, done

    # def get_info(self):
    #     return self.states_sell, self.states_buy, self.states_hold

    # def draw(self, save_name='a.png'):
    #     states_sell, states_buy, states_hold = self.get_info()
    #     total_gains = self.total_profit
    #     close = self.trend
    #     fig = plt.figure(figsize=(15, 5))
    #     plt.plot(close, color='b', lw=2.)
    #     plt.plot(close, 'v', markersize=4, color='r',
    #              label='selling signal', markevery=states_sell)
    #     plt.plot(close, '^', markersize=4, color='g',
    #              label='buying signal', markevery=states_buy)
    #     plt.plot(close, '_', markersize=6, color='gray',
    #              label='holding signal', markevery=states_hold)
    #     plt.title('total gains %f' %
    #               (total_gains))
    #     plt.legend()
    #     plt.savefig(save_name)
    #     plt.close()
