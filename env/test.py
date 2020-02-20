import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 25)

from utils import prepare, undo_state_transformations
from lobster.env.market import *

ACTIONS = {0: 'HOLD', 1: 'LIMIT ORDER', 2: 'MARKET ORDER'}

if __name__ == '__main__':

    reward_signal = {'method': 'VWAP_TARGET',  # supported: ['VWAP_TARGET', 'VWAP_RATIO', 'VWAP_PNL']
                     'collection': 'harvested',  # supported: ['terminal', 'harvested']
                     'target': 'beat'}  # supported: ['beat', 'track']

    params = {
        'tickers': ['MSFT'],  # reserve 'GOOG' for testing
        'data_dir': '/Users/mreiter/Desktop/drloe_v2/lobster/ob',
        'output_dir': '/Users/mreiter/Desktop/drloe_v2/lobster/output',
        'ob_name': '{}_2012-06-21_34200000_57600000_orderbook_10.csv',
        'episode_length': 60 * 1000,
        'inventory': 1000,
        'reward_signal': reward_signal,
        'side': 2,  # 2 for liquidation (SELL), 1 for acquisition (BUY)
        'boost_vol': 1,  # sometimes the volume is too small, so hack it here
        'look_back': 50,
        'depth': 5,  # how deep to render the market for visuals
        'impact': False,  # True to model for the actor's impact on the lob .. set to False until you sanity check this
        'verbose': True,
        'render': True,    # to render the environment or not
        'interactive': False  # whether the rendered plots will display during run
    }

    m = MarketEnvironment(params)

    done = False

    step, count = 0, 0
    step_limit, count_limit = 20, 1
    cumulative = 0
    rewards = 0

    while not done:
        m.display_lob()

        # choose a random action
        trade = np.random.randint(0, 2)

        # get the limit price
        best_bid = m.top_of_book()['BID'] if m.top_of_book()['BID'] != 0 else m.top_of_book()['ASK']
        limit_price = round(best_bid * np.random.uniform(0.95, 1.05), 4)

        # determine the amount
        amount = round(np.random.uniform(0, 1 - cumulative), 2)
        quantity = round(amount * m.inventory, 0)

        # if we reached the step limit then liquidate
        if step == step_limit:#m.steps_left == 1:
            trade, quantity = 2, m.inventory - m.executed

        if trade in [0, 3]:
            limit_price = 'NA'
            quantity = 'NA'
        else:
            cumulative += amount

        action = [trade, limit_price, quantity]

        obs, reward, done, info = m.step(action)
        rewards += reward
        summary = 'Summary:{}\n'.format(color.BOLD)
        summary += '{}Action: {} \t\tPrice: {}\t\tQuantity: {}\n'.format(color.RED,
                                                                         ACTIONS[trade],
                                                                         limit_price,
                                                                         quantity)
        summary += '{}Physical Inventory: {} \t\tInventory w/Orders: {}\n'.format(color.PURPLE,
                                                                                  m.inventory - m.executed,
                                                                                  m.inventory - m.executed - m.pending)
        summary += '{}Reward: {}{}\n\n'.format(color.DARKCYAN,
                                               reward,
                                               color.END)

        print(summary)
        m.render()
        time.sleep(2)

        step += 1

        if done:
            count += 1

            print('\nTotal Reward: {}\nOrder History: \n{}\nTrade History: \n{}'.format(rewards,
                                                                                        m.order_history,
                                                                                        m.trade_history))

            if count != count_limit:
                step, cumulative, rewards, done = 0, 0, 0, False

                m.reset()
                time.sleep(5)
