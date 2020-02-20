''' gym environment based on LOBSTER data '''
__author__ = 'Matthew Reiter'
__email__ = 'matthew.reiter@mail.utoronto.ca'
__version__ = '2.0.5'
__status__ = 'Production'
__copyright__   = 'Copyright 2020, Applications of Deep Reinforcement Learning, BASc Thesis, University of Toronto'


import gym
import time
import random
import collections

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from gym import spaces
from scipy import stats
from lobster.env.render import *
from lightmatchingengine.lightmatchingengine import LightMatchingEngine, Side

OH_COLS = ['order_id',          # unique identifier from the matching engine
           'i_placed',          # time index when order was placed on exchange
           'i_filled',          # time index when filled ... -1 if not filled
           'i_cancelled',       # time index when cancelled ... -1 if not cancelled
           'price',             # target price for limit order
           'quantity',          # target quantity for limit order
           'filled',            # quantity executed
           'pv',                # price times volume for each fill
           'active']            # False if not filled or cancelled

TH_COLS = ['trade_id',          # unique identifier for the trade
           'order_id',          # relation to order id
           'i_trade',           # time index of trade
           'price',             # price the trade executed at
           'quantity',          # trade volume
           'side']              # 1 for BUY and 2 for SELL

VWAP_COLS = ['i_time',          # time index when lob vwap is snapped
             'pv',              # price of all orders multiplied by their respective volumes
             'volume']          # volumes of all orders

ACTIONS = {'HOLD': 0,           # supported actions
           'LIMT': 1,
           'MKT_': 2,
           'CNCL': 3}

# start and end of the trading day, given in ms since midnight to match the format from the LOBSTER output files
START_OF_DAY = 9.5 * 60 * 60 * 1000
END_OF_DAY = 16 * 60 * 60 * 1000

LOBSTER_COLUMNS = [i for i in range(41)]
LOBSTER_PRICE_MULTIPLIER = 10000
LOBSTER_ASK_PRICES = LOBSTER_COLUMNS[1::4]
LOBSTER_ASK_VOLUMES = LOBSTER_COLUMNS[2::4]
LOBSTER_BID_PRICES = LOBSTER_COLUMNS[3::4]
LOBSTER_BID_VOLUMES = LOBSTER_COLUMNS[4::4]


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class MarketEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    # ******************************************************************************************************************
    # initialization
    # ******************************************************************************************************************
    def __init__(self, params):
        ''' notes:
                - the market simulates a client request for an acquisition (side = 1) or liquidation (side = 2)
                - data comes from LOBSTER, approximately at a 1ms frequency,  10 levels
                - the data is all from the date 2012-06-21
                - the request supports five tickers (AAPL, AMZN, INTC, MFST and GOOG), randomly chosen for each reset
                - requests come at any point during the day, provided that there is sufficient time to execute

            :param params: dictionary containing parameters for the market environment
        '''

        # just to see how long it takes to initialize the environment ... most costly part is loading all the data
        init_start = time.time()

        # determine the direction of trades and the impact consideration flag (impact = False --> 'infinite' recovery)
        self.side = params['side']
        self.impact = params['impact']
        self.boost_vol = params['boost_vol']  # data hack ... if volume is too low, amplify it

        # read all the lob data ... they are large files so we prevent reloading them on each reset
        self.tickers = params['tickers']
        self.lobs = dict()

        for ticker in self.tickers:
            self.lobs[ticker] = self.lob_data = pd.read_csv('{}/{}'.format(params['data_dir'],
                                                                           params['ob_name'].format(ticker)),
                                                            header=None)

        # randomly sample a ticker and get the corresponding order book
        self.ticker = random.choice(self.tickers)
        ob = self.lobs[self.ticker]

        # initialize the current_step counter, set the episode length and note the time of the final possible episode
        self.current_step = 0
        self.episode_length = params['episode_length']
        self.final_episode = END_OF_DAY - self.episode_length

        # cutoff point, when the last trade needs to be executed, and the starting randomly sampled index
        self.look_back = params['look_back']
        final_cutoff = len(ob.loc[ob[0] <= self.final_episode])
        start_index = np.random.randint(self.look_back, final_cutoff)

        # set the date and the time
        self.date = '20120621'
        start_obs = ob.iloc[start_index - self.look_back, 0]
        self.start_time = ob.iloc[start_index, 0]
        self.time = self.start_time

        # set the lob, corresponding to the desired subset of the complete lob
        self.lob_data = ob.loc[(ob[0] >= self.time) & (ob[0] <= (self.time + self.episode_length))]
        self.lob_data.iloc[:, LOBSTER_ASK_PRICES + LOBSTER_BID_PRICES] /= LOBSTER_PRICE_MULTIPLIER
        self.lob_data.iloc[:, LOBSTER_ASK_VOLUMES + LOBSTER_BID_VOLUMES] *= self.boost_vol

        # get the maximum number of steps left
        self.steps_left = len(self.lob_data.loc[self.lob_data[0] > self.time].drop_duplicates(subset=0))
        self.steps_length = len(self.lob_data[0].drop_duplicates())

        # matching engines and order tracking
        self.lme = LightMatchingEngine()
        self.order_history = pd.DataFrame(columns=OH_COLS)          # tracking orders posted by the actor
        self.trade_history = pd.DataFrame(columns=TH_COLS)          # tracking executed trades
        self.lobster_orders = pd.DataFrame(columns=['order_id',
                                                    'price'])       # historical orders from LOBSTER file
        self.dummy_orders = pd.DataFrame(columns=['order_id',
                                                  'price'])         # orders meant to approximate lasting market impacts
        self.dark_orders = pd.DataFrame(columns=['order_id',        # approximating dark orders
                                                 'i_placed',
                                                 'price'])
        self.lob_vwap = pd.DataFrame(columns=VWAP_COLS)             # tracking the vwap of the lob

        # set the lob
        self._set_ob()

        # inventory and reward related parameters
        self.reward_signal = params['reward_signal']                # outlines how the return should be calculated

        self.inventory = params['inventory']                        # the amount to execute on
        self.pending = 0                                            # quantity tied up in a limit order not yet executed
        self.executed = 0                                           # the quantity sold or bought

        self.pnl = 0                                                # the running profit (side=2) or cost (side=1)
        self.delayed = 0                                            # delayed reward from a limit order executed later

        # the observation data, which will be appended to after each action is taken
        self.obs_data = self._init_obs(ob.loc[(ob[0] >= start_obs) & (ob[0] < self.time)])

        # create a visual instance if we want to visualize the environment and create nice looking gifs
        if params['render']:
            self.visual = RenderMarket({'ticker': self.ticker,
                                        'output_dir': params['output_dir'],
                                        'display': params['interactive'],
                                        'depth': 5})
            self.visual._update(self.inventory, self.lme)
            self.visual.render()
        else:
            self.visual = None

        self.action_space = spaces.Box(low=np.array([0, 0]),
                                       high=np.array([3, 1]),
                                       dtype=np.float16)
        self.observation_space = spaces.Box(low=-99999999,
                                            high=99999999,
                                            shape=(len(self.obs_data), len(self.obs_data.columns)),
                                            dtype=np.float16)

        if params['verbose']:
            print('finished initializing the market for {} in {} seconds'.format(self.ticker, time.time() - init_start))

    # ******************************************************************************************************************
    # reset to a new (random) ticker with a new (random) starting point
    # ******************************************************************************************************************
    def reset(self):
        ''' resets the state of the environment '''

        # limit order book data for a randomly sampled ticker
        self.ticker = random.choice(self.tickers)
        ob = self.lobs[self.ticker]

        # reset the counter
        self.current_step = 0

        # cutoff point, when the last trade needs to be executed, and the starting randomly sampled index
        final_cutoff = len(ob.loc[ob[0] <= self.final_episode])
        start_index = np.random.randint(self.look_back, final_cutoff)

        # set the time
        start_obs = ob.iloc[start_index - self.look_back, 0]
        self.start_time = ob.iloc[start_index, 0]
        self.time = self.start_time

        # reset the lob, adjusting the prices to account for the multiplier and boost volume if we need to
        self.lob_data = ob.loc[(ob[0] >= self.time) & (ob[0] <= (self.time + self.episode_length))]
        self.lob_data.iloc[:, LOBSTER_ASK_PRICES + LOBSTER_BID_PRICES] /= LOBSTER_PRICE_MULTIPLIER
        self.lob_data.iloc[:, LOBSTER_ASK_VOLUMES + LOBSTER_BID_VOLUMES] *= self.boost_vol

        self.steps_left = len(self.lob_data.loc[self.lob_data[0] > self.time].drop_duplicates(subset=0))
        self.steps_length = len(self.lob_data[0].drop_duplicates())

        # reset the lob and order/trade tracking tools
        self.lme = LightMatchingEngine()
        self.order_history = pd.DataFrame(columns=OH_COLS)
        self.trade_history = pd.DataFrame(columns=TH_COLS)
        self.lobster_orders = pd.DataFrame(columns=['order_id', 'price'])
        self.dummy_orders = pd.DataFrame(columns=['order_id', 'price'])
        self.dark_orders = pd.DataFrame(columns=['order_id', 'i_placed','price'])
        self.lob_vwap = pd.DataFrame(columns=VWAP_COLS)

        # reset the lob
        self._set_ob()

        # inventory related parameters
        self.pending = 0
        self.executed = 0

        self.pnl = 0
        self.delayed = 0

        # reset observation data
        self.obs_data = self._init_obs(ob.loc[(ob[0] >= start_obs) & (ob[0] < self.time)])

        # reset the visual
        if self.visual is not None:
            self.visual.reset(self.ticker)
            self.visual._update(self.inventory, self.lme)

        return self._get_state()

    def _init_obs(self, lob_data):
        ''' columns: 41 (LOBSTER DATA) + 4 (INVENTORY DATA) = 44 FEATURES over a rolling window basis.
            Window corresponds approximately to the period defined by self.look_back
        '''

        # adjust the ask and bid prices, since they are multiplied by 10,000 in the file
        lob_data.loc[:, LOBSTER_ASK_PRICES + LOBSTER_BID_PRICES] /= LOBSTER_PRICE_MULTIPLIER

        # adjust our ob data into
        obs_data = lob_data.loc[:, [0]+
                                    LOBSTER_ASK_PRICES +
                                    LOBSTER_ASK_VOLUMES +
                                    LOBSTER_BID_PRICES +
                                    LOBSTER_BID_VOLUMES]

        # inventory that must still be liquidated
        obs_data[len(obs_data.columns)] = self.inventory

        # inventory that is held up in a limit order
        obs_data[len(obs_data.columns)] = self.pending

        # time remaining for liquidation
        obs_data[len(obs_data.columns)] = self.episode_length

        # steps remaining for liquidation
        obs_data[len(obs_data.columns)] = self.steps_left / self.steps_length

        # ensure that the size of the observation is consistent
        obs_data = obs_data.iloc[-self.look_back:, :]

        # pad with zero rows at the beginning if under sized
        padding = self.look_back - len(obs_data)
        if padding > 0:
            padding_rows = pd.DataFrame(np.zeros((padding, len(obs_data.columns))),
                                        columns= obs_data.columns)

            obs_data = padding_rows.append(obs_data)

        # adjust the index so that it makes sense in view of a period backview
        obs_data.index = reversed(range(len(obs_data)))

        return obs_data

    def _get_state(self):
        ''' returns the current state of the limit order book and the recent history '''

        # update the visual data
        if self.visual is not None:
            self.visual._update(self.inventory - self.executed, self.lme)

        self.obs_data = self.obs_data.shift(-1)

        # update the time
        current_time = self.time

        # get the prices and volumes
        ask_prices, ask_volumes = self.get_lob_side(Side.SELL)
        bid_prices, bid_volumes = self.get_lob_side(Side.BUY)

        # build the new observation
        current_obs = dict(zip(self.obs_data.columns,
                               [current_time] +
                               ask_prices +
                               ask_volumes +
                               bid_prices +
                               bid_volumes +
                               [self.inventory - self.executed] +
                               [self.pending] +
                               [self.start_time + self.episode_length - self.time] +
                               [self.steps_left / self.steps_length]))

        # append the new observation
        self.obs_data.iloc[-1, :] = current_obs

        return self.obs_data

    # ******************************************************************************************************************
    # displaying the limit order book
    # ******************************************************************************************************************

    def get_lob_side(self, side):
        ob = self.lme.order_books[self.ticker].asks if side == Side.SELL else self.lme.order_books[self.ticker].bids
        ob_quotes = sorted(ob.keys(), reverse=False if side == Side.SELL else True)

        if len(ob_quotes) == 0:
            return [0]*len(LOBSTER_ASK_PRICES), [0]*len(LOBSTER_ASK_PRICES)

        volumes = []
        for quote in ob_quotes:
            volumes += [sum([order.leaves_qty for order in ob[quote]])]

        # subset to the required depth given by the LOBSTER file, pad if necessary.
        ob_quotes = ob_quotes[0:len(LOBSTER_ASK_PRICES)]
        ob_quotes += [0] * max(len(LOBSTER_ASK_PRICES) - len(ob), 0)

        volumes = volumes[0:len(LOBSTER_ASK_PRICES)]
        volumes += [0] * max(len(LOBSTER_ASK_PRICES) - len(volumes), 0)

        return ob_quotes, volumes


    def get_volumes(self, side, levels=None):
        if side == Side.SELL:
            orders = [* self.get_quotes(Side.SELL).values()][0:levels]
        elif side == Side.BUY:
            orders = [* self.get_quotes(Side.BUY).values()][0:levels]
        else:
            orders = [* self.get_quotes(Side.SELL).values()] + [* self.get_quotes(Side.BUY).values()]

        volume = 0
        for order in orders:
            volume += sum([quote.leaves_qty for quote in order])

        return volume

    def get_quotes(self, side):
        if side == Side.SELL:
            return collections.OrderedDict(sorted(self.lme.order_books[self.ticker].asks.items(), reverse=False))
        else:
            return collections.OrderedDict(sorted(self.lme.order_books[self.ticker].bids.items(), reverse=True))

    def display_lob(self):
        asks = [* self.get_quotes(Side.SELL).values()]
        asks_pv, asks_volume = 0, 0

        bids = [* self.get_quotes(Side.BUY).values()]
        bids_pv, bids_volume = 0, 0

        title = '{}{}\n{} LIMIT ORDER BOOK FOR {} at {} on {} {}'.format(color.BOLD, color.END, ' '*20,
                                                                         self.ticker, self.time, self.date, ' '*20)
        line = '{}'.format('_'*len(title))
        strip = '{:<3s} {:<5s}: {:<12s} {:<6s}: {:<12s}'

        print(title)
        print(line)
        for i in range(max(len(asks), len(bids))):
            row = ''

            bid_volume = sum([bid.leaves_qty for bid in bids[i]]) if i < len(bids) else 0
            bids_pv += bids[i][0].price * bid_volume if i < len(bids) else 0
            bids_volume += bid_volume

            row += strip.format('BID' if i==0 else ' '*3,
                                'PRICE' if i == 0 else ' '*5,
                                str(bids[i][0].price) if i < len(bids) else ' '*12,
                                'VOLUME' if i == 0 else ' '*6,
                                str(bid_volume) if i < len(bids) else ' '*12)

            ask_volume = sum([ask.leaves_qty for ask in asks[i]]) if i < len(asks) else 0
            asks_pv += asks[i][0].price * ask_volume if i < len(asks) else 0
            asks_volume += ask_volume

            row += '|' + strip.format('ASK' if i==0 else ' '*3,
                                      'PRICE' if i == 0 else ' '*5,
                                      str(asks[i][0].price) if i < len(asks) else ' '*12,
                                      'VOLUME' if i == 0 else ' '*6,
                                      str(ask_volume) if i < len(asks) else ' '*12)

            print(row)

        print('~'*len(title))
        totals = strip.format(' '*3,
                              'VWABP',
                              str(round(bids_pv/bids_volume, 2)) if bids_volume > 0 else '0',
                              'TOTALV',
                              str(bids_volume) if i < len(asks) else '0')
        totals += '|' + strip.format(' '*3,
                              'VWAAP',
                              str(round(asks_pv/asks_volume,2)) if asks_volume > 0 else '0',
                              'TOTALV',
                              str(asks_volume) if i < len(asks) else '0')
        print(totals)
        print(line)

    def top_of_book(self):
        asks = [*self.lme.order_books[self.ticker].asks.keys()]
        bids = [*self.lme.order_books[self.ticker].bids.keys()]

        return {'BID': 0 if len(bids) == 0 else max(bids),
                'ASK': 0 if len(asks) == 0 else min(asks)}

    # ******************************************************************************************************************
    # updating the limit order book - setting, updating, clearing and cancelling
    # ******************************************************************************************************************
    def _set_ob(self):
        ''' sets the new lob by performing the following
            1. clear the old instance of the limit order book ... only keeping outstanding orders from the actor
            2. add the historical trades that were completed by the actor ... to approximate her market impact
            3. add the bid/ask prices from the lob snapshot

            note:
            the order here is important. cannot add new orders until the dummy orders are placed
        '''

        self._clear_ob()
        if self.impact:
            self._order_impacts()
        self._lob_snapshot()


    def _helper_clear(self, order):
        ''' helper function, enclosed in a try and except in case the limit does not exist in the book anymore '''
        try:
            self.lme.cancel_order(order['order_id'], self.ticker)
        except:
            pass

    def _clear_ob(self, dark=False):
        ''' clears the order book

            note:
            only clears the outstanding orders coming from the file and any dummy orders, any outstanding orders placed
            by the actor will remain ...
        '''

        if dark:
            clear_orders = self.dark_orders
            self.dark_orders = pd.DataFrame(columns=['order_id', 'i_placed', 'price'])
        else:
            clear_orders = self.lobster_orders.append(self.dummy_orders, ignore_index=True)
            self.lobster_orders = pd.DataFrame(columns=['order_id', 'price'])
            self.dummy_orders = pd.DataFrame(columns=['order_id', 'price'])

        clear_orders.apply(lambda order:
                           self._helper_clear(order), axis=1)

    def _impacts_helper(self, order):
        dummy, _ = self.lme.add_order(self.ticker, order['price'], order['quantity'], order['side'])
        self.dummy_orders = self.dummy_orders.append({'order_id': dummy.order_id,
                                                      'price': dummy.price}, ignore_index=True)

    def _order_impacts(self):
        ''' place orders for completed tracks in the lob to approximate market impacts of actors trades

            note:
            approximation because it does not take into consideration how other market participants would react
            to the actors trades ... !DO NOT TREAT THESE ORDERS AS REAL!
        '''

        self.trade_history.apply(lambda order:
                                 self._impacts_helper(order), axis=1)

    def _lob_snapshot(self):
        ''' based on the current_step index, add the orders for the bid and ask prices

            notes:
            1. Order to the file is: ASK_PRICE1, ASK_VOLUME1, BID_PRICE1, BID_VOLUME1 ...
            2. need to score delayed rewards ... consider trades executed with outstanding orders

        '''

        # parse the LOBSTER file for the lob prices and volumes
        ask = np.append([self.lob_data.iloc[self.current_step, 1::4].values],
                        [self.lob_data.iloc[self.current_step, 2::4].values], axis=0)

        bid = np.append([self.lob_data.iloc[self.current_step, 3::4].values],
                        [self.lob_data.iloc[self.current_step, 4::4].values], axis=0)

        # get all active orders which have been (partially) filled with the evolution of the limit order book
        trades = []
        for i in range(ask.shape[1]):
            ask_order, ask_trades = self.lme.add_order(self.ticker, ask[0,i], ask[1,i], Side.SELL)
            bid_order, bid_trades = self.lme.add_order(self.ticker, bid[0, i], bid[1, i], Side.BUY)

            # append both lists of trades
            trades += ask_trades + bid_trades

            # track the historical orders
            self.lobster_orders = self.lobster_orders.append({'order_id': ask_order.order_id,
                                                              'price': ask_order.price}, ignore_index=True)
            self.lobster_orders = self.lobster_orders.append({'order_id': bid_order.order_id,
                                                              'price': bid_order.price}, ignore_index=True)

        # update market vwap, store this in the tracking
        self._calc_vwap()

        # fill orders resulting from passive order being filled
        self._fill_order(trades)

    def _post_dark_orders(self):
        # get the quotes at the top of the book to determine the mid-quote and the spread
        top_of_book = self.top_of_book()
        best_ask, best_bid = top_of_book['ASK'], top_of_book['BID']
        mid_quote, spread = (best_ask + best_bid)/2, best_ask - best_bid

        # order imbalance
        ask_volume, bid_volume = self.get_volumes(Side.SELL, levels=1), self.get_volumes(Side.BUY, levels=1)
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        # find the direction to post the net dark orders
        side = Side.BUY if self.side == Side.SELL else Side.SELL
        target_volume = bid_volume if side == Side.BUY else ask_volume
        quantity = max(int(round(target_volume * 0.10, 0)), 1)

        skew, loc, scale = imbalance, mid_quote, spread/10
        prices = np.around(stats.skewnorm(skew, loc, scale).rvs(10), 2)

        trades = []
        for price in prices:
            dark_order, dark_trades = self.lme.add_order(self.ticker, price, quantity, side)
            trades += dark_trades

            self.dark_orders = self.dark_orders.append({'order_id': dark_order.order_id,
                                                        'i_placed': self.current_step,
                                                        'price': dark_order.price}, ignore_index=True)

        # handle the orders
        self._fill_order(trades)

    def _cancel_helper(self, order):
        # remove the order from the lob ... enclose in a try and catch for safety.
        try:
            self.lme.cancel_order(order['order_id'], self.ticker)
        except:
            pass

        # free up the inventory
        self.pending -= (order['quantity'] - order['filled'])

        # update the time when the order was cancelled, set active status to false
        self.order_history.loc[self.order_history['order_id'] == order['order_id'], 'i_cancelled'] = self.current_step
        self.order_history.loc[self.order_history['order_id'] == order['order_id'], 'active'] = False

    def _cancel_orders(self, orders=None):
        ''' cancels the actor's outstanding orders
        '''

        if orders is None:
            active_orders = self.order_history.loc[self.order_history['active'] == True]
        else:
            active_orders = orders

        # cancel the active orders specified
        active_orders.apply(lambda order:
                            self._cancel_helper(order), axis=1)

    # ******************************************************************************************************************
    # interactions with the market - execute trades
    # ******************************************************************************************************************
    def step(self, action):
        # if the action involves a cancel, clear the outstanding orders BEFORE we update the book ...
        if action[0] in [2, 3]:
            self._cancel_orders()

        # update the lob and accumulate delayed rewards
        self._set_ob()

        # add dark orders to the book ... these cannot be seen by the agent
        self._post_dark_orders()

        # execute the trade, evaluating immediate and delayed rewards ... !!!! the delay might cause problems training
        reward = self._execute_trade(action) + self.delayed
        self.delayed = 0

        # remove the dark orders
        self._clear_ob(dark=True)

        # get the state
        obs = self._get_state()

        # update the steps remaining
        self.steps_left -= 1

        # update the counter
        next_step = self.current_step
        while True:
            next_step += 1

            # if we are at the end of the day, set current_step = length of lob_data, which will later trigger done=True
            if next_step == len(self.lob_data):
                self.current_step = next_step
                break

            # if we have an updated lob, update the current step ... if multiple events, select the time randomly
            if self.lob_data.iloc[self.current_step, 0] != self.lob_data.iloc[next_step, 0]:
                count = len(self.lob_data.loc[self.lob_data[0] == self.lob_data.iloc[next_step, 0]])
                self.current_step = next_step + np.random.randint(0, count)
                break

        # check if done
        if (self.current_step == len(self.lob_data)) or (self.steps_left == 0):
            done, info = True, 'the liquidation period is over'
        elif self.executed == self.inventory:
            done, info = True, 'the inventory has been fully liquidated'
        else:
            # update time
            self.time = self.lob_data.iloc[self.current_step, 0]
            done, info = False, 'the episode continues'

        # if we are done and we configured the environment for a terminal reward, calculate it here
        if done:
            vwap_m = self.lob_vwap.sum()['pv'] / self.lob_vwap.sum()['volume']      # the market vwap over the period

            if self.reward_signal['collection'] == 'terminal':                      # only realize reward at the end
                pv = float((self.trade_history['price'] * self.trade_history['quantity']).sum())
                volume = float(self.trade_history.sum()['quantity'])

                reward = self._calc_reward(pv, volume, vwap_m, done)
            else:
                if self.reward_signal['method'] == 'VWAP_PNL':                       # liability terminal expense
                    reward += self.inventory * vwap_m * (-1)**(self.side + 1)

        return obs, reward, done, info

    def _calc_vwap(self, update=True):

        # infer the VWAP from the top of the book
        bids = self.get_lob_side(Side.BUY)
        asks = self.get_lob_side(Side.SELL)

        # initialize variables and then calculuate VWAP
        pv, volume = 0, 0
        try:
            pv += bids[0][0] * bids[1][0] + asks[0][0] * asks[1][0]
            volume += bids[1][0] + asks[1][0]
            vwap = pv / volume
        except:
            pv, volume, vwap =  0, 0, 0

        # add the record to the tracking dataframe if one does not already exist for the time. if it does, override it
        if update:
            if self.current_step in list(self.lob_vwap['i_time']):
                self.lob_vwap.loc[self.lob_vwap['i_time'] == self.current_step, 'pv'] = pv
                self.lob_vwap.loc[self.lob_vwap['i_time'] == self.current_step, 'volume'] = volume
            else:
                self.lob_vwap = self.lob_vwap.append(dict(zip(VWAP_COLS, [self.current_step,
                                                                          pv,
                                                                          volume])), ignore_index=True)

        return vwap

    def _calc_reward(self, pv, volume, vwap_m, done=False):
        vwap_e = pv / volume

        if self.reward_signal['collection'] == 'terminal' and not done:
            reward = 0
        else:
            if self.reward_signal['target'] == 'beat':
                if self.reward_signal['method'] == 'VWAP_RATIO':
                    reward = self.inventory * ((vwap_e / vwap_m) ** ((-1) ** self.side) - 1)
                elif self.reward_signal['method'] == 'VWAP_PNL':
                    reward = pv * (-1) ** self.side
                elif self.reward_signal['method'] == 'VWAP_TARGET':
                    reward = (vwap_e - vwap_m) * (-1) ** self.side# * (volume / self.inventory)
            elif self.reward_signal['target'] == 'track':
                reward = np.exp(-self.inventory * (vwap_m - vwap_e) ** 2)
            else:
                pass

        return reward

    def _fill_order(self, trades):
        ''' function updates outstanding orders and determines the delayed reward

        :param trades:
        :return:
        '''

        # all the active orders which may have been filled with the incoming trades
        active_orders = self.order_history.loc[self.order_history['active'] == True]

        # consolidate the trades and associate them to their respective orders
        fill_orders = {}
        for trade in trades:
            if trade.order_id in list(active_orders['order_id']):
                if trade.order_id in fill_orders:
                    fill_orders[trade.order_id] += [trade]
                else:
                    fill_orders[trade.order_id] = [trade]

        # process the orders
        if len(fill_orders) > 0:

            cum_pv, cum_vol = 0, 0
            for trades in fill_orders.keys():

                pv, volume = 0, 0
                for trade in fill_orders[trades]:
                    trade_details = dict(zip(TH_COLS, [trade.trade_id,
                                                       trade.order_id,
                                                       self.current_step,
                                                       trade.trade_price,
                                                       trade.trade_qty,
                                                       trade.trade_side]))

                    self.trade_history = self.trade_history.append(trade_details, ignore_index=True)

                    # pv and volume for each order
                    pv += trade.trade_price * trade.trade_qty
                    volume += trade.trade_qty

                # calculate vwap_m since the order was last filled
                last = self.order_history.loc[self.order_history['order_id'] == trade.order_id, 'i_filled'].iloc[-1]
                # period_totals = self.lob_vwap.loc[self.lob_vwap.i_time >= last].sum()
                # period_totals = self.lob_vwap.sum()
                # vwap_m = period_totals['pv'] / period_totals['volume']
                vwap_m = (self.lob_vwap.iloc[-1]['pv'] + pv) / (self.lob_vwap.iloc[-1]['volume'] + volume)

                # update the delayed reward tracker ...
                self.delayed += self._calc_reward(pv, volume, vwap_m)

                # update the order information to reflect the trade
                self.order_history.loc[self.order_history['order_id'] == trade.order_id, 'i_filled'] = self.current_step
                self.order_history.loc[self.order_history['order_id'] == trade.order_id, 'filled'] += volume
                self.order_history.loc[self.order_history['order_id'] == trade.order_id, 'pv'] += pv
                self.order_history.loc[self.order_history['active'] == trade.order_id, 'active'] = \
                    self.order_history.loc[self.order_history['order_id'] == trade.order_id, 'filled'] == \
                        self.order_history.loc[self.order_history['order_id'] == trade.order_id, 'quantity']

                # remove the quantity from the pending list and update the quantity sold
                self.pending -= volume
                self.executed += volume
                self.pnl += pv

    def _place_order(self, order, trades, vwap):
        ''' given an order that was sent to the matching engine, record it in the history of orders and score the reward

            note:
            the details of the order reflect efforts of the engine to match the order

            :param order:
            :param trades:
            :return:
        '''

        # there was no order execution so there is no reward
        if order is None:
            return 0
        else:
            # list of trades executed for the given order
            executed = [trade for trade in trades if trade.order_id == order.order_id]

            # initialize price * volume
            pv = 0
            volume = 0

            # add each executed trade to the trade history and update price * volume for the order
            for trade in executed:
                trade_details = dict(zip(TH_COLS, [trade.trade_id,
                                                   trade.order_id,
                                                   self.current_step,
                                                   trade.trade_price,
                                                   trade.trade_qty,
                                                   trade.trade_side]))
                self.trade_history = self.trade_history.append(trade_details, ignore_index=True)
                pv += trade.trade_price * trade.trade_qty
                volume += trade.trade_qty

            # add the order to the order history
            order_details = dict(zip(OH_COLS, [order.order_id,
                                               self.current_step,
                                               self.current_step if order.cum_qty > 0 else -1,
                                               -1,
                                               order.price,
                                               order.qty,
                                               order.cum_qty,
                                               pv,
                                               True if order.leaves_qty > 0 else False]))

            # add the order to the tracking records
            self.order_history = self.order_history.append(order_details, ignore_index=True)
            self.executed += order.cum_qty
            self.pending += order.leaves_qty
            self.pnl += pv

            return self._calc_reward(pv, order.cum_qty, vwap) if order.cum_qty > 0 else 0

    def _execute_trade(self, action):

        # calculate the vwap for the current lob, BEFORE we make our trade. (we calculate this when we update the lob)
        # vwap = float(self.lob_vwap.iloc[-1]['pv'] / self.lob_vwap.iloc[-1]['volume'])
        # period_totals = self.lob_vwap.sum()
        vwap = self.lob_vwap.iloc[-1]['pv'] / self.lob_vwap.iloc[-1]['volume']

        order_type = action[0]                                  # 0.HOLD, 1.LIMT, 2.MKT_ 3. CNCL
        price = action[1]                                       # target price for the limit order
        amount = action[2]                                      # quantity of the order

        # avoid a pointless trade
        if amount == 0:
            order_type = 0

        # initialize both order and trades to None, to be overwritten if an order is placed/trade is executed
        order, trades = None, None

        if order_type == 1:                                 # place a limit order
            order, trades = self.lme.add_order(self.ticker, price, amount, self.side)

        elif order_type == 2:                               # execute a market order to finish to session
            # get the opposite side of the book
            penalty_side = Side.BUY if self.side == Side.SELL else Side.BUY

            # get the volume of orders
            volume = self.get_volumes(penalty_side)

            # we need to ensure that the market order clears ... so add to the bottom with a punitive penalty
            if volume < amount:
                penalty = round(np.random.uniform(0, 0.05), 2)
                penalty_price = (1-penalty) * self.lobster_orders['price'].min() if penalty_side == Side.BUY \
                    else (1+penalty) * self.lobster_orders['price'].max()

                # add the order to the market to support the upcoming market order
                _, _ = self.lme.add_order(self.ticker, penalty_price, amount - volume, penalty_side)

            # make the market order
            order, trades = self.lme.add_order(self.ticker, 0, amount, self.side)

        elif order_type == 3:                               # clear limit orders
            self._cancel_orders()

        return self._place_order(order, trades, vwap)

    # ******************************************************************************************************************
    # render the market
    # ******************************************************************************************************************
    def render(self, mode='human', close=False):
        if self.visual is not None:
            self.visual.render()
