import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection


def make_series(side, depth):
    return pd.DataFrame(columns=['{} {}'.format(side, i)
                                 if i not in [1, depth]
                                 else ('Best {}'.format(side)
                                       if i != depth
                                       else 'Deeper VWA{}P'.format(side[0]))
                                 for i in range(1, depth+1)])


def get_quotes(ob_side, depth, descending=False):

    # get the quotes in the order book
    quotes = sorted(ob_side.keys(), reverse=descending)

    # get the volumes for each order
    volumes = []
    for quote in quotes:
        volumes += [sum([order.leaves_qty for order in ob_side[quote]])]

    # get the quotes up to the depth, and pad with zeros if too shallow
    if len(quotes) < depth:
        deeper_vwap = 0
    else:
        deeper_vwap = np.array(quotes[(depth-1)::]).dot(np.array(volumes[(depth-1)::])) / sum(volumes[(depth-1)::])

    # select the top quotes and append the deeper vwap calculation
    quotes = quotes[0:(depth-1)]
    quotes += [0] * (depth - 1 - len(quotes)) + [deeper_vwap]

    # get the % volumes at each
    volume_pie = list(np.array(volumes) / sum(volumes)) if len(volumes) > 0 else [0]*depth
    volume_share = volume_pie[0:(depth-1)]
    volume_share += [0] * (depth - 1 - len(volume_share)) + [sum(volume_pie[(depth-1)::])]

    return quotes, volume_share


class RenderMarket:

    def __init__(self, params):
        self.ticker = params['ticker']
        self.depth = params['depth']

        self.output_dir = params['output_dir']
        self.file_name = '{}_order_execution.gif'.format(self.ticker)

        # the inventory and lob dataframes
        self.inventory = pd.DataFrame(columns = ['Inventory'])

        self.bids = make_series('BID', depth=self.depth)
        self.bids_share = make_series('BID', depth=self.depth)

        self.asks = make_series('ASK', depth=self.depth)
        self.asks_share = make_series('ASK', depth=self.depth)

        # if we do not want to display interactively, turn off pyplot interactive mode
        if not params['display']:
            plt.ioff()
        else:
            plt.ion()

        # set the figure
        self.fig = plt.figure(constrained_layout=False)
        sns.set_style('dark')
        sns.set_context('paper')

        self.fig.suptitle('Performance Liquidating {}'.format(self.ticker))
        self.fig.tight_layout()

        # add the subplots
        gs = self.fig.add_gridspec(9, 1)
        self.lob_ax = self.fig.add_subplot(gs[0:4, 0])
        self.inventory_ax = self.fig.add_subplot(gs[5:, 0])

        # the list containing the snapshots
        self.snap_shots = []

    def reset(self, ticker):
        # reset the ticker
        self.ticker = ticker

        # reset the inventory and lob dataframes
        self.inventory = pd.DataFrame(columns=['Inventory'])

        self.bids = make_series('BID', depth=self.depth)
        self.bids_share = make_series('BID', depth=self.depth)

        self.asks = make_series('ASK', depth=self.depth)
        self.asks_share = make_series('ASK', depth=self.depth)

        # clear the figure
        self.lob_ax.clear()
        self.inventory_ax.clear()
        self.fig.suptitle('Performance Liquidating {}'.format(self.ticker))

        # the list containing the snapshots
        self.snap_shots = []

    def _update(self, inventory, lob):
        # add entries for the inventory
        self.inventory = self.inventory.append({'Inventory': inventory}, ignore_index=True)

        # add entries for the bid side
        bids, bids_share = get_quotes(lob.order_books[self.ticker].bids, self.depth, descending=True)
        self.bids = self.bids.append(dict(zip(self.bids.columns,
                                              bids)), ignore_index=True)
        self.bids_share = self.bids_share.append(dict(zip(self.bids_share.columns,
                                                          bids_share)), ignore_index=True)

        # add entries for the ask side
        asks, asks_share = get_quotes(lob.order_books[self.ticker].asks, self.depth, descending=False)
        self.asks = self.asks.append(dict(zip(self.asks.columns,
                                              asks)), ignore_index=True)
        self.asks_share = self.asks_share.append(dict(zip(self.asks_share.columns,
                                                          asks_share)), ignore_index=True)

    def render(self, window_size=25):
        current_step = len(self.inventory)

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step)

        # render the plots now
        self._render_lob(step_range)
        self._render_inventory(step_range)

        plt.pause(0.001)

    def _render_lob(self, step_range):
        # clear the axes and set x limit
        self.lob_ax.clear()

        if len(step_range) <= 1:
            self.lob_ax.set_xlim(0, 1)
        else:
            self.lob_ax.set_xlim(step_range[0], step_range[-1])

        # reformatting after the clear
        formatter = ticker.FormatStrFormatter('$%1.2f')
        self.lob_ax.yaxis.set_major_formatter(formatter)
        self.lob_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # quote levels
        quote_range = range(len(self.bids.columns)-1)

        # subset the bids and asks
        bids, bids_share = self.bids.iloc[step_range, quote_range], self.bids_share.iloc[step_range, quote_range]
        asks, asks_share = self.asks.iloc[step_range, quote_range], self.asks_share.iloc[step_range, quote_range]

        min_bid, max_bid = bids.min(axis=1).min(), bids.min(axis=1).min()
        min_ask, max_ask = asks.min(axis=1).min(), asks.max(axis=1).max()

        # careful, it is possible that the order book gets depleted!
        self.lob_ax.set_ylim(min_bid if min_bid != 0 else min_ask,
                             max_ask if max_ask != 0 else max_bid)

        # set the titles
        self.lob_ax.set_title('Limit Order Book')

        # plot the bid and ask sides
        self._quote_collections(bids, bids_share, color='#fcbc99', thickness=10)
        self._quote_collections(asks, asks_share, color='#434862', thickness=10)

    def _quote_collections(self, quotes, share, color='#434862', thickness=5):
        for quote in quotes:
            points = np.array([quotes.index.values, quotes[quote].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=thickness * share[quote].values, color=color)
            self.lob_ax.add_collection(lc)

    def _render_inventory(self, step_range):
        # get the time and the last inventory ... for annotating purposes
        time = len(self.inventory) - 1
        last_inventory = self.inventory.iloc[-1, -1]
        initial_inventory = self.inventory.iloc[0, 0]

        # clear the axes and set limits
        self.inventory_ax.clear()
        self.inventory_ax.set_ylim(0, initial_inventory)

        if len(step_range) <= 1:
            self.inventory_ax.set_xlim(0, 1)
        else:
            self.inventory_ax.set_xlim(step_range[0], step_range[-1])

        # reformatting
        self.inventory_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # set the titles
        self.inventory_ax.set_title('Inventory')

        # plot the inventory
        self.inventory_ax.plot(self.inventory.iloc[step_range], '-', color='#fcbc99', label='inventory')

        # annotate the plot
        self.inventory_ax.annotate('{0:.2f}'.format(last_inventory),
                                   (time, last_inventory),
                                   xytext=(time, last_inventory),
                                   bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                                   color="#3B4F66",
                                   fontsize="small")

    def _make_gif(self):
        pass
