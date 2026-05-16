"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function
import collections
import numpy as np
import pandas as pd
import copy
from .policies import MultiPeriodOpt


def getFiscalQuarter(dt):
    """Convert a time to a fiscal quarter.
    """
    year = dt.year
    quarter = (dt.month - 1) // 3 + 1
    return "Q%i %s" % (quarter, year)


class SimulationResult():
    """A container for the result of a simulation.

    Attributes:
        h_next: A dataframe of holdings over time.
        u: A dataframe of trades over time.
        tcosts: A series of transaction costs over time.
        borrow_costs: A series of borrow costs over time.
    """

    def __init__(self, initial_portfolio, policy, cash_key, simulator,
                 simulation_times=None, PPY=252,
                 timedelta=pd.Timedelta("1 days")):
        """
        Initialize the result object.

        Args:
            initial_portfolio:
            policy:
            simulator:
            simulation_times:
            PPY:
        """
        self.PPY = PPY
        self.timedelta = timedelta
        self.initial_val = sum(initial_portfolio)
        self.initial_portfolio = copy.copy(initial_portfolio)
        self.cash_key = cash_key
        self.simulator = simulator
        self.policy = policy

    def summary(self):
        print(self._summary_string())

    def _get_plot_ax(self, ax=None, figsize=(12, 5)):
        if ax is not None:
            return ax, None
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        return ax, fig

    def _finalize_plot(self, ax, fig=None, legend=True):
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend()
        if fig is not None:
            fig.tight_layout()
        return ax

    def _summary_string(self):
        data = collections.OrderedDict({
            'Number of periods':
                self.u.shape[0],
            'Initial timestamp':
                self.h.index[0],
            'Final timestamp':
                self.h.index[-1],
            'Cumulative return (%)':
                self.cumulative_return * 100,
            'Annualized return (%)':
                self.annual_return,
            'Annualized volatility (%)':
                self.volatility * 100,
            'Sharpe ratio':
                self.sharpe_ratio,
            'Max. drawdown (%)':
                self.max_drawdown,
            'Average daily turnover (%)':
                self.turnover.mean() * 100,
            'Annualized turnover (%)':
                self.turnover.mean() * 100 * self.PPY,
            'Total turnover (%)':
                self.turnover.sum() * 100,
            'Average policy time (sec)':
                self.policy_time.mean(),
            'Average simulator time (sec)':
                self.simulation_time.mean(),
        })

        return (pd.Series(data=data).
                to_string(float_format='{:,.3f}'.format))

    def log_data(self, name, t, entry):
        try:
            getattr(self, name).loc[t] = entry
        except AttributeError:
            setattr(self, name,
                    (pd.Series if np.isscalar(entry) else
                     pd.DataFrame)(index=[t], data=[entry]))

    def log_policy(self, t, exec_time):
        self.log_data("policy_time", t, exec_time)
        # TODO mpo policy requires changes in the optimization_log methods
        if not isinstance(self.policy, MultiPeriodOpt):
            for cost in self.policy.costs:
                self.log_data("policy_" + cost.__class__.__name__,
                              t, cost.optimization_log(t))

    def log_simulation(self, t, u, h_next, risk_free_return, exec_time):
        self.log_data("simulation_time", t, exec_time)
        self.log_data("u", t, u)
        self.log_data("h_next", t, h_next)
        self.log_data("risk_free_returns", t, risk_free_return)
        for cost in self.simulator.costs:
            self.log_data("simulator_" + cost.__class__.__name__,
                          t, cost.simulation_log(t))

    @property
    def h(self):
        """
        Concatenate initial portfolio and h_next dataframe.

        """
        tmp = self.h_next.copy()
        tmp.loc['last'] = np.nan
        tmp = self.h_next.shift(1)
        tmp.iloc[0] = self.initial_portfolio
        # TODO fix ?
        # tmp.loc[self.h_next.index[-1] + self.timedelta]=self.h_next.iloc[-1]
        return tmp

    @property
    def v(self):
        """The value of the portfolio over time.
        """
        return self.h.sum(axis=1)

    @property
    def profit(self):
        """The profit made, in dollars."""
        return self.v[-1] - self.v[0]

    @property
    def cumulative_return(self):
        """Total return over the full backtest."""
        return self.v.iloc[-1] / self.v.iloc[0] - 1

    @property
    def cumulative_returns(self):
        """Cumulative return series."""
        return self.v / self.v.iloc[0] - 1.0

    @property
    def w(self):
        """The weights of the portfolio over time."""
        return (self.h.T / self.v).T

    @property
    def leverage(self):
        """Portfolio gross leverage including cash."""
        return np.abs(self.w).sum(1)

    @property
    def risky_leverage(self):
        """Gross leverage excluding the cash account."""
        risky_weights = self.w.drop(columns=[self.cash_key], errors='ignore')
        return np.abs(risky_weights).sum(1)

    @property
    def volatility(self):
        """The annualized, realized portfolio volatility."""
        return np.sqrt(self.PPY) * np.std(self.returns)

    @property
    def mean_return(self):
        """The annualized mean portfolio return."""
        return self.PPY * np.mean(self.returns)

    @property
    def returns(self):
        """The returns R_t = (v_{t+1}-v_t)/v_t
        """
        val = self.v
        return pd.Series(data=val.values[1:] / val.values[:-1] - 1,
                         index=val.index[:-1])

    @property
    def growth_rates(self):
        """The growth rate log(v_{t+1}/v_t)"""
        return np.log(self.returns + 1)

    @property
    def annual_growth_rate(self):
        """The annualized growth rate PPY/T sum_{t=1}^T log(v_{t+1}/v_t)
        """
        return self.growth_rates.sum() * self.PPY / self.growth_rates.size

    @property
    def annual_return(self):
        """The annualized return in percent.
        """
        ret = self.growth_rates
        return self._growth_to_return(ret.mean())

    def _growth_to_return(self, growth):
        """Convert growth to annualized percentage return.
        """
        return 100 * (np.exp(self.PPY * growth) - 1)

    def get_quarterly_returns(self, benchmark=None):
        """The annualized returns for each fiscal quarter.
        """
        ret = self.growth_rates
        quarters = ret.groupby(getFiscalQuarter).aggregate(np.mean)
        return self._growth_to_return(quarters)

    def get_best_quarter(self, benchmark=None):
        ret = self.get_quarterly_returns(benchmark)
        return (ret.argmax(), ret.max())

    def get_worst_quarter(self, benchmark=None):
        ret = self.get_quarterly_returns(benchmark)
        return (ret.argmin(), ret.min())

    @property
    def excess_returns(self):
        return self.returns - self.risk_free_returns

    @property
    def sharpe_ratio(self):
        return np.sqrt(self.PPY) * np.mean(self.excess_returns) / \
            np.std(self.excess_returns)

    @property
    def turnover(self):
        """Turnover ||u_t||_1/v_t
        """
        noncash_trades = self.u.drop(self.cash_key, axis=1)
        return np.abs(noncash_trades).sum(axis=1) / self.v

    @property
    def trading_days(self):
        """The fraction of days with nonzero turnover.
        """
        return (self.turnover.values > 0).sum() / self.turnover.size

    @property
    def max_drawdown(self):
        """The maximum peak to trough drawdown in percent.
        """
        val_arr = self.v.values
        max_dd_so_far = 0
        cur_max = val_arr[0]
        for val in val_arr[1:]:
            if val >= cur_max:
                cur_max = val
            elif 100 * (cur_max - val) / cur_max > max_dd_so_far:
                max_dd_so_far = 100 * (cur_max - val) / cur_max
        return max_dd_so_far

    @property
    def drawdown(self):
        """Running drawdown series in percent."""
        running_max = self.v.cummax()
        return 100 * (running_max - self.v) / running_max

    def top_holdings(self, top_n=15, use_abs=True):
        """Return time series of the top holdings by average weight."""
        weights = self.w.drop(columns=[self.cash_key], errors='ignore')
        if use_abs:
            ranking = weights.abs().mean().sort_values(ascending=False)
        else:
            ranking = weights.mean().sort_values(ascending=False)
        selected = ranking.head(top_n).index.tolist()
        return weights.loc[:, selected]

    def plot_value(self, ax=None, figsize=(12, 5), title='Portfolio Value',
                   ylabel='Portfolio value', label=None, linewidth=2):
        ax, fig = self._get_plot_ax(ax=ax, figsize=figsize)
        series = self.v
        ax.plot(series.index, series.values,
                label=label or 'Portfolio value', linewidth=linewidth)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        return self._finalize_plot(ax, fig=fig)

    def plot_leverage(self, ax=None, figsize=(12, 5), title='Leverage Over Time',
                      ylabel='Leverage (x)', label=None, linewidth=2,
                      include_cash=False):
        ax, fig = self._get_plot_ax(ax=ax, figsize=figsize)
        series = self.leverage if include_cash else self.risky_leverage
        ax.plot(series.index, series.values,
                label=label or 'Leverage', linewidth=linewidth)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        return self._finalize_plot(ax, fig=fig)

    def plot_drawdown(self, ax=None, figsize=(12, 5), title='Drawdown Over Time',
                      ylabel='Drawdown (%)', label=None, linewidth=2):
        ax, fig = self._get_plot_ax(ax=ax, figsize=figsize)
        series = -self.drawdown
        ax.fill_between(series.index, series.values, 0, alpha=0.25,
                        label=label or 'Drawdown')
        ax.plot(series.index, series.values,
                linewidth=linewidth)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        return self._finalize_plot(ax, fig=fig)

    def plot_top_holdings(self, top_n=15, use_abs=True, ax=None, figsize=(16, 7),
                          title=None, ylabel='Portfolio weight',
                          linewidth=1.6):
        ax, fig = self._get_plot_ax(ax=ax, figsize=figsize)
        weights = self.top_holdings(top_n=top_n, use_abs=use_abs)
        for col in weights.columns:
            ax.plot(weights.index, weights[col].values, label=str(col), linewidth=linewidth)
        ax.set_title(title or f'Top {top_n} Holdings Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        return self._finalize_plot(ax, fig=fig)

    @staticmethod
    def comparison_table(results_dict):
        """Build a comparison table for a dict of named SimulationResult objects."""
        return pd.DataFrame({
            name: {
                'Cumulative return (%)': result.cumulative_return * 100,
                'Annualized return (%)': result.annual_return,
                'Annualized volatility (%)': result.volatility * 100,
                'Sharpe ratio': result.sharpe_ratio,
                'Max. drawdown (%)': result.max_drawdown,
                'Average daily turnover (%)': result.turnover.mean() * 100,
                'Annualized turnover (%)': result.turnover.mean() * 100 * result.PPY,
                'Total turnover (%)': result.turnover.sum() * 100,
                'Average risky leverage (x)': result.risky_leverage.mean(),
            }
            for name, result in results_dict.items()
        }).round(3)

    @staticmethod
    def _comparison_ax(ax=None, figsize=(15, 7)):
        if ax is not None:
            return ax, None
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        return ax, fig

    @classmethod
    def plot_value_compare(cls, results_dict, ax=None, figsize=(15, 7),
                           title='Portfolio Value Comparison',
                           ylabel='Portfolio value', linewidth=2.2):
        ax, fig = cls._comparison_ax(ax=ax, figsize=figsize)
        for name, result in results_dict.items():
            ax.plot(result.v.index, result.v.values, label=str(name), linewidth=linewidth)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        if fig is not None:
            fig.tight_layout()
        return ax

    @classmethod
    def plot_leverage_compare(cls, results_dict, ax=None, figsize=(15, 7),
                              title='Leverage Comparison',
                              ylabel='Leverage (x)', linewidth=2.0,
                              leverage_limits=None, include_cash=False):
        ax, fig = cls._comparison_ax(ax=ax, figsize=figsize)
        for name, result in results_dict.items():
            series = result.leverage if include_cash else result.risky_leverage
            ax.plot(series.index, series.values,
                    label=str(name), linewidth=linewidth)
        if leverage_limits is not None:
            for label, value in leverage_limits.items():
                ax.axhline(value, linestyle=':', linewidth=1.8, label=str(label))
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        if fig is not None:
            fig.tight_layout()
        return ax

    @classmethod
    def plot_drawdown_compare(cls, results_dict, ax=None, figsize=(15, 7),
                              title='Drawdown Comparison',
                              ylabel='Drawdown (%)', linewidth=1.6,
                              alpha_fill=0.22):
        ax, fig = cls._comparison_ax(ax=ax, figsize=figsize)
        for name, result in results_dict.items():
            series = -result.drawdown
            ax.fill_between(series.index, series.values, 0, alpha=alpha_fill, label=str(name))
            ax.plot(series.index, series.values, linewidth=linewidth)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='lower left')
        if fig is not None:
            fig.tight_layout()
        return ax

    @classmethod
    def plot_top_holdings_compare(cls, results_dict, top_n=15, use_abs=True,
                                  figsize=(16, 7), ylabel='Portfolio weight'):
        """Plot top holdings for each strategy in separate figures."""
        axes = {}
        for name, result in results_dict.items():
            axes[name] = result.plot_top_holdings(
                top_n=top_n,
                use_abs=use_abs,
                figsize=figsize,
                title=f'{name} Top {top_n} Holdings Over Time',
                ylabel=ylabel,
            )
        return axes
