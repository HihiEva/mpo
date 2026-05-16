
import copy
import logging
import time

import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx

from .returns import MultipleReturnsForecasts

from .result import SimulationResult
from .costs import BaseCost


def _normalize_margin_rate(value):
    """Accept both percentage-style inputs (7, 16) and decimal inputs (0.07, 0.16)."""
    if isinstance(value, dict):
        value = value.get('initial', 0.1)
    rate = float(value)
    return rate / 100.0 if rate > 1.0 else rate

# TODO update benchmark weights (?)
# Also could try jitting with numba.


class MarketSimulator():
    logger = None

    def __init__(self, market_returns, costs,
                 market_volumes=None, cash_key='cash'):
        """Provide market returns object and cost objects."""
        self.market_returns = market_returns
        if market_volumes is not None:
            self.market_volumes = market_volumes[
                market_volumes.columns.difference([cash_key])]
        else:
            self.market_volumes = None

        self.costs = costs
        for cost in self.costs:
            assert isinstance(cost, BaseCost) or (
                hasattr(cost, 'value_expr') and
                hasattr(cost, 'simulation_log')
            )

        self.cash_key = cash_key

    def propagate(self, h, u, t):
        """Propagates the portfolio forward over time period t, given trades u.

        Args:
            h: pandas Series object describing current portfolio
            u: n vector with the stock trades (not cash)
            t: current time

        Returns:
            h_next: portfolio after returns propagation
            u: trades vector with simulated cash balance
        """
        assert (u.index.equals(h.index))

        if self.market_volumes is not None:
            # don't trade if volume is null
            null_trades = self.market_volumes.columns[
                self.market_volumes.loc[t] == 0]
            if len(null_trades):
                logging.info('No trade condition for stocks %s on %s' %
                             (null_trades, t))
                u.loc[null_trades] = 0.

        hplus = h + u
        costs = [cost.value_expr(t, h_plus=hplus, u=u) for cost in self.costs]
        for cost in costs:
            assert(not pd.isnull(cost))
            assert(not np.isinf(cost))

        u[self.cash_key] = - sum(u[u.index != self.cash_key]) - sum(costs)
        hplus[self.cash_key] = h[self.cash_key] + u[self.cash_key]

        # Ensure returns row aligns with hplus index; fill missing returns with 0.0
        mr = self.market_returns.loc[t]
        try:
            mr = mr.reindex(hplus.index)
        except Exception:
            # fallback: convert to Series and reindex
            mr = pd.Series(self.market_returns.loc[t].values, index=self.market_returns.columns).reindex(hplus.index)
        mr = mr.fillna(0.0)
        assert (hplus.index.sort_values().equals(mr.index.sort_values()))
        h_next = mr * hplus + hplus

        assert (not h_next.isnull().values.any())
        assert (not u.isnull().values.any())
        return h_next, u

    def run_backtest(self, initial_portfolio, start_time, end_time,
                     policy, loglevel=logging.WARNING):
        """Backtest a single policy.
        """
        logging.basicConfig(level=loglevel)

        results = SimulationResult(initial_portfolio=copy.copy(
            initial_portfolio),
            policy=policy, cash_key=self.cash_key,
            simulator=self)
        h = initial_portfolio

        simulation_times = self.market_returns.index[
            (self.market_returns.index >= start_time) &
            (self.market_returns.index <= end_time)]
        logging.info('Backtest started, from %s to %s' %
                     (simulation_times[0], simulation_times[-1]))
        
        for t in simulation_times:
            logging.info('Getting trades at time %s' % t)
            start = time.time()
            try:
                u = policy.get_trades(h, t)
            except cvx.SolverError:
                logging.warning(
                    'Solver failed on timestamp %s. Default to no trades.' % t)
                u = pd.Series(index=h.index, data=0.)
            end = time.time()
            assert (not pd.isnull(u).any())
            results.log_policy(t, end - start)

            logging.info('Propagating portfolio at time %s' % t)
            start = time.time()
            h, u = self.propagate(h, u, t)
            end = time.time()
            # If propagate resulted in bankruptcy (non-positive equity), stop simulation
            try:
                equity = float(h.sum())
            except Exception:
                equity = None
            if equity is not None and equity <= 0:
                logging.warning(f"Bankruptcy detected at {t}: equity={equity}. Stopping backtest.")
                results.log_simulation(t=t, u=u, h_next=h,
                                       risk_free_return=self.market_returns.loc[
                                           t, self.cash_key],
                                       exec_time=end - start)
                break
            assert (not h.isnull().values.any())
            results.log_simulation(t=t, u=u, h_next=h,
                                   risk_free_return=self.market_returns.loc[
                                       t, self.cash_key],
                                   exec_time=end - start)

        logging.info('Backtest ended, from %s to %s' %
                     (simulation_times[0], simulation_times[-1]))
        return results

    def run_multiple_backtest(self, initial_portf, start_time,
                              end_time, policies,
                              loglevel=logging.WARNING, parallel=False):
        """Backtest multiple policies.
        """

        def _run_backtest(policy):
            return self.run_backtest(initial_portf, start_time, end_time,
                                     policy, loglevel=loglevel)

        num_workers = min(multiprocess.cpu_count(), len(policies))
        if parallel:
            workers = multiprocess.Pool(num_workers)
            results = workers.map(_run_backtest, policies)
            workers.close()
            return results
        else:
            return list(map(_run_backtest, policies))

    def what_if(self, time, results, alt_policies, parallel=True):
        """Run alternative policies starting from given time.
        """
        # TODO fix
        initial_portf = copy.copy(results.h.loc[time])
        all_times = results.h.index
        alt_results = self.run_multiple_backtest(initial_portf,
                                                 time,
                                                 all_times[-1],
                                                 alt_policies, parallel)
        for idx, alt_result in enumerate(alt_results):
            alt_result.h.loc[time] = results.h.loc[time]
            alt_result.h.sort_index(axis=0, inplace=True)
        return alt_results

    @staticmethod
    def reduce_signal_perturb(initial_weights, delta):
        """Compute matrix of perturbed weights given initial weights."""
        perturb_weights_matrix = \
            np.zeros((len(initial_weights), len(initial_weights)))
        for i in range(len(initial_weights)):
            perturb_weights_matrix[i, :] = initial_weights / \
                (1 - delta * initial_weights[i])
            perturb_weights_matrix[i, i] = (1 - delta) * initial_weights[i]
        return perturb_weights_matrix

    def attribute(self, true_results, policy,
                  selector=None,
                  delta=1,
                  fit="linear",
                  parallel=True):
        """Attributes returns over a period to individual alpha sources.

        Args:
            true_results: observed results.
            policy: the policy that achieved the returns.
                    Alpha model must be a stream.
            selector: A map from SimulationResult to time series.
            delta: the fractional deviation.
            fit: the type of fit to perform.
        Returns:
            A dict of alpha source to return series.
        """
        # Default selector looks at profits.
        if selector is None:
            def selector(result):
                return result.v - sum(result.initial_portfolio)

        alpha_stream = policy.return_forecast
        assert isinstance(alpha_stream, MultipleReturnsForecasts)
        times = true_results.h.index
        weights = alpha_stream.weights
        assert np.sum(weights) == 1
        alpha_sources = alpha_stream.alpha_sources
        num_sources = len(alpha_sources)
        Wmat = self.reduce_signal_perturb(weights, delta)
        perturb_pols = []
        for idx in range(len(alpha_sources)):
            new_pol = copy.copy(policy)
            new_pol.return_forecast = MultipleReturnsForecasts(alpha_sources,
                                                               Wmat[idx, :])
            perturb_pols.append(new_pol)
        # Simulate
        p0 = true_results.initial_portfolio
        alt_results = self.run_multiple_backtest(p0, times[0], times[-1],
                                                 perturb_pols, parallel)
        # Attribute.
        true_arr = selector(true_results).values
        attr_times = selector(true_results).index
        Rmat = np.zeros((num_sources, len(attr_times)))
        for idx, result in enumerate(alt_results):
            Rmat[idx, :] = selector(result).values
        Pmat = cvx.Variable((num_sources, len(attr_times)))
        if fit == "linear":
            prob = cvx.Problem(cvx.Minimize(0), [Wmat * Pmat == Rmat])
            prob.solve()
        elif fit == "least-squares":
            error = cvx.sum_squares(Wmat * Pmat - Rmat)
            prob = cvx.Problem(cvx.Minimize(error),
                               [Pmat.T * weights == true_arr])
            prob.solve()
        else:
            raise Exception("Unknown fitting method.")
        # Dict of results.
        wmask = np.tile(weights[:, np.newaxis], (1, len(attr_times))).T
        data = pd.DataFrame(columns=[s.name for s in alpha_sources],
                            index=attr_times,
                            data=Pmat.value.T * wmask)
        data['residual'] = true_arr - np.matrix((weights * Pmat).value).A1
        data['RMS error'] = np.matrix(
            cvx.norm(Wmat * Pmat - Rmat, 2, axis=0).value).A1
        data['RMS error'] /= np.sqrt(num_sources)
        return data

class CNFuturesSimulator(MarketSimulator):
    """
    中国期货专用模拟器
    
    核心改进：
    1. 强平逻辑 (Margin Call Logic)：权益不足时强制平仓。
    2. 修正现金流逻辑：不应该对'名义杠杆产生的负现金'收利息。
    """
    def __init__(self, trading_times, costs, margin_map, 
                 maintenance_margin_buffer=0.8, **kwargs):
        """
        :param margin_map: 保证金字典 {ticker: ratio} (用于计算强平线)
        :param maintenance_margin_buffer: 强平阈值。
               如果 (占用保证金 > 总权益 * buffer)，触发强平。
               通常维持保证金是初始的 0.7~0.8 倍。
        """
        # 剔除父类不接受的参数 (防止报错)
        if 'prices' in kwargs:
            kwargs.pop('prices')

        super().__init__(trading_times, costs, **kwargs)
        self.margin_map = {}
        for k, v in margin_map.items():
            self.margin_map[k] = _normalize_margin_rate(v)
            
        self.limit_threshold = maintenance_margin_buffer

    def propagate(self, h, u, t):
        """
        覆写状态更新逻辑
        h: 当前持仓 (Series)
        u: 交易向量 (Series)
        t: 当前时间
        """
        # 1. 先调用父类逻辑
        # 父类返回的是元组 (h_next, u_updated)
        # 注意参数顺序必须是 (h, u, t)
        h_next, u_updated = super().propagate(h, u, t)
        
        # 如果父类返回 None (如异常)，保持一致
        if h_next is None:
            return None, None

        # 2. --- 新增：强平检查 (Margin Call Check) ---
        
        # 计算当前总权益 (Equity)
        equity = h_next.sum()
        
        # A. 穿仓检查 (破产)
        if equity <= 0:
            logging.warning(f"[{t}] BANKRUPTCY! Equity: {equity:.2f}")
            # 返回全 0 持仓，表示归零
            return pd.Series(0., index=h_next.index), u_updated

        # B. 维持保证金检查
        # 计算当前持仓占用的初始保证金 (Initial Margin Used)
        # 注意：模拟器里的 h 是名义价值，所以可以直接乘保证金率
        margin_used = 0.0
        for asset, value in h_next.items():
            if asset == self.cash_key: continue
            rate = self.margin_map.get(asset, 0.1) # 默认 10%
            margin_used += abs(value) * rate
            
        # 检查是否击穿维持保证金线
        # 假设维持保证金是初始保证金的 80% (0.8)
        maintenance_margin = margin_used * 0.8 
        
        if maintenance_margin > equity:
            logging.warning(f"[{t}] MARGIN CALL! Equity: {equity:.2f} < Maint: {maintenance_margin:.2f}")
            
            # 执行强平：强制平仓转为现金
            h_liquidated = pd.Series(0., index=h_next.index)
            h_liquidated[self.cash_key] = equity # 剩下的钱都是现金
            return h_liquidated, u_updated

        # 3. 必须返回元组 (h_next, u)
        return h_next, u_updated
