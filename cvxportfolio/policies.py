from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import logging
import cvxpy as cvx
from datetime import datetime

from cvxportfolio.costs import BaseCost
from cvxportfolio.returns import BaseReturnsModel
from cvxportfolio.constraints import BaseConstraint
from cvxportfolio.utils import values_in_time, null_checker

__all__ = ['Hold', 'FixedTrade', 'PeriodicRebalance',   
           'SinglePeriodOpt', 'MultiPeriodOpt', 'ProportionalTrade',
           'RankAndLongShort','FuturesRoundTrade', 'FixedWeights']

class BasePolicy(object, metaclass=ABCMeta):
    def __init__(self):
        self.costs = []
        self.constraints = []

    @abstractmethod
    def get_trades(self, portfolio, t=datetime.today()):
        return NotImplemented

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.)

    def get_rounded_trades(self, portfolio, prices, t):
        return np.round(self.get_trades(portfolio, t) / values_in_time(prices, t))[:-1]

class Hold(BasePolicy):
    def get_trades(self, portfolio, t=datetime.today()):
        return self._nulltrade(portfolio)

class RankAndLongShort(BasePolicy):
    def __init__(self, return_forecast, num_long, num_short, 
                 target_turnover, target_leverage=1.0):
        self.target_turnover = target_turnover
        self.num_long = num_long
        self.num_short = num_short
        self.target_leverage = target_leverage 
        self.return_forecast = return_forecast
        super().__init__()

    def get_trades(self, portfolio, t=datetime.today()):
        prediction = values_in_time(self.return_forecast, t)
        sorted_ret = prediction.sort_values()

        short_trades = sorted_ret.index[:self.num_short]
        long_trades = sorted_ret.index[-self.num_long:]

        u = pd.Series(0., index=prediction.index)
        u[short_trades] = -1.
        u[long_trades] = 1.
        u /= sum(abs(u))
        u *= self.target_leverage
        u = sum(portfolio) * u * self.target_turnover

        return u

class ProportionalTrade(BasePolicy):
    def __init__(self, targetweight, time_steps):
        self.targetweight = targetweight
        self.time_steps = time_steps
        super().__init__()

    def get_trades(self, portfolio, t=datetime.today()):
        try:
            missing_time_steps = len(self.time_steps) - next(i for (i, x) in enumerate(self.time_steps) if x == t)
        except StopIteration:
            raise Exception("ProportionalTrade can only trade on the given time steps")
        deviation = self.targetweight - portfolio / sum(portfolio)
        return sum(portfolio) * deviation / missing_time_steps

class SellAll(BasePolicy):
    def get_trades(self, portfolio, t=datetime.today()):
        trade = -pd.Series(portfolio, copy=True)
        trade.ix[-1] = 0.
        return trade

class FixedTrade(BasePolicy):
    def __init__(self, tradevec=None, tradeweight=None):
        if tradevec is not None and tradeweight is not None:
            raise Exception
        if tradevec is None and tradeweight is None:
            raise Exception
        self.tradevec = tradevec
        self.tradeweight = tradeweight
        assert(self.tradevec is None or sum(self.tradevec) == 0.)
        assert(self.tradeweight is None or sum(self.tradeweight) == 0.)
        super().__init__()

    def get_trades(self, portfolio, t=datetime.today()):
        if self.tradevec is not None:
            return self.tradevec
        return sum(portfolio) * self.tradeweight

class BaseRebalance(BasePolicy):
    def _rebalance(self, portfolio):
        return sum(portfolio) * self.target - portfolio

class PeriodicRebalance(BaseRebalance):
    def __init__(self, target, period, **kwargs):
        self.target = target
        self.period = period
        super().__init__()

    def is_start_period(self, t):
        result = not getattr(t, self.period) == getattr(self.last_t, self.period) if hasattr(self, 'last_t') else True
        self.last_t = t
        return result

    def get_trades(self, portfolio, t=datetime.today()):
        return self._rebalance(portfolio) if self.is_start_period(t) else self._nulltrade(portfolio)

class SinglePeriodOpt(BasePolicy):
    def __init__(self, return_forecast, costs, constraints, solver=None, solver_opts=None):
        super().__init__()
        if not hasattr(return_forecast, 'weight_expr'):
            null_checker(return_forecast)
        self.return_forecast = return_forecast
        
        self.costs = []
        for cost in costs:
            assert isinstance(cost, BaseCost) or (
                hasattr(cost, 'weight_expr') and
                hasattr(cost, 'value_expr')
            )
            self.costs.append(cost)

        self.constraints = []
        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint) or hasattr(constraint, 'weight_expr')
            self.constraints.append(constraint)

        self.solver = solver if solver else 'OSQP'
        self.solver_opts = solver_opts if solver_opts else {'verbose': False}

    def get_trades(self, portfolio, t=None):
        if t is None:
            t = datetime.now()

        value = sum(portfolio)
        w = portfolio / value
        z = cvx.Variable(w.size) 
        wplus = w.values + z

        if hasattr(self.return_forecast, 'weight_expr'):
            alpha_term = self.return_forecast.weight_expr(t, wplus, z, value)
        else:
            alpha_term = cvx.sum(cvx.multiply(values_in_time(self.return_forecast, t).values, wplus))

        assert(alpha_term.is_concave())

        costs_exprs = []
        constraints = [cvx.sum(z) == 0]

        for cost in self.costs:
            c_expr, c_constr = cost.weight_expr(t, wplus, z, value)
            costs_exprs.append(c_expr)
            constraints += c_constr

        for con in self.constraints:
            c_expr = con.weight_expr(t, wplus, z, value)
            if isinstance(c_expr, list):
                constraints += c_expr
            else:
                constraints.append(c_expr)

        obj = cvx.Maximize(alpha_term - sum(costs_exprs))
        self.prob = cvx.Problem(obj, constraints)

        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)
            if self.prob.status in ['unbounded', 'infeasible']:
                logging.error(f'Optimization failed: {self.prob.status}. No trades.')
                return pd.Series(0., index=portfolio.index)
            return pd.Series(index=portfolio.index, data=(z.value * value))
        except cvx.SolverError as e:
            logging.error(f'Solver {self.solver} failed: {e}')
            return pd.Series(0., index=portfolio.index)

class FuturesRoundTrade(BasePolicy):
    """
    [Wrapper Policy] 期货手数取整策略
    直接从本地 targetinfo.csv 读取第三列(multiplier)
    """
    def __init__(self, base_policy, target_info_path, prices, round_mode='floor', cash_key='CNY'):
        super().__init__()
        self.base_policy = base_policy
        self.prices = prices
        self.round_mode = round_mode
        self.cash_key = cash_key
        self.multipliers = self._parse_multipliers(target_info_path)
        self.costs = getattr(base_policy, 'costs', [])
        self.constraints = getattr(base_policy, 'constraints', [])

    def _parse_multipliers(self, path):
        df = pd.read_csv(path)
        tickers = df.iloc[:, 0].astype(str).str.strip()
        multipliers = df.iloc[:, 2].astype(float)
        return dict(zip(tickers, multipliers))

    def _recursive_values_in_time(self, t, **kwargs):
        if hasattr(self.base_policy, '_recursive_values_in_time'):
            self.base_policy._recursive_values_in_time(t, **kwargs)

    def get_trades(self, portfolio, t=None):
        raw_trades = self.base_policy.get_trades(portfolio, t)
        if raw_trades.abs().sum() < 1e-4:
            return raw_trades

        rounded_trades = pd.Series(0.0, index=portfolio.index)
        portfolio_value = sum(portfolio)
        
        for asset in portfolio.index:
            if asset == self.cash_key:
                continue 
            
            multiplier = self.multipliers.get(asset)
            if multiplier is None:
                rounded_trades[asset] = raw_trades[asset]
                continue

            try:
                P = values_in_time(self.prices[asset], t)
            except KeyError:
                rounded_trades[asset] = 0.0
                continue
                
            if P <= 0: continue

            current_dollars = portfolio[asset]
            trade_dollars = raw_trades[asset]
            target_dollars = current_dollars + trade_dollars
            target_hands_float = target_dollars / (P * multiplier)
            
            if self.round_mode == 'floor':
                target_hands_int = int(abs(target_hands_float)) * np.sign(target_hands_float)
            else:
                target_hands_int = int(round(target_hands_float))
                
            real_target_dollars = target_hands_int * P * multiplier
            rounded_trades[asset] = real_target_dollars - current_dollars

        non_cash_trades_sum = rounded_trades.drop(self.cash_key, errors='ignore').sum()
        rounded_trades[self.cash_key] = -non_cash_trades_sum
        
        return rounded_trades
    
class MultiPeriodOpt(SinglePeriodOpt):
    def __init__(self, trading_times, terminal_weights, lookahead_periods=None, *args, **kwargs):
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        self.terminal_weights = terminal_weights
        super().__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=datetime.today()):
        value = sum(portfolio)
        if value <= 0:
            logging.getLogger().warning(f"Portfolio value non-positive at {t}: {value}. Returning zero trades.")
            return self._nulltrade(portfolio)
        w = cvx.Constant(portfolio.values / value)

        total_obj_expr = 0
        total_constraints = []
        first_z = None

        try:
            start_idx = self.trading_times.index(t)
        except ValueError:
            return self._nulltrade(portfolio)
            
        end_idx = start_idx + self.lookahead_periods if self.lookahead_periods else len(self.trading_times)
        planning_times = self.trading_times[start_idx:end_idx]

        for i, tau in enumerate(planning_times):
            z = cvx.Variable(*w.shape)
            if i == 0:
                first_z = z 
            wplus = w + z
            
            obj = self.return_forecast.weight_expr_ahead(t, tau, wplus)

            costs, constr = [], []
            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr_ahead(t, tau, wplus, z, value)
                costs.append(cost_expr)
                constr += const_expr

            obj -= sum(costs)
            constr += [cvx.sum(wplus) == 1]
            
            for con in self.constraints:
                c_expr = con.weight_expr(t, wplus, z, value)
                if isinstance(c_expr, list):
                    constr += c_expr
                else:
                    constr.append(c_expr)

            total_obj_expr += obj
            total_constraints += constr
            w = wplus

        if self.terminal_weights is not None:
            total_constraints += [wplus == self.terminal_weights.values]

        prob = cvx.Problem(cvx.Maximize(total_obj_expr), total_constraints)
        
        try:
            prob.solve(solver=self.solver, **self.solver_opts)
        except cvx.SolverError as e:
            logging.exception("Solver failed: %s", e)
            return pd.Series(0.0, index=portfolio.index)

        try:
            self._last_prob = prob
        except Exception:
            self._last_prob = None

        try:
            self._last_first_z_value = None if first_z is None else first_z.value
        except Exception:
            self._last_first_z_value = None

        if first_z is None or self._last_first_z_value is None:
            return pd.Series(0.0, index=portfolio.index)

        return pd.Series(index=portfolio.index, data=(self._last_first_z_value * value))

class FixedWeights(BasePolicy):
    def __init__(self, target_weights):
        super().__init__()
        self.target_weights = target_weights

    def get_trades(self, portfolio, t=datetime.today()):
        total_equity = portfolio.sum()
        target_holdings = total_equity * self.target_weights
        target_holdings = target_holdings.reindex(portfolio.index).fillna(0.0)
        trades = target_holdings - portfolio
        return trades
    
class DynamicFixedWeights(BasePolicy):
    def __init__(self, target_weights_df):
        super().__init__()
        self.target_weights_df = target_weights_df

    def get_trades(self, portfolio, t):
        total_equity = portfolio.sum()
        current_target_weights = self.target_weights_df.loc[t]
        target_holdings = total_equity * current_target_weights
        target_holdings = target_holdings.reindex(portfolio.index).fillna(0.0)
        trades = target_holdings - portfolio
        return trades
    
class RobustMultiPeriodOpt(MultiPeriodOpt):
    def get_trades(self, portfolio, t):
        try:
            return super().get_trades(portfolio, t)
        except (cvx.SolverError, TypeError, Exception) as e:
            logging.exception("Optimizer failed at %s: %s", t, e)
            return pd.Series(0.0, index=portfolio.index)
