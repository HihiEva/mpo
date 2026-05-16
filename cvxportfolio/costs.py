import copy
import re
from datetime import timedelta

import cvxpy as cvx
import numpy as np
import pandas as pd

from .expression import Expression
from .utils import values_in_time

__all__ = [
    'CNFuturesCommission',
    'CNFuturesTickSlippage',
    'CNFuturesVolSlippage',
    'CNFuturesSlippage',
    'CNFuturesMarginCost',
    'CNFuturesMPOVolumeSlippage',
]


def _normalize_margin_rate(value):
    """Accept both percentage-style inputs (7, 16) and decimal inputs (0.07, 0.16)."""
    if isinstance(value, dict):
        value = value.get('initial', 0.1)
    rate = float(value)
    return rate / 100.0 if rate > 1.0 else rate


def _infer_asset_index(data):
    if isinstance(data, dict) and data:
        sample = next(iter(data.values()))
        if hasattr(sample, 'index'):
            return sample.index
    if hasattr(data, 'columns'):
        return data.columns
    if hasattr(data, 'index'):
        return data.index
    return []


def _get_series_from_forecast(data, origin_t, target_t, volume_cap=None):
    """When data is keyed by (origin, target), always use forecasted values."""
    if not isinstance(data, dict):
        if isinstance(data, pd.Series):
            return data
        try:
            return values_in_time(data, target_t)
        except Exception:
            return None

    adjusted_target = target_t
    if volume_cap is not None and target_t > origin_t:
        capped_target = origin_t + timedelta(days=volume_cap)
        if target_t > capped_target:
            adjusted_target = capped_target

    # For future planning horizons, use forecasts generated at the current origin.
    if adjusted_target > origin_t:
        if (origin_t, adjusted_target) in data:
            return data[(origin_t, adjusted_target)]
        candidates = [tau for (t0, tau) in data.keys() if t0 == origin_t]
        if candidates:
            closest_tau = min(candidates, key=lambda tau: abs((tau - adjusted_target).days))
            return data[(origin_t, closest_tau)]
        return None

    # For the first stage / simulation at date t, use the latest forecast made before t for target t.
    prior_candidates = [
        (t0, tau) for (t0, tau) in data.keys()
        if tau == target_t and t0 < target_t
    ]
    if not prior_candidates:
        prior_candidates = [
            (t0, tau) for (t0, tau) in data.keys()
            if tau == target_t and t0 <= target_t
        ]
    if not prior_candidates:
        return None
    latest_origin = max(t0 for (t0, _) in prior_candidates)
    return data[(latest_origin, target_t)]


class BaseCost(Expression):

    def __init__(self):
        self.gamma = 1.
        self.cost_logs = {}

    def weight_expr(self, t, w_plus, z, value):
        cost, constr = self._estimate(t, w_plus, z, value)
        return self.gamma * cost, constr

    def weight_expr_ahead(self, t, tau, w_plus, z, value):
        cost, constr = self._estimate_ahead(t, tau, w_plus, z, value)
        return self.gamma * cost, constr

    def __mul__(self, other):
        newobj = copy.copy(self)
        newobj.gamma *= other
        return newobj

    def __rmul__(self, other):
        return self.__mul__(other)

    def simulation_log(self, t):
        return self.cost_logs.get(t, 0.0)

    def optimization_log(self, t):
        return self.cost_logs.get(t, 0.0)

    def _estimate_ahead(self, t, tau, w_plus, z, value):
        return self._estimate(tau, w_plus, z, value)


class CNFuturesCommission(BaseCost):
    """China futures commission model loaded from daily CSV files."""

    def __init__(self, commission_dir, target_info_path, prices, cash_key='CNY'):
        super(CNFuturesCommission, self).__init__()
        self.prices = prices
        self.cash_key = cash_key
        self.commission_dir = commission_dir
        self.multipliers = self._parse_multipliers(target_info_path)
        self._current_df = None
        self._current_date = None

    def _parse_multipliers(self, path):
        df = pd.read_csv(path)
        tickers = df.iloc[:, 0].astype(str).str.strip()
        multipliers = df.iloc[:, 2].astype(float)
        return dict(zip(tickers, multipliers))

    def _normalize_asset_code(self, code):
        return re.sub(r'\d+', '', str(code).strip().lower())

    def _prepare_daily_fee_table(self, df):
        df = df.copy()
        instrument_col = df.columns[0]
        df[instrument_col] = df[instrument_col].astype(str).str.strip().map(self._normalize_asset_code)
        numeric_cols = [
            'open_ratio_by_money',
            'open_fee_by_volume',
            'close_ratio_by_money',
            'close_fee_by_volume',
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df.groupby(instrument_col, sort=False)[numeric_cols].mean()

    def _get_daily_fee_table(self, t):
        date_str = t.strftime('%Y%m%d')
        if self._current_date == date_str:
            return self._current_df

        import os
        file_path = os.path.join(self.commission_dir, f"{date_str}.csv")

        try:
            df = pd.read_csv(file_path)
            self._current_df = self._prepare_daily_fee_table(df)
            self._current_date = date_str
            return self._current_df
        except FileNotFoundError:
            return None

    def _get_average_rate(self, t, asset):
        df = self._get_daily_fee_table(t)
        asset_key = self._normalize_asset_code(asset)
        if df is None or asset_key not in df.index:
            return 0.0

        row = df.loc[asset_key]
        ratio = (row['open_ratio_by_money'] + row['close_ratio_by_money']) / 2.0
        fixed = (row['open_fee_by_volume'] + row['close_fee_by_volume']) / 2.0

        rate = ratio
        try:
            price = values_in_time(self.prices[asset], t)
            multiplier = self.multipliers.get(asset, 1.0)
            if fixed > 0 and price > 0 and multiplier > 0:
                rate += fixed / (price * multiplier)
        except Exception:
            pass
        return rate

    def _estimate(self, t, w_plus, z, value):
        assets = self.prices.columns
        coeffs = []
        for asset in assets:
            coeffs.append(0.0 if asset == self.cash_key else self._get_average_rate(t, asset))
        c_vec = cvx.Parameter(len(coeffs), value=np.array(coeffs), nonneg=True)
        # In optimization we use normalized trade weights z, consistent with the
        # paper's single-period objective. The simulator still applies costs on
        # notional trade amounts via value_expr.
        return c_vec.T @ cvx.abs(z), []

    def value_expr(self, t, h_plus, u):
        u_val = u.values if hasattr(u, 'values') else u
        cost = 0.0
        for i, asset in enumerate(self.prices.columns):
            trade_amt = abs(u[asset] if hasattr(u, 'index') else u_val[i])
            if trade_amt < 1e-6 or asset == self.cash_key:
                continue
            cost += trade_amt * self._get_average_rate(t, asset)
        self.cost_logs[t] = cost
        return cost


class CNFuturesTickSlippage(BaseCost):
    """Tick-based slippage: cost = (k * tick / price) * |trade_amount|."""

    def __init__(self, prices, target_info_path, k=1.0, cash_key='CNY'):
        super(CNFuturesTickSlippage, self).__init__()
        self.prices = prices
        self.k = k
        self.cash_key = cash_key
        self.tick_sizes = self._parse_minimove(target_info_path)

    def _parse_minimove(self, path):
        df = pd.read_csv(path)
        tickers = df.iloc[:, 0].astype(str).str.strip()
        minimoves = df.iloc[:, 1].astype(float)
        return dict(zip(tickers, minimoves))

    def _linear_rates(self, t):
        rates = []
        for asset in self.prices.columns:
            if asset == self.cash_key:
                rates.append(0.0)
                continue
            try:
                price = values_in_time(self.prices[asset], t)
                tick = self.tick_sizes.get(asset, 0.0)
                rates.append((self.k * tick) / price if price > 0 else 0.0)
            except Exception:
                rates.append(0.0)
        return np.array(rates, dtype=float)

    def _estimate(self, t, w_plus, z, value):
        rates = self._linear_rates(t)
        c_vec = cvx.Parameter(len(rates), value=rates, nonneg=True)
        return c_vec.T @ cvx.abs(z), []

    def value_expr(self, t, h_plus, u):
        u_val = u.values if hasattr(u, 'values') else u
        rates = self._linear_rates(t)
        cost = 0.0
        for i, asset in enumerate(self.prices.columns):
            trade_val = abs(u[asset] if hasattr(u, 'index') else u_val[i])
            if trade_val < 1e-6 or asset == self.cash_key:
                continue
            cost += rates[i] * trade_val
        self.cost_logs[t] = cost
        return cost


class CNFuturesVolSlippage(BaseCost):
    """Volatility-based slippage: cost = k * sigma * |trade_amount|."""

    def __init__(self, sigmas, k=1.0, cash_key='CNY'):
        super(CNFuturesVolSlippage, self).__init__()
        self.sigmas = sigmas
        self.k = k
        self.cash_key = cash_key

    def _assets(self):
        return _infer_asset_index(self.sigmas)

    def _get_sigma_series(self, t, tau):
        return _get_series_from_forecast(self.sigmas, t, tau)

    def _linear_rates(self, t, tau):
        sigma_series = self._get_sigma_series(t, tau)
        rates = []
        for asset in self._assets():
            if asset == self.cash_key:
                rates.append(0.0)
                continue
            sigma = 0.0
            try:
                if sigma_series is not None:
                    sigma = float(sigma_series.get(asset, 0.0))
            except Exception:
                sigma = 0.0
            rates.append(self.k * sigma)
        return np.array(rates, dtype=float)

    def _estimate_ahead(self, t, tau, w_plus, z, value):
        rates = self._linear_rates(t, tau)
        c_vec = cvx.Parameter(len(rates), value=rates, nonneg=True)
        return c_vec.T @ cvx.abs(z), []

    def _estimate(self, t, w_plus, z, value):
        return self._estimate_ahead(t, t, w_plus, z, value)

    def value_expr(self, t, h_plus, u):
        u_val = u.values if hasattr(u, 'values') else u
        rates = self._linear_rates(t, t)
        assets = self._assets()
        cost = 0.0
        for i, asset in enumerate(assets):
            trade_val = abs(u[asset] if hasattr(u, 'index') else u_val[i])
            if trade_val < 1e-6 or asset == self.cash_key:
                continue
            cost += rates[i] * trade_val
        self.cost_logs[t] = cost
        return cost


class CNFuturesMPOVolumeSlippage(BaseCost):
    """Forecast volume market impact: c * |trade_amount|^1.5, c = Y * sigma / sqrt(V)."""

    def __init__(self, mpo_volume_data, sigmas, volume_cap=None, config=None, cash_key='CNY'):
        super(CNFuturesMPOVolumeSlippage, self).__init__()
        self.mpo_volume_data = mpo_volume_data
        self.sigmas = sigmas
        self.volume_cap = volume_cap
        self.config = config if config else {'Y': 1.0}
        self.cash_key = cash_key

    def _assets(self):
        assets = list(_infer_asset_index(self.sigmas))
        if not assets:
            assets = list(_infer_asset_index(self.mpo_volume_data))
        return assets

    def _get_sigma_series(self, t, tau):
        return _get_series_from_forecast(self.sigmas, t, tau)

    def _get_volume_series(self, t, tau):
        return _get_series_from_forecast(
            self.mpo_volume_data, t, tau, volume_cap=self.volume_cap
        )

    def _impact_coeffs(self, t, tau):
        sigma_series = self._get_sigma_series(t, tau)
        volume_series = self._get_volume_series(t, tau)
        Y = self.config.get('Y', 1.0)
        coeffs = []
        for asset in self._assets():
            if asset == self.cash_key:
                coeffs.append(0.0)
                continue
            sigma = 0.0
            volume = 0.0
            try:
                if sigma_series is not None:
                    sigma = float(sigma_series.get(asset, 0.0))
            except Exception:
                sigma = 0.0
            try:
                if volume_series is not None:
                    volume = float(volume_series.get(asset, 0.0))
            except Exception:
                volume = 0.0
            coeffs.append(Y * sigma / np.sqrt(volume) if sigma > 0 and volume > 0 else 0.0)
        return np.array(coeffs, dtype=float)

    def _estimate_ahead(self, t, tau, w_plus, z, value):
        coeffs = self._impact_coeffs(t, tau)
        c_imp = cvx.Parameter(len(coeffs), value=coeffs, nonneg=True)
        return c_imp.T @ cvx.power(cvx.abs(z), 1.5), []

    def _estimate(self, t, w_plus, z, value):
        return self._estimate_ahead(t, t, w_plus, z, value)

    def value_expr(self, t, h_plus, u):
        u_val = u.values if hasattr(u, 'values') else u
        coeffs = self._impact_coeffs(t, t)
        portfolio_value = float(np.sum(np.abs(h_plus.values if hasattr(h_plus, 'values') else h_plus)))
        if portfolio_value <= 0:
            self.cost_logs[t] = 0.0
            return 0.0

        cost = 0.0
        for i, asset in enumerate(self._assets()):
            trade_val = abs(u[asset] if hasattr(u, 'index') else u_val[i])
            if trade_val < 1e-6 or asset == self.cash_key:
                continue
            trade_weight = trade_val / portfolio_value
            # Keep simulation scaling consistent with the optimizer:
            # optimizer uses coeff * |z|^1.5, so realized dollar cost should be
            # coeff * value * |u/value|^1.5.
            cost += coeffs[i] * portfolio_value * np.power(trade_weight, 1.5)
        self.cost_logs[t] = cost
        return cost


class CNFuturesSlippage(CNFuturesTickSlippage):
    """Backward-compatible wrapper. Supports only 'tick' and 'vol'."""

    def __new__(cls, prices, target_info_path=None, volumes=None, sigmas=None, config=None, cash_key='CNY'):
        config = config if config else {'model': 'tick', 'k': 1.0}
        model = config.get('model', 'tick')
        if model == 'tick':
            return CNFuturesTickSlippage(
                prices=prices,
                target_info_path=target_info_path,
                k=config.get('k', 1.0),
                cash_key=cash_key,
            )
        if model == 'vol':
            return CNFuturesVolSlippage(
                sigmas=sigmas,
                k=config.get('k', 1.0),
                cash_key=cash_key,
            )
        raise ValueError("CNFuturesSlippage no longer supports model='impact'. Use CNFuturesMPOVolumeSlippage.")


class CNFuturesMarginCost(BaseCost):
    """Margin financing cost."""

    def __init__(self, margin_csv_path, risk_free_rate, cash_key='CNY', asset_list=None):
        super(CNFuturesMarginCost, self).__init__()
        self.rf = risk_free_rate
        self.cash_key = cash_key
        self.asset_list = asset_list
        self.initial_margins = self._parse_margin(margin_csv_path)

    def _parse_margin(self, path):
        df = pd.read_csv(path)
        tickers = df.iloc[:, 0].astype(str).str.strip()
        margins = df.iloc[:, 1].astype(float).map(_normalize_margin_rate)
        return dict(zip(tickers, margins))

    def _estimate(self, t, w_plus, z, value):
        assets = self.asset_list if self.asset_list is not None else self.initial_margins.keys()
        m_vec_list = [self.initial_margins.get(a, 0.1) for a in assets]
        m_vec = cvx.Parameter(len(m_vec_list), value=np.array(m_vec_list), nonneg=True)
        return (m_vec.T @ cvx.abs(w_plus[:-1])) * self.rf, []

    def value_expr(self, t, h_plus, u):
        h_val = h_plus.values if hasattr(h_plus, 'values') else h_plus
        assets = h_plus.index if hasattr(h_plus, 'index') else (
            self.asset_list if self.asset_list is not None else self.initial_margins.keys()
        )
        cost = 0.0
        for i, asset in enumerate(assets):
            if asset == self.cash_key:
                continue
            rate = self.initial_margins.get(asset, 0.1)
            holding_val = h_plus[asset] if hasattr(h_plus, 'index') else h_val[i]
            cost += abs(holding_val) * rate
        cost *= self.rf
        self.cost_logs[t] = cost
        return cost
