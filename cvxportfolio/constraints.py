from abc import ABCMeta, abstractmethod

import cvxpy as cvx
import numpy as np

from .utils import values_in_time

__all__ = [
    'LeverageLimit',
    'MarginMaxLeverage',
    'AssetAbsoluteLimit',
    'TradeAbsoluteLimit'
]


def _normalize_margin_rate(value):
    """Accept both percentage-style inputs (7, 16) and decimal inputs (0.07, 0.16)."""
    if isinstance(value, dict):
        value = value.get('initial', 0.1)
    rate = float(value)
    return rate / 100.0 if rate > 1.0 else rate


class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop('w_bench', 0.)

    def weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints."""
        if w_plus is None:
            return self._weight_expr(t, None, z, v)
        return self._weight_expr(t, w_plus - self.w_bench, z, v)

    @abstractmethod
    def _weight_expr(self, t, w_plus, z, v):
        pass


class LeverageLimit(BaseConstraint):
    """A limit on leverage."""

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return [cvx.norm(w_plus[:-1], 1) <= values_in_time(self.limit, t)]


class MarginMaxLeverage(BaseConstraint):
    """A leverage limit based on margin usage instead of gross notional."""

    def __init__(self, margin_map, limit=0.8, asset_list=None, cash_key='CNY', **kwargs):
        super().__init__(**kwargs)
        self.limit = limit
        self.asset_list = asset_list
        self.cash_key = cash_key

        if asset_list is not None:
            self.initial_margins = {
                asset: _normalize_margin_rate(margin_map.get(asset, 0.1))
                for asset in asset_list
                if asset != self.cash_key
            }
        else:
            self.initial_margins = {
                asset: _normalize_margin_rate(val)
                for asset, val in margin_map.items()
                if asset != self.cash_key
            }

    def _weight_expr(self, t, w_plus, z, value):
        if self.asset_list is not None:
            m_vals = np.array([
                self.initial_margins[a]
                for a in self.asset_list
                if a != self.cash_key
            ])
        else:
            m_vals = np.array(list(self.initial_margins.values()))

        m_vec = cvx.Parameter(len(m_vals), value=m_vals, nonneg=True)
        return [m_vec.T @ cvx.abs(w_plus[:-1]) <= self.limit]


class AssetAbsoluteLimit(BaseConstraint):
    """Per-asset absolute post-trade weight limit."""

    def __init__(self, limit_map, **kwargs):
        self.limit_map = limit_map
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        limits = values_in_time(self.limit_map, t)
        return [cvx.abs(w_plus[:-1]) <= limits]


class TradeAbsoluteLimit(BaseConstraint):
    """Per-asset absolute trade-weight limit."""

    def __init__(self, limit_map=None, **kwargs):
        self.limit_map = 0.1 if limit_map is None else limit_map
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        limits = values_in_time(self.limit_map, t)
        return [cvx.abs(z[:-1]) <= limits]


class MaxTrade(BaseConstraint):
    """A limit on maximum trading size."""

    def __init__(self, ADVs, max_fraction=0.05, **kwargs):
        self.ADVs = ADVs
        self.max_fraction = max_fraction
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return cvx.abs(z[:-1]) * v <= \
            np.array(values_in_time(self.ADVs, t)) * self.max_fraction


class FactorMaxLimit(BaseConstraint):
    """A max limit on portfolio-wide factor exposure."""

    def __init__(self, factor_exposure, limit, **kwargs):
        super().__init__(**kwargs)
        self.factor_exposure = factor_exposure
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        return values_in_time(self.factor_exposure, t).T * w_plus[:-1] <= \
            values_in_time(self.limit, t)


class FactorMinLimit(BaseConstraint):
    """A min limit on portfolio-wide factor exposure."""

    def __init__(self, factor_exposure, limit, **kwargs):
        super().__init__(**kwargs)
        self.factor_exposure = factor_exposure
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        return values_in_time(self.factor_exposure, t).T * w_plus[:-1] >= \
            values_in_time(self.limit, t)


class FixedAlpha(BaseConstraint):
    """A constraint to fix portfolio-wide alpha."""

    def __init__(self, return_forecast, alpha_target, **kwargs):
        super().__init__(**kwargs)
        self.return_forecast = return_forecast
        self.alpha_target = alpha_target

    def _weight_expr(self, t, w_plus, z, v):
        return values_in_time(self.return_forecast, t).T * w_plus[:-1] == \
            values_in_time(self.alpha_target, t)
