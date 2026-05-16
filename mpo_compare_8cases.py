from __future__ import annotations

from typing import Dict, Iterable, Optional

import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvxportfolio as cp
import cvxportfolio.result as cp_result
from cvxportfolio.utils import values_in_time

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


class SafeMPOReturnsForecast(cp.returns.BaseReturnsModel):
    def __init__(self, alpha_data, asset_columns, cash_key='CNY', gamma_decay=None):
        self.alpha_data = alpha_data
        self.asset_columns = pd.Index(asset_columns)
        self.cash_key = cash_key
        self.gamma_decay = gamma_decay

    def _lookup_series(self, t, tau):
        if (t, tau) in self.alpha_data:
            return self.alpha_data[(t, tau)]

        same_target = [
            (t0, tau0) for (t0, tau0) in self.alpha_data.keys()
            if tau0 == tau and t0 <= t
        ]
        if same_target:
            best_key = max(same_target, key=lambda key: key[0])
            return self.alpha_data[best_key]

        same_origin = [
            (t0, tau0) for (t0, tau0) in self.alpha_data.keys()
            if t0 == t and tau0 >= t
        ]
        if same_origin:
            best_key = min(same_origin, key=lambda key: abs((key[1] - tau).days))
            return self.alpha_data[best_key]

        prior_origin = [
            (t0, tau0) for (t0, tau0) in self.alpha_data.keys()
            if t0 <= t and tau0 >= t
        ]
        if prior_origin:
            best_key = min(
                prior_origin,
                key=lambda key: (abs((key[0] - t).days), abs((key[1] - tau).days)),
            )
            return self.alpha_data[best_key]

        raise KeyError((t, tau))

    def _aligned_series(self, t, tau):
        alpha_series = self._lookup_series(t, tau)
        if not isinstance(alpha_series, pd.Series):
            alpha_series = pd.Series(alpha_series)
        alpha_series = alpha_series.reindex(self.asset_columns).fillna(0.0)
        if self.cash_key in alpha_series.index:
            alpha_series.loc[self.cash_key] = 0.0
        return alpha_series

    def weight_expr_ahead(self, t, tau, wplus=None, **kwargs):
        if wplus is None and 'w_plus' in kwargs:
            wplus = kwargs.get('w_plus')

        alpha_series = self._aligned_series(t, tau)
        alpha = cvx.sum(cvx.multiply(alpha_series.values, wplus))

        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days ** (-self.gamma_decay)

        return alpha


def _looks_like_single_step_dict(alpha_data) -> bool:
    if not isinstance(alpha_data, dict) or not alpha_data:
        return False
    deltas = {(tau - t).days for t, tau in alpha_data.keys()}
    return deltas == {1}


def _build_return_models(
    mpo_alpha_data,
    return_predictions: pd.DataFrame,
    cash_key: str,
    mpo_gamma_decay: float,
    spo_gamma_decay: float,
    alpha_delta: float,
):
    asset_columns = return_predictions.columns

    mpo_alpha = SafeMPOReturnsForecast(
        alpha_data=mpo_alpha_data,
        asset_columns=asset_columns,
        cash_key=cash_key,
        gamma_decay=mpo_gamma_decay,
    )

    single_period_alpha = cp.returns.ReturnsForecast(
        returns=return_predictions.copy(),
        delta=alpha_delta,
        gamma_decay=spo_gamma_decay,
    )

    if cash_key in single_period_alpha.returns.columns:
        single_period_alpha.returns[cash_key] = 0.0

    return {
        "MPOAlpha": mpo_alpha,
        "T1Alpha": single_period_alpha,
    }


def _build_risk_models(
    clean_historical_returns: pd.DataFrame,
    cash_key: str,
    lookback: int,
):
    risk_returns_panel = clean_historical_returns.drop(
        columns=[cash_key], errors="ignore"
    ).copy()

    return {
        "RiskNoDecay": cp.risks.ShrinkageSigma(
            returns=risk_returns_panel,
            lookback=lookback,
            gamma_half_life=float("inf"),
        ),
        "RiskDecay": cp.risks.ShrinkageSigma(
            returns=risk_returns_panel,
            lookback=lookback,
            gamma_half_life=3.0,
        ),
    }


def _build_cost_sets(
    base_costs: Iterable,
    mpo_volume_data,
    historical_sigmas,
    cash_key: str,
):
    full_path_costs = list(base_costs) + [
        cp.costs.CNFuturesMPOVolumeSlippage(
            mpo_volume_data=mpo_volume_data,
            sigmas=historical_sigmas,
            volume_cap=None,
            cash_key=cash_key,
        )
    ]

    capped_costs = list(base_costs) + [
        cp.costs.CNFuturesMPOVolumeSlippage(
            mpo_volume_data=mpo_volume_data,
            sigmas=historical_sigmas,
            volume_cap=5,
            cash_key=cash_key,
        )
    ]

    return {
        "CostFullPath": full_path_costs,
        "CostCapped": capped_costs,
    }


def _weight_policy_costs(cost_list: Iterable, gamma_tcost: float):
    weighted = []
    for cost in cost_list:
        if isinstance(cost, cp.costs.CNFuturesMarginCost):
            weighted.append(cost)
        else:
            weighted.append(gamma_tcost * cost)
    return weighted


def _estimate_alpha_dollars(return_model, t, holdings_after_trade: pd.Series) -> float:
    portfolio_value = float(holdings_after_trade.sum())
    if portfolio_value <= 0:
        return 0.0

    weights = holdings_after_trade / portfolio_value

    if isinstance(return_model, SafeMPOReturnsForecast):
        alpha_series = return_model._aligned_series(t, t)
        return float(np.dot(alpha_series.values, weights.values) * portfolio_value)

    if hasattr(return_model, "returns"):
        alpha_series = values_in_time(return_model.returns, t)
        if not isinstance(alpha_series, pd.Series):
            alpha_series = pd.Series(alpha_series, index=weights.index)
        alpha_series = alpha_series.reindex(weights.index).fillna(0.0)

        delta = getattr(return_model, "delta", 0.0)
        if isinstance(delta, (pd.Series, pd.DataFrame)):
            delta_series = values_in_time(delta, t)
            if not isinstance(delta_series, pd.Series):
                delta_series = pd.Series(delta_series, index=weights.index)
            delta_series = delta_series.reindex(weights.index).fillna(0.0)
        else:
            delta_series = pd.Series(float(delta), index=weights.index)

        alpha = alpha_series.values * weights.values
        alpha -= delta_series.values * np.abs(weights.values)
        return float(alpha.sum() * portfolio_value)

    return 0.0


def _estimate_cost_dollars(cost_list: Iterable, t, holdings_after_trade: pd.Series, trades: pd.Series) -> float:
    total_cost = 0.0
    for cost in cost_list:
        total_cost += float(cost.value_expr(t, holdings_after_trade, trades))
    return total_cost


class TwoStepCostGatePolicy:
    def __init__(
        self,
        base_policy,
        gate_return_model,
        gate_costs: Iterable,
        gate_ratio: float = 1.0,
        gate_tolerance: float = 0.0,
    ):
        self.base_policy = base_policy
        self.gate_return_model = gate_return_model
        self.gate_costs = list(gate_costs)
        self.gate_ratio = gate_ratio
        self.gate_tolerance = gate_tolerance
        self.costs = getattr(base_policy, "costs", [])
        self.constraints = getattr(base_policy, "constraints", [])
        self.gate_logs = {}

    def get_trades(self, portfolio, t):
        candidate_trades = self.base_policy.get_trades(portfolio, t)
        candidate_trades = candidate_trades.reindex(portfolio.index).fillna(0.0)

        if float(candidate_trades.abs().sum()) <= 1e-12:
            self.gate_logs[t] = {
                "alpha": 0.0,
                "cost": 0.0,
                "passed": False,
                "reason": "no_candidate_trade",
            }
            return candidate_trades

        holdings_after_trade = portfolio.add(candidate_trades, fill_value=0.0).reindex(portfolio.index).fillna(0.0)
        alpha_est = _estimate_alpha_dollars(self.gate_return_model, t, holdings_after_trade)
        cost_est = _estimate_cost_dollars(self.gate_costs, t, holdings_after_trade, candidate_trades)

        passed = alpha_est > (self.gate_ratio * cost_est + self.gate_tolerance)
        self.gate_logs[t] = {
            "alpha": alpha_est,
            "cost": cost_est,
            "passed": passed,
            "reason": "alpha_gt_cost" if passed else "alpha_le_cost",
        }

        if passed:
            return candidate_trades
        return pd.Series(0.0, index=portfolio.index)


def run_mpo_8way_compare(
    mpo_alpha_data,
    return_predictions: pd.DataFrame,
    clean_historical_returns: pd.DataFrame,
    mpo_volume_data,
    base_costs: Iterable,
    margin_map: Dict,
    initial_portfolio: pd.Series,
    start_time,
    end_time,
    cash_key: str = "CNY",
    lookahead_periods: int = 2,
    solver: str = "CLARABEL",
    solver_opts: Optional[Dict] = None,
    gamma_risk: float = 3.0,
    gamma_tcost: float = 1.0,
    margin_limit: float = 0.8,
    trade_limit: float = 0.10,
    leverage_limit: Optional[float] = None,
    risk_lookback: int = 252,
    mpo_gamma_decay: float = 0.2,
    spo_gamma_decay: float = 2.0,
    alpha_delta: float = 1e-6,
    top_n: int = 15,
    plot: bool = True,
):
    if solver_opts is None:
        solver_opts = {"tol_gap_abs": 1e-7, "tol_gap_rel": 1e-7}

    if _looks_like_single_step_dict(mpo_alpha_data):
        print(
            "Warning: mpo_alpha_data currently looks like a single-step {(t, t+1): alpha} dict, "
            "not a true multi-horizon forecast cube. The MPOAlpha branch is therefore only a pseudo-MPO return model."
        )

    risky_assets = [c for c in clean_historical_returns.columns if c != cash_key]
    historical_sigmas = clean_historical_returns.std().copy()
    historical_sigmas[cash_key] = 1.0

    return_models = _build_return_models(
        mpo_alpha_data=mpo_alpha_data,
        return_predictions=return_predictions,
        cash_key=cash_key,
        mpo_gamma_decay=mpo_gamma_decay,
        spo_gamma_decay=spo_gamma_decay,
        alpha_delta=alpha_delta,
    )
    risk_models = _build_risk_models(
        clean_historical_returns=clean_historical_returns,
        cash_key=cash_key,
        lookback=risk_lookback,
    )
    cost_sets = _build_cost_sets(
        base_costs=base_costs,
        mpo_volume_data=mpo_volume_data,
        historical_sigmas=historical_sigmas,
        cash_key=cash_key,
    )

    common_constraints = [
        cp.MarginMaxLeverage(
            margin_map=margin_map,
            limit=margin_limit,
            asset_list=risky_assets,
            cash_key=cash_key,
        ),
        cp.TradeAbsoluteLimit(trade_limit),
    ]
    if leverage_limit is not None:
        common_constraints.insert(1, cp.LeverageLimit(leverage_limit))

    simulators = {
        cost_name: cp.CNFuturesSimulator(
            trading_times=clean_historical_returns,
            costs=list(cost_list),
            margin_map=margin_map,
            cash_key=cash_key,
        )
        for cost_name, cost_list in cost_sets.items()
    }

    policies = {}
    results = {}

    for return_name, return_model in return_models.items():
        for risk_name, risk_model in risk_models.items():
            for cost_name, cost_list in cost_sets.items():
                policy_name = f"{return_name}-{risk_name}-{cost_name}"
                weighted_costs = [gamma_risk * risk_model] + _weight_policy_costs(
                    cost_list=cost_list,
                    gamma_tcost=gamma_tcost,
                )

                policy = cp.MultiPeriodOpt(
                    return_forecast=return_model,
                    costs=weighted_costs,
                    constraints=common_constraints,
                    trading_times=list(clean_historical_returns.index),
                    lookahead_periods=lookahead_periods,
                    terminal_weights=None,
                    solver=solver,
                    solver_opts=solver_opts,
                )
                policies[policy_name] = policy

    print(
        f"Running {len(policies)} MPO backtests from "
        f"{pd.Timestamp(start_time).date()} to {pd.Timestamp(end_time).date()}..."
    )
    print(
        f"Parameter scale check: gamma_risk={gamma_risk}, gamma_tcost={gamma_tcost}. "
        "If trading starts too late, reduce gamma_tcost first."
    )
    for name, policy in policies.items():
        cost_name = name.split("-")[-1]
        print(f"  -> {name}")
        results[name] = simulators[cost_name].run_backtest(
            initial_portfolio=initial_portfolio.copy(),
            policy=policy,
            start_time=start_time,
            end_time=end_time,
        )

    table = cp_result.SimulationResult.comparison_table(results)
    display(table)

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(16, 20), sharex=True)
        cp_result.SimulationResult.plot_value_compare(
            results,
            ax=axes[0],
            title="MPO 8-Case Portfolio Value Comparison",
        )
        leverage_limits = None if leverage_limit is None else {
            "Leverage limit": leverage_limit
        }
        cp_result.SimulationResult.plot_leverage_compare(
            results,
            ax=axes[1],
            title="MPO 8-Case Leverage Comparison",
            leverage_limits=leverage_limits,
        )
        cp_result.SimulationResult.plot_drawdown_compare(
            results,
            ax=axes[2],
            title="MPO 8-Case Drawdown Comparison",
        )
        plt.tight_layout()
        plt.show()

        cp_result.SimulationResult.plot_top_holdings_compare(
            results,
            top_n=top_n,
        )

    return {
        "simulators": simulators,
        "policies": policies,
        "results": results,
        "table": table,
    }


def run_two_step_mpo_8way_compare(
    mpo_alpha_data,
    return_predictions: pd.DataFrame,
    clean_historical_returns: pd.DataFrame,
    mpo_volume_data,
    base_costs: Iterable,
    margin_map: Dict,
    initial_portfolio: pd.Series,
    start_time,
    end_time,
    cash_key: str = "CNY",
    lookahead_periods: int = 2,
    solver: str = "CLARABEL",
    solver_opts: Optional[Dict] = None,
    gamma_risk: float = 3.0,
    margin_limit: float = 0.8,
    trade_limit: float = 0.10,
    leverage_limit: Optional[float] = None,
    risk_lookback: int = 252,
    mpo_gamma_decay: float = 0.2,
    spo_gamma_decay: float = 2.0,
    alpha_delta: float = 1e-6,
    gate_ratio: float = 1.0,
    gate_tolerance: float = 0.0,
    top_n: int = 15,
    plot: bool = True,
):
    if solver_opts is None:
        solver_opts = {"tol_gap_abs": 1e-7, "tol_gap_rel": 1e-7}

    if _looks_like_single_step_dict(mpo_alpha_data):
        print(
            "Warning: mpo_alpha_data currently looks like a single-step {(t, t+1): alpha} dict, "
            "not a true multi-horizon forecast cube. The MPOAlpha branch is therefore only a pseudo-MPO return model."
        )

    risky_assets = [c for c in clean_historical_returns.columns if c != cash_key]
    historical_sigmas = clean_historical_returns.std().copy()
    historical_sigmas[cash_key] = 1.0

    return_models = _build_return_models(
        mpo_alpha_data=mpo_alpha_data,
        return_predictions=return_predictions,
        cash_key=cash_key,
        mpo_gamma_decay=mpo_gamma_decay,
        spo_gamma_decay=spo_gamma_decay,
        alpha_delta=alpha_delta,
    )
    risk_models = _build_risk_models(
        clean_historical_returns=clean_historical_returns,
        cash_key=cash_key,
        lookback=risk_lookback,
    )
    cost_sets = _build_cost_sets(
        base_costs=base_costs,
        mpo_volume_data=mpo_volume_data,
        historical_sigmas=historical_sigmas,
        cash_key=cash_key,
    )

    common_constraints = [
        cp.MarginMaxLeverage(
            margin_map=margin_map,
            limit=margin_limit,
            asset_list=risky_assets,
            cash_key=cash_key,
        ),
        cp.TradeAbsoluteLimit(trade_limit),
    ]
    if leverage_limit is not None:
        common_constraints.insert(1, cp.LeverageLimit(leverage_limit))

    simulators = {
        cost_name: cp.CNFuturesSimulator(
            trading_times=clean_historical_returns,
            costs=list(cost_list),
            margin_map=margin_map,
            cash_key=cash_key,
        )
        for cost_name, cost_list in cost_sets.items()
    }

    policies = {}
    planner_policies = {}
    results = {}

    for return_name, return_model in return_models.items():
        for risk_name, risk_model in risk_models.items():
            for cost_name, cost_list in cost_sets.items():
                policy_name = f"{return_name}-{risk_name}-{cost_name}"
                planner = cp.MultiPeriodOpt(
                    return_forecast=return_model,
                    costs=[gamma_risk * risk_model],
                    constraints=common_constraints,
                    trading_times=list(clean_historical_returns.index),
                    lookahead_periods=lookahead_periods,
                    terminal_weights=None,
                    solver=solver,
                    solver_opts=solver_opts,
                )
                planner_policies[policy_name] = planner
                policies[policy_name] = TwoStepCostGatePolicy(
                    base_policy=planner,
                    gate_return_model=return_model,
                    gate_costs=cost_list,
                    gate_ratio=gate_ratio,
                    gate_tolerance=gate_tolerance,
                )

    print(
        f"Running {len(policies)} two-step MPO backtests from "
        f"{pd.Timestamp(start_time).date()} to {pd.Timestamp(end_time).date()}..."
    )
    print(
        f"Step 1 uses return-risk only (gamma_risk={gamma_risk}); "
        f"step 2 trades only when alpha > {gate_ratio} * cost + {gate_tolerance}."
    )
    for name, policy in policies.items():
        cost_name = name.split("-")[-1]
        print(f"  -> {name}")
        results[name] = simulators[cost_name].run_backtest(
            initial_portfolio=initial_portfolio.copy(),
            policy=policy,
            start_time=start_time,
            end_time=end_time,
        )

    table = cp_result.SimulationResult.comparison_table(results)
    display(table)

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(16, 20), sharex=True)
        cp_result.SimulationResult.plot_value_compare(
            results,
            ax=axes[0],
            title="Two-Step MPO 8-Case Portfolio Value Comparison",
        )
        leverage_limits = None if leverage_limit is None else {
            "Leverage limit": leverage_limit
        }
        cp_result.SimulationResult.plot_leverage_compare(
            results,
            ax=axes[1],
            title="Two-Step MPO 8-Case Leverage Comparison",
            leverage_limits=leverage_limits,
        )
        cp_result.SimulationResult.plot_drawdown_compare(
            results,
            ax=axes[2],
            title="Two-Step MPO 8-Case Drawdown Comparison",
        )
        plt.tight_layout()
        plt.show()

        cp_result.SimulationResult.plot_top_holdings_compare(
            results,
            top_n=top_n,
        )

    return {
        "simulators": simulators,
        "planner_policies": planner_policies,
        "policies": policies,
        "results": results,
        "table": table,
    }


def run_mpo_4way_leverage_compare(
    mpo_alpha_data,
    return_predictions: pd.DataFrame,
    clean_historical_returns: pd.DataFrame,
    mpo_volume_data,
    base_costs: Iterable,
    margin_map: Dict,
    initial_portfolio: pd.Series,
    start_time,
    end_time,
    cash_key: str = "CNY",
    lookahead_periods: int = 7,
    solver: str = "CLARABEL",
    solver_opts: Optional[Dict] = None,
    gamma_risk: float = 3.0,
    gamma_tcost: float = 1.0,
    margin_limit: float = 0.8,
    trade_limit: float = 0.10,
    risk_lookback: int = 252,
    mpo_gamma_decay: float = 0.2,
    spo_gamma_decay: float = 2.0,
    alpha_delta: float = 1e-6,
    cost_name: str = "CostFullPath",
    leverage_levels: Optional[Iterable[Optional[float]]] = None,
    top_n: int = 15,
    plot: bool = True,
):
    if solver_opts is None:
        solver_opts = {"tol_gap_abs": 1e-7, "tol_gap_rel": 1e-7}
    if leverage_levels is None:
        leverage_levels = [None, 5.0, 10.0]

    risky_assets = [c for c in clean_historical_returns.columns if c != cash_key]
    historical_sigmas = clean_historical_returns.std().copy()
    historical_sigmas[cash_key] = 1.0

    return_models = _build_return_models(
        mpo_alpha_data=mpo_alpha_data,
        return_predictions=return_predictions,
        cash_key=cash_key,
        mpo_gamma_decay=mpo_gamma_decay,
        spo_gamma_decay=spo_gamma_decay,
        alpha_delta=alpha_delta,
    )
    risk_models = _build_risk_models(
        clean_historical_returns=clean_historical_returns,
        cash_key=cash_key,
        lookback=risk_lookback,
    )
    cost_sets = _build_cost_sets(
        base_costs=base_costs,
        mpo_volume_data=mpo_volume_data,
        historical_sigmas=historical_sigmas,
        cash_key=cash_key,
    )
    if cost_name not in cost_sets:
        raise KeyError(f"Unknown cost_name: {cost_name}")

    selected_costs = cost_sets[cost_name]
    weighted_costs = [gamma_tcost * cost for cost in selected_costs]

    results = {}
    policies = {}

    print(
        f"Running MPO 4-way leverage comparison from "
        f"{pd.Timestamp(start_time).date()} to {pd.Timestamp(end_time).date()} "
        f"with cost model {cost_name} and lookahead={lookahead_periods}..."
    )

    for return_name, return_model in return_models.items():
        for risk_name, risk_model in risk_models.items():
            for lev in leverage_levels:
                constraints = [
                    cp.MarginMaxLeverage(
                        margin_map=margin_map,
                        limit=margin_limit,
                        asset_list=risky_assets,
                        cash_key=cash_key,
                    ),
                    cp.TradeAbsoluteLimit(trade_limit),
                ]
                lev_label = "Base" if lev is None else f"Lev{int(lev)}"
                if lev is not None:
                    constraints.insert(1, cp.LeverageLimit(lev))

                policy_name = f"{return_name}-{risk_name}-{lev_label}"
                policy = cp.MultiPeriodOpt(
                    return_forecast=return_model,
                    costs=[gamma_risk * risk_model] + weighted_costs,
                    constraints=constraints,
                    trading_times=list(clean_historical_returns.index),
                    lookahead_periods=lookahead_periods,
                    terminal_weights=None,
                    solver=solver,
                    solver_opts=solver_opts,
                )
                simulator = cp.CNFuturesSimulator(
                    trading_times=clean_historical_returns,
                    costs=list(selected_costs),
                    margin_map=margin_map,
                    cash_key=cash_key,
                )
                print(f"  -> {policy_name}")
                result = simulator.run_backtest(
                    initial_portfolio=initial_portfolio.copy(),
                    policy=policy,
                    start_time=start_time,
                    end_time=end_time,
                )
                policies[policy_name] = policy
                results[policy_name] = result

    table = cp_result.SimulationResult.comparison_table(results)
    display(table)

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(16, 20), sharex=True)
        cp_result.SimulationResult.plot_value_compare(
            results,
            ax=axes[0],
            title=f"MPO 4-Way Value Compare ({cost_name})",
        )
        cp_result.SimulationResult.plot_leverage_compare(
            results,
            ax=axes[1],
            title=f"MPO 4-Way Leverage Compare ({cost_name})",
            leverage_limits={"Lev5 limit": 5.0, "Lev10 limit": 10.0},
        )
        cp_result.SimulationResult.plot_drawdown_compare(
            results,
            ax=axes[2],
            title=f"MPO 4-Way Drawdown Compare ({cost_name})",
        )
        plt.tight_layout()
        plt.show()

        cp_result.SimulationResult.plot_top_holdings_compare(
            results,
            top_n=top_n,
        )

    return {
        "policies": policies,
        "results": results,
        "table": table,
        "cost_name": cost_name,
        "lookahead_periods": lookahead_periods,
    }
