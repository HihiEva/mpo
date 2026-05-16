import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr  # 引入了 spearmanr

def normalize_factor(df, do_winsorize=True, multiplier=3.0, winsor_method='sigma', method='zscore'):
    """
    1. 因子去极值与标准化
    
    参数说明:
    - do_winsorize: 是否进行去极值处理，默认为 True。设为 False 则完全跳过去极值。
    - multiplier: 极值边界系数，默认 3.0 (例如 3倍MAD 或 3倍标准差)
    - winsor_method: 去极值方法，可选 'mad' (绝对中位差法) 或 'sigma' (标准差法)
    - method: 标准化方法，默认 'zscore'
    """
    df = df.copy()
    df = df.dropna(how='all')
    
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return df
        
    # ================= 去极值 =================
    if do_winsorize:
        if winsor_method == 'mad':
            median = numeric.median(axis=1)
            # 计算绝对中位差 (MAD)
            mad = numeric.sub(median, axis=0).abs().median(axis=1)
            lower = median - multiplier * mad
            upper = median + multiplier * mad
            
        elif winsor_method == 'sigma':
            mean = numeric.mean(axis=1)
            std = numeric.std(axis=1, ddof=0)
            lower = mean - multiplier * std
            upper = mean + multiplier * std
            
        else:
            raise ValueError("winsor_method 必须是 'mad' 或 'sigma'")

        # 填充边界值防止报错，并进行截断
        lower = lower.fillna(-np.inf)
        upper = upper.fillna(np.inf)
        clipped = numeric.clip(lower=lower, upper=upper, axis=0)
        df[numeric.columns] = clipped

    # ================= 标准化 =================
    numeric = df.select_dtypes(include=[np.number])
    if method == 'zscore':
        mu = numeric.mean(axis=1)
        sigma = numeric.std(axis=1, ddof=0).replace(0, np.nan)
        res = numeric.sub(mu, axis=0).div(sigma, axis=0)
        df[numeric.columns] = res
        
    return df

def backtest_long_short(factor_df, returns_df, lag=1, top_pct_extreme=0.1, top_pct_moderate=0.3, ic_method='rank'):
    """
    2. 多空组合回测与 IC 计算（支持分层加权）
    
    参数说明:
    - top_pct_extreme: 极端分组比例（默认 10%），这部分使用 2 倍权重
    - top_pct_moderate: 中等分组比例（默认 30%），10%~30% 这部分使用 1 倍权重
    - ic_method: IC计算方法，'pearson' 或 'rank'
    """
    factor_df = factor_df.copy()
    returns_df = returns_df.copy()
    
    # 统一列名
    factor_df.columns = [str(c).strip().upper() for c in factor_df.columns]
    returns_df.columns = [str(c).strip().upper() for c in returns_df.columns]
    factor_df = factor_df.loc[:, ~factor_df.columns.duplicated(keep='last')]
    returns_df = returns_df.loc[:, ~returns_df.columns.duplicated(keep='last')]
    
    # 统一时间索引
    factor_df.index = pd.to_datetime(factor_df.index.astype(str), errors='coerce').normalize()
    returns_df.index = pd.to_datetime(returns_df.index.astype(str), errors='coerce').normalize()
    if factor_df.index.tz is not None: factor_df.index = factor_df.index.tz_localize(None)
    if returns_df.index.tz is not None: returns_df.index = returns_df.index.tz_localize(None)

    # 去除无效索引并去重
    factor_df = factor_df.loc[factor_df.index.notna()]
    returns_df = returns_df.loc[returns_df.index.notna()]
    factor_df = factor_df[~factor_df.index.duplicated(keep='last')]
    returns_df = returns_df[~returns_df.index.duplicated(keep='last')]

    # 【关键】严格按时间升序排序
    factor_df = factor_df.sort_index()
    returns_df = returns_df.sort_index()

    common_cols = factor_df.columns.intersection(returns_df.columns)
    if len(common_cols) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float) 

    # 对齐
    factor = factor_df.reindex(index=returns_df.index, columns=common_cols)
    returns_next = returns_df[common_cols].shift(-lag)
    
    port_returns = []
    dates = []
    ic_list = []
    
    for date in factor.index:
        if date not in returns_next.index or pd.isna(date):
            port_returns.append(np.nan)
            ic_list.append(np.nan)
            dates.append(date)
            continue
            
        f_row = factor.loc[date].dropna()
        r_row = returns_next.loc[date].dropna()
        
        common = f_row.index.intersection(r_row.index)
        if len(common) == 0:
            port_returns.append(np.nan)
            ic_list.append(np.nan)
            dates.append(date)
            continue
            
        fvals = f_row[common]
        rval = r_row[common]
        
        n_total = len(fvals)
        n_extreme = max(1, int(n_total * top_pct_extreme))      # 极端组数量（前/后 10%）
        n_moderate = max(1, int(n_total * top_pct_moderate))    # 中等组数量（前/后 30%）
        
        sorted_idx = fvals.sort_values(ascending=False).index
        
        # 多头分组
        longs_extreme = sorted_idx[:n_extreme]                          # 前 10%，2 倍权重
        longs_moderate = sorted_idx[n_extreme:n_moderate]               # 10%~30%，1 倍权重
        
        # 空头分组
        shorts_extreme = sorted_idx[-n_extreme:]                        # 后 10%，2 倍权重
        shorts_moderate = sorted_idx[-n_moderate:-n_extreme]            # 后 10%~30%，1 倍权重
        
        # 加权计算多头收益（2 倍极端组 + 1 倍中等组）/ 3
        long_ret_extreme = rval[longs_extreme].mean() if len(longs_extreme) > 0 else 0
        long_ret_moderate = rval[longs_moderate].mean() if len(longs_moderate) > 0 else 0
        long_ret = (2 * long_ret_extreme + 1 * long_ret_moderate) / 3
        
        # 加权计算空头收益（2 倍极端组 + 1 倍中等组）/ 3
        short_ret_extreme = rval[shorts_extreme].mean() if len(shorts_extreme) > 0 else 0
        short_ret_moderate = rval[shorts_moderate].mean() if len(shorts_moderate) > 0 else 0
        short_ret = (2 * short_ret_extreme + 1 * short_ret_moderate) / 3
        
        port_returns.append(long_ret - short_ret)
        dates.append(date)
        
        # 计算 IC (Pearson / Rank)
        if len(common) >= 2:
            try:
                if ic_method.lower() == 'rank':
                    ic = spearmanr(fvals.values, rval.values)[0]
                else:
                    ic = pearsonr(fvals.values, rval.values)[0]
            except Exception:
                ic = np.nan
        else:
            ic = np.nan
        ic_list.append(ic)
            
    port_series = pd.Series(port_returns, index=pd.to_datetime(dates))
    ic_series = pd.Series(ic_list, index=pd.to_datetime(dates))
    
    return port_series, ic_series


def compare_factors(factors_dict, returns_df, lag=1, top_pct_extreme=0.1, top_pct_moderate=0.3, periods_per_year=252, ic_method='pearson'):
    """
    3. 多因子对比与作图
    """
    stats_list = []
    port_cum_dict = {}
    ic_cum_dict = {}
    dd_cum_dict = {}
    
    for name, factor_df in factors_dict.items():
        port_series, ic_series = backtest_long_short(
            factor_df, returns_df, lag=lag, 
            top_pct_extreme=top_pct_extreme, 
            top_pct_moderate=top_pct_moderate,
            ic_method=ic_method
        )
        s = port_series.dropna()
        
        if len(s) == 0:
            print(f"⚠️ 警告：因子 '{name}' 未生成有效回测数据！")
            continue
            
        cum_series = (1 + port_series.fillna(0)).cumprod()
        total_ret = cum_series.iloc[-1] if len(cum_series.dropna()) > 0 else np.nan
        
        total_for_ann = (1 + s).prod() - 1
        years = len(s) / periods_per_year
        ann_ret = (1 + total_for_ann) ** (1 / years) - 1 if years > 0 else 0
        
        mean_ann = s.mean() * periods_per_year
        std_ann = s.std(ddof=1) * np.sqrt(periods_per_year)
        sharpe = mean_ann / std_ann if std_ann != 0 else np.nan
        
        cum = cum_series.dropna()
        peak = cum.cummax()
        drawdowns = (peak - cum) / peak
        max_dd = drawdowns.max()
        
        drawdowns_plot = (cum - peak) / peak
        
        calmar = ann_ret / max_dd if (max_dd != 0 and not np.isnan(max_dd)) else np.nan
        win_rate = (s > 0).mean()
        
        stats_list.append({
            'factor': name,
            'daily_win_rate': win_rate,
            'cumulative_return': total_ret,
            'annualized_return': ann_ret,
            'sharpe': sharpe,
            'calmar': calmar,
            'max_drawdown': max_dd,
            'ic_mean': ic_series.mean(),
            'ic_median': ic_series.median()
        })
        port_cum_dict[name] = cum_series.dropna()
        ic_cum_dict[name] = ic_series.dropna().cumsum()
        dd_cum_dict[name] = drawdowns_plot
        
    if not stats_list:
        print("❌ 错误：所有因子的回测结果均为空。")
        return pd.DataFrame()
        
    stats_df = pd.DataFrame(stats_list).set_index('factor')
    
    # --- 作图 ---
    plt.figure(figsize=(12, 6))
    for name, cum_ret in port_cum_dict.items():
        plt.plot(cum_ret.index, cum_ret.values, label=name, linewidth=1.5)
    plt.title('Cumulative Returns (Daily Long-Short Strategy)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for name, cum_ic in ic_cum_dict.items():
        plt.plot(cum_ic.index, cum_ic.values, label=name, linewidth=1.5)
    
    ic_title = 'Rank IC' if ic_method.lower() == 'rank' else 'Pearson IC'
    plt.title(f'Cumulative {ic_title} (Daily)')
    plt.axhline(0, color='k', lw=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for name, dd in dd_cum_dict.items():
        line, = plt.plot(dd.index, dd.values, label=name, linewidth=1)
        plt.fill_between(dd.index, dd.values, 0, color=line.get_color(), alpha=0.3)
        
    plt.title('Drawdown Series (Daily)')
    plt.axhline(0, color='k', lw=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return stats_df


def compare_multi_lags(factors_dict, returns_df, lags=[1, 2], top_pct_extreme=0.1, top_pct_moderate=0.3, periods_per_year=252, ic_method='pearson', plot=True):
    """
    5. 多因子 + 多滞后期对比与作图 (已添加防穿仓保护与作图开关)
    """
    stats_list = []
    port_cum_dict = {}
    ic_cum_dict = {}
    dd_cum_dict = {}
    
    for lag in lags:
        for name, factor_df in factors_dict.items():
            label_name = f"{name}_lag{lag}"
            
            port_series, ic_series = backtest_long_short(
                factor_df, returns_df, lag=lag, 
                top_pct_extreme=top_pct_extreme, 
                top_pct_moderate=top_pct_moderate,
                ic_method=ic_method
            )
            s = port_series.dropna()
            
            if len(s) == 0:
                print(f"⚠️ 警告：'{label_name}' 未生成有效回测数据！")
                continue
                
            cum_series = (1 + port_series.fillna(0)).cumprod()
            total_ret = cum_series.iloc[-1] if len(cum_series.dropna()) > 0 else np.nan
            
            total_for_ann = (1 + s).prod() - 1
            years = len(s) / periods_per_year
            
            # --- 修复：强行将负净值托底为 0，防止数学运算报错 ---
            total_compounded = max(0.0, 1 + total_for_ann)
            ann_ret = total_compounded ** (1 / years) - 1 if years > 0 else 0
            # ----------------------------------------------------
            
            mean_ann = s.mean() * periods_per_year
            std_ann = s.std(ddof=1) * np.sqrt(periods_per_year)
            sharpe = mean_ann / std_ann if std_ann != 0 else np.nan
            
            cum = cum_series.dropna()
            peak = cum.cummax()
            drawdowns = (peak - cum) / peak
            max_dd = drawdowns.max()
            
            drawdowns_plot = (cum - peak) / peak
            
            calmar = ann_ret / max_dd if (max_dd != 0 and not np.isnan(max_dd)) else np.nan
            win_rate = (s > 0).mean()
            
            stats_list.append({
                'factor_and_lag': label_name,
                'daily_win_rate': win_rate,
                'cumulative_return': total_ret,
                'annualized_return': ann_ret,
                'sharpe': sharpe,
                'calmar': calmar,
                'max_drawdown': max_dd,
                'ic_mean': ic_series.mean(),
                'ic_median': ic_series.median()
            })
            port_cum_dict[label_name] = cum_series.dropna()
            ic_cum_dict[label_name] = ic_series.dropna().cumsum()
            dd_cum_dict[label_name] = drawdowns_plot
            
    if not stats_list:
        print("❌ 错误：所有因子的回测结果均为空。")
        return pd.DataFrame()
        
    stats_df = pd.DataFrame(stats_list).set_index('factor_and_lag')
    
    # --- 作图 ---
    if plot:
        plt.figure(figsize=(12, 6))
        for label, cum_ret in port_cum_dict.items():
            plt.plot(cum_ret.index, cum_ret.values, label=label, linewidth=1.5)
        plt.title('Cumulative Returns (Multi-Lags Comparison)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        for label, cum_ic in ic_cum_dict.items():
            plt.plot(cum_ic.index, cum_ic.values, label=label, linewidth=1.5)
        ic_title = 'Rank IC' if ic_method.lower() == 'rank' else 'Pearson IC'
        plt.title(f'Cumulative {ic_title} (Multi-Lags Comparison)')
        plt.axhline(0, color='k', lw=0.7)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        for label, dd in dd_cum_dict.items():
            line, = plt.plot(dd.index, dd.values, label=label, linewidth=1)
            plt.fill_between(dd.index, dd.values, 0, color=line.get_color(), alpha=0.3)
        plt.title('Drawdown Series (Multi-Lags Comparison)')
        plt.axhline(0, color='k', lw=0.7)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return stats_df
def compare_custom_lags(factors_dict, returns_df, lags_dict, top_pct_extreme=0.1, top_pct_moderate=0.3, periods_per_year=252, ic_method='pearson'):
    """
    6. 自定义每个因子的专属滞后期进行对比
    """
    stats_list = []
    port_cum_dict = {}
    ic_cum_dict = {}
    dd_cum_dict = {}
    
    for name, factor_df in factors_dict.items():
        if isinstance(lags_dict, dict):
            lag = lags_dict.get(name, 1)
        else:
            lag = lags_dict
            
        label_name = f"{name}_lag{lag}"
        
        port_series, ic_series = backtest_long_short(
            factor_df, returns_df, lag=lag, 
            top_pct_extreme=top_pct_extreme, 
            top_pct_moderate=top_pct_moderate,
            ic_method=ic_method
        )
        s = port_series.dropna()
        
        if len(s) == 0:
            print(f"⚠️ 警告：因子 '{label_name}' 未生成有效回测数据！")
            continue
            
        cum_series = (1 + port_series.fillna(0)).cumprod()
        total_ret = cum_series.iloc[-1] if len(cum_series.dropna()) > 0 else np.nan
        
        total_for_ann = (1 + s).prod() - 1
        years = len(s) / periods_per_year
        ann_ret = (1 + total_for_ann) ** (1 / years) - 1 if years > 0 else 0
        
        mean_ann = s.mean() * periods_per_year
        std_ann = s.std(ddof=1) * np.sqrt(periods_per_year)
        sharpe = mean_ann / std_ann if std_ann != 0 else np.nan
        
        cum = cum_series.dropna()
        peak = cum.cummax()
        drawdowns = (peak - cum) / peak
        max_dd = drawdowns.max()
        
        drawdowns_plot = (cum - peak) / peak
        
        calmar = ann_ret / max_dd if (max_dd != 0 and not np.isnan(max_dd)) else np.nan
        win_rate = (s > 0).mean()
        
        stats_list.append({
            'factor': label_name,
            'daily_win_rate': win_rate,
            'cumulative_return': total_ret,
            'annualized_return': ann_ret,
            'sharpe': sharpe,
            'calmar': calmar,
            'max_drawdown': max_dd,
            'ic_mean': ic_series.mean(),
            'ic_median': ic_series.median()
        })
        port_cum_dict[label_name] = cum_series.dropna()
        ic_cum_dict[label_name] = ic_series.dropna().cumsum()
        dd_cum_dict[label_name] = drawdowns_plot
        
    if not stats_list:
        print("❌ 错误：所有因子的回测结果均为空。")
        return pd.DataFrame()
        
    stats_df = pd.DataFrame(stats_list).set_index('factor')
    
    # --- 作图 ---
    plt.figure(figsize=(12, 6))
    for name, cum_ret in port_cum_dict.items():
        plt.plot(cum_ret.index, cum_ret.values, label=name, linewidth=1.5)
    plt.title('Cumulative Returns (Custom Lags per Factor)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for name, cum_ic in ic_cum_dict.items():
        plt.plot(cum_ic.index, cum_ic.values, label=name, linewidth=1.5)
    ic_title = 'Rank IC' if ic_method.lower() == 'rank' else 'Pearson IC'
    plt.title(f'Cumulative {ic_title} (Custom Lags per Factor)')
    plt.axhline(0, color='k', lw=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for name, dd in dd_cum_dict.items():
        line, = plt.plot(dd.index, dd.values, label=name, linewidth=1)
        plt.fill_between(dd.index, dd.values, 0, color=line.get_color(), alpha=0.3)
    plt.title('Drawdown Series (Custom Lags per Factor)')
    plt.axhline(0, color='k', lw=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return stats_df


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

def calculate_ic_series(factor_df, target_df, lag=1, ic_method='rank'):
    """只计算截面 IC 序列，剔除所有收益率逻辑"""
    factor_df = factor_df.copy()
    target_df = target_df.copy()
    
    # 统一时间索引
    factor_df.index = pd.to_datetime(factor_df.index.astype(str), errors='coerce').normalize()
    target_df.index = pd.to_datetime(target_df.index.astype(str), errors='coerce').normalize()
    
    factor_df = factor_df.sort_index()
    target_df = target_df.sort_index()
    
    # 提取公共列并错位对齐(预测未来 lag 天)
    common_cols = factor_df.columns.intersection(target_df.columns)
    factor = factor_df[common_cols]
    target_next = target_df[common_cols].shift(-lag)
    
    ic_list = []
    dates = []
    
    for date in factor.index:
        if date not in target_next.index or pd.isna(date):
            continue
            
        f_row = factor.loc[date].dropna()
        t_row = target_next.loc[date].dropna()
        
        common = f_row.index.intersection(t_row.index)
        if len(common) < 2:
            continue
            
        fvals = f_row[common]
        tvals = t_row[common]
        
        try:
            if ic_method.lower() == 'rank':
                ic = spearmanr(fvals.values, tvals.values)[0]
            else:
                ic = pearsonr(fvals.values, tvals.values)[0]
        except:
            ic = np.nan
            
        ic_list.append(ic)
        dates.append(date)
        
    return pd.Series(ic_list, index=pd.to_datetime(dates)).dropna()

def compare_multi_lags_only_ic(factors_dict, target_df, lags=[1, 2, 3, 4, 5], ic_method='rank'):
    """多滞后期 IC 评估与画图"""
    stats_list = []
    ic_cum_dict = {}
    
    for lag in lags:
        for name, factor_df in factors_dict.items():
            label_name = f"{name}_lag{lag}"
            
            # 计算单期 IC 序列
            ic_series = calculate_ic_series(factor_df, target_df, lag=lag, ic_method=ic_method)
            
            if len(ic_series) == 0:
                print(f"⚠️ 警告：'{label_name}' 未生成有效 IC 数据！")
                continue
            
            # 计算核心评价指标
            ic_mean = ic_series.mean()
            ic_std = ic_series.std(ddof=1)
            ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
            ic_win_rate = (ic_series > 0).mean() # IC大于0的胜率
            
            stats_list.append({
                'Factor_Lag': label_name,
                'IC_Mean': ic_mean,
                'IC_IR': ic_ir,
                'IC_Win_Rate(>0)': ic_win_rate,
            })
            
            ic_cum_dict[label_name] = ic_series.cumsum()
            
    stats_df = pd.DataFrame(stats_list).set_index('Factor_Lag')
    
    # --- 只画有意义的图：累计 IC 图 ---
    plt.figure(figsize=(12, 6))
    for label, cum_ic in ic_cum_dict.items():
        plt.plot(cum_ic.index, cum_ic.values, label=label, linewidth=1.5)
        
    ic_title = 'Rank IC' if ic_method.lower() == 'rank' else 'Pearson IC'
    plt.title(f'Cumulative {ic_title} (Volume Prediction)')
    plt.axhline(0, color='k', lw=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return stats_df

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import numpy as np
import pandas as pd

def prepare_volume_target(volume_df, window=20):
    # 完全不碰时间，直接算！
    clean_vol = volume_df.replace(0, np.nan).ffill()
    log_vol = np.log1p(clean_vol)
    baseline_df = log_vol.rolling(window=window, min_periods=window//2).mean()
    target_df = log_vol - baseline_df.shift(1)
    return target_df, baseline_df

def restore_volume_prediction(pred_df, baseline_df):
    restored_df = pred_df.copy()
    for col in restored_df.columns:
        if col not in baseline_df.columns:
            continue
        for idx in restored_df.index:
            if idx not in baseline_df.index:
                continue
            pred_val = restored_df.at[idx, col]
            base_val = baseline_df.at[idx, col]
            if pd.isna(base_val):
                restored_df.at[idx, col] = np.nan
                continue
            if isinstance(pred_val, (float, int)) and pd.isna(pred_val):
                continue
            if isinstance(pred_val, (list, np.ndarray)):
                real_v_array = np.expm1(np.array(pred_val) + base_val)
                restored_df.at[idx, col] = real_v_array.tolist()
            else:
                restored_df.at[idx, col] = np.expm1(pred_val + base_val)
    return restored_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sensitivity_1d(results_df, param_name, metrics=['annualized_return', 'ic_mean', 'max_drawdown']):
    """
    绘制单参数的敏感性分析图 (横轴为参数，纵轴为指标)
    
    参数:
    results_df: 包含参数列和各个指标列的 DataFrame
    param_name: 参数的列名 (例如 'lag' 或 'threshold')
    metrics: 需要绘制的指标列表
    """
    # 按参数值排序，确保连线顺序正确
    df_sorted = results_df.sort_values(by=param_name)
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]
        
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(df_sorted[param_name], df_sorted[metric], marker='o', linestyle='-', linewidth=2, markersize=6)
        
        # 增加标题和标签
        ax.set_title(f'{metric} vs {param_name}', fontsize=12)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 针对最大回撤，可以把坐标轴翻转，或者用不同的颜色 (可选)
        if 'drawdown' in metric.lower():
            ax.invert_yaxis() # 回撤越小越在上面
            
    axes[-1].set_xlabel(param_name, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_sensitivity_2d(results_df, param_x, param_y, metrics=['annualized_return', 'ic_mean', 'max_drawdown']):
    """
    绘制双参数的热力图 (X轴为参数1，Y轴为参数2，颜色深浅为指标)
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
        
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 将数据透视为热力图需要的矩阵格式
        pivot_table = results_df.pivot(index=param_y, columns=param_x, values=metric)
        
        # 设置不同的颜色映射 (收益和IC用红/绿，回撤用红/蓝等)
        if 'drawdown' in metric.lower():
            cmap = 'YlOrRd'  # 回撤越大数据越红
        else:
            cmap = 'RdYlGn'  # 收益/IC越大越绿/红 (取决于你的习惯，这里绿代表好)
            
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap=cmap, ax=ax, 
                    cbar_kws={'label': metric}, linewidths=.5)
        
        ax.set_title(f'{metric} Heatmap\n({param_y} vs {param_x})', fontsize=12)
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        
    plt.tight_layout()
    plt.show()

# ==========================================
# 使用示例 (Dummy Data)
# ==========================================
if __name__ == "__main__":
    # 1. 单参数示例数据 (比如测试不同的 Lag)
    df_1d = pd.DataFrame({
        'lag': [1, 2, 3, 4, 5],
        'annualized_return': [0.15, 0.18, 0.16, 0.12, 0.08],
        'ic_mean': [0.06, 0.07, 0.05, 0.03, 0.02],
        'max_drawdown': [0.10, 0.08, 0.12, 0.15, 0.20]
    })
    
    print("绘制单参数敏感性折线图：")
    plot_sensitivity_1d(df_1d, param_name='lag')
    
    # 2. 双参数示例数据 (比如同时测试 移动平均窗口 和 截面极值百分比)
    import itertools
    window_list = [5, 10, 20]
    pct_list = [0.1, 0.2, 0.3]
    
    data_2d = []
    for w, p in itertools.product(window_list, pct_list):
        # 伪造一些回测指标
        ret = 0.2 - abs(w - 10)*0.01 - abs(p - 0.2)*0.1
        ic = 0.08 - abs(w - 10)*0.005 - abs(p - 0.2)*0.05
        dd = 0.05 + abs(w - 10)*0.01 + abs(p - 0.2)*0.2
        data_2d.append({'window': w, 'top_pct': p, 'annualized_return': ret, 'ic_mean': ic, 'max_drawdown': dd})
        
    df_2d = pd.DataFrame(data_2d)
    
    print("绘制双参数敏感性热力图：")
    plot_sensitivity_2d(df_2d, param_x='window', param_y='top_pct')
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sensitivity_1d_periods(results_df, param_name, metrics=['ic_mean', 'annualized_return']):
    """画出分时间段的 1D 折线矩阵图"""
    periods = results_df['period'].unique()
    n_periods = len(periods)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_periods, n_metrics, figsize=(5 * n_metrics, 3 * n_periods), sharex=True)
    
    if n_periods == 1: axes = np.expand_dims(axes, axis=0)
    if n_metrics == 1: axes = np.expand_dims(axes, axis=1)
        
    for i, period in enumerate(periods):
        df_period = results_df[results_df['period'] == period].sort_values(by=param_name)
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            ax.plot(df_period[param_name], df_period[metric], marker='o', linestyle='-', linewidth=2, markersize=6)
            
            ax.set_title(f'[{period}]\n{metric} vs {param_name}', fontsize=10)
            ax.set_ylabel(metric, fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if 'drawdown' in metric.lower():
                ax.invert_yaxis()
            if i == n_periods - 1:
                ax.set_xlabel(param_name, fontsize=10)
                
    plt.tight_layout()
    plt.show()


def compare_multi_lags_periods(factors_dict, returns_df, lags=[1, 2, 3, 4, 5], n_periods=6, 
                               ic_method='rank', metrics=['ic_mean', 'annualized_return', 'max_drawdown']):
    """
    终极封装：自动分段回测并输出 1D 矩阵图
    """
    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index.astype(str), errors='coerce').normalize()
    all_valid_dates = sorted(returns_df.index.dropna().unique())
    
    # 自动切分为 n 份
    date_chunks = np.array_split(all_valid_dates, n_periods)
    periods_info = []
    for chunk in date_chunks:
        if len(chunk) > 0:
            start_str = chunk[0].strftime('%Y-%m-%d')
            end_str = chunk[-1].strftime('%Y-%m-%d')
            periods_info.append({'dates': chunk, 'label': f"{start_str} ~ {end_str}"})
            
    results_list = []
    print(f"🚀 开始执行分时间段 ({n_periods}组) 回测...")
    
    for p_info in periods_info:
        chunk_dates = p_info['dates']
        period_label = p_info['label']
        
        # 截取该时间段的收益率
        returns_chunk = returns_df.loc[returns_df.index.isin(chunk_dates)]
        
        # 调用基础的多滞后期回测 (关闭普通画图)
        stats = compare_multi_lags(factors_dict, returns_chunk, lags=lags, ic_method=ic_method, plot=False)
        
        for idx, row in stats.iterrows():
            factor_name = idx.rsplit('_lag', 1)[0]
            lag_val = int(idx.rsplit('_lag', 1)[1])
            
            res_dict = {'period': period_label, 'factor': factor_name, 'lag': lag_val}
            for col in stats.columns:
                res_dict[col] = row[col]
            results_list.append(res_dict)
            
    df_results = pd.DataFrame(results_list)
    
    # 按照不同因子分别画分段矩阵图
    for f in df_results['factor'].unique():
        print(f"\n================== [{f}] 分时间段 1D 衰减图 ==================")
        df_plot = df_results[df_results['factor'] == f]
        plot_sensitivity_1d_periods(df_plot, param_name='lag', metrics=metrics)
        
    # 整理表格结构返回，方便你用 display() 查看
    final_stats_df = df_results.set_index(['factor', 'period', 'lag'])
    return final_stats_df

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sensitivity_2d_periods(results_df, param_x, param_y, metrics=['ic_mean', 'annualized_return', 'max_drawdown']):
    """
    绘制分时间段的 2D 参数敏感性热力图矩阵。
    
    参数:
    results_df: DataFrame, 包含回测结果的压平后的宽表 (必须包含 period, factor, 以及 param_x 和 param_y 列)
    param_x: str, 画图的横坐标参数名 (如 'lag')
    param_y: str, 画图的纵坐标参数名 (如 'k', 'T', 'window')
    metrics: list, 需要画的指标列名
    """
    # 提取共有多少个时间段
    if 'period' not in results_df.columns:
        raise ValueError("传入的 DataFrame 缺少 'period' 列，请确保是 reset_index() 后的结果！")
        
    periods = results_df['period'].unique()
    n_periods = len(periods)
    n_metrics = len(metrics)
    
    # 动态构建画布大小，保证图表清晰不拥挤
    fig, axes = plt.subplots(n_periods, n_metrics, figsize=(6 * n_metrics, 4.5 * n_periods))
    
    # 兼容只有 1 个时间段或 1 个指标的特殊维度情况
    if n_periods == 1 and n_metrics == 1:
        axes = np.array([[axes]])
    elif n_periods == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_metrics == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i, period in enumerate(periods):
        # 截取当前时间段的数据
        df_period = results_df[results_df['period'] == period].copy()
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # 使用 pivot_table 将长表转换为 2D 热力图需要的矩阵表
            # aggfunc='mean' 是为了防止有重复参数报错
            pivot_df = df_period.pivot_table(index=param_y, columns=param_x, values=metric, aggfunc='mean')
            
            # 智能排序：如果行列名称是数字字符串，强制转为 float/int 排序，防止 '10' 排在 '2' 前面
            try:
                pivot_df.index = pivot_df.index.astype(float)
                pivot_df = pivot_df.sort_index()
                # 去掉没必要的小数点
                if pivot_df.index.to_series().apply(lambda x: x.is_integer()).all():
                    pivot_df.index = pivot_df.index.astype(int)
            except:
                pivot_df = pivot_df.sort_index()
                
            try:
                pivot_df.columns = pivot_df.columns.astype(float)
                pivot_df = pivot_df.sort_index(axis=1)
                if pivot_df.columns.to_series().apply(lambda x: x.is_integer()).all():
                    pivot_df.columns = pivot_df.columns.astype(int)
            except:
                pivot_df = pivot_df.sort_index(axis=1)

            # 智能动态配色与中心点逻辑
            # IC 和 收益率：以 0 为界，绿色代表正收益(好)，红色代表负收益(坏)
            if metric in ['ic_mean', 'annualized_return', 'sharpe_ratio']:
                cmap = 'RdYlGn'
                center = 0
            # 回撤/波动：不需要以 0 为界，直接用从黄到红渐变，越红说明回撤越大(越危险)
            elif 'drawdown' in metric or 'volatility' in metric:
                cmap = 'YlOrRd'
                center = None
            else:
                cmap = 'viridis'
                center = None

            # 绘制 Seaborn 热力图
            if center is not None:
                sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap=cmap, center=center, 
                            ax=ax, linewidths=0.5, cbar=True)
            else:
                sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap=cmap, 
                            ax=ax, linewidths=0.5, cbar=True)
            
            # 设置标题与坐标轴
            ax.set_title(f"[{period}]\n{metric} ({param_y} vs {param_x})", fontsize=13)
            ax.set_xlabel(param_x, fontsize=11)
            ax.set_ylabel(param_y, fontsize=11)

    plt.tight_layout()
    plt.show()
def compare_multi_lags_periods(factors_dict, returns_df, lags=[1, 2, 3, 4, 5], n_periods=6, ic_method='rank'):
    """
    终极封装：自动分段回测 (纯净版，不自动画图，只返回数据表)
    """
    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index.astype(str), errors='coerce').normalize()
    all_valid_dates = sorted(returns_df.index.dropna().unique())
    
    # 自动切分为 n 份
    date_chunks = np.array_split(all_valid_dates, n_periods)
    periods_info = []
    for chunk in date_chunks:
        if len(chunk) > 0:
            start_str = chunk[0].strftime('%Y-%m-%d')
            end_str = chunk[-1].strftime('%Y-%m-%d')
            periods_info.append({'dates': chunk, 'label': f"{start_str} ~ {end_str}"})
            
    results_list = []
    print(f"🚀 开始执行分时间段 ({n_periods}组) 回测...")
    
    for p_info in periods_info:
        chunk_dates = p_info['dates']
        period_label = p_info['label']
        
        # 截取该时间段的收益率
        returns_chunk = returns_df.loc[returns_df.index.isin(chunk_dates)]
        
        # 调用基础的多滞后期回测 (关闭普通画图)
        stats = compare_multi_lags(factors_dict, returns_chunk, lags=lags, ic_method=ic_method, plot=False)
        
        for idx, row in stats.iterrows():
            factor_name = idx.rsplit('_lag', 1)[0]
            lag_val = int(idx.rsplit('_lag', 1)[1])
            
            res_dict = {'period': period_label, 'factor': factor_name, 'lag': lag_val}
            for col in stats.columns:
                res_dict[col] = row[col]
            results_list.append(res_dict)
            
    df_results = pd.DataFrame(results_list)
    
    # 整理表格结构返回
    final_stats_df = df_results.set_index(['factor', 'period', 'lag'])
    return final_stats_df