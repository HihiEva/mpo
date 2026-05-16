# file: fmb_predict_combined.py
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =========================================================================
# 内部核心辅助模块
# =========================================================================
def _safe_parse_dates(df, name=""):
    """
    内部辅助函数：终极安全的时间解析器。
    彻底规避 Pandas 将大数字错误解析的问题，自动剔除无效行。
    """
    df_copy = df.copy()
    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = df_copy.index.normalize()
        if df_copy.index.tz is not None:
            df_copy.index = df_copy.index.tz_localize(None)
        return df_copy.loc[df_copy.index.notna()]

    raw_values = df_copy.index.values
    parsed_dates = []
    for val in raw_values:
        if pd.isna(val):
            parsed_dates.append(pd.NaT)
            continue
        if isinstance(val, (int, float, np.number)):
            try:
                s = str(int(float(val)))
            except Exception:
                s = str(val)
        else:
            s = str(val).strip()
            
        s = s.replace('.0', '').split(' ')[0].replace('-', '')[:8]
        try:
            parsed_dates.append(pd.to_datetime(s, format='%Y%m%d'))
        except Exception:
            parsed_dates.append(pd.NaT)
            
    df_copy.index = pd.DatetimeIndex(parsed_dates)
    return df_copy.loc[df_copy.index.notna()]

# =========================================================================
# 1. 预测矩阵生成模块 (回归主干)
# =========================================================================
def fmb_enet_trajectory(factors_dict, returns_df, horizons=[1, 2, 3, 4, 5, 6, 7], 
                        window_type='rolling', lookback=252, beta_window=20, 
                        alpha=0.01, l1_ratio=0.5, min_listing_days=60, winsorize_limits=(0.01, 0.99)):
    """全品种池化基准模型 (Pooled Full Universe) 预测矩阵生成"""
    print(f"初始化数据对齐与去均值化 ({window_type} 模式, 严格满窗口={lookback})...")
    
    returns_aligned = _safe_parse_dates(returns_df)
    factors_aligned = {k: _safe_parse_dates(v) for k, v in factors_dict.items()}
    
    common_dates = returns_aligned.index
    for v in factors_aligned.values():
        common_dates = common_dates.intersection(v.index)
    common_dates = common_dates.sort_values().dropna()
    
    if len(common_dates) == 0:
        raise ValueError("数据对齐后为空！请检查收益率和因子的时间格式及起止范围。")
        
    returns_df = returns_aligned.loc[common_dates].copy()
    factor_names = list(factors_dict.keys())
    factors_aligned = {k: v.loc[common_dates].copy() for k, v in factors_aligned.items()}
    
    # 严格满窗口掩码
    all_valid = returns_df.notna()
    for k in factor_names:
        all_valid = all_valid & factors_aligned[k].notna()
        
    if window_type == 'rolling':
        strict_mask = all_valid.rolling(window=lookback).sum() == lookback
        min_periods_val = lookback
    else:
        strict_mask = (all_valid.cumsum() >= lookback) & all_valid
        min_periods_val = lookback

    # 均值计算
    if window_type == 'rolling':
        ret_mean = returns_df.shift(1).rolling(window=lookback, min_periods=min_periods_val).mean()
        fact_mean = {k: v.rolling(window=lookback, min_periods=min_periods_val).mean() for k, v in factors_aligned.items()}
    else:
        ret_mean = returns_df.shift(1).expanding(min_periods=min_periods_val).mean()
        fact_mean = {k: v.expanding(min_periods=min_periods_val).mean() for k, v in factors_aligned.items()}

    ret_resid = returns_df - ret_mean
    fact_resid = {k: factors_aligned[k] - fact_mean[k] for k in factor_names}
    
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, random_state=42)
    temp_preds = {}
    
    for h in horizons:
        lagged_fact_resid = {k: v.shift(h) for k, v in fact_resid.items()}
        beta_records = []
        
        for s in tqdm(common_dates, desc=f"截面回归 (h={h})", leave=False):
            y_s = ret_resid.loc[s]
            X_s_df = pd.DataFrame({k: lagged_fact_resid[k].loc[s] for k in factor_names})
            
            # 【核心修改1】彻底清理 Inf，并 dropna
            df_s = pd.concat([y_s.rename('y'), X_s_df], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            
            # 叠加严格窗口掩码
            valid_assets_today = strict_mask.loc[s]
            df_s = df_s[df_s.index.isin(valid_assets_today[valid_assets_today].index)]

            if len(df_s) < 10:  
                continue
                
            if winsorize_limits:
                l_q, u_q = winsorize_limits
                for col in factor_names + ['y']: 
                    df_s[col] = df_s[col].clip(lower=df_s[col].quantile(l_q), upper=df_s[col].quantile(u_q))
            
            # ================= 🚨 探针与异常拦截 🚨 =================
            X_vals = df_s[factor_names].values
            y_vals = df_s['y'].values
            
            try:
                enet.fit(X_vals, y_vals)
            except Exception as e:
                print(f"\n\n{'='*50}")
                print(f"💥 [崩溃现场抓取] 模块: fmb_enet_trajectory")
                print(f"💥 日期: {s}")
                print(f"💥 当天符合条件的品种数: {len(df_s)}")
                print(f"--> X_vals 包含 NaN 的数量: {np.isnan(X_vals).sum()}")
                print(f"--> X_vals 包含 Inf 的数量: {np.isinf(X_vals).sum()}")
                print(f"--> y_vals 包含 NaN 的数量: {np.isnan(y_vals).sum()}")
                print(f"--> y_vals 包含 Inf 的数量: {np.isinf(y_vals).sum()}")
                if len(X_vals) > 0 and not np.isnan(X_vals).all():
                    print(f"--> X_vals 最大绝对值: {np.nanmax(np.abs(X_vals))}")
                print(f"💥 [原始报错信息] {e}")
                print(f"{'='*50}\n")
                raise RuntimeError("数据异常导致模型拟合失败，请查看上方打印的 [崩溃现场抓取] 信息！")
            # ========================================================
            
            record = {'Date': s} 
            for idx, col in enumerate(factor_names):
                record[f'Beta_{col}'] = enet.coef_[idx]
            beta_records.append(record)
            
        # 【核心修改2】检查是否一天都没运行
        if len(beta_records) == 0:
            print(f"\n⚠️ 警告：在 horizon={h} 时，【没有任何一天】的数据满足最小样本量和满窗口条件！这会导致返回全空预测矩阵！")

        beta_df = pd.DataFrame(beta_records).set_index('Date') if beta_records else pd.DataFrame(columns=['Date']+factor_names).set_index('Date')
        smoothed_betas = beta_df.rolling(window=beta_window, min_periods=5).mean()
        
        pred_matrix = ret_mean.copy()
        for col in factor_names:
            if f'Beta_{col}' in smoothed_betas.columns:
                pred_matrix += fact_resid[col].mul(smoothed_betas[f'Beta_{col}'], axis=0)
        temp_preds[h] = pred_matrix

    arr3d = np.array([temp_preds[h].values for h in horizons]) 
    arr_t_n_h = np.transpose(arr3d, (1, 2, 0))
    
    data_for_df = [[np.nan if pd.isna(arr_t_n_h[i, j, :]).all() else arr_t_n_h[i, j, :].tolist() 
                    for j in range(arr_t_n_h.shape[1])] for i in range(arr_t_n_h.shape[0])]
        
    result_df = pd.DataFrame(data_for_df, index=returns_df.index, columns=returns_df.columns)
    result_df.attrs['horizons'] = list(horizons)
    return result_df


def fmb_clustered_enet_trajectory(factors_dict, returns_df, cluster_df, interval_size=20, 
                                  horizons=[1, 2, 3, 4, 5, 6, 7], window_type='rolling', 
                                  lookback=252, beta_window=20, alpha=0.005, l1_ratio=0.9,
                                  min_listing_days=60, winsorize_limits=(0.01, 0.99), min_cluster_size=5):
    """带动态聚类结构支持的 Fama-MacBeth 预测矩阵生成"""
    print(f"\n初始化聚类回归数据对齐 (聚类稳定区间={interval_size}, 严格满窗口={lookback})...")
    
    returns_aligned = _safe_parse_dates(returns_df)
    cluster_aligned = _safe_parse_dates(cluster_df)
    factors_aligned = {k: _safe_parse_dates(v) for k, v in factors_dict.items()}
        
    common_dates = returns_aligned.index.intersection(cluster_aligned.index)
    for v in factors_aligned.values():
        common_dates = common_dates.intersection(v.index)
    common_dates = common_dates.sort_values().dropna()
    
    returns_df = returns_aligned.loc[common_dates].copy()
    cluster_df = cluster_aligned.loc[common_dates].copy()
    factor_names = list(factors_dict.keys())
    factors_aligned = {k: v.loc[common_dates].copy() for k, v in factors_aligned.items()}
    
    interval_cluster_df = pd.DataFrame(index=common_dates, columns=cluster_df.columns, dtype=float)
    for i in range(0, len(common_dates), interval_size):
        chunk_dates = common_dates[i : i + interval_size]
        valid_cluster_rows = cluster_df.loc[chunk_dates].dropna(how='all')
        if len(valid_cluster_rows) > 0:
            for d in chunk_dates:
                interval_cluster_df.loc[d] = valid_cluster_rows.iloc[0].values

    # 严格满窗口掩码
    all_valid = returns_df.notna()
    for k in factor_names:
        all_valid = all_valid & factors_aligned[k].notna()
        
    if window_type == 'rolling':
        strict_mask = all_valid.rolling(window=lookback).sum() == lookback
        min_periods_val = lookback
    else:
        strict_mask = (all_valid.cumsum() >= lookback) & all_valid
        min_periods_val = lookback

    # 均值计算
    if window_type == 'rolling':
        ret_mean = returns_df.shift(1).rolling(window=lookback, min_periods=min_periods_val).mean()
        fact_mean = {k: v.rolling(window=lookback, min_periods=min_periods_val).mean() for k, v in factors_aligned.items()}
    else:
        ret_mean = returns_df.shift(1).expanding(min_periods=min_periods_val).mean()
        fact_mean = {k: v.expanding(min_periods=min_periods_val).mean() for k, v in factors_aligned.items()}

    ret_resid = returns_df - ret_mean
    fact_resid = {k: factors_aligned[k] - fact_mean[k] for k in factor_names}
    
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, random_state=42)
    temp_preds = {}
    
    for h in horizons:
        lagged_fact_resid = {k: v.shift(h) for k, v in fact_resid.items()}
        cluster_beta_records = {}
        
        for s in tqdm(common_dates, desc=f"截面回归 (h={h})", leave=False):
            y_s = ret_resid.loc[s]
            X_s_df = pd.DataFrame({k: lagged_fact_resid[k].loc[s] for k in factor_names})
            
            # 【核心修改1】彻底清理 Inf，并 dropna
            df_s = pd.concat([y_s.rename('y'), X_s_df], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            
            valid_assets_today = strict_mask.loc[s]
            df_s = df_s[df_s.index.isin(valid_assets_today[valid_assets_today].index)]

            clusters_today = interval_cluster_df.loc[s].dropna()
            for c in clusters_today.unique():
                assets_in_c = clusters_today[clusters_today == c].index
                df_c = df_s.loc[df_s.index.intersection(assets_in_c)].copy()
                
                if len(df_c) < min_cluster_size: 
                    continue
                    
                if winsorize_limits:
                    l_q, u_q = winsorize_limits
                    for col in factor_names + ['y']: 
                        df_c[col] = df_c[col].clip(lower=df_c[col].quantile(l_q), upper=df_c[col].quantile(u_q))
                
                # ================= 🚨 探针与异常拦截 🚨 =================
                X_vals = df_c[factor_names].values
                y_vals = df_c['y'].values
                
                try:
                    enet.fit(X_vals, y_vals)
                except Exception as e:
                    print(f"\n\n{'='*50}")
                    print(f"💥 [崩溃现场抓取] 模块: fmb_clustered_enet_trajectory")
                    print(f"💥 日期: {s} | 聚类簇: {c}")
                    print(f"💥 该簇符合条件的品种数: {len(df_c)}")
                    print(f"--> X_vals 包含 NaN 的数量: {np.isnan(X_vals).sum()}")
                    print(f"--> X_vals 包含 Inf 的数量: {np.isinf(X_vals).sum()}")
                    print(f"--> y_vals 包含 NaN 的数量: {np.isnan(y_vals).sum()}")
                    print(f"--> y_vals 包含 Inf 的数量: {np.isinf(y_vals).sum()}")
                    if len(X_vals) > 0 and not np.isnan(X_vals).all():
                        print(f"--> X_vals 最大绝对值: {np.nanmax(np.abs(X_vals))}")
                    print(f"💥 [原始报错信息] {e}")
                    print(f"{'='*50}\n")
                    raise RuntimeError("数据异常导致模型拟合失败，请查看上方打印的 [崩溃现场抓取] 信息！")
                # ========================================================
                
                if c not in cluster_beta_records: 
                    cluster_beta_records[c] = []
                record = {'Date': s}
                for idx, col in enumerate(factor_names):
                    record[f'Beta_{col}'] = enet.coef_[idx]
                cluster_beta_records[c].append(record)
        
        # 【核心修改2】检查是否一天都没运行
        if len(cluster_beta_records) == 0:
            print(f"\n⚠️ 警告：在 horizon={h} 时，【没有任何一个聚类簇】的数据满足最小样本量和满窗口条件！")

        smoothed_cluster_betas = {}
        for c, records in cluster_beta_records.items():
            b_df = pd.DataFrame(records).set_index('Date')
            smoothed_cluster_betas[c] = b_df.rolling(window=beta_window, min_periods=max(1, beta_window // 4)).mean()

        beta_asset_matrices = {col: pd.DataFrame(np.nan, index=common_dates, columns=returns_df.columns) 
                               for col in factor_names}
        
        for s in common_dates:
            clusters_today = interval_cluster_df.loc[s].dropna()
            for asset, c in clusters_today.items():
                if c in smoothed_cluster_betas and s in smoothed_cluster_betas[c].index:
                    b_s = smoothed_cluster_betas[c].loc[s]
                    if not b_s.isna().all():
                        for col in factor_names:
                            if f'Beta_{col}' in b_s:
                                beta_asset_matrices[col].loc[s, asset] = b_s[f'Beta_{col}']
                            
        pred_matrix = ret_mean.copy()
        for col in factor_names:
            pred_matrix += fact_resid[col].mul(beta_asset_matrices[col], fill_value=0)
        temp_preds[h] = pred_matrix

    arr3d = np.array([temp_preds[h].values for h in horizons]) 
    arr_t_n_h = np.transpose(arr3d, (1, 2, 0))
    
    data_for_df = [[np.nan if pd.isna(arr_t_n_h[i, j, :]).all() else arr_t_n_h[i, j, :].tolist() 
                    for j in range(arr_t_n_h.shape[1])] for i in range(arr_t_n_h.shape[0])]
        
    result_df = pd.DataFrame(data_for_df, index=returns_df.index, columns=returns_df.columns)
    result_df.attrs['horizons'] = list(horizons)
    return result_df

# =========================================================================
# 2. 系数提取与可视化模块
# =========================================================================
def plot_full_universe_factor_premiums(factors_dict, returns_df, horizon=1, window_type='rolling', 
                                       lookback=252, alpha=0.01, l1_ratio=0.5,
                                       min_listing_days=60, winsorize_limits=(0.01, 0.99)):
    """全品种回归：各个因子的平均系数（因子溢价）柱状图"""
    print(f"提取全品种平均因子系数 (Horizon={horizon}, 严格满窗口={lookback})...")
    returns_aligned = _safe_parse_dates(returns_df)
    factors_aligned = {k: _safe_parse_dates(v) for k, v in factors_dict.items()}
        
    common_dates = returns_aligned.index
    for v in factors_aligned.values(): common_dates = common_dates.intersection(v.index)
    common_dates = common_dates.sort_values().dropna()
    
    returns_df = returns_aligned.loc[common_dates].copy()
    factor_names = list(factors_dict.keys())
    factors_aligned = {k: v.loc[common_dates].copy() for k, v in factors_aligned.items()}
    
    # 严格满窗口掩码
    all_valid = returns_df.notna()
    for k in factor_names:
        all_valid = all_valid & factors_aligned[k].notna()
        
    if window_type == 'rolling':
        strict_mask = all_valid.rolling(window=lookback).sum() == lookback
        min_periods_val = lookback
    else:
        strict_mask = (all_valid.cumsum() >= lookback) & all_valid
        min_periods_val = lookback

    if window_type == 'rolling':
        ret_mean = returns_df.shift(1).rolling(window=lookback, min_periods=min_periods_val).mean()
        fact_mean = {k: v.rolling(window=lookback, min_periods=min_periods_val).mean() for k, v in factors_aligned.items()}
    else:
        ret_mean = returns_df.shift(1).expanding(min_periods=min_periods_val).mean()
        fact_mean = {k: v.expanding(min_periods=min_periods_val).mean() for k, v in factors_aligned.items()}

    ret_resid = returns_df - ret_mean
    fact_resid = {k: factors_aligned[k] - fact_mean[k] for k in factor_names}
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, random_state=42)
    
    lagged_fact_resid = {k: v.shift(horizon) for k, v in fact_resid.items()}
    beta_records = []
    
    for s in tqdm(common_dates, desc=f"截面系数提取", leave=False):
        y_s = ret_resid.loc[s]
        X_s_df = pd.DataFrame({k: lagged_fact_resid[k].loc[s] for k in factor_names})
        
        # 彻底清理 Inf
        df_s = pd.concat([y_s.rename('y'), X_s_df], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        
        valid_assets_today = strict_mask.loc[s]
        df_s = df_s[df_s.index.isin(valid_assets_today[valid_assets_today].index)]

        if len(df_s) < 10: continue
            
        if winsorize_limits:
            l_q, u_q = winsorize_limits
            for col in factor_names + ['y']: 
                df_s[col] = df_s[col].clip(lower=df_s[col].quantile(l_q), upper=df_s[col].quantile(u_q))
                
        enet.fit(df_s[factor_names].values, df_s['y'].values)
        beta_records.append({col: enet.coef_[idx] for idx, col in enumerate(factor_names)})

    if len(beta_records) == 0:
        print("⚠️ 警告：数据不足以计算系数！返回全空。")
        return pd.Series()

    mean_betas = pd.DataFrame(beta_records).mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    colors = ['#d62728' if val > 0 else '#1f77b4' for val in mean_betas.values]
    bars = plt.bar(mean_betas.index, mean_betas.values, color=colors, edgecolor='black', alpha=0.8)
    
    max_y = abs(mean_betas.values).max()
    offset_val = max_y * 0.05 if max_y != 0 else 0.0001
    
    for bar in bars:
        yval = bar.get_height()
        va = 'bottom' if yval > 0 else 'top'
        offset = offset_val if yval > 0 else -offset_val
        
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.6f}', 
                 ha='center', va=va, fontsize=9, rotation=45 if len(factor_names)>8 else 0)

    plt.title(f'Full Universe: Mean Factor Premiums (h={horizon})', fontsize=15, fontweight='bold')
    plt.ylabel('Mean Coefficient (Beta)')
    plt.axhline(0, color='black', linewidth=1.2)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return mean_betas


def plot_clustered_fmb_coefficients(factors_dict, returns_df, cluster_df, target_date=-1,
                                    interval_size=20, horizon=1, window_type='rolling', 
                                    lookback=252, alpha=0.01, l1_ratio=0.5, 
                                    min_listing_days=60, winsorize_limits=(0.01, 0.99), min_cluster_size=5):
    """截面聚类系数截影"""
    print(f"初始化聚类截面截影 (Horizon={horizon}, 区间={interval_size}, 严格满窗口={lookback})...")
    
    returns_aligned = _safe_parse_dates(returns_df)
    cluster_aligned = _safe_parse_dates(cluster_df)
    factors_aligned = {k: _safe_parse_dates(v) for k, v in factors_dict.items()}
        
    common_dates = returns_aligned.index.intersection(cluster_aligned.index)
    for v in factors_aligned.values(): common_dates = common_dates.intersection(v.index)
    common_dates = common_dates.sort_values().dropna()
    
    if isinstance(target_date, int):
        if len(common_dates) == 0:
            return pd.DataFrame()
        query_date = common_dates[target_date]
    else:
        query_date = pd.to_datetime(target_date)
        if query_date not in common_dates:
            raise ValueError(f"目标日期 {query_date} 不在有效重叠时间轴内！")
            
    print(f"🎯 正在分析目标截面日期: {query_date.strftime('%Y-%m-%d')} ...")
    
    returns_df = returns_aligned.loc[common_dates].copy()
    cluster_df = cluster_aligned.loc[common_dates].copy()
    factor_names = list(factors_dict.keys())
    factors_aligned = {k: v.loc[common_dates].copy() for k, v in factors_aligned.items()}
    
    interval_cluster_df = pd.DataFrame(index=common_dates, columns=cluster_df.columns, dtype=float)
    for i in range(0, len(common_dates), interval_size):
        chunk_dates = common_dates[i : i + interval_size]
        valid_cluster_rows = cluster_df.loc[chunk_dates].dropna(how='all')
        if len(valid_cluster_rows) > 0:
            for d in chunk_dates: interval_cluster_df.loc[d] = valid_cluster_rows.iloc[0].values

    # 严格满窗口掩码
    all_valid = returns_df.notna()
    for k in factor_names:
        all_valid = all_valid & factors_aligned[k].notna()
        
    if window_type == 'rolling':
        strict_mask = all_valid.rolling(window=lookback).sum() == lookback
        min_periods_val = lookback
    else:
        strict_mask = (all_valid.cumsum() >= lookback) & all_valid
        min_periods_val = lookback

    if window_type == 'rolling':
        ret_mean = returns_df.shift(1).rolling(window=lookback, min_periods=min_periods_val).mean()
        fact_mean = {k: v.rolling(window=lookback, min_periods=min_periods_val).mean() for k, v in factors_aligned.items()}
    else:
        ret_mean = returns_df.shift(1).expanding(min_periods=min_periods_val).mean()
        fact_mean = {k: v.expanding(min_periods=min_periods_val).mean() for k, v in factors_aligned.items()}

    ret_resid = returns_df - ret_mean
    fact_resid = {k: factors_aligned[k] - fact_mean[k] for k in factor_names}
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, random_state=42)
    lagged_fact_resid = {k: v.shift(horizon) for k, v in fact_resid.items()}
    
    y_s = ret_resid.loc[query_date]
    X_s_df = pd.DataFrame({k: lagged_fact_resid[k].loc[query_date] for k in factor_names})
    
    # 彻底清理 Inf
    df_s = pd.concat([y_s.rename('y'), X_s_df], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    
    valid_assets_today = strict_mask.loc[query_date]
    df_s = df_s[df_s.index.isin(valid_assets_today[valid_assets_today].index)]

    clusters_today = interval_cluster_df.loc[query_date].dropna()
    cluster_betas = {}
    
    for c in clusters_today.unique():
        assets_in_c = clusters_today[clusters_today == c].index
        df_c = df_s.loc[df_s.index.intersection(assets_in_c)].copy()
        
        if len(df_c) < min_cluster_size: continue
            
        if winsorize_limits:
            l_q, u_q = winsorize_limits
            for col in factor_names + ['y']: 
                df_c[col] = df_c[col].clip(lower=df_c[col].quantile(l_q), upper=df_c[col].quantile(u_q))
                
        enet.fit(df_c[factor_names].values, df_c['y'].values)
        cluster_betas[c] = {col: enet.coef_[idx] for idx, col in enumerate(factor_names)}

    if not cluster_betas:
        print(f"⚠️ {query_date.strftime('%Y-%m-%d')} 当天没有任何聚类簇满足最小样本量要求。")
        return pd.DataFrame()

    sorted_clusters = sorted(list(cluster_betas.keys()))
    for c in sorted_clusters:
        betas = pd.Series(cluster_betas[c]).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 3.5))
        colors = ['#d62728' if val > 0 else '#1f77b4' for val in betas.values]
        bars = plt.bar(betas.index, betas.values, color=colors, edgecolor='black', alpha=0.8)
        
        max_y = abs(betas.values).max()
        offset_val = max_y * 0.08 if max_y != 0 else 0.0001
        
        for bar in bars:
            yval = bar.get_height()
            va = 'bottom' if yval > 0 else 'top'
            offset = offset_val if yval > 0 else -offset_val
            plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.5f}', 
                     ha='center', va=va, fontsize=9, rotation=0)

        plt.title(f'Cluster {int(c)} Factor Premiums on {query_date.strftime("%Y-%m-%d")}', fontsize=14, fontweight='bold')
        plt.ylabel('Coefficient (Beta)')
        plt.axhline(0, color='black', linewidth=1.2)
        plt.xticks(rotation=0 if len(factor_names) <= 6 else 30)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
    return pd.DataFrame(cluster_betas)