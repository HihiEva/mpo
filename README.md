# 多期组合优化（MPO）量化交易策略框架说明

本框架是一个基于全流程闭环的多期资产配置与算法交易系统。整个工程分为三个阶段：**数据预处理与特征工程**、**异构预测引擎构建（收益率与成交量路径）**、**多期凸优化（MPO）组合求解**。
```
原始日频数据 (./data/raw/)
│
▼ 第一阶段：特征工程与异构阵列预测 (./src/features_and_prediction/)
│
├─ factor.ipynb          → 基础特征池构建 (时序量价与基本面 Carry)
├─ ret_forest.ipynb      → 非线性树模型群预测
├─ ret_regression.ipynb  → Fama-Macbeth 正则化截面回归预测
├─ ret_combine.ipynb     → 收益率多步前瞻路径聚合矩阵 (1~7步 MPO Alpha Input)
└─ vol_predict.ipynb     → 5日滑动均值递推成交量预测矩阵 (MPO Volume Input)
│
▼ 第二阶段：多期组合规划最优化求解 (./src/optimization/)
│
└─ MPO.ipynb             ← 读取前瞻矩阵，调用 CVXPortfolio 底层最优化器
└─ cvxportfolio/   ← 凸优化核心算子（成本项/风控约束/二次锥规划求解）

```
---

## 一、 系统架构与执行流

### 1. 异构预测阵列层 (Prediction Engine)

本层通过提取多品种期货时间序列特征，并行驱动线性与非线性两个子系统，对未来 $H$ 步（7步）的前瞻路径进行精细估计。

* **线性系统 (`ret_regression.ipynb`)**：执行基于 Elastic Net 正则化的 Fama-Macbeth 截面回归。通过引入 L1/L2 惩罚项抑制共线性因子，并在时序轴上执行滚动平滑（Rolling Window），输出纯化后的预期收益矩阵。
* **非线性系统 (`ret_forest.ipynb`)**：通过高频信号分段降频（日内均值法与尾盘切片法），使用随机森林、LightGBM、XGBoost 及 CatBoost 挖掘微观非线性动量，损失函数结合 Huber Loss 增强对厚尾离群点的鲁棒性。
* **路径聚合 (`ret_combine.ipynb`)**：内层通过 120 日滚动 Rank IC 赋权，外层通过 5 日动态风险平价（Risk Parity）将线性和非线性系统的多步预测矩阵进行分层多模态聚合，输出量纲稳定的最终收益率预测路径。
* **流动性预测 (`vol_predict.ipynb`)**：通过稳健的 5 日滑动均值（MA-5）递推，预测各品种未来多步的市场成交量，作为多期优化中冲击成本和调仓边界的独立输入参数。

### 2. 多期最优化组合层 (Multi-Period Optimization)

* **策略执行入口 (`MPO.ipynb`)**：负责加载前序生成的收益率路径与成交量路径，配置效用函数惩罚系数，启动组合动态回测系统。
* **优化求解核心 (`cvxportfolio/`)**：基于 **CVXPY** 凸优化建模包，引入高精度锥规划求解器（如 `CLARABEL` 与 `OSQP`），在统一的最优化问题中联合调谐未来多步的收益、风险和交易摩擦。

---

## 二、 核心模块目录树
```
mpo_strategy_framework/
│
├── src/
│   ├── features_and_prediction/   # 因子挖掘与异构预测模型
│   │   ├── factor.ipynb           # 原始因子面板生成
│   │   ├── ret_forest.ipynb       # 机器学习非线性树模型阵列
│   │   ├── ret_regression.ipynb   # Fama-Macbeth 截面回归模型
│   │   └── ret_combine.ipynb      # 模型分层多模态聚合
│   │
│   ├── optimization/              # 多期组合优化器核心逻辑
│   │   ├── MPO.ipynb              # 策略回测主入口
│   │   ├── my_backtest.py         # 因子多空回测与分段测试引擎
│   │   ├── quant_eval.py          # 统计评估指标（Rank IC/ICIR/MAE/RMSE）
│   │   └── cluster_toolkit.py     # 横截面谱聚类与分层聚类分析工具
│   │
│   └── cvxportfolio/              # CVX凸最优化底层架构（开源魔改版）
│       ├── simulator.py           # 期货市场模拟器（含保证金强平逻辑）
│       ├── policies.py            # 优化策略基类（Single-Period / Multi-Period）
│       ├── returns.py             # 预期收益输入接口
│       ├── risks.py               # 风险矩阵构建（含协方差收缩及结构化因子模型）
│       ├── costs.py               # 交易摩擦模型（固定滑点/持仓资金利息/动态手续费）
│       └── constraints.py         # 物理与风控约束（预算中性/杠杆上限/调仓手数边界）
│
└── data/
├── raw/                       # 原始多品种期货量价面板
├── processed/                 # 因子清洗及去极值化（MAD/Z-score）数据
├── predictions/               # 收益率与成交量前瞻路径中间件 (.pkl)
└── target_config/             # 交易所保证金率与手续费率动态映射表
```

---

## 三、 标准回测与评测流

1. **单因子与多空分析**：通过 `my_backtest.py` 对特征池中的因子执行基于特定滞后期（Lags）的多空组合切分回测，计算其在不同 Split 分段下的方向稳定性。
2. **预测指标检验**：通过 `quant_eval.py` 计算截面相关性指标（Rank IC、ICIR、IC 胜率）来考察模型的**排序选资产能力**；通过 MAE 和 RMSE 来考察模型的**绝对数值及量纲拟合精度**。
3. **MPO 参数空间搜索**：在 `MPO.ipynb` 中，对风险厌恶系数 $\gamma_{risk}$ 和成本敏感因子 $\gamma_{tcost}$ 实施网格搜索（Grid Search），定位全局最优且结构稳定的风险收益折中区间（如 $\gamma_{risk} \in [8, 12]$），确保模型在样本外的强泛化表现。
