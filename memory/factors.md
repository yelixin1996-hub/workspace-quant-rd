# 量化因子知识库
> 更新: 2026-03-29

## 1. 因子分类

### 动量因子 (Momentum)
| 因子名 | 计算方法 | 说明 |
|--------|----------|------|
| `ret_N` | close.pct_change(N) | N日收益率 |
| `mom_20` | (close - close_20d_ago) / close_20d_ago | 20日动量 |
| `rsrs` | 阻力支撑相对强度 | 喜胖哥RSRS |
| `momentum_rev` | -ret_20 | 反转动量 |
| `high_freq_mom` | 微趋势跟踪 | 高频因子 |

### 价值因子 (Value)
| 因子名 | 计算方法 | 说明 |
|--------|----------|------|
| `pe_ratio` | price / earnings | 市盈率 |
| `pb_ratio` | price / book | 市净率 |
| `pcf_ratio` | price / cashflow | 现金流率 |
| `ps_ratio` | price / sales | 市销率 |

### 质量因子 (Quality)
| 因子名 | 计算方法 | 说明 |
|--------|----------|------|
| `roe` | net_income / equity | 净资产收益率 |
| `roa` | net_income / assets | 资产收益率 |
| `gross_margin` | gross_profit / revenue | 毛利率 |
| `operating_margin` | op_income / revenue | 营利率 |
| `debt_to_equity` | total_debt / equity | 资产负债率 |

### 量价因子 (Volume-Price)
| 因子名 | 计算方法 | 说明 |
|--------|----------|------|
| `vol_ratio` | volume / ma(volume,5) | 量比 |
| `turnover_rate` | volume / float_shares | 换手率 |
| `amount_ratio` | amount / ma(amount,5) | 成交额比 |
| `volatility_20` | std(ret_1,20) | 波动率 |
| `price_position` | (close - low_N) / (high_N - low_N) | 价格位置 |
| `amihud_illiq` | abs(ret) / volume | Amihud非流动性 |

### 趋势因子 (Trend)
| 因子名 | 计算方法 | 说明 |
|--------|----------|------|
| `ma_slope` | (ma_N - ma_N.shift(5)) / ma_N.shift(5) | 均线斜率 |
| `ma_bull` | ma5 > ma10 > ma20 | 均线多头排列 |
| `bbi_ratio` | close / BBI | BBI多空比 |
| `macd_hist` | DIF - DEA | MACD柱状图 |
| `kdj_j` | 3*K - 2*D | KDJ的J值 |
| `rsi_14` | 100 - 100/(1+RS) | RSI指标 |

### 市场情绪因子 (Sentiment)
| 因子名 | 计算方法 | 说明 |
|--------|----------|------|
| `shortInterest` | 融券余额变化 | 做空情绪 |
| `margin_balance` | 融资余额变化 | 杠杆情绪 |
| `fund_flow` | 资金净流入 | 大单资金流 |
| `board_strength` | 涨跌股票数比 | 市场广度 |

## 2. 因子预处理

### 去极值 (Winsorize)
```python
def winsorize(series, upper=0.99, lower=0.01):
    q_high = series.quantile(upper)
    q_low = series.quantile(lower)
    return series.clip(q_low, q_high)
```

### 标准化 (Standardize)
```python
def standardize(series):
    return (series - series.mean()) / series.std()
```

### 中性化 (Neutralize)
```python
# 行业中性
factor_by_industry = factor.groupby(industry).transform('mean')
factor_neutral = factor - factor_by_industry

# 市值中性
factor_vs_mktcap = pd.merge(factor, mktcap, on='code')
# 回归残差
```

## 3. 因子有效性评估

### IC (Information Coefficient)
```python
def calc_ic(factor, returns, n=20):
    """计算IC值 (Pearson相关系数)"""
    ic_series = []
    for i in range(n, len(factor)):
        f = factor.iloc[i-n:i].mean()
        r = returns.iloc[i]
        ic = f.corr(r)
        ic_series.append(ic)
    return np.mean(ic_series), np.std(ic_series)
```

### IC评判标准
| IC均值 | IC标准差 | 评价 |
|--------|----------|------|
| > 0.03 | < 0.06 | ⭐ 有效因子 |
| 0.02-0.03 | < 0.08 | 可用 |
| < 0.02 | > 0.08 | 无效 |

### IR (Information Ratio)
```python
IR = IC_mean / IC_std
```
- IR > 0.5: 稳定有效
- IR > 1.0: 强有效因子

## 4. 多因子组合

### 等权组合
```python
factor_combined = (factor1 + factor2 + factor3) / 3
```

### IC加权组合
```python
weights = [ic1, ic2, ic3] / sum([ic1, ic2, ic3])
factor_combined = w1 * factor1 + w2 * factor2 + w3 * factor3
```

### 最大化IC组合
```python
# 用优化器找最优权重
from scipy.optimize import minimize

def neg_ic(weights):
    combined = w1*f1 + w2*f2 + w3*f3
    return -calc_ic(combined, returns)

result = minimize(neg_ic, [1/3, 1/3, 1/3], bounds=[(0,1)]*3)
```

## 5. A股有效因子 (经验总结)

### 一线因子 (IC > 0.03)
- 20日动量 (mom_20)
- 5日收益率 (ret_5)
- 量比 (vol_ratio)
- 波动率倒数 (1/vol_20)

### 二线因子 (IC 0.02-0.03)
- BBI多空比 (bbi_ratio)
- 均线斜率 (ma_slope)
- RSI
- 价格位置 (price_pos)

### 待验证因子
- RSI背离
- 布林带位置
- KDJ金叉/死叉
- 缠论笔

## 6. 常见陷阱

1. **过拟合** - 样本内IC高，样本外失效
2. **未来函数** - 使用了未来数据
3. **偷价** - 假设按收盘价成交
4. **流动性** - 小市值股票无法实际执行
5. **手续费** - 忽视交易成本
6. **滑点** - 假设零滑点执行

## 7. 改进方向

### 机器学习因子
- XGBoost/LightGBM特征重要性
- 神经网络因子挖掘
- 自然语言处理因子

### 高频因子
- 逐笔数据因子
- 订单流因子
- 盘口信息因子

### 另类数据
- 卫星图像
- 社交媒体情绪
- 产业链数据