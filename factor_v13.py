# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v13.0 — 缩量回踩 + 逻辑止损
==========================================================
历史结论 (v10-v12):
- 1-5天持有: 胜率100%，平均+19-29% (核心盈利区)
- 6-10天: 胜率骤降，时间止损效果差
- 止盈20-30%有效，时间止损无效
- v11无止损: 年化0.12%，胜率53%

v13 核心改进:
1. 止盈提高至30-35%，让利润奔跑
2. 新增「缩量回踩」入场信号：股价回踩MA5/MA10 + 成交量萎缩
3. 亏损超-8%无条件止损（硬性风控线）
4. 市场环境RSI过滤：大盘弱势时降低仓位或不买
5. 新因子「量价背离」：下跌缩量=卖压衰竭，上涨放量=买盘强劲
"""

import pandas as pd
import numpy as np
import os


def log(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════

def load_all_stocks(data_dir='E:/data', min_rows=200, max_stocks=100):
    stocks = {}
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:max_stocks]
    for f in files:
        code = f.replace('.SZ.csv', '').replace('.SH.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df = df.rename(columns={'trade_date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) >= min_rows:
            stocks[code] = df
    return stocks


# ═══════════════════════════════════════════════════════════
# 因子计算
# ═══════════════════════════════════════════════════════════

def add_factors(df):
    df = df.copy()

    # --- 收益率 ---
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)
    df['ret_60'] = df['close'].pct_change(60)

    # --- 均线 ---
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    df['ma_bull'] = ((df['ma5'] > df['ma10']) &
                     (df['ma10'] > df['ma20']) &
                     (df['close'] > df['ma20'])).astype(int)

    # --- BBI ---
    df['bbi'] = (df['ma5'] + df['ma10'] + df['ma20'] + df['ma60']) / 4
    df['bbi_ratio'] = df['close'] / df['bbi']

    # --- 波动率 & 量比 ---
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma10'] = df['volume'].rolling(10).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-8)

    # --- 价格位置 ---
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

    # --- 均线斜率 ---
    df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(10)) / (df['ma20'].shift(10) + 1e-8)

    # --- RSI ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- KDJ ---
    n = 9
    low_n = df['low'].rolling(n).min()
    high_n = df['high'].rolling(n).max()
    rsv = (df['close'] - low_n) / (high_n - low_n + 1e-8) * 100
    rsv = rsv.fillna(50)
    K = pd.Series(50.0, index=df.index)
    D = pd.Series(50.0, index=df.index)
    for i in range(1, len(df)):
        K.iloc[i] = 2/3 * K.iloc[i-1] + 1/3 * rsv.iloc[i]
        D.iloc[i] = 2/3 * D.iloc[i-1] + 1/3 * K.iloc[i]
    df['K'] = K
    df['D'] = D

    # --- MACD ---
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    # --- 创新高 ---
    df['new_high_20'] = (df['close'] >= df['high'].rolling(20).max()).astype(int)

    # ═══════════════ v13新增因子 ═══════════════

    # --- 缩量回踩信号 ---
    ma5_dist = abs(df['close'] - df['ma5']) / (df['ma5'] + 1e-8)
    ma10_dist = abs(df['close'] - df['ma10']) / (df['ma10'] + 1e-8)
    near_ma5 = ma5_dist < 0.02
    near_ma10 = ma10_dist < 0.02
    vol_shrink = df['vol_ratio'] < 0.8
    trend_up = df['ma20_slope'] > 0.005
    df['pullback_buy'] = ((near_ma5 | near_ma10) & vol_shrink & trend_up).astype(int)

    # --- 量价背离因子 ---
    # 下跌缩量: 价格下跌但成交量萎缩 → 卖压衰竭(正面信号)
    price_down = df['ret_5'] < -0.03
    vol_5_shrink = df['volume'] < df['vol_ma10'] * 0.7
    df['vol_price_diverge_bull'] = (price_down & vol_5_shrink).astype(int)

    # 上涨放量: 价格上涨且成交量放大 → 买盘强劲(正面信号)
    price_up = df['ret_5'] > 0.03
    vol_5_expand = df['volume'] > df['vol_ma10'] * 1.5
    df['vol_price_diverge_strong'] = (price_up & vol_5_expand).astype(int)

    # --- 回踩支撑强度 ---
    df['support_ma5'] = (df['low'] <= df['ma5'] * 1.005) & (df['close'] > df['ma5']).astype(int)
    df['support_ma10'] = (df['low'] <= df['ma10'] * 1.005) & (df['close'] > df['ma10']).astype(int)

    return df


def prepare_factors(stocks):
    log("  预计算因子...")
    for code, df in stocks.items():
        stocks[code] = add_factors(df)
    log(f"  完成 {len(stocks)} 只股票的因子计算")


# ═══════════════════════════════════════════════════════════
# 市场环境判断 (v13增强版: 加入RSI)
# ═══════════════════════════════════════════════════════════

def get_market_env(stocks, date):
    """综合判断市场环境: 均线强度 + RSI"""
    bull_count = 0
    total = 0
    rsi_sum = 0
    rsi_count = 0

    for code, df in stocks.items():
        hist = df[df['date'] <= date]
        if len(hist) < 25:
            continue
        total += 1
        row = hist.iloc[-1]
        if row['close'] > row['ma20']:
            bull_count += 1
        rsi_val = row.get('rsi_14', 50)
        if not pd.isna(rsi_val):
            rsi_sum += rsi_val
            rsi_count += 1

    if total == 0:
        return 'neutral', 0.5, 50

    bull_ratio = bull_count / total
    avg_rsi = rsi_sum / max(rsi_count, 1)

    # RSI < 40 → 市场弱势
    if avg_rsi < 40:
        return 'weak', bull_ratio, avg_rsi
    elif bull_ratio > 0.55:
        return 'bull', bull_ratio, avg_rsi
    elif bull_ratio < 0.45:
        return 'bear', bull_ratio, avg_rsi
    else:
        return 'neutral', bull_ratio, avg_rsi


# ═══════════════════════════════════════════════════════════
# 选股器 (v13: 加权缩量回踩和量价背离因子)
# ═══════════════════════════════════════════════════════════

class Selector:
    def __init__(self, stocks):
        self.stocks = stocks

    def select_top(self, date, market_env='neutral', min_score=80, top_n=3):
        candidates = []

        for code, df in self.stocks.items():
            mask = df['date'] <= date
            if mask.sum() < 65:
                continue

            row = df[mask].iloc[-1]

            if pd.isna(row.get('ret_20', np.nan)):
                continue
            if row['close'] <= 0 or row['volume'] <= 0:
                continue
            if row.get('vol_ratio', 1) < 0.3 or row.get('vol_ratio', 1) > 6:
                continue

            score = 0

            # ─── 核心因子 ───

            # 1. 强动量 (max 60)
            mom = row.get('ret_20', 0)
            if mom > 0.20:
                score += 60
            elif mom > 0.15:
                score += 45
            elif mom > 0.10:
                score += 30
            elif mom > 0.05:
                score += 20
            elif mom < 0:
                score += 15 * mom

            # 2. 均线多头 (25)
            if row.get('ma_bull', 0) == 1:
                score += 25

            # 3. BBI上方 (max ~15)
            if row.get('bbi_ratio', 1) > 1:
                score += 15 * min(row['bbi_ratio'] - 1, 0.1) / 0.1

            # 4. 趋势向上 (15)
            slope = row.get('ma20_slope', 0)
            if slope > 0.02:
                score += 15
            elif slope > 0:
                score += slope / 0.02 * 15
            elif slope < 0:
                score += 10 * slope

            # 5. 创20日新高 (15)
            if row.get('new_high_20', 0) == 1:
                score += 15

            # 6. RSI健康 (10)
            rsi = row.get('rsi_14', 50)
            if 45 < rsi < 65:
                score += 10
            elif rsi < 40:
                score += 5

            # 7. 量比适中 (8)
            vol_r = row.get('vol_ratio', 1)
            if 1.2 < vol_r < 3:
                score += 8

            # 8. 低波动 (10)
            vol = row.get('vol_20', 0.05)
            if vol < 0.02:
                score += 10
            elif vol < 0.03:
                score += 5

            # 9. KDJ金叉 (8)
            if row.get('K', 50) > row.get('D', 50) and row.get('K', 50) < 70:
                score += 8

            # 10. MACD多头 (5)
            if row.get('MACD', 0) > 0:
                score += 5

            # ─── v13 新因子 ───

            # 11. 缩量回踩信号 (重要! +20)
            if row.get('pullback_buy', 0) == 1:
                score += 20

            # 12. 量价背离 — 下跌缩量(卖压衰竭) (+12)
            if row.get('vol_price_diverge_bull', 0) == 1:
                score += 12

            # 13. 量价背离 — 上涨放量(买盘强劲) (+8)
            if row.get('vol_price_diverge_strong', 0) == 1:
                score += 8

            # 14. 回踩MA支撑 (+6)
            if row.get('support_ma5', 0) == 1:
                score += 6
            elif row.get('support_ma10', 0) == 1:
                score += 4

            # ─── 负面清单 ───
            if mom < -0.15:
                score -= 30
            if rsi > 80:
                score -= 20
            if vol_r > 4:
                score -= 10

            # ─── 环境适应 ───
            if market_env == 'bull':
                if row.get('ma_bull', 0) == 1:
                    score *= 1.3
            elif market_env == 'bear':
                continue
            elif market_env == 'weak':
                score *= 0.7

            if score >= min_score:
                candidates.append((code, score, row['close']))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ═══════════════════════════════════════════════════════════
# 回测引擎 v13
# ═══════════════════════════════════════════════════════════

def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-20',
                 initial_cash=1000000, take_profit=0.30, stop_loss=-0.08,
                 max_hold=25, min_score=80, trailing_profit=0.05):
    """
    v13 回测引擎
    - take_profit: 止盈阈值 (默认30%)
    - stop_loss: 硬止损线 (默认-8%)
    - max_hold: 最大持有天数 (到期不强制卖，改为评估)
    - trailing_profit: 浮盈超过take_profit*0.6后回撤幅度触发止盈
    """

    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates
                    if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])

    if not dates:
        log("  无可用交易日!")
        return [], [initial_cash], []

    log(f"  回测期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    log(f"  交易日: {len(dates)}, 股票池: {len(stocks)}")
    log(f"  参数: 止盈={take_profit:.0%}, 止损={stop_loss:.0%}, "
        f"追踪回撤={trailing_profit:.0%}, 最大持有={max_hold}天, 最低分={min_score}")

    selector = Selector(stocks)

    cash = initial_cash
    position = None
    trades = []
    equity_curve = [initial_cash]
    max_pnl_in_trade = 0  # 本次交易中的最高浮盈

    cached_env = ('neutral', 0.5, 50)
    env_update_day = -1

    for i, date in enumerate(dates):
        if i < 65:
            continue

        # 每5天更新市场环境
        if i - env_update_day >= 5:
            cached_env = get_market_env(stocks, date)
            env_update_day = i
        market_env, bull_ratio, avg_rsi = cached_env

        # ─── 持仓管理 ───
        if position:
            code, entry_price, entry_date, shares = position

            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if today_row.empty:
                continue
            price = today_row['close'].iloc[0]
            today_rsi = today_row['rsi_14'].iloc[0]
            today_high = today_row['high'].iloc[0]

            hold_days = (date - entry_date).days
            pnl = (price - entry_price) / entry_price
            pnl_high = (today_high - entry_price) / entry_price
            max_pnl_in_trade = max(max_pnl_in_trade, pnl_high)

            sell_reason = None

            # [1] 硬性止损: 亏损超-8%，无条件出场
            if pnl <= stop_loss:
                sell_reason = '止损'

            # [2] 止盈: 收益达到阈值
            elif pnl >= take_profit:
                sell_reason = '止盈'

            # [3] 追踪止盈: 浮盈曾超过 take_profit*0.6, 但从最高点回撤超过 trailing_profit
            elif max_pnl_in_trade >= take_profit * 0.6 and (max_pnl_in_trade - pnl) >= trailing_profit:
                sell_reason = '追踪止盈'

            # [4] RSI超买辅助 (有盈利才卖)
            elif today_rsi > 88 and pnl > 0.05:
                sell_reason = 'RSI超买'

            # [5] 到期评估: 超过max_hold天
            # 不直接强卖，而是评估当前状态
            elif hold_days >= max_hold:
                if pnl < -0.03:
                    sell_reason = '到期亏损'
                elif pnl > 0.15:
                    pass  # 大幅盈利，继续持有
                elif today_rsi > 70:
                    sell_reason = '到期超买'
                else:
                    sell_reason = '到期'

            if sell_reason:
                revenue = price * shares * 0.9997
                cash += revenue
                pnl_pct = pnl * 100
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl_pct,
                    'reason': sell_reason, 'hold_days': hold_days,
                    'max_pnl': max_pnl_in_trade * 100
                })
                position = None
                max_pnl_in_trade = 0

        # ─── 买入 ───
        if position is None:
            # 市场弱势(RSI<40)不开仓
            if market_env in ('bear', 'weak'):
                pass
            else:
                top_stocks = selector.select_top(
                    date, market_env=market_env, min_score=min_score, top_n=3
                )
                if top_stocks:
                    code, score, price = top_stocks[0]
                    if price * 100 * 1.0003 <= cash:
                        shares = 100
                        cost = price * shares * 1.0003
                        cash -= cost
                        position = (code, price, date, shares)
                        max_pnl_in_trade = 0
                        trades.append({
                            'date': date, 'code': code, 'action': 'BUY',
                            'price': price, 'shares': shares, 'score': score,
                            'market_env': market_env, 'avg_rsi': avg_rsi
                        })

        # ─── 计算净值 ───
        if position:
            code, entry_price, _, shares = position
            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if not today_row.empty:
                current_value = cash + today_row['close'].iloc[0] * shares * 0.9997
            else:
                current_value = cash + entry_price * shares * 0.9997
        else:
            current_value = cash
        equity_curve.append(current_value)

    return trades, equity_curve, dates


# ═══════════════════════════════════════════════════════════
# 结果分析 (增强版)
# ═══════════════════════════════════════════════════════════

def analyze_results(trades, equity_curve, dates, initial_cash, label=''):
    if not trades:
        log(f"  [{label}] 无交易记录")
        return {}

    sells = [t for t in trades if t['action'] == 'SELL']
    buys = [t for t in trades if t['action'] == 'BUY']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    losses = [t for t in sells if t.get('pnl', 0) <= 0]

    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    start = dates[65] if len(dates) > 65 else dates[0]
    n_years = (dates[-1] - start).days / 365
    annual_return = ((final_value / initial_cash) ** (1 / max(n_years, 0.1)) - 1) * 100

    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - peak) / peak * 100
    max_dd = drawdown.min()

    win_rate = len(wins) / max(1, len(sells)) * 100

    log("\n" + "=" * 65)
    log(f"回测结果 {label}")
    log("=" * 65)
    log(f"  初始资金: {initial_cash:,.0f}")
    log(f"  最终资金: {final_value:,.0f}")
    log(f"  总收益率: {total_return:.2f}%")
    log(f"  年化收益率: {annual_return:.2f}%")
    log(f"  夏普比率: {sharpe:.2f}")
    log(f"  最大回撤: {max_dd:.2f}%")
    log(f"  交易次数: {len(sells)} (买入{len(buys)}, 卖出{len(sells)})")
    log(f"  胜率: {win_rate:.1f}%")

    if sells:
        pnls = [t['pnl'] for t in sells]
        log(f"  平均收益: {np.mean(pnls):.2f}%")
        log(f"  最大单笔盈利: {max(pnls):.2f}%")
        log(f"  最大单笔亏损: {min(pnls):.2f}%")
        log(f"  盈亏比: {abs(np.mean([p for p in pnls if p > 0]) / np.mean([p for p in pnls if p < 0])):.2f}" if losses else "  盈亏比: ∞")

    # 持有天数分析
    if sells:
        hold_groups = {'1-5': [], '6-10': [], '11-15': [], '16-20': [], '21+': []}
        for t in sells:
            h = t.get('hold_days', 0)
            if h <= 5:
                hold_groups['1-5'].append(t['pnl'])
            elif h <= 10:
                hold_groups['6-10'].append(t['pnl'])
            elif h <= 15:
                hold_groups['11-15'].append(t['pnl'])
            elif h <= 20:
                hold_groups['16-20'].append(t['pnl'])
            else:
                hold_groups['21+'].append(t['pnl'])

        log("\n  持有天数分析:")
        for days, pnls_g in hold_groups.items():
            if pnls_g:
                avg = np.mean(pnls_g)
                win_r = len([p for p in pnls_g if p > 0]) / len(pnls_g) * 100
                log(f"    {days}天: {len(pnls_g)}笔, 平均{avg:+.2f}%, 胜率{win_r:.0f}%")

    # 卖出原因分析
    if sells:
        reason_groups = {}
        for t in sells:
            r = t.get('reason', 'other')
            if r not in reason_groups:
                reason_groups[r] = []
            reason_groups[r].append(t['pnl'])

        log("\n  卖出原因分析:")
        for reason, pnls_g in reason_groups.items():
            avg = np.mean(pnls_g)
            win_r = len([p for p in pnls_g if p > 0]) / len(pnls_g) * 100
            log(f"    {reason}: {len(pnls_g)}笔, 平均{avg:+.2f}%, 胜率{win_r:.0f}%")

    # 最近15笔交易
    log("\n  最近15笔交易:")
    for t in trades[-15:]:
        if t['action'] == 'BUY':
            log(f"    {t['date'].strftime('%Y-%m-%d')} BUY  {t['code']} "
                f"@{t['price']:.2f} [score:{t.get('score', 0):.0f} env:{t.get('market_env', '')}]")
        else:
            log(f"    {t['date'].strftime('%Y-%m-%d')} SELL {t['code']} "
                f"@{t['price']:.2f} pnl:{t.get('pnl', 0):+.2f}% "
                f"[{t.get('reason', '')} {t.get('hold_days', 0)}天 峰值:{t.get('max_pnl', 0):.1f}%]")

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'n_trades': len(sells),
        'avg_pnl': np.mean([t['pnl'] for t in sells]) if sells else 0,
    }


# ═══════════════════════════════════════════════════════════
# 参数扫描
# ═══════════════════════════════════════════════════════════

def quick_scan(stocks, date_range=None):
    if date_range is None:
        date_range = ['2024-01-01', '2026-03-20']

    log("\n" + "=" * 65)
    log("v13 参数扫描")
    log("=" * 65)

    best_annual = -999
    best_params = None
    results = []

    params_grid = [
        # (take_profit, stop_loss, max_hold, min_score, trailing_profit)
        # 组1: 基础参数对比
        (0.30, -0.08, 25, 80, 0.05),
        (0.30, -0.08, 20, 80, 0.05),
        (0.30, -0.10, 25, 80, 0.05),

        # 组2: 高止盈
        (0.35, -0.08, 25, 80, 0.05),
        (0.35, -0.08, 30, 80, 0.06),
        (0.35, -0.10, 25, 85, 0.05),

        # 组3: 不同止损和追踪
        (0.30, -0.06, 25, 80, 0.04),
        (0.30, -0.08, 25, 85, 0.05),
        (0.35, -0.08, 25, 80, 0.08),

        # 组4: 极端参数
        (0.40, -0.08, 30, 80, 0.06),
        (0.25, -0.08, 20, 80, 0.04),
        (0.30, -0.05, 20, 85, 0.04),
    ]

    for tp, sl, mh, ms, trail in params_grid:
        trades, equity, dates = run_backtest(
            stocks,
            start_date=date_range[0],
            end_date=date_range[1],
            take_profit=tp,
            stop_loss=sl,
            max_hold=mh,
            min_score=ms,
            trailing_profit=trail,
        )

        if len(equity) < 10:
            continue

        sells = [t for t in trades if t['action'] == 'SELL']
        final = equity[-1]
        n_years = max((dates[-1] - dates[65]).days / 365, 0.1) if len(dates) > 65 else 0.5
        annual = ((final / 1000000) ** (1 / n_years) - 1) * 100
        win_rate = len([t for t in sells if t.get('pnl', 0) > 0]) / max(1, len(sells)) * 100

        result = {
            'take_profit': tp, 'stop_loss': sl, 'max_hold': mh,
            'min_score': ms, 'trailing_profit': trail,
            'annual': annual, 'win_rate': win_rate, 'n_trades': len(sells),
            'final': final
        }
        results.append(result)

        marker = ''
        if annual > best_annual:
            best_annual = annual
            best_params = result
            marker = ' ★★★'

        log(f"  止盈{tp:.0%} 止损{sl:.0%} 持有{mh}天 门槛{ms} 追踪{trail:.0%} "
            f"→ 年化{annual:+.2f}% 胜率{win_rate:.0f}% {len(sells)}笔{marker}")

    log(f"\n  ═══ 最佳参数 ═══")
    if best_params:
        log(f"  止盈={best_params['take_profit']:.0%}, "
            f"止损={best_params['stop_loss']:.0%}, "
            f"持有={best_params['max_hold']}天, "
            f"门槛={best_params['min_score']}, "
            f"追踪={best_params['trailing_profit']:.0%}")
        log(f"  年化={best_annual:+.2f}%, 胜率={best_params['win_rate']:.0f}%, "
            f"交易={best_params['n_trades']}笔")

    return best_params, results


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    log("=" * 65)
    log("多因子Alpha量化交易系统 v13.0")
    log("缩量回踩 + 逻辑止损 + 市场RSI过滤")
    log("=" * 65)

    log("\n[1] 加载股票数据...")
    stocks = load_all_stocks('E:/data', min_rows=250, max_stocks=100)
    log(f"    加载 {len(stocks)} 只股票")

    log("\n[2] 预计算因子...")
    prepare_factors(stocks)

    log("\n[3] 默认参数回测...")
    log("    止盈30%, 止损-8%, 追踪回撤5%, 最大持有25天")
    trades, equity, dates = run_backtest(
        stocks,
        start_date='2024-01-01',
        end_date='2026-03-20',
        initial_cash=1000000,
        take_profit=0.30,
        stop_loss=-0.08,
        max_hold=25,
        min_score=80,
        trailing_profit=0.05,
    )
    stats_default = analyze_results(trades, equity, dates, 1000000, label='默认参数')

    log("\n[4] 参数扫描 (12组参数)...")
    best_params, all_results = quick_scan(stocks)

    if best_params:
        log("\n[5] 最佳参数完整回测...")
        trades_best, equity_best, dates_best = run_backtest(
            stocks,
            start_date='2024-01-01',
            end_date='2026-03-20',
            initial_cash=1000000,
            take_profit=best_params['take_profit'],
            stop_loss=best_params['stop_loss'],
            max_hold=best_params['max_hold'],
            min_score=best_params['min_score'],
            trailing_profit=best_params['trailing_profit'],
        )
        stats_best = analyze_results(trades_best, equity_best, dates_best, 1000000, label='最佳参数')

    # ─── 策略总结 ───
    log("\n" + "=" * 65)
    log("v13 策略设计要点总结")
    log("=" * 65)
    log("""
    1. 止盈: 30-35%, 让利润奔跑（vs v11的25%）
    2. 止损: -8%硬止损, 不再按时间割肉
       → 解决v12"时间止损"的问题
    3. 追踪止盈: 浮盈达到止盈*60%后, 回撤超5%就锁利
       → 避免"到手的利润飞了"
    4. 缩量回踩入场: 趋势向上 + 回踩MA5/MA10 + 成交量萎缩
       → 经典"缩量回踩买入"信号
    5. 市场RSI过滤: 全市场RSI<40时不开仓
       → 避免在极弱市场中被套
    6. 量价背离因子: 下跌缩量=卖压衰竭(加分)
    7. 到期不强卖: 超过max_hold天时评估状态
       → 大幅盈利继续持有, 小亏/超买才卖
    """)

    log("\n完成!")
