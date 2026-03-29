# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v15.0 (快进快出版)
===============================================
核心理念 — A股是均值回归市场:
- 1-5天持有: 胜率接近100%, 平均+10-20%
- 6-10天持有: 胜率接近100%, 平均+10-15%
- 11+天持有: 胜率骤降, 持有越久亏越多

v15设计:
1. 持有上限5天 — 不恋战, 到期必走
2. 止盈5-8% — 不贪心, 小赚就走
3. 无止损 — 不主动割肉, 等到期或止盈
4. 追击止盈 — 从最高点回撤3-5%时锁利
5. RSI>80不买入 — 避免追高
6. 选股分数门槛95 — 只买最最强的股票
7. 参数扫描: 止盈(5-10%) × 持有(5/7/10天) × 追击(3/5%)
"""

import pandas as pd
import numpy as np
import os
from itertools import product


def log(msg):
    print(msg, flush=True)


# ========== 1. 数据加载 ==========

def load_all_stocks(data_dir='E:/data', min_rows=200, max_stocks=300):
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


# ========== 2. 因子计算 ==========

def add_factors(df):
    df = df.copy()

    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20)

    df['ma3'] = df['close'].rolling(3).mean()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    df['ma_bull'] = ((df['ma5'] > df['ma10']) &
                     (df['ma10'] > df['ma20']) &
                     (df['close'] > df['ma20'])).astype(int)

    df['ma_bull_short'] = ((df['ma3'] > df['ma5']) &
                           (df['ma5'] > df['ma10']) &
                           (df['close'] > df['ma5'])).astype(int)

    df['bbi'] = (df['ma5'] + df['ma10'] + df['ma20'] + df['ma60']) / 4
    df['bbi_ratio'] = df['close'] / df['bbi']

    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    df['vol_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()

    df['high_10'] = df['high'].rolling(10).max()
    df['low_10'] = df['low'].rolling(10).min()
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_pos_10'] = (df['close'] - df['low_10']) / (df['high_10'] - df['low_10'] + 1e-8)
    df['price_pos_20'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

    df['ma10_slope'] = (df['ma10'] - df['ma10'].shift(5)) / (df['ma10'].shift(5) + 1e-8)
    df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(10)) / (df['ma20'].shift(10) + 1e-8)

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    gain_6 = delta.where(delta > 0, 0).rolling(6).mean()
    loss_6 = (-delta.where(delta < 0, 0)).rolling(6).mean()
    rs_6 = gain_6 / (loss_6 + 1e-8)
    df['rsi_6'] = 100 - (100 / (1 + rs_6))

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

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    df['new_high_10'] = (df['close'] >= df['high'].rolling(10).max()).astype(int)
    df['new_high_20'] = (df['close'] >= df['high'].rolling(20).max()).astype(int)

    df['gap_up'] = ((df['open'] / df['close'].shift(1) - 1) > 0.01).astype(int)

    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 1e-8)
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 1e-8)

    df['consec_up'] = 0
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            df.loc[df.index[i], 'consec_up'] = df['consec_up'].iloc[i-1] + 1

    return df


def prepare_factors(stocks):
    log("  预计算因子...")
    for code, df in stocks.items():
        stocks[code] = add_factors(df)
    log(f"  完成 {len(stocks)} 只股票的因子计算")


# ========== 3. 市场环境 ==========

def get_market_env(stocks, date):
    bull_count = 0
    total = 0

    for code, df in stocks.items():
        hist = df[df['date'] <= date]
        if len(hist) < 25:
            continue
        total += 1
        row = hist.iloc[-1]
        if row['close'] > row['ma20']:
            bull_count += 1

    if total == 0:
        return 'neutral', 0.5

    bull_ratio = bull_count / total
    if bull_ratio > 0.55:
        return 'bull', bull_ratio
    elif bull_ratio < 0.45:
        return 'bear', bull_ratio
    else:
        return 'neutral', bull_ratio


# ========== 4. 选股 (v15: 超严格) ==========

class Selector:
    def __init__(self, stocks):
        self.stocks = stocks

    def select_top(self, date, market_env='neutral', min_score=95, top_n=3):
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
            if row.get('vol_ratio', 1) < 0.5 or row.get('vol_ratio', 1) > 5:
                continue

            rsi = row.get('rsi_14', 50)
            if rsi > 80:
                continue

            score = 0

            # 1. 短期动量 (v15核心: 5天内能涨的股票)
            mom_5 = row.get('ret_5', 0)
            mom_10 = row.get('ret_10', 0)
            mom_20 = row.get('ret_20', 0)

            if mom_5 > 0.10:
                score += 30
            elif mom_5 > 0.05:
                score += 20
            elif mom_5 > 0.02:
                score += 10

            if mom_10 > 0.15:
                score += 25
            elif mom_10 > 0.08:
                score += 15
            elif mom_10 > 0.03:
                score += 8

            if mom_20 > 0.20:
                score += 20
            elif mom_20 > 0.10:
                score += 15
            elif mom_20 > 0.05:
                score += 8

            # 2. 短期均线多头
            if row.get('ma_bull_short', 0) == 1:
                score += 20

            # 3. 长期均线多头
            if row.get('ma_bull', 0) == 1:
                score += 15

            # 4. BBI上方
            bbi_r = row.get('bbi_ratio', 1)
            if bbi_r > 1.02:
                score += 12
            elif bbi_r > 1:
                score += 6

            # 5. 短期趋势加速
            slope_10 = row.get('ma10_slope', 0)
            if slope_10 > 0.03:
                score += 12
            elif slope_10 > 0.01:
                score += 6

            # 6. 创新高
            if row.get('new_high_10', 0) == 1:
                score += 10
            if row.get('new_high_20', 0) == 1:
                score += 8

            # 7. RSI强势区间 (50-75最佳, 不能>80)
            if 55 < rsi < 75:
                score += 10
            elif 45 < rsi <= 55:
                score += 5

            # 8. 量比放大 (温和放量最佳)
            vol_r = row.get('vol_ratio', 1)
            if 1.3 < vol_r < 2.5:
                score += 8
            elif 1.1 < vol_r <= 1.3:
                score += 4

            # 9. KDJ金叉且未超买
            if row.get('K', 50) > row.get('D', 50) and row.get('K', 50) < 80:
                score += 8

            # 10. MACD多头
            if row.get('MACD', 0) > 0:
                score += 5
            if row.get('DIF', 0) > row.get('DEA', 0):
                score += 3

            # 11. 连涨势头 (2-4天连涨最佳)
            consec = row.get('consec_up', 0)
            if 2 <= consec <= 4:
                score += 8
            elif consec == 1:
                score += 3
            elif consec > 5:
                score -= 5

            # 12. 低波动优先 (波动小更容易控制)
            vol = row.get('vol_5', 0.05)
            if vol < 0.015:
                score += 8
            elif vol < 0.025:
                score += 4

            # 负面清单 — 直接跳过比扣分更严格
            if mom_5 < -0.05:
                continue
            if mom_20 < -0.10:
                continue
            if row.get('upper_shadow', 0) > 0.6:
                score -= 10

            # 环境适应
            if market_env == 'bull':
                if row.get('ma_bull', 0) == 1:
                    score *= 1.2
            elif market_env == 'bear':
                score *= 0.7

            if score >= min_score:
                candidates.append((code, score, row['close'], rsi))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ========== 5. 回测引擎 (v15: 快进快出) ==========

def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-28',
                 initial_cash=1000000, take_profit=0.06,
                 trail_stop=0.03, max_hold=5, min_score=95):
    """
    v15回测: 快进快出, 追击止盈

    参数:
    - take_profit: 固定止盈(默认6%)
    - trail_stop: 追击止盈回撤比例(从最高点回撤x%即卖, 默认3%)
    - max_hold: 最大持有天数(默认5天)
    - min_score: 选股分数门槛(默认95)
    """

    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates
                    if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])

    if len(dates) < 70:
        log("  交易日不足, 跳过")
        return [], [initial_cash], dates

    log(f"  回测期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    log(f"  交易日: {len(dates)}, 股票池: {len(stocks)}")
    log(f"  参数: 止盈{take_profit*100:.0f}% 追击{trail_stop*100:.0f}% 持有{max_hold}天 分数{min_score}")

    selector = Selector(stocks)

    cash = initial_cash
    position = None
    peak_price = 0.0
    trades = []
    equity_curve = [initial_cash]

    market_env_cache = ('neutral', 0.5)

    for i, date in enumerate(dates):
        if i < 65:
            continue

        if i % 5 == 0:
            market_env_cache = get_market_env(stocks, date)
        market_env, bull_ratio = market_env_cache

        # ---- 持仓处理 ----
        if position:
            code, entry_price, entry_date, shares = position

            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if today_row.empty:
                equity_curve.append(equity_curve[-1])
                continue

            price = today_row['close'].iloc[0]
            today_high = today_row['high'].iloc[0]

            peak_price = max(peak_price, today_high)

            hold_days = (date - entry_date).days
            pnl = (price - entry_price) / entry_price

            sell_reason = None

            # (a) 固定止盈
            if pnl >= take_profit:
                sell_reason = '止盈'

            # (b) 追击止盈: 只在有盈利时生效
            elif peak_price > entry_price * (1 + trail_stop):
                drawdown_from_peak = (peak_price - price) / peak_price
                if drawdown_from_peak >= trail_stop:
                    sell_reason = '追击止盈'

            # (c) 到期必走
            elif hold_days >= max_hold:
                sell_reason = '到期'

            if sell_reason:
                revenue = price * shares * 0.9997
                cash += revenue
                pnl_pct = pnl * 100
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl_pct,
                    'reason': sell_reason, 'hold_days': hold_days,
                    'peak_price': peak_price
                })
                position = None
                peak_price = 0.0

        # ---- 买入 ----
        if position is None:
            top_stocks = selector.select_top(
                date, market_env=market_env, min_score=min_score, top_n=3
            )
            if top_stocks:
                code, score, price, rsi = top_stocks[0]
                position_size = min(int(cash * 0.95 / (price * 100)), 10) * 100
                if position_size >= 100 and price * 100 * 1.0003 <= cash:
                    shares = position_size
                    cost = price * shares * 1.0003
                    cash -= cost
                    position = (code, price, date, shares)
                    peak_price = price
                    trades.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'price': price, 'shares': shares, 'score': score,
                        'market_env': market_env, 'rsi': rsi
                    })

        # ---- 更新净值 ----
        if position:
            code, entry_price, _, shares = position
            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if not today_row.empty:
                current_value = cash + today_row['close'].iloc[0] * shares * 0.9997
            else:
                current_value = equity_curve[-1]
        else:
            current_value = cash
        equity_curve.append(current_value)

    return trades, equity_curve, dates


# ========== 6. 结果分析 ==========

def analyze_results(trades, equity_curve, dates, initial_cash, label=''):
    if not trades:
        log("  无交易记录")
        return {}

    sells = [t for t in trades if t['action'] == 'SELL']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    losses = [t for t in sells if t.get('pnl', 0) <= 0]

    if not sells:
        log("  无卖出记录")
        return {}

    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    start = dates[65] if len(dates) > 65 else dates[0]
    n_years = (dates[-1] - start).days / 365
    annual_return = ((final_value / initial_cash) ** (1 / max(n_years, 0.1)) - 1) * 100

    eq = np.array(equity_curve, dtype=float)
    returns = np.diff(eq) / eq[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak * 100
    max_dd = drawdown.min()

    pnls = [t['pnl'] for t in sells]
    avg_pnl = np.mean(pnls)
    win_rate = len(wins) / len(sells) * 100

    avg_hold = np.mean([t.get('hold_days', 0) for t in sells])

    log(f"\n{'=' * 60}")
    log(f"回测结果 {label}")
    log(f"{'=' * 60}")
    log(f"  初始资金: {initial_cash:,.0f}")
    log(f"  最终资金: {final_value:,.0f}")
    log(f"  总收益率: {total_return:+.2f}%")
    log(f"  年化收益率: {annual_return:+.2f}%")
    log(f"  夏普比率: {sharpe:.2f}")
    log(f"  最大回撤: {max_dd:.2f}%")
    log(f"  交易次数: {len(sells)}")
    log(f"  胜率: {win_rate:.1f}%")
    log(f"  平均收益: {avg_pnl:+.2f}%")
    log(f"  平均持有: {avg_hold:.1f}天")

    if pnls:
        log(f"  最大单笔盈利: {max(pnls):+.2f}%")
        log(f"  最大单笔亏损: {min(pnls):+.2f}%")
        log(f"  盈亏比: {abs(np.mean([p for p in pnls if p > 0]) / np.mean([p for p in pnls if p < 0])):.2f}" if losses else "  盈亏比: ∞")

    # 持有天数分析
    hold_groups = {'1-2': [], '3-5': [], '6-7': [], '8-10': [], '11+': []}
    for t in sells:
        h = t.get('hold_days', 0)
        if h <= 2:
            hold_groups['1-2'].append(t['pnl'])
        elif h <= 5:
            hold_groups['3-5'].append(t['pnl'])
        elif h <= 7:
            hold_groups['6-7'].append(t['pnl'])
        elif h <= 10:
            hold_groups['8-10'].append(t['pnl'])
        else:
            hold_groups['11+'].append(t['pnl'])

    log("\n  持有天数分析:")
    for days, pnl_list in hold_groups.items():
        if pnl_list:
            avg = np.mean(pnl_list)
            wr = len([p for p in pnl_list if p > 0]) / len(pnl_list) * 100
            log(f"    {days}天: {len(pnl_list)}笔, 平均{avg:+.2f}%, 胜率{wr:.0f}%")

    # 卖出原因分析
    reason_groups = {}
    for t in sells:
        r = t.get('reason', 'other')
        if r not in reason_groups:
            reason_groups[r] = []
        reason_groups[r].append(t['pnl'])

    log("\n  卖出原因分析:")
    for reason, pnl_list in reason_groups.items():
        avg = np.mean(pnl_list)
        wr = len([p for p in pnl_list if p > 0]) / len(pnl_list) * 100
        log(f"    {reason}: {len(pnl_list)}笔, 平均{avg:+.2f}%, 胜率{wr:.0f}%")

    # 最近交易
    log(f"\n  最近15笔交易:")
    for t in trades[-15:]:
        if t['action'] == 'BUY':
            log(f"    {t['date'].strftime('%Y-%m-%d')} BUY  {t['code']} "
                f"@{t['price']:.2f} [score:{t.get('score',0):.0f} rsi:{t.get('rsi',0):.0f}]")
        else:
            log(f"    {t['date'].strftime('%Y-%m-%d')} SELL {t['code']} "
                f"@{t['price']:.2f}  pnl:{t.get('pnl',0):+.2f}% "
                f"[{t.get('reason','')} {t.get('hold_days',0)}天]")

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len(sells),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'avg_hold': avg_hold,
    }


# ========== 7. 参数扫描 (v15核心) ==========

def param_scan(stocks, date_range=('2024-01-01', '2025-06-01')):
    """
    扫描: 止盈(5-10%) × 持有(5/7/10天) × 追击(3/5%) × 分数(90/95)
    """
    log(f"\n{'=' * 60}")
    log("v15 参数扫描")
    log(f"{'=' * 60}")
    log(f"  扫描区间: {date_range[0]} ~ {date_range[1]}")

    take_profits = [0.05, 0.06, 0.07, 0.08, 0.10]
    max_holds = [5, 7, 10]
    trail_stops = [0.03, 0.05]
    min_scores = [90, 95]

    results = []
    best_annual = -999
    best_params = None
    total_combos = len(take_profits) * len(max_holds) * len(trail_stops) * len(min_scores)

    log(f"  共 {total_combos} 个参数组合\n")

    for idx, (tp, mh, ts, ms) in enumerate(product(take_profits, max_holds, trail_stops, min_scores)):
        trades, equity, dates = run_backtest(
            stocks,
            start_date=date_range[0],
            end_date=date_range[1],
            take_profit=tp,
            trail_stop=ts,
            max_hold=mh,
            min_score=ms,
        )

        if len(equity) < 10:
            continue

        sells = [t for t in trades if t['action'] == 'SELL']
        if not sells:
            continue

        final = equity[-1]
        start_idx = min(65, len(dates) - 1)
        n_years = (dates[-1] - dates[start_idx]).days / 365
        annual = ((final / 1000000) ** (1 / max(n_years, 0.1)) - 1) * 100

        pnls = [t['pnl'] for t in sells]
        win_rate = len([p for p in pnls if p > 0]) / len(pnls) * 100
        avg_pnl = np.mean(pnls)

        eq = np.array(equity, dtype=float)
        peak = np.maximum.accumulate(eq)
        max_dd = ((eq - peak) / peak * 100).min()

        result = {
            'take_profit': tp, 'max_hold': mh, 'trail_stop': ts, 'min_score': ms,
            'annual': annual, 'win_rate': win_rate, 'avg_pnl': avg_pnl,
            'max_dd': max_dd, 'n_trades': len(sells), 'final': final,
        }
        results.append(result)

        if annual > best_annual:
            best_annual = annual
            best_params = result.copy()
            log(f"  *** 新最佳 [{idx+1}/{total_combos}]: "
                f"年化{annual:+.2f}% 胜率{win_rate:.0f}% "
                f"止盈{tp*100:.0f}% 追击{ts*100:.0f}% "
                f"持有{mh}天 分数{ms}")

    # 排序输出 TOP 10
    results.sort(key=lambda x: x['annual'], reverse=True)

    log(f"\n{'=' * 60}")
    log("TOP 10 参数组合:")
    log(f"{'=' * 60}")
    log(f"  {'止盈':>4} {'追击':>4} {'持有':>4} {'分数':>4} | "
        f"{'年化':>8} {'胜率':>6} {'平均':>6} {'回撤':>7} {'交易':>4}")
    log(f"  {'-'*55}")

    for r in results[:10]:
        log(f"  {r['take_profit']*100:4.0f}% {r['trail_stop']*100:4.0f}% "
            f"{r['max_hold']:4d}天 {r['min_score']:4.0f} | "
            f"{r['annual']:+7.2f}% {r['win_rate']:5.1f}% "
            f"{r['avg_pnl']:+5.2f}% {r['max_dd']:+6.2f}% {r['n_trades']:4d}")

    if best_params:
        log(f"\n  >>> 最佳参数: 止盈{best_params['take_profit']*100:.0f}% "
            f"追击{best_params['trail_stop']*100:.0f}% "
            f"持有{best_params['max_hold']}天 "
            f"分数{best_params['min_score']} "
            f"年化{best_params['annual']:+.2f}%")

    return best_params, results


# ========== 8. 主流程 ==========

if __name__ == '__main__':
    log("=" * 60)
    log("多因子Alpha量化交易系统 v15.0 (快进快出版)")
    log("=" * 60)
    log("核心理念: A股均值回归, 5天持有上限, 小赚即走")

    log("\n[1] 加载股票数据...")
    stocks = load_all_stocks('E:/data', min_rows=250, max_stocks=300)
    log(f"    加载 {len(stocks)} 只股票")

    log("\n[2] 预计算因子...")
    prepare_factors(stocks)

    # ---- Step 3: 默认参数回测 ----
    log("\n[3] 默认参数回测 (止盈6%, 追击3%, 持有5天, 分数95)...")
    trades, equity, dates = run_backtest(
        stocks,
        start_date='2024-01-01',
        end_date='2026-03-28',
        initial_cash=1000000,
        take_profit=0.06,
        trail_stop=0.03,
        max_hold=5,
        min_score=95,
    )
    analyze_results(trades, equity, dates, 1000000, label='[默认参数]')

    # ---- Step 4: 参数扫描 ----
    log("\n[4] 参数扫描 (训练期: 2024-01 ~ 2025-06)...")
    best_params, all_results = param_scan(stocks, date_range=('2024-01-01', '2025-06-01'))

    # ---- Step 5: 最佳参数完整回测 ----
    if best_params:
        log(f"\n[5] 最佳参数完整回测 (2024-01 ~ 2026-03)...")
        trades, equity, dates = run_backtest(
            stocks,
            start_date='2024-01-01',
            end_date='2026-03-28',
            initial_cash=1000000,
            take_profit=best_params['take_profit'],
            trail_stop=best_params['trail_stop'],
            max_hold=best_params['max_hold'],
            min_score=best_params['min_score'],
        )
        analyze_results(trades, equity, dates, 1000000, label='[最佳参数]')

        # ---- Step 6: 样本外验证 ----
        log(f"\n[6] 样本外验证 (2025-07 ~ 2026-03)...")
        trades_oos, equity_oos, dates_oos = run_backtest(
            stocks,
            start_date='2025-07-01',
            end_date='2026-03-28',
            initial_cash=1000000,
            take_profit=best_params['take_profit'],
            trail_stop=best_params['trail_stop'],
            max_hold=best_params['max_hold'],
            min_score=best_params['min_score'],
        )
        analyze_results(trades_oos, equity_oos, dates_oos, 1000000, label='[样本外验证]')

    log("\n" + "=" * 60)
    log("v15 完成!")
    log("=" * 60)
