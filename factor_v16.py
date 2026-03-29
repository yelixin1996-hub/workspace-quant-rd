# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v16.0
===============================================
基于v15验证数据的核心改进:

v15数据:
- 止盈(10%): 50笔, 胜率100%, 平均+12.22% ✅ 完美
- 追击止盈(3%): 117笔, 胜率52%, +0.50% ⚠️ 太紧，打断盈利
- 到期(5天): 56笔, 胜率7%, -7.49% ❌ 亏损根源
- 1-2天持有: 胜率64%, +3.31%
- 3-5天持有: 胜率44-62%, +0~3%

v16改进:
1. 仓位管理 — 可用资金的10-20%，而非固定100股
2. 高阈值追击止盈 — 8%触发（替代3%），避免打断盈利交易
3. 智能到期 — 5天到期时盈利>0%则延长至8天，亏损则立即卖出
4. 亏损不加仓 — 持仓浮亏时不开新仓位
5. 参数扫描 — 止盈(8-15%) × 持有(5/8天) × 追击(8/10%)
6. 样本外验证 — 2025-07之后独立测试
"""

import pandas as pd
import numpy as np
import os
from itertools import product


def log(msg):
    print(msg, flush=True)


# ========== 1. 数据加载 ==========

def load_all_stocks(data_dir='E:/data', min_rows=200, max_stocks=100):
    stocks = {}
    if not os.path.exists(data_dir):
        log(f"  ⚠ 数据目录不存在: {data_dir}")
        return stocks
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:max_stocks]
    for f in files:
        code = f.replace('.SZ.csv', '').replace('.SH.csv', '')
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            df = df.rename(columns={'trade_date': 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            if len(df) >= min_rows:
                stocks[code] = df
        except Exception:
            continue
    return stocks


# ========== 2. 因子计算 ==========

def add_factors(df):
    df = df.copy()

    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)
    df['ret_60'] = df['close'].pct_change(60)

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    df['ma_bull'] = ((df['ma5'] > df['ma10']) &
                     (df['ma10'] > df['ma20']) &
                     (df['close'] > df['ma20'])).astype(int)

    df['bbi'] = (df['ma5'] + df['ma10'] + df['ma20'] + df['ma60']) / 4
    df['bbi_ratio'] = df['close'] / df['bbi']

    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()

    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

    df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(10)) / (df['ma20'].shift(10) + 1e-8)

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

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

    df['new_high_20'] = (df['close'] >= df['high'].rolling(20).max()).astype(int)

    return df


def prepare_factors(stocks):
    log("  预计算因子...")
    for code in list(stocks.keys()):
        stocks[code] = add_factors(stocks[code])
    log(f"  完成 {len(stocks)} 只股票的因子计算")


# ========== 3. 市场环境判断 ==========

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


# ========== 4. 选股器 ==========

class Selector:
    def __init__(self, stocks):
        self.stocks = stocks

    def select_top(self, date, market_env='neutral', min_score=90, top_n=3):
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

            score = 0

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

            if row.get('ma_bull', 0) == 1:
                score += 25

            if row.get('bbi_ratio', 1) > 1:
                score += 15 * (row['bbi_ratio'] - 1)

            slope = row.get('ma20_slope', 0)
            if slope > 0.02:
                score += 15
            elif slope < 0:
                score += 10 * slope

            if row.get('new_high_20', 0) == 1:
                score += 15

            rsi = row.get('rsi_14', 50)
            if 45 < rsi < 65:
                score += 10
            elif rsi < 40:
                score += 5

            vol_r = row.get('vol_ratio', 1)
            if 1.2 < vol_r < 3:
                score += 8

            vol = row.get('vol_20', 0.05)
            if vol < 0.02:
                score += 10
            elif vol < 0.03:
                score += 5

            if row.get('K', 50) > row.get('D', 50) and row.get('K', 50) < 70:
                score += 8

            if row.get('MACD', 0) > 0:
                score += 5

            if mom < -0.15:
                score -= 30
            if rsi > 80:
                score -= 20

            if market_env == 'bull':
                if row.get('ma_bull', 0) == 1:
                    score *= 1.3
            elif market_env == 'bear':
                continue

            if score >= min_score:
                candidates.append((code, score, row['close']))

        if not candidates:
            return []
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ========== 5. v16 回测引擎 ==========

def calc_position_shares(price, cash, position_pct=0.15):
    """
    按可用资金百分比计算持仓股数（A股最小100股，向下取整到100的倍数）。
    position_pct: 目标仓位占可用资金比例 (0.10~0.20)
    """
    target_value = cash * position_pct
    raw_shares = int(target_value / price)
    shares = (raw_shares // 100) * 100
    return max(shares, 100)


def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-29',
                 initial_cash=1000000,
                 take_profit=0.10,
                 trailing_trigger=0.08,
                 base_hold=5,
                 max_hold=8,
                 position_pct=0.15,
                 min_score=90):
    """
    v16 回测引擎

    核心逻辑:
    - 仓位: 可用资金 * position_pct（而非固定100股）
    - 止盈: 收益 >= take_profit 时卖出
    - 追击止盈: 收益达到 trailing_trigger 后，从最高点回落2%则卖出
    - 智能到期: base_hold天到期时，盈利(>0%)则续持至max_hold天；亏损则立即卖出
    - 亏损不加仓: 有持仓且浮亏时不开新仓
    """

    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates
                    if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])

    if not dates:
        return [], [initial_cash], []

    log(f"  回测期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    log(f"  交易日: {len(dates)}, 股票池: {len(stocks)}")

    selector = Selector(stocks)

    cash = initial_cash
    position = None  # (code, entry_price, entry_date, shares, peak_pnl)
    trades = []
    equity_curve = [initial_cash]

    market_env_cache = ('neutral', 0.5)
    last_env_check = -10

    for i, date in enumerate(dates):
        if i < 65:
            equity_curve.append(initial_cash)
            continue

        if i - last_env_check >= 5:
            market_env_cache = get_market_env(stocks, date)
            last_env_check = i
        market_env, bull_ratio = market_env_cache

        # --- 持仓管理 ---
        if position:
            code, entry_price, entry_date, shares, peak_pnl = position

            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if today_row.empty:
                equity_curve.append(equity_curve[-1])
                continue

            price = today_row['close'].iloc[0]
            hold_days = (date - entry_date).days
            pnl = (price - entry_price) / entry_price
            peak_pnl = max(peak_pnl, pnl)
            position = (code, entry_price, entry_date, shares, peak_pnl)

            sell_reason = None

            # 规则1: 止盈 — 达到目标收益
            if pnl >= take_profit:
                sell_reason = '止盈'

            # 规则2: 追击止盈 — 到达高位后回落
            # 当收益曾超过trailing_trigger，但从最高点回落超过2个百分点
            elif peak_pnl >= trailing_trigger and pnl < (peak_pnl - 0.02):
                sell_reason = '追击止盈'

            # 规则3: 智能到期
            # base_hold天到期 → 亏损立即卖；盈利则延长
            elif hold_days >= base_hold:
                if pnl <= 0:
                    sell_reason = '到期止损'
                elif hold_days >= max_hold:
                    sell_reason = '延期到期'

            # 规则4: RSI极端超买 + 有盈利
            elif today_row['rsi_14'].iloc[0] > 88 and pnl > 0.05:
                sell_reason = 'RSI超买'

            if sell_reason:
                revenue = price * shares * 0.9997  # 扣手续费
                cash += revenue
                pnl_pct = pnl * 100
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl_pct,
                    'reason': sell_reason, 'hold_days': hold_days,
                    'peak_pnl': peak_pnl * 100
                })
                position = None

        # --- 买入逻辑 ---
        if position is None and market_env != 'bear':
            top_stocks = selector.select_top(date, market_env=market_env,
                                             min_score=min_score, top_n=3)
            if top_stocks:
                code, score, price = top_stocks[0]
                shares = calc_position_shares(price, cash, position_pct)
                cost = price * shares * 1.0003
                if cost <= cash and shares >= 100:
                    cash -= cost
                    position = (code, price, date, shares, 0.0)
                    trades.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'price': price, 'shares': shares, 'score': score,
                        'market_env': market_env,
                        'position_value': price * shares,
                        'position_pct_actual': (price * shares) / (cash + price * shares) * 100
                    })

        # 更新净值曲线
        if position:
            code, entry_price, _, shares, _ = position
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
    buys = [t for t in trades if t['action'] == 'BUY']

    if not sells:
        log("  无卖出交易")
        return {}

    wins = [t for t in sells if t.get('pnl', 0) > 0]
    losses = [t for t in sells if t.get('pnl', 0) <= 0]

    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    start = dates[65] if len(dates) > 65 else dates[0]
    n_years = (dates[-1] - start).days / 365
    annual_return = ((final_value / initial_cash) ** (1 / max(n_years, 0.1)) - 1) * 100

    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    returns = returns[np.isfinite(returns)]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - np.array(peak)) / np.array(peak) * 100
    max_dd = drawdown.min()

    avg_pos_value = np.mean([t.get('position_value', 0) for t in buys]) if buys else 0
    avg_pos_pct = np.mean([t.get('position_pct_actual', 0) for t in buys]) if buys else 0

    header = f"回测结果 [{label}]" if label else "回测结果"
    log("\n" + "=" * 60)
    log(header)
    log("=" * 60)
    log(f"  初始资金: {initial_cash:,.0f}")
    log(f"  最终资金: {final_value:,.0f}")
    log(f"  总收益率: {total_return:.2f}%")
    log(f"  年化收益率: {annual_return:.2f}%")
    log(f"  夏普比率: {sharpe:.2f}")
    log(f"  最大回撤: {max_dd:.2f}%")
    log(f"  交易次数: {len(sells)}")
    log(f"  盈利次数: {len(wins)}")
    log(f"  亏损次数: {len(losses)}")
    log(f"  胜率: {len(wins)/max(1,len(sells))*100:.1f}%")
    log(f"  平均持仓金额: {avg_pos_value:,.0f}")
    log(f"  平均仓位占比: {avg_pos_pct:.1f}%")

    if sells:
        pnls = [t['pnl'] for t in sells]
        log(f"  平均收益: {np.mean(pnls):.2f}%")
        log(f"  收益中位数: {np.median(pnls):.2f}%")
        log(f"  最大单笔盈利: {max(pnls):.2f}%")
        log(f"  最大单笔亏损: {min(pnls):.2f}%")

    # 卖出原因分析
    if sells:
        reason_groups = {}
        for t in sells:
            r = t.get('reason', 'other')
            if r not in reason_groups:
                reason_groups[r] = []
            reason_groups[r].append(t['pnl'])

        log("\n  卖出原因分析:")
        for reason, pnls in sorted(reason_groups.items()):
            avg = np.mean(pnls)
            win_r = len([p for p in pnls if p > 0]) / len(pnls) * 100
            log(f"    {reason}: {len(pnls)}笔, 平均{avg:+.2f}%, 胜率{win_r:.0f}%")

    # 持有天数分析
    if sells:
        hold_groups = {'1-2天': [], '3-5天': [], '6-8天': [], '8+天': []}
        for t in sells:
            h = t.get('hold_days', 0)
            if h <= 2:
                hold_groups['1-2天'].append(t['pnl'])
            elif h <= 5:
                hold_groups['3-5天'].append(t['pnl'])
            elif h <= 8:
                hold_groups['6-8天'].append(t['pnl'])
            else:
                hold_groups['8+天'].append(t['pnl'])

        log("\n  持有天数分析:")
        for days, pnls in hold_groups.items():
            if pnls:
                avg = np.mean(pnls)
                win_r = len([p for p in pnls if p > 0]) / len(pnls) * 100
                log(f"    {days}: {len(pnls)}笔, 平均{avg:+.2f}%, 胜率{win_r:.0f}%")

    # 最近交易
    log(f"\n  最近10笔交易:")
    for t in trades[-10:]:
        if t['action'] == 'BUY':
            log(f"    {t['date'].strftime('%Y-%m-%d')} BUY  {t['code']} @{t['price']:.2f}"
                f" [{t.get('shares',0)}股, {t.get('position_pct_actual',0):.1f}%仓位]")
        else:
            log(f"    {t['date'].strftime('%Y-%m-%d')} SELL {t['code']} @{t['price']:.2f}"
                f"  pnl:{t.get('pnl',0):+.2f}% [{t.get('reason','')}]"
                f" peak:{t.get('peak_pnl',0):+.1f}%")

    return {
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': len(wins) / max(1, len(sells)) * 100,
        'num_trades': len(sells),
        'avg_pnl': np.mean([t['pnl'] for t in sells]),
        'total_return': total_return,
        'final_value': final_value,
    }


# ========== 7. 参数扫描 ==========

def parameter_sweep(stocks, date_range=('2024-01-01', '2025-06-30'),
                    initial_cash=1000000):
    """
    参数扫描: 止盈(8-15%) × 持有(5/8天) × 追击(8/10%)
    """
    log("\n" + "=" * 60)
    log("参数扫描 (样本内)")
    log(f"  区间: {date_range[0]} ~ {date_range[1]}")
    log("=" * 60)

    take_profits = [0.08, 0.10, 0.12, 0.15]
    base_holds = [5, 8]
    trailing_triggers = [0.08, 0.10]
    position_pcts = [0.10, 0.15, 0.20]

    results = []
    best_sharpe = -999
    best_params = None

    total = len(take_profits) * len(base_holds) * len(trailing_triggers) * len(position_pcts)
    log(f"  共 {total} 组参数待测试\n")

    count = 0
    for tp, bh, tt, pp in product(take_profits, base_holds, trailing_triggers, position_pcts):
        count += 1
        max_h = max(bh + 3, 8)

        trades, equity, dates = run_backtest(
            stocks,
            start_date=date_range[0],
            end_date=date_range[1],
            initial_cash=initial_cash,
            take_profit=tp,
            trailing_trigger=tt,
            base_hold=bh,
            max_hold=max_h,
            position_pct=pp,
            min_score=90
        )

        if len(equity) < 10:
            continue

        sells = [t for t in trades if t['action'] == 'SELL']
        if not sells:
            continue

        final = equity[-1]
        start_idx = min(65, len(dates) - 1)
        n_years = (dates[-1] - dates[start_idx]).days / 365
        annual = ((final / initial_cash) ** (1 / max(n_years, 0.1)) - 1) * 100

        returns = np.diff(equity) / np.array(equity[:-1])
        returns = returns[np.isfinite(returns)]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        peak = np.maximum.accumulate(equity)
        dd = ((np.array(equity) - np.array(peak)) / np.array(peak) * 100).min()

        wins = len([t for t in sells if t.get('pnl', 0) > 0])
        win_rate = wins / len(sells) * 100
        avg_pnl = np.mean([t['pnl'] for t in sells])

        r = {
            'take_profit': tp, 'base_hold': bh, 'trailing_trigger': tt,
            'position_pct': pp, 'max_hold': max_h,
            'annual': annual, 'sharpe': sharpe, 'max_dd': dd,
            'win_rate': win_rate, 'num_trades': len(sells), 'avg_pnl': avg_pnl,
        }
        results.append(r)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = r
            log(f"  [{count}/{total}] ★ 新最佳 夏普{sharpe:.2f} "
                f"年化{annual:.2f}% 回撤{dd:.2f}% 胜率{win_rate:.0f}% "
                f"| 止盈{tp*100:.0f}% 持有{bh}/{max_h}天 追击{tt*100:.0f}% 仓位{pp*100:.0f}%")

    # 输出Top5
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    log(f"\n  === Top 5 参数组合 ===")
    for i, r in enumerate(results[:5]):
        log(f"  #{i+1}: 夏普{r['sharpe']:.2f} 年化{r['annual']:.2f}% "
            f"回撤{r['max_dd']:.2f}% 胜率{r['win_rate']:.0f}% {r['num_trades']}笔 "
            f"| 止盈{r['take_profit']*100:.0f}% 持有{r['base_hold']}/{r['max_hold']}天 "
            f"追击{r['trailing_trigger']*100:.0f}% 仓位{r['position_pct']*100:.0f}%")

    return best_params, results


# ========== 8. 样本外验证 ==========

def out_of_sample_test(stocks, params, oos_start='2025-07-01', oos_end='2026-03-29',
                       initial_cash=1000000):
    """样本外独立验证"""
    log("\n" + "=" * 60)
    log(f"样本外验证 ({oos_start} ~ {oos_end})")
    log("=" * 60)

    if params is None:
        log("  无最佳参数，跳过样本外测试")
        return {}

    log(f"  参数: 止盈{params['take_profit']*100:.0f}% "
        f"持有{params['base_hold']}/{params['max_hold']}天 "
        f"追击{params['trailing_trigger']*100:.0f}% "
        f"仓位{params['position_pct']*100:.0f}%")

    trades, equity, dates = run_backtest(
        stocks,
        start_date=oos_start,
        end_date=oos_end,
        initial_cash=initial_cash,
        take_profit=params['take_profit'],
        trailing_trigger=params['trailing_trigger'],
        base_hold=params['base_hold'],
        max_hold=params['max_hold'],
        position_pct=params['position_pct'],
        min_score=90
    )

    stats = analyze_results(trades, equity, dates, initial_cash, label='样本外')

    # 样本外vs样本内对比
    if stats:
        log("\n  样本外 vs 样本内:")
        log(f"    样本内: 夏普{params['sharpe']:.2f} 年化{params['annual']:.2f}%")
        log(f"    样本外: 夏普{stats.get('sharpe', 0):.2f} 年化{stats.get('annual_return', 0):.2f}%")

        is_decay = stats.get('sharpe', 0) / max(params['sharpe'], 0.01)
        if is_decay > 0.7:
            log(f"    ✅ 样本外表现良好 (衰减{(1-is_decay)*100:.0f}%)")
        elif is_decay > 0.4:
            log(f"    ⚠ 样本外有一定衰减 ({(1-is_decay)*100:.0f}%)")
        else:
            log(f"    ❌ 样本外严重衰减 ({(1-is_decay)*100:.0f}%)")

    return stats


# ========== 9. 完整流水线 ==========

def run_full_pipeline(data_dir='E:/data', max_stocks=100):
    """完整回测流水线"""
    log("=" * 60)
    log("多因子Alpha量化交易系统 v16.0")
    log("=" * 60)
    log("改进: 仓位管理 + 高阈值追击 + 智能到期 + 亏损不加仓")

    # Step 1: 加载数据
    log("\n[1] 加载股票数据...")
    stocks = load_all_stocks(data_dir, min_rows=250, max_stocks=max_stocks)
    log(f"    加载 {len(stocks)} 只股票")

    if not stocks:
        log("    ❌ 无数据，检查数据目录")
        return

    # Step 2: 计算因子
    log("\n[2] 预计算因子...")
    prepare_factors(stocks)

    # Step 3: 默认参数回测 (基线)
    log("\n[3] 默认参数回测 (止盈10%, 持有5/8天, 追击8%, 仓位15%)...")
    trades, equity, dates = run_backtest(
        stocks,
        start_date='2024-01-01',
        end_date='2026-03-29',
        initial_cash=1000000,
        take_profit=0.10,
        trailing_trigger=0.08,
        base_hold=5,
        max_hold=8,
        position_pct=0.15,
        min_score=90
    )
    baseline_stats = analyze_results(trades, equity, dates, 1000000, label='v16默认')

    # Step 4: 参数扫描 (样本内: 2024-01 ~ 2025-06)
    log("\n[4] 参数扫描 (样本内: 2024-01 ~ 2025-06)...")
    best_params, all_results = parameter_sweep(
        stocks,
        date_range=('2024-01-01', '2025-06-30'),
        initial_cash=1000000
    )

    # Step 5: 样本外验证 (2025-07 ~ 2026-03)
    log("\n[5] 样本外验证 (2025-07 ~ 2026-03)...")
    oos_stats = out_of_sample_test(
        stocks, best_params,
        oos_start='2025-07-01',
        oos_end='2026-03-29',
        initial_cash=1000000
    )

    # Step 6: 最佳参数全量回测
    if best_params:
        log("\n[6] 最佳参数完整回测 (2024-01 ~ 2026-03)...")
        trades, equity, dates = run_backtest(
            stocks,
            start_date='2024-01-01',
            end_date='2026-03-29',
            initial_cash=1000000,
            take_profit=best_params['take_profit'],
            trailing_trigger=best_params['trailing_trigger'],
            base_hold=best_params['base_hold'],
            max_hold=best_params['max_hold'],
            position_pct=best_params['position_pct'],
            min_score=90
        )
        full_stats = analyze_results(trades, equity, dates, 1000000, label='最佳参数全量')

    # 总结
    log("\n" + "=" * 60)
    log("v16 策略总结")
    log("=" * 60)
    if best_params:
        log(f"  最佳参数:")
        log(f"    止盈: {best_params['take_profit']*100:.0f}%")
        log(f"    追击止盈触发: {best_params['trailing_trigger']*100:.0f}%")
        log(f"    基础持有: {best_params['base_hold']}天")
        log(f"    最大持有: {best_params['max_hold']}天")
        log(f"    仓位占比: {best_params['position_pct']*100:.0f}%")
        log(f"  样本内夏普: {best_params['sharpe']:.2f}")
        if oos_stats:
            log(f"  样本外夏普: {oos_stats.get('sharpe', 0):.2f}")
            log(f"  样本外年化: {oos_stats.get('annual_return', 0):.2f}%")
            log(f"  样本外回撤: {oos_stats.get('max_dd', 0):.2f}%")

    log("\nv16 vs v15 改进点:")
    log("  1. ✅ 仓位管理: 10-20%资金 (vs 固定100股)")
    log("  2. ✅ 追击止盈: 高阈值8-10% (vs 3%太紧)")
    log("  3. ✅ 智能到期: 盈利延长至8天 (vs 5天强制卖出)")
    log("  4. ✅ 亏损不加仓保护")
    log("  5. ✅ 参数扫描: 48组参数网格")
    log("  6. ✅ 样本外验证: 2025-07后独立测试")

    log("\n完成!")
    return best_params, oos_stats


if __name__ == '__main__':
    run_full_pipeline(data_dir='E:/data', max_stocks=100)
