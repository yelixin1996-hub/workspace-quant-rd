# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v17.0 (双持仓高仓位版)
===============================================
基于v11数据和经验，v17核心改进:
1. 去掉固定止损 — v11已验证止损是亏损根源
2. 仓位提升到20-25% — 每只股票配置总资产的20-25%
3. 新增连涨天数因子 — 捕捉趋势延续信号
4. 同时持有2只股 — 分散风险，提高资金利用率

目标: 年化30%+

策略逻辑:
- 选股: 多因子打分(动量+均线+连涨天数+技术指标)
- 买入: 同时持有最多2只，每只配置总资产20-25%
- 卖出: 止盈/到期/RSI超买（无固定止损）
- 风控: 市场熊市时空仓，连亏熔断
"""

import pandas as pd
import numpy as np
import os


def log(msg):
    print(msg, flush=True)


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


def add_factors(df):
    """计算所有因子，包括v17新增的连涨天数因子"""
    df = df.copy()

    # 收益率
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)
    df['ret_60'] = df['close'].pct_change(60)

    # 均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    # 均线多头排列
    df['ma_bull'] = ((df['ma5'] > df['ma10']) &
                      (df['ma10'] > df['ma20']) &
                      (df['close'] > df['ma20'])).astype(int)

    # BBI
    df['bbi'] = (df['ma5'] + df['ma10'] + df['ma20'] + df['ma60']) / 4
    df['bbi_ratio'] = df['close'] / df['bbi']

    # 波动率和量比
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()

    # 价格位置
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

    # 均线斜率
    df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(10)) / (df['ma20'].shift(10) + 1e-8)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # KDJ
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

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    # 创20日新高
    df['new_high_20'] = (df['close'] >= df['high'].rolling(20).max()).astype(int)

    # === v17新增: 连涨天数因子 ===
    up_days = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            up_days.iloc[i] = up_days.iloc[i-1] + 1
        else:
            up_days.iloc[i] = 0
    df['consecutive_up'] = up_days

    # 连涨伴随放量（连涨且量比>1说明有资金持续流入）
    df['up_with_vol'] = ((df['consecutive_up'] >= 2) &
                          (df['vol_ratio'] > 1.0)).astype(int)

    return df


def prepare_factors(stocks):
    log("  预计算因子...")
    for code, df in stocks.items():
        stocks[code] = add_factors(df)
    log(f"  完成 {len(stocks)} 只股票的因子计算")


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


class Selector:
    """v17选股器：新增连涨天数因子加分"""

    def __init__(self, stocks):
        self.stocks = stocks

    def select_top(self, date, market_env='neutral', min_score=85,
                   top_n=3, exclude_codes=None):
        """选股打分，排除已持仓的股票"""
        candidates = []
        exclude_codes = exclude_codes or []

        for code, df in self.stocks.items():
            if code in exclude_codes:
                continue

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

            # 1. 强动量 (权重最高)
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

            # 2. 均线多头
            if row.get('ma_bull', 0) == 1:
                score += 25

            # 3. BBI上方
            if row.get('bbi_ratio', 1) > 1:
                score += 15 * (row['bbi_ratio'] - 1)

            # 4. 趋势向上
            slope = row.get('ma20_slope', 0)
            if slope > 0.02:
                score += 15
            elif slope < 0:
                score += 10 * slope

            # 5. 创20日新高
            if row.get('new_high_20', 0) == 1:
                score += 15

            # 6. RSI健康区间
            rsi = row.get('rsi_14', 50)
            if 45 < rsi < 70:
                score += 10
            elif rsi < 40:
                score += 5

            # 7. 量比适中
            vol_r = row.get('vol_ratio', 1)
            if 1.2 < vol_r < 3:
                score += 8

            # 8. 低波动
            vol = row.get('vol_20', 0.05)
            if vol < 0.02:
                score += 10
            elif vol < 0.03:
                score += 5

            # 9. KDJ金叉
            if row.get('K', 50) > row.get('D', 50) and row.get('K', 50) < 70:
                score += 8

            # 10. MACD多头
            if row.get('MACD', 0) > 0:
                score += 5

            # === v17新增: 连涨天数因子 ===
            consec_up = row.get('consecutive_up', 0)
            if consec_up >= 5:
                score += 20
            elif consec_up >= 3:
                score += 15
            elif consec_up >= 2:
                score += 8

            # 连涨+放量 额外加分
            if row.get('up_with_vol', 0) == 1:
                score += 10

            # 负面清单
            if mom < -0.15:
                score -= 30
            if rsi > 80:
                score -= 20

            # 环境适应
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


def calc_position_shares(price, total_equity, position_pct=0.22):
    """计算仓位对应的股数（A股100股整数倍）"""
    target_value = total_equity * position_pct
    shares = int(target_value / price / 100) * 100
    return max(shares, 100)


def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-20',
                 initial_cash=1000000, take_profit=0.25,
                 max_hold=20, min_score=85, position_pct=0.22,
                 max_positions=2):
    """
    v17回测引擎:
    - 无固定止损
    - 同时持有最多max_positions只股票
    - 每只股票配置总资产的position_pct (20-25%)
    """

    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates
                    if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])

    log(f"  回测期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    log(f"  交易日: {len(dates)}, 股票池: {len(stocks)}")
    log(f"  仓位: {position_pct*100:.0f}%, 最大持仓: {max_positions}只")

    selector = Selector(stocks)

    cash = initial_cash
    positions = []  # [(code, entry_price, entry_date, shares), ...]
    trades = []
    equity_curve = [initial_cash]

    consecutive_losses = 0
    fuse_active = False
    fuse_end_date = None

    for i, date in enumerate(dates):
        if i < 65:
            continue

        # 熔断检查
        if fuse_active and date >= fuse_end_date:
            fuse_active = False
            consecutive_losses = 0

        market_env = 'neutral'
        if i % 5 == 0:
            market_env, bull_ratio = get_market_env(stocks, date)

        # === 卖出逻辑: 逐个检查持仓 ===
        positions_to_remove = []
        for idx, (code, entry_price, entry_date, shares) in enumerate(positions):
            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if today_row.empty:
                continue
            price = today_row['close'].iloc[0]
            today_rsi = today_row['rsi_14'].iloc[0]

            hold_days = (date - entry_date).days
            pnl = (price - entry_price) / entry_price

            sell_reason = None

            # 止盈
            if pnl >= take_profit:
                sell_reason = '止盈'

            # 到期
            elif hold_days >= max_hold:
                sell_reason = '到期'

            # RSI超买辅助（有盈利才卖）
            elif today_rsi > 88 and pnl > 0.05:
                sell_reason = 'RSI超买'

            if sell_reason:
                revenue = price * shares * 0.9997
                cash += revenue
                pnl_pct = pnl * 100
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl_pct,
                    'reason': sell_reason, 'hold_days': hold_days
                })
                positions_to_remove.append(idx)

                if pnl < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= 4:
                        fuse_active = True
                        fuse_end_date = date + pd.Timedelta(days=7)
                        log(f"  熔断: 连亏{consecutive_losses}次, 暂停至{fuse_end_date.strftime('%Y-%m-%d')}")
                else:
                    consecutive_losses = 0

        for idx in sorted(positions_to_remove, reverse=True):
            positions.pop(idx)

        # === 买入逻辑: 补仓到max_positions ===
        if len(positions) < max_positions and not fuse_active and market_env != 'bear':
            held_codes = [p[0] for p in positions]

            # 计算当前总权益
            total_equity = cash
            for code, ep, ed, sh in positions:
                df = stocks.get(code)
                tr = df[df['date'] == date]
                if not tr.empty:
                    total_equity += tr['close'].iloc[0] * sh * 0.9997

            slots = max_positions - len(positions)
            top_stocks = selector.select_top(
                date, market_env=market_env, min_score=min_score,
                top_n=slots + 2, exclude_codes=held_codes
            )

            for code, score, price in top_stocks:
                if len(positions) >= max_positions:
                    break

                shares = calc_position_shares(price, total_equity, position_pct)
                cost = price * shares * 1.0003

                if cost <= cash and shares >= 100:
                    cash -= cost
                    positions.append((code, price, date, shares))
                    trades.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'price': price, 'shares': shares, 'score': score,
                        'market_env': market_env,
                        'position_pct': round(cost / total_equity * 100, 1)
                    })

        # 计算当日权益
        current_value = cash
        for code, ep, ed, sh in positions:
            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if not today_row.empty:
                current_value += today_row['close'].iloc[0] * sh * 0.9997
        equity_curve.append(current_value)

    return trades, equity_curve, dates


def analyze_results(trades, equity_curve, dates, initial_cash):
    if not trades:
        log("  无交易记录")
        return

    sells = [t for t in trades if t['action'] == 'SELL']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    losses = [t for t in sells if t.get('pnl', 0) <= 0]

    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    start = dates[65] if len(dates) > 65 else dates[0]
    n_years = (dates[-1] - start).days / 365
    annual_return = ((final_value / initial_cash) ** (1 / max(n_years, 0.1)) - 1) * 100

    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    max_dd = drawdown.min()

    log("\n" + "=" * 60)
    log("回测结果 (v17 双持仓高仓位版)")
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

    if sells:
        pnls = [t['pnl'] for t in sells]
        log(f"  平均收益: {np.mean(pnls):.2f}%")
        log(f"  最大单笔盈利: {max(pnls):.2f}%")
        log(f"  最大单笔亏损: {min(pnls):.2f}%")

        big_wins = [p for p in pnls if p > 15]
        log(f"  大赚(>15%): {len(big_wins)}笔")
        big_loss = [p for p in pnls if p < -15]
        log(f"  大亏(<-15%): {len(big_loss)}笔")

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
        for days, pnls in hold_groups.items():
            if pnls:
                avg = np.mean(pnls)
                win_r = len([p for p in pnls if p > 0]) / len(pnls) * 100
                log(f"    {days}天: {len(pnls)}笔, 平均{avg:.2f}%, 胜率{win_r:.0f}%")

    # 卖出原因分析
    if sells:
        reason_groups = {}
        for t in sells:
            r = t.get('reason', 'other')
            if r not in reason_groups:
                reason_groups[r] = []
            reason_groups[r].append(t['pnl'])

        log("\n  卖出原因分析:")
        for reason, pnls in reason_groups.items():
            avg = np.mean(pnls)
            win_r = len([p for p in pnls if p > 0]) / len(pnls) * 100
            log(f"    {reason}: {len(pnls)}笔, 平均{avg:.2f}%, 胜率{win_r:.0f}%")

    # 仓位使用统计
    buys = [t for t in trades if t['action'] == 'BUY']
    if buys:
        pcts = [t.get('position_pct', 0) for t in buys if t.get('position_pct')]
        if pcts:
            log(f"\n  仓位使用: 平均{np.mean(pcts):.1f}%, 最大{max(pcts):.1f}%, 最小{min(pcts):.1f}%")

    log("\n最近20笔交易:")
    for t in trades[-20:]:
        if t['action'] == 'BUY':
            pct_str = f" 仓位:{t.get('position_pct', 0):.0f}%" if t.get('position_pct') else ""
            log(f"  {t['date'].strftime('%Y-%m-%d')} BUY  {t['code']} @{t['price']:.2f} "
                f"x{t['shares']} [score:{t.get('score',0):.0f}]{pct_str}")
        else:
            log(f"  {t['date'].strftime('%Y-%m-%d')} SELL {t['code']} @{t['price']:.2f} "
                f"x{t['shares']}  pnl:{t.get('pnl',0):+.2f}% [{t.get('reason','')}]")


def quick_scan(stocks, date_range=None):
    """参数扫描: 搜索最佳参数组合"""
    if date_range is None:
        date_range = ['2024-01-01', '2025-06-01']

    log("\n" + "=" * 60)
    log("v17 参数扫描")
    log("=" * 60)

    best_annual = -999
    best_params = None

    params = [
        # (take_profit, max_hold, min_score, position_pct)
        (0.20, 15, 85, 0.20),
        (0.25, 15, 85, 0.22),
        (0.25, 20, 85, 0.22),
        (0.30, 20, 85, 0.22),
        (0.25, 20, 85, 0.25),
        (0.30, 20, 85, 0.25),
        (0.25, 20, 90, 0.22),
        (0.30, 20, 90, 0.22),
        (0.30, 25, 85, 0.22),
        (0.35, 25, 85, 0.22),
        (0.30, 25, 85, 0.25),
        (0.35, 25, 85, 0.25),
        (0.30, 25, 90, 0.25),
        (0.35, 20, 85, 0.22),
    ]

    for take_profit, max_hold, min_score, position_pct in params:
        trades, equity, dt = run_backtest(
            stocks,
            start_date=date_range[0],
            end_date=date_range[1],
            take_profit=take_profit,
            max_hold=max_hold,
            min_score=min_score,
            position_pct=position_pct,
            max_positions=2
        )

        if len(equity) < 10:
            continue

        final = equity[-1]
        n_years = (dt[-1] - dt[65]).days / 365 if len(dt) > 65 else 0.5
        annual = ((final / 1000000) ** (1 / max(n_years, 0.1)) - 1) * 100

        sells = [t for t in trades if t['action'] == 'SELL']
        win_rate = len([t for t in sells if t.get('pnl', 0) > 0]) / max(1, len(sells)) * 100

        if annual > best_annual:
            best_annual = annual
            best_params = {
                'take_profit': take_profit,
                'max_hold': max_hold,
                'min_score': min_score,
                'position_pct': position_pct
            }
            log(f"*** 新最佳: 年化{annual:.2f}% 胜率{win_rate:.0f}% "
                f"止盈{take_profit} 持有{max_hold} 仓位{position_pct}")

    log(f"\n最佳参数: {best_params} 年化{best_annual:.2f}%")
    return best_params


if __name__ == '__main__':
    log("=" * 60)
    log("多因子Alpha量化交易系统 v17.0 (双持仓高仓位版)")
    log("=" * 60)
    log("核心改进: 无止损 | 仓位20-25% | 连涨天数因子 | 同时持2只")

    log("\n[1] 加载股票数据...")
    stocks = load_all_stocks('E:/data', min_rows=250, max_stocks=100)
    log(f"    加载 {len(stocks)} 只股票")

    log("\n[2] 预计算因子 (含连涨天数)...")
    prepare_factors(stocks)

    log("\n[3] 默认参数回测 (止盈25%, 持有20天, 仓位22%, 持2只)...")
    trades, equity, dates = run_backtest(
        stocks,
        start_date='2024-01-01',
        end_date='2026-03-20',
        initial_cash=1000000,
        take_profit=0.25,
        max_hold=20,
        min_score=85,
        position_pct=0.22,
        max_positions=2
    )
    analyze_results(trades, equity, dates, 1000000)

    log("\n[4] 参数扫描...")
    best_params = quick_scan(stocks, ['2024-01-01', '2025-06-01'])

    if best_params:
        log("\n[5] 最佳参数完整回测...")
        trades, equity, dates = run_backtest(
            stocks,
            start_date='2024-01-01',
            end_date='2026-03-20',
            initial_cash=1000000,
            max_positions=2,
            **best_params
        )
        analyze_results(trades, equity, dates, 1000000)

    log("\n完成!")
