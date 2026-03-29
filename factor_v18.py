# -*- coding: utf-8 -*-
"""
Z哥实战量化系统 v18.0 — 少妇战法 + N型战法
=============================================
核心战法:
  1. 少妇战法: BBI上升 + J值<-1 + DIF>0 (趋势中超跌回踩)
  2. N型战法:  N型结构 + 波动≥6% + 间隔≥5天 (趋势延续形态)

关键改进:
  - 去掉固定止损 (v10实验证明止损是亏损根源)
  - 仓位 20-25% (单只不超过总资金25%, 同时持4-5只)
  - 样本外验证: 训练2024-01~2025-06, 验证2025-07~2026-03
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def log(msg):
    print(msg, flush=True)


# ============================================================
#  1. 数据加载
# ============================================================

def load_all_stocks(data_dir='E:/data', min_rows=200, max_stocks=300):
    stocks = {}
    if not os.path.exists(data_dir):
        log(f"  [WARN] 数据目录不存在: {data_dir}")
        return stocks
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


def generate_test_data(n_stocks=80, n_days=600, seed=42):
    """生成模拟A股数据用于离线测试"""
    np.random.seed(seed)
    stocks = {}
    start = pd.Timestamp('2023-06-01')
    dates = pd.bdate_range(start, periods=n_days, freq='B')

    for i in range(n_stocks):
        code = f'{300000 + i}'
        base = np.random.uniform(10, 80)
        ret = np.random.normal(0.0004, 0.025, n_days)
        # 注入一些趋势段
        regime = np.random.choice([0, 1, 2], n_days, p=[0.4, 0.35, 0.25])
        for j in range(n_days):
            if regime[j] == 1:
                ret[j] += 0.002
            elif regime[j] == 2:
                ret[j] -= 0.001
        close = base * np.cumprod(1 + ret)
        high = close * (1 + np.abs(np.random.normal(0, 0.012, n_days)))
        low = close * (1 - np.abs(np.random.normal(0, 0.012, n_days)))
        opn = low + (high - low) * np.random.uniform(0.3, 0.7, n_days)
        volume = np.random.lognormal(14, 0.8, n_days).astype(int)

        df = pd.DataFrame({
            'date': dates, 'open': opn, 'high': high,
            'low': low, 'close': close, 'volume': volume
        })
        stocks[code] = df
    return stocks


# ============================================================
#  2. 技术指标计算
# ============================================================

def add_factors(df):
    df = df.copy()

    # --- 基础均线 ---
    for w in (3, 5, 6, 10, 12, 20, 24, 30, 60):
        df[f'ma{w}'] = df['close'].rolling(w).mean()

    df['ma_bull'] = (
        (df['ma5'] > df['ma10']) &
        (df['ma10'] > df['ma20']) &
        (df['close'] > df['ma20'])
    ).astype(int)

    # --- BBI (3/6/12/24 经典参数) ---
    df['bbi'] = (df['ma3'] + df['ma6'] + df['ma12'] + df['ma24']) / 4
    df['bbi_up'] = (df['bbi'] > df['bbi'].shift(1)).astype(int)
    df['bbi_ratio'] = df['close'] / df['bbi']

    # --- KDJ (9,3,3) ---
    n_kdj = 9
    low_n = df['low'].rolling(n_kdj).min()
    high_n = df['high'].rolling(n_kdj).max()
    rsv = (df['close'] - low_n) / (high_n - low_n + 1e-8) * 100
    rsv = rsv.fillna(50)

    K = pd.Series(50.0, index=df.index, dtype=float)
    D = pd.Series(50.0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        K.iloc[i] = 2 / 3 * K.iloc[i - 1] + 1 / 3 * rsv.iloc[i]
        D.iloc[i] = 2 / 3 * D.iloc[i - 1] + 1 / 3 * K.iloc[i]
    df['K'] = K
    df['D'] = D
    df['J'] = 3 * K - 2 * D

    # --- MACD (12,26,9) ---
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    # --- RSI(14) ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # --- 波动率 & 量比 ---
    df['vol_20'] = df['close'].pct_change().rolling(20).std()
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()

    # --- 辅助: 20日价格位置 ---
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

    # --- 动量 ---
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)

    # --- N型结构辅助: 局部极值 ---
    df['local_min'] = (
        (df['low'] < df['low'].shift(1)) &
        (df['low'] < df['low'].shift(-1)) &
        (df['low'] < df['low'].shift(2)) &
        (df['low'] < df['low'].shift(-2))
    ).astype(int)

    df['local_max'] = (
        (df['high'] > df['high'].shift(1)) &
        (df['high'] > df['high'].shift(-1)) &
        (df['high'] > df['high'].shift(2)) &
        (df['high'] > df['high'].shift(-2))
    ).astype(int)

    return df


def prepare_factors(stocks):
    log("  预计算因子 (BBI/KDJ/MACD/N型)...")
    for code in list(stocks.keys()):
        stocks[code] = add_factors(stocks[code])
    log(f"  完成 {len(stocks)} 只股票")


# ============================================================
#  3. 信号检测
# ============================================================

def detect_shaofu(row, prev_rows=None):
    """少妇战法: BBI上升 + J值低位 + DIF>0 (趋势中超跌回踩)

    放宽条件版:
      - BBI上升 (bbi_up=1)
      - J < 10 (经典超卖区, 原版 J<-1 太严格)
      - DIF > 0 或 DIF 金叉 (MACD>0 也可)
    """
    bbi_rising = row.get('bbi_up', 0) == 1

    j_val = row.get('J', 50)
    j_oversold = j_val < 10

    dif_val = row.get('DIF', 0)
    dif_positive = dif_val > 0 or row.get('MACD', 0) > 0

    if bbi_rising and j_oversold and dif_positive:
        confidence = 0
        # J值越低 → 超跌越深 → 信心越高 (上限 +40)
        confidence += min(40, max(0, (10 - j_val) * 3))
        # DIF离零轴越远 → 趋势越强
        confidence += min(20, max(0, dif_val / (row.get('close', 1) + 1e-8) * 800))
        # BBI上方加分
        if row.get('bbi_ratio', 1) > 1:
            confidence += 12
        # 量比温和放大
        vr = row.get('vol_ratio', 1)
        if 0.8 < vr < 3.0:
            confidence += 8
        # 均线多头
        if row.get('ma_bull', 0):
            confidence += 15
        # RSI不超买
        rsi = row.get('rsi_14', 50)
        if rsi < 60:
            confidence += 5
        elif rsi < 40:
            confidence += 10  # 超卖区域额外加分

        return True, confidence
    return False, 0


def detect_n_shape(df, idx, min_amplitude=0.06, min_interval=5):
    """N型战法: 寻找N型反转/延续结构

    N型结构:
      A(低点) -> B(反弹高点) -> C(回踩低点, C>A) -> D(当前价接近/突破B)
      要求: (B-A)/A >= min_amplitude, 总间隔 >= min_interval天
    """
    if idx < 30:
        return False, 0

    window = 30
    start_idx = max(0, idx - window)
    segment = df.iloc[start_idx:idx + 1]

    if len(segment) < 12:
        return False, 0

    closes = segment['close'].values
    lows = segment['low'].values
    highs = segment['high'].values
    n = len(closes)

    best_score = 0
    found = False

    for a_off in range(0, max(1, n - min_interval - 3)):
        a_price = lows[a_off]
        b_range_end = min(a_off + 20, n - 2)
        if a_off + 2 > b_range_end:
            continue
        b_off = a_off + 2 + int(np.argmax(highs[a_off + 2:b_range_end + 1]))
        b_price = highs[b_off]

        amplitude = (b_price - a_price) / (a_price + 1e-8)
        if amplitude < min_amplitude:
            continue

        c_range_end = min(b_off + 15, n - 1)
        if b_off + 1 > c_range_end:
            continue
        c_off = b_off + 1 + int(np.argmin(lows[b_off + 1:c_range_end + 1]))
        c_price = lows[c_off]

        if c_price <= a_price:
            continue

        interval = c_off - a_off
        if interval < min_interval:
            continue

        d_price = closes[-1]
        # D 至少回到 B 的 95% (放宽突破要求)
        if d_price < b_price * 0.95:
            continue

        score = 0
        score += min(25, amplitude * 250)
        # C高于A → 底部抬高, 趋势健康
        ca_ratio = (c_price - a_price) / (a_price + 1e-8)
        score += min(15, ca_ratio * 150)
        score += min(10, interval * 1.0)
        if d_price > b_price:
            score += 20  # 真正突破
        elif d_price > b_price * 0.98:
            score += 10  # 接近突破

        # 趋势辅助
        row = segment.iloc[-1]
        if row.get('ma_bull', 0):
            score += 10
        if row.get('DIF', 0) > 0:
            score += 5
        if row.get('bbi_up', 0):
            score += 5

        # 量价配合
        if n >= 5:
            recent_vol = segment['volume'].iloc[-3:].mean()
            past_vol = segment['volume'].iloc[:-3].mean()
            if past_vol > 0 and recent_vol / past_vol > 1.1:
                score += 8

        if score > best_score:
            best_score = score
            found = True

    return found, best_score


# ============================================================
#  4. 市场环境
# ============================================================

def get_market_env(stocks, date):
    bull_count, total = 0, 0
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
    ratio = bull_count / total
    if ratio > 0.55:
        return 'bull', ratio
    elif ratio < 0.40:
        return 'bear', ratio
    return 'neutral', ratio


# ============================================================
#  5. 选股器
# ============================================================

class ZSelector:
    """Z哥战法选股器 — 少妇 + N型"""

    def __init__(self, stocks):
        self.stocks = stocks

    def select(self, date, market_env='neutral', top_n=5):
        candidates = []
        for code, df in self.stocks.items():
            mask = df['date'] <= date
            if mask.sum() < 65:
                continue
            hist = df[mask]
            row = hist.iloc[-1]

            if pd.isna(row.get('bbi', np.nan)) or row['close'] <= 0 or row['volume'] <= 0:
                continue
            vr = row.get('vol_ratio', 1)
            if vr < 0.3 or vr > 8:
                continue

            signals = []
            total_score = 0

            # --- 少妇战法 ---
            sf_hit, sf_score = detect_shaofu(row)
            if sf_hit:
                signals.append('少妇')
                total_score += sf_score

            # --- N型战法 ---
            idx = hist.index[-1]
            nshape_hit, ns_score = detect_n_shape(df, idx)
            if nshape_hit:
                signals.append('N型')
                total_score += ns_score

            if not signals:
                continue

            # 共性加分
            if row.get('ma_bull', 0):
                total_score += 10
            if 0.3 < row.get('price_pos', 0.5) < 0.75:
                total_score += 5
            rsi = row.get('rsi_14', 50)
            if rsi > 80:
                total_score -= 15

            # 熊市过滤
            if market_env == 'bear':
                if 'N型' not in signals:
                    continue
                total_score *= 0.7

            candidates.append({
                'code': code,
                'score': total_score,
                'price': row['close'],
                'signals': signals,
                'J': row.get('J', 0),
                'DIF': row.get('DIF', 0),
                'bbi_ratio': row.get('bbi_ratio', 1),
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_n]


# ============================================================
#  6. 回测引擎 (多仓位, 20-25%, 无固定止损)
# ============================================================

def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-20',
                 initial_cash=1000000, position_pct=0.22, max_positions=4,
                 take_profit=0.30, max_hold=30, rsi_sell=85,
                 bbi_trail=True):
    """
    多仓位回测:
      - position_pct: 单只仓位占比 (20-25%)
      - max_positions: 最大同时持仓数
      - 无固定止损
      - 卖出条件: 止盈 / 到期 / RSI超买 / BBI严重跌穿
    """

    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates
                    if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])

    if len(dates) < 70:
        log("  日期不足, 跳过")
        return [], [], dates

    log(f"  回测期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    log(f"  交易日: {len(dates)}, 股票池: {len(stocks)}")

    selector = ZSelector(stocks)
    cash = initial_cash
    positions = []
    trades = []
    equity_curve = []

    market_env = 'neutral'

    for i, date in enumerate(dates):
        if i < 65:
            equity_curve.append(initial_cash)
            continue

        if i % 5 == 0:
            market_env, _ = get_market_env(stocks, date)

        # ----- 卖出检查 -----
        to_remove = []
        for pi, pos in enumerate(positions):
            df = stocks.get(pos['code'])
            today = df[df['date'] == date]
            if today.empty:
                continue
            price = today['close'].iloc[0]
            pos['high'] = max(pos['high'], price)

            hold_days = (date - pos['entry_date']).days
            pnl = (price - pos['entry_price']) / pos['entry_price']

            sell_reason = None

            # 止盈
            if pnl >= take_profit:
                sell_reason = '止盈'

            # 到期
            elif hold_days >= max_hold:
                sell_reason = '到期'

            # RSI超买 + 有盈利
            elif today['rsi_14'].iloc[0] > rsi_sell and pnl > 0.03:
                sell_reason = 'RSI超买'

            # BBI严重跌穿: 价格持续低于BBI + BBI转跌 + 已亏损
            elif bbi_trail and hold_days > 8:
                bbi_val = today['bbi'].iloc[0]
                bbi_up_val = today['bbi_up'].iloc[0]
                if price < bbi_val * 0.95 and bbi_up_val == 0 and pnl < -0.05:
                    sell_reason = '破BBI'

            # 移动止盈: 从高点回撤超 12% 且已有盈利
            if not sell_reason and pos['high'] > pos['entry_price'] * 1.08:
                retrace = (pos['high'] - price) / pos['high']
                if retrace > 0.12:
                    sell_reason = '移动止盈'

            if sell_reason:
                revenue = price * pos['shares'] * 0.9997
                cash += revenue
                trades.append({
                    'date': date, 'code': pos['code'], 'action': 'SELL',
                    'price': price, 'shares': pos['shares'],
                    'pnl': pnl * 100, 'reason': sell_reason,
                    'hold_days': hold_days, 'signals': pos.get('signals', [])
                })
                to_remove.append(pi)

        for pi in sorted(to_remove, reverse=True):
            positions.pop(pi)

        # ----- 买入 -----
        open_slots = max_positions - len(positions)
        if open_slots > 0 and market_env != 'bear':
            held_codes = {p['code'] for p in positions}
            candidates = selector.select(date, market_env=market_env,
                                         top_n=open_slots + 3)
            for cand in candidates:
                if open_slots <= 0:
                    break
                if cand['code'] in held_codes:
                    continue

                total_equity = cash
                for p in positions:
                    d = stocks.get(p['code'])
                    t = d[d['date'] == date]
                    if not t.empty:
                        total_equity += t['close'].iloc[0] * p['shares']

                alloc = total_equity * position_pct
                alloc = min(alloc, cash * 0.90)
                price = cand['price']
                shares = int(alloc / (price * 1.0003) / 100) * 100
                if shares < 100:
                    continue
                cost = price * shares * 1.0003
                if cost > cash:
                    continue
                cash -= cost
                positions.append({
                    'code': cand['code'],
                    'entry_price': price,
                    'entry_date': date,
                    'shares': shares,
                    'high': price,
                    'signals': cand['signals'],
                })
                trades.append({
                    'date': date, 'code': cand['code'], 'action': 'BUY',
                    'price': price, 'shares': shares,
                    'score': cand['score'],
                    'signals': cand['signals'],
                    'market_env': market_env,
                })
                held_codes.add(cand['code'])
                open_slots -= 1

        # ----- 权益曲线 -----
        pos_value = 0
        for pos in positions:
            df = stocks.get(pos['code'])
            today = df[df['date'] == date]
            if not today.empty:
                pos_value += today['close'].iloc[0] * pos['shares'] * 0.9997
        equity_curve.append(cash + pos_value)

    return trades, equity_curve, dates


# ============================================================
#  7. 结果分析
# ============================================================

def analyze_results(trades, equity_curve, dates, initial_cash, label=''):
    if not trades:
        log("  无交易记录")
        return {}

    sells = [t for t in trades if t['action'] == 'SELL']
    buys = [t for t in trades if t['action'] == 'BUY']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    losses = [t for t in sells if t.get('pnl', 0) <= 0]

    ec = np.array(equity_curve, dtype=float)
    final_value = ec[-1] if len(ec) else initial_cash
    total_return = (final_value - initial_cash) / initial_cash * 100

    start_idx = min(65, len(dates) - 1)
    n_years = max((dates[-1] - dates[start_idx]).days / 365, 0.1)
    annual_return = ((final_value / initial_cash) ** (1 / n_years) - 1) * 100

    returns = np.diff(ec) / ec[:-1] if len(ec) > 1 else np.array([0])
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-12)) * np.sqrt(252)

    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / (peak + 1e-8) * 100
    max_dd = dd.min()

    calmar = annual_return / (-max_dd + 1e-8) if max_dd < 0 else 0

    log(f"\n{'='*60}")
    log(f"回测结果 {label}")
    log(f"{'='*60}")
    log(f"  初始资金: {initial_cash:,.0f}")
    log(f"  最终资金: {final_value:,.0f}")
    log(f"  总收益率: {total_return:.2f}%")
    log(f"  年化收益率: {annual_return:.2f}%")
    log(f"  夏普比率: {sharpe:.2f}")
    log(f"  最大回撤: {max_dd:.2f}%")
    log(f"  Calmar比率: {calmar:.2f}")
    log(f"  买入次数: {len(buys)}, 卖出次数: {len(sells)}")
    if sells:
        log(f"  胜率: {len(wins)/len(sells)*100:.1f}%")
        pnls = [t['pnl'] for t in sells]
        log(f"  平均收益: {np.mean(pnls):.2f}%")
        log(f"  最大单笔盈利: {max(pnls):.2f}%")
        log(f"  最大单笔亏损: {min(pnls):.2f}%")

        # 信号来源分析
        sig_groups = {}
        for t in sells:
            for s in t.get('signals', ['未知']):
                sig_groups.setdefault(s, []).append(t['pnl'])
        log("\n  信号来源分析:")
        for sig, plist in sig_groups.items():
            avg = np.mean(plist)
            wr = len([p for p in plist if p > 0]) / len(plist) * 100
            log(f"    {sig}: {len(plist)}笔, 平均{avg:.2f}%, 胜率{wr:.0f}%")

        # 卖出原因
        reason_groups = {}
        for t in sells:
            r = t.get('reason', 'other')
            reason_groups.setdefault(r, []).append(t['pnl'])
        log("\n  卖出原因分析:")
        for reason, plist in reason_groups.items():
            avg = np.mean(plist)
            wr = len([p for p in plist if p > 0]) / len(plist) * 100
            log(f"    {reason}: {len(plist)}笔, 平均{avg:.2f}%, 胜率{wr:.0f}%")

        # 持有天数分析
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
        for days, plist in hold_groups.items():
            if plist:
                avg = np.mean(plist)
                wr = len([p for p in plist if p > 0]) / len(plist) * 100
                log(f"    {days}天: {len(plist)}笔, 平均{avg:.2f}%, 胜率{wr:.0f}%")

    # 最近15笔交易
    log(f"\n  最近15笔交易:")
    for t in trades[-15:]:
        d = t['date'].strftime('%Y-%m-%d')
        if t['action'] == 'BUY':
            sigs = '+'.join(t.get('signals', []))
            log(f"    {d} BUY  {t['code']} @{t['price']:.2f} [{sigs}] score:{t.get('score',0):.0f}")
        else:
            log(f"    {d} SELL {t['code']} @{t['price']:.2f} pnl:{t.get('pnl',0):+.2f}% [{t.get('reason','')}]")

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'n_trades': len(sells),
        'win_rate': len(wins) / max(1, len(sells)) * 100,
    }


# ============================================================
#  8. 样本外验证
# ============================================================

def out_of_sample_validation(stocks, initial_cash=1000000,
                             train_start='2024-01-01', train_end='2025-06-30',
                             test_start='2025-07-01', test_end='2026-03-20',
                             param_grid=None):
    """
    训练期选最优参数 → 测试期验证 (样本外)
    """

    if param_grid is None:
        param_grid = [
            {'take_profit': 0.25, 'max_hold': 25, 'position_pct': 0.22, 'max_positions': 4},
            {'take_profit': 0.30, 'max_hold': 30, 'position_pct': 0.22, 'max_positions': 4},
            {'take_profit': 0.30, 'max_hold': 25, 'position_pct': 0.20, 'max_positions': 5},
            {'take_profit': 0.25, 'max_hold': 30, 'position_pct': 0.25, 'max_positions': 4},
            {'take_profit': 0.35, 'max_hold': 30, 'position_pct': 0.22, 'max_positions': 4},
            {'take_profit': 0.30, 'max_hold': 35, 'position_pct': 0.20, 'max_positions': 5},
            {'take_profit': 0.25, 'max_hold': 25, 'position_pct': 0.25, 'max_positions': 4},
            {'take_profit': 0.35, 'max_hold': 25, 'position_pct': 0.22, 'max_positions': 5},
        ]

    log("\n" + "=" * 60)
    log("样本外验证")
    log("=" * 60)
    log(f"  训练期: {train_start} ~ {train_end}")
    log(f"  测试期: {test_start} ~ {test_end}")

    # ---- 训练期: 选最优参数 ----
    log("\n[训练期] 参数扫描...")
    best_annual = -999
    best_params = param_grid[0]

    for params in param_grid:
        trades, equity, dates = run_backtest(
            stocks, start_date=train_start, end_date=train_end,
            initial_cash=initial_cash, **params
        )
        if len(equity) < 20:
            continue
        ec = np.array(equity, dtype=float)
        final = ec[-1]
        start_idx = min(65, len(dates) - 1)
        n_years = max((dates[-1] - dates[start_idx]).days / 365, 0.1)
        annual = ((final / initial_cash) ** (1 / n_years) - 1) * 100
        log(f"  止盈{params['take_profit']:.0%} 持有{params['max_hold']}天 "
            f"仓位{params['position_pct']:.0%}x{params['max_positions']} → 年化{annual:.2f}%")
        if annual > best_annual:
            best_annual = annual
            best_params = params

    log(f"\n  [训练期最佳] 年化{best_annual:.2f}%, 参数: {best_params}")

    # ---- 训练期最优结果 ----
    log("\n[训练期] 最佳参数完整回测...")
    tr_trades, tr_equity, tr_dates = run_backtest(
        stocks, start_date=train_start, end_date=train_end,
        initial_cash=initial_cash, **best_params
    )
    tr_stats = analyze_results(tr_trades, tr_equity, tr_dates, initial_cash, label='[训练期]')

    # ---- 测试期 ----
    log("\n[测试期] 样本外验证...")
    te_trades, te_equity, te_dates = run_backtest(
        stocks, start_date=test_start, end_date=test_end,
        initial_cash=initial_cash, **best_params
    )
    te_stats = analyze_results(te_trades, te_equity, te_dates, initial_cash, label='[测试期/样本外]')

    # ---- 对比 ----
    log("\n" + "=" * 60)
    log("训练 vs 测试 对比")
    log("=" * 60)
    for key in ['annual_return', 'sharpe', 'max_dd', 'calmar', 'win_rate', 'n_trades']:
        tr_val = tr_stats.get(key, 0)
        te_val = te_stats.get(key, 0)
        log(f"  {key:20s}: 训练={tr_val:8.2f}  测试={te_val:8.2f}")

    # 过拟合判定
    if te_stats.get('annual_return', 0) > 0 and tr_stats.get('annual_return', 0) > 0:
        decay = 1 - te_stats['annual_return'] / (tr_stats['annual_return'] + 1e-8)
        if decay < 0.3:
            log(f"\n  ✅ 样本外衰减 {decay:.1%} < 30%, 策略鲁棒")
        elif decay < 0.6:
            log(f"\n  ⚠️ 样本外衰减 {decay:.1%}, 可能存在轻微过拟合")
        else:
            log(f"\n  ❌ 样本外衰减 {decay:.1%}, 过拟合风险较高")
    else:
        log("\n  ⚠️ 训练或测试收益为负, 需调整策略")

    return best_params, tr_stats, te_stats


# ============================================================
#  9. 主程序
# ============================================================

if __name__ == '__main__':
    log("=" * 60)
    log("Z哥实战量化系统 v18.0 — 少妇战法 + N型战法")
    log("=" * 60)

    # 加载数据
    log("\n[1] 加载股票数据...")
    data_dir = 'E:/data'
    stocks = load_all_stocks(data_dir, min_rows=250, max_stocks=300)
    if not stocks:
        log("  未找到真实数据, 使用模拟数据")
        stocks = generate_test_data(n_stocks=80, n_days=700, seed=42)
    log(f"    加载 {len(stocks)} 只股票")

    # 预计算因子
    log("\n[2] 预计算因子...")
    prepare_factors(stocks)

    # 全量回测 (默认参数)
    log("\n[3] 全量回测 (默认参数)...")
    trades, equity, dates = run_backtest(
        stocks,
        start_date='2024-01-01', end_date='2026-03-20',
        initial_cash=1000000,
        position_pct=0.22, max_positions=4,
        take_profit=0.30, max_hold=25,
    )
    analyze_results(trades, equity, dates, 1000000, label='[全量]')

    # 样本外验证
    log("\n[4] 样本外验证...")
    best_params, tr_stats, te_stats = out_of_sample_validation(
        stocks,
        initial_cash=1000000,
        train_start='2024-01-01', train_end='2025-06-30',
        test_start='2025-07-01', test_end='2026-03-20',
    )

    # 最佳参数全量回测
    log("\n[5] 最佳参数全量回测...")
    trades, equity, dates = run_backtest(
        stocks,
        start_date='2024-01-01', end_date='2026-03-20',
        initial_cash=1000000, **best_params
    )
    final_stats = analyze_results(trades, equity, dates, 1000000, label='[最佳参数全量]')

    log("\n" + "=" * 60)
    log("完成!")
    log("=" * 60)
