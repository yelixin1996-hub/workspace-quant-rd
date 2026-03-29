# -*- coding: utf-8 -*-
"""
多因子Alpha量化系统 v5.0 (极速版)
================================
优化: 预计算市场广度, 极速回测
"""

import pandas as pd
import numpy as np
import os

def log(msg):
    print(msg, flush=True)

# ========== 1. 数据加载 ==========

def load_stocks(data_dir='E:/data', max_stocks=300):
    stocks = {}
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:max_stocks]
    for f in files:
        code = f.replace('.SZ.csv', '').replace('.SH.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df = df.rename(columns={'trade_date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) >= 250:
            stocks[code] = df
    return stocks


def add_factors(df):
    df = df.copy()
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma_above'] = (df['close'] > df['ma20']).astype(int)
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    return df


# ========== 2. 预计算市场广度 ==========

def precompute_breadth(stocks):
    """预计算每日市场广度"""
    log("  预计算市场广度...")
    
    # 获取所有日期
    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted(all_dates)
    
    breadth_dict = {}
    for i, date in enumerate(dates):
        if i < 25:
            breadth_dict[date] = 0.5  # 默认中立
            continue
        
        count = 0
        total = 0
        for code, df in stocks.items():
            hist = df[df['date'] <= date]
            if len(hist) < 25:
                continue
            total += 1
            if hist.iloc[-1]['close'] > hist.iloc[-1]['ma20']:
                count += 1
        
        breadth_dict[date] = count / max(total, 1)
    
    log("  完成市场广度计算")
    return breadth_dict


# ========== 3. 选股 ==========

def select_stocks(stocks, date, breadth_dict, top_n=3):
    """选股"""
    candidates = []
    market_bull = breadth_dict.get(date, 0.5) > 0.5
    
    if not market_bull:
        return []
    
    for code, df in stocks.items():
        hist = df[df['date'] <= date]
        if len(hist) < 60:
            continue
        
        row = hist.iloc[-1]
        if row['close'] <= 0 or row['volume'] <= 0:
            continue
        
        # 简单打分
        score = 0
        mom = row.get('ret_20', 0)
        if mom > 0.05:
            score += 30 * mom
        if row.get('ma_above', 0) == 1:
            score += 20
        vr = row.get('vol_ratio', 1)
        if 1.2 < vr < 4:
            score += 10
        
        if score > 25:
            candidates.append((code, score, row['close']))
    
    if not candidates:
        return []
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


# ========== 4. 回测 ==========

def backtest(stocks, breadth_dict, start='2024-01-01', end='2026-03-20',
             init=1000000, stop=0.03, min_hold=5, max_hold=15):
    """回测"""
    
    all_dates = sorted(breadth_dict.keys())
    dates = [d for d in all_dates if pd.to_datetime(start) <= d <= pd.to_datetime(end)]
    
    log(f"  回测: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}, {len(dates)}天")
    
    cash = init
    pos = None  # (code, entry_price, entry_date, shares, high_price)
    trades = []
    equity = [init]
    
    for i, date in enumerate(dates):
        if i < 65:
            continue
        
        if pos:
            code, entry, entry_date, shares, high = pos
            
            df = stocks.get(code)
            today = df[df['date'] == date]
            if today.empty:
                continue
            
            price = today['close'].iloc[0]
            high = max(high, price)
            hold = (date - entry_date).days
            pnl = (price - entry) / entry
            
            pos = (code, entry, entry_date, shares, high)
            
            # 止损
            if pnl <= -stop:
                cash += price * shares * 0.9997
                trades.append({'d': date, 'c': code, 'a': 'SELL', 'p': price, 'pnl': pnl*100, 'r': '止损', 'h': hold})
                pos = None
            
            # 移动止盈 (持有5天后, 最高回撤10%)
            elif hold >= min_hold:
                trail = (high - price) / high
                if trail >= 0.10 or (hold >= max_hold and trail >= 0.05):
                    cash += price * shares * 0.9997
                    trades.append({'d': date, 'c': code, 'a': 'SELL', 'p': price, 'pnl': pnl*100, 
                                  'r': '止盈' if trail >= 0.10 else '到期', 'h': hold})
                    pos = None
        
        # 买入
        if pos is None:
            tops = select_stocks(stocks, date, breadth_dict, top_n=3)
            if tops:
                code, score, price = tops[0]
                if price * 100 * 1.0003 <= cash:
                    cash -= price * 100 * 1.0003
                    pos = (code, price, date, 100, price)
                    trades.append({'d': date, 'c': code, 'a': 'BUY', 'p': price})
        
        # 权益
        if pos:
            code, _, _, shares, _ = pos
            df = stocks.get(code)
            today = df[df['date'] == date]
            if not today.empty:
                val = cash + today['close'].iloc[0] * shares * 0.9997
        else:
            val = cash
        equity.append(val)
    
    return trades, equity, dates


def analyze(trades, equity, dates, init):
    sells = [t for t in trades if t['a'] == 'SELL']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    
    final = equity[-1]
    ret = (final - init) / init * 100
    
    n_years = (dates[-1] - dates[65]).days / 365
    annual = ((final / init) ** (1 / max(n_years, 0.1)) - 1) * 100
    
    rets = np.diff(equity) / equity[:-1]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()
    
    log("\n" + "=" * 50)
    log("结果")
    log("=" * 50)
    log(f"  初始: {init:,.0f}  最终: {final:,.0f}")
    log(f"  总收益: {ret:.2f}%  年化: {annual:.2f}%")
    log(f"  夏普: {sharpe:.2f}  最大回撤: {max_dd:.2f}%")
    log(f"  交易: {len(sells)}  胜率: {len(wins)/max(1,len(sells))*100:.1f}%")
    
    if sells:
        pnls = [t['pnl'] for t in sells]
        log(f"  均值: {np.mean(pnls):.2f}%  最大: {max(pnls):.2f}%  最小: {min(pnls):.2f}%")
        
        # 持有分布
        dist = {'1-5': [], '6-10': [], '11+': []}
        for t in sells:
            h = t.get('h', 0)
            if h <= 5: dist['1-5'].append(t['pnl'])
            elif h <= 10: dist['6-10'].append(t['pnl'])
            else: dist['11+'].append(t['pnl'])
        
        log("\n  持有分布:")
        for k, v in dist.items():
            if v:
                log(f"    {k}天: {len(v)}笔, 均值{np.mean(v):.2f}%, 胜率{len([p for p in v if p>0])/len(v)*100:.0f}%")
    
    log("\n最近5笔:")
    for t in trades[-5:]:
        d = t['d'].strftime('%Y-%m-%d')
        if t['a'] == 'BUY':
            log(f"  {d} BUY  {t['c']} @{t['p']:.2f}")
        else:
            log(f"  {d} SELL {t['c']} @{t['p']:.2f} {t.get('pnl',0):+.2f}%")


# ========== 5. 主程序 ==========

if __name__ == '__main__':
    log("=" * 50)
    log("多因子Alpha量化系统 v5.0 (极速版)")
    log("=" * 50)
    
    log("\n[1] 加载数据...")
    stocks = load_stocks('E:/data', max_stocks=300)
    log(f"    {len(stocks)} 只股票")
    
    log("\n[2] 计算因子...")
    for code, df in stocks.items():
        stocks[code] = add_factors(df)
    log("    完成")
    
    log("\n[3] 预计算市场广度...")
    breadth = precompute_breadth(stocks)
    
    # 测试参数
    log("\n[4] 参数测试...")
    for stop in [0.03, 0.05]:
        for min_hold in [5, 8]:
            for max_hold in [12, 20]:
                trades, equity, dates = backtest(stocks, breadth, stop=stop, min_hold=min_hold, max_hold=max_hold)
                if len(equity) > 10:
                    sells = [t for t in trades if t['a'] == 'SELL']
                    final = equity[-1]
                    ret = (final - 1000000) / 1000000 * 100
                    wr = len([t for t in sells if t.get('pnl', 0) > 0]) / max(1, len(sells)) * 100
                    if ret > -5:  # 只显示不太差的结果
                        log(f"  止损{stop} 持有{min_hold}-{max_hold}: 收益{ret:.2f}% 胜率{wr:.0f}%")
    
    log("\n完成!")
