# -*- coding: utf-8 -*-
"""
均值回归策略 v1.0
================
核心思想: 买超跌, 卖超涨
- 买入20日跌幅最大的股票 (超跌反弹)
- 价格需要在20日均线下方(确认超跌)
- 持有5-10天,不止损(均值回归需要时间)
- 涨5%止盈或跌3%止损
"""

import pandas as pd
import numpy as np
import os

def log(msg):
    print(msg, flush=True)

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


def add_indicators(df):
    df = df.copy()
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma5'] = df['close'].rolling(5).mean()
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    return df


def backtest(stocks, start='2024-01-01', end='2026-03-20', init=1000000, 
             stop=0.03, profit=0.05, min_hold=5, max_hold=10):
    """均值回归回测"""
    
    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates if pd.to_datetime(start) <= d <= pd.to_datetime(end)])
    
    log(f"回测: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}, {len(dates)}天")
    
    cash = init
    pos = None  # (code, entry_price, entry_date, shares)
    trades = []
    equity = [init]
    
    for i, date in enumerate(dates):
        if i < 65:
            continue
        
        # 选股: 20日跌幅最大且价格在20日均线下方
        if pos is None:
            candidates = []
            for code, df in stocks.items():
                hist = df[df['date'] <= date]
                if len(hist) < 60:
                    continue
                row = hist.iloc[-1]
                if row['close'] <= 0 or pd.isna(row.get('ret_20')):
                    continue
                # 超跌: 20日跌幅>5% 且价格<均线
                if row['ret_20'] < -0.05 and row['close'] < row['ma20']:
                    # 避免过度超跌
                    if row['ret_20'] > -0.40:
                        candidates.append((code, row['ret_20'], row['close'], row['vol_ratio']))
            
            if candidates:
                # 按跌幅排序 (最跌的优先) + 考虑量比
                candidates.sort(key=lambda x: (x[1], x[3]), reverse=False)
                code, ret20, price, vr = candidates[0]
                if price * 100 * 1.0003 <= cash:
                    cash -= price * 100 * 1.0003
                    pos = (code, price, date, 100)
                    trades.append({'d': date, 'c': code, 'a': 'BUY', 'p': price, 'ret20': ret20, 'vr': vr})
        
        # 持仓检查
        if pos:
            code, entry, entry_date, shares = pos
            
            df = stocks.get(code)
            today = df[df['date'] == date]
            if today.empty:
                continue
            
            price = today['close'].iloc[0]
            hold = (date - entry_date).days
            pnl = (price - entry) / entry
            
            # 止损
            if pnl <= -stop:
                cash += price * shares * 0.9997
                trades.append({'d': date, 'c': code, 'a': 'SELL', 'p': price, 'pnl': pnl*100, 'r': '止损', 'h': hold})
                pos = None
            
            # 止盈
            elif pnl >= profit:
                cash += price * shares * 0.9997
                trades.append({'d': date, 'c': code, 'a': 'SELL', 'p': price, 'pnl': pnl*100, 'r': '止盈', 'h': hold})
                pos = None
            
            # 到期强制卖出
            elif hold >= max_hold:
                cash += price * shares * 0.9997
                trades.append({'d': date, 'c': code, 'a': 'SELL', 'p': price, 'pnl': pnl*100, 'r': '到期', 'h': hold})
                pos = None
            
            # 持有至少min_hold天后,如果盈利也可以出
            elif hold >= min_hold and pnl > 0:
                # 5日线下方出
                ma5_today = today['ma5'].iloc[0]
                if price < ma5_today:
                    cash += price * shares * 0.9997
                    trades.append({'d': date, 'c': code, 'a': 'SELL', 'p': price, 'pnl': pnl*100, 'r': '破5日线', 'h': hold})
                    pos = None
        
        # 权益
        if pos:
            code, _, _, shares = pos
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
    
    log(f"\n{'='*50}")
    log("结果")
    log(f"{'='*50}")
    log(f"  初始: {init:,.0f}  最终: {final:,.0f}")
    log(f"  总收益: {ret:.2f}%  年化: {annual:.2f}%")
    log(f"  夏普: {sharpe:.2f}  最大回撤: {max_dd:.2f}%")
    log(f"  交易: {len(sells)}  胜率: {len(wins)/max(1,len(sells))*100:.1f}%")
    
    if sells:
        pnls = [t['pnl'] for t in sells]
        log(f"  均值: {np.mean(pnls):.2f}%  最大: {max(pnls):.2f}%  最小: {min(pnls):.2f}%")
        
        # 按持有天数
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
        
        # 止损 vs 止盈 vs 到期
        by_r = {}
        for t in sells:
            r = t.get('r', '')
            if r not in by_r:
                by_r[r] = []
            by_r[r].append(t['pnl'])
        
        log("\n  出场原因:")
        for r, pnls in by_r.items():
            log(f"    {r}: {len(pnls)}笔, 均值{np.mean(pnls):.2f}%")
    
    log("\n最近10笔:")
    for t in trades[-10:]:
        d = t['d'].strftime('%Y-%m-%d')
        if t['a'] == 'BUY':
            log(f"  {d} BUY  {t['c']} @{t['p']:.2f} (20日{t.get('ret20',0)*100:.1f}% 量比{t.get('vr',1):.1f})")
        else:
            log(f"  {d} SELL {t['c']} @{t['p']:.2f} {t.get('pnl',0):+.2f}% [{t.get('r','')}]")


if __name__ == '__main__':
    log("=" * 50)
    log("均值回归策略 v1.0")
    log("=" * 50)
    
    log("\n[1] 加载数据...")
    stocks = load_stocks('E:/data', max_stocks=300)
    log(f"    {len(stocks)} 只股票")
    
    log("\n[2] 计算指标...")
    for code, df in stocks.items():
        stocks[code] = add_indicators(df)
    log("    完成")
    
    log("\n[3] 回测...")
    for stop in [0.03, 0.05]:
        for profit in [0.05, 0.08, 0.10]:
            for min_hold in [3, 5]:
                for max_hold in [8, 12]:
                    trades, equity, dates = backtest(stocks, stop=stop, profit=profit, min_hold=min_hold, max_hold=max_hold)
                    if len(equity) > 10:
                        sells = [t for t in trades if t['a'] == 'SELL']
                        final = equity[-1]
                        ret = (final - 1000000) / 1000000 * 100
                        wr = len([t for t in sells if t.get('pnl', 0) > 0]) / max(1, len(sells)) * 100
                        if ret > -1:
                            log(f"  止{stop}/盈{profit} 持{min_hold}-{max_hold}: 收益{ret:.2f}% 胜率{wr:.0f}% ({len(sells)}笔)")
    
    log("\n完成!")
