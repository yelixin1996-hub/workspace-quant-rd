# -*- coding: utf-8 -*-
"""
简单动量策略 v1.0
================
核心思想: 追涨杀跌
- 买入20日涨幅最高且在20日均线上方的股票
- 持有5天后检查,跌破BBI则卖出
- 止损3%
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
    df['ret_20'] = df['close'].pct_change(20)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma5'] = df['close'].rolling(5).mean()
    df['bbi'] = (df['ma5'] + df['ma20'] + df['close'].rolling(10).mean() + df['close'].rolling(30).mean()) / 4
    df['above_ma20'] = (df['close'] > df['ma20']).astype(int)
    return df


def backtest(stocks, start='2024-01-01', end='2026-03-20', init=1000000, stop=0.03, min_win=5):
    """简单动量回测"""
    
    # 获取交易日
    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates if pd.to_datetime(start) <= d <= pd.to_datetime(end)])
    
    log(f"回测: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}, {len(dates)}天")
    
    cash = init
    pos = None  # (code, entry_price, entry_date, shares, high)
    trades = []
    equity = [init]
    
    for i, date in enumerate(dates):
        if i < 65:
            continue
        
        # 选股: 20日涨幅最高且站在20日均线上
        if pos is None:
            candidates = []
            for code, df in stocks.items():
                hist = df[df['date'] <= date]
                if len(hist) < 60:
                    continue
                row = hist.iloc[-1]
                if row['close'] <= 0 or pd.isna(row.get('ret_20')):
                    continue
                if row['above_ma20'] == 1 and row['ret_20'] > 0:
                    candidates.append((code, row['ret_20'], row['close']))
            
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                code, ret20, price = candidates[0]
                if price * 100 * 1.0003 <= cash:
                    cash -= price * 100 * 1.0003
                    pos = (code, price, date, 100, price)
                    trades.append({'d': date, 'c': code, 'a': 'BUY', 'p': price, 'ret20': ret20})
        
        # 持仓检查
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
            
            # 持有5天后,跌破BBI卖出
            elif hold >= min_win:
                if price < today['bbi'].iloc[0] * 0.98:
                    cash += price * shares * 0.9997
                    trades.append({'d': date, 'c': code, 'a': 'SELL', 'p': price, 'pnl': pnl*100, 'r': '破BBI', 'h': hold})
                    pos = None
                
                # 或者超过20天强制卖出
                elif hold >= 20:
                    cash += price * shares * 0.9997
                    trades.append({'d': date, 'c': code, 'a': 'SELL', 'p': price, 'pnl': pnl*100, 'r': '到期', 'h': hold})
                    pos = None
        
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
        
        # 持有分布
        dist = {'1-5': [], '6-10': [], '11-20': [], '21+': []}
        for t in sells:
            h = t.get('h', 0)
            if h <= 5: dist['1-5'].append(t['pnl'])
            elif h <= 10: dist['6-10'].append(t['pnl'])
            elif h <= 20: dist['11-20'].append(t['pnl'])
            else: dist['21+'].append(t['pnl'])
        
        log("\n  持有分布:")
        for k, v in dist.items():
            if v:
                log(f"    {k}天: {len(v)}笔, 均值{np.mean(v):.2f}%, 胜率{len([p for p in v if p>0])/len(v)*100:.0f}%")
        
        # 止损 vs 非止损
        stopped = [t for t in sells if t.get('r') == '止损']
        not_stopped = [t for t in sells if t.get('r') != '止损']
        log(f"\n  止损触发: {len(stopped)}笔, 均值{np.mean([t['pnl'] for t in stopped]):.2f}%")
        if not_stopped:
            log(f"  非止损: {len(not_stopped)}笔, 均值{np.mean([t['pnl'] for t in not_stopped]):.2f}%")
    
    log("\n最近10笔:")
    for t in trades[-10:]:
        d = t['d'].strftime('%Y-%m-%d')
        if t['a'] == 'BUY':
            log(f"  {d} BUY  {t['c']} @{t['p']:.2f} (20日涨幅{t.get('ret20',0)*100:.1f}%)")
        else:
            log(f"  {d} SELL {t['c']} @{t['p']:.2f} {t.get('pnl',0):+.2f}% [{t.get('r','')}]")


# 主程序
if __name__ == '__main__':
    log("=" * 50)
    log("简单动量策略 v1.0")
    log("=" * 50)
    
    log("\n[1] 加载数据...")
    stocks = load_stocks('E:/data', max_stocks=300)
    log(f"    {len(stocks)} 只股票")
    
    log("\n[2] 计算指标...")
    for code, df in stocks.items():
        stocks[code] = add_indicators(df)
    log("    完成")
    
    log("\n[3] 回测...")
    for stop in [0.03, 0.05, 0.08]:
        for min_win in [3, 5, 8]:
            trades, equity, dates = backtest(stocks, stop=stop, min_win=min_win)
            if len(equity) > 10:
                sells = [t for t in trades if t['a'] == 'SELL']
                final = equity[-1]
                ret = (final - 1000000) / 1000000 * 100
                wr = len([t for t in sells if t.get('pnl', 0) > 0]) / max(1, len(sells)) * 100
                log(f"  止损{stop} 最低{min_win}天: 收益{ret:.2f}% 胜率{wr:.0f}% ({len(sells)}笔)")
    
    log("\n完成!")
