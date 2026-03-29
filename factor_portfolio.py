# -*- coding: utf-8 -*-
"""
多持仓组合策略 v1.0
===================
同时持有3只股票,每月rebalance
- 选20日涨幅前3的股票
- 持有1个月
- 止损10%
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
    return df


def get_top_stocks(stocks, date, n=3):
    """获取指定日期动量最强的n只股票"""
    candidates = []
    for code, df in stocks.items():
        hist = df[df['date'] <= date]
        if len(hist) < 60:
            continue
        row = hist.iloc[-1]
        if row['close'] <= 0 or pd.isna(row.get('ret_20')):
            continue
        # 20日涨幅>0且价格在均线上方
        if row['ret_20'] > 0 and row['close'] > row['ma20']:
            candidates.append((code, row['ret_20'], row['close']))
    
    if not candidates:
        return []
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]


def backtest(stocks, start='2024-01-01', end='2026-03-20', init=1000000,
             n_stocks=3, stop=0.10, hold_days=20):
    """月度rebalance组合回测"""
    
    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates if pd.to_datetime(start) <= d <= pd.to_datetime(end)])
    
    log(f"回测: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}, {len(dates)}天")
    log(f"  持仓: {n_stocks}只, 持有: {hold_days}天, 止损: {stop}")
    
    # 初始持仓
    positions = {}  # {code: (entry_price, entry_date, shares)}
    
    # 第一个仓位
    first_date = dates[65]
    tops = get_top_stocks(stocks, first_date, n=n_stocks)
    per_stock = init / n_stocks
    
    for code, ret20, price in tops:
        shares = int(per_stock / (price * 100)) * 100
        if shares > 0:
            positions[code] = (price, first_date, shares)
    
    trades = []
    equity = [init]
    last_rebalance = first_date
    
    for i, date in enumerate(dates):
        if i < 65:
            continue
        
        # 检查是否需要rebalance
        need_rebalance = (date - last_rebalance).days >= hold_days
        
        # 检查止损
        for code in list(positions.keys()):
            df = stocks.get(code)
            today_data = df[df['date'] == date]
            if today_data.empty:
                continue
            
            price = today_data['close'].iloc[0]
            entry_price, entry_date, shares = positions[code]
            pnl = (price - entry_price) / entry_price
            
            if pnl <= -stop:
                # 止损出局
                proceeds = price * shares * 0.9997
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl * 100,
                    'reason': '止损', 'hold': (date - entry_date).days
                })
                del positions[code]
                need_rebalance = True
        
        # Rebalance
        if need_rebalance and positions:
            # 变现所有持仓
            total_value = 0
            for code in list(positions.keys()):
                df = stocks.get(code)
                today_data = df[df['date'] == date]
                if today_data.empty:
                    continue
                price = today_data['close'].iloc[0]
                entry_price, entry_date, shares = positions[code]
                proceeds = price * shares * 0.9997
                total_value += proceeds
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': (price - entry_price) / entry_price * 100,
                    'reason': 'rebalance', 'hold': (date - entry_date).days
                })
            positions = {}
            
            # 重新买入
            tops = get_top_stocks(stocks, date, n=n_stocks)
            per_stock = total_value / n_stocks if tops else total_value
            
            for code, ret20, price in tops:
                shares = int(per_stock / (price * 100)) * 100
                if shares > 0 and price * shares * 1.0003 <= total_value:
                    positions[code] = (price, date, shares)
                    trades.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'price': price, 'shares': shares, 'ret20': ret20
                    })
            
            last_rebalance = date
        
        # 计算当前权益
        total_val = 0
        for code, (entry_price, entry_date, shares) in positions.items():
            df = stocks.get(code)
            today_data = df[df['date'] == date]
            if not today_data.empty:
                total_val += today_data['close'].iloc[0] * shares * 0.9997
        
        if total_val == 0 and positions:
            total_val = equity[-1]  # 没有价格数据
        elif total_val == 0:
            total_val = equity[-1]
        
        equity.append(total_val)
    
    return trades, equity, dates


def analyze(trades, equity, dates, init):
    sells = [t for t in trades if t['action'] == 'SELL']
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
    
    log("\n最近5笔:")
    for t in trades[-5:]:
        d = t['date'].strftime('%Y-%m-%d')
        if t['action'] == 'BUY':
            log(f"  {d} BUY  {t['code']} @{t['price']:.2f}")
        else:
            log(f"  {d} SELL {t['code']} @{t['price']:.2f} {t.get('pnl',0):+.2f}%")


if __name__ == '__main__':
    log("=" * 50)
    log("多持仓组合策略 v1.0")
    log("=" * 50)
    
    log("\n[1] 加载数据...")
    stocks = load_stocks('E:/data', max_stocks=300)
    log(f"    {len(stocks)} 只股票")
    
    log("\n[2] 计算指标...")
    for code, df in stocks.items():
        stocks[code] = add_indicators(df)
    log("    完成")
    
    log("\n[3] 回测...")
    for n in [2, 3, 5]:
        for hold in [15, 20, 30]:
            for stop in [0.08, 0.10, 0.15]:
                trades, equity, dates = backtest(stocks, n_stocks=n, hold_days=hold, stop=stop)
                if len(equity) > 10:
                    sells = [t for t in trades if t['action'] == 'SELL']
                    final = equity[-1]
                    ret = (final - 1000000) / 1000000 * 100
                    wr = len([t for t in sells if t.get('pnl', 0) > 0]) / max(1, len(sells)) * 100
                    if ret > -5:
                        log(f"  {n}持仓{hold}天止损{stop}: 收益{ret:.2f}% 胜率{wr:.0f}%")
    
    log("\n完成!")
