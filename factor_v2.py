# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v2.0 (优化版)
===================================
预计算所有因子，按日期快速查询
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ========== 1. 数据加载 ==========

def load_all_stocks(data_dir='E:/data', min_rows=200, max_stocks=100):
    """加载所有股票"""
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
    """为单只股票添加因子列"""
    df = df.copy()
    
    # 基础指标
    df['returns'] = df['close'].pct_change()
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_20'] = df['close'].pct_change(20)
    
    # BBI
    df['bbi'] = (df['close'].rolling(5).mean() + 
                 df['close'].rolling(10).mean() + 
                 df['close'].rolling(20).mean() + 
                 df['close'].rolling(30).mean()) / 4
    df['bbi_ratio'] = df['close'] / df['bbi']
    
    # 波动率
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # KDJ (简化版, 使用ewm代替循环)
    n = 9
    low_n = df['low'].rolling(n).min()
    high_n = df['high'].rolling(n).max()
    rsv = (df['close'] - low_n) / (high_n - low_n + 1e-8) * 100
    rsv = rsv.fillna(50)
    
    # 使用ewm近似KDJ
    df['K'] = rsv.ewm(alpha=1/3).mean()
    df['D'] = df['K'].ewm(alpha=1/3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    # 量比
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # 均线斜率
    ma10 = df['close'].rolling(10).mean()
    df['ma_slope'] = (ma10 - ma10.shift(5)) / (ma10.shift(5) + 1e-8)
    
    # 价格位置
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['price_pos'] = (df['close'] - ma20) / (std20 + 1e-8)
    
    # 区间振幅
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['range_ratio'] = (df['high_20'] - df['low_20']) / (df['low_20'] + 1e-8)
    
    return df


def prepare_factors(stocks):
    """预计算所有股票的因子"""
    print("  预计算因子...")
    for code, df in stocks.items():
        stocks[code] = add_factors(df)
    print(f"  完成 {len(stocks)} 只股票的因子计算")


# ========== 2. 选股 ==========

class Selector:
    """选股器"""
    
    def __init__(self, stocks):
        self.stocks = stocks
        # 因子权重
        self.weights = {
            'returns_20': 0.20,      # 20日动量
            'volatility_20': -0.05,  # 波动率(负向)
            'bbi_ratio': 0.15,       # BBI多空
            'K': 0.10,               # KDJ_K
            'vol_ratio': 0.10,      # 量比
            'ma_slope': 0.15,        # 均线斜率
            'price_pos': 0.10,      # 价格位置
            'range_ratio': 0.15,   # 区间振幅
        }
    
    def select_top(self, date, top_n=3):
        """在指定日期选择最佳股票"""
        candidates = []
        
        for code, df in self.stocks.items():
            # 获取date之前的数据
            mask = df['date'] <= date
            if mask.sum() < 60:  # 需要足够历史
                continue
            
            row = df[mask].iloc[-1]
            
            # 检查必要因子
            if pd.isna(row.get('returns_20', np.nan)):
                continue
            if pd.isna(row.get('bbi_ratio', np.nan)):
                continue
            if row['close'] <= 0:
                continue
            
            # 跳过ST、退市等
            if row['volume'] <= 0:
                continue
            
            # 计算因子得分 (简化版)
            score = 0
            score += row.get('returns_20', 0) * 50 * 0.20  # 动量
            score += (0.02 - row.get('volatility_20', 0.02)) * 10 * 0.05  # 低波动
            score += (row.get('bbi_ratio', 1) - 1) * 10 * 0.15  # BBI
            score += (row.get('K', 50) - 50) / 50 * 0.10  # KDJ
            score += np.log1p(row.get('vol_ratio', 1)) * 0.10  # 量比
            score += row.get('ma_slope', 0) * 20 * 0.15  # 均线
            score += row.get('price_pos', 0) * 0.10  # 价格位置
            score += row.get('range_ratio', 0) * 0.15  # 振幅
            
            candidates.append((code, score, row['close']))
        
        if not candidates:
            return []
        
        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ========== 3. 回测 ==========

def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-20', 
                 initial_cash=1000000, stop_loss=0.08, take_profit=0.20, max_hold=15):
    """运行回测"""
    
    # 获取所有交易日
    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])
    
    print(f"  回测期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  交易日: {len(dates)}")
    
    selector = Selector(stocks)
    
    cash = initial_cash
    position = None  # (code, entry_price, entry_date, shares)
    trades = []
    equity_curve = [initial_cash]
    
    for i, date in enumerate(dates):
        # 跳过前60天(需要历史数据)
        if i < 60:
            continue
        
        # 每日选股池更新 (每5天选一次)
        if (i % 5 == 0) or (position is None):
            top_stocks = selector.select_top(date, top_n=5)
        
        # 持仓检查
        if position:
            code, entry_price, entry_date, shares = position
            
            # 获取当前价格
            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if today_row.empty:
                continue
            price = today_row['close'].iloc[0]
            
            hold_days = (date - entry_date).days
            pnl = (price - entry_price) / entry_price
            
            # 卖出条件
            sell_reason = None
            if pnl <= -stop_loss:
                sell_reason = '止损'
            elif pnl >= take_profit:
                sell_reason = '止盈'
            elif hold_days >= max_hold:
                sell_reason = '到期'
            
            if sell_reason:
                revenue = price * shares * 0.9997
                cash += revenue
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl * 100,
                    'reason': sell_reason
                })
                position = None
        
        # 买入
        if position is None and top_stocks:
            for code, score, price in top_stocks:
                if price * 100 * 1.0003 <= cash:
                    shares = 100
                    cost = price * shares * 1.0003
                    cash -= cost
                    position = (code, price, date, shares)
                    trades.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'price': price, 'shares': shares
                    })
                    break
        
        # 记录权益
        if position:
            code, entry_price, _, shares = position
            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if not today_row.empty:
                current_value = cash + today_row['close'].iloc[0] * shares * 0.9997
        else:
            current_value = cash
        equity_curve.append(current_value)
    
    return trades, equity_curve, dates


def analyze_results(trades, equity_curve, dates, initial_cash):
    """分析回测结果"""
    
    if not trades:
        print("  无交易记录")
        return
    
    sells = [t for t in trades if t['action'] == 'SELL']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    losses = [t for t in sells if t.get('pnl', 0) <= 0]
    
    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    # 年化
    start = dates[60] if len(dates) > 60 else dates[0]
    n_years = (dates[-1] - start).days / 365
    annual_return = ((final_value / initial_cash) ** (1 / max(n_years, 0.1)) - 1) * 100
    
    # 夏普比率 (简化)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # 最大回撤
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    max_dd = drawdown.min()
    
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"  初始资金: {initial_cash:,.0f}")
    print(f"  最终资金: {final_value:,.0f}")
    print(f"  总收益率: {total_return:.2f}%")
    print(f"  年化收益率: {annual_return:.2f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.2f}%")
    print(f"  交易次数: {len(sells)}")
    print(f"  盈利次数: {len(wins)}")
    print(f"  亏损次数: {len(losses)}")
    print(f"  胜率: {len(wins)/max(1,len(sells))*100:.1f}%")
    
    if sells:
        pnls = [t['pnl'] for t in sells]
        print(f"  平均收益: {np.mean(pnls):.2f}%")
        print(f"  最大单笔盈利: {max(pnls):.2f}%")
        print(f"  最大单笔亏损: {min(pnls):.2f}%")
    
    print("\n最近10笔交易:")
    for t in trades[-10:]:
        if t['action'] == 'BUY':
            print(f"  {t['date'].strftime('%Y-%m-%d')} BUY  {t['code']} @{t['price']:.2f}")
        else:
            print(f"  {t['date'].strftime('%Y-%m-%d')} SELL {t['code']} @{t['price']:.2f}  pnl:{t.get('pnl', 0):+.2f}% [{t.get('reason','')}]")


# ========== 4. 参数优化 ==========

def optimize_params(stocks, date_range=['2024-01-01', '2025-01-01']):
    """参数优化"""
    print("\n" + "=" * 60)
    print("参数优化")
    print("=" * 60)
    
    best_return = -999
    best_params = None
    
    # 网格搜索
    for stop_loss in [0.05, 0.08, 0.10]:
        for take_profit in [0.15, 0.20, 0.25, 0.30]:
            for max_hold in [10, 15, 20]:
                trades, equity, dates = run_backtest(
                    stocks, 
                    start_date=date_range[0],
                    end_date=date_range[1],
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    max_hold=max_hold
                )
                
                if equity:
                    final = equity[-1]
                    ret = (final - 1000000) / 1000000 * 100
                    if ret > best_return:
                        best_return = ret
                        best_params = {'stop_loss': stop_loss, 'take_profit': take_profit, 'max_hold': max_hold}
                        print(f"*** 新最佳: 收益{ret:.2f}% 止损{stop_loss} 止盈{take_profit} 持有{max_hold}")
    
    print(f"\n最佳参数: {best_params}")
    print(f"最佳收益: {best_return:.2f}%")
    return best_params


# ========== 5. 主程序 ==========

if __name__ == '__main__':
    import sys
    
    def log(msg):
        print(msg, flush=True)
    
    log("=" * 60)
    log("多因子Alpha量化交易系统 v2.0")
    log("=" * 60)
    
    # 加载数据
    log("\n[1] 加载股票数据...")
    stocks = load_all_stocks('E:/data', min_rows=250)
    print(f"    加载 {len(stocks)} 只股票")
    
    # 预计算因子
    print("\n[2] 预计算因子...")
    prepare_factors(stocks)
    
    # 回测
    print("\n[3] 运行回测...")
    trades, equity, dates = run_backtest(
        stocks, 
        start_date='2024-01-01',
        end_date='2026-03-20',
        initial_cash=1000000,
        stop_loss=0.08,
        take_profit=0.20,
        max_hold=15
    )
    
    # 分析结果
    analyze_results(trades, equity, dates, 1000000)
    
    # 参数优化 (用较短时期)
    print("\n[4] 参数优化 (2024全年)...")
    best_params = optimize_params(stocks, ['2024-01-01', '2024-12-31'])
    
    # 用最佳参数跑完整回测
    if best_params:
        print("\n[5] 最佳参数完整回测...")
        trades, equity, dates = run_backtest(
            stocks,
            start_date='2024-01-01',
            end_date='2026-03-20',
            initial_cash=1000000,
            **best_params
        )
        analyze_results(trades, equity, dates, 1000000)
    
    print("\n完成!")
