# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v4.0 (趋势跟随版)
=========================================
核心改进:
1. 延长持有期 - 让盈利股跑起来
2. 放宽止损 - 减少被洗出
3. 趋势确认 - 过滤假信号
4. 减少交易频率 - 避免过度交易
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def log(msg):
    print(msg, flush=True)

# ========== 1. 数据加载 ==========

def load_all_stocks(data_dir='E:/data', min_rows=200, max_stocks=200):
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
    
    # 收益率
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)
    df['ret_60'] = df['close'].pct_change(60)
    
    # 均线系统
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    # 均线多头排列
    df['ma_bull'] = ((df['ma5'] > df['ma10']) & 
                      (df['ma10'] > df['ma20']) & 
                      (df['close'] > df['ma20'])).astype(int)
    
    # 均线空头排列
    df['ma_bear'] = ((df['ma5'] < df['ma10']) & 
                      (df['ma10'] < df['ma20']) & 
                      (df['close'] < df['ma20'])).astype(int)
    
    # BBI
    df['bbi'] = (df['ma5'] + df['ma10'] + df['ma20'] + df['ma60']) / 4
    df['bbi_ratio'] = df['close'] / df['bbi']
    
    # 波动率
    df['vol_20'] = df['ret_1'].rolling(20).std()
    
    # 量比
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # 价格位置
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    
    # 均线斜率
    df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(10)) / (df['ma20'].shift(10) + 1e-8)
    
    # 趋势持续性 (过去5天有多少天上涨)
    df['up_days'] = (df['ret_1'] > 0).rolling(5).sum()
    
    # 创N日新高
    df['new_high_20'] = (df['close'] >= df['high'].rolling(20).max()).astype(int)
    
    return df


def prepare_factors(stocks):
    """预计算所有股票的因子"""
    log("  预计算因子...")
    for code, df in stocks.items():
        stocks[code] = add_factors(df)
    log(f"  完成 {len(stocks)} 只股票的因子计算")


# ========== 2. 市场环境判断 ==========

def get_market_env(stocks, date):
    """判断市场环境"""
    bull_count = 0
    bear_count = 0
    total = 0
    
    for code, df in stocks.items():
        hist = df[df['date'] <= date]
        if len(hist) < 25:
            continue
        total += 1
        row = hist.iloc[-1]
        if row['close'] > row['ma20']:
            bull_count += 1
        elif row['close'] < row['ma20']:
            bear_count += 1
    
    if total == 0:
        return 'neutral', 0.5
    
    bull_ratio = bull_count / total
    if bull_ratio > 0.6:
        return 'bull', bull_ratio
    elif bull_ratio < 0.4:
        return 'bear', bull_ratio
    else:
        return 'neutral', bull_ratio


# ========== 3. 选股 ==========

class Selector:
    """趋势跟随选股器"""
    
    def __init__(self, stocks):
        self.stocks = stocks
    
    def select_top(self, date, market_env='neutral', top_n=3):
        """选择最佳股票"""
        candidates = []
        
        for code, df in self.stocks.items():
            mask = df['date'] <= date
            if mask.sum() < 65:
                continue
            
            row = df[mask].iloc[-1]
            
            # 基础过滤
            if pd.isna(row.get('ret_20', np.nan)):
                continue
            if row['close'] <= 0 or row['volume'] <= 0:
                continue
            if row['vol_ratio'] < 0.5 or row['vol_ratio'] > 10:  # 排除极度缩量或巨量
                continue
            
            score = 0
            reasons = []
            
            # ========== 趋势因子 ==========
            
            # 因子1: 20日动量 (正动量才好)
            mom = row.get('ret_20', 0)
            if mom > 0.05:
                score += 30 * mom
                reasons.append('强动量')
            elif mom > 0.15:
                score += 50 * mom  # 强势动量加分
                reasons.append('超强动量')
            elif mom < 0:
                score += 20 * mom  # 负动量减分
                reasons.append('负动量')
            
            # 因子2: 均线多头排列
            if row.get('ma_bull', 0) == 1:
                score += 25
                reasons.append('均线多头')
            
            # 因子3: 趋势向上 (均线斜率为正)
            slope = row.get('ma20_slope', 0)
            if slope > 0.01:
                score += 15
                reasons.append('趋势向上')
            elif slope < -0.01:
                score -= 15
                reasons.append('趋势向下')
            
            # 因子4: 价格位置适中 (不太高不太低)
            pos = row.get('price_pos', 0.5)
            if 0.2 < pos < 0.8:
                score += 10 * (1 - abs(pos - 0.5) * 2)
                reasons.append('位置适中')
            
            # 因子5: 创20日新高
            if row.get('new_high_20', 0) == 1:
                score += 15
                reasons.append('创20日新高')
            
            # 因子6: 趋势持续性
            up_days = row.get('up_days', 2)
            if up_days >= 4:
                score += 10
                reasons.append('连续上涨')
            elif up_days <= 1:
                score -= 5
            
            # 因子7: BBI上方
            if row.get('bbi_ratio', 1) > 1:
                score += 10 * (row['bbi_ratio'] - 1)
                reasons.append('BBI多头')
            
            # ========== 环境适应 ==========
            if market_env == 'bull':
                # 多头市场: 偏好强势股
                if row.get('ma_bull', 0) == 1:
                    score *= 1.2
            elif market_env == 'bear':
                # 熊市: 只做超跌反弹
                if pos < 0.2 and mom < -0.1:
                    score = 60  # 设定固定分
            
            # ========== 负面清单 ==========
            # 跌幅过大可能是下跌中继
            if mom < -0.25:
                score -= 40
            
            # 涨幅过大不追
            if mom > 0.40:
                score -= 20
            
            # 排除均线空头
            if row.get('ma_bear', 0) == 1:
                score -= 30
            
            if score > 0:
                candidates.append((code, score, row['close'], reasons))
        
        if not candidates:
            return []
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ========== 4. 回测 ==========

def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-20', 
                 initial_cash=1000000, stop_loss=0.08, take_profit=0.20, 
                 max_hold=12, trailing_stop=0.05):
    """运行回测"""
    
    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])
    
    log(f"  回测期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    log(f"  交易日: {len(dates)}")
    
    selector = Selector(stocks)
    
    cash = initial_cash
    position = None
    trades = []
    equity_curve = [initial_cash]
    
    # 追踪最高价(用于跟踪止盈)
    highest_price = 0
    
    for i, date in enumerate(dates):
        if i < 65:
            continue
        
        # 每10天判断一次市场环境
        market_env = 'neutral'
        if i % 10 == 0:
            market_env, _ = get_market_env(stocks, date)
        
        # 持仓检查
        if position:
            code, entry_price, entry_date, shares, entry_breadth = position
            
            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if today_row.empty:
                continue
            price = today_row['close'].iloc[0]
            
            # 更新最高价
            if price > highest_price:
                highest_price = price
            
            hold_days = (date - entry_date).days
            pnl = (price - entry_price) / entry_price
            
            # 跟踪止盈 (从最高点回撤)
            if highest_price > entry_price:
                retrace = (highest_price - price) / highest_price
                if retrace > trailing_stop and pnl > 0:
                    sell_reason = '跟踪止盈'
                else:
                    sell_reason = None
            else:
                sell_reason = None
            
            # 固定止损
            if pnl <= -stop_loss and not sell_reason:
                sell_reason = '止损'
            
            # 止盈
            if pnl >= take_profit and not sell_reason:
                sell_reason = '止盈'
            
            # 到期
            if hold_days >= max_hold and not sell_reason:
                sell_reason = '到期'
            
            # 均线空头排列
            if today_row['ma_bear'].iloc[0] == 1 and not sell_reason:
                sell_reason = '均线死叉'
            
            if sell_reason:
                revenue = price * shares * 0.9997
                cash += revenue
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl * 100,
                    'reason': sell_reason, 'hold_days': hold_days
                })
                position = None
                highest_price = 0
        
        # 买入
        if position is None:
            top_stocks = selector.select_top(date, market_env=market_env, top_n=3)
            if top_stocks:
                code, score, price, reasons = top_stocks[0]
                if price * 100 * 1.0003 <= cash:
                    shares = 100
                    cost = price * shares * 1.0003
                    cash -= cost
                    position = (code, price, date, shares, market_env)
                    highest_price = price
                    trades.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'price': price, 'shares': shares, 'score': score,
                        'market_env': market_env, 'reasons': reasons
                    })
        
        # 记录权益
        if position:
            code, entry_price, _, shares, _ = position
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
    log("回测结果")
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
    
    # 按持有天数分析
    if sells:
        hold_groups = {1: [], 2: [], 3: [], '4-7': [], '8+': []}
        for t in sells:
            h = t.get('hold_days', 0)
            if h <= 1:
                hold_groups[1].append(t['pnl'])
            elif h == 2:
                hold_groups[2].append(t['pnl'])
            elif h == 3:
                hold_groups[3].append(t['pnl'])
            elif h <= 7:
                hold_groups['4-7'].append(t['pnl'])
            else:
                hold_groups['8+'].append(t['pnl'])
        
        log("\n  持有天数分析:")
        for days, pnls in hold_groups.items():
            if pnls:
                avg = np.mean(pnls)
                win_r = len([p for p in pnls if p>0])/len(pnls)*100
                log(f"    {days}天: {len(pnls)}笔, 平均{avg:.2f}%, 胜率{win_r:.0f}%")
    
    # 按卖出原因分析
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
            win_r = len([p for p in pnls if p>0])/len(pnls)*100
            log(f"    {reason}: {len(pnls)}笔, 平均{avg:.2f}%, 胜率{win_r:.0f}%")
    
    log("\n最近10笔交易:")
    for t in trades[-10:]:
        if t['action'] == 'BUY':
            log(f"  {t['date'].strftime('%Y-%m-%d')} BUY  {t['code']} @{t['price']:.2f} [score:{t.get('score',0):.0f}]")
        else:
            log(f"  {t['date'].strftime('%Y-%m-%d')} SELL {t['code']} @{t['price']:.2f}  pnl:{t.get('pnl',0):+.2f}% [{t.get('reason','')}]")


# ========== 5. 快速参数扫描 ==========

def quick_optimize(stocks, date_range=['2024-01-01', '2024-06-01']):
    """快速参数扫描 (用前半年的数据)"""
    log("\n" + "=" * 60)
    log("快速参数扫描")
    log("=" * 60)
    
    best_return = -999
    best_annual = -999
    best_params = None
    
    # 粗网格
    params_grid = [
        (0.05, 0.15, 8),
        (0.06, 0.18, 10),
        (0.07, 0.20, 10),
        (0.08, 0.20, 12),
        (0.08, 0.25, 12),
        (0.10, 0.20, 15),
        (0.10, 0.25, 15),
        (0.10, 0.30, 15),
    ]
    
    for stop_loss, take_profit, max_hold in params_grid:
        trades, equity, dates = run_backtest(
            stocks, 
            start_date=date_range[0],
            end_date=date_range[1],
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold=max_hold
        )
        
        if len(equity) < 10:
            continue
        
        final = equity[-1]
        ret = (final - 1000000) / 1000000 * 100
        
        n_years = (dates[-1] - dates[65]).days / 365 if len(dates) > 65 else 0.5
        annual = ((final / 1000000) ** (1 / max(n_years, 0.1)) - 1) * 100
        
        if annual > best_annual:
            best_annual = annual
            best_return = ret
            best_params = {'stop_loss': stop_loss, 'take_profit': take_profit, 'max_hold': max_hold}
            log(f"*** 新最佳: 年化{annual:.2f}% 收益{ret:.2f}% 止损{stop_loss} 止盈{take_profit} 持有{max_hold}")
    
    log(f"\n最佳参数: {best_params}")
    log(f"最佳年化: {best_annual:.2f}%")
    return best_params


# ========== 6. 主程序 ==========

if __name__ == '__main__':
    log("=" * 60)
    log("多因子Alpha量化交易系统 v4.0 (趋势跟随版)")
    log("=" * 60)
    
    # 加载数据
    log("\n[1] 加载股票数据...")
    stocks = load_all_stocks('E:/data', min_rows=250, max_stocks=200)
    log(f"    加载 {len(stocks)} 只股票")
    
    # 预计算因子
    log("\n[2] 预计算因子...")
    prepare_factors(stocks)
    
    # 默认参数回测
    log("\n[3] 默认参数回测 (止损8%, 止盈20%, 持有12天)...")
    trades, equity, dates = run_backtest(
        stocks, 
        start_date='2024-01-01',
        end_date='2026-03-20',
        initial_cash=1000000,
        stop_loss=0.08,
        take_profit=0.20,
        max_hold=12,
        trailing_stop=0.05
    )
    analyze_results(trades, equity, dates, 1000000)
    
    # 快速参数扫描
    log("\n[4] 快速参数扫描 (2024上半年)...")
    best_params = quick_optimize(stocks, ['2024-01-01', '2024-06-01'])
    
    # 用最佳参数跑完整回测
    if best_params:
        log("\n[5] 最佳参数完整回测...")
        trades, equity, dates = run_backtest(
            stocks,
            start_date='2024-01-01',
            end_date='2026-03-20',
            initial_cash=1000000,
            **best_params
        )
        analyze_results(trades, equity, dates, 1000000)
    
    log("\n完成!")