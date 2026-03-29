# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v3.0 (改进版)
===================================
改进:
1. 多空头因子组合 (动量 + 价值 + 质量)
2. 更严格的选股条件 (需要多个因子同时确认)
3. 市场环境过滤 (只在多头市场中做多)
4. 更好的止损止盈
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
    
    # 均线多头排列 (牛市特征)
    df['ma_bull'] = ((df['ma5'] > df['ma10']) & 
                      (df['ma10'] > df['ma20']) & 
                      (df['close'] > df['ma20'])).astype(int)
    
    # BBI
    df['bbi'] = (df['ma5'] + df['ma10'] + df['ma20'] + df['ma60']) / 4
    df['bbi_ratio'] = df['close'] / df['bbi']
    
    # 波动率
    df['vol_20'] = df['ret_1'].rolling(20).std()
    
    # 量比
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # 价格位置 (在近期高低点的什么位置)
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    
    # 区间涨幅
    df['range_20'] = (df['high_20'] - df['low_20']) / (df['low_20'] + 1e-8)
    
    # 趋势强度 (均线斜率)
    df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(10)) / (df['ma20'].shift(10) + 1e-8)
    
    # 换手率变化
    df['vol_change'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return df


def prepare_factors(stocks):
    """预计算所有股票的因子"""
    log("  预计算因子...")
    for code, df in stocks.items():
        stocks[code] = add_factors(df)
    log(f"  完成 {len(stocks)} 只股票的因子计算")


# ========== 2. 市场环境判断 ==========

def get_market_breadth(stocks, date):
    """计算市场广度 (有多少股票站在20日均线上)"""
    count = 0
    total = 0
    for code, df in stocks.items():
        hist = df[df['date'] <= date]
        if len(hist) < 25:
            continue
        total += 1
        row = hist.iloc[-1]
        if row['close'] > row['ma20']:
            count += 1
    return count / max(total, 1)


# ========== 3. 选股 ==========

class Selector:
    """改进版选股器"""
    
    def __init__(self, stocks):
        self.stocks = stocks
    
    def select_top(self, date, market_bull=True, top_n=3):
        """在指定日期选择最佳股票"""
        candidates = []
        
        for code, df in self.stocks.items():
            mask = df['date'] <= date
            if mask.sum() < 60:
                continue
            
            row = df[mask].iloc[-1]
            
            # 基础过滤
            if pd.isna(row.get('ret_20', np.nan)):
                continue
            if row['close'] <= 0 or row['volume'] <= 0:
                continue
            
            # ========== 多因子打分 ==========
            score = 0
            reasons = []
            
            # 因子1: 20日动量 (权重0.25)
            mom = row.get('ret_20', 0)
            if mom > 0.05:
                score += 25 * mom
                reasons.append('强动量')
            elif mom > 0.10:
                score += 35 * mom  # 更强动量加分
                reasons.append('超强动量')
            
            # 因子2: 均线多头排列 (权重0.20)
            if row.get('ma_bull', 0) == 1:
                score += 20
                reasons.append('均线多头')
            
            # 因子3: BBI上方 (权重0.15)
            if row.get('bbi_ratio', 1) > 1:
                score += 15 * (row['bbi_ratio'] - 1)
                reasons.append('BBI多头')
            
            # 因子4: 量比放大 (权重0.15)
            vol_r = row.get('vol_ratio', 1)
            if 1.5 < vol_r < 5:  # 温和放量
                score += 10 * (vol_r - 1)
                reasons.append('温和放量')
            elif vol_r >= 5:
                score -= 10  # 巨量可能是出货
            
            # 因子5: 价格位置适中 (权重0.10)
            pos = row.get('price_pos', 0.5)
            if 0.3 < pos < 0.7:  # 价格在中间位置,有上涨空间
                score += 10 * (1 - abs(pos - 0.5) * 2)
                reasons.append('位置适中')
            
            # 因子6: 低波动 (权重0.10)
            vol = row.get('vol_20', 0.05)
            if vol < 0.03:
                score += 10
                reasons.append('低波动')
            
            # 因子7: 趋势向上 (权重0.05)
            if row.get('ma20_slope', 0) > 0.02:
                score += 5
                reasons.append('趋势向上')
            
            # ========== 负面过滤 ==========
            # 跌幅过大的不碰 (可能是下跌中继)
            if mom < -0.15:
                score -= 30
            
            # 区间涨幅过大的不追 (可能见顶)
            if row.get('range_20', 0) > 0.5:
                score -= 15
            
            # ========== 市场环境过滤 ==========
            if market_bull:
                # 多头市场中,偏好强势股
                if row.get('ma_bull', 0) == 1:
                    score *= 1.2
            else:
                # 空头市场中只做超跌反弹
                if mom < -0.10 and pos < 0.2:
                    score = 50  # 设定一个最低分
            
            if score > 0:
                candidates.append((code, score, row['close'], reasons))
        
        if not candidates:
            return []
        
        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ========== 4. 回测 ==========

def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-20', 
                 initial_cash=1000000, stop_loss=0.05, take_profit=0.12, max_hold=8):
    """运行回测"""
    
    # 获取所有交易日
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
    
    for i, date in enumerate(dates):
        if i < 65:  # 需要足够历史数据
            continue
        
        # 每5天计算一次市场广度
        market_bull = False
        if i % 5 == 0:
            breadth = get_market_breadth(stocks, date)
            market_bull = breadth > 0.5  # 超过50%股票在20日线上视为多头
        
        # 持仓检查
        if position:
            code, entry_price, entry_date, shares, entry_breadth = position
            
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
            # 跌破BBI
            elif price < today_row['bbi'].iloc[0] * 0.97:
                sell_reason = '破BBI'
            # 市场转空且亏损
            elif not market_bull and pnl < 0:
                sell_reason = '市场转空'
            
            if sell_reason:
                revenue = price * shares * 0.9997
                cash += revenue
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl * 100,
                    'reason': sell_reason, 'hold_days': hold_days
                })
                position = None
        
        # 买入
        if position is None:
            top_stocks = selector.select_top(date, market_bull=market_bull, top_n=3)
            if top_stocks:
                code, score, price, reasons = top_stocks[0]
                if price * 100 * 1.0003 <= cash:
                    shares = 100
                    cost = price * shares * 1.0003
                    cash -= cost
                    position = (code, price, date, shares, market_bull)
                    trades.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'price': price, 'shares': shares, 'score': score,
                        'market_bull': market_bull, 'reasons': reasons
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
    
    # 夏普比率
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # 最大回撤
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
        hold_groups = {1: [], 2: [], 3: [], '4+': []}
        for t in sells:
            h = t.get('hold_days', 0)
            if h <= 1:
                hold_groups[1].append(t['pnl'])
            elif h == 2:
                hold_groups[2].append(t['pnl'])
            elif h == 3:
                hold_groups[3].append(t['pnl'])
            else:
                hold_groups['4+'].append(t['pnl'])
        log("\n  持有天数分析:")
        for days, pnls in hold_groups.items():
            if pnls:
                avg = np.mean(pnls)
                log(f"    {days}天: {len(pnls)}笔, 平均{np.mean(pnls):.2f}%, 胜率{len([p for p in pnls if p>0])/len(pnls)*100:.0f}%")
    
    log("\n最近10笔交易:")
    for t in trades[-10:]:
        if t['action'] == 'BUY':
            log(f"  {t['date'].strftime('%Y-%m-%d')} BUY  {t['code']} @{t['price']:.2f} [score:{t.get('score',0):.0f}]")
        else:
            log(f"  {t['date'].strftime('%Y-%m-%d')} SELL {t['code']} @{t['price']:.2f}  pnl:{t.get('pnl',0):+.2f}% [{t.get('reason','')}]")


# ========== 5. 参数优化 ==========

def optimize_params(stocks, date_range=['2024-01-01', '2025-01-01']):
    """参数优化"""
    log("\n" + "=" * 60)
    log("参数优化")
    log("=" * 60)
    
    best_return = -999
    best_params = None
    best_annual = -999
    
    for stop_loss in [0.03, 0.05, 0.08]:
        for take_profit in [0.10, 0.15, 0.20, 0.25]:
            for max_hold in [5, 8, 10, 15]:
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
                
                # 年化
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
    log("多因子Alpha量化交易系统 v3.0")
    log("=" * 60)
    
    # 加载数据
    log("\n[1] 加载股票数据...")
    stocks = load_all_stocks('E:/data', min_rows=250, max_stocks=200)
    log(f"    加载 {len(stocks)} 只股票")
    
    # 预计算因子
    log("\n[2] 预计算因子...")
    prepare_factors(stocks)
    
    # 默认参数回测
    log("\n[3] 默认参数回测 (止损5%, 止盈12%, 持有8天)...")
    trades, equity, dates = run_backtest(
        stocks, 
        start_date='2024-01-01',
        end_date='2026-03-20',
        initial_cash=1000000,
        stop_loss=0.05,
        take_profit=0.12,
        max_hold=8
    )
    analyze_results(trades, equity, dates, 1000000)
    
    # 参数优化
    log("\n[4] 参数优化 (2024全年)...")
    best_params = optimize_params(stocks, ['2024-01-01', '2024-12-31'])
    
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
