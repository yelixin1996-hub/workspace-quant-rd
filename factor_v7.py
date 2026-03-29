# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v7.0 (年化10%目标版)
=============================================
根本性改进:
1. 缩小选股池到100只 - 质量优先
2. 市场多头才持仓 - 空仓躲避熊市
3. 提高选股门槛 - 分数>80才买
4. 扩大止损减少被洗 - 止损10%
5. 缩小止盈早落袋 - 止盈15%
"""

import pandas as pd
import numpy as np
import os

def log(msg):
    print(msg, flush=True)

# ========== 1. 数据加载 ==========

def load_all_stocks(data_dir='E:/data', min_rows=200, max_stocks=100):
    """缩小到100只优质股票"""
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
    """因子计算"""
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
    
    # 均线排列
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
    
    return df


def prepare_factors(stocks):
    log("  预计算因子...")
    for code, df in stocks.items():
        stocks[code] = add_factors(df)
    log(f"  完成 {len(stocks)} 只股票的因子计算")


# ========== 2. 市场环境 ==========

def get_market_env(stocks, date):
    """判断市场环境 - 用沪深300成分股判断"""
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


# ========== 3. 严格选股 ==========

class Selector:
    def __init__(self, stocks):
        self.stocks = stocks
    
    def select_top(self, date, market_env='neutral', min_score=80, top_n=3):
        """严格选股 - 分数不够不要"""
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
            if row.get('vol_ratio', 1) < 0.5 or row.get('vol_ratio', 1) > 6:
                continue
            if row.get('vol_20', 0.05) > 0.08:  # 排除高波动
                continue
            
            score = 0
            
            # ========== 核心因子 ==========
            
            # 1. 20日动量 (最重要)
            mom = row.get('ret_20', 0)
            if mom > 0.08:
                score += 35
            elif mom > 0.15:
                score += 50
            elif mom > 0.05:
                score += 25
            elif mom < 0:
                score += 20 * mom  # 负动量惩罚
            
            # 2. 均线多头排列 (重要)
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
            if 45 < rsi < 65:
                score += 10
            elif rsi < 40:
                score += 5
            
            # 7. 量比温和
            vol_r = row.get('vol_ratio', 1)
            if 1.2 < vol_r < 3:
                score += 8
            
            # 8. 低波动加分
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
            
            # ========== 负面清单 ==========
            if mom < -0.15:
                score -= 30
            if rsi > 80:
                score -= 20
            if row.get('bbi_ratio', 1) < 0.95:
                score -= 15
            
            # ========== 环境适应 ==========
            if market_env == 'bull':
                if row.get('ma_bull', 0) == 1:
                    score *= 1.2
            elif market_env == 'bear':
                if row.get('ma_bull', 0) == 1 and mom > 0.05:
                    score *= 1.1  # 熊市只选最强的
                else:
                    continue  # 熊市不做
            
            if score >= min_score:
                candidates.append((code, score, row['close']))
        
        if not candidates:
            return []
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ========== 4. 回测 ==========

def run_backtest(stocks, start_date='2024-01-01', end_date='2026-03-20', 
                 initial_cash=1000000, stop_loss=0.10, take_profit=0.15, 
                 max_hold=10, min_score=80):
    """回测 - 市场环境过滤+严格选股"""
    
    all_dates = set()
    for df in stocks.values():
        all_dates.update(df['date'].tolist())
    dates = sorted([d for d in all_dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)])
    
    log(f"  回测期: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    log(f"  交易日: {len(dates)}, 股票池: {len(stocks)}")
    
    selector = Selector(stocks)
    
    cash = initial_cash
    position = None
    trades = []
    equity_curve = [initial_cash]
    
    highest_price = 0
    
    # 熔断
    consecutive_losses = 0
    fuse_active = False
    fuse_end_date = None
    
    for i, date in enumerate(dates):
        if i < 65:
            continue
        
        # 熔断恢复
        if fuse_active and date >= fuse_end_date:
            fuse_active = False
            consecutive_losses = 0
        
        # 市场环境
        market_env = 'neutral'
        if i % 5 == 0:
            market_env, bull_ratio = get_market_env(stocks, date)
        
        # 持仓检查
        if position:
            code, entry_price, entry_date, shares = position
            
            df = stocks.get(code)
            today_row = df[df['date'] == date]
            if today_row.empty:
                continue
            price = today_row['close'].iloc[0]
            today_rsi = today_row['rsi_14'].iloc[0]
            
            if price > highest_price:
                highest_price = price
            
            hold_days = (date - entry_date).days
            pnl = (price - entry_price) / entry_price
            
            sell_reason = None
            
            # 跟踪止盈
            if highest_price > entry_price * 1.05:
                retrace = (highest_price - price) / highest_price
                if retrace > 0.08:
                    sell_reason = '跟踪止盈'
            
            # 止损
            if pnl <= -stop_loss and not sell_reason:
                sell_reason = '止损'
            
            # 止盈
            if pnl >= take_profit and not sell_reason:
                sell_reason = '止盈'
            
            # 到期
            if hold_days >= max_hold and not sell_reason:
                sell_reason = '到期'
            
            # RSI超买
            if today_rsi > 85 and pnl > 0 and not sell_reason:
                sell_reason = 'RSI超买'
            
            # 市场转熊且亏损
            if market_env == 'bear' and pnl < 0 and not sell_reason:
                sell_reason = '市场转熊'
            
            if sell_reason:
                revenue = price * shares * 0.9997
                cash += revenue
                pnl_pct = pnl * 100
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'shares': shares, 'pnl': pnl_pct,
                    'reason': sell_reason, 'hold_days': hold_days
                })
                
                # 熔断
                if pnl < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= 4:
                        fuse_active = True
                        fuse_end_date = date + pd.Timedelta(days=5)
                        log(f"  熔断: 连亏{consecutive_losses}次, 暂停至{fuse_end_date.strftime('%Y-%m-%d')}")
                else:
                    consecutive_losses = 0
                
                position = None
                highest_price = 0
        
        # 买入 - 只在非熊市且有熔断保护
        if position is None and not fuse_active and market_env != 'bear':
            top_stocks = selector.select_top(date, market_env=market_env, min_score=min_score, top_n=3)
            if top_stocks:
                code, score, price = top_stocks[0]
                if price * 100 * 1.0003 <= cash:
                    shares = 100
                    cost = price * shares * 1.0003
                    cash -= cost
                    position = (code, price, date, shares)
                    highest_price = price
                    trades.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'price': price, 'shares': shares, 'score': score,
                        'market_env': market_env
                    })
        
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
    
    # 持有天数分析
    if sells:
        hold_groups = {'1-3': [], '4-7': [], '8+': []}
        for t in sells:
            h = t.get('hold_days', 0)
            if h <= 3:
                hold_groups['1-3'].append(t['pnl'])
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
    
    # 卖出原因
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


def quick_scan(stocks, date_range=['2024-01-01', '2024-06-01']):
    log("\n" + "=" * 60)
    log("参数扫描")
    log("=" * 60)
    
    best_annual = -999
    best_params = None
    
    params = [
        (0.08, 0.12, 8, 80),
        (0.10, 0.15, 8, 80),
        (0.10, 0.15, 10, 80),
        (0.10, 0.15, 12, 80),
        (0.10, 0.18, 10, 80),
        (0.10, 0.18, 12, 80),
        (0.12, 0.15, 10, 80),
        (0.12, 0.18, 12, 80),
        (0.08, 0.12, 8, 85),
        (0.10, 0.15, 10, 85),
    ]
    
    for stop_loss, take_profit, max_hold, min_score in params:
        trades, equity, dates = run_backtest(
            stocks, 
            start_date=date_range[0],
            end_date=date_range[1],
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold=max_hold,
            min_score=min_score
        )
        
        if len(equity) < 10:
            continue
        
        final = equity[-1]
        n_years = (dates[-1] - dates[65]).days / 365 if len(dates) > 65 else 0.5
        annual = ((final / 1000000) ** (1 / max(n_years, 0.1)) - 1) * 100
        
        if annual > best_annual:
            best_annual = annual
            best_params = {
                'stop_loss': stop_loss, 
                'take_profit': take_profit, 
                'max_hold': max_hold,
                'min_score': min_score
            }
            log(f"*** 新最佳: 年化{annual:.2f}% 止损{stop_loss} 止盈{take_profit} 持有{max_hold} 分数{min_score}")
    
    log(f"\n最佳: {best_params} 年化{best_annual:.2f}%")
    return best_params


if __name__ == '__main__':
    log("=" * 60)
    log("多因子Alpha量化交易系统 v7.0 (年化10%目标版)")
    log("=" * 60)
    
    log("\n[1] 加载股票数据 (100只优质股)...")
    stocks = load_all_stocks('E:/data', min_rows=250, max_stocks=100)
    log(f"    加载 {len(stocks)} 只股票")
    
    log("\n[2] 预计算因子...")
    prepare_factors(stocks)
    
    log("\n[3] 默认参数回测...")
    trades, equity, dates = run_backtest(
        stocks, 
        start_date='2024-01-01',
        end_date='2026-03-20',
        initial_cash=1000000,
        stop_loss=0.10,
        take_profit=0.15,
        max_hold=10,
        min_score=80
    )
    analyze_results(trades, equity, dates, 1000000)
    
    log("\n[4] 参数扫描...")
    best_params = quick_scan(stocks, ['2024-01-01', '2024-06-01'])
    
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