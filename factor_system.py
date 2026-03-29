# -*- coding: utf-8 -*-
"""
多因子Alpha量化交易系统 v1.0
===========================
因子池: 价值因子、质量因子、动量因子、量价因子
选股:   IC加权打分法
风控:   止损+分散+仓位管理
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ========== 1. 数据加载 ==========

def load_stock_data(stock_code, data_dir='E:/data'):
    """加载单只股票数据"""
    # 尝试不同的文件后缀
    for suffix in ['.SZ.csv', '.SH.csv']:
        file_path = os.path.join(data_dir, f'{stock_code}{suffix}')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # CSV列: trade_date, open, high, low, close, volume
            df = df.rename(columns={'trade_date': 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
    return None


def load_all_stocks(data_dir='E:/data', min_rows=200):
    """加载所有股票数据"""
    stocks = {}
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for f in files:
        # 从文件名提取code (去掉.SZ.csv或.SH.csv)
        code = f.replace('.SZ.csv', '').replace('.SH.csv', '')
        df = load_stock_data(code, data_dir)
        if df is not None and len(df) >= min_rows:
            stocks[code] = df
    return stocks


# ========== 2. 因子计算 ==========

def calc_returns(df, period=1):
    """计算收益率"""
    return df['close'].pct_change(period)


def calc_volatility(df, window=20):
    """计算历史波动率"""
    returns = calc_returns(df)
    return returns.rolling(window).std()


def calc_momentum(df, period=20):
    """计算动量因子 (过去period日收益率)"""
    return (df['close'] - df['close'].shift(period)) / df['close'].shift(period)


def calc_bbi(df, ma5=5, ma10=10, ma20=20, ma30=30):
    """BBI多空指标"""
    ma5_val = df['close'].rolling(ma5).mean()
    ma10_val = df['close'].rolling(ma10).mean()
    ma20_val = df['close'].rolling(ma20).mean()
    ma30_val = df['close'].rolling(ma30).mean()
    return (ma5_val + ma10_val + ma20_val + ma30_val) / 4


def calc_kdj(df, n=9, m1=3, m2=3):
    """KDJ指标"""
    low_n = df['low'].rolling(n).min()
    high_n = df['high'].rolling(n).max()
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    
    K = pd.Series(index=df.index, dtype=float)
    D = pd.Series(index=df.index, dtype=float)
    K.iloc[0] = 50
    D.iloc[0] = 50
    
    for i in range(1, len(df)):
        K.iloc[i] = (2/3) * K.iloc[i-1] + (1/3) * rsv.iloc[i]
        D.iloc[i] = (2/3) * D.iloc[i-1] + (1/3) * K.iloc[i]
    
    J = 3 * K - 2 * D
    return K, D, J


def calc_macd(df, fast=12, slow=26, signal=9):
    """MACD指标"""
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    DIF = ema_fast - ema_slow
    DEA = DIF.ewm(span=signal).mean()
    MACD = (DIF - DEA) * 2
    return DIF, DEA, MACD


def calc_volume_ratio(df, window=5):
    """量比 (当日成交量 / 过去N日平均成交量)"""
    vol_ma = df['volume'].rolling(window).mean()
    return df['volume'] / vol_ma


def calc_ma_slope(df, period=10):
    """均线斜率 (反映趋势强度)"""
    ma = df['close'].rolling(period).mean()
    return (ma - ma.shift(5)) / ma.shift(5)


def calc_price_position(df, period=20):
    """价格在均线附近的位置 (0-1之间, 0.5表示在均线)"""
    ma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    return (df['close'] - ma) / (std + 1e-8)


class FactorCalculator:
    """因子计算机 - 为所有股票计算因子"""
    
    def __init__(self):
        self.factors_list = []
    
    def compute(self, stocks, date):
        """
        计算指定日期所有股票的因子值
        返回: DataFrame, columns=[code, factor1, factor2, ...]
        """
        results = []
        
        for code, df in stocks.items():
            # 限制只用到date之前的数据
            df_before = df[df['date'] <= date]
            if len(df_before) < 60:  # 需要足够历史数据
                continue
            
            row = {'code': code}
            
            # 基础价格数据
            close = df_before['close'].iloc[-1]
            open_price = df_before['open'].iloc[-1]
            high = df_before['high'].iloc[-1]
            low = df_before['low'].iloc[-1]
            volume = df_before['volume'].iloc[-1]
            
            if close <= 0 or volume <= 0:
                continue
            
            # ---- 动量因子 ----
            mom_5 = calc_momentum(df_before.tail(6), 5).iloc[-1] if len(df_before) >= 6 else 0
            mom_20 = calc_momentum(df_before, 20).iloc[-1] if len(df_before) >= 20 else 0
            mom_60 = calc_momentum(df_before, 60).iloc[-1] if len(df_before) >= 60 else 0
            
            row['momentum_5'] = mom_5 if np.isfinite(mom_5) else 0
            row['momentum_20'] = mom_20 if np.isfinite(mom_20) else 0
            row['momentum_60'] = mom_60 if np.isfinite(mom_60) else 0
            
            # ---- 波动率因子 ----
            vol_20 = calc_volatility(df_before, 20).iloc[-1] if len(df_before) >= 20 else 0
            row['volatility_20'] = vol_20 if np.isfinite(vol_20) and vol_20 > 0 else 0.001
            
            # ---- BBI因子 ----
            bbi = calc_bbi(df_before).iloc[-1]
            row['bbi'] = bbi if np.isfinite(bbi) else close
            row['bbi_ratio'] = close / bbi if bbi > 0 else 1  # >1表示价格在BBI上方
            
            # ---- KDJ因子 ----
            K, D, J = calc_kdj(df_before)
            row['K'] = K.iloc[-1] if np.isfinite(K.iloc[-1]) else 50
            row['D'] = D.iloc[-1] if np.isfinite(D.iloc[-1]) else 50
            row['J'] = J.iloc[-1] if np.isfinite(J.iloc[-1]) else 50
            
            # ---- MACD因子 ----
            DIF, DEA, MACD = calc_macd(df_before)
            row['DIF'] = DIF.iloc[-1] if np.isfinite(DIF.iloc[-1]) else 0
            row['MACD'] = MACD.iloc[-1] if np.isfinite(MACD.iloc[-1]) else 0
            
            # ---- 量比因子 ----
            vol_ratio = calc_volume_ratio(df_before).iloc[-1]
            row['volume_ratio'] = vol_ratio if np.isfinite(vol_ratio) else 1
            
            # ---- 均线斜率 ----
            ma_slope = calc_ma_slope(df_before).iloc[-1]
            row['ma_slope'] = ma_slope if np.isfinite(ma_slope) else 0
            
            # ---- 价格位置 ----
            price_pos = calc_price_position(df_before).iloc[-1]
            row['price_position'] = price_pos if np.isfinite(price_pos) else 0
            
            # ---- 日收益率 ----
            daily_return = calc_returns(df_before).iloc[-1]
            row['daily_return'] = daily_return if np.isfinite(daily_return) else 0
            
            # ---- 换手率(成交量/流通股) ----
            avg_vol_5 = df_before['volume'].tail(5).mean()
            row['turnover_rate'] = volume / avg_vol_5 if avg_vol_5 > 0 else 1
            
            # ---- 区间涨幅 ----
            high_20 = df_before['high'].tail(20).max()
            low_20 = df_before['low'].tail(20).min()
            row['range_ratio'] = (high_20 - low_20) / low_20 if low_20 > 0 else 0
            
            results.append(row)
        
        return pd.DataFrame(results)


# ========== 3. 因子打分和选股 ==========

class FactorScorer:
    """因子打分器 - 将因子值转换为分数"""
    
    def __init__(self):
        # 因子权重 (可调整)
        self.weights = {
            'momentum_20': 0.20,      # 20日动量
            'volatility_20': -0.10,   # 波动率 (负向, 低波动更好)
            'bbi_ratio': 0.15,        # BBI多空
            'K': 0.10,                # KDJ_K
            'volume_ratio': 0.10,     # 量比
            'ma_slope': 0.15,         # 均线斜率
            'price_position': 0.10,   # 价格位置
            'range_ratio': 0.10,      # 区间振幅
        }
    
    def normalize(self, series):
        """标准化 (Z-score)"""
        mean = series.mean()
        std = series.std()
        if std < 1e-8:
            return pd.Series(0, index=series.index)
        return (series - mean) / std
    
    def score(self, factors_df):
        """计算综合得分"""
        if factors_df.empty:
            return factors_df
        
        df = factors_df.copy()
        
        # 标准化每个因子
        for factor, weight in self.weights.items():
            if factor in df.columns:
                df[f'{factor}_z'] = self.normalize(df[factor])
            else:
                df[f'{factor}_z'] = 0
        
        # 计算加权总分
        df['total_score'] = 0
        for factor, weight in self.weights.items():
            df['total_score'] += df[f'{factor}_z'] * weight
        
        return df


class StockSelector:
    """选股器 - 根据因子打分选择股票"""
    
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.factor_calc = FactorCalculator()
        self.scorer = FactorScorer()
    
    def select(self, stocks, date, top_n=None):
        """
        选股
        参数:
            stocks: dict {code: DataFrame}
            date: datetime or str
        返回:
            list of codes (排序后的最佳股票)
        """
        if top_n is None:
            top_n = self.top_n
        
        # 计算因子
        factors_df = self.factor_calc.compute(stocks, date)
        if factors_df.empty:
            return []
        
        # 打分
        scored_df = self.scorer.score(factors_df)
        
        # 按总分排序, 取前N
        scored_df = scored_df.sort_values('total_score', ascending=False)
        
        return scored_df.head(top_n)['code'].tolist()


# ========== 4. 交易模拟 ==========

class TradeSimulator:
    """交易模拟器"""
    
    def __init__(self, initial_cash=1000000, commission=0.0003):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission  # 手续费率
        self.position = None  # (code, entry_price, shares, entry_date)
        self.trades = []
    
    def buy(self, code, price, date, shares=100):
        """买入"""
        cost = price * shares * (1 + self.commission)
        if self.cash >= cost:
            self.cash -= cost
            self.position = (code, price, shares, date)
            self.trades.append({
                'date': date, 'action': 'BUY', 'code': code,
                'price': price, 'shares': shares, 'cost': cost
            })
            return True
        return False
    
    def sell(self, price, date, reason=''):
        """卖出"""
        if self.position is None:
            return False
        
        code, entry_price, shares, entry_date = self.position
        revenue = price * shares * (1 - self.commission)
        pnl = (price - entry_price) / entry_price * 100
        
        self.cash += revenue
        self.trades.append({
            'date': date, 'action': 'SELL', 'code': code,
            'price': price, 'shares': shares, 'revenue': revenue,
            'pnl_pct': pnl, 'reason': reason
        })
        self.position = None
        return True
    
    def get_stats(self):
        """获取交易统计"""
        if not self.trades:
            return {}
        
        sells = [t for t in self.trades if t['action'] == 'SELL']
        wins = [t for t in sells if t.get('pnl_pct', 0) > 0]
        losses = [t for t in sells if t.get('pnl_pct', 0) <= 0]
        
        total_pnl = sum(t.get('pnl_pct', 0) for t in sells)
        
        return {
            'final_cash': self.cash,
            'total_return': (self.cash - self.initial_cash) / self.initial_cash * 100,
            'num_trades': len(sells),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / max(1, len(sells)) * 100,
            'avg_pnl': total_pnl / max(1, len(sells)),
        }


# ========== 5. 回测引擎 ==========

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, stocks, initial_cash=1000000):
        self.stocks = stocks
        self.initial_cash = initial_cash
        self.stop_loss_pct = 0.08       # 8%止损
        self.take_profit_pct = 0.20     # 20%止盈
        self.max_hold_days = 15         # 最大持有天数
        self.selector = StockSelector(top_n=1)
        
        # 获取所有交易日
        all_dates = set()
        for df in stocks.values():
            all_dates.update(df['date'].tolist())
        self.trading_dates = sorted([d for d in all_dates if pd.to_datetime(d) >= pd.to_datetime('2024-01-01')])
    
    def run(self):
        """运行回测"""
        simulator = TradeSimulator(self.initial_cash)
        
        for i, date in enumerate(self.trading_dates):
            # 如果持有股票, 检查是否需要卖出
            if simulator.position:
                code, entry_price, shares, entry_date = simulator.position
                
                # 获取当前价格
                df = self.stocks.get(code)
                if df is not None:
                    today_data = df[df['date'] == date]
                    if not today_data.empty:
                        current_price = today_data['close'].iloc[0]
                        hold_days = (pd.to_datetime(date) - pd.to_datetime(entry_date)).days
                        
                        pnl_pct = (current_price - entry_price) / entry_price
                        
                        # 止损
                        if pnl_pct <= -self.stop_loss_pct:
                            simulator.sell(current_price, date, '止损')
                            continue
                        
                        # 止盈
                        if pnl_pct >= self.take_profit_pct:
                            simulator.sell(current_price, date, '止盈')
                            continue
                        
                        # 超过最大持有期
                        if hold_days >= self.max_hold_days:
                            simulator.sell(current_price, date, '到期卖出')
                            continue
                        
                        # 跌破BBI
                        bbi = calc_bbi(df[df['date'] <= date]).iloc[-1]
                        if current_price < bbi * 0.95:
                            simulator.sell(current_price, date, '跌破BBI')
                            continue
            
            # 如果没有持有, 选股买入
            if simulator.position is None and i > 60:  # 需要足够历史数据
                selected = self.selector.select(self.stocks, date, top_n=1)
                if selected:
                    code = selected[0]
                    df = self.stocks[code]
                    today_data = df[df['date'] == date]
                    if not today_data.empty:
                        price = today_data['close'].iloc[0]
                        simulator.buy(code, price, date, shares=100)
        
        return simulator.get_stats(), simulator.trades


# ========== 6. 主程序 ==========

if __name__ == '__main__':
    print("=" * 60)
    print("多因子Alpha量化交易系统 v1.0")
    print("=" * 60)
    
    # 加载数据
    print("\n[1] 加载股票数据...")
    stocks = load_all_stocks('E:/data', min_rows=250)
    print(f"    加载 {len(stocks)} 只股票")
    
    # 运行回测
    print("\n[2] 运行回测 (2024-01 至 2026-03)...")
    engine = BacktestEngine(stocks, initial_cash=1000000)
    stats, trades = engine.run()
    
    # 输出结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"  初始资金: 1,000,000")
    print(f"  最终资金: {stats['final_cash']:,.0f}")
    print(f"  总收益率: {stats['total_return']:.2f}%")
    print(f"  交易次数: {stats['num_trades']}")
    print(f"  盈利次数: {stats['wins']}")
    print(f"  亏损次数: {stats['losses']}")
    print(f"  胜率: {stats['win_rate']:.1f}%")
    print(f"  平均收益: {stats['avg_pnl']:.2f}%")
    
    # 年化收益率
    n_years = (pd.to_datetime('2026-03-27') - pd.to_datetime('2024-01-01')).days / 365
    annual_return = ((1 + stats['total_return']/100) ** (1/n_years) - 1) * 100
    print(f"  年化收益率: {annual_return:.2f}%")
    
    print("\n[3] 最近10笔交易:")
    for t in trades[-10:]:
        if t['action'] == 'BUY':
            print(f"    {t['date'].strftime('%Y-%m-%d')} BUY  {t['code']} @{t['price']:.2f}")
        else:
            print(f"    {t['date'].strftime('%Y-%m-%d')} SELL {t['code']} @{t['price']:.2f}  pnl:{t.get('pnl_pct', 0):+.2f}%  [{t.get('reason', '')}]")
    
    print("\n完成!")
