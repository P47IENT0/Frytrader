# 量化策略回测框架
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('default')
sns.set_palette("husl")


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # 持仓
        self.trades = []  # 交易记录
        self.portfolio_values = []  # 组合价值历史
        self.transaction_costs = 0.003  # 交易成本 (0.3%)

    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """计算当前组合价值"""
        portfolio_value = self.capital

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                portfolio_value += position['shares'] * current_prices[symbol]

        return portfolio_value

    def execute_trade(self, symbol: str, shares: float, price: float, trade_type: str,
                     date: pd.Timestamp):
        """执行交易"""
        trade_value = shares * price
        cost = trade_value * self.transaction_costs

        if trade_type == 'buy':
            if trade_value + cost > self.capital:
                return False  # 资金不足

            self.capital -= (trade_value + cost)
            if symbol in self.positions:
                self.positions[symbol]['shares'] += shares
                self.positions[symbol]['cost_basis'] = (
                    (self.positions[symbol]['cost_basis'] * self.positions[symbol]['shares'] +
                     trade_value) / (self.positions[symbol]['shares'] + shares)
                )
            else:
                self.positions[symbol] = {
                    'shares': shares,
                    'cost_basis': price,
                    'entry_date': date
                }

        elif trade_type == 'sell':
            if symbol not in self.positions or self.positions[symbol]['shares'] < shares:
                return False  # 持仓不足

            self.capital += (trade_value - cost)
            self.positions[symbol]['shares'] -= shares

            # 如果持仓为0，删除该持仓
            if self.positions[symbol]['shares'] <= 0:
                del self.positions[symbol]

        # 记录交易
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'value': trade_value,
            'cost': cost,
            'type': trade_type
        })

        return True

    def get_positions_value(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """获取各持仓市值"""
        positions_value = {}

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value[symbol] = position['shares'] * current_prices[symbol]

        return positions_value


class TradingStrategy(ABC):
    """交易策略基类"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, current_positions: Dict[str, Dict]) -> Dict[str, str]:
        """生成交易信号"""
        pass

    @abstractmethod
    def calculate_position_size(self, symbol: str, signal: str, capital: float,
                              current_price: float) -> float:
        """计算仓位大小"""
        pass


class MomentumStrategy(TradingStrategy):
    """动量策略"""

    def __init__(self, momentum_window: int = 20, top_n: int = 5):
        self.momentum_window = momentum_window
        self.top_n = top_n

    def generate_signals(self, data: pd.DataFrame, current_positions: Dict[str, Dict]) -> Dict[str, str]:
        """生成动量信号"""
        signals = {}

        # 计算动量
        momentum_scores = {}
        for symbol in data.columns.levels[0]:  # 多层索引的股票代码
            if 'close' in data[symbol].columns:
                close_prices = data[symbol]['close']
                momentum = (close_prices.iloc[-1] / close_prices.iloc[-self.momentum_window] - 1)
                momentum_scores[symbol] = momentum

        # 选择前N名
        sorted_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_stocks = [stock for stock, _ in sorted_momentum[:self.top_n]]

        # 生成信号
        current_holding = set(current_positions.keys())

        # 对不在前N名但在持仓中的股票发出卖出信号
        for symbol in current_holding:
            if symbol not in top_stocks:
                signals[symbol] = 'sell'

        # 对在前N名但不在持仓中的股票发出买入信号
        for symbol in top_stocks:
            if symbol not in current_holding:
                signals[symbol] = 'buy'

        return signals

    def calculate_position_size(self, symbol: str, signal: str, capital: float,
                              current_price: float) -> float:
        """等权重仓位"""
        if signal == 'buy':
            # 每个股票分配相等资金
            position_value = capital / self.top_n
            shares = position_value / current_price
            return shares
        return 0


class MeanReversionStrategy(TradingStrategy):
    """均值回归策略"""

    def __init__(self, ma_window: int = 20, deviation_threshold: float = 0.05):
        self.ma_window = ma_window
        self.deviation_threshold = deviation_threshold

    def generate_signals(self, data: pd.DataFrame, current_positions: Dict[str, Dict]) -> Dict[str, str]:
        """生成均值回归信号"""
        signals = {}

        for symbol in data.columns.levels[0]:
            if 'close' in data[symbol].columns:
                close_prices = data[symbol]['close']
                ma = close_prices.rolling(self.ma_window).mean()
                current_price = close_prices.iloc[-1]
                ma_value = ma.iloc[-1]

                if pd.isna(ma_value):
                    continue

                deviation = (current_price - ma_value) / ma_value

                # 偏离均线过远的信号
                if deviation < -self.deviation_threshold:  # 价格低于均线太多，买入
                    if symbol not in current_positions:
                        signals[symbol] = 'buy'
                elif deviation > self.deviation_threshold:  # 价格高于均线太多，卖出
                    if symbol in current_positions:
                        signals[symbol] = 'sell'

        return signals

    def calculate_position_size(self, symbol: str, signal: str, capital: float,
                              current_price: float) -> float:
        """固定比例仓位"""
        if signal == 'buy':
            position_value = capital * 0.1  # 10%资金
            shares = position_value / current_price
            return shares
        return 0


class FactorStrategy(TradingStrategy):
    """多因子策略"""

    def __init__(self, factors: List[str], factor_weights: Optional[Dict[str, float]] = None):
        self.factors = factors
        self.factor_weights = factor_weights or {factor: 1.0/len(factors) for factor in factors}

    def generate_signals(self, data: pd.DataFrame, current_positions: Dict[str, Dict]) -> Dict[str, str]:
        """基于因子得分生成信号"""
        signals = {}

        # 计算综合因子得分
        factor_scores = {}

        for symbol in data.columns.levels[0]:
            score = 0
            for factor in self.factors:
                if factor in data[symbol].columns:
                    factor_value = data[symbol][factor].iloc[-1]
                    if pd.isna(factor_value):
                        continue
                    score += factor_value * self.factor_weights.get(factor, 1.0)

            factor_scores[symbol] = score

        # 选择得分最高的股票
        if factor_scores:
            sorted_scores = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
            top_stock = sorted_scores[0][0]

            # 持有得分最高的股票
            current_holding = set(current_positions.keys())

            # 卖出其他股票
            for symbol in current_holding:
                if symbol != top_stock:
                    signals[symbol] = 'sell'

            # 买入得分最高的股票（如果不在持仓中）
            if top_stock not in current_holding:
                signals[top_stock] = 'buy'

        return signals

    def calculate_position_size(self, symbol: str, signal: str, capital: float,
                              current_price: float) -> float:
        """全仓买入"""
        if signal == 'buy':
            shares = (capital * 0.95) / current_price  # 保留5%现金
            return shares
        return 0


class Backtester:
    """回测器"""

    def __init__(self, strategy: TradingStrategy, initial_capital: float = 1000000):
        self.strategy = strategy
        self.engine = BacktestEngine(initial_capital)
        self.results = {}

    def run_backtest(self, data: pd.DataFrame, rebalance_freq: str = 'M') -> Dict:
        """运行回测"""
        self.engine.reset()

        # 准备数据
        data.index = pd.to_datetime(data.index)

        # 确定调仓日期
        if rebalance_freq == 'M':  # 月度调仓
            rebalance_dates = data.resample('M').last().index
        elif rebalance_freq == 'W':  # 周度调仓
            rebalance_dates = data.resample('W').last().index
        else:  # 每日调仓
            rebalance_dates = data.index

        print(f"回测期间: {data.index[0]} 到 {data.index[-1]}")
        print(f"调仓频率: {rebalance_freq}, 调仓次数: {len(rebalance_dates)}")

        # 逐期执行回测
        for i, date in enumerate(rebalance_dates):
            if date not in data.index:
                continue

            # 获取当前价格数据
            current_data = data.loc[:date]

            # 获取当前价格
            current_prices = {}
            for symbol in data.columns.levels[0]:
                if 'close' in data[symbol].columns:
                    current_prices[symbol] = data[symbol]['close'].loc[date]

            # 生成交易信号
            signals = self.strategy.generate_signals(current_data, self.engine.positions)

            # 执行交易
            for symbol, signal in signals.items():
                if symbol in current_prices:
                    price = current_prices[symbol]
                    shares = self.strategy.calculate_position_size(
                        symbol, signal, self.engine.capital, price
                    )

                    if shares > 0:
                        success = self.engine.execute_trade(symbol, shares, price, signal, date)
                        if success:
                            print(".4f"
            # 记录组合价值
            portfolio_value = self.engine.calculate_portfolio_value(current_prices)
            self.engine.portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'capital': self.engine.capital
            })

        # 计算回测结果
        results = self.calculate_performance_metrics()
        self.results = results

        return results

    def calculate_performance_metrics(self) -> Dict:
        """计算绩效指标"""
        if not self.engine.portfolio_values:
            return {}

        # 组合价值时间序列
        portfolio_df = pd.DataFrame(self.engine.portfolio_values)
        portfolio_df.set_index('date', inplace=True)

        # 计算收益率
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1

        # 计算年化收益率
        total_days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        total_return = portfolio_df['cumulative_returns'].iloc[-1]
        annual_return = (1 + total_return) ** (365 / total_days) - 1

        # 计算波动率
        daily_vol = portfolio_df['returns'].std()
        annual_vol = daily_vol * np.sqrt(252)

        # 计算夏普比率 (假设无风险利率为3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol

        # 计算最大回撤
        cumulative = (1 + portfolio_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 计算胜率 (基于交易)
        winning_trades = 0
        total_trades = 0

        for trade in self.engine.trades:
            if trade['type'] == 'sell':
                # 计算该交易的盈亏
                # 这里简化处理，实际需要匹配买卖交易对
                total_trades += 1

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.engine.trades),
            'portfolio_values': portfolio_df,
            'trades': self.engine.trades
        }

        return results

    def plot_results(self):
        """绘制回测结果"""
        if not self.results:
            print("请先运行回测")
            return

        portfolio_df = self.results['portfolio_values']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 组合价值曲线
        axes[0, 0].plot(portfolio_df.index, portfolio_df['value'])
        axes[0, 0].set_title('Portfolio Value')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True)

        # 2. 累计收益率
        axes[0, 1].plot(portfolio_df.index, portfolio_df['cumulative_returns'] * 100)
        axes[0, 1].set_title('Cumulative Returns (%)')
        axes[0, 1].set_ylabel('Returns (%)')
        axes[0, 1].grid(True)

        # 3. 回撤图
        cumulative = (1 + portfolio_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        axes[1, 0].fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown (%)')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True)

        # 4. 月度收益率分布
        monthly_returns = portfolio_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        axes[1, 1].hist(monthly_returns, bins=20, alpha=0.7)
        axes[1, 1].set_title('Monthly Returns Distribution')
        axes[1, 1].set_xlabel('Returns (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def print_summary(self):
        """打印回测摘要"""
        if not self.results:
            print("请先运行回测")
            return

        print("=" * 50)
        print("回测结果摘要")
        print("=" * 50)

        metrics = self.results
        print(".2%")
        print(".2%")
        print(".2%")
        print(".2f")
        print(".2%")
        print(".1f")
        print(f"总交易次数: {metrics['total_trades']}")

        # 绩效分析
        if metrics['sharpe_ratio'] > 1:
            print("✅ 夏普比率良好 (>1)")
        if metrics['max_drawdown'] > -0.2:
            print("✅ 最大回撤可控 (<20%)")
        if metrics['win_rate'] > 0.5:
            print("✅ 胜率良好 (>50%)")


# === 使用示例 ===

def demo_backtest():
    """回测演示"""
    from data_sources import QuantDataManager

    # 获取数据
    dm = QuantDataManager()
    symbols = ['000001', '600000', '000002', '600036', '600519']  # 示例股票
    start_date = '20240101'
    end_date = '20241201'

    print("=== 获取历史数据 ===")
    price_data = dm.get_daily_prices(symbols, start_date, end_date)

    # 转换为多层索引格式
    combined_data = []
    for symbol, df in price_data.items():
        df_copy = df.copy()
        df_copy.columns = pd.MultiIndex.from_product([[symbol], df_copy.columns])
        combined_data.append(df_copy)

    if combined_data:
        market_data = pd.concat(combined_data, axis=1)
        market_data = market_data.dropna()

        print(f"数据形状: {market_data.shape}")

        # 1. 动量策略回测
        print("\n=== 动量策略回测 ===")
        momentum_strategy = MomentumStrategy(momentum_window=20, top_n=3)
        momentum_backtester = Backtester(momentum_strategy, initial_capital=1000000)

        momentum_results = momentum_backtester.run_backtest(market_data, rebalance_freq='M')
        momentum_backtester.print_summary()

        # 2. 均值回归策略回测
        print("\n=== 均值回归策略回测 ===")
        mr_strategy = MeanReversionStrategy(ma_window=20, deviation_threshold=0.03)
        mr_backtester = Backtester(mr_strategy, initial_capital=1000000)

        mr_results = mr_backtester.run_backtest(market_data, rebalance_freq='W')
        mr_backtester.print_summary()

        return momentum_results, mr_results

    return None, None


if __name__ == "__main__":
    demo_backtest()
