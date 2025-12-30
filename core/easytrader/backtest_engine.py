# -*- coding: utf-8 -*-
"""
量化回测引擎
提供完整的策略回测功能，包括交易模拟、绩效分析等
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from easytrader.log import logger
from easytrader.strategy_engine import BaseStrategy, Order, OrderStatus, Position


class BacktestResult:
    """回测结果类"""

    def __init__(self):
        self.portfolio_values = []  # 组合价值时间序列
        self.returns = []  # 收益率序列
        self.trades = []  # 交易记录
        self.positions = {}  # 持仓记录
        self.dates = []  # 日期序列
        self.metrics = {}  # 绩效指标

    def calculate_metrics(self, risk_free_rate: float = 0.03):
        """计算绩效指标"""
        if not self.portfolio_values:
            return {}

        portfolio_values = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_values.pct_change().fillna(0)

        # 基础指标
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)
        annual_return = total_return / (len(portfolio_values) / 252)  # 假设252个交易日

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 胜率（基于交易）
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 盈亏比
        profits = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trades if t['pnl'] < 0]
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')

        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 索提诺比率
        downside_returns = returns[returns < 0]
        sortino_ratio = (annual_return - risk_free_rate) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0

        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': total_trades,
            'avg_trade_pnl': np.mean([t['pnl'] for t in self.trades]) if self.trades else 0,
        }

        return self.metrics

    def plot_results(self, save_path: Optional[str] = None):
        """绘制回测结果图表"""
        if not self.portfolio_values:
            logger.warning("没有可绘制的数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('回测结果分析', fontsize=16)

        portfolio_values = pd.Series(self.portfolio_values, index=self.dates)

        # 1. 组合价值曲线
        axes[0, 0].plot(portfolio_values.index, portfolio_values.values)
        axes[0, 0].set_title('组合价值')
        axes[0, 0].set_ylabel('价值')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 回撤图
        returns = portfolio_values.pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[0, 1].set_title('回撤')
        axes[0, 1].set_ylabel('回撤比例')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 收益分布
        if self.trades:
            pnl_values = [t['pnl'] for t in self.trades]
            axes[1, 0].hist(pnl_values, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('交易盈亏分布')
            axes[1, 0].set_xlabel('盈亏')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].axvline(0, color='red', linestyle='--')

        # 4. 月度收益热力图（如果数据足够）
        if len(portfolio_values) > 30:
            monthly_returns = portfolio_values.resample('M').last().pct_change().fillna(0)
            if len(monthly_returns) > 1:
                # 创建热力图数据
                heatmap_data = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
                heatmap_data = heatmap_data.unstack().fillna(0)

                if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
                    sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=axes[1, 1])
                    axes[1, 1].set_title('月度收益热力图')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存至: {save_path}")

        plt.show()


class BacktestEngine:
    """回测引擎"""

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.0003,  # 佣金率
                 slippage: float = 0.0001,   # 滑点
                 risk_free_rate: float = 0.03):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission: 交易佣金率
            slippage: 滑点
            risk_free_rate: 无风险利率
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

        self.result = BacktestResult()

    def run_backtest(self,
                    strategy_class: type,
                    data: pd.DataFrame,
                    strategy_params: Optional[Dict] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> BacktestResult:
        """
        运行回测

        Args:
            strategy_class: 策略类
            data: 市场数据，必须包含日期列和价格数据
            strategy_params: 策略参数
            start_date: 开始日期
            end_date: 结束日期
        """
        # 数据预处理
        data = data.copy()
        if 'date' not in data.columns and data.index.name == 'date':
            data = data.reset_index()

        if start_date:
            data = data[data['date'] >= start_date]
        if end_date:
            data = data[data['date'] <= end_date]

        data = data.sort_values('date').reset_index(drop=True)

        # 初始化策略
        strategy_params = strategy_params or {}
        strategy = strategy_class(**strategy_params)
        strategy.current_capital = self.initial_capital
        strategy.portfolio_value = self.initial_capital

        # 获取所有交易标的
        symbols = [col for col in data.columns if col not in ['date', 'Date']]

        # 初始化持仓
        for symbol in symbols:
            strategy.positions[symbol] = Position(symbol)

        # 逐日回测
        for idx, row in data.iterrows():
            current_date = row['date']
            self.result.dates.append(current_date)

            # 更新价格
            prices = {}
            for symbol in symbols:
                if symbol in row:
                    prices[symbol] = row[symbol]

            strategy.update_positions(prices)

            # 执行策略
            try:
                # 这里需要根据策略类型调用不同的方法
                if hasattr(strategy, 'generate_signals'):
                    # 信号类策略
                    for symbol in symbols:
                        signal = strategy.generate_signals(symbol)
                        self._execute_signal(strategy, symbol, signal, prices.get(symbol, 0))
                elif hasattr(strategy, 'on_data'):
                    # 事件驱动策略
                    strategy.on_data(pd.DataFrame([row]))
                else:
                    # 自定义策略逻辑
                    pass

            except Exception as e:
                logger.error(f"策略执行错误 ({current_date}): {e}")
                continue

            # 处理挂单
            self._process_pending_orders(strategy, prices)

            # 记录组合价值
            self.result.portfolio_values.append(strategy.portfolio_value)
            self.result.positions[current_date] = deepcopy(strategy.positions)

        # 计算绩效指标
        self.result.calculate_metrics(self.risk_free_rate)

        return self.result

    def _execute_signal(self, strategy: BaseStrategy, symbol: str, signal: str, price: float):
        """执行交易信号"""
        if signal == 'buy' and price > 0:
            # 计算可买入数量（简化版）
            available_capital = strategy.current_capital * 0.1  # 每次最多使用10%资金
            quantity = int(available_capital / (price * (1 + self.commission + self.slippage)))
            if quantity > 0:
                order = strategy.buy(symbol, quantity, price * (1 + self.slippage))
                if order:
                    self._fill_order(strategy, order, price)

        elif signal == 'sell':
            position = strategy.positions.get(symbol)
            if position and position.quantity > 0:
                order = strategy.sell(symbol, position.quantity, price * (1 - self.slippage))
                if order:
                    self._fill_order(strategy, order, price)

        elif signal == 'close':
            position = strategy.positions.get(symbol)
            if position and position.quantity > 0:
                order = strategy.sell(symbol, position.quantity, price * (1 - self.slippage))
                if order:
                    self._fill_order(strategy, order, price)

    def _fill_order(self, strategy: BaseStrategy, order: Order, execution_price: float):
        """成交订单"""
        if order.status != OrderStatus.PENDING:
            return

        # 计算实际成交价格和成本
        actual_price = execution_price
        commission_cost = actual_price * order.quantity * self.commission

        if order.order_type.name.startswith('BUY'):
            # 买入
            total_cost = actual_price * order.quantity + commission_cost

            if total_cost <= strategy.current_capital:
                strategy.current_capital -= total_cost
                position = strategy.positions[order.symbol]
                position.add_position(order.quantity, actual_price)

                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.filled_price = actual_price

                # 记录交易
                self.result.trades.append({
                    'date': order.timestamp,
                    'symbol': order.symbol,
                    'type': 'buy',
                    'quantity': order.quantity,
                    'price': actual_price,
                    'commission': commission_cost,
                    'pnl': 0  # 买入时的盈亏为0
                })

        elif order.order_type.name.startswith('SELL'):
            # 卖出
            position = strategy.positions[order.symbol]
            if position.quantity >= order.quantity:
                proceeds = actual_price * order.quantity - commission_cost
                strategy.current_capital += proceeds

                # 计算卖出盈亏
                cost_basis = position.avg_price * order.quantity
                realized_pnl = (actual_price - position.avg_price) * order.quantity - commission_cost

                position.add_position(-order.quantity, actual_price)

                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.filled_price = actual_price

                # 记录交易
                self.result.trades.append({
                    'date': order.timestamp,
                    'symbol': order.symbol,
                    'type': 'sell',
                    'quantity': order.quantity,
                    'price': actual_price,
                    'commission': commission_cost,
                    'pnl': realized_pnl
                })

    def _process_pending_orders(self, strategy: BaseStrategy, prices: Dict[str, float]):
        """处理挂单（简化版，立即成交）"""
        for order in strategy.orders[:]:  # 复制列表避免修改时的问题
            if order.status == OrderStatus.PENDING:
                symbol_price = prices.get(order.symbol, 0)
                if symbol_price > 0:
                    self._fill_order(strategy, order, symbol_price)
                    strategy.orders.remove(order)

    def compare_strategies(self,
                          strategies: List[Tuple[type, Dict]],
                          data: pd.DataFrame,
                          benchmark_symbol: Optional[str] = None) -> pd.DataFrame:
        """
        比较多个策略的回测结果

        Args:
            strategies: [(策略类, 参数字典), ...]
            data: 市场数据
            benchmark_symbol: 基准标的
        """
        results = {}

        for strategy_class, params in strategies:
            strategy_name = params.get('name', strategy_class.__name__)
            logger.info(f"回测策略: {strategy_name}")

            result = self.run_backtest(strategy_class, data, params)
            results[strategy_name] = result.metrics

        # 如果有基准，添加基准收益
        if benchmark_symbol and benchmark_symbol in data.columns:
            benchmark_returns = data[benchmark_symbol].pct_change().fillna(0)
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_total_return = benchmark_cumulative.iloc[-1] - 1

            results['Benchmark'] = {
                'total_return': benchmark_total_return,
                'annual_return': benchmark_total_return / (len(data) / 252),
                'volatility': benchmark_returns.std() * np.sqrt(252),
                'sharpe_ratio': (benchmark_returns.mean() - self.risk_free_rate/252) / benchmark_returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(benchmark_cumulative),
            }

        return pd.DataFrame(results).T

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """计算最大回撤"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def walk_forward_analysis(self,
                            strategy_class: type,
                            data: pd.DataFrame,
                            train_window: int = 252,
                            test_window: int = 63,
                            step_size: int = 21) -> List[BacktestResult]:
        """
        步前进分析

        Args:
            strategy_class: 策略类
            data: 数据
            train_window: 训练窗口长度（天）
            test_window: 测试窗口长度（天）
            step_size: 步长（天）
        """
        results = []
        n_periods = len(data)

        for start_idx in range(0, n_periods - train_window - test_window + 1, step_size):
            train_end_idx = start_idx + train_window
            test_end_idx = train_end_idx + test_window

            # 训练数据
            train_data = data.iloc[start_idx:train_end_idx]

            # 测试数据
            test_data = data.iloc[train_end_idx:test_end_idx]

            # 这里可以添加参数优化逻辑
            # 简化版：直接使用默认参数

            # 在测试数据上回测
            result = self.run_backtest(strategy_class, test_data)
            results.append(result)

        return results


def create_backtest_engine(initial_capital: float = 100000,
                          commission: float = 0.0003,
                          slippage: float = 0.0001) -> BacktestEngine:
    """创建回测引擎"""
    return BacktestEngine(initial_capital, commission, slippage)


# 绩效分析工具
class PerformanceAnalyzer:
    """绩效分析器"""

    @staticmethod
    def calculate_rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
        """计算滚动夏普比率"""
        excess_returns = returns - 0.03/252  # 假设无风险利率3%
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = excess_returns.rolling(window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)

    @staticmethod
    def calculate_alpha_beta(strategy_returns: pd.Series,
                           market_returns: pd.Series) -> Tuple[float, float]:
        """计算Alpha和Beta"""
        # 协方差和方差
        covariance = np.cov(strategy_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        beta = covariance / market_variance if market_variance > 0 else 0
        alpha = strategy_returns.mean() - beta * market_returns.mean()

        return alpha, beta

    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """计算VaR（历史模拟法）"""
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
        """计算期望短缺（CVaR）"""
        var = PerformanceAnalyzer.calculate_var(returns, confidence)
        return returns[returns <= var].mean()


# 使用示例
if __name__ == "__main__":
    from easytrader.strategy_engine import MeanReversionStrategy

    # 创建示例数据
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    np.random.seed(42)

    # 生成两只股票的价格数据
    n_days = len(dates)
    price1 = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
    price2 = 50 * np.exp(np.cumsum(np.random.randn(n_days) * 0.025))

    data = pd.DataFrame({
        'date': dates,
        '000001': price1,
        '000002': price2
    })

    # 创建回测引擎
    engine = create_backtest_engine(initial_capital=100000)

    # 运行回测
    result = engine.run_backtest(
        MeanReversionStrategy,
        data,
        strategy_params={'name': 'Test_MeanReversion', 'initial_capital': 100000}
    )

    # 输出结果
    print("回测结果:")
    for metric, value in result.metrics.items():
        print(f"{metric}: {value:.4f}")

    # 绘制结果
    result.plot_results()
