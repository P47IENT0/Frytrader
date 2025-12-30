# FryTrader 量化交易系统集成
import pandas as pd
import numpy as np
import json
import time
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# 导入FryTrader
try:
    import easytrader
    FRYTRADER_AVAILABLE = True
except ImportError:
    FRYTRADER_AVAILABLE = False
    print("警告: FryTrader 未安装，将使用模拟模式")

# 导入我们的量化模块
from data_sources import QuantDataManager
from factor_analysis import FactorAnalyzer
from strategy_backtest import Backtester, MomentumStrategy, MeanReversionStrategy, FactorStrategy
from portfolio_optimization import PortfolioOptimizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuantTradingSystem:
    """完整的量化交易系统"""

    def __init__(self, config_path: str = "config.json"):
        """
        初始化量化交易系统

        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.data_manager = QuantDataManager()
        self.factor_analyzer = FactorAnalyzer()
        self.trader = None
        self.current_positions = {}
        self.portfolio_value = 0

        # 初始化交易接口
        if FRYTRADER_AVAILABLE:
            self.initialize_trader()
        else:
            logger.warning("FryTrader不可用，使用模拟模式")

    def load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "user": "your_username",
            "password": "your_password",
            "exe_path": "path_to_trading_client",
            "client_type": "universal_client",
            "initial_capital": 1000000,
            "max_position_size": 0.1,  # 单股票最大仓位10%
            "rebalance_frequency": "daily",  # 调仓频率
            "risk_management": {
                "max_drawdown": 0.1,  # 最大回撤10%
                "stop_loss": 0.05,  # 止损5%
                "take_profit": 0.1   # 止盈10%
            },
            "strategy": {
                "type": "momentum",  # 策略类型
                "parameters": {
                    "momentum_window": 20,
                    "top_n": 5
                }
            }
        }

    def initialize_trader(self):
        """初始化交易接口"""
        try:
            client_type = self.config.get('client_type', 'universal_client')
            self.trader = easytrader.use(client_type)

            # 连接交易客户端
            if hasattr(self.trader, 'prepare'):
                self.trader.prepare(self.config)
            elif hasattr(self.trader, 'connect'):
                exe_path = self.config.get('exe_path', '')
                self.trader.connect(exe_path)

            logger.info(f"交易接口初始化成功: {client_type}")

        except Exception as e:
            logger.error(f"交易接口初始化失败: {e}")
            self.trader = None

    def update_market_data(self, symbols: List[str], lookback_days: int = 252) -> pd.DataFrame:
        """更新市场数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y%m%d')

        logger.info(f"更新市场数据: {len(symbols)}只股票, {start_date} 到 {end_date}")

        # 获取价格数据
        price_data = self.data_manager.get_daily_prices(symbols, start_date, end_date)

        # 计算技术指标
        for symbol in price_data:
            price_data[symbol] = self.data_manager.calculate_technical_indicators(price_data[symbol])

        # 转换为宽表格式用于分析
        market_data = self._convert_to_wide_format(price_data)

        # 保存数据
        self.data_manager.save_data(price_data, f"market_data_{end_date}")

        return market_data

    def _convert_to_wide_format(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """转换为宽表格式"""
        combined_data = []
        for symbol, df in price_data.items():
            df_copy = df.copy()
            df_copy.columns = pd.MultiIndex.from_product([[symbol], df_copy.columns])
            combined_data.append(df_copy)

        if combined_data:
            market_data = pd.concat(combined_data, axis=1)
            return market_data.dropna()
        return pd.DataFrame()

    def analyze_factors(self, market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """因子分析"""
        logger.info("开始因子分析...")

        # 加载价格数据到因子分析器
        price_data = {}
        for symbol in market_data.columns.levels[0]:
            symbol_data = market_data[symbol].copy()
            price_data[symbol] = symbol_data

        self.factor_analyzer.load_price_data(price_data)

        # 计算各类因子
        factors = {}

        # 技术因子
        factors['momentum'] = self.factor_analyzer.calculate_momentum_factors()
        factors['mean_reversion'] = self.factor_analyzer.calculate_mean_reversion_factors()
        factors['volatility'] = self.factor_analyzer.calculate_volatility_factors()

        # 合并因子
        all_factors = pd.concat(factors.values(), axis=1, keys=factors.keys())

        # 保存因子数据
        self.factor_analyzer.save_factors(factors)

        logger.info(f"因子分析完成，共生成{len(factors)}类因子")

        return factors

    def run_strategy_backtest(self, market_data: pd.DataFrame) -> Dict:
        """运行策略回测"""
        logger.info("开始策略回测...")

        strategy_type = self.config.get('strategy', {}).get('type', 'momentum')
        strategy_params = self.config.get('strategy', {}).get('parameters', {})

        # 创建策略
        if strategy_type == 'momentum':
            strategy = MomentumStrategy(**strategy_params)
        elif strategy_type == 'mean_reversion':
            strategy = MeanReversionStrategy(**strategy_params)
        elif strategy_type == 'factor':
            strategy = FactorStrategy(**strategy_params)
        else:
            logger.warning(f"未知策略类型: {strategy_type}，使用动量策略")
            strategy = MomentumStrategy(**strategy_params)

        # 创建回测器
        initial_capital = self.config.get('initial_capital', 1000000)
        backtester = Backtester(strategy, initial_capital)

        # 运行回测
        rebalance_freq = self.config.get('rebalance_frequency', 'M')
        results = backtester.run_backtest(market_data, rebalance_freq)

        # 保存回测结果
        backtester.print_summary()

        return results

    def optimize_portfolio(self, returns: pd.DataFrame) -> Dict:
        """组合优化"""
        logger.info("开始组合优化...")

        optimizer = PortfolioOptimizer(returns)

        # 多种优化方法
        optimizations = {}

        # 1. 最大夏普比率
        max_sharpe = optimizer.maximize_sharpe_ratio()
        if max_sharpe['success']:
            optimizations['max_sharpe'] = max_sharpe

        # 2. 最小波动率
        min_vol = optimizer.minimize_volatility()
        if min_vol['success']:
            optimizations['min_volatility'] = min_vol

        # 3. 风险平价
        risk_parity = optimizer.risk_parity_portfolio()
        if risk_parity['success']:
            optimizations['risk_parity'] = risk_parity

        # 选择最优组合 (这里选择最大夏普比率)
        if 'max_sharpe' in optimizations:
            optimal_portfolio = optimizations['max_sharpe']
            logger.info("选择最大夏普比率组合作为最优组合")
        elif optimizations:
            optimal_portfolio = list(optimizations.values())[0]
        else:
            logger.error("所有优化方法都失败了")
            return {}

        # 输出结果
        self.print_portfolio_summary(optimal_portfolio, returns.columns)

        return optimal_portfolio

    def print_portfolio_summary(self, portfolio: Dict, asset_names: List[str]):
        """打印组合摘要"""
        print("\n" + "="*50)
        print("最优投资组合")
        print("="*50)
        print(".2%")
        print(".2%")
        print(".2f")

        print("\n资产配置:")
        weights_dict = dict(zip(asset_names, portfolio['weights']))
        for asset, weight in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.01:  # 只显示权重>1%的资产
                print(".1f")

    def generate_trading_signals(self, market_data: pd.DataFrame) -> Dict[str, str]:
        """生成交易信号"""
        logger.info("生成交易信号...")

        strategy_type = self.config.get('strategy', {}).get('type', 'momentum')
        strategy_params = self.config.get('strategy', {}).get('parameters', {})

        # 获取最新数据
        latest_data = market_data.tail(50)  # 使用最近50天数据

        # 创建策略实例
        if strategy_type == 'momentum':
            strategy = MomentumStrategy(**strategy_params)
        elif strategy_type == 'mean_reversion':
            strategy = MeanReversionStrategy(**strategy_params)
        else:
            strategy = MomentumStrategy(**strategy_params)

        # 生成信号
        signals = strategy.generate_signals(latest_data, self.current_positions)

        logger.info(f"生成{len(signals)}个交易信号")

        return signals

    def execute_trades(self, signals: Dict[str, str], current_prices: Dict[str, float]):
        """执行交易"""
        if not self.trader:
            logger.warning("交易接口不可用，使用模拟模式")
            self.simulate_trades(signals, current_prices)
            return

        logger.info("开始执行交易...")

        max_position_size = self.config.get('max_position_size', 0.1)
        available_capital = self.get_available_capital()

        executed_trades = []

        for symbol, signal in signals.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

            if signal == 'buy':
                # 计算买入数量
                position_value = available_capital * max_position_size
                shares = int(position_value / (price * 1.003))  # 考虑交易成本

                if shares > 0:
                    try:
                        result = self.trader.buy(symbol, price=price, amount=shares)
                        executed_trades.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'shares': shares,
                            'price': price,
                            'result': result
                        })
                        logger.info(f"买入 {symbol}: {shares}股 @ {price}")
                    except Exception as e:
                        logger.error(f"买入{symbol}失败: {e}")

            elif signal == 'sell':
                # 获取当前持仓
                current_shares = self.current_positions.get(symbol, {}).get('shares', 0)

                if current_shares > 0:
                    try:
                        result = self.trader.sell(symbol, price=price, amount=current_shares)
                        executed_trades.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'shares': current_shares,
                            'price': price,
                            'result': result
                        })
                        logger.info(f"卖出 {symbol}: {current_shares}股 @ {price}")
                    except Exception as e:
                        logger.error(f"卖出{symbol}失败: {e}")

        # 更新持仓信息
        self.update_positions()

        return executed_trades

    def simulate_trades(self, signals: Dict[str, str], current_prices: Dict[str, float]):
        """模拟交易执行"""
        logger.info("模拟交易执行...")

        max_position_size = self.config.get('max_position_size', 0.1)
        available_capital = 1000000  # 模拟可用资金

        for symbol, signal in signals.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

            if signal == 'buy':
                position_value = available_capital * max_position_size
                shares = int(position_value / price)

                print(f"[模拟] 买入 {symbol}: {shares}股 @ {price}")

            elif signal == 'sell':
                current_shares = self.current_positions.get(symbol, {}).get('shares', 0)
                if current_shares > 0:
                    print(f"[模拟] 卖出 {symbol}: {current_shares}股 @ {price}")

    def get_available_capital(self) -> float:
        """获取可用资金"""
        if not self.trader:
            return 1000000  # 模拟资金

        try:
            balance = self.trader.balance
            if balance:
                return float(balance[0].get('可用资金', 0))
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")

        return 0

    def update_positions(self):
        """更新持仓信息"""
        if not self.trader:
            return

        try:
            positions = self.trader.position
            self.current_positions = {}

            if positions:
                for pos in positions:
                    symbol = pos.get('证券代码', '')
                    shares = float(pos.get('股份可用', 0))
                    if shares > 0:
                        self.current_positions[symbol] = {
                            'shares': shares,
                            'cost_price': float(pos.get('参考成本价', 0)),
                            'market_value': float(pos.get('参考市值', 0))
                        }

            logger.info(f"更新持仓信息: {len(self.current_positions)}只股票")

        except Exception as e:
            logger.error(f"更新持仓信息失败: {e}")

    def run_daily_routine(self, symbols: List[str]):
        """运行每日例行任务"""
        logger.info("开始每日量化交易例行任务...")

        try:
            # 1. 更新市场数据
            market_data = self.update_market_data(symbols, lookback_days=100)

            if market_data.empty:
                logger.error("无法获取市场数据")
                return

            # 2. 分析因子
            factors = self.analyze_factors(market_data)

            # 3. 生成交易信号
            signals = self.generate_trading_signals(market_data)

            # 4. 获取实时价格
            realtime_data = self.data_manager.get_realtime_quotes(symbols[:10])  # 只获取前10只股票的实时数据
            current_prices = dict(zip(realtime_data['代码'], realtime_data['最新价']))

            # 5. 执行交易
            if signals:
                executed_trades = self.execute_trades(signals, current_prices)
                logger.info(f"执行了{len(executed_trades)}笔交易")
            else:
                logger.info("今日无交易信号")

            # 6. 风险管理检查
            self.check_risk_limits()

            logger.info("每日例行任务完成")

        except Exception as e:
            logger.error(f"每日例行任务失败: {e}")

    def check_risk_limits(self):
        """检查风险限额"""
        risk_config = self.config.get('risk_management', {})

        try:
            # 获取当前组合表现
            if self.trader:
                balance = self.trader.balance
                positions = self.trader.position

                if balance and positions:
                    total_assets = float(balance[0].get('总资产', 0))
                    total_value = sum(float(pos.get('参考市值', 0)) for pos in positions)

                    # 检查回撤
                    max_drawdown_limit = risk_config.get('max_drawdown', 0.1)
                    # 这里需要历史数据来计算实际回撤，暂时跳过

                    logger.info(".2f")

        except Exception as e:
            logger.error(f"风险检查失败: {e}")

    def run_full_backtest_analysis(self, symbols: List[str], start_date: str, end_date: str):
        """运行完整回测分析"""
        logger.info("开始完整回测分析...")

        # 1. 获取历史数据
        price_data = self.data_manager.get_daily_prices(symbols, start_date, end_date)

        if not price_data:
            logger.error("无法获取历史数据")
            return

        # 2. 计算收益率
        returns_data = {}
        for symbol, df in price_data.items():
            returns_data[symbol] = df['close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)

        # 3. 运行策略回测
        market_data = self._convert_to_wide_format(price_data)
        backtest_results = self.run_strategy_backtest(market_data)

        # 4. 组合优化
        optimal_portfolio = self.optimize_portfolio(returns_df)

        # 5. 输出综合报告
        self.generate_comprehensive_report(backtest_results, optimal_portfolio)

        return {
            'backtest_results': backtest_results,
            'optimal_portfolio': optimal_portfolio,
            'returns_data': returns_df
        }

    def generate_comprehensive_report(self, backtest_results: Dict, optimal_portfolio: Dict):
        """生成综合报告"""
        print("\n" + "="*60)
        print("量化交易系统综合报告")
        print("="*60)

        print("\n📊 策略回测结果:")
        if 'sharpe_ratio' in backtest_results:
            print(".2%")
            print(".2%")
            print(".2f")
            print(".2%")

        print("\n💼 最优组合配置:")
        if optimal_portfolio and optimal_portfolio.get('success'):
            print(".2%")
            print(".2%")
            print(".2f")

        print("\n✅ 系统状态:")
        print(f"数据源: {'✅ 已配置' if self.data_manager else '❌ 未配置'}")
        print(f"因子分析: {'✅ 已配置' if self.factor_analyzer else '❌ 未配置'}")
        print(f"交易接口: {'✅ 已连接' if self.trader else '⚠️ 模拟模式'}")

        print("\n📝 使用建议:")
        print("1. 定期更新市场数据和因子")
        print("2. 监控交易信号和执行情况")
        print("3. 根据回测结果调整策略参数")
        print("4. 实施风险管理措施")


# === 使用示例 ===

def demo_full_system():
    """完整系统演示"""
    print("🚀 量化交易系统演示")
    print("=" * 50)

    # 初始化系统
    system = QuantTradingSystem()

    # 定义股票池
    symbols = ['000001', '600000', '000002', '600036', '600519', '000858', '002142']

    try:
        # 1. 运行完整回测分析
        print("\n1. 运行策略回测和组合优化...")
        analysis_results = system.run_full_backtest_analysis(
            symbols=symbols,
            start_date='20240101',
            end_date='20241201'
        )

        # 2. 运行每日例行任务
        print("\n2. 运行每日交易任务...")
        system.run_daily_routine(symbols[:5])  # 只用前5只股票测试

        # 3. 生成综合报告
        print("\n3. 生成系统报告...")
        system.generate_comprehensive_report(
            analysis_results.get('backtest_results', {}),
            analysis_results.get('optimal_portfolio', {})
        )

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("可能是网络连接或数据源问题，请检查配置")

    print("\n✨ 演示完成！")
    print("\n如需实际交易，请:")
    print("1. 配置真实交易账户信息")
    print("2. 测试交易接口连接")
    print("3. 从模拟模式逐步过渡到实盘交易")


if __name__ == "__main__":
    demo_full_system()
