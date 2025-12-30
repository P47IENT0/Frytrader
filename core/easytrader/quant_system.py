# -*- coding: utf-8 -*-
"""
FryTrader 量化交易系统
整合数据获取、因子计算、策略开发、组合优化和回测功能的完整量化平台
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from easytrader.log import logger

# 导入各个模块
from .data_provider import DataManager, DataProviderFactory
from .factor_engine import FactorEngine
from .strategy_engine import StrategyEngine, MeanReversionStrategy, MomentumStrategy
from .portfolio_optimizer import PortfolioOptimizer, optimize_portfolio
from .backtest_engine import BacktestEngine, create_backtest_engine, PerformanceAnalyzer


class QuantSystem:
    """量化交易系统主类"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化量化系统

        Args:
            config: 系统配置
        """
        self.config = config or self._default_config()
        self.data_manager = None
        self.factor_engine = FactorEngine()
        self.strategy_engine = StrategyEngine()
        self.backtest_engine = create_backtest_engine(
            initial_capital=self.config.get('initial_capital', 100000),
            commission=self.config.get('commission', 0.0003),
            slippage=self.config.get('slippage', 0.0001)
        )

        logger.info("量化交易系统初始化完成")

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'initial_capital': 100000,
            'commission': 0.0003,
            'slippage': 0.0001,
            'risk_free_rate': 0.03,
            'data_provider': 'tushare',  # 'akshare' 或 'tushare'
            'benchmark_symbol': '000001'  # 基准标的
        }

    def setup_data_provider(self, provider_type: str = 'akshare', **kwargs):
        """设置数据提供商"""
        try:
            provider = DataProviderFactory.create_provider(provider_type, **kwargs)
            self.data_manager = DataManager(provider)
            logger.info(f"数据提供商设置成功: {provider_type}")
        except Exception as e:
            logger.error(f"设置数据提供商失败: {e}")
            raise

    def load_market_data(self,
                        symbols: List[str],
                        start_date: str,
                        end_date: str,
                        use_cache: bool = True) -> pd.DataFrame:
        """加载市场数据"""
        if not self.data_manager:
            raise ValueError("请先设置数据提供商")

        logger.info(f"加载数据: {len(symbols)} 个标的, {start_date} 至 {end_date}")
        data = self.data_manager.get_price_data(symbols, start_date, end_date, use_cache)

        if data.empty:
            logger.warning("未获取到数据")
            return pd.DataFrame()

        logger.info(f"成功加载 {len(data)} 条记录")
        return data

    def calculate_factors(self,
                         data: pd.DataFrame,
                         factors: Optional[List[str]] = None) -> pd.DataFrame:
        """计算因子"""
        if factors is None:
            factors = ['returns', 'MA_5', 'MA_10', 'RSI_14', 'volatility']

        logger.info(f"计算因子: {factors}")
        factor_data = self.factor_engine.calculate_factors(factors, data)
        logger.info(f"因子计算完成，共 {len(factor_data.columns)} 个因子")
        return factor_data

    def create_custom_factor(self,
                           name: str,
                           func: callable,
                           **params) -> None:
        """创建自定义因子"""
        self.factor_engine.create_custom_factor(name, func, **params)
        logger.info(f"自定义因子创建成功: {name}")

    def register_strategy(self, strategy_class: type, name: str, **params):
        """注册策略"""
        strategy = strategy_class(name=name, **params)
        strategy.initialize()
        self.strategy_engine.register_strategy(strategy)
        logger.info(f"策略注册成功: {name}")

    def run_backtest(self,
                    strategy_name: str,
                    data: pd.DataFrame,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """运行策略回测"""
        strategy = None
        for s in self.strategy_engine.strategies.values():
            if s.name == strategy_name:
                strategy = s.__class__
                break

        if not strategy:
            raise ValueError(f"策略未找到: {strategy_name}")

        logger.info(f"开始回测策略: {strategy_name}")

        result = self.backtest_engine.run_backtest(
            strategy, data,
            strategy_params={'name': strategy_name},
            start_date=start_date,
            end_date=end_date
        )

        logger.info("回测完成")
        return {
            'metrics': result.metrics,
            'trades': result.trades,
            'portfolio_values': result.portfolio_values,
            'dates': result.dates
        }

    def optimize_portfolio(self,
                          returns: pd.DataFrame,
                          method: str = 'sharpe',
                          **kwargs) -> Dict:
        """组合优化"""
        logger.info(f"执行组合优化: {method}")
        result = optimize_portfolio(returns, method, **kwargs)

        if result['success']:
            logger.info("组合优化成功")
            return result
        else:
            logger.error("组合优化失败")
            return result

    def compare_strategies(self,
                          strategies: List[Tuple[str, Dict]],
                          data: pd.DataFrame,
                          benchmark_symbol: Optional[str] = None) -> pd.DataFrame:
        """比较多个策略"""
        logger.info(f"比较 {len(strategies)} 个策略")

        strategy_classes = []
        for strategy_name, params in strategies:
            strategy_class = None
            for s in self.strategy_engine.strategies.values():
                if s.name == strategy_name:
                    strategy_class = s.__class__
                    break
            if strategy_class:
                strategy_classes.append((strategy_class, params))

        result = self.backtest_engine.compare_strategies(
            strategy_classes, data, benchmark_symbol
        )

        logger.info("策略比较完成")
        return result

    def walk_forward_test(self,
                         strategy_name: str,
                         data: pd.DataFrame,
                         train_window: int = 252,
                         test_window: int = 63,
                         step_size: int = 21) -> List[Dict]:
        """步前进分析"""
        strategy = None
        for s in self.strategy_engine.strategies.values():
            if s.name == strategy_name:
                strategy = s.__class__
                break

        if not strategy:
            raise ValueError(f"策略未找到: {strategy_name}")

        logger.info(f"执行步前进分析: {strategy_name}")

        results = self.backtest_engine.walk_forward_analysis(
            strategy, data, train_window, test_window, step_size
        )

        # 汇总结果
        metrics_list = [r.metrics for r in results]
        summary = {
            'total_periods': len(results),
            'avg_sharpe': np.mean([m.get('sharpe_ratio', 0) for m in metrics_list]),
            'avg_return': np.mean([m.get('annual_return', 0) for m in metrics_list]),
            'avg_max_drawdown': np.mean([m.get('max_drawdown', 0) for m in metrics_list]),
            'sharpe_std': np.std([m.get('sharpe_ratio', 0) for m in metrics_list]),
        }

        logger.info("步前进分析完成")
        return {
            'summary': summary,
            'detailed_results': metrics_list
        }

    def get_available_factors(self) -> List[str]:
        """获取可用因子列表"""
        return self.factor_engine.get_available_factors()

    def get_registered_strategies(self) -> List[str]:
        """获取已注册策略列表"""
        return list(self.strategy_engine.strategies.keys())

    def clear_cache(self):
        """清空数据缓存"""
        if self.data_manager:
            self.data_manager.clear_cache()
            logger.info("数据缓存已清空")


# 便捷函数
def create_quant_system(config: Optional[Dict] = None) -> QuantSystem:
    """创建量化系统"""
    return QuantSystem(config)


def quick_backtest(strategy_class: type,
                  symbols: List[str],
                  start_date: str,
                  end_date: str,
                  initial_capital: float = 100000,
                  data_provider: str = 'akshare') -> Dict:
    """快速回测函数"""
    # 创建系统
    system = create_quant_system({
        'initial_capital': initial_capital
    })

    # 设置数据源
    system.setup_data_provider(data_provider)

    # 加载数据
    data = system.load_market_data(symbols, start_date, end_date)

    if data.empty:
        return {'error': '无法获取数据'}

    # 运行回测
    result = system.backtest_engine.run_backtest(
        strategy_class, data,
        strategy_params={'name': f'Quick_{strategy_class.__name__}'}
    )

    return {
        'metrics': result.metrics,
        'trades_count': len(result.trades),
        'final_value': result.portfolio_values[-1] if result.portfolio_values else initial_capital
    }


# 示例工作流
def example_workflow():
    """示例工作流"""
    print("=== FryTrader 量化交易系统示例 ===\n")

    # 1. 创建量化系统
    print("1. 创建量化系统...")
    system = create_quant_system()

    # 2. 设置数据源
    print("2. 设置数据源...")
    system.setup_data_provider('akshare')

    # 3. 加载数据
    print("3. 加载市场数据...")
    symbols = ['000001', '000002', '600036']  # 平安银行、万科A、招商银行
    data = system.load_market_data(symbols, '2022-01-01', '2023-01-01')

    if data.empty:
        print("❌ 数据加载失败，请检查网络连接和数据源")
        return

    print(f"✅ 成功加载 {len(data)} 条记录")

    # 4. 计算因子
    print("4. 计算技术因子...")
    factors = system.calculate_factors(data, ['returns', 'MA_5', 'RSI_14'])
    print(f"✅ 计算完成 {len(factors.columns)} 个因子")

    # 5. 注册策略
    print("5. 注册交易策略...")
    system.register_strategy(MeanReversionStrategy, "均值回归策略",
                           initial_capital=100000)

    # 6. 运行回测
    print("6. 执行策略回测...")
    backtest_result = system.run_backtest("均值回归策略", data)

    print("✅ 回测完成！")
    print("\n=== 回测结果 ===")
    metrics = backtest_result['metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 7. 组合优化
    print("\n7. 执行组合优化...")
    returns_data = data.set_index('date')[symbols].pct_change().dropna()
    opt_result = system.optimize_portfolio(returns_data, method='sharpe')

    if opt_result['success']:
        print("✅ 组合优化成功！")
        print("最优权重:")
        for symbol, weight in opt_result['weights'].items():
            print(".4f")
        else:
            print("❌ 组合优化失败")

    print("\n=== 示例完成 ===")
    print("你现在可以:")
    print("1. 修改策略参数和逻辑")
    print("2. 添加新的技术指标")
    print("3. 尝试不同的组合优化方法")
    print("4. 集成真实的交易接口")


if __name__ == "__main__":
    example_workflow()
