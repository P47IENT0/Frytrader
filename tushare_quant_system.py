# -*- coding: utf-8 -*-
"""
基于Tushare的FryTrader量化交易系统
专门为tushare数据源优化的量化平台
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from easytrader.log import logger

# 导入tushare
import tushare as ts
import time
import warnings
warnings.filterwarnings('ignore')

# 导入各个模块
from easytrader.factor_engine import FactorEngine
from easytrader.strategy_engine import StrategyEngine, MeanReversionStrategy, MomentumStrategy
from easytrader.portfolio_optimizer import PortfolioOptimizer, optimize_portfolio
from easytrader.backtest_engine import BacktestEngine, create_backtest_engine, PerformanceAnalyzer


class TushareQuantSystem:
    """基于Tushare的量化交易系统"""

    def __init__(self, config: Optional[Dict] = None, token: str = None):
        """
        初始化Tushare量化系统

        Args:
            config: 系统配置
            token: tushare token，如果不提供则使用默认token
        """
        self.config = config or self._default_config()
        self.token = token or "f6e0a7687738aae0631a7015aac4d91488983113b10962ad66ab3142"

        # 初始化tushare
        ts.set_token(self.token)
        self.pro = ts.pro_api()

        # 初始化各个组件
        self.factor_engine = FactorEngine()
        self.strategy_engine = StrategyEngine()
        self.backtest_engine = create_backtest_engine(
            initial_capital=self.config.get('initial_capital', 100000),
            commission=self.config.get('commission', 0.0003),
            slippage=self.config.get('slippage', 0.0001)
        )

        logger.info("Tushare量化交易系统初始化完成")

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'initial_capital': 100000,
            'commission': 0.0003,
            'slippage': 0.0001,
            'risk_free_rate': 0.03,
            'benchmark_symbol': '000001.SZ'
        }

    def get_daily_prices_tushare(self, symbols: List[str], start_date: str, end_date: str,
                                adj: str = 'qfq') -> pd.DataFrame:
        """
        使用tushare获取日线数据

        Args:
            symbols: 股票代码列表 (格式: ['000001.SZ', '600036.SH'])
            start_date: 开始日期 (格式: '20230101')
            end_date: 结束日期 (格式: '20231201')
            adj: 复权方式 ('qfq': 前复权, 'hfq': 后复权, None: 不复权)
        """
        all_data = []

        for ts_code in symbols:
            try:
                logger.info(f"获取 {ts_code} 数据...")

                # tushare获取日线数据
                df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

                if df.empty:
                    logger.warning(f"未获取到 {ts_code} 数据")
                    continue

                # 数据预处理
                df = df.sort_values('trade_date').reset_index(drop=True)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)

                # 添加symbol列
                df['symbol'] = ts_code.split('.')[0]  # 移除.SZ/.SH后缀用于后续处理

                # 复权处理
                if adj:
                    try:
                        # 获取复权因子
                        adj_df = self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
                        if not adj_df.empty:
                            adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'])
                            adj_df.set_index('trade_date', inplace=True)

                            # 合并复权因子
                            df = df.join(adj_df[['adj_factor']], how='left')
                            df['adj_factor'] = df['adj_factor'].fillna(method='ffill').fillna(1)

                            # 计算复权价格
                            if adj == 'qfq':  # 前复权
                                df['open'] = df['open'] * df['adj_factor']
                                df['high'] = df['high'] * df['adj_factor']
                                df['low'] = df['low'] * df['adj_factor']
                                df['close'] = df['close'] * df['adj_factor']
                            elif adj == 'hfq':  # 后复权
                                latest_adj = df['adj_factor'].iloc[-1]
                                df['open'] = df['open'] * df['adj_factor'] / latest_adj
                                df['high'] = df['high'] * df['adj_factor'] / latest_adj
                                df['low'] = df['low'] * df['adj_factor'] / latest_adj
                                df['close'] = df['close'] * df['adj_factor'] / latest_adj
                    except Exception as e:
                        logger.warning(f"{ts_code} 复权处理失败: {e}")

                all_data.append(df)
                logger.info(f"✅ {ts_code} 数据获取完成: {len(df)} 条记录")

                # 控制请求频率
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"获取 {ts_code} 数据失败: {e}")
                continue

        if not all_data:
            logger.error("未获取到任何数据")
            return pd.DataFrame()

        # 合并所有数据
        combined_data = pd.concat(all_data, axis=0)
        logger.info(f"总共获取到 {len(combined_data)} 条记录")
        return combined_data

    def get_financial_data_tushare(self, symbols: List[str], report_type: str = 'income',
                                  periods: int = 4) -> Dict[str, pd.DataFrame]:
        """
        获取财务数据

        Args:
            symbols: 股票代码列表
            report_type: 财务报表类型 ('income', 'balance', 'cashflow')
            periods: 获取的报告期数
        """
        results = {}

        for ts_code in symbols:
            try:
                logger.info(f"获取 {ts_code} {report_type} 财务数据...")

                if report_type == 'income':
                    df = self.pro.income(ts_code=ts_code, limit=periods*4)  # 每个季度一条记录
                elif report_type == 'balance':
                    df = self.pro.balancesheet(ts_code=ts_code, limit=periods*4)
                elif report_type == 'cashflow':
                    df = self.pro.cashflow(ts_code=ts_code, limit=periods*4)

                if not df.empty:
                    df['ann_date'] = pd.to_datetime(df['ann_date'])
                    df.set_index('ann_date', inplace=True)
                    results[ts_code.split('.')[0]] = df
                    logger.info(f"✅ {ts_code} 财务数据获取完成")
                else:
                    logger.warning(f"未获取到 {ts_code} 财务数据")

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"获取 {ts_code} 财务数据失败: {e}")
                continue

        return results

    def get_index_data_tushare(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数数据

        Args:
            index_code: 指数代码 (如: '000001.SH' 上证指数, '399001.SZ' 深证成指)
            start_date: 开始日期
            end_date: 结束日期
        """
        try:
            logger.info(f"获取指数 {index_code} 数据...")

            df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)

            if not df.empty:
                df = df.sort_values('trade_date').reset_index(drop=True)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                logger.info(f"✅ 指数数据获取完成: {len(df)} 条记录")
                return df
            else:
                logger.warning(f"未获取到指数 {index_code} 数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            return pd.DataFrame()

    def load_market_data(self,
                        symbols: List[str],
                        start_date: str,
                        end_date: str,
                        use_cache: bool = True) -> pd.DataFrame:
        """加载市场数据"""
        logger.info(f"加载数据: {len(symbols)} 个标的, {start_date} 至 {end_date}")
        data = self.get_daily_prices_tushare(symbols, start_date, end_date)

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

    def get_available_factors(self) -> List[str]:
        """获取可用因子列表"""
        return self.factor_engine.get_available_factors()

    def get_registered_strategies(self) -> List[str]:
        """获取已注册策略列表"""
        return list(self.strategy_engine.strategies.keys())


# 便捷函数
def create_tushare_quant_system(config: Optional[Dict] = None, token: str = None) -> TushareQuantSystem:
    """创建Tushare量化系统"""
    return TushareQuantSystem(config, token)


def quick_backtest_tushare(strategy_class: type,
                          symbols: List[str],
                          start_date: str,
                          end_date: str,
                          initial_capital: float = 100000,
                          token: str = None) -> Dict:
    """快速回测函数（Tushare版本）"""
    # 创建系统
    system = create_tushare_quant_system({
        'initial_capital': initial_capital
    }, token)

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
def example_tushare_workflow():
    """Tushare示例工作流"""
    print("=== FryTrader Tushare量化系统示例 ===\n")

    # 1. 创建Tushare量化系统
    print("1. 创建Tushare量化系统...")
    system = create_tushare_quant_system()

    # 2. 加载数据
    print("2. 加载Tushare数据...")
    symbols = ['000001.SZ', '000002.SZ', '600036.SH']  # 平安银行、万科A、招商银行
    data = system.load_market_data(symbols, '20230101', '20231201')

    if data.empty:
        print("❌ 数据加载失败，请检查tushare token和网络连接")
        return

    print(f"✅ 成功加载 {len(data)} 条记录")

    # 3. 计算因子
    print("3. 计算技术因子...")
    factors = system.calculate_factors(data, ['returns', 'MA_5', 'RSI_14'])
    print(f"✅ 计算完成 {len(factors.columns)} 个因子")

    # 4. 注册策略
    print("4. 注册交易策略...")
    system.register_strategy(MeanReversionStrategy, "均值回归策略",
                           initial_capital=100000)

    # 5. 运行回测
    print("5. 执行策略回测...")
    backtest_result = system.run_backtest("均值回归策略", data)

    print("✅ 回测完成！")
    print("\n=== 回测结果 ===")
    metrics = backtest_result['metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 6. 获取财务数据
    print("\n6. 获取财务数据...")
    financial_data = system.get_financial_data_tushare(symbols[:2], 'income', periods=2)
    if financial_data:
        print(f"✅ 获取到 {len(financial_data)} 只股票的财务数据")

    print("\n=== Tushare示例完成 ===")
    print("你现在可以:")
    print("- 使用更多tushare数据接口")
    print("- 结合财务数据进行选股")
    print("- 开发多因子模型")


if __name__ == "__main__":
    example_tushare_workflow()
