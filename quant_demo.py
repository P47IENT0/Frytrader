# -*- coding: utf-8 -*-
"""
FryTrader 量化交易系统演示
展示完整量化工作流：数据获取 -> 因子计算 -> 策略开发 -> 组合优化 -> 回测分析
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入FryTrader量化模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
import easytrader
from easytrader.quant_system import create_quant_system, quick_backtest
from easytrader.factor_engine import momentum_factor, mean_reversion_factor
from easytrader.strategy_engine import MeanReversionStrategy, MomentumStrategy


def main():
    print("🚀 FryTrader 量化交易系统演示\n")

    try:
        # 1. 加载配置文件
        import json
        with open('config_quant.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 2. 创建量化系统
        print("1️⃣ 初始化量化系统...")
        system = create_quant_system(config)

        # 2. 设置数据源
        print("2️⃣ 配置数据源...")
        # 使用tushare作为数据源（需要token）
        system.setup_data_provider('tushare', token="f6e0a7687738aae0631a7015aac4d91488983113b10962ad66ab3142")

        # 3. 定义交易标的
        symbols = ['000001', '000002', '600036', '600519']  # 平安银行、万科A、招商银行、贵州茅台
        symbol_names = {
            '000001': '平安银行',
            '000002': '万科A',
            '600036': '招商银行',
            '600519': '贵州茅台'
        }

        print(f"🎯 交易标的: {', '.join([f'{k}({v})' for k, v in symbol_names.items()])}")

        # 4. 加载历史数据
        print("3️⃣ 加载历史数据...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        data = system.load_market_data(symbols, start_date, end_date)

        if data.empty:
            print("❌ 数据加载失败！请检查网络连接")
            print("💡 提示: 可以尝试使用tushare数据源（需要注册获取token）")
            return

        print(f"✅ 成功加载 {len(data)} 条日线数据")

        # 5. 计算技术因子
        print("4️⃣ 计算技术因子...")
        factors = [
            'returns',      # 收益率
            'log_returns',  # 对数收益率
            'MA_5',         # 5日均线
            'MA_20',        # 20日均线
            'RSI_14',       # RSI指标
            'volatility',   # 波动率
            'MACD'          # MACD指标
        ]

        factor_data = system.calculate_factors(data, factors)
        print(f"✅ 计算完成 {len(factor_data.columns)} 个因子")

        # 6. 添加自定义因子
        print("5️⃣ 添加自定义因子...")

        # 动量因子
        system.create_custom_factor('momentum_20', momentum_factor, period=20)

        # 均值回归因子
        system.create_custom_factor('mean_reversion_20', mean_reversion_factor, period=20)

        # 重新计算包含自定义因子的数据
        all_factors = factors + ['momentum_20', 'mean_reversion_20']
        factor_data = system.calculate_factors(data, all_factors)
        print(f"✅ 自定义因子添加完成，总共 {len(factor_data.columns)} 个因子")

        # 7. 注册交易策略
        print("6️⃣ 注册交易策略...")

        # 均值回归策略
        system.register_strategy(
            MeanReversionStrategy,
            "均值回归策略",
            initial_capital=100000,
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5
        )

        # 动量策略
        system.register_strategy(
            MomentumStrategy,
            "动量策略",
            initial_capital=100000,
            momentum_period=20,
            top_n=3,
            rebalance_period=20
        )

        print(f"✅ 策略注册完成: {system.get_registered_strategies()}")

        # 8. 执行策略回测
        print("7️⃣ 执行策略回测...")

        strategies_to_test = ["均值回归策略", "动量策略"]

        for strategy_name in strategies_to_test:
            print(f"\n📊 回测策略: {strategy_name}")
            result = system.run_backtest(strategy_name, data)

            metrics = result['metrics']
            trades = result['trades']

            print("   绩效指标:"            print(f"   • 年化收益率: {metrics['annual_return']:.2%}")
            print(f"   • 夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"   • 最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"   • 胜率: {metrics['win_rate']:.1%}")
            print(f"   • 总交易次数: {metrics['total_trades']}")
            print(f"   • 最终权益: ¥{result['portfolio_values'][-1]:,.0f}")

        # 9. 策略对比分析
        print("\n8️⃣ 策略对比分析...")
        strategy_configs = [
            ("均值回归策略", {}),
            ("动量策略", {})
        ]

        comparison = system.compare_strategies(strategy_configs, data)
        print("策略对比结果:")
        print(comparison.round(4))

        # 10. 组合优化
        print("\n9️⃣ 执行组合优化...")

        # 计算收益率数据用于优化
        returns_data = data.set_index('date')[symbols].pct_change().dropna()

        # 最大化夏普比率
        sharpe_result = system.optimize_portfolio(returns_data, method='sharpe')
        if sharpe_result['success']:
            print("🎯 最优组合 (最大化夏普比率):")
            for symbol, weight in sharpe_result['weights'].items():
                print(".2%")
            print(".3f"            print(".2%")

        # 风险平价
        rp_result = system.optimize_portfolio(returns_data, method='risk_parity')
        if rp_result['success']:
            print("\n⚖️ 风险平价组合:")
            for symbol, weight in rp_result['weights'].items():
                print(".2%")

        # 11. 步前进分析
        print("\n🔄 执行步前进分析...")
        wf_result = system.walk_forward_test(
            "均值回归策略", data,
            train_window=252,  # 1年训练
            test_window=63,    # 3个月测试
            step_size=21       # 每月前进
        )

        print("步前进分析结果:")
        summary = wf_result['summary']
        print(".3f"        print(".2%"        print(".2%"        print(".3f"
        # 12. 生成回测报告
        print("\n📋 生成分析报告...")

        # 这里可以添加报告生成功能
        print("✅ 量化分析完成！")
        print("\n" + "="*60)
        print("🎉 恭喜！你已经完成了完整的量化交易工作流！")
        print("="*60)
        print("\n📚 接下来你可以:")
        print("   • 修改策略参数，优化交易逻辑")
        print("   • 添加新的技术指标和因子")
        print("   • 尝试不同的组合优化方法")
        print("   • 将策略连接到真实的交易接口")
        print("   • 搭建实时交易系统")

        print("\n🔧 进阶功能:")
        print("   • 实现机器学习选股模型")
        print("   • 添加风险管理系统")
        print("   • 构建高频交易策略")
        print("   • 开发Web监控界面")

        print("\n💡 提示:")
        print("   • 数据质量对策略表现至关重要")
        print("   • 避免过度优化（过拟合）")
        print("   • 考虑交易成本和市场冲击")
        print("   • 风险管理永远是第一位的")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def quick_start_example():
    """快速开始示例"""
    print("\n⚡ 快速开始示例:")

    try:
        # 一键回测
        result = quick_backtest(
            MeanReversionStrategy,
            ['000001', '000002'],  # 交易标的
            '2022-01-01',          # 开始日期
            '2023-01-01',          # 结束日期
            initial_capital=50000, # 初始资金
            data_provider='akshare'
        )

        if 'error' not in result:
            print("快速回测结果:")
            print(f"• 年化收益率: {result['metrics']['annual_return']:.2%}")
            print(f"• 夏普比率: {result['metrics']['sharpe_ratio']:.3f}")
            print(f"• 交易次数: {result['trades_count']}")
            print(f"• 最终价值: ¥{result['final_value']:,.0f}")
        else:
            print(f"快速回测失败: {result['error']}")

    except Exception as e:
        print(f"快速开始示例失败: {e}")


if __name__ == "__main__":
    main()
    quick_start_example()
