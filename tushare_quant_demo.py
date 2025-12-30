# -*- coding: utf-8 -*-
"""
FryTrader Tushare量化交易演示
使用tushare数据源进行完整的量化工作流
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入FryTrader量化模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
import easytrader
from easytrader.quant_system import create_quant_system, quick_backtest
from easytrader.factor_engine import momentum_factor, mean_reversion_factor
from easytrader.strategy_engine import MeanReversionStrategy, MomentumStrategy

# 导入tushare数据管理器
from data_sources import QuantDataManager


def main():
    print("🚀 FryTrader Tushare量化交易演示\n")

    try:
        # 1. 初始化tushare数据管理器
        print("1️⃣ 初始化Tushare数据管理器...")
        dm = QuantDataManager()

        # 2. 创建量化系统
        print("2️⃣ 创建量化系统...")
        system = create_quant_system({
            'initial_capital': 100000,  # 初始资金10万
            'commission': 0.0003,       # 手续费0.03%
            'slippage': 0.0001,         # 滑点0.01%
            'data_provider': 'tushare'  # 指定使用tushare
        })

        # 3. 设置数据源为tushare
        print("3️⃣ 配置数据源...")
        system.setup_data_provider('tushare')

        # 4. 定义交易标的
        symbols = ['000001.SZ', '000002.SZ', '600036.SH', '600519.SH']  # 平安银行、万科A、招商银行、贵州茅台
        symbol_names = {
            '000001.SZ': '平安银行',
            '000002.SZ': '万科A',
            '600036.SH': '招商银行',
            '600519.SH': '贵州茅台'
        }

        print(f"🎯 交易标的: {', '.join([f'{k}({v})' for k, v in symbol_names.items()])}")

        # 5. 使用tushare获取历史数据
        print("4️⃣ 使用Tushare获取历史数据...")
        start_date = '20230101'
        end_date = datetime.now().strftime('%Y%m%d')

        # 使用data_sources中的方法获取数据
        price_data = dm.get_daily_prices([s.split('.')[0] for s in symbols], start_date, end_date)

        if not price_data:
            print("❌ 使用akshare获取数据...")
            # 如果tushare获取失败，使用akshare作为备选
            system.setup_data_provider('akshare')
            data = system.load_market_data([s.split('.')[0] for s in symbols], start_date, end_date)
        else:
            print("✅ Tushare数据获取成功！")
            # 将数据转换为FryTrader期望的格式
            data = pd.DataFrame()
            for symbol, df in price_data.items():
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                df_copy['date'] = df_copy.index
                if data.empty:
                    data = df_copy
                else:
                    data = pd.concat([data, df_copy])

        if data.empty:
            print("❌ 数据加载失败！请检查网络连接或tushare token")
            return

        print(f"✅ 成功加载 {len(data)} 条日线数据")

        # 6. 计算技术因子
        print("5️⃣ 计算技术因子...")
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

        # 7. 添加自定义因子
        print("6️⃣ 添加自定义因子...")

        # 动量因子
        system.create_custom_factor('momentum_20', momentum_factor, period=20)

        # 均值回归因子
        system.create_custom_factor('mean_reversion_20', mean_reversion_factor, period=20)

        # 重新计算包含自定义因子的数据
        all_factors = factors + ['momentum_20', 'mean_reversion_20']
        factor_data = system.calculate_factors(data, all_factors)
        print(f"✅ 自定义因子添加完成，总共 {len(factor_data.columns)} 个因子")

        # 8. 注册交易策略
        print("7️⃣ 注册交易策略...")

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

        # 9. 执行策略回测
        print("8️⃣ 执行策略回测...")

        strategies_to_test = ["均值回归策略", "动量策略"]

        for strategy_name in strategies_to_test:
            print(f"\n📊 回测策略: {strategy_name}")
            result = system.run_backtest(strategy_name, data)

            metrics = result['metrics']
            trades = result['trades']

            print("   绩效指标:")
            print(f"   - 年化收益率: {metrics['annual_return']:.2%}")
            print(f"   - 夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"   - 最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"   - 胜率: {metrics['win_rate']:.1%}")
            print(f"   - 总交易次数: {metrics['total_trades']}")
            print(f"   - 最终权益: {result['portfolio_values'][-1]:,.0f}元")
        # 10. 策略对比分析
        print("\n9️⃣ 策略对比分析...")
        strategy_configs = [
            ("均值回归策略", {}),
            ("动量策略", {})
        ]

        comparison = system.compare_strategies(strategy_configs, data)
        print("策略对比结果:")
        print(comparison.round(4))

        # 11. 组合优化
        print("\n🔟 执行组合优化...")

        # 计算收益率数据用于优化
        returns_data = data.set_index('date')[list(symbol_names.keys())].pct_change().dropna()

        # 最大化夏普比率
        sharpe_result = system.optimize_portfolio(returns_data, method='sharpe')
        if sharpe_result['success']:
            print("最优组合 (最大化夏普比率):")
            for symbol, weight in sharpe_result['weights'].items():
                print(f"   - {symbol}: {weight:.2%}")
            print(f"   - 预期年化收益率: {sharpe_result['expected_return']:.2%}")
            print(f"   - 预期波动率: {sharpe_result['volatility']:.2%}")

        # 风险平价
        rp_result = system.optimize_portfolio(returns_data, method='risk_parity')
        if rp_result['success']:
            print("\n风险平价组合:")
            for symbol, weight in rp_result['weights'].items():
                print(f"   - {symbol}: {weight:.2%}")

        # 12. 使用tushare获取财务数据
        print("\n📈 获取财务数据...")
        try:
            financial_data = dm.get_financial_data([s.split('.')[0] for s in symbols], 'income', periods=4)
            if financial_data:
                print(f"✅ 获取到 {len(financial_data)} 只股票的财务数据")
                # 这里可以进一步分析财务因子
            else:
                print("⚠️ 财务数据获取失败，使用技术因子分析")
        except Exception as e:
            print(f"财务数据获取失败: {e}")

        # 13. 生成分析报告
        print("\n📋 生成分析报告...")

        # 这里可以添加报告生成功能
        print("✅ Tushare量化分析完成！")
        print("\n" + "="*60)
        print("🎉 恭喜！你已经完成了基于Tushare的完整量化工作流！")
        print("="*60)
        print("\n📚 Tushare数据源优势:")
        print("   • 丰富的财务数据和基本面信息")
        print("   • 分钟级和高频数据支持")
        print("   • 宏观经济和行业数据")
        print("   • 稳定的API服务")

        print("\n🚀 接下来你可以:")
        print("   • 探索更多tushare数据接口")
        print("   • 结合财务因子和估值指标")
        print("   • 开发多因子选股模型")
        print("   • 添加宏观经济数据分析")

        print("\n💡 Tushare积分说明:")
        print("   • 日线数据: 120积分/只股票")
        print("   • 财务数据: 2000积分/季度")
        print("   • 分钟数据: 500积分/只股票/日")
        print("   • 建议合理控制数据获取频率")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def tushare_quick_demo():
    """Tushare快速演示"""
    print("\n⚡ Tushare快速演示:")

    try:
        from data_sources import QuantDataManager

        dm = QuantDataManager()

        # 获取基础股票信息
        print("获取A股股票列表...")
        stocks = dm.get_stock_basic_info()
        print(f"获取到{len(stocks)}只股票")

        # 获取单只股票数据
        print("获取平安银行数据...")
        price_data = dm.get_daily_prices(['000001'], '20240101', '20241201')

        if '000001' in price_data:
            df = price_data['000001']
            print(f"平安银行: {len(df)}条记录")
            print(f"最新收盘价: {df['close'].iloc[-1]:.2f}")
            print(f"期间涨幅: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")

            # 计算技术指标
            df_with_indicators = dm.calculate_technical_indicators(df)
            print(f"计算了 {len([col for col in df_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']])} 个技术指标")

    except Exception as e:
        print(f"快速演示失败: {e}")


if __name__ == "__main__":
    main()
    tushare_quick_demo()
