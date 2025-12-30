# -*- coding: utf-8 -*-
"""
FryTrader Tushare简单演示
基于tushare数据源的基础量化分析
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    print("FryTrader Tushare简单演示\n")

    try:
        # 设置tushare token
        token = "f6e0a7687738aae0631a7015aac4d91488983113b10962ad66ab3142"
        ts.set_token(token)
        pro = ts.pro_api()

        print("1. 初始化完成")

        # 定义交易标的
        symbols = ['000001.SZ', '000002.SZ', '600036.SH', '600519.SH']
        symbol_names = {
            '000001.SZ': '平安银行',
            '000002.SZ': '万科A',
            '600036.SH': '招商银行',
            '600519.SH': '贵州茅台'
        }

        print(f"2. 交易标的: {', '.join([f'{k}({v})' for k, v in symbol_names.items()])}")

        # 获取数据
        print("\n3. 获取历史数据...")
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

        all_data = []
        for ts_code in symbols:
            try:
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df['symbol'] = ts_code.split('.')[0]
                    df['date'] = pd.to_datetime(df['trade_date'])
                    df.set_index('date', inplace=True)
                    all_data.append(df)
                    print(f"   {symbol_names[ts_code]}: {len(df)}条记录")
            except Exception as e:
                print(f"   {symbol_names[ts_code]}: 获取失败 - {e}")

        if not all_data:
            print("无法获取数据，请检查网络连接")
            return

        # 合并数据
        data = pd.concat(all_data, axis=0)
        print(f"\n4. 数据加载完成，共{len(data)}条记录")

        # 计算基本因子
        print("\n5. 计算技术因子...")

        # 按股票分组计算因子
        factor_data = []
        for symbol in data['symbol'].unique():
            stock_data = data[data['symbol'] == symbol].copy()

            # 收益率
            stock_data['returns'] = stock_data['close'].pct_change()

            # 移动平均线
            stock_data['MA5'] = stock_data['close'].rolling(5).mean()
            stock_data['MA20'] = stock_data['close'].rolling(20).mean()

            # RSI指标
            delta = stock_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            stock_data['RSI'] = 100 - (100 / (1 + rs))

            # 波动率
            stock_data['volatility'] = stock_data['returns'].rolling(20).std() * np.sqrt(252)

            factor_data.append(stock_data)

        factor_df = pd.concat(factor_data)
        print(f"   计算完成，新增因子: returns, MA5, MA20, RSI, volatility")

        # 简单策略实现
        print("\n6. 执行简单策略回测...")

        # 均值回归策略示例
        strategy_results = []

        for symbol in data['symbol'].unique():
            stock_data = factor_df[factor_df['symbol'] == symbol].copy()

            # 生成交易信号 (简化版)
            stock_data['signal'] = 0

            # 当价格低于MA20-2倍标准差时买入
            ma20 = stock_data['MA20']
            std20 = stock_data['close'].rolling(20).std()
            lower_bound = ma20 - 2 * std20

            stock_data.loc[stock_data['close'] < lower_bound, 'signal'] = 1

            # 当价格高于MA20+2倍标准差时卖出
            upper_bound = ma20 + 2 * std20
            stock_data.loc[stock_data['close'] > upper_bound, 'signal'] = -1

            # 模拟交易 (简化)
            capital = 100000
            position = 0
            trades = []

            for idx, row in stock_data.iterrows():
                if row['signal'] == 1 and position == 0:  # 买入
                    shares = int(capital / row['close'])
                    position = shares
                    capital -= shares * row['close']
                    trades.append({'date': idx, 'action': 'BUY', 'price': row['close'], 'shares': shares})

                elif row['signal'] == -1 and position > 0:  # 卖出
                    capital += position * row['close']
                    trades.append({'date': idx, 'action': 'SELL', 'price': row['close'], 'shares': position})
                    position = 0

            # 计算最终价值
            final_value = capital + position * stock_data['close'].iloc[-1]
            total_return = (final_value - 100000) / 100000

            strategy_results.append({
                'symbol': symbol,
                'name': symbol_names.get(f"{symbol}.SZ", symbol),
                'trades': len(trades),
                'final_value': final_value,
                'total_return': total_return
            })

        # 显示结果
        print("\n7. 策略回测结果:")
        for result in strategy_results:
            print("   {}: 交易{}次, 最终价值{:.0f}, 总收益率{:.1%}".format(
                result['name'],
                result['trades'],
                result['final_value'],
                result['total_return']
            ))

        # 基础统计
        returns = [r['total_return'] for r in strategy_results]
        print("\n   统计信息:")
        print("   平均收益率: {:.1%}".format(np.mean(returns)))
        print("   胜率: {:.1%}".format(len([r for r in returns if r > 0]) / len(returns)))
        print("   最大收益率: {:.1%}".format(max(returns)))
        print("   最小收益率: {:.1%}".format(min(returns)))

        print("\n8. 演示完成！")
        print("\n接下来你可以:")
        print("- 完善交易信号逻辑")
        print("- 添加更多的技术指标")
        print("- 实现风险管理")
        print("- 连接真实的交易接口")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
