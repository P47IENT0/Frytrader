# -*- coding: utf-8 -*-
"""
Tushare连接测试
验证tushare token是否有效
"""

import tushare as ts
import pandas as pd
import sys

def test_tushare_connection():
    """测试tushare连接"""
    print("测试Tushare连接...")

    try:
        # 设置token
        token = "f6e0a7687738aae0631a7015aac4d91488983113b10962ad66ab3142"
        ts.set_token(token)
        pro = ts.pro_api()

        print("Token设置成功")

        # 测试获取股票基本信息
        print("获取股票基本信息...")
        df = pro.stock_basic(limit=10)
        if not df.empty:
            print(f"获取到 {len(df)} 只股票信息")
            print("前5只股票:")
            print(df[['ts_code', 'symbol', 'name']].head())
        else:
            print("获取股票基本信息失败")
            return False

        # 测试获取日线数据
        print("\n获取平安银行日线数据...")
        df_daily = pro.daily(ts_code='000001.SZ', start_date='20241201', end_date='20241230')
        if not df_daily.empty:
            print(f"获取到 {len(df_daily)} 条日线数据")
            print("最新数据:")
            latest = df_daily.iloc[0]  # tushare返回的数据是倒序的
            print(f"日期: {latest['trade_date']}")
            print(f"开盘: {latest['open']:.2f}")
            print(f"收盘: {latest['close']:.2f}")
            print(f"最高: {latest['high']:.2f}")
            print(f"最低: {latest['low']:.2f}")
            print(f"成交量: {latest['vol']}")
        else:
            print("获取日线数据失败")
            return False

        print("\nTushare连接测试成功！")
        print("你现在可以使用tushare数据源进行量化分析了。")

        return True

    except Exception as e:
        print(f"Tushare连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tushare_quant_features():
    """测试tushare量化功能"""
    print("\n测试Tushare量化功能...")

    try:
        token = "f6e0a7687738aae0631a7015aac4d91488983113b10962ad66ab3142"
        ts.set_token(token)
        pro = ts.pro_api()

        # 测试财务数据获取
        print("获取财务数据...")
        income_df = pro.income(ts_code='000001.SZ', limit=2)
        if not income_df.empty:
            print(f"获取到 {len(income_df)} 条财务数据")
            print("财务数据列:", list(income_df.columns))
        else:
            print("财务数据获取失败（可能是积分不足）")

        # 测试指数数据
        print("获取指数数据...")
        index_df = pro.index_daily(ts_code='000001.SH', start_date='20241201', end_date='20241230')
        if not index_df.empty:
            print(f"获取到 {len(index_df)} 条指数数据")
            latest_index = index_df.iloc[0]
            print(f"上证指数最新: {latest_index['close']:.2f}")
        else:
            print("指数数据获取失败")

        return True

    except Exception as e:
        print(f"量化功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("FryTrader Tushare测试\n")

    success = test_tushare_connection()
    if success:
        test_tushare_quant_features()

    print("\n" + "="*50)
    if success:
        print("所有测试通过！可以开始你的量化之旅了！")
        print("\n使用建议:")
        print("1. 运行 tushare_quant_demo.py 体验完整功能")
        print("2. 查看 tushare_quant_system.py 了解高级用法")
        print("3. 根据需要调整策略参数")
        print("4. 注意tushare的积分消耗")
    else:
        print("测试失败，请检查:")
        print("1. 网络连接")
        print("2. tushare token是否有效")
        print("3. 是否有足够的积分")
