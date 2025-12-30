# 量化数据源获取方案
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import akshare as ak
import tushare as ts
import baostock as bs
from jqdatasdk import *
import yfinance as yf
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class QuantDataManager:
    """量化数据管理器"""

    def __init__(self):
        self.ts_token = "f6e0a7687738aae0631a7015aac4d91488983113b10962ad66ab3142"  # 用户的token
        self.jq_token = "your_joinquant_token_here"  # 需要申请
        ts.set_token(self.ts_token)

    # === 基础数据获取 ===

    def get_stock_basic_info(self, market: str = 'CN') -> pd.DataFrame:
        """获取基础股票信息"""
        try:
            if market == 'CN':
                # 中证A股数据
                df = ak.stock_info_a_code_name()
                return df
            elif market == 'HK':
                # 港股数据
                df = ak.stock_hk_daily(symbol="00001", adjust="")
                return df.head(1)  # 只返回结构
            elif market == 'US':
                # 美股数据
                df = ak.stock_us_daily(symbol="AAPL", adjust="")
                return df.head(1)
        except Exception as e:
            print(f"获取股票基础信息失败: {e}")
            return pd.DataFrame()

    def get_daily_prices(self, symbols: List[str], start_date: str, end_date: str,
                        market: str = 'CN') -> Dict[str, pd.DataFrame]:
        """获取日线数据"""
        results = {}

        if market == 'CN':
            # 使用akshare获取A股数据
            for symbol in symbols:
                try:
                    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date,
                                           end_date=end_date, adjust="hfq")
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        results[symbol] = df
                    time.sleep(0.5)  # 避免请求过频
                except Exception as e:
                    print(f"获取{symbol}数据失败: {e}")
                    continue

        elif market == 'US':
            # 美股数据
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    if not df.empty:
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        df.columns = ['open', 'high', 'low', 'close', 'volume']
                        results[symbol] = df
                    time.sleep(0.5)
                except Exception as e:
                    print(f"获取{symbol}美股数据失败: {e}")

        return results

    # === 财务数据获取 ===

    def get_financial_data(self, symbols: List[str], report_type: str = 'income',
                          periods: int = 4) -> Dict[str, pd.DataFrame]:
        """获取财务数据"""
        results = {}

        try:
            pro = ts.pro_api()

            for symbol in symbols:
                try:
                    if report_type == 'income':
                        df = pro.income(ts_code=symbol, period='', start_date='20200101',
                                      end_date=datetime.now().strftime('%Y%m%d'))
                    elif report_type == 'balance':
                        df = pro.balancesheet(ts_code=symbol, period='', start_date='20200101',
                                            end_date=datetime.now().strftime('%Y%m%d'))
                    elif report_type == 'cashflow':
                        df = pro.cashflow(ts_code=symbol, period='', start_date='20200101',
                                        end_date=datetime.now().strftime('%Y%m%d'))

                    if not df.empty:
                        df['ann_date'] = pd.to_datetime(df['ann_date'])
                        df.set_index('ann_date', inplace=True)
                        results[symbol] = df

                    time.sleep(0.5)

                except Exception as e:
                    print(f"获取{symbol}财务数据失败: {e}")
                    continue

        except Exception as e:
            print(f"财务数据API连接失败: {e}")

        return results

    # === 技术指标计算 ===

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        # 简单移动平均
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()

        # 指数移动平均
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 布林带
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

        # 威廉指标
        df['Williams_R'] = (df['high'].rolling(14).max() - df['close']) / \
                          (df['high'].rolling(14).max() - df['low'].rolling(14).min()) * -100

        return df

    # === 市场数据获取 ===

    def get_market_index(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数数据"""
        try:
            if index_code.startswith('000'):  # 上海指数
                df = ak.stock_zh_index_daily(symbol=index_code)
            elif index_code.startswith('399'):  # 深圳指数
                df = ak.stock_zh_index_daily_em(symbol=index_code)
            elif index_code == 'HSI':  # 恒生指数
                df = ak.stock_hk_index_daily_sina(symbol="HSI")
            else:  # 其他指数
                df = ak.stock_zh_index_daily(symbol=index_code)

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                mask = (df.index >= start_date) & (df.index <= end_date)
                return df[mask]

        except Exception as e:
            print(f"获取指数{index_code}数据失败: {e}")

        return pd.DataFrame()

    # === 实时数据获取 ===

    def get_realtime_quotes(self, symbols: List[str], market: str = 'CN') -> pd.DataFrame:
        """获取实时行情"""
        try:
            if market == 'CN':
                df = ak.stock_zh_a_spot_em()
                # 过滤指定股票
                df = df[df['代码'].isin(symbols)]
                df.rename(columns={
                    '代码': 'symbol', '名称': 'name', '最新价': 'price',
                    '涨跌幅': 'change_pct', '涨跌额': 'change', '成交量': 'volume',
                    '成交额': 'amount', '振幅': 'amplitude', '最高': 'high',
                    '最低': 'low', '今开': 'open', '昨收': 'pre_close'
                }, inplace=True)
                return df

        except Exception as e:
            print(f"获取实时行情失败: {e}")

        return pd.DataFrame()

    # === 数据存储和加载 ===

    def save_data(self, data: Dict[str, pd.DataFrame], filename: str, path: str = './data/'):
        """保存数据到本地"""
        import os
        os.makedirs(path, exist_ok=True)

        for symbol, df in data.items():
            filepath = f"{path}{symbol}_{filename}.csv"
            df.to_csv(filepath)
            print(f"保存{symbol}数据到{filepath}")

    def load_data(self, symbols: List[str], filename: str, path: str = './data/') -> Dict[str, pd.DataFrame]:
        """从本地加载数据"""
        results = {}

        for symbol in symbols:
            filepath = f"{path}{symbol}_{filename}.csv"
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                results[symbol] = df
            except FileNotFoundError:
                print(f"文件{filepath}不存在")
                continue

        return results


# === 使用示例 ===

def demo_data_acquisition():
    """数据获取演示"""
    dm = QuantDataManager()

    # 1. 获取股票列表
    print("=== 获取A股股票列表 ===")
    stocks = dm.get_stock_basic_info()
    print(f"获取到{len(stocks)}只股票")
    print(stocks.head())

    # 2. 获取日线数据
    print("\n=== 获取日线数据 ===")
    symbols = ['000001', '600000', '000002']  # 平安银行、浦发银行、万科A
    start_date = '20240101'
    end_date = '20241201'

    price_data = dm.get_daily_prices(symbols, start_date, end_date)
    for symbol, df in price_data.items():
        print(f"{symbol}: {len(df)}条记录")

    # 3. 计算技术指标
    print("\n=== 计算技术指标 ===")
    if '000001' in price_data:
        df_with_indicators = dm.calculate_technical_indicators(price_data['000001'])
        print("技术指标列:", [col for col in df_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']])

    # 4. 获取指数数据
    print("\n=== 获取指数数据 ===")
    index_data = dm.get_market_index('000001', start_date, end_date)  # 上证指数
    print(f"上证指数: {len(index_data)}条记录")

    # 5. 获取实时行情
    print("\n=== 获取实时行情 ===")
    realtime_data = dm.get_realtime_quotes(symbols[:2])
    print(realtime_data)

    return price_data, index_data


if __name__ == "__main__":
    demo_data_acquisition()
