# -*- coding: utf-8 -*-
"""
量化数据提供模块
提供多种数据源的统一接口
"""
import abc
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
import time
from easytrader.log import logger


class BaseDataProvider(abc.ABC):
    """数据提供者基类"""

    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False

    @abc.abstractmethod
    def get_daily_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        pass

    @abc.abstractmethod
    def get_realtime_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时报价"""
        pass

    def get_stock_info(self, symbols: List[str]) -> pd.DataFrame:
        """获取股票基本信息"""
        pass


class AkShareDataProvider(BaseDataProvider):
    """基于akshare的数据提供者"""

    def __init__(self):
        super().__init__()
        try:
            import akshare as ak
            self.ak = ak
        except ImportError:
            raise ImportError("需要安装akshare: pip install akshare")

    def get_daily_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        all_data = []

        for symbol in symbols:
            try:
                # 获取A股日线数据
                df = self.ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
                time.sleep(0.5)  # 避免请求过频繁
            except Exception as e:
                logger.warning(f"获取 {symbol} 数据失败: {e}")
                continue

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # 统一列名
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            }
            result = result.rename(columns=column_mapping)
            result['date'] = pd.to_datetime(result['date'])
            return result

        return pd.DataFrame()

    def get_realtime_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时报价"""
        try:
            # 获取实时行情
            df = self.ak.stock_zh_a_spot_em()
            df_filtered = df[df['代码'].isin(symbols)]

            # 统一列名
            column_mapping = {
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'price',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '成交量': 'volume',
                '成交额': 'amount',
                '最高': 'high',
                '最低': 'low',
                '今开': 'open',
                '昨收': 'prev_close'
            }
            df_filtered = df_filtered.rename(columns=column_mapping)
            return df_filtered
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return pd.DataFrame()


class TushareDataProvider(BaseDataProvider):
    """基于tushare的数据提供者"""

    def __init__(self, token: str):
        super().__init__()
        try:
            import tushare as ts
            self.ts = ts
            self.pro = ts.pro_api(token)
        except ImportError:
            raise ImportError("需要安装tushare: pip install tushare")

    def get_daily_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        all_data = []

        for symbol in symbols:
            try:
                df = self.pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"获取 {symbol} 数据失败: {e}")
                continue

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result['date'] = pd.to_datetime(result['trade_date'])
            return result

        return pd.DataFrame()

    def get_realtime_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时报价"""
        try:
            # 转换为tushare格式
            ts_codes = [f"{symbol}.SH" if symbol.startswith(('6', '9')) else f"{symbol}.SZ"
                       for symbol in symbols]
            df = self.pro.realtime_quote(ts_codes=ts_codes)
            return df
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return pd.DataFrame()


class DataManager:
    """数据管理器"""

    def __init__(self, provider: BaseDataProvider):
        self.provider = provider
        self.cache = {}

    def get_price_data(self, symbols: List[str], start_date: str, end_date: str,
                      use_cache: bool = True) -> pd.DataFrame:
        """获取价格数据，支持缓存"""
        cache_key = f"price_{'_'.join(sorted(symbols))}_{start_date}_{end_date}"

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        data = self.provider.get_daily_prices(symbols, start_date, end_date)

        if use_cache and not data.empty:
            self.cache[cache_key] = data

        return data

    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时数据"""
        return self.provider.get_realtime_quotes(symbols)

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()


# 数据源工厂
class DataProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> BaseDataProvider:
        """创建数据提供者"""
        if provider_type.lower() == 'akshare':
            return AkShareDataProvider()
        elif provider_type.lower() == 'tushare':
            token = kwargs.get('token')
            if not token:
                raise ValueError("tushare需要提供token")
            return TushareDataProvider(token)
        else:
            raise ValueError(f"不支持的数据提供者类型: {provider_type}")


# 使用示例
def create_data_manager(provider_type: str = 'akshare', **kwargs) -> DataManager:
    """创建数据管理器"""
    provider = DataProviderFactory.create_provider(provider_type, **kwargs)
    return DataManager(provider)
