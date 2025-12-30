# -*- coding: utf-8 -*-
"""
量化因子计算引擎
提供常用的技术指标和基本面因子计算
"""
import abc
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime
import talib
from easytrader.log import logger


class BaseFactor(abc.ABC):
    """因子基类"""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        pass

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return self.calculate(data)


class TechnicalFactor(BaseFactor):
    """技术指标因子"""

    def __init__(self, name: str, func: Callable, **params):
        super().__init__(name)
        self.func = func
        self.params = params

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算技术指标"""
        try:
            # 确保数据按日期排序
            if 'date' in data.columns:
                data = data.sort_values('date')

            # 获取价格数据
            close = data['close'].values if 'close' in data.columns else data['price'].values
            high = data['high'].values if 'high' in data.columns else close
            low = data['low'].values if 'low' in data.columns else close
            open_price = data['open'].values if 'open' in data.columns else close
            volume = data['volume'].values if 'volume' in data.columns else np.ones_like(close)

            # 调用相应的技术指标函数
            if self.name == 'RSI':
                result = talib.RSI(close, timeperiod=self.params.get('period', 14))
            elif self.name == 'MACD':
                macd, signal, hist = talib.MACD(close,
                                              fastperiod=self.params.get('fast', 12),
                                              slowperiod=self.params.get('slow', 26),
                                              signalperiod=self.params.get('signal', 9))
                result = macd  # 返回MACD线
            elif self.name == 'MA':
                result = talib.SMA(close, timeperiod=self.params.get('period', 20))
            elif self.name == 'EMA':
                result = talib.EMA(close, timeperiod=self.params.get('period', 20))
            elif self.name == 'BBANDS':
                upper, middle, lower = talib.BBANDS(close,
                                                   timeperiod=self.params.get('period', 20),
                                                   nbdevup=self.params.get('nbdev', 2),
                                                   nbdevdn=self.params.get('nbdev', 2))
                result = (close - lower) / (upper - lower)  # 布林带位置
            elif self.name == 'ATR':
                result = talib.ATR(high, low, close, timeperiod=self.params.get('period', 14))
            elif self.name == 'MOM':
                result = talib.MOM(close, timeperiod=self.params.get('period', 10))
            else:
                # 自定义函数
                result = self.func(close, **self.params)

            return pd.Series(result, index=data.index, name=self.name)

        except Exception as e:
            logger.error(f"计算因子 {self.name} 失败: {e}")
            return pd.Series(index=data.index, name=self.name)


class PriceFactor(BaseFactor):
    """价格类因子"""

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算价格类因子"""
        try:
            if self.name == 'returns':
                # 计算收益率
                close = data['close'] if 'close' in data.columns else data['price']
                return close.pct_change()

            elif self.name == 'log_returns':
                # 对数收益率
                close = data['close'] if 'close' in data.columns else data['price']
                return np.log(close / close.shift(1))

            elif self.name == 'volatility':
                # 波动率（20日）
                returns = self.calculate_returns(data)
                return returns.rolling(window=20).std()

            elif self.name.startswith('MA_'):
                # 简单移动平均
                period = int(self.name.split('_')[1])
                close = data['close'] if 'close' in data.columns else data['price']
                return close.rolling(window=period).mean()

            elif self.name.startswith('EMA_'):
                # 指数移动平均
                period = int(self.name.split('_')[1])
                close = data['close'] if 'close' in data.columns else data['price']
                return close.ewm(span=period).mean()

        except Exception as e:
            logger.error(f"计算价格因子 {self.name} 失败: {e}")
            return pd.Series(index=data.index, name=self.name)

    def calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """计算收益率的辅助方法"""
        close = data['close'] if 'close' in data.columns else data['price']
        return close.pct_change()


class VolumeFactor(BaseFactor):
    """成交量因子"""

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量因子"""
        try:
            volume = data['volume']
            close = data['close'] if 'close' in data.columns else data['price']

            if self.name == 'volume_ratio':
                # 量比（相对5日平均）
                avg_volume = volume.rolling(window=5).mean()
                return volume / avg_volume

            elif self.name == 'volume_price_trend':
                # 量价趋势
                return volume * (close - close.shift(1))

            elif self.name.startswith('VMA_'):
                # 成交量移动平均
                period = int(self.name.split('_')[1])
                return volume.rolling(window=period).mean()

        except Exception as e:
            logger.error(f"计算成交量因子 {self.name} 失败: {e}")
            return pd.Series(index=data.index, name=self.name)


class CustomFactor(BaseFactor):
    """自定义因子"""

    def __init__(self, name: str, func: Callable, **params):
        super().__init__(name)
        self.func = func
        self.params = params

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算自定义因子"""
        try:
            return self.func(data, **self.params)
        except Exception as e:
            logger.error(f"计算自定义因子 {self.name} 失败: {e}")
            return pd.Series(index=data.index, name=self.name)


class FactorEngine:
    """因子计算引擎"""

    def __init__(self):
        self.factors = {}
        self._register_builtin_factors()

    def _register_builtin_factors(self):
        """注册内置因子"""
        # 技术指标因子
        self.register_factor(TechnicalFactor('RSI_14', talib.RSI, timeperiod=14))
        self.register_factor(TechnicalFactor('MACD', talib.MACD))
        self.register_factor(TechnicalFactor('MA_20', talib.SMA, timeperiod=20))
        self.register_factor(TechnicalFactor('EMA_20', talib.EMA, timeperiod=20))
        self.register_factor(TechnicalFactor('BBANDS_POSITION', talib.BBANDS))

        # 价格因子
        self.register_factor(PriceFactor('returns'))
        self.register_factor(PriceFactor('log_returns'))
        self.register_factor(PriceFactor('volatility'))
        self.register_factor(PriceFactor('MA_5'))
        self.register_factor(PriceFactor('MA_10'))
        self.register_factor(PriceFactor('EMA_12'))
        self.register_factor(PriceFactor('EMA_26'))

        # 成交量因子
        self.register_factor(VolumeFactor('volume_ratio'))
        self.register_factor(VolumeFactor('volume_price_trend'))
        self.register_factor(VolumeFactor('VMA_5'))

    def register_factor(self, factor: BaseFactor):
        """注册因子"""
        self.factors[factor.name] = factor

    def unregister_factor(self, factor_name: str):
        """注销因子"""
        if factor_name in self.factors:
            del self.factors[factor_name]

    def calculate_factor(self, factor_name: str, data: pd.DataFrame) -> pd.Series:
        """计算单个因子"""
        if factor_name not in self.factors:
            raise ValueError(f"因子 {factor_name} 未注册")

        return self.factors[factor_name].calculate(data)

    def calculate_factors(self, factor_names: List[str], data: pd.DataFrame) -> pd.DataFrame:
        """计算多个因子"""
        results = {}

        for factor_name in factor_names:
            try:
                factor_data = self.calculate_factor(factor_name, data)
                results[factor_name] = factor_data
            except Exception as e:
                logger.warning(f"计算因子 {factor_name} 失败: {e}")
                continue

        return pd.DataFrame(results)

    def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有注册的因子"""
        return self.calculate_factors(list(self.factors.keys()), data)

    def create_custom_factor(self, name: str, func: Callable, **params) -> CustomFactor:
        """创建自定义因子"""
        factor = CustomFactor(name, func, **params)
        self.register_factor(factor)
        return factor

    def get_available_factors(self) -> List[str]:
        """获取所有可用因子"""
        return list(self.factors.keys())


# 便捷函数
def create_factor_engine() -> FactorEngine:
    """创建因子引擎"""
    return FactorEngine()


def calculate_technical_factors(data: pd.DataFrame, factors: List[str] = None) -> pd.DataFrame:
    """计算技术因子（便捷函数）"""
    engine = create_factor_engine()

    if factors is None:
        # 默认计算常用技术因子
        factors = ['RSI_14', 'MACD', 'MA_20', 'EMA_20', 'returns', 'volatility']

    return engine.calculate_factors(factors, data)


# 自定义因子示例
def momentum_factor(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """动量因子"""
    close = data['close'] if 'close' in data.columns else data['price']
    return (close - close.shift(period)) / close.shift(period)


def mean_reversion_factor(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """均值回归因子"""
    close = data['close'] if 'close' in data.columns else data['price']
    ma = close.rolling(window=period).mean()
    return (close - ma) / ma


# 注册示例自定义因子
if __name__ == "__main__":
    engine = create_factor_engine()

    # 注册自定义因子
    engine.create_custom_factor('momentum_20', momentum_factor, period=20)
    engine.create_custom_factor('mean_reversion_20', mean_reversion_factor, period=20)

    print("可用因子:", engine.get_available_factors())
