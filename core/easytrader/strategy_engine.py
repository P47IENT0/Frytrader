# -*- coding: utf-8 -*-
"""
量化策略引擎
提供策略开发、执行和管理的框架
"""
import abc
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
from easytrader.log import logger


class OrderType(Enum):
    """订单类型"""
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_MARKET = "buy_market"
    SELL_MARKET = "sell_market"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Order:
    """交易订单"""

    def __init__(self, symbol: str, order_type: OrderType, quantity: int,
                 price: Optional[float] = None, order_id: Optional[str] = None):
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.order_id = order_id or f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.filled_price = 0.0
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"Order({self.symbol}, {self.order_type.value}, qty={self.quantity}, price={self.price})"


class Position:
    """持仓信息"""

    def __init__(self, symbol: str, quantity: int = 0, avg_price: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.current_price = 0.0
        self.market_value = 0.0
        self.unrealized_pnl = 0.0

    def update_price(self, price: float):
        """更新价格"""
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity

    def add_position(self, quantity: int, price: float):
        """增加持仓"""
        if self.quantity + quantity == 0:
            self.quantity = 0
            self.avg_price = 0.0
        else:
            total_cost = self.avg_price * self.quantity + price * quantity
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity if self.quantity != 0 else 0.0

    def __repr__(self):
        return f"Position({self.symbol}: qty={self.quantity}, avg_price={self.avg_price:.2f})"


class BaseStrategy(abc.ABC):
    """策略基类"""

    def __init__(self, name: str, initial_capital: float = 100000):
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> Position
        self.orders = []  # List[Order]
        self.portfolio_value = initial_capital
        self.trades = []  # 交易记录

        # 策略参数
        self.params = {}

    @abc.abstractmethod
    def initialize(self):
        """策略初始化"""
        pass

    @abc.abstractmethod
    def on_data(self, data: pd.DataFrame):
        """数据更新时的处理"""
        pass

    def buy(self, symbol: str, quantity: int, price: Optional[float] = None) -> Order:
        """买入"""
        if price is None:
            price = self.get_current_price(symbol)

        cost = quantity * price
        if cost > self.current_capital:
            logger.warning(f"资金不足: 需要 {cost}, 可用 {self.current_capital}")
            return None

        order = Order(symbol, OrderType.BUY, quantity, price)
        self.orders.append(order)
        return order

    def sell(self, symbol: str, quantity: int, price: Optional[float] = None) -> Order:
        """卖出"""
        if symbol not in self.positions:
            logger.warning(f"没有 {symbol} 的持仓")
            return None

        current_quantity = self.positions[symbol].quantity
        if quantity > current_quantity:
            logger.warning(f"卖出数量超过持仓: 请求 {quantity}, 持仓 {current_quantity}")
            quantity = current_quantity

        if price is None:
            price = self.get_current_price(symbol)

        order = Order(symbol, OrderType.SELL, quantity, price)
        self.orders.append(order)
        return order

    def get_current_price(self, symbol: str) -> float:
        """获取当前价格（需要子类实现）"""
        raise NotImplementedError

    def update_positions(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

        # 计算组合价值
        total_value = self.current_capital
        for position in self.positions.values():
            total_value += position.market_value
        self.portfolio_value = total_value

    def get_portfolio_info(self) -> Dict:
        """获取组合信息"""
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.current_capital,
            'positions': {k: {'quantity': v.quantity, 'avg_price': v.avg_price,
                            'current_price': v.current_price, 'pnl': v.unrealized_pnl}
                         for k, v in self.positions.items()},
            'total_positions': len([p for p in self.positions.values() if p.quantity != 0])
        }


class TechnicalStrategy(BaseStrategy):
    """技术分析策略基类"""

    def __init__(self, name: str, initial_capital: float = 100000):
        super().__init__(name, initial_capital)
        self.price_data = {}  # symbol -> DataFrame
        self.indicators = {}  # symbol -> indicators dict

    def add_symbol(self, symbol: str, data: pd.DataFrame):
        """添加交易标的"""
        self.price_data[symbol] = data.copy()
        self.positions[symbol] = Position(symbol)

    def update_price_data(self, symbol: str, new_data: pd.DataFrame):
        """更新价格数据"""
        if symbol in self.price_data:
            self.price_data[symbol] = pd.concat([self.price_data[symbol], new_data]).drop_duplicates()
        else:
            self.price_data[symbol] = new_data

    def calculate_indicators(self, symbol: str):
        """计算技术指标"""
        if symbol not in self.price_data:
            return {}

        data = self.price_data[symbol]

        # 计算常用技术指标
        indicators = {}

        # 移动平均线
        indicators['MA5'] = data['close'].rolling(5).mean()
        indicators['MA10'] = data['close'].rolling(10).mean()
        indicators['MA20'] = data['close'].rolling(20).mean()

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        indicators['MACD'] = exp1 - exp2
        indicators['Signal'] = indicators['MACD'].ewm(span=9).mean()

        self.indicators[symbol] = indicators
        return indicators

    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        if symbol in self.price_data and not self.price_data[symbol].empty:
            return self.price_data[symbol]['close'].iloc[-1]
        return 0.0


class MeanReversionStrategy(TechnicalStrategy):
    """均值回归策略"""

    def initialize(self):
        """策略初始化"""
        self.params = {
            'lookback_period': 20,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'max_positions': 5,
            'position_size': 0.2  # 每只股票最大仓位比例
        }

    def on_data(self, data: pd.DataFrame):
        """数据更新处理"""
        # 这里应该处理实时数据更新
        # 简化版本，假设data包含了所有symbol的数据
        pass

    def generate_signals(self, symbol: str) -> str:
        """生成交易信号"""
        if symbol not in self.price_data:
            return 'hold'

        data = self.price_data[symbol]
        if len(data) < self.params['lookback_period']:
            return 'hold'

        # 计算均值回归指标
        price = data['close']
        ma = price.rolling(self.params['lookback_period']).mean()
        std = price.rolling(self.params['lookback_period']).std()
        z_score = (price - ma) / std

        current_z = z_score.iloc[-1]

        # 交易逻辑
        if current_z < -self.params['entry_threshold']:
            return 'buy'  # 价格偏低，买入
        elif current_z > self.params['entry_threshold']:
            return 'sell'  # 价格偏高，卖出
        elif abs(current_z) < self.params['exit_threshold']:
            return 'close'  # 价格回归，平仓

        return 'hold'


class MomentumStrategy(TechnicalStrategy):
    """动量策略"""

    def initialize(self):
        """策略初始化"""
        self.params = {
            'momentum_period': 20,
            'top_n': 10,
            'rebalance_period': 20,  # 调仓周期
            'max_positions': 5
        }
        self.day_count = 0

    def on_data(self, data: pd.DataFrame):
        """数据更新处理"""
        self.day_count += 1

        # 定期调仓
        if self.day_count % self.params['rebalance_period'] == 0:
            self.rebalance_portfolio()

    def calculate_momentum(self, symbol: str) -> float:
        """计算动量"""
        if symbol not in self.price_data:
            return 0.0

        data = self.price_data[symbol]
        if len(data) < self.params['momentum_period'] + 1:
            return 0.0

        # 计算动量：当前价格相对于N天前的收益率
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-self.params['momentum_period']-1]
        return (current_price - past_price) / past_price

    def rebalance_portfolio(self):
        """组合调仓"""
        # 计算所有股票的动量
        momentum_scores = {}
        for symbol in self.price_data.keys():
            momentum_scores[symbol] = self.calculate_momentum(symbol)

        # 选择动量最强的股票
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_stocks = [stock for stock, score in sorted_stocks[:self.params['top_n']]]

        # 平掉不在榜单上的持仓
        current_positions = [s for s, p in self.positions.items() if p.quantity > 0]
        for symbol in current_positions:
            if symbol not in top_stocks:
                position = self.positions[symbol]
                if position.quantity > 0:
                    self.sell(symbol, position.quantity)

        # 等权重分配资金给入选股票
        if top_stocks:
            capital_per_stock = self.current_capital / len(top_stocks)
            for symbol in top_stocks:
                current_price = self.get_current_price(symbol)
                if current_price > 0:
                    quantity = int(capital_per_stock / current_price)
                    if quantity > 0:
                        self.buy(symbol, quantity)


class StrategyEngine:
    """策略引擎"""

    def __init__(self):
        self.strategies = {}
        self.active_strategies = []

    def register_strategy(self, strategy: BaseStrategy):
        """注册策略"""
        self.strategies[strategy.name] = strategy

    def unregister_strategy(self, strategy_name: str):
        """注销策略"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            if strategy_name in self.active_strategies:
                self.active_strategies.remove(strategy_name)

    def activate_strategy(self, strategy_name: str):
        """激活策略"""
        if strategy_name in self.strategies and strategy_name not in self.active_strategies:
            self.active_strategies.append(strategy_name)

    def deactivate_strategy(self, strategy_name: str):
        """停用策略"""
        if strategy_name in self.active_strategies:
            self.active_strategies.remove(strategy_name)

    def run_strategies(self, data: pd.DataFrame):
        """运行所有激活的策略"""
        for strategy_name in self.active_strategies:
            strategy = self.strategies[strategy_name]
            try:
                strategy.on_data(data)
            except Exception as e:
                logger.error(f"运行策略 {strategy_name} 失败: {e}")

    def get_strategy_status(self) -> Dict:
        """获取策略状态"""
        status = {}
        for name, strategy in self.strategies.items():
            status[name] = {
                'active': name in self.active_strategies,
                'portfolio_value': strategy.portfolio_value,
                'positions_count': len([p for p in strategy.positions.values() if p.quantity != 0]),
                'orders_count': len([o for o in strategy.orders if o.status == OrderStatus.PENDING])
            }
        return status


# 便捷函数
def create_strategy_engine() -> StrategyEngine:
    """创建策略引擎"""
    return StrategyEngine()


def create_mean_reversion_strategy(name: str, capital: float = 100000) -> MeanReversionStrategy:
    """创建均值回归策略"""
    strategy = MeanReversionStrategy(name, capital)
    strategy.initialize()
    return strategy


def create_momentum_strategy(name: str, capital: float = 100000) -> MomentumStrategy:
    """创建动量策略"""
    strategy = MomentumStrategy(name, capital)
    strategy.initialize()
    return strategy


# 使用示例
if __name__ == "__main__":
    # 创建策略引擎
    engine = create_strategy_engine()

    # 创建并注册策略
    mean_rev_strategy = create_mean_reversion_strategy("MeanReversion_001", 100000)
    momentum_strategy = create_momentum_strategy("Momentum_001", 100000)

    engine.register_strategy(mean_rev_strategy)
    engine.register_strategy(momentum_strategy)

    # 激活策略
    engine.activate_strategy("MeanReversion_001")

    print("策略引擎初始化完成")
    print("可用策略:", list(engine.strategies.keys()))
    print("激活策略:", engine.active_strategies)
