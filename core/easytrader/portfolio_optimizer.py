# -*- coding: utf-8 -*-
"""
组合优化模块
提供现代投资组合理论（MPT）的实现和多种优化算法
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize, Bounds
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

from easytrader.log import logger


class PortfolioOptimizer:
    """投资组合优化器"""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.03):
        """
        初始化优化器

        Args:
            returns: 资产收益率数据，DataFrame格式，列为资产，行为日期
            risk_free_rate: 无风险利率，默认3%
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(self.mean_returns)

        # 使用Ledoit-Wolf收缩估计器改进协方差矩阵
        try:
            lw = LedoitWolf().fit(returns.values)
            self.cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        except:
            logger.warning("Ledoit-Wolf协方差估计失败，使用样本协方差")

    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """计算组合的收益、风险等指标"""
        weights = np.array(weights)

        # 组合期望收益
        portfolio_return = np.dot(weights, self.mean_returns)

        # 组合方差和标准差
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # 夏普比率
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0

        # 最大回撤（简化计算）
        cumulative_returns = (1 + self.returns.dot(weights)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 索提诺比率（使用负收益计算）
        negative_returns = self.returns[self.returns < 0].dot(weights)
        downside_std = negative_returns.std() if len(negative_returns) > 0 else portfolio_std
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': np.percentile(self.returns.dot(weights), 5),  # 95% VaR
        }

    def minimize_volatility(self, target_return: Optional[float] = None) -> Dict:
        """最小化波动率（或在目标收益下最小化波动率）"""
        def objective(weights):
            return self.calculate_portfolio_metrics(weights)['volatility']

        def constraint_return(weights):
            return np.dot(weights, self.mean_returns) - (target_return or self.mean_returns.mean())

        constraints = []
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': constraint_return})

        bounds = Bounds(0, 1)
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        initial_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(objective, initial_weights,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)

        if result.success:
            weights = result.x
            metrics = self.calculate_portfolio_metrics(weights)
            return {
                'weights': dict(zip(self.returns.columns, weights)),
                'metrics': metrics,
                'success': True
            }
        else:
            logger.error(f"最小波动率优化失败: {result.message}")
            return {'success': False, 'message': result.message}

    def maximize_sharpe_ratio(self) -> Dict:
        """最大化夏普比率"""
        def objective(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return -metrics['sharpe_ratio']  # 最小化负夏普比率

        bounds = Bounds(0, 1)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        initial_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(objective, initial_weights,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)

        if result.success:
            weights = result.x
            metrics = self.calculate_portfolio_metrics(weights)
            return {
                'weights': dict(zip(self.returns.columns, weights)),
                'metrics': metrics,
                'success': True
            }
        else:
            logger.error(f"最大夏普比率优化失败: {result.message}")
            return {'success': False, 'message': result.message}

    def efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """计算有效前沿"""
        results = []

        # 生成随机权重组合
        for _ in range(n_portfolios):
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)  # 归一化

            metrics = self.calculate_portfolio_metrics(weights)
            results.append({
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'weights': weights
            })

        return pd.DataFrame(results)

    def risk_parity_weights(self) -> Dict:
        """风险平价权重"""
        def objective(weights):
            # 计算每个资产的边际风险贡献
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_risk = np.dot(self.cov_matrix, weights) / portfolio_vol
            risk_contributions = weights * marginal_risk

            # 风险平价目标：所有资产的风险贡献相等
            target_risk = np.mean(risk_contributions)
            return np.sum((risk_contributions - target_risk) ** 2)

        bounds = Bounds(0.001, 1)  # 避免权重为0
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        initial_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(objective, initial_weights,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)

        if result.success:
            weights = result.x
            metrics = self.calculate_portfolio_metrics(weights)
            return {
                'weights': dict(zip(self.returns.columns, weights)),
                'metrics': metrics,
                'success': True
            }
        else:
            logger.error(f"风险平价优化失败: {result.message}")
            return {'success': False, 'message': result.message}

    def equal_weight_portfolio(self) -> Dict:
        """等权重组合"""
        weights = np.ones(self.n_assets) / self.n_assets
        metrics = self.calculate_portfolio_metrics(weights)

        return {
            'weights': dict(zip(self.returns.columns, weights)),
            'metrics': metrics,
            'success': True
        }

    def market_cap_weighted_portfolio(self, market_caps: Dict[str, float]) -> Dict:
        """市值加权组合"""
        # 确保market_caps包含所有资产
        missing_assets = set(self.returns.columns) - set(market_caps.keys())
        if missing_assets:
            logger.warning(f"缺少市值数据: {missing_assets}")

        # 归一化权重
        total_cap = sum(market_caps.get(asset, 0) for asset in self.returns.columns)
        if total_cap == 0:
            logger.error("市值数据无效")
            return {'success': False, 'message': '市值数据无效'}

        weights = np.array([market_caps.get(asset, 0) / total_cap for asset in self.returns.columns])
        metrics = self.calculate_portfolio_metrics(weights)

        return {
            'weights': dict(zip(self.returns.columns, weights)),
            'metrics': metrics,
            'success': True
        }

    def hierarchical_risk_parity(self) -> Dict:
        """层次风险平价（简化版本）"""
        # 计算相关性矩阵
        corr_matrix = self.returns.corr()

        # 使用层次聚类简化版：按波动率排序分配权重
        volatilities = np.sqrt(np.diag(self.cov_matrix.values))
        sorted_indices = np.argsort(volatilities)

        # 按波动率倒序分配权重（波动率低的权重更高）
        weights = np.zeros(self.n_assets)
        for i, idx in enumerate(sorted_indices):
            weights[idx] = 1.0 / (i + 1)  # 简化权重分配

        weights /= np.sum(weights)  # 归一化

        metrics = self.calculate_portfolio_metrics(weights)
        return {
            'weights': dict(zip(self.returns.columns, weights)),
            'metrics': metrics,
            'success': True
        }


class BlackLittermanOptimizer:
    """Black-Litterman模型优化器"""

    def __init__(self, returns: pd.DataFrame, market_weights: np.ndarray,
                 risk_aversion: float = 2.5, tau: float = 0.05):
        """
        Black-Litterman模型

        Args:
            returns: 资产收益率数据
            market_weights: 市场均衡权重
            risk_aversion: 风险厌恶系数
            tau: 不确定性参数
        """
        self.returns = returns
        self.market_weights = market_weights
        self.risk_aversion = risk_aversion
        self.tau = tau

        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(self.mean_returns)

    def optimize_with_views(self, views: Dict[str, float], view_confidences: Dict[str, float]) -> Dict:
        """
        基于观点的优化

        Args:
            views: 观点字典，格式如 {'000001': 0.1} 表示对000001的超额收益预期
            view_confidences: 观点置信度字典
        """
        # 构建观点矩阵
        view_assets = list(views.keys())
        view_returns = np.array(list(views.values()))
        view_confidences = np.array(list(view_confidences.values()))

        # 简化实现：只处理绝对观点
        P = np.eye(len(view_assets))
        Q = view_returns

        # 观点协方差矩阵
        omega = np.diag(1.0 / view_confidences)

        # Black-Litterman公式
        tau_sigma = self.tau * self.cov_matrix.loc[view_assets, view_assets].values

        # 后验期望收益
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega)

        temp = inv_tau_sigma + P.T @ inv_omega @ P
        posterior_mean = np.linalg.inv(temp) @ (inv_tau_sigma @ np.zeros(len(view_assets)) + P.T @ inv_omega @ Q)

        # 使用后验均值进行传统优化
        bl_returns = self.mean_returns.copy()
        for i, asset in enumerate(view_assets):
            bl_returns[asset] = posterior_mean[i]

        # 使用传统方法优化
        optimizer = PortfolioOptimizer(self.returns, 0.03)
        optimizer.mean_returns = bl_returns

        return optimizer.maximize_sharpe_ratio()


def optimize_portfolio(returns: pd.DataFrame,
                      method: str = 'sharpe',
                      **kwargs) -> Dict:
    """
    便捷的组合优化函数

    Args:
        returns: 资产收益率数据
        method: 优化方法 ('sharpe', 'min_vol', 'equal_weight', 'risk_parity', 'hrp')
        **kwargs: 其他参数
    """
    optimizer = PortfolioOptimizer(returns)

    if method == 'sharpe':
        return optimizer.maximize_sharpe_ratio()
    elif method == 'min_vol':
        return optimizer.minimize_volatility()
    elif method == 'equal_weight':
        return optimizer.equal_weight_portfolio()
    elif method == 'risk_parity':
        return optimizer.risk_parity_weights()
    elif method == 'hrp':
        return optimizer.hierarchical_risk_parity()
    else:
        raise ValueError(f"不支持的优化方法: {method}")


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_assets = 5
    n_periods = 252

    # 模拟资产收益率
    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.02,
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    )

    # 创建优化器
    optimizer = PortfolioOptimizer(returns)

    # 最大化夏普比率
    sharpe_result = optimizer.maximize_sharpe_ratio()
    print("最大夏普比率组合:")
    print(f"权重: {sharpe_result['weights']}")
    print(f"预期收益: {sharpe_result['metrics']['return']:.4f}")
    print(f"波动率: {sharpe_result['metrics']['volatility']:.4f}")
    print(f"夏普比率: {sharpe_result['metrics']['sharpe_ratio']:.4f}")

    # 等权重组合
    equal_result = optimizer.equal_weight_portfolio()
    print("\n等权重组合:")
    print(f"权重: {equal_result['weights']}")
    print(f"预期收益: {equal_result['metrics']['return']:.4f}")

    # 风险平价
    rp_result = optimizer.risk_parity_weights()
    print("\n风险平价组合:")
    print(f"权重: {rp_result['weights']}")
    print(f"波动率: {rp_result['metrics']['volatility']:.4f}")
